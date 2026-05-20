//! Chat Loop
//!
//! Main interactive chat loop implementation.

use anyhow::{Context, Result};
use dialoguer::{Input, theme::ColorfulTheme};
use std::io::{self, BufRead, IsTerminal};
use std::sync::Arc;

use super::ai_processing::{process_ai_response, process_ai_response_mdap};
use super::command_dispatch::handle_command;
use crate::auth::SessionManager;
use crate::commands::CommandExecutor;
use crate::config::ConfigManager;
use crate::mdap::MdapConfig;
use crate::providers::ProviderFactory;
use crate::storage::VectorDatabase;
use crate::types::agent::AgentContext;
use crate::types::message::{Message, MessageContent, Role};
use crate::utils::checkpoint::CheckpointManager;
use crate::utils::conversation::ConversationManager;
use crate::utils::logger::Logger;
use crate::utils::rich_output::RichOutput;
use crate::utils::system_prompt::build_system_prompt;
use brainwires::knowledge::bks_pks::personal::PksIntegration;

/// Handle chat with conversation management
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_with_conversation(
    model: Option<String>,
    provider: Option<String>,
    system: Option<String>,
    conversation_id: Option<String>,
    json_output: bool,
    quiet: bool,
    format: &str,
    mdap_config: Option<MdapConfig>,
    backend_url_override: Option<String>,
) -> Result<()> {
    // Load configuration and session
    let config_manager = ConfigManager::new()?;
    let session = SessionManager::load()?;

    // Resolve provider (CLI flag > env > config) and model.
    let config = config_manager.get();
    let active_provider =
        ProviderFactory::effective_provider(provider.as_deref(), config.provider_type)?;
    let model_id = match model {
        Some(m) => m,
        None => config.model.clone(),
    };

    if let Some(ref url) = backend_url_override {
        Logger::info(format!(
            "Starting chat session with {} via {} (dev backend: {})",
            model_id,
            active_provider.as_str(),
            url
        ));
    } else {
        Logger::info(format!(
            "Starting chat session with {} via {}",
            model_id,
            active_provider.as_str()
        ));
    }

    // Create provider using factory with optional backend URL override
    let factory = ProviderFactory;
    let provider_instance = factory
        .create_with_overrides(
            model_id.clone(),
            Some(active_provider),
            backend_url_override,
        )
        .await
        .context("Failed to create provider — run `brainwires auth status` to diagnose")?;

    // Initialize agent context with core tools only to reduce token cost
    let user_id = session.as_ref().map(|s| s.user.user_id.clone());

    let registry = brainwires_tool_builtins::registry_with_builtins();
    let mut context = AgentContext {
        working_directory: std::env::current_dir()?.to_string_lossy().to_string(),
        user_id,
        conversation_history: Vec::new(),
        tools: crate::tools::select_non_tui_tools(&registry),
        metadata: std::collections::HashMap::new(),
        working_set: crate::types::WorkingSet::new(),
        // Use full_access for CLI mode - users expect agents to have write access
        capabilities: brainwires::permissions::AgentCapabilities::full_access(),
    };

    // Initialize conversation manager for auto-save
    let mut conversation_manager = ConversationManager::new(128000);
    conversation_manager.set_model(model_id.clone());

    // Store for cleared conversation (for /resume)
    let mut cleared_conversation_manager: Option<ConversationManager> = None;

    // Initialize command executor
    let command_executor =
        CommandExecutor::new().context("Failed to initialize command executor")?;

    // Initialize checkpoint manager
    let checkpoint_manager =
        CheckpointManager::new().context("Failed to initialize checkpoint manager")?;

    // Initialize PKS integration for implicit fact detection
    let mut pks_integration = PksIntegration::default();
    // Record initial working directory for context inference
    pks_integration.record_working_directory(&context.working_directory);

    // Initialize storage for loading conversations
    let db_path = crate::config::PlatformPaths::conversations_db_path()?;
    let lance_client = Arc::new(
        crate::storage::LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
            .await
            .context("Failed to create LanceDB client")?,
    );
    let embeddings = Arc::new(
        crate::storage::CachedEmbeddingProvider::new()
            .context("Failed to create embedding provider")?,
    );
    lance_client
        .initialize(embeddings.dimension())
        .await
        .context("Failed to initialize LanceDB")?;
    let message_store = crate::storage::MessageStore::new(lance_client.clone(), embeddings);

    // Load existing conversation if conversation_id is provided
    let is_resuming = if let Some(conv_id) = conversation_id {
        Logger::info(format!("Loading conversation: {}", conv_id));
        conversation_manager
            .load_from_db(&conv_id)
            .await
            .context("Failed to load conversation from database")?;

        // Update context with loaded messages
        context.conversation_history = conversation_manager.get_messages().to_vec();

        Logger::info(format!(
            "Loaded {} messages from conversation",
            conversation_manager.get_messages().len()
        ));
        true
    } else {
        false
    };

    // Build system message for chat (only for new conversations)
    if !is_resuming {
        let system_prompt = build_system_prompt(system)?;
        let sys_message = Message {
            role: Role::System,
            content: MessageContent::Text(system_prompt),
            name: None,
            metadata: None,
        };
        context.conversation_history.push(sys_message.clone());
        conversation_manager.add_message(sys_message);
    }

    // Print welcome message (unless quiet)
    if !quiet {
        println!("{}", RichOutput::header("Brainwires Chat", "cyan"));
        println!("Model: {} (brainwires)", model_id);
        println!(
            "Conversation ID: {}",
            console::style(conversation_manager.conversation_id()).dim()
        );
        if is_resuming {
            println!(
                "{}",
                console::style("(Resuming previous conversation)").yellow()
            );
        }
        if mdap_config.is_some() {
            println!(
                "{}",
                console::style("MDAP Mode: Enabled (high-reliability execution)").green()
            );
        }
        println!("Type your message or 'exit' to quit\n");
    }

    // Detect if running in interactive mode
    let is_interactive = io::stdin().is_terminal();
    let stdin = io::stdin();
    let mut stdin_reader = stdin.lock();

    // Load layered harness settings + hook dispatcher once per chat
    // session. Used for UserPromptSubmit and Stop events; tool-level
    // hooks are wired separately inside the ToolExecutor.
    let hook_dispatcher: Arc<crate::hooks::HookDispatcher> = {
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        crate::hooks::load_for_cwd(&cwd).1
    };

    // Main chat loop
    loop {
        // Get user input
        let input: String = if is_interactive {
            Input::with_theme(&ColorfulTheme::default())
                .with_prompt("You")
                .allow_empty(false)
                .interact_text()?
        } else {
            let mut line = String::new();
            match stdin_reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => line.trim().to_string(),
                Err(e) => return Err(e.into()),
            }
        };

        // Check for exit
        if input.trim().eq_ignore_ascii_case("exit") || input.trim().eq_ignore_ascii_case("quit") {
            handle_exit(&mut conversation_manager, &model_id, json_output, quiet).await?;
            break;
        }

        // Skip empty lines in non-interactive mode
        if !is_interactive && input.is_empty() {
            continue;
        }

        // Check if input is a slash command
        if let Some((cmd_name, cmd_args)) = command_executor.parse_input(&input) {
            let should_continue = handle_command(
                &command_executor,
                &cmd_name,
                &cmd_args,
                &model_id,
                &mut context,
                &mut conversation_manager,
                &mut cleared_conversation_manager,
                &checkpoint_manager,
                &message_store,
            )
            .await?;

            if !should_continue {
                break;
            }

            // If command didn't produce a message for AI, continue the loop
            if !matches!(
                command_executor.execute(&cmd_name, &cmd_args),
                Ok(crate::commands::executor::CommandResult::Message(_))
                    | Ok(crate::commands::executor::CommandResult::ActionWithMessage(
                        _,
                        _
                    ))
            ) {
                continue;
            }
        } else {
            // Not a command, add user message to conversation manager
            let user_message = Message {
                role: Role::User,
                content: MessageContent::Text(input.clone()),
                name: None,
                metadata: None,
            };
            conversation_manager.add_message(user_message);

            // UserPromptSubmit hook — fired after the prompt is in the
            // conversation. Exit 2 from a hook blocks the turn, returning
            // the hook's stderr to the user as a reason.
            match hook_dispatcher.dispatch_user_prompt(&input).await {
                crate::hooks::HookOutcome::Continue => {}
                crate::hooks::HookOutcome::Block { reason } => {
                    Logger::warn(format!("Blocked by UserPromptSubmit hook: {}", reason));
                    continue;
                }
                crate::hooks::HookOutcome::SoftError(msg) => {
                    tracing::warn!("UserPromptSubmit hook error: {}", msg);
                }
            }

            // PKS: Process user message for implicit fact detection
            let detected_count = pks_integration.process_user_message(&input);
            if detected_count > 0 {
                tracing::debug!(
                    "PKS: Detected {} implicit facts from user message",
                    detected_count
                );
            }
        }

        // Process the message with AI
        if let Some(ref config) = mdap_config {
            // MDAP mode: high-reliability execution with voting
            process_ai_response_mdap(
                &provider_instance,
                &mut context,
                &mut conversation_manager,
                &model_id,
                &input,
                quiet,
                format,
                config,
            )
            .await?;
        } else {
            // Standard mode
            process_ai_response(
                &provider_instance,
                &mut context,
                &mut conversation_manager,
                &model_id,
                &input,
                quiet,
                format,
            )
            .await?;
        }
    }

    Ok(())
}

/// Handle exit command
async fn handle_exit(
    conversation_manager: &mut ConversationManager,
    model_id: &str,
    json_output: bool,
    quiet: bool,
) -> Result<()> {
    use serde_json::json;

    if !quiet {
        Logger::info("Saving conversation...");
    }
    if let Err(e) = conversation_manager.save_to_db().await {
        if !quiet {
            Logger::warn(format!("Failed to save conversation: {}", e));
        }
    } else if !quiet {
        Logger::info("Conversation saved");
    }

    if json_output {
        let messages: Vec<serde_json::Value> = conversation_manager
            .get_messages()
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                json!({
                    "role": role,
                    "content": msg.text().unwrap_or(""),
                })
            })
            .collect();

        let output = json!({
            "conversation_id": conversation_manager.conversation_id(),
            "model": model_id,
            "messages": messages,
        });

        println!("{}", serde_json::to_string_pretty(&output)?);
    } else if !quiet {
        Logger::info("Ending chat session");
    }

    Ok(())
}
