//! Prompt Mode
//!
//! Single-shot prompt mode handler.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;

use crate::auth::SessionManager;
use crate::cli::chat::streaming::process_chat_stream;
use crate::config::ConfigManager;
use crate::providers::ProviderFactory;
use crate::types::agent::AgentContext;
use crate::types::message::{Message, MessageContent, Role};
use crate::utils::logger::Logger;
use crate::utils::system_prompt::build_system_prompt;

/// Handle single-shot prompt mode
pub async fn handle_prompt_mode(
    model: Option<String>,
    provider: Option<String>,
    system: Option<String>,
    prompt: String,
    quiet: bool,
    format: &str,
    backend_url_override: Option<String>,
) -> Result<()> {
    // Reject empty/whitespace prompts client-side so users get a clean error
    // instead of a leaky backend 4xx that mentions internal field names.
    if prompt.trim().is_empty() {
        return Err(anyhow::anyhow!("--prompt cannot be empty"));
    }

    // Load configuration and session
    let config_manager = ConfigManager::new()?;
    let session = SessionManager::load()?;

    // Resolve provider (CLI flag > env > config) and model (CLI flag > config).
    let config = config_manager.get();
    let active_provider =
        ProviderFactory::effective_provider(provider.as_deref(), config.provider_type)?;
    let model_id = match model {
        Some(m) => m,
        None => config.model.clone(),
    };

    if !quiet && format != "json" {
        // Route progress to stderr so --format json / piped stdout stays clean for jq.
        if let Some(ref url) = backend_url_override {
            eprintln!(
                "{} Processing prompt with {} via {} (dev backend: {})",
                console::style("ℹ").blue(),
                model_id,
                active_provider.as_str(),
                url
            );
        } else {
            eprintln!(
                "{} Processing prompt with {} via {}",
                console::style("ℹ").blue(),
                model_id,
                active_provider.as_str()
            );
        }
    }

    // Create provider with optional backend URL override
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

    // Build system message
    let system_prompt = build_system_prompt(system)?;
    let sys_message = Message {
        role: Role::System,
        content: MessageContent::Text(system_prompt),
        name: None,
        metadata: None,
    };
    context.conversation_history.push(sys_message);

    // Add user prompt
    let user_message = Message {
        role: Role::User,
        content: MessageContent::Text(prompt),
        name: None,
        metadata: None,
    };
    context.conversation_history.push(user_message);

    // Process the request
    let spinner = if !quiet {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        spinner.set_message("Thinking...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));
        Some(spinner)
    } else {
        None
    };

    let response_text =
        process_chat_stream(&provider_instance, &context, &spinner, &model_id, None).await;

    if let Some(s) = spinner {
        s.finish_and_clear();
    }

    match response_text {
        Ok(text) => {
            // Output based on format
            match format {
                "plain" => {
                    println!("{}", text);
                }
                "json" => {
                    let output = json!({
                        "model": model_id,
                        "response": text,
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                _ => {
                    // Full format
                    if !quiet {
                        println!(
                            "\n{}: {}\n",
                            console::style("Assistant").green().bold(),
                            text
                        );
                    } else {
                        println!("{}", text);
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            if !quiet {
                Logger::error(format!("Error: {}", e));
            }
            eprintln!("{}: {}", console::style("Error").red().bold(), e);
            Err(e)
        }
    }
}

/// Handle single-shot prompt mode with MDAP
#[allow(clippy::too_many_arguments)]
pub async fn handle_prompt_mode_mdap(
    model: Option<String>,
    provider: Option<String>,
    system: Option<String>,
    prompt: String,
    quiet: bool,
    format: &str,
    mdap_config: Option<crate::mdap::MdapConfig>,
    backend_url_override: Option<String>,
) -> Result<()> {
    use crate::agents::OrchestratorAgent;
    use crate::types::agent::PermissionMode;

    // Reject empty/whitespace prompts client-side (same rationale as handle_prompt_mode).
    if prompt.trim().is_empty() {
        return Err(anyhow::anyhow!("--prompt cannot be empty"));
    }

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

    // Check if MDAP is actually enabled
    let mdap_config = match mdap_config {
        Some(c) => c,
        None => {
            // Fall back to regular prompt mode
            return handle_prompt_mode(
                Some(model_id),
                provider,
                system,
                prompt,
                quiet,
                format,
                backend_url_override,
            )
            .await;
        }
    };

    if !quiet && format != "json" {
        // Route progress to stderr so --format json / piped stdout stays clean for jq.
        if let Some(ref url) = backend_url_override {
            eprintln!(
                "{} Processing prompt with {} via {} in MDAP mode (dev backend: {})",
                console::style("ℹ").blue(),
                model_id,
                active_provider.as_str(),
                url
            );
        } else {
            eprintln!(
                "{} Processing prompt with {} via {} in MDAP mode",
                console::style("ℹ").blue(),
                model_id,
                active_provider.as_str()
            );
        }
        eprintln!(
            "{} MDAP config: k={}, target={}%",
            console::style("ℹ").blue(),
            mdap_config.k,
            mdap_config.target_success_rate * 100.0
        );
    }

    // Create provider with optional backend URL override
    let factory = ProviderFactory;
    let provider_instance = factory
        .create_with_overrides(
            model_id.clone(),
            Some(active_provider),
            backend_url_override,
        )
        .await
        .context("Failed to create provider — run `brainwires auth status` to diagnose")?;

    // Initialize agent context with core tools
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

    // Build system message
    let system_prompt = build_system_prompt(system)?;
    let sys_message = Message {
        role: Role::System,
        content: MessageContent::Text(system_prompt),
        name: None,
        metadata: None,
    };
    context.conversation_history.push(sys_message);

    // Create orchestrator and execute with MDAP
    let mut orchestrator = OrchestratorAgent::new(provider_instance, PermissionMode::Auto);

    let spinner = if !quiet {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        spinner.set_message("Processing with MDAP...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));
        Some(spinner)
    } else {
        None
    };

    let result = orchestrator
        .execute_mdap(&prompt, &mut context, mdap_config)
        .await;

    if let Some(s) = spinner {
        s.finish_and_clear();
    }

    match result {
        Ok((response, metrics)) => {
            // Output based on format
            match format {
                "plain" => {
                    println!("{}", response.message);
                }
                "json" => {
                    let output = json!({
                        "model": model_id,
                        "response": response.message,
                        "mdap": {
                            "success": metrics.final_success,
                            "steps_completed": metrics.completed_steps,
                            "total_samples": metrics.total_samples,
                            "red_flagged": metrics.red_flagged_samples,
                            "cost_usd": metrics.actual_cost_usd,
                            "time_seconds": metrics.total_time_seconds,
                        }
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                _ => {
                    // Full format
                    if !quiet {
                        println!(
                            "\n{}: {}\n",
                            console::style("Assistant").green().bold(),
                            response.message
                        );
                        println!("{}", console::style("MDAP Metrics:").cyan().bold());
                        println!("{}", metrics.summary());
                    } else {
                        println!("{}", response.message);
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            if !quiet {
                Logger::error(format!("MDAP execution error: {}", e));
            }
            eprintln!("{}: {}", console::style("Error").red().bold(), e);
            Err(e)
        }
    }
}
