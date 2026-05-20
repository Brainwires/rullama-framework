use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};

use crate::agents::AgentManager;
use crate::auth::SessionManager;
use crate::config::{ConfigManager, ModelRegistry};
use crate::providers::ProviderFactory;
use crate::types::agent::{AgentContext, PermissionMode};
use crate::utils::logger::Logger;
use crate::utils::rich_output::RichOutput;

pub async fn handle_task(
    prompt: String,
    model: Option<String>,
    provider: Option<String>,
) -> Result<()> {
    // Load configuration and session
    let config_manager = ConfigManager::new()?;
    let session = SessionManager::load()?;

    // Resolve provider (CLI flag > env > config) and model.
    let active_provider = ProviderFactory::effective_provider(
        provider.as_deref(),
        config_manager.get().provider_type,
    )?;
    let model_id = match model {
        Some(m) => m,
        None => ModelRegistry::default_model().await,
    };

    Logger::info(format!(
        "Executing task with {} via {}",
        model_id,
        active_provider.as_str()
    ));

    // Create provider using factory (requires active session)
    let factory = ProviderFactory;
    let provider_instance = factory
        .create_with_overrides(model_id.clone(), Some(active_provider), None)
        .await
        .context("Failed to create provider — run `brainwires auth status` to diagnose")?;

    // Create agent manager with Full permission mode for quick tasks
    let agent_manager = AgentManager::new(
        provider_instance,
        PermissionMode::Full,
        3, // Fewer workers for quick tasks
    )
    .await?;

    // Initialize agent context
    let user_id = session.as_ref().map(|s| s.user.user_id.clone());

    let mut context = AgentContext {
        working_directory: std::env::current_dir()?.to_string_lossy().to_string(),
        user_id,
        conversation_history: Vec::new(),
        tools: brainwires_tool_builtins::registry_with_builtins()
            .get_all()
            .to_vec(),
        metadata: std::collections::HashMap::new(),
        working_set: crate::types::WorkingSet::new(),
        capabilities: brainwires::permissions::AgentCapabilities::full_access(),
    };

    // Print header
    println!("\n{}", RichOutput::header("Brainwires Task", "blue"));
    println!(
        "Model: {} (provider: {})",
        model_id,
        active_provider.as_str()
    );
    println!("Task: {}\n", console::style(&prompt).cyan());

    // Show execution indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    spinner.set_message("Executing task...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Execute task
    let response = agent_manager.execute_task(&prompt, &mut context).await;

    spinner.finish_and_clear();

    match response {
        Ok(agent_response) => {
            // Print result
            println!("\n{}\n", console::style("Result:").green().bold());

            // Print with typing effect
            for chunk in agent_response.message.chars() {
                print!("{}", chunk);
                io::stdout().flush()?;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            println!("\n");

            // Show statistics
            if !agent_response.tasks.is_empty() {
                use crate::types::agent::TaskStatus;
                let completed = agent_response
                    .tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Completed)
                    .count();

                println!(
                    "{} (Iterations: {}, Tasks: {}/{})",
                    console::style("Stats").dim(),
                    agent_response.iterations,
                    completed,
                    agent_response.tasks.len()
                );
            }

            Logger::info("Task completed successfully");
        }
        Err(e) => {
            Logger::error(format!("Task error: {}", e));
            println!("\n{}: {}", console::style("Error").red().bold(), e);
        }
    }

    Ok(())
}
