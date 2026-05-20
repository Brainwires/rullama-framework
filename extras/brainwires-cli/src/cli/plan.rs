use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use clap::Subcommand;
use console::style;
use dialoguer::{Confirm, Editor, theme::ColorfulTheme};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use std::sync::Arc;

use crate::agents::AgentManager;
use crate::auth::SessionManager;
use crate::config::{ConfigManager, ModelRegistry, PlatformPaths};
use crate::providers::ProviderFactory;
use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};
use crate::types::agent::{AgentContext, PermissionMode};
use crate::types::plan::{PlanMetadata, PlanStatus};
use crate::utils::entity_extraction::EntityExtractor;
use crate::utils::logger::Logger;
use crate::utils::rich_output::RichOutput;

/// Plan subcommands
#[derive(Subcommand)]
pub enum PlanCommands {
    /// Create a new execution plan for a task
    Create {
        /// The task to create a plan for
        task: String,

        #[arg(short, long)]
        model: Option<String>,

        #[arg(short, long)]
        provider: Option<String>,

        /// Execute the plan immediately after creation
        #[arg(short, long)]
        execute: bool,
    },

    /// List saved plans
    List {
        /// Filter by conversation ID
        #[arg(short, long)]
        conversation: Option<String>,

        /// Maximum number of plans to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Show a specific plan
    Show {
        /// Plan ID to show
        plan_id: String,
    },

    /// Export a plan to markdown
    Export {
        /// Plan ID to export
        plan_id: String,

        /// Output path (defaults to plans directory)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Delete a plan
    Delete {
        /// Plan ID to delete
        plan_id: String,

        /// Skip confirmation
        #[arg(short, long)]
        confirm: bool,
    },

    /// Edit a plan in your editor
    Edit {
        /// Plan ID to edit
        plan_id: String,
    },
}

/// Handle plan subcommands
pub async fn handle_plan_command(cmd: PlanCommands) -> Result<()> {
    match cmd {
        PlanCommands::Create {
            task,
            model,
            provider,
            execute,
        } => handle_plan(task, model, provider, execute).await,
        PlanCommands::List {
            conversation,
            limit,
        } => handle_list(conversation, limit).await,
        PlanCommands::Show { plan_id } => handle_show(&plan_id).await,
        PlanCommands::Export { plan_id, output } => handle_export(&plan_id, output).await,
        PlanCommands::Delete { plan_id, confirm } => handle_delete(&plan_id, confirm).await,
        PlanCommands::Edit { plan_id } => handle_edit(&plan_id).await,
    }
}

/// Create a new plan (legacy handler)
pub async fn handle_plan(
    task: String,
    model: Option<String>,
    _provider: Option<String>,
    execute: bool,
) -> Result<()> {
    // Load configuration and session
    let _config_manager = ConfigManager::new()?;
    let session = SessionManager::load()?;

    // Resolve model (provider is always Brainwires)
    let model_id = match model {
        Some(m) => m,
        None => ModelRegistry::default_model().await,
    };

    Logger::info(format!("Planning task with {}", model_id));

    // Create provider using factory (requires active session)
    let factory = ProviderFactory;
    let provider_instance = factory
        .create(model_id.clone())
        .await
        .context("Failed to create provider — run `brainwires auth status` to diagnose")?;

    // Create agent manager
    let agent_manager = AgentManager::new(provider_instance, PermissionMode::Auto, 5).await?;

    // Initialize agent context with planning system prompt
    let user_id = session.as_ref().map(|s| s.user.user_id.clone());

    let planning_prompt = format!(
        "You are an expert planning assistant. Create a detailed, step-by-step execution plan for the following task:\n\n{}\n\n\
        Your plan should:\n\
        1. Break down the task into clear, actionable steps\n\
        2. Identify dependencies between steps\n\
        3. Consider potential challenges or edge cases\n\
        4. Suggest tools or approaches for each step\n\
        5. Provide the plan in a clear, numbered format\n\n\
        Focus on creating a comprehensive plan rather than executing it.",
        task
    );

    let mut context = AgentContext {
        working_directory: std::env::current_dir()?.to_string_lossy().to_string(),
        user_id,
        conversation_history: Vec::new(),
        tools: brainwires_tool_builtins::registry_with_builtins()
            .get_all()
            .to_vec(),
        metadata: std::collections::HashMap::new(),
        working_set: crate::types::WorkingSet::new(),
        capabilities: brainwires::permissions::AgentCapabilities::standard_dev(),
    };

    // Print header
    println!("\n{}", RichOutput::header("Brainwires Plan", "magenta"));
    println!("Model: {} (brainwires)", model_id);
    println!("Task: {}\n", console::style(&task).cyan());

    // Show planning indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.magenta} {msg}")
            .unwrap(),
    );
    spinner.set_message("Creating execution plan...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Generate plan
    let response = agent_manager
        .execute_task(&planning_prompt, &mut context)
        .await;

    spinner.finish_and_clear();

    match response {
        Ok(agent_response) => {
            // Print plan
            println!("\n{}\n", console::style("Execution Plan:").green().bold());
            println!("{}\n", agent_response.message);

            // Show statistics
            if !agent_response.tasks.is_empty() {
                use crate::types::agent::TaskStatus;
                let completed = agent_response
                    .tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Completed)
                    .count();

                println!(
                    "{} (Iterations: {}, Planning Tasks: {}/{})\n",
                    console::style("Stats").dim(),
                    agent_response.iterations,
                    completed,
                    agent_response.tasks.len()
                );
            }

            // Save the plan to database and file
            let saved_plan = save_plan(
                &task,
                &agent_response.message,
                &model_id,
                agent_response.iterations,
            )
            .await;

            if let Ok((plan_id, file_path)) = &saved_plan {
                println!(
                    "{} Plan ID: {}",
                    console::style("Saved:").green(),
                    console::style(plan_id).cyan()
                );
                println!(
                    "{} {}",
                    console::style("File:").green(),
                    console::style(file_path).dim()
                );
                println!();
            }

            // Ask if user wants to execute the plan
            if execute
                || Confirm::with_theme(&ColorfulTheme::default())
                    .with_prompt("Execute this plan now?")
                    .default(false)
                    .interact()?
            {
                println!("\n{}\n", console::style("Executing Plan:").green().bold());

                // Show execution indicator
                let exec_spinner = ProgressBar::new_spinner();
                exec_spinner.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {msg}")
                        .unwrap(),
                );
                exec_spinner.set_message("Executing plan...");
                exec_spinner.enable_steady_tick(std::time::Duration::from_millis(100));

                // Execute the original task
                let exec_response = agent_manager.execute_task(&task, &mut context).await;

                exec_spinner.finish_and_clear();

                match exec_response {
                    Ok(exec_result) => {
                        println!("\n{}\n", console::style("Execution Result:").green().bold());

                        // Print with typing effect
                        for chunk in exec_result.message.chars() {
                            print!("{}", chunk);
                            io::stdout().flush()?;
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        println!("\n");

                        // Show execution statistics
                        if !exec_result.tasks.is_empty() {
                            use crate::types::agent::TaskStatus;
                            let completed = exec_result
                                .tasks
                                .iter()
                                .filter(|t| t.status == TaskStatus::Completed)
                                .count();

                            println!(
                                "{} (Iterations: {}, Tasks: {}/{})",
                                console::style("Execution Stats").dim(),
                                exec_result.iterations,
                                completed,
                                exec_result.tasks.len()
                            );
                        }
                    }
                    Err(e) => {
                        Logger::error(format!("Execution error: {}", e));
                        println!(
                            "\n{}: {}",
                            console::style("Execution Error").red().bold(),
                            e
                        );
                    }
                }
            } else if saved_plan.is_ok() {
                println!(
                    "{}",
                    console::style("Plan saved. Use 'brainwires plan list' to view saved plans.")
                        .dim()
                );
            }
        }
        Err(e) => {
            Logger::error(format!("Planning error: {}", e));
            println!("\n{}: {}", console::style("Error").red().bold(), e);
        }
    }

    Ok(())
}

/// Save a plan to the database and export to markdown
async fn save_plan(
    task: &str,
    plan_content: &str,
    model_id: &str,
    iterations: u32,
) -> Result<(String, String)> {
    // Initialize storage
    let db_path = PlatformPaths::conversations_db_path()?;
    let client = Arc::new(
        LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
            .await
            .context("Failed to create LanceDatabase")?,
    );

    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client.initialize(embeddings.dimension()).await?;

    let plan_store = PlanStore::new(client, embeddings);

    // Create plan metadata
    // Generate a conversation ID for standalone plans
    let conversation_id = uuid::Uuid::new_v4().to_string();
    let mut plan = PlanMetadata::new(conversation_id, task.to_string(), plan_content.to_string());
    plan = plan
        .with_model(model_id.to_string())
        .with_iterations(iterations);
    plan.set_status(PlanStatus::Active);

    // Extract entities from plan content for Infinite Context integration
    let entity_extractor = EntityExtractor::new();
    let plan_id = plan.plan_id.clone();
    let extraction = entity_extractor.extract(plan_content, &plan_id);

    // Log extracted entities (for debugging/info)
    if !extraction.entities.is_empty() {
        let entity_names: Vec<_> = extraction
            .entities
            .iter()
            .take(5)
            .map(|(name, _)| name.as_str())
            .collect();
        Logger::debug(format!(
            "Extracted {} entities from plan: {:?}",
            extraction.entities.len(),
            entity_names
        ));
    }

    // Save to database and export to markdown
    let file_path = plan_store.save_and_export(&mut plan).await?;

    Ok((plan.plan_id, file_path.to_string_lossy().to_string()))
}

/// Initialize plan storage
async fn initialize_plan_storage() -> Result<(Arc<LanceDatabase>, PlanStore)> {
    let db_path = PlatformPaths::conversations_db_path()?;
    let client = Arc::new(
        LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
            .await
            .context("Failed to create LanceDatabase")?,
    );

    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client.initialize(embeddings.dimension()).await?;

    let plan_store = PlanStore::new(client.clone(), embeddings);
    Ok((client, plan_store))
}

/// List saved plans
async fn handle_list(conversation: Option<String>, limit: usize) -> Result<()> {
    let (_client, plan_store) = initialize_plan_storage().await?;

    let plans = if let Some(conv_id) = conversation {
        plan_store.get_by_conversation(&conv_id).await?
    } else {
        plan_store.list_recent(limit).await?
    };

    if plans.is_empty() {
        println!("{}", style("No plans found.").yellow());
        return Ok(());
    }

    println!("\n{}\n", style("Saved Plans:").cyan().bold());

    for plan in plans.iter().take(limit) {
        let created_at = DateTime::from_timestamp(plan.created_at, 0)
            .map(|dt| {
                dt.with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string()
            })
            .unwrap_or_else(|| "Unknown".to_string());

        let status_style = match plan.status {
            PlanStatus::Draft => style("draft").dim(),
            PlanStatus::Active => style("active").green(),
            PlanStatus::Paused => style("paused").yellow(),
            PlanStatus::Completed => style("completed").cyan(),
            PlanStatus::Abandoned => style("abandoned").red(),
        };

        println!(
            "{} {} [{}]",
            style(&plan.plan_id[..8]).cyan(),
            style(&plan.title).bold(),
            status_style
        );
        println!(
            "    {} | {} | {}",
            style(&created_at).dim(),
            plan.model_id.as_deref().unwrap_or("unknown"),
            if plan.executed {
                style("executed").green()
            } else {
                style("not executed").dim()
            }
        );
        println!();
    }

    println!(
        "{} {} plans shown",
        style("Total:").dim(),
        plans.len().min(limit)
    );

    Ok(())
}

/// Show a specific plan
async fn handle_show(plan_id: &str) -> Result<()> {
    let (_client, plan_store) = initialize_plan_storage().await?;

    // Try to find by full ID or prefix
    let plan = plan_store.get(plan_id).await?;

    let plan = match plan {
        Some(p) => p,
        None => {
            // Try searching by prefix
            let plans = plan_store.list_recent(100).await?;
            plans
                .into_iter()
                .find(|p| p.plan_id.starts_with(plan_id))
                .ok_or_else(|| anyhow::anyhow!("Plan not found: {}", plan_id))?
        }
    };

    let created_at = DateTime::from_timestamp(plan.created_at, 0)
        .map(|dt| {
            dt.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| "Unknown".to_string());

    println!("\n{}", style("═".repeat(60)).dim());
    println!("{}", style(format!(" Plan: {}", plan.title)).cyan().bold());
    println!("{}", style("═".repeat(60)).dim());

    println!("\n{}: {}", style("ID").dim(), plan.plan_id);
    println!("{}: {}", style("Status").dim(), plan.status);
    println!("{}: {}", style("Created").dim(), created_at);
    println!(
        "{}: {}",
        style("Model").dim(),
        plan.model_id.as_deref().unwrap_or("unknown")
    );
    println!("{}: {}", style("Iterations").dim(), plan.iterations_used);
    println!(
        "{}: {}",
        style("Executed").dim(),
        if plan.executed { "yes" } else { "no" }
    );

    if let Some(ref file_path) = plan.file_path {
        println!("{}: {}", style("File").dim(), file_path);
    }

    println!("\n{}", style("─".repeat(60)).dim());
    println!("{}", style("Original Task:").yellow().bold());
    println!("{}", plan.task_description);

    println!("\n{}", style("─".repeat(60)).dim());
    println!("{}", style("Plan:").green().bold());
    println!("{}", plan.plan_content);

    println!("\n{}", style("═".repeat(60)).dim());

    Ok(())
}

/// Export a plan to markdown
async fn handle_export(plan_id: &str, output: Option<String>) -> Result<()> {
    let (_client, plan_store) = initialize_plan_storage().await?;

    let file_path = if let Some(output_path) = output {
        // Custom output path
        let plan = plan_store
            .get(plan_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Plan not found: {}", plan_id))?;

        let path = std::path::PathBuf::from(&output_path);
        std::fs::write(&path, plan.to_markdown())
            .with_context(|| format!("Failed to write to {}", output_path))?;
        path
    } else {
        // Default export location
        plan_store.export_to_markdown(plan_id).await?
    };

    println!(
        "{} Exported plan to: {}",
        style("✓").green(),
        style(file_path.display()).cyan()
    );

    Ok(())
}

/// Delete a plan
async fn handle_delete(plan_id: &str, confirm: bool) -> Result<()> {
    let (_client, plan_store) = initialize_plan_storage().await?;

    // Verify plan exists
    let plan = plan_store
        .get(plan_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Plan not found: {}", plan_id))?;

    if !confirm {
        println!("\n{}", style("About to delete plan:").yellow());
        println!("  ID: {}", plan.plan_id);
        println!("  Title: {}", plan.title);

        if !Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Are you sure?")
            .default(false)
            .interact()?
        {
            println!("{}", style("Cancelled.").dim());
            return Ok(());
        }
    }

    // Delete from database
    plan_store.delete(plan_id).await?;

    // Delete markdown file if it exists
    if let Some(ref file_path) = plan.file_path {
        let path = std::path::Path::new(file_path);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }

    println!(
        "{} Deleted plan: {}",
        style("✓").green(),
        style(&plan.plan_id[..8]).cyan()
    );

    Ok(())
}

/// Edit a plan in the user's editor
async fn handle_edit(plan_id: &str) -> Result<()> {
    let (_client, plan_store) = initialize_plan_storage().await?;

    let mut plan = plan_store
        .get(plan_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Plan not found: {}", plan_id))?;

    // Open editor with plan content
    let edited = Editor::new().extension(".md").edit(&plan.plan_content)?;

    match edited {
        Some(new_content) => {
            if new_content == plan.plan_content {
                println!("{}", style("No changes made.").dim());
                return Ok(());
            }

            plan.plan_content = new_content;
            plan.updated_at = chrono::Utc::now().timestamp();

            // Save updated plan
            plan_store.save(&plan).await?;

            // Re-export markdown file
            if plan.file_path.is_some() {
                plan_store.export_to_markdown(plan_id).await?;
            }

            println!(
                "{} Updated plan: {}",
                style("✓").green(),
                style(&plan.plan_id[..8]).cyan()
            );
        }
        None => {
            println!("{}", style("Edit cancelled.").dim());
        }
    }

    Ok(())
}
