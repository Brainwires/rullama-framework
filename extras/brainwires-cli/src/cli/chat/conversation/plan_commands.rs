//! Plan Commands
//!
//! Handlers for plan management commands.

use anyhow::Result;
use std::sync::Arc;

use crate::commands::executor::CommandAction;

/// Handle plan-related command actions
pub async fn handle_plan_action(action: CommandAction) -> Result<bool> {
    match action {
        CommandAction::ListPlans(conversation_id) => {
            use crate::config::PlatformPaths;
            use crate::storage::{
                CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase,
            };

            match PlatformPaths::conversations_db_path() {
                Ok(db_path) => {
                    if let Some(db_str) = db_path.to_str() {
                        match LanceDatabase::new(db_str).await {
                            Ok(client) => {
                                let client = Arc::new(client);
                                let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
                                let _ = client.initialize(embeddings.dimension()).await;
                                let plan_store = PlanStore::new(client, embeddings);

                                let plans = if let Some(conv_id) = conversation_id {
                                    plan_store
                                        .get_by_conversation(&conv_id)
                                        .await
                                        .unwrap_or_default()
                                } else {
                                    plan_store.list_recent(20).await.unwrap_or_default()
                                };

                                if plans.is_empty() {
                                    println!("{}\n",
                                        console::style("No plans found. Create a plan with: /plan <task description>").yellow());
                                } else {
                                    println!("{}\n", console::style("Saved Plans:").cyan().bold());
                                    for plan in plans {
                                        let created =
                                            chrono::DateTime::from_timestamp(plan.created_at, 0)
                                                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                                .unwrap_or_else(|| "Unknown".to_string());
                                        let title_preview = if plan.title.len() > 50 {
                                            format!("{}...", &plan.title[..47])
                                        } else {
                                            plan.title.clone()
                                        };
                                        println!(
                                            "  {} - {} [{}]",
                                            console::style(&plan.plan_id[..8]).green(),
                                            title_preview,
                                            console::style(plan.status.to_string()).dim()
                                        );
                                        println!(
                                            "    Created: {} | Iterations: {}",
                                            console::style(created).dim(),
                                            plan.iterations_used
                                        );
                                    }
                                    println!("\nView a plan: /plan:show <plan_id>\n");
                                }
                            }
                            Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
                        }
                    }
                }
                Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
            }
            Ok(true)
        }
        CommandAction::ShowPlan(plan_id) => {
            use crate::config::PlatformPaths;
            use crate::storage::{
                CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase,
            };

            match PlatformPaths::conversations_db_path() {
                Ok(db_path) => {
                    if let Some(db_str) = db_path.to_str() {
                        match LanceDatabase::new(db_str).await {
                            Ok(client) => {
                                let client = Arc::new(client);
                                let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
                                let _ = client.initialize(embeddings.dimension()).await;
                                let plan_store = PlanStore::new(client, embeddings);

                                // Try partial ID match
                                let plan = if plan_id.len() < 36 {
                                    let all_plans =
                                        plan_store.list_recent(100).await.unwrap_or_default();
                                    all_plans
                                        .into_iter()
                                        .find(|p| p.plan_id.starts_with(&plan_id))
                                } else {
                                    plan_store.get(&plan_id).await.unwrap_or(None)
                                };

                                if let Some(plan) = plan {
                                    let created =
                                        chrono::DateTime::from_timestamp(plan.created_at, 0)
                                            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                                            .unwrap_or_else(|| "Unknown".to_string());

                                    println!("{}\n", console::style("Plan Details:").cyan().bold());
                                    println!("  ID: {}", console::style(&plan.plan_id).green());
                                    println!("  Status: {}", plan.status);
                                    println!("  Created: {}", created);
                                    println!("  Iterations: {}", plan.iterations_used);
                                    println!("  Task: {}\n", plan.task_description);
                                    println!("{}", console::style("---").dim());
                                    println!("\n{}\n", plan.plan_content);
                                } else {
                                    println!(
                                        "{}\n",
                                        console::style(format!("Plan not found: {}", plan_id))
                                            .yellow()
                                    );
                                }
                            }
                            Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
                        }
                    }
                }
                Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
            }
            Ok(true)
        }
        CommandAction::DeletePlan(plan_id) => {
            use crate::config::PlatformPaths;
            use crate::storage::{
                CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase,
            };

            match PlatformPaths::conversations_db_path() {
                Ok(db_path) => {
                    if let Some(db_str) = db_path.to_str() {
                        match LanceDatabase::new(db_str).await {
                            Ok(client) => {
                                let client = Arc::new(client);
                                let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
                                let _ = client.initialize(embeddings.dimension()).await;
                                let plan_store = PlanStore::new(client, embeddings);

                                // Try partial ID match
                                let full_plan_id = if plan_id.len() < 36 {
                                    let all_plans =
                                        plan_store.list_recent(100).await.unwrap_or_default();
                                    all_plans
                                        .into_iter()
                                        .find(|p| p.plan_id.starts_with(&plan_id))
                                        .map(|p| p.plan_id)
                                } else {
                                    Some(plan_id.clone())
                                };

                                if let Some(id) = full_plan_id {
                                    match plan_store.delete(&id).await {
                                        Ok(()) => println!(
                                            "{}\n",
                                            console::style(format!("Plan deleted: {}", &id[..8]))
                                                .green()
                                        ),
                                        Err(e) => println!(
                                            "{}: {}\n",
                                            console::style("Error").red().bold(),
                                            e
                                        ),
                                    }
                                } else {
                                    println!(
                                        "{}\n",
                                        console::style(format!("Plan not found: {}", plan_id))
                                            .yellow()
                                    );
                                }
                            }
                            Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
                        }
                    }
                }
                Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
            }
            Ok(true)
        }
        CommandAction::SearchPlans(query) => {
            use crate::config::PlatformPaths;
            use crate::storage::{
                CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase,
            };

            match PlatformPaths::conversations_db_path() {
                Ok(db_path) => {
                    if let Some(db_str) = db_path.to_str() {
                        match LanceDatabase::new(db_str).await {
                            Ok(client) => {
                                let client = Arc::new(client);
                                let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
                                let _ = client.initialize(embeddings.dimension()).await;
                                let plan_store = PlanStore::new(client, embeddings);

                                match plan_store.search(&query, 20).await {
                                    Ok(plans) => {
                                        if plans.is_empty() {
                                            println!(
                                                "{}\n",
                                                console::style(format!(
                                                    "No plans found matching '{}'",
                                                    query
                                                ))
                                                .yellow()
                                            );
                                        } else {
                                            println!(
                                                "{}\n",
                                                console::style(format!(
                                                    "Search results for '{}':",
                                                    query
                                                ))
                                                .cyan()
                                                .bold()
                                            );
                                            for plan in &plans {
                                                println!(
                                                    "  {} - {} [{}]",
                                                    console::style(&plan.plan_id[..8]).green(),
                                                    if plan.title.len() > 40 {
                                                        format!("{}...", &plan.title[..37])
                                                    } else {
                                                        plan.title.clone()
                                                    },
                                                    plan.status
                                                );
                                            }
                                            println!();
                                        }
                                    }
                                    Err(e) => println!(
                                        "{}: {}\n",
                                        console::style("Error").red().bold(),
                                        e
                                    ),
                                }
                            }
                            Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
                        }
                    }
                }
                Err(e) => println!("{}: {}\n", console::style("Error").red().bold(), e),
            }
            Ok(true)
        }
        CommandAction::ActivatePlan(_) => {
            println!(
                "{}\n",
                console::style(
                    "Plan activation is only available in TUI mode (--tui).\n\
                Use the TUI for interactive plan execution."
                )
                .yellow()
            );
            Ok(true)
        }
        CommandAction::DeactivatePlan => {
            println!(
                "{}\n",
                console::style("Plan deactivation is only available in TUI mode (--tui).").yellow()
            );
            Ok(true)
        }
        CommandAction::PlanStatus => {
            println!(
                "{}\n",
                console::style(
                    "Plan status tracking is only available in TUI mode (--tui).\n\
                Use /plans to list saved plans."
                )
                .yellow()
            );
            Ok(true)
        }
        CommandAction::PausePlan => {
            println!(
                "{}\n",
                console::style("Plan pause is only available in TUI mode (--tui).").yellow()
            );
            Ok(true)
        }
        CommandAction::ResumePlan(_) => {
            println!(
                "{}\n",
                console::style("Plan resume is only available in TUI mode (--tui).").yellow()
            );
            Ok(true)
        }
        CommandAction::BranchPlan(_, _)
        | CommandAction::MergePlan(_)
        | CommandAction::PlanTree(_) => {
            println!(
                "{}\n",
                console::style("Plan branching commands are only available in TUI mode (--tui).")
                    .yellow()
            );
            Ok(true)
        }
        _ => Ok(true),
    }
}
