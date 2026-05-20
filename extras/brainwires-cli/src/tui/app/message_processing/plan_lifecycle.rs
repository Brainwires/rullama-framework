//! Plan Lifecycle Management
//!
//! Handles plan lifecycle operations: activate, deactivate, status, pause, resume.

use super::super::state::{App, LogLevel, TuiMessage};
use crate::types::plan::PlanStatus;
use anyhow::Result;
use std::sync::Arc;

impl App {
    /// Handle activate plan command
    pub(super) async fn handle_activate_plan(&mut self, plan_id: String) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};
        use crate::utils::plan_parser::{parse_plan_steps, steps_to_tasks};

        // Initialize plan store
        let db_path = PlatformPaths::conversations_db_path()?;
        let client = Arc::new(
            LanceDatabase::new(
                db_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?,
            )
            .await?,
        );
        let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
        client.initialize(embeddings.dimension()).await?;
        let plan_store = PlanStore::new(client.clone(), embeddings);

        // Try to find plan by full ID or partial ID
        let plan = if plan_id.len() < 36 {
            let all_plans = plan_store.list_recent(100).await?;
            all_plans
                .into_iter()
                .find(|p| p.plan_id.starts_with(&plan_id))
        } else {
            plan_store.get(&plan_id).await?
        };

        let content = if let Some(mut plan) = plan {
            // Update plan status to Active
            plan.set_status(PlanStatus::Active);
            let _ = plan_store.update(&plan).await;

            // Parse plan content into tasks
            let steps = parse_plan_steps(&plan.plan_content);
            let tasks = steps_to_tasks(&steps, &plan.plan_id);

            // Load tasks into task manager
            {
                let task_mgr = self.task_manager.write().await;
                task_mgr.clear().await;
                task_mgr.load_tasks(tasks.clone()).await;
            }

            // Persist tasks to storage
            for task in &tasks {
                let _ = self.task_store.save(task, &self.session_id).await;
            }

            // Update task cache
            self.update_task_cache().await;

            let plan_summary = format!(
                "Plan activated: {}\n\n\
                 Task: {}\n\
                 Steps: {} tasks created from plan\n\n\
                 The agent will now follow this plan.\n\
                 Use /tasks to see task list, /plan:current for status.",
                &plan.plan_id[..8],
                plan.task_description,
                tasks.len()
            );

            self.active_plan = Some(plan);
            self.completed_plan_steps.clear();
            self.set_status(LogLevel::Info, "Plan active");

            plan_summary
        } else {
            format!("Plan not found: {}", plan_id)
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle deactivate plan command
    pub(super) fn handle_deactivate_plan(&mut self) {
        let content = if let Some(ref plan) = self.active_plan {
            let msg = format!("Plan deactivated: {}", &plan.plan_id[..8]);
            self.active_plan = None;
            self.completed_plan_steps.clear();
            self.plan_progress = None;
            self.set_status(LogLevel::Info, "Ready");
            msg
        } else {
            "No active plan to deactivate".to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle plan status command
    pub(super) fn handle_plan_status(&mut self) {
        let content = if let Some(ref plan) = self.active_plan {
            let completed = self.completed_plan_steps.len();
            format!(
                "Active Plan: {}\n\
                 Status: {}\n\
                 Task: {}\n\
                 Progress: {} steps completed\n\n\
                 ---\n\n\
                 {}",
                &plan.plan_id[..8],
                plan.status,
                plan.task_description,
                completed,
                plan.plan_content
            )
        } else {
            "No active plan.\n\n\
             Use /plans to list saved plans, then /plan:activate <id> to set one."
                .to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle pause plan command - saves current task state and pauses plan
    pub(super) async fn handle_pause_plan(&mut self) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        let content = if let Some(ref mut plan) = self.active_plan {
            // Persist all current task states before pausing
            let tasks = {
                let task_mgr = self.task_manager.read().await;
                task_mgr.get_all_tasks().await
            };

            for task in &tasks {
                let _ = self.task_store.save(task, &self.session_id).await;
            }

            // Update plan status to Paused
            let db_path = PlatformPaths::conversations_db_path()?;
            let client = Arc::new(
                LanceDatabase::new(
                    db_path
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?,
                )
                .await?,
            );
            let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
            client.initialize(embeddings.dimension()).await?;
            let plan_store = PlanStore::new(client, embeddings);

            plan.set_status(PlanStatus::Paused);
            let _ = plan_store.update(plan).await;

            let completed_count = tasks
                .iter()
                .filter(|t| t.status == crate::types::agent::TaskStatus::Completed)
                .count();
            let total_count = tasks.len();

            let msg = format!(
                "Plan paused: {}\n\
                 Progress: {}/{} tasks completed\n\n\
                 Task state has been saved. Use /plan:resume {} to continue.",
                &plan.plan_id[..8],
                completed_count,
                total_count,
                &plan.plan_id[..8]
            );

            // Clear active plan but keep tasks in storage
            self.active_plan = None;
            self.plan_progress = None;
            self.set_status(LogLevel::Info, "Ready");

            msg
        } else {
            "No active plan to pause.".to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle resume plan command - restores task state and reactivates plan
    pub(super) async fn handle_resume_plan(&mut self, plan_id: String) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        // Initialize stores
        let db_path = PlatformPaths::conversations_db_path()?;
        let client = Arc::new(
            LanceDatabase::new(
                db_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?,
            )
            .await?,
        );
        let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
        client.initialize(embeddings.dimension()).await?;
        let plan_store = PlanStore::new(client.clone(), embeddings);

        // Try to find plan by full ID or partial ID
        let plan = if plan_id.len() < 36 {
            let all_plans = plan_store.list_recent(100).await?;
            all_plans
                .into_iter()
                .find(|p| p.plan_id.starts_with(&plan_id))
        } else {
            plan_store.get(&plan_id).await?
        };

        let content = if let Some(mut plan) = plan {
            // Load saved tasks for this plan from storage
            let saved_tasks = self
                .task_store
                .get_by_plan(&plan.plan_id)
                .await
                .unwrap_or_default();

            if saved_tasks.is_empty() {
                // No saved tasks - parse from plan content
                use crate::utils::plan_parser::{parse_plan_steps, steps_to_tasks};
                let steps = parse_plan_steps(&plan.plan_content);
                let tasks = steps_to_tasks(&steps, &plan.plan_id);

                {
                    let task_mgr = self.task_manager.write().await;
                    task_mgr.clear().await;
                    task_mgr.load_tasks(tasks.clone()).await;
                }

                for task in &tasks {
                    let _ = self.task_store.save(task, &self.session_id).await;
                }
            } else {
                // Restore saved task state
                {
                    let task_mgr = self.task_manager.write().await;
                    task_mgr.clear().await;
                    task_mgr.load_tasks(saved_tasks.clone()).await;
                }
            }

            // Update plan status to Active
            plan.set_status(PlanStatus::Active);
            let _ = plan_store.update(&plan).await;

            // Update task cache
            self.update_task_cache().await;

            let task_stats = {
                let task_mgr = self.task_manager.read().await;
                task_mgr.get_stats().await
            };

            let msg = format!(
                "Plan resumed: {}\n\n\
                 Task: {}\n\
                 Progress: {}/{} tasks completed, {} in progress\n\n\
                 Use /tasks to see current state.",
                &plan.plan_id[..8],
                plan.task_description,
                task_stats.completed,
                task_stats.total,
                task_stats.in_progress
            );

            self.active_plan = Some(plan);
            self.completed_plan_steps.clear();
            self.set_status(LogLevel::Info, "Plan active");

            msg
        } else {
            format!("Plan not found: {}", plan_id)
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }
}
