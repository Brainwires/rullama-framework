//! Plan Execution
//!
//! Handles plan execution: starting automated plan execution with various approval modes.

use super::super::state::{App, LogLevel, TuiMessage};
use crate::types::plan::PlanStatus;
use anyhow::Result;
use std::sync::Arc;

impl App {
    /// Handle execute plan command - activate plan and start execution
    pub(super) async fn handle_execute_plan(
        &mut self,
        plan_id: String,
        mode: Option<String>,
    ) -> Result<()> {
        use crate::agents::{ExecutionApprovalMode, PlanExecutionConfig, PlanExecutorAgent};
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};
        use crate::utils::plan_parser::{parse_plan_steps, steps_to_tasks};

        // Parse approval mode
        let approval_mode = if let Some(m) = mode {
            m.parse::<ExecutionApprovalMode>()
                .unwrap_or(ExecutionApprovalMode::FullAuto)
        } else {
            ExecutionApprovalMode::FullAuto
        };

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

            // Create plan executor
            let config = PlanExecutionConfig {
                approval_mode,
                auto_advance: true,
                stop_on_error: true,
                ..Default::default()
            };
            let _executor = PlanExecutorAgent::new(plan.clone(), self.task_manager.clone(), config);

            // Start first task
            let first_task = {
                let task_mgr = self.task_manager.read().await;
                let ready = task_mgr.get_ready_tasks().await;
                ready.into_iter().next()
            };

            if let Some(task) = first_task {
                {
                    let task_mgr = self.task_manager.write().await;
                    let _ = task_mgr.start_task(&task.id).await;
                }
                self.update_task_cache().await;
            }

            self.active_plan = Some(plan.clone());
            self.completed_plan_steps.clear();

            // Set approval mode in TUI state (so completion detector respects it)
            self.approval_mode = match approval_mode {
                ExecutionApprovalMode::Suggest => super::super::state::ApprovalMode::Suggest,
                ExecutionApprovalMode::AutoEdit => super::super::state::ApprovalMode::AutoEdit,
                ExecutionApprovalMode::FullAuto => super::super::state::ApprovalMode::FullAuto,
            };

            self.set_status(
                LogLevel::Info,
                format!("Executing plan ({})", approval_mode),
            );

            // Generate initial prompt to start execution
            let stats = {
                let task_mgr = self.task_manager.read().await;
                task_mgr.get_stats().await
            };

            format!(
                "Plan execution started: {}\n\n\
                 Task: {}\n\
                 Mode: {}\n\
                 Tasks: {} total ({} ready to execute)\n\n\
                 The AI will now work through the plan. Progress is tracked automatically.\n\
                 Use /tasks to see current progress, /task:complete to manually complete tasks.\n\n\
                 Starting first task...",
                &plan.plan_id[..8],
                plan.task_description,
                approval_mode,
                tasks.len(),
                stats.pending
            )
        } else {
            format!("Plan not found: {}", plan_id)
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();

        // Queue a message to start execution if we have an active plan
        if self.active_plan.is_some() {
            let first_task = {
                let task_mgr = self.task_manager.read().await;
                task_mgr
                    .get_tasks_by_status(crate::types::agent::TaskStatus::InProgress)
                    .await
            };

            if let Some(task) = first_task.first() {
                // Queue a prompt to start working on the first task
                self.queued_messages.push(format!(
                    "Please begin working on this task from the plan: {}",
                    task.description
                ));
            }
        }

        Ok(())
    }
}
