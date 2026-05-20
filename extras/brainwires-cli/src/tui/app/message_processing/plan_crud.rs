//! Plan CRUD Operations
//!
//! Handles basic plan operations: list, show, delete.

use super::super::state::{App, TuiMessage};
use anyhow::Result;
use std::sync::Arc;

impl App {
    /// Handle list plans command
    pub(super) async fn handle_list_plans(
        &mut self,
        conversation_id: Option<String>,
    ) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

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
        let plan_store = PlanStore::new(client, embeddings);

        // Get plans
        let plans = if let Some(conv_id) = conversation_id {
            plan_store.get_by_conversation(&conv_id).await?
        } else {
            plan_store.list_recent(20).await?
        };

        // Format output
        let content = if plans.is_empty() {
            "No plans found.\n\nCreate a plan with: /plan <task description>".to_string()
        } else {
            let mut lines = vec!["Saved Plans:".to_string(), "".to_string()];
            for plan in plans {
                let created = chrono::DateTime::from_timestamp(plan.created_at, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_else(|| "Unknown".to_string());
                let title_preview = if plan.title.len() > 50 {
                    format!("{}...", &plan.title[..47])
                } else {
                    plan.title.clone()
                };
                lines.push(format!(
                    "  {} - {} [{}]",
                    &plan.plan_id[..8],
                    title_preview,
                    plan.status
                ));
                lines.push(format!(
                    "    Created: {} | Iterations: {}",
                    created, plan.iterations_used
                ));
            }
            lines.push("".to_string());
            lines.push("View a plan: /plan:show <plan_id>".to_string());
            lines.join("\n")
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle show plan command
    pub(super) async fn handle_show_plan(&mut self, plan_id: String) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

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
        let plan_store = PlanStore::new(client, embeddings);

        // Try to find plan by full ID or partial ID
        let plan = if plan_id.len() < 36 {
            // Search for partial match
            let all_plans = plan_store.list_recent(100).await?;
            all_plans
                .into_iter()
                .find(|p| p.plan_id.starts_with(&plan_id))
        } else {
            plan_store.get(&plan_id).await?
        };

        let content = if let Some(plan) = plan {
            let created = chrono::DateTime::from_timestamp(plan.created_at, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            format!(
                "Plan: {}\n\
                 Status: {}\n\
                 Created: {}\n\
                 Iterations: {}\n\
                 Task: {}\n\n\
                 ---\n\n\
                 {}",
                plan.plan_id,
                plan.status,
                created,
                plan.iterations_used,
                plan.task_description,
                plan.plan_content
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
        Ok(())
    }

    /// Handle delete plan command
    pub(super) async fn handle_delete_plan(&mut self, plan_id: String) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

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
        let plan_store = PlanStore::new(client, embeddings);

        // Try to find and delete the plan
        let full_plan_id = if plan_id.len() < 36 {
            // Search for partial match
            let all_plans = plan_store.list_recent(100).await?;
            all_plans
                .into_iter()
                .find(|p| p.plan_id.starts_with(&plan_id))
                .map(|p| p.plan_id)
        } else {
            Some(plan_id.clone())
        };

        let content = if let Some(id) = full_plan_id {
            // If deleting the active plan, deactivate it first
            if let Some(ref active) = self.active_plan
                && active.plan_id == id
            {
                self.active_plan = None;
                self.completed_plan_steps.clear();
                self.plan_progress = None;
            }
            match plan_store.delete(&id).await {
                Ok(()) => format!("Plan deleted: {}", &id[..8]),
                Err(e) => format!("Failed to delete plan: {}", e),
            }
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
