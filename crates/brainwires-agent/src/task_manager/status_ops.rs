//! Status Operations
//!
//! Task status update operations: start, complete, fail, skip, block.

use anyhow::{Context, Result};

use super::TaskManager;
use brainwires_core::TaskStatus;

impl TaskManager {
    /// Update task status
    #[tracing::instrument(name = "agent.task.update_status", skip(self, summary), fields(status = ?status))]
    pub async fn update_status(
        &self,
        task_id: &str,
        status: TaskStatus,
        summary: Option<String>,
    ) -> Result<()> {
        let parent_id = {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .context(format!("Task '{}' not found", task_id))?;

            task.status = status.clone();
            if let Some(s) = summary {
                task.summary = Some(s);
            }
            task.updated_at = chrono::Utc::now().timestamp();

            task.parent_id.clone()
        };

        // If completing a task, unblock dependents and check parent
        if status == TaskStatus::Completed {
            self.unblock_dependents(task_id).await?;
            if let Some(ref parent_id) = parent_id {
                self.check_parent_completion(parent_id).await?;
            }
        }

        Ok(())
    }

    /// Mark a task as started
    pub async fn start_task(&self, task_id: &str) -> Result<()> {
        self.update_status(task_id, TaskStatus::InProgress, None)
            .await
    }

    /// Mark a task as completed
    pub async fn complete_task(&self, task_id: &str, summary: String) -> Result<()> {
        self.update_status(task_id, TaskStatus::Completed, Some(summary))
            .await
    }

    /// Mark a task as failed
    pub async fn fail_task(&self, task_id: &str, error: String) -> Result<()> {
        self.update_status(task_id, TaskStatus::Failed, Some(error))
            .await
    }

    /// Mark a task as skipped
    pub async fn skip_task(&self, task_id: &str, reason: Option<String>) -> Result<()> {
        let parent_id = {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .context(format!("Task '{}' not found", task_id))?;

            let now = chrono::Utc::now().timestamp();
            task.status = TaskStatus::Skipped;
            task.completed_at = Some(now);
            task.updated_at = now;

            if let Some(r) = reason {
                task.summary = Some(r);
            }

            task.parent_id.clone()
        };

        // Check if parent can be updated (lock released above)
        if let Some(ref pid) = parent_id {
            self.check_parent_completion(pid).await?;
        }

        // Unblock tasks that depend on this one
        self.unblock_dependents(task_id).await?;

        Ok(())
    }

    /// Mark a task as blocked with optional reason
    pub async fn block_task(&self, task_id: &str, reason: Option<String>) -> Result<()> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .context(format!("Task '{}' not found", task_id))?;

        task.status = TaskStatus::Blocked;
        task.updated_at = chrono::Utc::now().timestamp();

        if let Some(r) = reason {
            task.summary = Some(r);
        }

        Ok(())
    }

    /// Check if parent task should be auto-completed when all children are done
    pub(crate) async fn check_parent_completion(&self, parent_id: &str) -> Result<()> {
        let tasks = self.tasks.read().await;

        if let Some(parent) = tasks.get(parent_id) {
            // Check if all children are complete
            let all_complete = parent.children.iter().all(|child_id| {
                tasks
                    .get(child_id)
                    .map(|t| t.status == TaskStatus::Completed)
                    .unwrap_or(false)
            });

            if all_complete && !parent.children.is_empty() {
                let grandparent_id = parent.parent_id.clone();
                drop(tasks);

                // Auto-complete the parent task
                let mut tasks = self.tasks.write().await;
                if let Some(parent) = tasks.get_mut(parent_id)
                    && (parent.status == TaskStatus::InProgress
                        || parent.status == TaskStatus::Pending)
                {
                    parent.status = TaskStatus::Completed;
                    parent.summary =
                        Some(format!("All {} subtasks completed", parent.children.len()));
                    parent.updated_at = chrono::Utc::now().timestamp();
                }
                drop(tasks);

                // Recursively check grandparent
                if let Some(gp_id) = grandparent_id {
                    Box::pin(self.check_parent_completion(&gp_id)).await?;
                }
            }
        }

        Ok(())
    }
}
