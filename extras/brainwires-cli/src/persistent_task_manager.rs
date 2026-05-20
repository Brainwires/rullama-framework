//! Persistent Task Manager - TaskManager with automatic persistence
//!
//! Wraps TaskManager to automatically save/load tasks from the storage backend,
//! enabling task lists to survive app restarts and be linked to plans.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use brainwires_agent::task_manager::TaskManager;
use brainwires_core::{Task, TaskPriority, TaskStatus};

use brainwires_storage::LanceDatabase;

use brainwires_stores::TaskStore;

/// A TaskManager that automatically persists to the storage backend
pub struct PersistentTaskManager {
    /// In-memory task manager
    manager: TaskManager,
    /// Task store for persistence
    store: TaskStore,
    /// Current conversation ID (for storage association)
    conversation_id: String,
    /// Current plan ID (if tasks are linked to a plan)
    plan_id: Option<String>,
}

impl PersistentTaskManager {
    /// Create a new persistent task manager
    pub async fn new(db: Arc<LanceDatabase>, conversation_id: String) -> Result<Self> {
        let store = TaskStore::new(db);
        let manager = TaskManager::new();

        // Load existing tasks for this conversation
        let tasks = store
            .get_by_conversation(&conversation_id)
            .await
            .unwrap_or_default();
        if !tasks.is_empty() {
            manager.load_tasks(tasks).await;
        }

        Ok(Self {
            manager,
            store,
            conversation_id,
            plan_id: None,
        })
    }

    /// Create for a specific plan (loads tasks for that plan)
    pub async fn new_for_plan(
        db: Arc<LanceDatabase>,
        conversation_id: String,
        plan_id: String,
    ) -> Result<Self> {
        let store = TaskStore::new(db);
        let manager = TaskManager::new();

        // Load existing tasks for this plan
        let tasks = store.get_by_plan(&plan_id).await.unwrap_or_default();
        if !tasks.is_empty() {
            manager.load_tasks(tasks).await;
        }

        Ok(Self {
            manager,
            store,
            conversation_id,
            plan_id: Some(plan_id),
        })
    }

    /// Set the active plan ID (new tasks will be linked to this plan)
    pub fn set_plan_id(&mut self, plan_id: Option<String>) {
        self.plan_id = plan_id.clone();
    }

    /// Get the current plan ID
    pub fn plan_id(&self) -> Option<&String> {
        self.plan_id.as_ref()
    }

    /// Create a new task and persist it
    pub async fn create_task(
        &self,
        description: String,
        parent_id: Option<String>,
        priority: TaskPriority,
    ) -> Result<String> {
        let task_id = self
            .manager
            .create_task(description, parent_id, priority)
            .await?;

        // Set plan_id on the task and persist
        if let Some(mut task) = self.manager.get_task(&task_id).await {
            task.plan_id = self.plan_id.clone();
            self.store.save(&task, &self.conversation_id).await?;
        }

        Ok(task_id)
    }

    /// Add a subtask and persist it
    pub async fn add_subtask(&self, parent_id: String, description: String) -> Result<String> {
        let task_id = self.manager.add_subtask(parent_id, description).await?;

        // Set plan_id on the task and persist
        if let Some(mut task) = self.manager.get_task(&task_id).await {
            task.plan_id = self.plan_id.clone();
            self.store.save(&task, &self.conversation_id).await?;
        }

        Ok(task_id)
    }

    /// Start a task and persist the change
    pub async fn start_task(&self, task_id: &str) -> Result<()> {
        self.manager.start_task(task_id).await?;

        if let Some(task) = self.manager.get_task(task_id).await {
            self.store.save(&task, &self.conversation_id).await?;
        }

        Ok(())
    }

    /// Complete a task and persist the change
    pub async fn complete_task(&self, task_id: &str, summary: String) -> Result<()> {
        self.manager.complete_task(task_id, summary).await?;

        if let Some(task) = self.manager.get_task(task_id).await {
            self.store.save(&task, &self.conversation_id).await?;
        }

        // Also persist any parent that was auto-completed
        let all_tasks = self.manager.get_all_tasks().await;
        for task in all_tasks {
            if task.status == TaskStatus::Completed {
                self.store.save(&task, &self.conversation_id).await?;
            }
        }

        Ok(())
    }

    /// Fail a task and persist the change
    pub async fn fail_task(&self, task_id: &str, error: String) -> Result<()> {
        self.manager.fail_task(task_id, error).await?;

        if let Some(task) = self.manager.get_task(task_id).await {
            self.store.save(&task, &self.conversation_id).await?;
        }

        Ok(())
    }

    /// Add a dependency and persist
    pub async fn add_dependency(&self, task_id: &str, depends_on: &str) -> Result<()> {
        self.manager.add_dependency(task_id, depends_on).await?;

        if let Some(task) = self.manager.get_task(task_id).await {
            self.store.save(&task, &self.conversation_id).await?;
        }

        Ok(())
    }

    /// Clear all tasks (in memory and storage)
    pub async fn clear(&self) -> Result<()> {
        self.manager.clear().await;

        if let Some(ref plan_id) = self.plan_id {
            self.store.delete_by_plan(plan_id).await?;
        } else {
            self.store
                .delete_by_conversation(&self.conversation_id)
                .await?;
        }

        Ok(())
    }

    /// Persist all current tasks to storage
    pub async fn persist_all(&self) -> Result<()> {
        let tasks = self.manager.get_all_tasks().await;
        for task in tasks {
            self.store.save(&task, &self.conversation_id).await?;
        }
        Ok(())
    }

    /// Load tasks from storage (refreshes in-memory state)
    pub async fn reload(&self) -> Result<()> {
        let tasks = if let Some(ref plan_id) = self.plan_id {
            self.store.get_by_plan(plan_id).await?
        } else {
            self.store
                .get_by_conversation(&self.conversation_id)
                .await?
        };

        self.manager.load_tasks(tasks).await;
        Ok(())
    }

    // Delegate read-only methods to inner manager

    /// Get a task by ID
    pub async fn get_task(&self, task_id: &str) -> Option<Task> {
        self.manager.get_task(task_id).await
    }

    /// Get all tasks ready to execute
    pub async fn get_ready_tasks(&self) -> Vec<Task> {
        self.manager.get_ready_tasks().await
    }

    /// Get all root tasks
    pub async fn get_root_tasks(&self) -> Vec<Task> {
        self.manager.get_root_tasks().await
    }

    /// Get task tree
    pub async fn get_task_tree(&self, root_id: Option<&str>) -> Vec<Task> {
        self.manager.get_task_tree(root_id).await
    }

    /// Get all tasks
    pub async fn get_all_tasks(&self) -> Vec<Task> {
        self.manager.get_all_tasks().await
    }

    /// Get tasks by status
    pub async fn get_tasks_by_status(&self, status: TaskStatus) -> Vec<Task> {
        self.manager.get_tasks_by_status(status).await
    }

    /// Get task count
    pub async fn count(&self) -> usize {
        self.manager.count().await
    }

    /// Get statistics
    pub async fn get_stats(&self) -> brainwires_agent::task_manager::TaskStats {
        self.manager.get_stats().await
    }

    /// Get progress percentage
    pub async fn get_progress(&self, task_id: &str) -> f64 {
        self.manager.get_progress(task_id).await
    }

    /// Get overall progress
    pub async fn get_overall_progress(&self) -> f64 {
        self.manager.get_overall_progress().await
    }

    /// Format task tree as text
    pub async fn format_tree(&self) -> String {
        self.manager.format_tree().await
    }
}

/// Thread-safe wrapper for PersistentTaskManager
pub type SharedPersistentTaskManager = Arc<RwLock<PersistentTaskManager>>;

#[cfg(test)]
mod tests {
    // Note: Integration tests require a running LanceDB instance
    // Unit tests can be added for the wrapper logic
}
