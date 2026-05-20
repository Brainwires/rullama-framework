//! Task Manager - Manages a tree-structured task list
//!
//! Provides functionality for the AI to create, update, and manage tasks
//! in a hierarchical structure with dependencies.

mod dependency_ops;
mod query_ops;
mod status_ops;
mod time_tracking;

#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use brainwires_core::{Task, TaskPriority};

// Re-export public types
pub use time_tracking::{TaskStats, TaskTimeInfo, TimeStats, format_duration_secs};

/// Manages a tree of tasks with dependencies
#[derive(Debug, Clone)]
pub struct TaskManager {
    /// All tasks indexed by ID
    pub(crate) tasks: Arc<RwLock<HashMap<String, Task>>>,
}

impl TaskManager {
    /// Create a new task manager
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new task
    #[tracing::instrument(name = "agent.task.create", skip(self, description))]
    pub async fn create_task(
        &self,
        description: String,
        parent_id: Option<String>,
        priority: TaskPriority,
    ) -> Result<String> {
        let task_id = uuid::Uuid::new_v4().to_string();
        let mut task = Task::new(task_id.clone(), description);
        task.priority = priority;

        let mut tasks = self.tasks.write().await;

        // If parent_id is provided, validate and update parent
        if let Some(ref pid) = parent_id {
            let parent = tasks
                .get_mut(pid)
                .context(format!("Parent task '{}' not found", pid))?;
            parent.add_child(task_id.clone());
            task.parent_id = Some(pid.clone());
        }

        tasks.insert(task_id.clone(), task);
        Ok(task_id)
    }

    /// Add a subtask to an existing task
    pub async fn add_subtask(&self, parent_id: String, description: String) -> Result<String> {
        self.create_task(description, Some(parent_id), TaskPriority::Normal)
            .await
    }

    /// Get a task by ID
    pub async fn get_task(&self, task_id: &str) -> Option<Task> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).cloned()
    }

    /// Clear all tasks
    pub async fn clear(&self) {
        let mut tasks = self.tasks.write().await;
        tasks.clear();
    }

    /// Get task count
    pub async fn count(&self) -> usize {
        let tasks = self.tasks.read().await;
        tasks.len()
    }

    /// Load tasks from storage
    pub async fn load_tasks(&self, tasks_to_load: Vec<Task>) {
        let mut tasks = self.tasks.write().await;
        tasks.clear();
        for task in tasks_to_load {
            tasks.insert(task.id.clone(), task);
        }
    }

    /// Export all tasks for persistence
    pub async fn export_tasks(&self) -> Vec<Task> {
        self.get_all_tasks().await
    }

    /// Assign a task to an agent (sets the `assigned_to` field).
    pub async fn assign_task(&self, task_id: &str, agent_id: &str) -> Result<()> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .context(format!("Task '{}' not found", task_id))?;

        task.assigned_to = Some(agent_id.to_string());
        task.updated_at = chrono::Utc::now().timestamp();
        Ok(())
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}
