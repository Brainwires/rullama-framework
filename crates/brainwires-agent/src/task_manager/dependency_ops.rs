//! Dependency Operations
//!
//! Task dependency management: add, remove, check dependencies, cycle detection.

use anyhow::{Context, Result};

use super::TaskManager;
use brainwires_core::TaskStatus;

impl TaskManager {
    /// Add a dependency between tasks
    pub async fn add_dependency(&self, task_id: &str, depends_on: &str) -> Result<()> {
        // Check for circular dependency before acquiring write lock
        if self.would_create_cycle(task_id, depends_on).await? {
            anyhow::bail!(
                "Adding dependency '{}' -> '{}' would create a circular dependency",
                task_id,
                depends_on
            );
        }

        let mut tasks = self.tasks.write().await;

        // Verify both tasks exist
        if !tasks.contains_key(depends_on) {
            anyhow::bail!("Dependency task '{}' not found", depends_on);
        }

        let task = tasks
            .get_mut(task_id)
            .context(format!("Task '{}' not found", task_id))?;

        task.add_dependency(depends_on.to_string());

        // If dependency is not complete/skipped, mark task as blocked
        let dep_status = tasks
            .get(depends_on)
            .expect("dependency existence verified above")
            .status
            .clone();
        if dep_status != TaskStatus::Completed && dep_status != TaskStatus::Skipped {
            tasks
                .get_mut(task_id)
                .expect("task existence verified above")
                .status = TaskStatus::Blocked;
        }

        Ok(())
    }

    /// Check if adding a dependency would create a circular dependency
    async fn would_create_cycle(&self, task_id: &str, depends_on: &str) -> Result<bool> {
        // If task_id == depends_on, it's a self-dependency (cycle)
        if task_id == depends_on {
            return Ok(true);
        }

        let tasks = self.tasks.read().await;

        // BFS to check if depends_on can reach task_id through its dependencies
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(depends_on.to_string());

        while let Some(current) = queue.pop_front() {
            if current == task_id {
                return Ok(true); // Found a cycle
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(task) = tasks.get(&current) {
                for dep in &task.depends_on {
                    if !visited.contains(dep) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        Ok(false)
    }

    /// Check if a task can be started (all dependencies are complete/skipped)
    /// Returns Ok(true) if task can start, or Err with list of blocking task IDs
    pub async fn can_start(&self, task_id: &str) -> std::result::Result<bool, Vec<String>> {
        let tasks = self.tasks.read().await;

        let task = match tasks.get(task_id) {
            Some(t) => t,
            None => return Ok(false), // Task doesn't exist
        };

        // If task is already completed, failed, or skipped, it can't be started
        if matches!(
            task.status,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Skipped
        ) {
            return Ok(false);
        }

        // Collect blocking dependencies
        let blocking: Vec<String> = task
            .depends_on
            .iter()
            .filter(|dep_id| {
                tasks
                    .get(*dep_id)
                    .map(|t| t.status != TaskStatus::Completed && t.status != TaskStatus::Skipped)
                    .unwrap_or(true) // Missing dependency is considered blocking
            })
            .cloned()
            .collect();

        if blocking.is_empty() {
            Ok(true)
        } else {
            Err(blocking)
        }
    }

    /// Remove a dependency between tasks
    pub async fn remove_dependency(&self, task_id: &str, depends_on: &str) -> Result<()> {
        let mut tasks = self.tasks.write().await;

        // First get the task, update it, and collect info for the check
        let (is_blocked, remaining_deps) = {
            let task = tasks
                .get_mut(task_id)
                .context(format!("Task '{}' not found", task_id))?;

            task.depends_on.retain(|d| d != depends_on);
            task.updated_at = chrono::Utc::now().timestamp();

            (task.status == TaskStatus::Blocked, task.depends_on.clone())
        };

        // Check if task should be unblocked
        if is_blocked {
            let all_deps_done = remaining_deps.iter().all(|dep_id| {
                tasks
                    .get(dep_id)
                    .map(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Skipped)
                    .unwrap_or(false)
            });

            if all_deps_done && let Some(task) = tasks.get_mut(task_id) {
                task.status = TaskStatus::Pending;
            }
        }

        Ok(())
    }

    /// Unblock tasks that depend on a completed/skipped task
    pub async fn unblock_dependents(&self, completed_task_id: &str) -> Result<()> {
        let mut tasks = self.tasks.write().await;

        // Find all tasks that depend on the completed task
        let dependent_ids: Vec<String> = tasks
            .values()
            .filter(|t| t.depends_on.contains(&completed_task_id.to_string()))
            .map(|t| t.id.clone())
            .collect();

        // Collect the dependency lists for each task first
        let mut tasks_to_check: Vec<(String, Vec<String>)> = Vec::new();
        for dep_id in &dependent_ids {
            if let Some(task) = tasks.get(dep_id)
                && task.status == TaskStatus::Blocked
            {
                tasks_to_check.push((dep_id.clone(), task.depends_on.clone()));
            }
        }

        // Now update tasks based on dependency status
        for (dep_id, deps) in tasks_to_check {
            // Check if all dependencies are now complete/skipped
            let all_deps_done = deps.iter().all(|d| {
                tasks
                    .get(d)
                    .map(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Skipped)
                    .unwrap_or(false)
            });

            if all_deps_done && let Some(task) = tasks.get_mut(&dep_id) {
                task.status = TaskStatus::Pending;
                task.updated_at = chrono::Utc::now().timestamp();
            }
        }

        Ok(())
    }
}
