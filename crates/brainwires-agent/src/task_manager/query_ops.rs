//! Query Operations
//!
//! Task query operations: get ready tasks, get all tasks, get tree, format tree, get stats, progress.

use std::collections::HashMap;

use super::TaskManager;
use super::time_tracking::{TaskStats, TaskTimeInfo, TimeStats};
use brainwires_core::{Task, TaskPriority, TaskStatus};

impl TaskManager {
    /// Get all tasks ready to execute (no incomplete dependencies)
    pub async fn get_ready_tasks(&self) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        let mut ready = Vec::new();

        for task in tasks.values() {
            if task.status == TaskStatus::Pending || task.status == TaskStatus::Blocked {
                // Check if all dependencies are complete or skipped
                let deps_complete = task.depends_on.iter().all(|dep_id| {
                    tasks
                        .get(dep_id)
                        .map(|t| {
                            t.status == TaskStatus::Completed || t.status == TaskStatus::Skipped
                        })
                        .unwrap_or(false)
                });

                if deps_complete {
                    ready.push(task.clone());
                }
            }
        }

        // Sort by priority (highest first)
        ready.sort_by(|a, b| b.priority.cmp(&a.priority));
        ready
    }

    /// Get all root tasks (tasks without parents)
    pub async fn get_root_tasks(&self) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        tasks.values().filter(|t| t.is_root()).cloned().collect()
    }

    /// Get task tree starting from a task (or all roots if None)
    pub async fn get_task_tree(&self, root_id: Option<&str>) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        let mut result = Vec::new();

        match root_id {
            Some(id) => {
                if let Some(task) = tasks.get(id) {
                    Self::collect_tree_recursive(&tasks, task, &mut result);
                }
            }
            None => {
                // Get all root tasks and their trees
                for task in tasks.values().filter(|t| t.is_root()) {
                    Self::collect_tree_recursive(&tasks, task, &mut result);
                }
            }
        }

        result
    }

    /// Recursively collect tasks in tree order
    fn collect_tree_recursive(tasks: &HashMap<String, Task>, task: &Task, result: &mut Vec<Task>) {
        result.push(task.clone());
        for child_id in &task.children {
            if let Some(child) = tasks.get(child_id) {
                Self::collect_tree_recursive(tasks, child, result);
            }
        }
    }

    /// Get all tasks
    pub async fn get_all_tasks(&self) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        tasks.values().cloned().collect()
    }

    /// Get tasks by status
    pub async fn get_tasks_by_status(&self, status: TaskStatus) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        tasks
            .values()
            .filter(|t| t.status == status)
            .cloned()
            .collect()
    }

    /// Get summary statistics
    pub async fn get_stats(&self) -> TaskStats {
        let tasks = self.tasks.read().await;
        let mut stats = TaskStats::default();

        for task in tasks.values() {
            stats.total += 1;
            match task.status {
                TaskStatus::Pending => stats.pending += 1,
                TaskStatus::InProgress => stats.in_progress += 1,
                TaskStatus::Completed => stats.completed += 1,
                TaskStatus::Failed => stats.failed += 1,
                TaskStatus::Blocked => stats.blocked += 1,
                TaskStatus::Skipped => stats.skipped += 1,
            }
        }

        stats
    }

    /// Get time tracking info for a task
    pub async fn get_task_time_info(&self, task_id: &str) -> Option<TaskTimeInfo> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).map(|task| TaskTimeInfo {
            task_id: task.id.clone(),
            description: task.description.clone(),
            status: task.status.clone(),
            started_at: task.started_at,
            completed_at: task.completed_at,
            duration_secs: task.duration_secs(),
            elapsed_secs: task.elapsed_secs(),
        })
    }

    /// Get time statistics for all tasks
    pub async fn get_time_stats(&self) -> TimeStats {
        let tasks = self.tasks.read().await;

        let mut total_duration: i64 = 0;
        let mut completed_count: usize = 0;
        let mut total_elapsed: i64 = 0;
        let mut in_progress_count: usize = 0;

        for task in tasks.values() {
            if let Some(duration) = task.duration_secs() {
                total_duration += duration;
                completed_count += 1;
            }
            if task.status == TaskStatus::InProgress
                && let Some(elapsed) = task.elapsed_secs()
            {
                total_elapsed += elapsed;
                in_progress_count += 1;
            }
        }

        TimeStats {
            total_duration_secs: total_duration,
            completed_tasks: completed_count,
            average_duration_secs: if completed_count > 0 {
                Some(total_duration / completed_count as i64)
            } else {
                None
            },
            current_elapsed_secs: total_elapsed,
            in_progress_tasks: in_progress_count,
        }
    }

    /// Calculate progress percentage (0.0 to 1.0) for a task
    /// For tasks with children, this is based on completed children
    /// For leaf tasks, returns 1.0 if completed, 0.5 if in progress, 0.0 otherwise
    pub async fn get_progress(&self, task_id: &str) -> f64 {
        let tasks = self.tasks.read().await;

        if let Some(task) = tasks.get(task_id) {
            if task.children.is_empty() {
                // Leaf task
                match task.status {
                    TaskStatus::Completed => 1.0,
                    TaskStatus::InProgress => 0.5,
                    _ => 0.0,
                }
            } else {
                // Parent task - calculate based on children
                let completed = task
                    .children
                    .iter()
                    .filter(|id| {
                        tasks
                            .get(*id)
                            .map(|t| t.status == TaskStatus::Completed)
                            .unwrap_or(false)
                    })
                    .count();
                let in_progress = task
                    .children
                    .iter()
                    .filter(|id| {
                        tasks
                            .get(*id)
                            .map(|t| t.status == TaskStatus::InProgress)
                            .unwrap_or(false)
                    })
                    .count();

                let total = task.children.len() as f64;
                if total == 0.0 {
                    return 0.0;
                }

                (completed as f64 + (in_progress as f64 * 0.5)) / total
            }
        } else {
            0.0
        }
    }

    /// Get overall progress for all tasks
    pub async fn get_overall_progress(&self) -> f64 {
        let stats = self.get_stats().await;
        if stats.total == 0 {
            return 0.0;
        }

        let completed = stats.completed as f64;
        let in_progress = stats.in_progress as f64 * 0.5;
        let total = stats.total as f64;

        (completed + in_progress) / total
    }

    /// Get average task duration in seconds (from completed tasks)
    pub async fn get_average_duration(&self) -> Option<i64> {
        let tasks = self.tasks.read().await;
        let durations: Vec<i64> = tasks.values().filter_map(|t| t.duration_secs()).collect();

        if durations.is_empty() {
            None
        } else {
            Some(durations.iter().sum::<i64>() / durations.len() as i64)
        }
    }

    /// Estimate remaining time in seconds based on average duration
    pub async fn estimate_remaining_time(&self) -> Option<i64> {
        let avg_duration = self.get_average_duration().await?;
        let stats = self.get_stats().await;
        let remaining = stats.pending + stats.blocked;

        Some(avg_duration * remaining as i64)
    }

    /// Format task tree as indented text for display
    pub async fn format_tree(&self) -> String {
        let tasks = self.tasks.read().await;
        let mut output = String::new();

        // Get root tasks
        let mut roots: Vec<_> = tasks.values().filter(|t| t.is_root()).collect();
        roots.sort_by(|a, b| b.priority.cmp(&a.priority));

        let root_count = roots.len();
        for (idx, root) in roots.iter().enumerate() {
            let is_last = idx == root_count - 1;
            Self::format_task_recursive(&tasks, root, 0, is_last, "", &mut output);
        }

        if output.is_empty() {
            output = "No tasks".to_string();
        }

        output
    }

    fn format_task_recursive(
        tasks: &HashMap<String, Task>,
        task: &Task,
        depth: usize,
        is_last: bool,
        parent_prefix: &str,
        output: &mut String,
    ) {
        let status_icon = match task.status {
            TaskStatus::Pending => "○",
            TaskStatus::InProgress => "◐",
            TaskStatus::Completed => "●",
            TaskStatus::Failed => "✗",
            TaskStatus::Blocked => "◌",
            TaskStatus::Skipped => "⊘",
        };
        let priority_icon = match task.priority {
            TaskPriority::Urgent => "🔴 ",
            TaskPriority::High => "🟠 ",
            TaskPriority::Normal => "",
            TaskPriority::Low => "🔵 ",
        };

        // Build the prefix for this task
        let current_prefix = if depth == 0 {
            String::new()
        } else if is_last {
            format!("{}└── ", parent_prefix)
        } else {
            format!("{}├── ", parent_prefix)
        };

        output.push_str(&format!(
            "{}{} {}{}\n",
            current_prefix, status_icon, priority_icon, task.description
        ));

        // Build prefix for children
        let child_prefix = if depth == 0 {
            String::new()
        } else if is_last {
            format!("{}    ", parent_prefix)
        } else {
            format!("{}│   ", parent_prefix)
        };

        // Format children
        let child_count = task.children.len();
        for (idx, child_id) in task.children.iter().enumerate() {
            if let Some(child) = tasks.get(child_id) {
                let child_is_last = idx == child_count - 1;
                Self::format_task_recursive(
                    tasks,
                    child,
                    depth + 1,
                    child_is_last,
                    &child_prefix,
                    output,
                );
            }
        }
    }
}
