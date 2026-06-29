use serde::{Deserialize, Serialize};

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task is waiting to be started.
    Pending,
    /// Task is currently being executed.
    InProgress,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task is blocked by dependencies.
    Blocked,
    /// Task was skipped.
    Skipped,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum TaskPriority {
    /// Low priority.
    Low = 0,
    /// Normal (default) priority.
    #[default]
    Normal = 1,
    /// High priority.
    High = 2,
    /// Urgent priority.
    Urgent = 3,
}

/// A task being executed by an agent (supports tree structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task ID
    pub id: String,
    /// Task description
    pub description: String,
    /// Current status
    pub status: TaskStatus,
    /// Associated plan ID (links task to a plan)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    /// Parent task ID (for tree structure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Child task IDs
    #[serde(default)]
    pub children: Vec<String>,
    /// Task IDs this task depends on (must complete before this can start)
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Task priority
    #[serde(default)]
    pub priority: TaskPriority,
    /// Assigned agent (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigned_to: Option<String>,
    /// Number of iterations executed
    #[serde(default)]
    pub iterations: u32,
    /// Result summary (when completed or failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Creation timestamp
    #[serde(default = "default_timestamp")]
    pub created_at: i64,
    /// Last update timestamp
    #[serde(default = "default_timestamp")]
    pub updated_at: i64,
    /// When the task was started (for time tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<i64>,
    /// When the task was completed (for time tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
}

fn default_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Task {
    /// Create a new pending task
    pub fn new<S: Into<String>>(id: S, description: S) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: id.into(),
            description: description.into(),
            status: TaskStatus::Pending,
            plan_id: None,
            parent_id: None,
            children: Vec::new(),
            depends_on: Vec::new(),
            priority: TaskPriority::Normal,
            assigned_to: None,
            iterations: 0,
            summary: None,
            created_at: now,
            updated_at: now,
            started_at: None,
            completed_at: None,
        }
    }

    /// Create a new task associated with a plan
    pub fn new_for_plan<S: Into<String>>(id: S, description: S, plan_id: S) -> Self {
        let mut task = Self::new(id, description);
        task.plan_id = Some(plan_id.into());
        task
    }

    /// Create a new task with a parent (subtask)
    pub fn new_subtask<S: Into<String>>(id: S, description: S, parent_id: S) -> Self {
        let mut task = Self::new(id, description);
        task.parent_id = Some(parent_id.into());
        task
    }

    /// Mark task as in progress
    pub fn start(&mut self) {
        let now = chrono::Utc::now().timestamp();
        self.status = TaskStatus::InProgress;
        self.started_at = Some(now);
        self.updated_at = now;
    }

    /// Mark task as completed
    pub fn complete<S: Into<String>>(&mut self, summary: S) {
        let now = chrono::Utc::now().timestamp();
        self.status = TaskStatus::Completed;
        self.summary = Some(summary.into());
        self.completed_at = Some(now);
        self.updated_at = now;
    }

    /// Get task duration in seconds (if started and completed)
    pub fn duration_secs(&self) -> Option<i64> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Get elapsed time since task started (in seconds)
    pub fn elapsed_secs(&self) -> Option<i64> {
        self.started_at
            .map(|start| chrono::Utc::now().timestamp() - start)
    }

    /// Mark task as failed
    pub fn fail<S: Into<String>>(&mut self, error: S) {
        self.status = TaskStatus::Failed;
        self.summary = Some(error.into());
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Mark task as blocked (waiting on dependencies)
    pub fn block(&mut self) {
        self.status = TaskStatus::Blocked;
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Mark task as skipped
    pub fn skip<S: Into<String>>(&mut self, reason: Option<S>) {
        let now = chrono::Utc::now().timestamp();
        self.status = TaskStatus::Skipped;
        if let Some(r) = reason {
            self.summary = Some(r.into());
        }
        self.completed_at = Some(now);
        self.updated_at = now;
    }

    /// Increment iterations
    pub fn increment_iteration(&mut self) {
        self.iterations += 1;
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Add a child task ID
    pub fn add_child(&mut self, child_id: String) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
            self.updated_at = chrono::Utc::now().timestamp();
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, task_id: String) {
        if !self.depends_on.contains(&task_id) {
            self.depends_on.push(task_id);
            self.updated_at = chrono::Utc::now().timestamp();
        }
    }

    /// Check if task has any incomplete dependencies
    pub fn has_dependencies(&self) -> bool {
        !self.depends_on.is_empty()
    }

    /// Check if task has children
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Check if task is a root task (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Set priority
    pub fn set_priority(&mut self, priority: TaskPriority) {
        self.priority = priority;
        self.updated_at = chrono::Utc::now().timestamp();
    }
}

/// Agent response after processing
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// The response message
    pub message: String,
    /// Whether the task is complete
    pub is_complete: bool,
    /// Tasks created or updated
    pub tasks: Vec<Task>,
    /// Number of iterations executed
    pub iterations: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_lifecycle() {
        let mut task = Task::new("task-1", "Test task");
        assert_eq!(task.status, TaskStatus::Pending);
        task.start();
        assert_eq!(task.status, TaskStatus::InProgress);
        task.complete("Done!");
        assert_eq!(task.status, TaskStatus::Completed);
    }

    #[test]
    fn test_task_failure() {
        let mut task = Task::new("task-2", "Failing task");
        task.start();
        task.fail("Error occurred");
        assert_eq!(task.status, TaskStatus::Failed);
    }
}
