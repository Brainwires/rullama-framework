//! Plan Executor Agent - Executes plans by orchestrating task execution
//!
//! Runs through a plan's tasks, respecting dependencies and approval modes.
//! Integrates with completion detection to auto-progress tasks.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use brainwires_core::{PlanMetadata, Task};

use brainwires_agent::task_manager::TaskManager;

/// Approval mode for plan execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionApprovalMode {
    /// Suggest mode - ask user before each task (safest)
    Suggest,
    /// Auto-edit mode - auto-approve file edits, ask for shell commands
    AutoEdit,
    /// Full-auto mode - auto-approve everything (default for plan execution)
    #[default]
    FullAuto,
}

impl std::fmt::Display for ExecutionApprovalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Suggest => write!(f, "suggest"),
            Self::AutoEdit => write!(f, "auto-edit"),
            Self::FullAuto => write!(f, "full-auto"),
        }
    }
}

impl std::str::FromStr for ExecutionApprovalMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "suggest" => Ok(Self::Suggest),
            "auto-edit" | "autoedit" => Ok(Self::AutoEdit),
            "full-auto" | "fullauto" | "auto" => Ok(Self::FullAuto),
            _ => Err(format!("Unknown approval mode: {}", s)),
        }
    }
}

/// Status of plan execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanExecutionStatus {
    /// Not started
    Idle,
    /// Currently executing
    Running,
    /// Waiting for user approval
    WaitingForApproval(String),
    /// Paused by user
    Paused,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
}

impl std::fmt::Display for PlanExecutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Running => write!(f, "Running"),
            Self::WaitingForApproval(task) => write!(f, "Waiting for approval: {}", task),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(err) => write!(f, "Failed: {}", err),
        }
    }
}

/// Configuration for plan execution
#[derive(Debug, Clone)]
pub struct PlanExecutionConfig {
    /// Approval mode
    pub approval_mode: ExecutionApprovalMode,
    /// Maximum iterations per task
    pub max_iterations_per_task: u32,
    /// Whether to auto-start next task after completion
    pub auto_advance: bool,
    /// Stop on first error
    pub stop_on_error: bool,
}

impl Default for PlanExecutionConfig {
    fn default() -> Self {
        Self {
            approval_mode: ExecutionApprovalMode::FullAuto,
            max_iterations_per_task: 15,
            auto_advance: true,
            stop_on_error: true,
        }
    }
}

/// Plan Executor Agent - coordinates execution of a plan's tasks
pub struct PlanExecutorAgent {
    /// The plan being executed
    plan: Arc<RwLock<PlanMetadata>>,
    /// Task manager
    task_manager: Arc<RwLock<TaskManager>>,
    /// Execution configuration
    config: PlanExecutionConfig,
    /// Current execution status
    status: Arc<RwLock<PlanExecutionStatus>>,
    /// Current task being executed (if any)
    current_task_id: Arc<RwLock<Option<String>>>,
}

impl PlanExecutorAgent {
    /// Create a new plan executor
    pub fn new(
        plan: PlanMetadata,
        task_manager: Arc<RwLock<TaskManager>>,
        config: PlanExecutionConfig,
    ) -> Self {
        Self {
            plan: Arc::new(RwLock::new(plan)),
            task_manager,
            config,
            status: Arc::new(RwLock::new(PlanExecutionStatus::Idle)),
            current_task_id: Arc::new(RwLock::new(None)),
        }
    }

    /// Get the plan
    #[tracing::instrument(name = "agent.plan.get", skip(self))]
    pub async fn plan(&self) -> PlanMetadata {
        self.plan.read().await.clone()
    }

    /// Get the execution status
    pub async fn status(&self) -> PlanExecutionStatus {
        self.status.read().await.clone()
    }

    /// Get the current task ID
    pub async fn current_task_id(&self) -> Option<String> {
        self.current_task_id.read().await.clone()
    }

    /// Get the approval mode
    pub fn approval_mode(&self) -> ExecutionApprovalMode {
        self.config.approval_mode
    }

    /// Set the approval mode
    pub fn set_approval_mode(&mut self, mode: ExecutionApprovalMode) {
        self.config.approval_mode = mode;
    }

    /// Check if a task needs approval based on current mode
    pub fn needs_approval(&self, _task: &Task) -> bool {
        match self.config.approval_mode {
            ExecutionApprovalMode::Suggest => true,   // Always ask
            ExecutionApprovalMode::AutoEdit => false, // Auto-approve (shell commands need separate handling)
            ExecutionApprovalMode::FullAuto => false, // Never ask
        }
    }

    /// Get the next task to execute
    #[tracing::instrument(name = "agent.plan.next_task", skip(self))]
    pub async fn get_next_task(&self) -> Option<Task> {
        let task_mgr = self.task_manager.read().await;
        let ready_tasks = task_mgr.get_ready_tasks().await;
        ready_tasks.into_iter().next()
    }

    /// Start executing a specific task
    #[tracing::instrument(name = "agent.plan.start_task", skip(self))]
    pub async fn start_task(&self, task_id: &str) -> Result<()> {
        let task_mgr = self.task_manager.write().await;

        // Check if task can start
        match task_mgr.can_start(task_id).await {
            Ok(true) => {}
            Ok(false) => {
                anyhow::bail!(
                    "Task '{}' cannot be started (may already be completed)",
                    task_id
                );
            }
            Err(blocking_tasks) => {
                anyhow::bail!(
                    "Task '{}' is blocked by incomplete dependencies: {}",
                    task_id,
                    blocking_tasks.join(", ")
                );
            }
        }

        // Start the task
        task_mgr.start_task(task_id).await?;

        // Update current task
        *self.current_task_id.write().await = Some(task_id.to_string());

        // Update status
        *self.status.write().await = PlanExecutionStatus::Running;

        Ok(())
    }

    /// Complete the current task
    #[tracing::instrument(name = "agent.plan.complete_task", skip(self, summary))]
    pub async fn complete_current_task(&self, summary: String) -> Result<Option<Task>> {
        let task_id = {
            let current = self.current_task_id.read().await;
            current.clone()
        };

        if let Some(task_id) = task_id {
            let task_mgr = self.task_manager.write().await;
            task_mgr.complete_task(&task_id, summary).await?;

            // Clear current task
            *self.current_task_id.write().await = None;

            // Check if plan is complete
            let stats = task_mgr.get_stats().await;
            if stats.completed == stats.total {
                *self.status.write().await = PlanExecutionStatus::Completed;
            }

            // Get and return the next task if auto-advance is enabled
            if self.config.auto_advance {
                let ready_tasks = task_mgr.get_ready_tasks().await;
                return Ok(ready_tasks.into_iter().next());
            }
        }

        Ok(None)
    }

    /// Skip the current task
    pub async fn skip_current_task(&self, reason: Option<String>) -> Result<Option<Task>> {
        let task_id = {
            let current = self.current_task_id.read().await;
            current.clone()
        };

        if let Some(task_id) = task_id {
            let task_mgr = self.task_manager.write().await;
            task_mgr.skip_task(&task_id, reason).await?;

            // Clear current task
            *self.current_task_id.write().await = None;

            // Get next task if auto-advance
            if self.config.auto_advance {
                let ready_tasks = task_mgr.get_ready_tasks().await;
                return Ok(ready_tasks.into_iter().next());
            }
        }

        Ok(None)
    }

    /// Fail the current task
    pub async fn fail_current_task(&self, error: String) -> Result<()> {
        let task_id = {
            let current = self.current_task_id.read().await;
            current.clone()
        };

        if let Some(task_id) = task_id {
            let task_mgr = self.task_manager.write().await;
            task_mgr.fail_task(&task_id, error.clone()).await?;

            // Clear current task
            *self.current_task_id.write().await = None;

            if self.config.stop_on_error {
                *self.status.write().await = PlanExecutionStatus::Failed(error);
            }
        }

        Ok(())
    }

    /// Pause execution
    pub async fn pause(&self) {
        *self.status.write().await = PlanExecutionStatus::Paused;
    }

    /// Resume execution
    pub async fn resume(&self) -> Option<Task> {
        *self.status.write().await = PlanExecutionStatus::Running;

        // Return current task or get next ready task
        let current = self.current_task_id.read().await.clone();
        if current.is_some() {
            let task_mgr = self.task_manager.read().await;
            if let Some(id) = current {
                return task_mgr.get_task(&id).await;
            }
        }

        self.get_next_task().await
    }

    /// Request approval for a task (in Suggest mode)
    pub async fn request_approval(&self, task: &Task) {
        *self.status.write().await =
            PlanExecutionStatus::WaitingForApproval(task.description.clone());
    }

    /// Approve and start a task
    pub async fn approve_and_start(&self, task_id: &str) -> Result<()> {
        self.start_task(task_id).await
    }

    /// Get execution progress
    pub async fn get_progress(&self) -> ExecutionProgress {
        let task_mgr = self.task_manager.read().await;
        let stats = task_mgr.get_stats().await;
        let time_stats = task_mgr.get_time_stats().await;

        ExecutionProgress {
            total_tasks: stats.total,
            completed_tasks: stats.completed,
            in_progress_tasks: stats.in_progress,
            pending_tasks: stats.pending,
            blocked_tasks: stats.blocked,
            skipped_tasks: stats.skipped,
            failed_tasks: stats.failed,
            total_duration_secs: time_stats.total_duration_secs,
            average_task_duration_secs: time_stats.average_duration_secs,
            estimated_remaining_secs: task_mgr.estimate_remaining_time().await,
        }
    }

    /// Format progress as a string
    pub async fn format_progress(&self) -> String {
        let progress = self.get_progress().await;
        let status = self.status().await;

        let mut output = format!(
            "Plan Execution Status: {}\n\
             Progress: {}/{} tasks completed\n",
            status, progress.completed_tasks, progress.total_tasks
        );

        if progress.in_progress_tasks > 0 {
            output.push_str(&format!("  In Progress: {}\n", progress.in_progress_tasks));
        }
        if progress.blocked_tasks > 0 {
            output.push_str(&format!("  Blocked: {}\n", progress.blocked_tasks));
        }
        if progress.skipped_tasks > 0 {
            output.push_str(&format!("  Skipped: {}\n", progress.skipped_tasks));
        }
        if progress.failed_tasks > 0 {
            output.push_str(&format!("  Failed: {}\n", progress.failed_tasks));
        }

        if progress.total_duration_secs > 0 {
            output.push_str(&format!(
                "Time: {} elapsed",
                format_duration(progress.total_duration_secs)
            ));

            if let Some(remaining) = progress.estimated_remaining_secs {
                output.push_str(&format!(", ~{} remaining", format_duration(remaining)));
            }
            output.push('\n');
        }

        output
    }
}

/// Execution progress information
#[derive(Debug, Clone)]
pub struct ExecutionProgress {
    /// Total number of tasks in the plan.
    pub total_tasks: usize,
    /// Number of completed tasks.
    pub completed_tasks: usize,
    /// Number of tasks currently in progress.
    pub in_progress_tasks: usize,
    /// Number of pending tasks.
    pub pending_tasks: usize,
    /// Number of blocked tasks.
    pub blocked_tasks: usize,
    /// Number of skipped tasks.
    pub skipped_tasks: usize,
    /// Number of failed tasks.
    pub failed_tasks: usize,
    /// Total elapsed duration in seconds.
    pub total_duration_secs: i64,
    /// Average task duration in seconds.
    pub average_task_duration_secs: Option<i64>,
    /// Estimated remaining time in seconds.
    pub estimated_remaining_secs: Option<i64>,
}

/// Format duration in human readable form
fn format_duration(secs: i64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{PlanStatus, TaskPriority, TaskStatus};

    fn create_test_plan() -> PlanMetadata {
        PlanMetadata {
            plan_id: "test-plan-1".to_string(),
            conversation_id: "conv-1".to_string(),
            title: "Test Plan".to_string(),
            task_description: "Test the plan executor".to_string(),
            plan_content: "1. First task\n2. Second task".to_string(),
            model_id: None,
            status: PlanStatus::Active,
            executed: false,
            iterations_used: 0,
            created_at: 0,
            updated_at: 0,
            file_path: None,
            embedding: None,
            // Branching fields
            parent_plan_id: None,
            child_plan_ids: Vec::new(),
            branch_name: None,
            merged: false,
            depth: 0,
        }
    }

    async fn create_test_task_manager() -> Arc<RwLock<TaskManager>> {
        let task_mgr = TaskManager::new();

        // Add test tasks
        task_mgr
            .create_task("First task".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();
        task_mgr
            .create_task("Second task".to_string(), None, TaskPriority::Normal)
            .await
            .unwrap();

        Arc::new(RwLock::new(task_mgr))
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let executor = PlanExecutorAgent::new(plan, task_mgr, config);

        assert_eq!(executor.status().await, PlanExecutionStatus::Idle);
        assert!(executor.current_task_id().await.is_none());
    }

    #[tokio::test]
    async fn test_approval_modes() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let mut executor = PlanExecutorAgent::new(plan, task_mgr, config);

        // Default is FullAuto
        assert_eq!(executor.approval_mode(), ExecutionApprovalMode::FullAuto);

        // Change mode
        executor.set_approval_mode(ExecutionApprovalMode::Suggest);
        assert_eq!(executor.approval_mode(), ExecutionApprovalMode::Suggest);
    }

    #[tokio::test]
    async fn test_get_next_task() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let executor = PlanExecutorAgent::new(plan, task_mgr, config);

        let next = executor.get_next_task().await;
        assert!(next.is_some());
        // Don't check specific task - order is non-deterministic
        let desc = next.unwrap().description;
        assert!(desc == "First task" || desc == "Second task");
    }

    #[tokio::test]
    async fn test_start_task() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        // Get the first task ID
        let task_id = {
            let mgr = task_mgr.read().await;
            let tasks = mgr.get_all_tasks().await;
            tasks[0].id.clone()
        };

        let executor = PlanExecutorAgent::new(plan, task_mgr.clone(), config);

        // Start the task
        executor.start_task(&task_id).await.unwrap();

        assert_eq!(executor.status().await, PlanExecutionStatus::Running);
        assert_eq!(executor.current_task_id().await, Some(task_id.clone()));

        // Verify task status in manager
        let mgr = task_mgr.read().await;
        let task = mgr.get_task(&task_id).await.unwrap();
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[tokio::test]
    async fn test_complete_task() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let task_id = {
            let mgr = task_mgr.read().await;
            let tasks = mgr.get_all_tasks().await;
            tasks[0].id.clone()
        };

        let executor = PlanExecutorAgent::new(plan, task_mgr.clone(), config);

        // Start and complete task
        executor.start_task(&task_id).await.unwrap();
        let next = executor
            .complete_current_task("Done".to_string())
            .await
            .unwrap();

        // Should get next task due to auto-advance
        assert!(next.is_some());
        // Don't check specific task - the other task should be returned
        let next_desc = next.unwrap().description;
        let started_desc = {
            let mgr = task_mgr.read().await;
            mgr.get_task(&task_id).await.unwrap().description.clone()
        };
        // Next task should be different from the one we completed
        assert_ne!(next_desc, started_desc);

        // Current task should be cleared
        assert!(executor.current_task_id().await.is_none());
    }

    #[tokio::test]
    async fn test_pause_resume() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let executor = PlanExecutorAgent::new(plan, task_mgr, config);

        executor.pause().await;
        assert_eq!(executor.status().await, PlanExecutionStatus::Paused);

        let next = executor.resume().await;
        assert_eq!(executor.status().await, PlanExecutionStatus::Running);
        assert!(next.is_some());
    }

    #[tokio::test]
    async fn test_progress() {
        let plan = create_test_plan();
        let task_mgr = create_test_task_manager().await;
        let config = PlanExecutionConfig::default();

        let task_id = {
            let mgr = task_mgr.read().await;
            let tasks = mgr.get_all_tasks().await;
            tasks[0].id.clone()
        };

        let executor = PlanExecutorAgent::new(plan, task_mgr, config);

        // Start task
        executor.start_task(&task_id).await.unwrap();

        let progress = executor.get_progress().await;
        assert_eq!(progress.total_tasks, 2);
        assert_eq!(progress.in_progress_tasks, 1);
        assert_eq!(progress.pending_tasks, 1);
        assert_eq!(progress.completed_tasks, 0);
    }

    #[test]
    fn test_approval_mode_parsing() {
        assert_eq!(
            "suggest".parse::<ExecutionApprovalMode>().unwrap(),
            ExecutionApprovalMode::Suggest
        );
        assert_eq!(
            "auto-edit".parse::<ExecutionApprovalMode>().unwrap(),
            ExecutionApprovalMode::AutoEdit
        );
        assert_eq!(
            "full-auto".parse::<ExecutionApprovalMode>().unwrap(),
            ExecutionApprovalMode::FullAuto
        );
        assert_eq!(
            "auto".parse::<ExecutionApprovalMode>().unwrap(),
            ExecutionApprovalMode::FullAuto
        );
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m");
    }
}
