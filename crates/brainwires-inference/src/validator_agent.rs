//! ValidatorAgent - Standalone read-only agent that runs external validators
//!
//! Unlike the inline validation inside `TaskAgent`, the `ValidatorAgent` can be
//! triggered independently by an orchestrator — e.g., after multiple task agents
//! finish work — without coupling validation to any single task agent.
//!
//! The agent acquires **read locks** on the working-set files, calls
//! [`run_validation`], and returns a structured [`ValidatorAgentResult`].
//!
//! This is intentionally **not** an `AgentRuntime` implementation: it is a
//! deterministic pipeline (no AI provider loop), following the same pattern as
//! [`PlanExecutorAgent`](crate::plan_executor::PlanExecutorAgent).

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::sync::RwLock;

use crate::validation_loop::{
    ValidationConfig, ValidationResult, format_validation_feedback, run_validation,
};
use brainwires_agent::communication::{AgentMessage, CommunicationHub};
use brainwires_agent::file_locks::{FileLockManager, LockGuard, LockType};

// ── Types ────────────────────────────────────────────────────────────────────

/// Current status of a `ValidatorAgent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatorAgentStatus {
    /// Not yet started.
    Idle,
    /// Acquiring read locks on working-set files.
    AcquiringLocks,
    /// Running validation checks.
    Validating,
    /// All checks passed.
    Passed,
    /// One or more checks failed. The `usize` is the issue count.
    Failed(usize),
    /// An unrecoverable error occurred.
    Error(String),
}

impl std::fmt::Display for ValidatorAgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::AcquiringLocks => write!(f, "Acquiring locks"),
            Self::Validating => write!(f, "Validating"),
            Self::Passed => write!(f, "Passed"),
            Self::Failed(n) => write!(f, "Failed ({} issues)", n),
            Self::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

/// Configuration for [`ValidatorAgent`].
#[derive(Debug, Clone)]
pub struct ValidatorAgentConfig {
    /// The underlying validation pipeline configuration.
    pub validation_config: ValidationConfig,
    /// Wall-clock timeout in seconds for the entire validation run.
    /// Default: 120.
    pub timeout_secs: u64,
}

impl Default for ValidatorAgentConfig {
    fn default() -> Self {
        Self {
            validation_config: ValidationConfig::default(),
            timeout_secs: 120,
        }
    }
}

impl ValidatorAgentConfig {
    /// Create a config wrapping a [`ValidationConfig`].
    pub fn new(validation_config: ValidationConfig) -> Self {
        Self {
            validation_config,
            ..Default::default()
        }
    }

    /// Set the wall-clock timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// Result returned by [`ValidatorAgent::validate`].
#[derive(Debug, Clone)]
pub struct ValidatorAgentResult {
    /// The validator agent's unique ID.
    pub agent_id: String,
    /// Whether all checks passed.
    pub success: bool,
    /// The raw validation result from the pipeline.
    pub validation_result: ValidationResult,
    /// Human-readable feedback string.
    pub feedback: String,
    /// Wall-clock duration of the validation run.
    pub duration: std::time::Duration,
    /// Number of files that were checked.
    pub files_checked: usize,
    /// Number of read locks that were successfully acquired.
    pub locks_acquired: usize,
}

// ── ValidatorAgent ───────────────────────────────────────────────────────────

/// A standalone, read-only agent that runs external validators and returns a
/// structured result to the orchestrator.
pub struct ValidatorAgent {
    /// Unique identifier for this validator agent.
    pub id: String,
    /// Configuration.
    pub config: ValidatorAgentConfig,
    /// Communication hub for broadcasting status messages.
    pub communication_hub: Arc<CommunicationHub>,
    /// File lock manager for acquiring read locks.
    pub file_lock_manager: Arc<FileLockManager>,
    /// Observable status.
    pub status: Arc<RwLock<ValidatorAgentStatus>>,
}

impl ValidatorAgent {
    /// Create a new `ValidatorAgent`.
    pub fn new(
        id: impl Into<String>,
        config: ValidatorAgentConfig,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
    ) -> Self {
        Self {
            id: id.into(),
            config,
            communication_hub,
            file_lock_manager,
            status: Arc::new(RwLock::new(ValidatorAgentStatus::Idle)),
        }
    }

    /// Run the full validation pipeline.
    ///
    /// 1. Register with the communication hub, broadcast `AgentSpawned`.
    /// 2. Acquire **read** locks on all `working_set_files` (best-effort).
    /// 3. Run [`run_validation`] with a wall-clock timeout.
    /// 4. Release locks, broadcast `AgentCompleted`, unregister.
    #[tracing::instrument(name = "validator_agent.validate", skip(self), fields(agent_id = %self.id))]
    pub async fn validate(&self) -> Result<ValidatorAgentResult> {
        let start = Instant::now();

        // ── 1. Register & broadcast spawn ────────────────────────────────
        self.communication_hub
            .register_agent(self.id.clone())
            .await?;

        if let Err(e) = self
            .communication_hub
            .broadcast(
                self.id.clone(),
                AgentMessage::AgentSpawned {
                    agent_id: self.id.clone(),
                    task_id: format!("validation-{}", self.id),
                },
            )
            .await
        {
            tracing::warn!(agent_id = %self.id, "Failed to broadcast validator spawn: {}", e);
        }

        // ── 2. Acquire read locks (best-effort) ─────────────────────────
        self.set_status(ValidatorAgentStatus::AcquiringLocks).await;

        let mut lock_guards: Vec<LockGuard> = Vec::new();
        for file in &self.config.validation_config.working_set_files {
            let path = std::path::PathBuf::from(&self.config.validation_config.working_directory)
                .join(file);
            match self
                .file_lock_manager
                .acquire_lock(&self.id, &path, LockType::Read)
                .await
            {
                Ok(guard) => {
                    lock_guards.push(guard);
                }
                Err(e) => {
                    tracing::warn!(
                        agent_id = %self.id,
                        file = %file,
                        "Failed to acquire read lock (best-effort, continuing): {}",
                        e
                    );
                }
            }
        }
        let locks_acquired = lock_guards.len();

        // ── 3. Run validation with timeout ──────────────────────────────
        self.set_status(ValidatorAgentStatus::Validating).await;

        let timeout = tokio::time::Duration::from_secs(self.config.timeout_secs);
        let validation_result =
            match tokio::time::timeout(timeout, run_validation(&self.config.validation_config))
                .await
            {
                Ok(Ok(result)) => result,
                Ok(Err(e)) => {
                    let err_msg = format!("Validation error: {}", e);
                    self.set_status(ValidatorAgentStatus::Error(err_msg.clone()))
                        .await;
                    self.cleanup(&lock_guards, false, &err_msg, start).await;
                    return Err(e);
                }
                Err(_elapsed) => {
                    let err_msg =
                        format!("Validation timed out after {}s", self.config.timeout_secs);
                    self.set_status(ValidatorAgentStatus::Error(err_msg.clone()))
                        .await;
                    self.cleanup(&lock_guards, false, &err_msg, start).await;
                    return Err(anyhow::anyhow!("{}", err_msg));
                }
            };

        // ── 4. Build result, drop locks, broadcast completion ───────────
        let success = validation_result.passed;
        let issue_count = validation_result.issues.len();
        let files_checked = self.config.validation_config.working_set_files.len();
        let feedback = format_validation_feedback(&validation_result);

        if success {
            self.set_status(ValidatorAgentStatus::Passed).await;
        } else {
            self.set_status(ValidatorAgentStatus::Failed(issue_count))
                .await;
        }

        let summary = if success {
            "All validation checks passed".to_string()
        } else {
            format!("Validation failed with {} issues", issue_count)
        };

        self.cleanup(&lock_guards, success, &summary, start).await;

        Ok(ValidatorAgentResult {
            agent_id: self.id.clone(),
            success,
            validation_result,
            feedback,
            duration: start.elapsed(),
            files_checked,
            locks_acquired,
        })
    }

    /// Set the observable status.
    async fn set_status(&self, status: ValidatorAgentStatus) {
        *self.status.write().await = status;
    }

    /// Drop lock guards, broadcast completion, unregister from hub.
    async fn cleanup(
        &self,
        _lock_guards: &[LockGuard],
        _success: bool,
        summary: &str,
        _start: Instant,
    ) {
        // Lock guards are borrowed — they'll be dropped when the caller's
        // `lock_guards` Vec goes out of scope. As a safety net, release all
        // locks explicitly.
        self.file_lock_manager.release_all_locks(&self.id).await;

        if let Err(e) = self
            .communication_hub
            .broadcast(
                self.id.clone(),
                AgentMessage::AgentCompleted {
                    agent_id: self.id.clone(),
                    task_id: format!("validation-{}", self.id),
                    summary: summary.to_string(),
                },
            )
            .await
        {
            tracing::warn!(agent_id = %self.id, "Failed to broadcast validator completion: {}", e);
        }

        if let Err(e) = self.communication_hub.unregister_agent(&self.id).await {
            tracing::warn!(agent_id = %self.id, "Failed to unregister validator agent: {}", e);
        }
    }
}

/// Spawn a `ValidatorAgent` on a Tokio task and return a join handle.
///
/// Mirrors [`spawn_task_agent`](crate::task_agent::spawn_task_agent).
pub fn spawn_validator_agent(
    agent: Arc<ValidatorAgent>,
) -> tokio::task::JoinHandle<Result<ValidatorAgentResult>> {
    tokio::spawn(async move { agent.validate().await })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hub_and_locks() -> (Arc<CommunicationHub>, Arc<FileLockManager>) {
        (
            Arc::new(CommunicationHub::new()),
            Arc::new(FileLockManager::new()),
        )
    }

    #[tokio::test]
    async fn test_validator_disabled() {
        let (hub, locks) = make_hub_and_locks();
        let config = ValidatorAgentConfig::new(ValidationConfig::disabled());
        let agent = ValidatorAgent::new("val-disabled", config, hub, locks);

        let result = agent.validate().await.unwrap();
        assert!(result.success);
        assert!(result.validation_result.issues.is_empty());
    }

    #[tokio::test]
    async fn test_validator_detects_missing_file() {
        let (hub, locks) = make_hub_and_locks();

        let dir = tempfile::tempdir().unwrap();
        let vc = ValidationConfig {
            working_directory: dir.path().to_string_lossy().to_string(),
            working_set_files: vec!["nonexistent.rs".to_string()],
            ..Default::default()
        };

        let config = ValidatorAgentConfig::new(vc);
        let agent = ValidatorAgent::new("val-missing", config, hub, locks);

        let result = agent.validate().await.unwrap();
        assert!(!result.success);
        assert!(
            result
                .validation_result
                .issues
                .iter()
                .any(|i| i.check == "file_existence")
        );
    }

    #[tokio::test]
    async fn test_validator_registers_and_unregisters() {
        let (hub, locks) = make_hub_and_locks();
        let config = ValidatorAgentConfig::new(ValidationConfig::disabled());
        let agent = ValidatorAgent::new("val-hub", config, Arc::clone(&hub), locks);

        // Before validate — not registered
        assert!(!hub.is_registered("val-hub").await);

        let _result = agent.validate().await.unwrap();

        // After validate — unregistered
        assert!(!hub.is_registered("val-hub").await);
    }

    #[tokio::test]
    async fn test_validator_acquires_read_locks() {
        let (hub, locks) = make_hub_and_locks();

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("locked.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        // Pre-acquire a WRITE lock as another agent on the same file
        let _write_guard = locks
            .acquire_lock("other-agent", &file_path, LockType::Write)
            .await
            .unwrap();

        let mut vc = ValidationConfig::disabled();
        vc.working_directory = dir.path().to_string_lossy().to_string();
        vc.working_set_files = vec!["locked.rs".to_string()];

        let config = ValidatorAgentConfig::new(vc);
        let agent = ValidatorAgent::new("val-lock", config, hub, locks);

        // Validation should still succeed (best-effort lock acquisition)
        let result = agent.validate().await.unwrap();
        assert!(result.success);
        // Lock was blocked so 0 acquired
        assert_eq!(result.locks_acquired, 0);
    }

    #[tokio::test]
    async fn test_validator_concurrent_read_locks() {
        let (hub, locks) = make_hub_and_locks();

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("shared.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        // Pre-acquire a READ lock as another agent
        let _read_guard = locks
            .acquire_lock("reader-agent", &file_path, LockType::Read)
            .await
            .unwrap();

        let mut vc = ValidationConfig::disabled();
        vc.working_directory = dir.path().to_string_lossy().to_string();
        vc.working_set_files = vec!["shared.rs".to_string()];

        let config = ValidatorAgentConfig::new(vc);
        let agent = ValidatorAgent::new("val-read", config, hub, locks);

        let result = agent.validate().await.unwrap();
        assert!(result.success);
        // Both readers can hold the lock simultaneously
        assert_eq!(result.locks_acquired, 1);
    }

    #[tokio::test]
    async fn test_spawn_validator_agent() {
        let (hub, locks) = make_hub_and_locks();
        let config = ValidatorAgentConfig::new(ValidationConfig::disabled());
        let agent = Arc::new(ValidatorAgent::new("val-spawn", config, hub, locks));

        let handle = spawn_validator_agent(agent);
        let result = handle.await.unwrap().unwrap();
        assert!(result.success);
        assert_eq!(result.agent_id, "val-spawn");
    }

    #[tokio::test]
    async fn test_result_metadata() {
        let (hub, locks) = make_hub_and_locks();

        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("a.rs");
        let file2 = dir.path().join("b.rs");
        std::fs::write(&file1, "fn a() {}").unwrap();
        std::fs::write(&file2, "fn b() {}").unwrap();

        let mut vc = ValidationConfig::disabled();
        vc.working_directory = dir.path().to_string_lossy().to_string();
        vc.working_set_files = vec!["a.rs".to_string(), "b.rs".to_string()];

        let config = ValidatorAgentConfig::new(vc);
        let agent = ValidatorAgent::new("val-meta", config, hub, locks);

        let result = agent.validate().await.unwrap();
        assert_eq!(result.agent_id, "val-meta");
        assert_eq!(result.files_checked, 2);
        assert!(result.duration.as_nanos() > 0);
        assert!(result.success);
    }
}
