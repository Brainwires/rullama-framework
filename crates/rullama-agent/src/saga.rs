//! Saga-Style Compensating Transactions
//!
//! Based on SagaLLM (arXiv:2503.11951), this module implements the Saga pattern
//! for multi-step operations. When an operation fails mid-way, compensation
//! actions are executed in reverse order to undo completed sub-operations.
//!
//! # Key Concepts
//!
//! - **CompensableOperation**: A trait for operations that can be undone
//! - **SagaExecutor**: Manages execution and compensation of operation sequences
//! - **Checkpoint**: Captures state before operations for rollback
//! - **CompensationReport**: Summary of compensation actions taken
//!
//! # Example
//!
//! ```ignore
//! let mut saga = SagaExecutor::new("agent-1", "edit-and-build");
//!
//! // Execute operations in sequence
//! saga.execute_step(Arc::new(FileEditOp { path, content })).await?;
//! saga.execute_step(Arc::new(GitStageOp { files })).await?;
//! saga.execute_step(Arc::new(BuildOp { project })).await?;
//!
//! // If any step fails, compensate all completed operations
//! if failed {
//!     let report = saga.compensate_all().await?;
//!     // Files restored, staging undone
//! }
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// A compensable operation that can be undone
#[async_trait]
pub trait CompensableOperation: Send + Sync {
    /// Execute the forward operation
    async fn execute(&self) -> Result<OperationResult>;

    /// Compensate (undo) the operation
    async fn compensate(&self, result: &OperationResult) -> Result<()>;

    /// Get operation description for logging
    fn description(&self) -> String;

    /// Get the operation type (for categorization)
    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::Generic
    }
}

/// Result of an operation, needed for compensation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResult {
    /// Unique identifier for this operation.
    pub operation_id: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// State captured before operation (for rollback)
    #[serde(skip)]
    pub checkpoint: Option<Checkpoint>,
    /// Metadata needed for compensation
    pub compensation_data: serde_json::Value,
    /// Output from the operation
    pub output: Option<String>,
}

impl OperationResult {
    /// Create a successful operation result.
    pub fn success(operation_id: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            success: true,
            checkpoint: None,
            compensation_data: serde_json::Value::Null,
            output: None,
        }
    }

    /// Create a successful operation result with compensation data.
    pub fn success_with_data(operation_id: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            operation_id: operation_id.into(),
            success: true,
            checkpoint: None,
            compensation_data: data,
            output: None,
        }
    }

    /// Create a failed operation result.
    pub fn failure(operation_id: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            success: false,
            checkpoint: None,
            compensation_data: serde_json::Value::Null,
            output: None,
        }
    }

    /// Attach a checkpoint for rollback.
    pub fn with_checkpoint(mut self, checkpoint: Checkpoint) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Attach output text to the result.
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = Some(output.into());
        self
    }
}

/// Checkpoint for state restoration
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint identifier.
    pub id: String,
    /// When the checkpoint was created.
    pub timestamp: Instant,
    /// File states before modification
    pub file_states: Vec<FileState>,
    /// Git state before modification
    pub git_state: Option<GitCheckpoint>,
}

impl Checkpoint {
    /// Create a new checkpoint with the given identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            timestamp: Instant::now(),
            file_states: Vec::new(),
            git_state: None,
        }
    }

    /// Add a file state to the checkpoint.
    pub fn with_file(mut self, path: PathBuf, content: String) -> Self {
        self.file_states.push(FileState {
            path,
            content_hash: Self::hash_content(&content),
            original_content: Some(content),
        });
        self
    }

    /// Set all file states at once.
    pub fn with_files(mut self, files: Vec<FileState>) -> Self {
        self.file_states = files;
        self
    }

    /// Attach a git checkpoint.
    pub fn with_git(mut self, git_state: GitCheckpoint) -> Self {
        self.git_state = Some(git_state);
        self
    }

    fn hash_content(content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Snapshot of a file's state for restoration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    /// File path.
    pub path: PathBuf,
    /// Hash of the file content.
    pub content_hash: String,
    /// Original content for small files (for direct restoration)
    pub original_content: Option<String>,
}

impl FileState {
    /// Create a file state with path and content hash.
    pub fn new(path: PathBuf, content_hash: String) -> Self {
        Self {
            path,
            content_hash,
            original_content: None,
        }
    }

    /// Create a file state with path and full content (auto-hashed).
    pub fn with_content(path: PathBuf, content: String) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);

        Self {
            path,
            content_hash: format!("{:x}", hasher.finish()),
            original_content: Some(content),
        }
    }
}

/// Snapshot of git state for restoration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCheckpoint {
    /// HEAD commit hash.
    pub head_commit: String,
    /// List of staged files.
    pub staged_files: Vec<String>,
    /// Current branch name.
    pub branch: String,
}

impl GitCheckpoint {
    /// Create a new git checkpoint.
    pub fn new(head_commit: impl Into<String>, branch: impl Into<String>) -> Self {
        Self {
            head_commit: head_commit.into(),
            staged_files: Vec::new(),
            branch: branch.into(),
        }
    }

    /// Set staged files for this checkpoint.
    pub fn with_staged(mut self, files: Vec<String>) -> Self {
        self.staged_files = files;
        self
    }
}

/// Types of saga operations for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaOperationType {
    /// File write operation
    FileWrite,
    /// File edit operation
    FileEdit,
    /// File delete operation
    FileDelete,
    /// Git staging operation
    GitStage,
    /// Git unstaging operation
    GitUnstage,
    /// Git commit operation
    GitCommit,
    /// Git branch creation
    GitBranchCreate,
    /// Git branch deletion
    GitBranchDelete,
    /// Build operation
    Build,
    /// Test operation
    Test,
    /// Generic operation
    Generic,
}

impl SagaOperationType {
    /// Returns true if this operation type can be compensated
    pub fn is_compensable(&self) -> bool {
        match self {
            SagaOperationType::FileWrite
            | SagaOperationType::FileEdit
            | SagaOperationType::FileDelete
            | SagaOperationType::GitStage
            | SagaOperationType::GitUnstage
            | SagaOperationType::GitCommit
            | SagaOperationType::GitBranchCreate
            | SagaOperationType::GitBranchDelete => true,
            // Build and test are idempotent, no compensation needed
            SagaOperationType::Build | SagaOperationType::Test | SagaOperationType::Generic => {
                false
            }
        }
    }

    /// Get the compensation description for this operation type
    pub fn compensation_description(&self) -> &'static str {
        match self {
            SagaOperationType::FileWrite => "Delete written file or restore from backup",
            SagaOperationType::FileEdit => "Restore original file content",
            SagaOperationType::FileDelete => "Restore deleted file",
            SagaOperationType::GitStage => "git reset HEAD <files>",
            SagaOperationType::GitUnstage => "git add <files>",
            SagaOperationType::GitCommit => "git reset --soft HEAD~1",
            SagaOperationType::GitBranchCreate => "git branch -d <branch>",
            SagaOperationType::GitBranchDelete => "Restore branch from reflog",
            SagaOperationType::Build => "No compensation (idempotent)",
            SagaOperationType::Test => "No compensation (idempotent)",
            SagaOperationType::Generic => "Custom compensation",
        }
    }
}

/// Saga executor that manages compensating transactions
pub struct SagaExecutor {
    /// Unique saga identifier
    pub saga_id: String,
    /// Agent executing this saga
    pub agent_id: String,
    /// Description of the saga's purpose
    pub description: String,
    /// When the saga started
    pub started_at: Instant,
    /// Completed operations in execution order
    completed_ops: RwLock<Vec<(Arc<dyn CompensableOperation>, OperationResult)>>,
    /// Compensation callbacks
    #[allow(clippy::type_complexity)]
    compensation_hooks: RwLock<Vec<Box<dyn Fn(&str, bool) + Send + Sync>>>,
    /// Current status
    status: RwLock<SagaStatus>,
}

/// Current status of a saga execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SagaStatus {
    /// Saga is in progress
    Running,
    /// Saga completed successfully
    Completed,
    /// Saga failed and compensation is needed
    Failed,
    /// Compensation is in progress
    Compensating,
    /// Compensation completed
    Compensated,
}

impl SagaExecutor {
    /// Create a new saga executor for the given agent and description.
    pub fn new(agent_id: impl Into<String>, description: impl Into<String>) -> Self {
        let agent_id = agent_id.into();
        let description = description.into();
        let saga_id = format!(
            "saga-{}-{}",
            agent_id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX epoch")
                .as_millis()
        );

        Self {
            saga_id,
            agent_id,
            description,
            started_at: Instant::now(),
            completed_ops: RwLock::new(Vec::new()),
            compensation_hooks: RwLock::new(Vec::new()),
            status: RwLock::new(SagaStatus::Running),
        }
    }

    /// Get current status
    pub async fn status(&self) -> SagaStatus {
        *self.status.read().await
    }

    /// Get number of completed operations
    pub async fn operation_count(&self) -> usize {
        self.completed_ops.read().await.len()
    }

    /// Execute an operation within the saga
    pub async fn execute_step(&self, op: Arc<dyn CompensableOperation>) -> Result<OperationResult> {
        // Check if saga is still running
        if *self.status.read().await != SagaStatus::Running {
            anyhow::bail!("Cannot execute step: saga is not running");
        }

        tracing::debug!(
            saga_id = %self.saga_id,
            operation = %op.description(),
            "Executing saga step"
        );

        let result = op.execute().await?;

        if result.success {
            self.completed_ops.write().await.push((op, result.clone()));
            tracing::debug!(
                saga_id = %self.saga_id,
                "Saga step completed successfully"
            );
        } else {
            *self.status.write().await = SagaStatus::Failed;
            tracing::warn!(
                saga_id = %self.saga_id,
                "Saga step failed"
            );
        }

        Ok(result)
    }

    /// Mark the saga as successfully completed
    pub async fn complete(&self) {
        *self.status.write().await = SagaStatus::Completed;
        tracing::info!(
            saga_id = %self.saga_id,
            operations = self.completed_ops.read().await.len(),
            "Saga completed successfully"
        );
    }

    /// Mark the saga as failed (triggers compensation need)
    pub async fn fail(&self) {
        *self.status.write().await = SagaStatus::Failed;
        tracing::warn!(
            saga_id = %self.saga_id,
            operations = self.completed_ops.read().await.len(),
            "Saga marked as failed"
        );
    }

    /// Compensate all completed operations in reverse order
    pub async fn compensate_all(&self) -> Result<CompensationReport> {
        *self.status.write().await = SagaStatus::Compensating;

        let mut report = CompensationReport::new(self.saga_id.clone());

        tracing::info!(
            saga_id = %self.saga_id,
            "Starting saga compensation"
        );

        // Pop operations in reverse order
        while let Some((op, result)) = self.completed_ops.write().await.pop() {
            let description = op.description();

            // Skip non-compensable operations
            if !op.operation_type().is_compensable() {
                tracing::debug!(
                    operation = %description,
                    "Skipping non-compensable operation"
                );
                report.add_skipped(&description, "Non-compensable operation type");
                continue;
            }

            tracing::debug!(
                operation = %description,
                "Compensating operation"
            );

            match op.compensate(&result).await {
                Ok(()) => {
                    report.add_success(&description);
                    tracing::debug!(
                        operation = %description,
                        "Compensation successful"
                    );
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    report.add_failure(&description, error_msg.clone());
                    tracing::error!(
                        operation = %description,
                        error = %error_msg,
                        "Compensation failed"
                    );
                    // Continue compensating other operations even if one fails
                }
            }
        }

        *self.status.write().await = SagaStatus::Compensated;

        // Call compensation hooks
        let hooks = self.compensation_hooks.read().await;
        let summary = report.summary();
        let all_successful = report.all_successful();
        for hook in hooks.iter() {
            hook(&summary, all_successful);
        }

        tracing::info!(
            saga_id = %self.saga_id,
            summary = %summary,
            "Saga compensation completed"
        );

        Ok(report)
    }

    /// Add a hook called after compensation
    pub async fn on_compensation<F>(&self, hook: F)
    where
        F: Fn(&str, bool) + Send + Sync + 'static,
    {
        self.compensation_hooks.write().await.push(Box::new(hook));
    }

    /// Get descriptions of all completed operations
    pub async fn get_operation_descriptions(&self) -> Vec<String> {
        self.completed_ops
            .read()
            .await
            .iter()
            .map(|(op, _)| op.description())
            .collect()
    }
}

/// Report of compensation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationReport {
    /// Saga this report belongs to.
    pub saga_id: String,
    /// Status of each compensation action.
    pub operations: Vec<CompensationStatus>,
    /// When compensation started (epoch millis).
    pub started_at: u64,
    /// When compensation finished (epoch millis).
    pub completed_at: Option<u64>,
}

impl CompensationReport {
    /// Create a new empty compensation report.
    pub fn new(saga_id: String) -> Self {
        Self {
            saga_id,
            operations: Vec::new(),
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX epoch")
                .as_millis() as u64,
            completed_at: None,
        }
    }

    /// Record a successful compensation action.
    pub fn add_success(&mut self, description: &str) {
        self.operations.push(CompensationStatus {
            description: description.to_string(),
            status: CompensationOutcome::Success,
            error: None,
        });
    }

    /// Record a failed compensation action.
    pub fn add_failure(&mut self, description: &str, error: String) {
        self.operations.push(CompensationStatus {
            description: description.to_string(),
            status: CompensationOutcome::Failed,
            error: Some(error),
        });
    }

    /// Record a skipped compensation action.
    pub fn add_skipped(&mut self, description: &str, reason: &str) {
        self.operations.push(CompensationStatus {
            description: description.to_string(),
            status: CompensationOutcome::Skipped,
            error: Some(reason.to_string()),
        });
    }

    /// Returns true if all compensations succeeded or were skipped.
    pub fn all_successful(&self) -> bool {
        self.operations.iter().all(|s| {
            matches!(
                s.status,
                CompensationOutcome::Success | CompensationOutcome::Skipped
            )
        })
    }

    /// Generate a human-readable summary of compensation outcomes.
    pub fn summary(&self) -> String {
        let successful = self
            .operations
            .iter()
            .filter(|s| s.status == CompensationOutcome::Success)
            .count();
        let failed = self
            .operations
            .iter()
            .filter(|s| s.status == CompensationOutcome::Failed)
            .count();
        let skipped = self
            .operations
            .iter()
            .filter(|s| s.status == CompensationOutcome::Skipped)
            .count();

        format!(
            "{} successful, {} failed, {} skipped (total: {})",
            successful,
            failed,
            skipped,
            self.operations.len()
        )
    }

    /// Mark the compensation report as completed with a timestamp.
    pub fn mark_completed(&mut self) {
        self.completed_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX epoch")
                .as_millis() as u64,
        );
    }
}

/// Status of a single compensation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationStatus {
    /// Description of the compensated operation.
    pub description: String,
    /// Outcome of the compensation attempt.
    pub status: CompensationOutcome,
    /// Error message if compensation failed.
    pub error: Option<String>,
}

/// Outcome of a compensation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompensationOutcome {
    /// Compensation succeeded.
    Success,
    /// Compensation failed.
    Failed,
    /// Compensation was skipped.
    Skipped,
}

// =============================================================================
// Common Compensable Operations
// =============================================================================

/// File write operation with compensation
pub struct FileWriteOp {
    /// Path to the file being written.
    pub path: PathBuf,
    /// Content to write.
    pub content: String,
    /// Whether this creates a new file or overwrites an existing one.
    pub is_new_file: bool,
}

#[cfg(feature = "native")]
#[async_trait]
impl CompensableOperation for FileWriteOp {
    async fn execute(&self) -> Result<OperationResult> {
        // Capture existing content if file exists
        let checkpoint = if self.path.exists() {
            let original_content = tokio::fs::read_to_string(&self.path).await?;
            Some(
                Checkpoint::new(format!("file-write-{}", self.path.display()))
                    .with_file(self.path.clone(), original_content),
            )
        } else {
            None
        };

        // Write new content
        tokio::fs::write(&self.path, &self.content).await?;

        let mut result = OperationResult::success_with_data(
            format!("file-write-{}", self.path.display()),
            serde_json::json!({
                "path": self.path.display().to_string(),
                "is_new_file": self.is_new_file,
            }),
        );

        if let Some(cp) = checkpoint {
            result = result.with_checkpoint(cp);
        }

        Ok(result)
    }

    async fn compensate(&self, result: &OperationResult) -> Result<()> {
        if self.is_new_file {
            // Delete the new file
            if self.path.exists() {
                tokio::fs::remove_file(&self.path).await?;
            }
        } else if let Some(checkpoint) = &result.checkpoint {
            // Restore original content
            if let Some(file_state) = checkpoint.file_states.first()
                && let Some(original_content) = &file_state.original_content
            {
                tokio::fs::write(&self.path, original_content).await?;
            }
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!(
            "Write file: {} ({})",
            self.path.display(),
            if self.is_new_file { "new" } else { "existing" }
        )
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::FileWrite
    }
}

/// File edit operation with compensation
pub struct FileEditOp {
    /// Path to the file being edited.
    pub path: PathBuf,
    /// Original file content before edit.
    pub old_content: String,
    /// New file content after edit.
    pub new_content: String,
}

#[cfg(feature = "native")]
#[async_trait]
impl CompensableOperation for FileEditOp {
    async fn execute(&self) -> Result<OperationResult> {
        let checkpoint = Checkpoint::new(format!("file-edit-{}", self.path.display()))
            .with_file(self.path.clone(), self.old_content.clone());

        // Write new content
        tokio::fs::write(&self.path, &self.new_content).await?;

        Ok(
            OperationResult::success(format!("file-edit-{}", self.path.display()))
                .with_checkpoint(checkpoint),
        )
    }

    async fn compensate(&self, result: &OperationResult) -> Result<()> {
        if let Some(checkpoint) = &result.checkpoint
            && let Some(file_state) = checkpoint.file_states.first()
            && let Some(original_content) = &file_state.original_content
        {
            tokio::fs::write(&self.path, original_content).await?;
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!("Edit file: {}", self.path.display())
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::FileEdit
    }
}

/// Git stage operation with compensation
pub struct GitStageOp {
    /// Files to stage.
    pub files: Vec<PathBuf>,
    /// Working directory for the git command.
    pub working_dir: PathBuf,
}

#[cfg(feature = "native")]
#[async_trait]
impl CompensableOperation for GitStageOp {
    async fn execute(&self) -> Result<OperationResult> {
        use tokio::process::Command;

        for file in &self.files {
            Command::new("git")
                .args(["add", &file.display().to_string()])
                .current_dir(&self.working_dir)
                .output()
                .await?;
        }

        Ok(OperationResult::success_with_data(
            "git-stage",
            serde_json::json!({
                "files": self.files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>(),
            }),
        ))
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        use tokio::process::Command;

        for file in &self.files {
            Command::new("git")
                .args(["reset", "HEAD", &file.display().to_string()])
                .current_dir(&self.working_dir)
                .output()
                .await?;
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!("Git stage: {} files", self.files.len())
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::GitStage
    }
}

/// Git commit operation with compensation
pub struct GitCommitOp {
    /// Commit message.
    pub message: String,
    /// Working directory for the git command.
    pub working_dir: PathBuf,
}

#[cfg(feature = "native")]
#[async_trait]
impl CompensableOperation for GitCommitOp {
    async fn execute(&self) -> Result<OperationResult> {
        use tokio::process::Command;

        // Get current HEAD before commit
        let head_output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .await?;
        let previous_head = String::from_utf8_lossy(&head_output.stdout)
            .trim()
            .to_string();

        // Perform commit
        let output = Command::new("git")
            .args(["commit", "-m", &self.message])
            .current_dir(&self.working_dir)
            .output()
            .await?;

        if output.status.success() {
            Ok(OperationResult::success_with_data(
                "git-commit",
                serde_json::json!({
                    "previous_head": previous_head,
                    "message": self.message,
                }),
            ))
        } else {
            anyhow::bail!(
                "Git commit failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
        }
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        use tokio::process::Command;

        // Soft reset to undo the commit but keep changes staged
        Command::new("git")
            .args(["reset", "--soft", "HEAD~1"])
            .current_dir(&self.working_dir)
            .output()
            .await?;
        Ok(())
    }

    fn description(&self) -> String {
        format!("Git commit: {}", self.message)
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::GitCommit
    }
}

/// No-op compensable operation (for operations that don't need compensation)
pub struct NoOpCompensation {
    /// Description of the operation.
    pub description: String,
    /// Type of the operation.
    pub op_type: SagaOperationType,
}

#[async_trait]
impl CompensableOperation for NoOpCompensation {
    async fn execute(&self) -> Result<OperationResult> {
        Ok(OperationResult::success(&self.description))
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        // No compensation needed
        Ok(())
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn operation_type(&self) -> SagaOperationType {
        self.op_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_saga_executor_basic() {
        let saga = SagaExecutor::new("test-agent", "test saga");

        assert_eq!(saga.status().await, SagaStatus::Running);
        assert_eq!(saga.operation_count().await, 0);
    }

    #[tokio::test]
    async fn test_saga_execute_and_complete() {
        let saga = SagaExecutor::new("test-agent", "test saga");

        let op = Arc::new(NoOpCompensation {
            description: "test op".to_string(),
            op_type: SagaOperationType::Generic,
        });

        let result = saga.execute_step(op).await.unwrap();
        assert!(result.success);

        saga.complete().await;
        assert_eq!(saga.status().await, SagaStatus::Completed);
    }

    #[tokio::test]
    async fn test_saga_compensation() {
        let saga = SagaExecutor::new("test-agent", "test saga");

        // Execute a compensable operation
        let op = Arc::new(NoOpCompensation {
            description: "compensable op".to_string(),
            op_type: SagaOperationType::FileWrite,
        });

        saga.execute_step(op).await.unwrap();

        // Trigger compensation
        saga.fail().await;
        let report = saga.compensate_all().await.unwrap();

        assert_eq!(saga.status().await, SagaStatus::Compensated);
        assert_eq!(report.operations.len(), 1);
    }

    #[tokio::test]
    async fn test_file_write_op_compensation() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        // Create initial file
        std::fs::write(&file_path, "original content").unwrap();

        // Create file write operation
        let op = FileWriteOp {
            path: file_path.clone(),
            content: "new content".to_string(),
            is_new_file: false,
        };

        // Execute
        let result = op.execute().await.unwrap();
        assert!(result.success);

        // Verify new content
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "new content");

        // Compensate
        op.compensate(&result).await.unwrap();

        // Verify original content restored
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "original content");
    }

    #[tokio::test]
    async fn test_file_write_new_file_compensation() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");

        // Create file write operation for new file
        let op = FileWriteOp {
            path: file_path.clone(),
            content: "new content".to_string(),
            is_new_file: true,
        };

        // Execute
        let result = op.execute().await.unwrap();
        assert!(result.success);
        assert!(file_path.exists());

        // Compensate (should delete the file)
        op.compensate(&result).await.unwrap();
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_compensation_report() {
        let mut report = CompensationReport::new("test-saga".to_string());

        report.add_success("op1");
        report.add_failure("op2", "error".to_string());
        report.add_skipped("op3", "non-compensable");

        assert!(!report.all_successful());
        assert!(report.summary().contains("1 successful"));
        assert!(report.summary().contains("1 failed"));
        assert!(report.summary().contains("1 skipped"));
    }

    #[tokio::test]
    async fn test_operation_type_compensable() {
        assert!(SagaOperationType::FileWrite.is_compensable());
        assert!(SagaOperationType::FileEdit.is_compensable());
        assert!(SagaOperationType::GitStage.is_compensable());
        assert!(SagaOperationType::GitCommit.is_compensable());

        assert!(!SagaOperationType::Build.is_compensable());
        assert!(!SagaOperationType::Test.is_compensable());
        assert!(!SagaOperationType::Generic.is_compensable());
    }

    #[tokio::test]
    async fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new("test-cp")
            .with_file(PathBuf::from("/test/file.txt"), "content".to_string());

        assert_eq!(checkpoint.id, "test-cp");
        assert_eq!(checkpoint.file_states.len(), 1);
        assert_eq!(
            checkpoint.file_states[0].path,
            PathBuf::from("/test/file.txt")
        );
        assert!(checkpoint.file_states[0].original_content.is_some());
    }
}
