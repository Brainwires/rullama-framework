//! Persistent state for crash recovery — checkpoints and resume tracking.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// State of the git repository at crash time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitState {
    /// Current branch name.
    pub branch: String,
    /// Last commit hash.
    pub last_commit: String,
    /// Files with uncommitted changes.
    pub dirty_files: Vec<String>,
    /// Whether there are uncommitted changes.
    pub has_uncommitted_changes: bool,
}

/// Checkpoint persisted before each improvement cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleCheckpoint {
    /// Index of the current cycle (0-based).
    pub cycle_index: u32,
    /// Total number of cycles planned.
    pub total_cycles: u32,
    /// ID of the task being executed.
    pub task_id: Option<String>,
    /// Strategy name of the current task.
    pub strategy: Option<String>,
    /// Git state at checkpoint time.
    pub git_state: GitState,
    /// Timestamp of the checkpoint.
    pub timestamp: DateTime<Utc>,
}

/// Crash context captured when a self-improvement session fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashContext {
    /// When the crash occurred.
    pub crash_time: DateTime<Utc>,
    /// Process exit code, if available.
    pub exit_code: Option<i32>,
    /// Signal that killed the process, if applicable.
    pub signal: Option<i32>,
    /// Last N lines of stderr output.
    pub stderr_tail: String,
    /// Index of the cycle that was running when the crash occurred.
    pub last_cycle_index: u32,
    /// ID of the task that was being executed.
    pub last_task_id: Option<String>,
    /// Strategy that was running.
    pub last_strategy: Option<String>,
    /// Working directory of the session.
    pub working_directory: String,
    /// Git state at crash time.
    pub git_state: GitState,
}

/// Recovery state file persisted across process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryState {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Unique crash identifier.
    pub crash_id: String,
    /// The crash context.
    pub crash_context: CrashContext,
    /// Number of fix attempts already made for this crash.
    pub fix_attempts: u32,
    /// Maximum fix attempts allowed.
    pub max_fix_attempts: u32,
    /// Recovery plan, populated after diagnosis.
    pub recovery_plan: Option<RecoveryPlanState>,
}

/// Serializable recovery plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlanState {
    /// Root cause analysis from the AI.
    pub root_cause: String,
    /// Strategy to apply.
    pub fix_strategy: String,
    /// Files that need fixing.
    pub files_to_fix: Vec<String>,
    /// Whether a git rollback is needed before fixing.
    pub rollback_needed: bool,
    /// Cycle index to resume from after fix.
    pub resume_from_cycle: u32,
}

impl RecoveryState {
    /// Load recovery state from a file, returning `None` if the file doesn't exist.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(path)?;
        let state: Self = serde_json::from_str(&content)?;
        Ok(Some(state))
    }

    /// Save recovery state to a file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Delete the recovery state file.
    pub fn cleanup(path: &Path) -> anyhow::Result<()> {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Check if this is a meta-crash (crash handler itself crashed).
    pub fn is_meta_crash(&self) -> bool {
        self.fix_attempts >= self.max_fix_attempts
    }
}

impl CycleCheckpoint {
    /// Save checkpoint to a file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load checkpoint from a file.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(path)?;
        let checkpoint: Self = serde_json::from_str(&content)?;
        Ok(Some(checkpoint))
    }
}

/// Capture the current git state of a repository.
pub async fn capture_git_state(repo_path: &str) -> anyhow::Result<GitState> {
    let branch = tokio::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(repo_path)
        .output()
        .await?;
    let branch = String::from_utf8_lossy(&branch.stdout).trim().to_string();

    let commit = tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_path)
        .output()
        .await?;
    let last_commit = String::from_utf8_lossy(&commit.stdout).trim().to_string();

    let status = tokio::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_path)
        .output()
        .await?;
    let status_output = String::from_utf8_lossy(&status.stdout);
    let dirty_files: Vec<String> = status_output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect();
    let has_uncommitted_changes = !dirty_files.is_empty();

    Ok(GitState {
        branch,
        last_commit,
        dirty_files,
        has_uncommitted_changes,
    })
}

/// Derive a checkpoint file path from the recovery state file path.
pub fn checkpoint_path(state_file: &Path) -> PathBuf {
    let stem = state_file.file_stem().unwrap_or_default().to_string_lossy();
    state_file.with_file_name(format!("{stem}-checkpoint.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_state_roundtrip() {
        let state = RecoveryState {
            version: 1,
            crash_id: "test-123".to_string(),
            crash_context: CrashContext {
                crash_time: Utc::now(),
                exit_code: Some(1),
                signal: None,
                stderr_tail: "panicked at 'test'".to_string(),
                last_cycle_index: 3,
                last_task_id: Some("task-1".to_string()),
                last_strategy: Some("clippy".to_string()),
                working_directory: "/tmp/test".to_string(),
                git_state: GitState {
                    branch: "self-improve/test".to_string(),
                    last_commit: "abc123".to_string(),
                    dirty_files: vec!["src/main.rs".to_string()],
                    has_uncommitted_changes: true,
                },
            },
            fix_attempts: 0,
            max_fix_attempts: 3,
            recovery_plan: None,
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let deserialized: RecoveryState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.crash_id, "test-123");
        assert_eq!(deserialized.crash_context.last_cycle_index, 3);
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn sample_git_state() -> GitState {
        GitState {
            branch: "self-improve/test".to_string(),
            last_commit: "abc123".to_string(),
            dirty_files: vec!["src/main.rs".to_string()],
            has_uncommitted_changes: true,
        }
    }

    fn sample_checkpoint() -> CycleCheckpoint {
        CycleCheckpoint {
            cycle_index: 4,
            total_cycles: 10,
            task_id: Some("task-42".to_string()),
            strategy: Some("clippy".to_string()),
            git_state: sample_git_state(),
            timestamp: Utc::now(),
        }
    }

    fn sample_recovery_state() -> RecoveryState {
        RecoveryState {
            version: 1,
            crash_id: "crash-xyz".to_string(),
            crash_context: CrashContext {
                crash_time: Utc::now(),
                exit_code: Some(101),
                signal: None,
                stderr_tail: "panicked at main.rs:1".to_string(),
                last_cycle_index: 4,
                last_task_id: Some("task-42".to_string()),
                last_strategy: Some("clippy".to_string()),
                working_directory: "/tmp/work".to_string(),
                git_state: sample_git_state(),
            },
            fix_attempts: 0,
            max_fix_attempts: 3,
            recovery_plan: None,
        }
    }

    // ── Recovery tests ──────────────────────────────────────────────────

    #[test]
    fn recovery_with_no_checkpoint_is_safe() {
        // Calling load on a missing path returns Ok(None) — the "no prior checkpoint"
        // case must not panic and must not pretend there's state.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("does-not-exist.json");
        let loaded = RecoveryState::load(&path).expect("load should be Ok");
        assert!(loaded.is_none(), "no checkpoint → no state");

        let checkpoint = CycleCheckpoint::load(&path).expect("load should be Ok");
        assert!(checkpoint.is_none(), "no checkpoint → no checkpoint");

        // Cleanup on a missing path is a no-op, not an error.
        RecoveryState::cleanup(&path).expect("cleanup on missing path should be a no-op");
    }

    #[test]
    fn recovery_restores_last_checkpoint() {
        // Simulate the "crash" path: write a checkpoint, then load it back and
        // assert the state matches what was saved.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("state.json");

        let original = sample_recovery_state();
        original.save(&path).expect("save should succeed");

        let restored = RecoveryState::load(&path)
            .expect("load should succeed")
            .expect("saved state should load back");

        assert_eq!(restored.crash_id, original.crash_id);
        assert_eq!(restored.version, original.version);
        assert_eq!(
            restored.crash_context.last_cycle_index,
            original.crash_context.last_cycle_index
        );
        assert_eq!(
            restored.crash_context.git_state.branch,
            original.crash_context.git_state.branch
        );
        assert_eq!(restored.fix_attempts, 0);
        assert!(!restored.is_meta_crash());
    }

    #[test]
    fn cycle_checkpoint_save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("cycle-checkpoint.json");

        let original = sample_checkpoint();
        original.save(&path).expect("save should succeed");

        let loaded = CycleCheckpoint::load(&path)
            .expect("load should succeed")
            .expect("checkpoint should exist on disk");

        assert_eq!(loaded.cycle_index, original.cycle_index);
        assert_eq!(loaded.total_cycles, original.total_cycles);
        assert_eq!(loaded.task_id, original.task_id);
        assert_eq!(loaded.strategy, original.strategy);
    }

    #[test]
    fn cleanup_removes_saved_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("state.json");

        sample_recovery_state().save(&path).unwrap();
        assert!(path.exists(), "state file should exist after save");

        RecoveryState::cleanup(&path).expect("cleanup should succeed");
        assert!(!path.exists(), "state file should be gone after cleanup");

        // Re-cleanup on now-missing path should still succeed.
        RecoveryState::cleanup(&path).expect("double cleanup should not error");
    }

    #[test]
    fn checkpoint_path_is_sibling_of_state_file() {
        let state = Path::new("/tmp/foo/bar.json");
        let cp = checkpoint_path(state);
        assert_eq!(
            cp,
            PathBuf::from("/tmp/foo/bar-checkpoint.json"),
            "checkpoint should be derived from state stem"
        );
    }

    #[test]
    fn save_creates_parent_directories() {
        // Safety guard: saving into a path whose parent doesn't exist should
        // not panic — `save` must create the directory tree.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("nested").join("deeper").join("state.json");
        sample_recovery_state()
            .save(&path)
            .expect("save should create missing parent dirs");
        assert!(path.exists(), "nested state file should exist after save");
    }

    #[test]
    fn is_meta_crash_when_max_attempts_reached() {
        let state = RecoveryState {
            version: 1,
            crash_id: "test".to_string(),
            crash_context: CrashContext {
                crash_time: Utc::now(),
                exit_code: None,
                signal: None,
                stderr_tail: String::new(),
                last_cycle_index: 0,
                last_task_id: None,
                last_strategy: None,
                working_directory: ".".to_string(),
                git_state: GitState {
                    branch: "main".to_string(),
                    last_commit: "abc".to_string(),
                    dirty_files: Vec::new(),
                    has_uncommitted_changes: false,
                },
            },
            fix_attempts: 3,
            max_fix_attempts: 3,
            recovery_plan: None,
        };
        assert!(state.is_meta_crash());
    }
}
