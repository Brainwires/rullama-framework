//! Persistent workflow state for crash-safe agent retry.
//!
//! When an agent crashes or is killed mid-execution, naïvely re-running it from
//! scratch risks duplicating already-completed side effects (file writes, API
//! calls, database inserts). `WorkflowCheckpoint` records which tool calls have
//! already been executed so that a re-started agent can skip them.
//!
//! # Storage
//!
//! [`FsWorkflowStateStore`](crate::workflow_state::FsWorkflowStateStore)
//! persists checkpoints as JSON files under
//! `~/.brainwires/workflow/<task_id>.json` using an atomic write (write to a
//! temp file, then rename) so the checkpoint is never left in a partially-written
//! state.
//!
//! [`InMemoryWorkflowStateStore`](crate::workflow_state::InMemoryWorkflowStateStore)
//! is provided for tests.
//!
//! # Usage
//!
//! ```ignore
//! let store = FsWorkflowStateStore::default();
//!
//! // On agent start — check for a prior checkpoint
//! if let Some(cp) = store.load_checkpoint(&task_id).await? {
//!     // Restore iteration count, skip completed tool_use_ids
//! }
//!
//! // After each successful tool call
//! store.mark_step_complete(&task_id, &tool_use_id, SideEffectRecord { ... }).await?;
//!
//! // On clean completion
//! store.delete_checkpoint(&task_id).await?;
//! ```

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

// ─── Data types ─────────────────────────────────────────────────────────────

/// Snapshot of an agent's execution progress that survives process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCheckpoint {
    /// ID of the task being executed.
    pub task_id: String,
    /// ID of the agent executing the task.
    pub agent_id: String,
    /// Number of loop iterations completed so far.
    pub step_index: u32,
    /// `tool_use_id` values for calls that have already been executed.
    /// An agent resuming from this checkpoint must skip these IDs.
    pub completed_tool_ids: HashSet<String>,
    /// Ordered log of side effects that have been applied.
    pub side_effects_log: Vec<SideEffectRecord>,
    /// Unix timestamp (seconds) of the last update.
    pub updated_at: i64,
}

impl WorkflowCheckpoint {
    /// Create a fresh checkpoint for the given task/agent pair.
    pub fn new(task_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            agent_id: agent_id.into(),
            step_index: 0,
            completed_tool_ids: HashSet::new(),
            side_effects_log: Vec::new(),
            updated_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Return `true` if this tool call has already been completed.
    pub fn is_completed(&self, tool_use_id: &str) -> bool {
        self.completed_tool_ids.contains(tool_use_id)
    }
}

/// Record of a single side effect that has been durably applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffectRecord {
    /// The `tool_use_id` of the call that produced this side effect.
    pub tool_use_id: String,
    /// Name of the tool that was called.
    pub tool_name: String,
    /// Primary target of the operation (file path, URL, etc.), if applicable.
    pub target: Option<String>,
    /// Unix timestamp (seconds) when the side effect was applied.
    pub completed_at: i64,
    /// Whether this side effect can be undone / is safe to retry.
    pub reversible: bool,
}

impl SideEffectRecord {
    /// Create a new `SideEffectRecord` for a completed tool call.
    pub fn new(
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        target: Option<String>,
        reversible: bool,
    ) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            tool_name: tool_name.into(),
            target,
            completed_at: chrono::Utc::now().timestamp(),
            reversible,
        }
    }
}

// ─── Trait ──────────────────────────────────────────────────────────────────

/// Persistence backend for workflow checkpoints.
#[async_trait]
pub trait WorkflowStateStore: Send + Sync {
    /// Persist or overwrite the checkpoint for `task_id`.
    async fn save_checkpoint(&self, cp: &WorkflowCheckpoint) -> Result<()>;

    /// Load the most recent checkpoint for `task_id`, or `None` if not found.
    async fn load_checkpoint(&self, task_id: &str) -> Result<Option<WorkflowCheckpoint>>;

    /// Record a completed tool call and its side effect.
    ///
    /// Implementations must apply the update atomically relative to concurrent
    /// callers for the same `task_id`.
    async fn mark_step_complete(
        &self,
        task_id: &str,
        tool_use_id: &str,
        effect: SideEffectRecord,
    ) -> Result<()>;

    /// Delete the checkpoint once a task completes successfully.
    async fn delete_checkpoint(&self, task_id: &str) -> Result<()>;
}

// ─── Filesystem implementation ───────────────────────────────────────────────

/// Stores workflow checkpoints as JSON files under `~/.brainwires/workflow/`.
///
/// Writes are atomic: the file is written to a `.tmp` path and then renamed,
/// so the checkpoint is never partially written.
#[derive(Debug, Clone)]
pub struct FsWorkflowStateStore {
    dir: PathBuf,
}

impl FsWorkflowStateStore {
    /// Create a store using the default directory (`~/.brainwires/workflow/`).
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("cannot determine home directory")?;
        Ok(home.join(".brainwires").join("workflow"))
    }

    /// Create a store rooted at `dir`. The directory is created if absent.
    pub fn new(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create workflow state dir: {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// Create a store using the default path, creating directories as needed.
    pub fn with_default_path() -> Result<Self> {
        Self::new(Self::default_path()?)
    }

    fn checkpoint_path(&self, task_id: &str) -> PathBuf {
        // Sanitise task_id so it's safe as a filename.
        let safe_id: String = task_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.dir.join(format!("{safe_id}.json"))
    }
}

#[async_trait]
impl WorkflowStateStore for FsWorkflowStateStore {
    async fn save_checkpoint(&self, cp: &WorkflowCheckpoint) -> Result<()> {
        let path = self.checkpoint_path(&cp.task_id);
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(cp).context("serialize checkpoint")?;
        tokio::fs::write(&tmp, &json)
            .await
            .with_context(|| format!("write checkpoint tmp: {}", tmp.display()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .with_context(|| format!("rename checkpoint: {}", path.display()))?;
        Ok(())
    }

    async fn load_checkpoint(&self, task_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        let path = self.checkpoint_path(task_id);
        match tokio::fs::read_to_string(&path).await {
            Ok(json) => {
                let cp: WorkflowCheckpoint =
                    serde_json::from_str(&json).context("deserialize checkpoint")?;
                Ok(Some(cp))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("read checkpoint: {}", path.display())),
        }
    }

    async fn mark_step_complete(
        &self,
        task_id: &str,
        tool_use_id: &str,
        effect: SideEffectRecord,
    ) -> Result<()> {
        // Load → mutate → save. This is not concurrent-safe across processes,
        // but task agents are single-threaded w.r.t. tool execution.
        let mut cp = self
            .load_checkpoint(task_id)
            .await?
            .unwrap_or_else(|| WorkflowCheckpoint::new(task_id, "unknown"));
        cp.completed_tool_ids.insert(tool_use_id.to_string());
        cp.side_effects_log.push(effect);
        cp.step_index += 1;
        cp.updated_at = chrono::Utc::now().timestamp();
        self.save_checkpoint(&cp).await
    }

    async fn delete_checkpoint(&self, task_id: &str) -> Result<()> {
        let path = self.checkpoint_path(task_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("delete checkpoint: {}", path.display())),
        }
    }
}

// ─── In-memory implementation (tests) ───────────────────────────────────────

/// In-memory workflow state store for tests — no filesystem I/O.
#[derive(Debug, Default)]
pub struct InMemoryWorkflowStateStore {
    checkpoints: Arc<Mutex<std::collections::HashMap<String, WorkflowCheckpoint>>>,
}

impl InMemoryWorkflowStateStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WorkflowStateStore for InMemoryWorkflowStateStore {
    async fn save_checkpoint(&self, cp: &WorkflowCheckpoint) -> Result<()> {
        self.checkpoints
            .lock()
            .await
            .insert(cp.task_id.clone(), cp.clone());
        Ok(())
    }

    async fn load_checkpoint(&self, task_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        Ok(self.checkpoints.lock().await.get(task_id).cloned())
    }

    async fn mark_step_complete(
        &self,
        task_id: &str,
        tool_use_id: &str,
        effect: SideEffectRecord,
    ) -> Result<()> {
        let mut map = self.checkpoints.lock().await;
        let cp = map
            .entry(task_id.to_string())
            .or_insert_with(|| WorkflowCheckpoint::new(task_id, "unknown"));
        cp.completed_tool_ids.insert(tool_use_id.to_string());
        cp.side_effects_log.push(effect);
        cp.step_index += 1;
        cp.updated_at = chrono::Utc::now().timestamp();
        Ok(())
    }

    async fn delete_checkpoint(&self, task_id: &str) -> Result<()> {
        self.checkpoints.lock().await.remove(task_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_roundtrip() {
        let store = InMemoryWorkflowStateStore::new();
        assert!(store.load_checkpoint("t1").await.unwrap().is_none());

        let cp = WorkflowCheckpoint::new("t1", "agent-1");
        store.save_checkpoint(&cp).await.unwrap();

        let loaded = store.load_checkpoint("t1").await.unwrap().unwrap();
        assert_eq!(loaded.task_id, "t1");
        assert_eq!(loaded.agent_id, "agent-1");
    }

    #[tokio::test]
    async fn mark_step_and_skip() {
        let store = InMemoryWorkflowStateStore::new();

        let effect = SideEffectRecord::new("use-1", "write_file", Some("src/main.rs".into()), true);
        store
            .mark_step_complete("t2", "use-1", effect)
            .await
            .unwrap();

        let cp = store.load_checkpoint("t2").await.unwrap().unwrap();
        assert!(cp.is_completed("use-1"));
        assert!(!cp.is_completed("use-2"));
        assert_eq!(cp.step_index, 1);
    }

    #[tokio::test]
    async fn delete_removes_checkpoint() {
        let store = InMemoryWorkflowStateStore::new();
        let cp = WorkflowCheckpoint::new("t3", "a");
        store.save_checkpoint(&cp).await.unwrap();
        store.delete_checkpoint("t3").await.unwrap();
        assert!(store.load_checkpoint("t3").await.unwrap().is_none());
    }

    // ── FsWorkflowStateStore tests ───────────────────────────────────────────

    #[tokio::test]
    async fn fs_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsWorkflowStateStore::new(dir.path().to_path_buf()).unwrap();

        assert!(store.load_checkpoint("task-a").await.unwrap().is_none());

        let cp = WorkflowCheckpoint::new("task-a", "agent-x");
        store.save_checkpoint(&cp).await.unwrap();

        let loaded = store.load_checkpoint("task-a").await.unwrap().unwrap();
        assert_eq!(loaded.task_id, "task-a");
        assert_eq!(loaded.agent_id, "agent-x");
    }

    #[tokio::test]
    async fn fs_atomic_write_produces_no_tmp_file_after_save() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsWorkflowStateStore::new(dir.path().to_path_buf()).unwrap();

        let cp = WorkflowCheckpoint::new("atomic-task", "a");
        store.save_checkpoint(&cp).await.unwrap();

        // The .tmp file must not be left behind after a successful save
        let tmp = dir.path().join("atomic-task.json.tmp");
        assert!(!tmp.exists(), ".tmp file should be gone after rename");

        // The real checkpoint file must exist
        let real = dir.path().join("atomic-task.json");
        assert!(real.exists());
    }

    #[tokio::test]
    async fn fs_mark_step_creates_checkpoint_implicitly() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsWorkflowStateStore::new(dir.path().to_path_buf()).unwrap();

        let effect = SideEffectRecord::new("use-99", "write_file", Some("foo.rs".into()), true);
        store
            .mark_step_complete("fresh-task", "use-99", effect)
            .await
            .unwrap();

        let cp = store.load_checkpoint("fresh-task").await.unwrap().unwrap();
        assert!(cp.is_completed("use-99"));
        assert_eq!(cp.step_index, 1);
        assert_eq!(cp.side_effects_log.len(), 1);
        assert_eq!(cp.side_effects_log[0].tool_name, "write_file");
    }

    #[tokio::test]
    async fn fs_delete_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsWorkflowStateStore::new(dir.path().to_path_buf()).unwrap();

        let cp = WorkflowCheckpoint::new("del-task", "a");
        store.save_checkpoint(&cp).await.unwrap();
        store.delete_checkpoint("del-task").await.unwrap();

        // Second delete of a non-existent file must not error
        store.delete_checkpoint("del-task").await.unwrap();

        // Also fine when the file never existed
        store.delete_checkpoint("never-existed").await.unwrap();
    }

    #[tokio::test]
    async fn fs_checkpoint_path_sanitizes_special_chars() {
        let dir = tempfile::tempdir().unwrap();
        let store = FsWorkflowStateStore::new(dir.path().to_path_buf()).unwrap();

        // task IDs with slashes, dots, and spaces must not create subdirs or fail
        let cp = WorkflowCheckpoint::new("proj/task.1 final", "a");
        store.save_checkpoint(&cp).await.unwrap();

        let loaded = store
            .load_checkpoint("proj/task.1 final")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.task_id, "proj/task.1 final");

        // Verify the file is directly in `dir`, not in a subdirectory
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "should be exactly one file, no subdirs");
    }
}
