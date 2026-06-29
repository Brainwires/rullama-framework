//! MCP Tasks primitive (SEP-1686).
//!
//! Provides a standardised lifecycle for long-running asynchronous tool calls:
//!
//! ```text
//! Working → Completed
//!          ↘ Failed
//!          ↘ Cancelled
//! Working → InputRequired → Working (loop)
//! ```
//!
//! [`McpTaskStore`] is a thread-safe in-memory store with optional TTL-based
//! expiry. Wire it into [`McpServer`](crate::McpServer) to expose
//! `tasks/create`, `tasks/get`, and `tasks/cancel` JSON-RPC methods.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use uuid::Uuid;

/// Default maximum number of retries before a task transitions to `Failed`.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// All possible states in the MCP task lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTaskState {
    /// Task is actively running.
    Working,
    /// Task is paused waiting for additional input from the caller.
    InputRequired,
    /// Task finished successfully.
    Completed,
    /// Task finished with an error.
    Failed,
    /// Task was explicitly cancelled by the caller.
    Cancelled,
}

impl std::fmt::Display for McpTaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Working => "working",
            Self::InputRequired => "input_required",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        };
        write!(f, "{}", s)
    }
}

/// A single MCP task entry.
#[derive(Debug, Clone)]
pub struct McpTask {
    /// Unique task identifier (UUID v4).
    pub id: String,
    /// Current lifecycle state.
    pub state: McpTaskState,
    /// Wall-clock creation time.
    pub created_at: Instant,
    /// When this task entry expires and will be evicted (if set).
    pub expires_at: Option<Instant>,
    /// Result payload for completed tasks.
    pub result: Option<serde_json::Value>,
    /// Error message for failed tasks.
    pub error: Option<String>,
    /// Number of times execution has been retried.
    pub retry_count: u32,
    /// Maximum allowed retries before the task automatically fails.
    pub max_retries: u32,
}

impl McpTask {
    /// Create a new task in the `Working` state.
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            state: McpTaskState::Working,
            created_at: Instant::now(),
            expires_at: None,
            result: None,
            error: None,
            retry_count: 0,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Set a TTL so the store evicts this task after the given duration.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.expires_at = Some(Instant::now() + ttl);
        self
    }

    /// Whether this task entry has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Instant::now() >= exp)
            .unwrap_or(false)
    }
}

impl Default for McpTask {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe in-memory store for [`McpTask`] entries.
///
/// Spawn with [`McpTaskStore::new`]; wire JSON-RPC dispatch into
/// [`McpServer`](crate::McpServer) by calling [`insert`], [`get`], and
/// [`cancel`] from your handler's `tasks/*` method implementations.
///
/// [`insert`]: McpTaskStore::insert
/// [`get`]: McpTaskStore::get
/// [`cancel`]: McpTaskStore::cancel
#[derive(Clone)]
pub struct McpTaskStore {
    inner: Arc<RwLock<HashMap<String, McpTask>>>,
}

impl McpTaskStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a task and return its ID.
    pub async fn insert(&self, task: McpTask) -> String {
        let id = task.id.clone();
        self.inner.write().await.insert(id.clone(), task);
        id
    }

    /// Retrieve a task by ID, returning `None` if not found or expired.
    pub async fn get(&self, id: &str) -> Option<McpTask> {
        let map = self.inner.read().await;
        let task = map.get(id)?;
        if task.is_expired() {
            None
        } else {
            Some(task.clone())
        }
    }

    /// Transition a task to `Cancelled`. Returns `false` if the task is not
    /// found, expired, or already in a terminal state.
    pub async fn cancel(&self, id: &str) -> bool {
        let mut map = self.inner.write().await;
        match map.get_mut(id) {
            Some(task)
                if !task.is_expired()
                    && !matches!(
                        task.state,
                        McpTaskState::Completed | McpTaskState::Failed | McpTaskState::Cancelled
                    ) =>
            {
                task.state = McpTaskState::Cancelled;
                true
            }
            _ => false,
        }
    }

    /// Update the state of a task. Returns `false` if the task is not found or expired.
    pub async fn update_state(&self, id: &str, state: McpTaskState) -> bool {
        let mut map = self.inner.write().await;
        match map.get_mut(id) {
            Some(task) if !task.is_expired() => {
                task.state = state;
                true
            }
            _ => false,
        }
    }

    /// Set a completed result on a task, transitioning it to `Completed`.
    pub async fn complete(&self, id: &str, result: serde_json::Value) -> bool {
        let mut map = self.inner.write().await;
        match map.get_mut(id) {
            Some(task) if !task.is_expired() => {
                task.state = McpTaskState::Completed;
                task.result = Some(result);
                true
            }
            _ => false,
        }
    }

    /// Fail a task with an error message, transitioning it to `Failed`.
    pub async fn fail(&self, id: &str, error: impl Into<String>) -> bool {
        let mut map = self.inner.write().await;
        match map.get_mut(id) {
            Some(task) if !task.is_expired() => {
                task.state = McpTaskState::Failed;
                task.error = Some(error.into());
                true
            }
            _ => false,
        }
    }

    /// Evict all expired task entries.
    pub async fn evict_expired(&self) -> usize {
        let mut map = self.inner.write().await;
        let before = map.len();
        map.retain(|_, task| !task.is_expired());
        before - map.len()
    }

    /// Return the number of tasks currently in the store (including expired).
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Return `true` if the store has no tasks.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

impl Default for McpTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_lifecycle_working_to_completed() {
        let store = McpTaskStore::new();
        let task = McpTask::new();
        let id = store.insert(task).await;

        assert_eq!(store.get(&id).await.unwrap().state, McpTaskState::Working);
        store.complete(&id, serde_json::json!({"ok": true})).await;
        assert_eq!(store.get(&id).await.unwrap().state, McpTaskState::Completed);
    }

    #[tokio::test]
    async fn test_task_lifecycle_working_to_failed() {
        let store = McpTaskStore::new();
        let id = store.insert(McpTask::new()).await;
        store.fail(&id, "timeout").await;
        let task = store.get(&id).await.unwrap();
        assert_eq!(task.state, McpTaskState::Failed);
        assert_eq!(task.error.as_deref(), Some("timeout"));
    }

    #[tokio::test]
    async fn test_task_lifecycle_working_to_cancelled() {
        let store = McpTaskStore::new();
        let id = store.insert(McpTask::new()).await;
        assert!(store.cancel(&id).await);
        assert_eq!(store.get(&id).await.unwrap().state, McpTaskState::Cancelled);
    }

    #[tokio::test]
    async fn test_cancel_terminal_task_returns_false() {
        let store = McpTaskStore::new();
        let id = store.insert(McpTask::new()).await;
        store.complete(&id, serde_json::json!({})).await;
        // Already completed — cancel should return false
        assert!(!store.cancel(&id).await);
    }

    #[tokio::test]
    async fn test_input_required_state() {
        let store = McpTaskStore::new();
        let id = store.insert(McpTask::new()).await;
        store.update_state(&id, McpTaskState::InputRequired).await;
        assert_eq!(
            store.get(&id).await.unwrap().state,
            McpTaskState::InputRequired
        );
    }

    #[tokio::test]
    async fn test_ttl_expiry_eviction() {
        let store = McpTaskStore::new();
        let task = McpTask::new().with_ttl(Duration::from_millis(1));
        let id = store.insert(task).await;

        // Wait for TTL to elapse
        tokio::time::sleep(Duration::from_millis(5)).await;

        // get() returns None for expired task
        assert!(store.get(&id).await.is_none());

        // evict_expired cleans up the backing map
        let evicted = store.evict_expired().await;
        assert_eq!(evicted, 1);
        assert_eq!(store.len().await, 0);
    }
}
