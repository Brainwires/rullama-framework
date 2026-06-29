//! Permission Relay — manages pending tool-approval requests between CLI and remote UI.
//!
//! When the CLI agent wants to execute a dangerous tool while a remote bridge is
//! active, it sends a `PermissionRequest` and waits on a oneshot channel.
//! The backend relays the request to the web UI; the user approves/denies;
//! the backend sends a `PermissionResponse` back; this module resolves
//! the waiting oneshot.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, oneshot};

/// Result of a permission request.
#[derive(Debug, Clone)]
pub struct PermissionDecision {
    /// Whether the tool execution was approved.
    pub approved: bool,
    /// Remember this decision for the rest of the session.
    pub remember_for_session: bool,
    /// Always allow this specific tool (no future prompts).
    pub always_allow: bool,
}

/// Manages in-flight permission requests.
///
/// Thread-safe — all operations go through an inner `Mutex`.
#[derive(Clone)]
pub struct PermissionRelay {
    inner: Arc<Mutex<RelayInner>>,
}

struct RelayInner {
    /// Pending requests: request_id → oneshot sender for the decision.
    pending: HashMap<String, oneshot::Sender<PermissionDecision>>,
    /// Tools that have been permanently allowed for this session.
    session_allowed: Vec<String>,
    /// Default timeout for permission requests.
    default_timeout: Duration,
}

impl PermissionRelay {
    /// Create a new permission relay with a default 60-second timeout.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RelayInner {
                pending: HashMap::new(),
                session_allowed: Vec::new(),
                default_timeout: Duration::from_secs(60),
            })),
        }
    }

    /// Check if a tool has been permanently allowed for this session.
    pub async fn is_session_allowed(&self, tool_name: &str) -> bool {
        let inner = self.inner.lock().await;
        inner.session_allowed.contains(&tool_name.to_string())
    }

    /// Register a new permission request. Returns the request_id and a receiver
    /// that will resolve when the remote user responds (or times out).
    pub async fn register_request(
        &self,
        request_id: String,
    ) -> oneshot::Receiver<PermissionDecision> {
        let (tx, rx) = oneshot::channel();
        let mut inner = self.inner.lock().await;
        inner.pending.insert(request_id, tx);
        rx
    }

    /// Resolve a pending request with the remote user's decision.
    /// Returns `true` if the request was found and resolved.
    pub async fn resolve(&self, request_id: &str, decision: PermissionDecision) -> bool {
        let mut inner = self.inner.lock().await;

        // If always_allow, remember for session
        if decision.always_allow || decision.remember_for_session {
            // We don't have tool_name here, but the caller can also call
            // add_session_allowed separately if needed.
        }

        if let Some(tx) = inner.pending.remove(request_id) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    /// Resolve a pending request and optionally mark the tool as session-allowed.
    pub async fn resolve_with_tool(
        &self,
        request_id: &str,
        tool_name: &str,
        decision: PermissionDecision,
    ) -> bool {
        let mut inner = self.inner.lock().await;

        if decision.always_allow && !inner.session_allowed.contains(&tool_name.to_string()) {
            inner.session_allowed.push(tool_name.to_string());
        }

        if let Some(tx) = inner.pending.remove(request_id) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    /// Cancel a pending request (e.g., on timeout). The receiver will get a
    /// `RecvError`, which the caller should treat as denial.
    pub async fn cancel(&self, request_id: &str) -> bool {
        let mut inner = self.inner.lock().await;
        inner.pending.remove(request_id).is_some()
    }

    /// Get the number of pending requests.
    pub async fn pending_count(&self) -> usize {
        let inner = self.inner.lock().await;
        inner.pending.len()
    }

    /// Get the default timeout duration.
    pub async fn default_timeout(&self) -> Duration {
        let inner = self.inner.lock().await;
        inner.default_timeout
    }

    /// Add a tool to the session-allowed list.
    pub async fn add_session_allowed(&self, tool_name: &str) {
        let mut inner = self.inner.lock().await;
        if !inner.session_allowed.contains(&tool_name.to_string()) {
            inner.session_allowed.push(tool_name.to_string());
        }
    }

    /// Clear all pending requests (e.g., on disconnect).
    pub async fn clear(&self) {
        let mut inner = self.inner.lock().await;
        inner.pending.clear();
    }
}

impl Default for PermissionRelay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_resolve() {
        let relay = PermissionRelay::new();

        let rx = relay.register_request("req-1".to_string()).await;
        assert_eq!(relay.pending_count().await, 1);

        let resolved = relay
            .resolve(
                "req-1",
                PermissionDecision {
                    approved: true,
                    remember_for_session: false,
                    always_allow: false,
                },
            )
            .await;
        assert!(resolved);
        assert_eq!(relay.pending_count().await, 0);

        let decision = rx.await.unwrap();
        assert!(decision.approved);
    }

    #[tokio::test]
    async fn test_resolve_unknown_request() {
        let relay = PermissionRelay::new();
        let resolved = relay
            .resolve(
                "nonexistent",
                PermissionDecision {
                    approved: true,
                    remember_for_session: false,
                    always_allow: false,
                },
            )
            .await;
        assert!(!resolved);
    }

    #[tokio::test]
    async fn test_session_allowed() {
        let relay = PermissionRelay::new();

        assert!(!relay.is_session_allowed("bash").await);

        relay
            .resolve_with_tool(
                "req-1",
                "bash",
                PermissionDecision {
                    approved: true,
                    remember_for_session: false,
                    always_allow: true,
                },
            )
            .await;

        assert!(relay.is_session_allowed("bash").await);
    }

    #[tokio::test]
    async fn test_cancel_request() {
        let relay = PermissionRelay::new();

        let rx = relay.register_request("req-1".to_string()).await;
        assert!(relay.cancel("req-1").await);
        assert_eq!(relay.pending_count().await, 0);

        // Receiver should get an error (sender dropped)
        assert!(rx.await.is_err());
    }
}
