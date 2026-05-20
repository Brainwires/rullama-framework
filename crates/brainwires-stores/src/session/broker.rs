//! Interactive surface for live session orchestration.
//!
//! [`SessionStore`](crate::SessionStore) is the **persistence** side of the
//! session abstraction: load/save the messages of one session. [`SessionBroker`]
//! is the **interactive** side: list peers, read history, push a message, or
//! spawn a child session — all against a host-provided live session registry.
//!
//! These two sit in the same crate because both are about "a session", but
//! they're paired interfaces, not a single one — a host can implement either
//! independently.
//!
//! `brainwires-session` is a framework crate — it does not know about a
//! specific gateway's per-user session map or any concrete agent type.
//! Hosts (e.g., `brainwires-tools::SessionsTool` or
//! `extras/brainclaw/gateway/src/sessions_broker.rs`) implement
//! [`SessionBroker`] over their real registry and hand an
//! `Arc<dyn SessionBroker>` to the consumer.
//!
//! Lives here so all session abstractions sit together and the
//! `SessionId` type is shared between persistence and brokering.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::types::SessionId;

/// Summary metadata for a single session, returned by `sessions_list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// The session identifier.
    pub id: SessionId,
    /// Originating channel (e.g. `"discord"`, `"web"`, `"internal"`).
    pub channel: String,
    /// Peer handle — user id on the channel, or `"spawned-by-<parent>"`.
    pub peer: String,
    /// When the session was first created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// When the session last received or produced a message.
    pub last_active: chrono::DateTime<chrono::Utc>,
    /// Number of messages currently in the session's transcript.
    pub message_count: usize,
    /// Parent session that spawned this one, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<SessionId>,
}

/// A single message from a session's transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    /// `"user"` | `"assistant"` | `"system"` | `"tool"`.
    pub role: String,
    /// Message text. Tool calls/results are stringified.
    pub content: String,
    /// When the message was recorded (may be approximate if the underlying
    /// agent does not track per-message timestamps).
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Parameters for [`SessionBroker::spawn`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnRequest {
    /// Initial user message to seed the new session with.
    pub prompt: String,
    /// Optional provider/model override. `None` = inherit from parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional system prompt override. `None` = inherit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Tools to allow in the spawned session. `None` = inherit parent's toolset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// If `true`, block until the spawned session produces its first
    /// assistant message (or [`Self::wait_timeout_secs`] elapses) and return
    /// that in the tool result. Default: `false` — return immediately with
    /// just the new session id.
    #[serde(default)]
    pub wait_for_first_reply: bool,
    /// Seconds to wait when [`Self::wait_for_first_reply`] is `true`.
    /// Default: `60`.
    #[serde(default = "default_wait_timeout_secs")]
    pub wait_timeout_secs: u64,
}

fn default_wait_timeout_secs() -> u64 {
    60
}

impl Default for SpawnRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            model: None,
            system: None,
            tools: None,
            wait_for_first_reply: false,
            wait_timeout_secs: default_wait_timeout_secs(),
        }
    }
}

/// Result of [`SessionBroker::spawn`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnedSession {
    /// The id of the newly-created session.
    pub id: SessionId,
    /// Set iff `wait_for_first_reply` was `true` and the first assistant
    /// message arrived within the timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_reply: Option<SessionMessage>,
}

/// Host-provided bridge from session-control tools to the real session registry.
///
/// Implementations must be cheap to clone-via-`Arc` and safe to call from any
/// async context.
#[async_trait]
pub trait SessionBroker: Send + Sync {
    /// List every live session the host knows about.
    async fn list(&self) -> anyhow::Result<Vec<SessionSummary>>;

    /// Read a session's transcript, newest-last, capped at `limit` entries
    /// (`None` = use the host's sensible default).
    async fn history(
        &self,
        id: &SessionId,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<SessionMessage>>;

    /// Inject a user-role message into `id`'s inbound queue. Fire-and-forget
    /// — the target session processes it asynchronously.
    async fn send(&self, id: &SessionId, text: String) -> anyhow::Result<()>;

    /// Create a new session as a child of `parent`, seeded with `req.prompt`.
    async fn spawn(&self, parent: &SessionId, req: SpawnRequest) -> anyhow::Result<SpawnedSession>;
}
