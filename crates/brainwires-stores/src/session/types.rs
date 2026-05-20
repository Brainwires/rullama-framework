//! Shared value types used by every [`crate::SessionStore`] impl.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Opaque identifier for a persisted session.
///
/// Callers pick the format — typical values are `"user-42"`, `"discord:12345"`,
/// or a bare UUID. Implementations treat it as an arbitrary UTF-8 token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Build a `SessionId` from anything convertible into a `String`.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }

    /// Borrow the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Metadata row returned by [`crate::SessionStore::list`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRecord {
    /// The session identifier.
    pub id: SessionId,
    /// Number of messages in the transcript.
    pub message_count: usize,
    /// When the session was first persisted.
    pub created_at: DateTime<Utc>,
    /// When the session was last written.
    pub updated_at: DateTime<Utc>,
}
