//! Typed errors surfaced by session stores.

use thiserror::Error;

/// Errors surfaced by session-store implementations.
#[derive(Debug, Error)]
pub enum SessionError {
    /// Serialising or deserialising messages failed.
    #[error("session serialization: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Underlying storage layer (sqlite, filesystem) failed.
    #[error("session storage: {0}")]
    Storage(String),
}
