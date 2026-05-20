//! Typed error taxonomy for `brainwires-storage`.
//!
//! Public APIs in this crate return `anyhow::Result<T>` for historical reasons
//! and to keep the trait surface stable across nine backend impls. [`StorageError`]
//! gives callers a structured view on top of that: backends and domain stores
//! attach `StorageError` values to the anyhow context, and downstream code
//! recovers the typed variant with `err.downcast_ref::<StorageError>()`.
//!
//! This mirrors the `ResilienceError` pattern already used by
//! `brainwires-call-policy`.
//!
//! ```rust,ignore
//! use brainwires_storage::StorageError;
//! match err.downcast_ref::<StorageError>() {
//!     Some(StorageError::NotFound { .. })   => /* treat as empty */,
//!     Some(StorageError::Conflict { .. })   => /* retry or surface */,
//!     Some(StorageError::Backend { .. })    => /* log + bubble up */,
//!     _ => /* opaque anyhow::Error */,
//! }
//! ```

use thiserror::Error;

/// Structured error variants attachable to `anyhow::Error` returned by
/// `StorageBackend` / `VectorDatabase` impls and the domain stores.
///
/// Construct via the helper methods below; surface via
/// `anyhow::Error::from(StorageError::…)` or `Err(StorageError::….into())`.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Backend-level failure — connection, schema, query execution.
    /// `backend` identifies the origin (e.g. `"lance"`, `"postgres"`, `"qdrant"`).
    #[error("{backend} backend error: {message}")]
    Backend {
        /// Backend identifier.
        backend: &'static str,
        /// Human-readable context.
        message: String,
    },

    /// Serialisation / deserialisation of a stored payload failed.
    #[error("storage serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Underlying I/O (file-backed stores, SQLite WAL, local vector indexes).
    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Row/document was not found. Prefer this over returning `Ok(None)` at
    /// the `anyhow` layer when a caller explicitly asked for a known id.
    #[error("not found: {kind} {id}")]
    NotFound {
        /// Object kind (e.g. `"message"`, `"session"`, `"plan"`).
        kind: &'static str,
        /// Caller-provided identifier.
        id: String,
    },

    /// Unique-constraint or optimistic-concurrency violation.
    #[error("conflict on {kind} {id}: {reason}")]
    Conflict {
        /// Object kind.
        kind: &'static str,
        /// Caller-provided identifier.
        id: String,
        /// Why it conflicted (duplicate, stale version, etc.).
        reason: String,
    },

    /// Input that violates a store-level invariant (empty vector, wrong
    /// dimension, unsupported filter shape).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Feature requested is not compiled or not supported by the active
    /// backend. Emitted by the capability gate in `BackendCapabilities`.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl StorageError {
    /// Shorthand for the `Backend` variant.
    pub fn backend(backend: &'static str, message: impl Into<String>) -> Self {
        Self::Backend {
            backend,
            message: message.into(),
        }
    }

    /// Shorthand for the `NotFound` variant.
    pub fn not_found(kind: &'static str, id: impl Into<String>) -> Self {
        Self::NotFound {
            kind,
            id: id.into(),
        }
    }

    /// Shorthand for the `Conflict` variant.
    pub fn conflict(kind: &'static str, id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Conflict {
            kind,
            id: id.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_displays_include_context() {
        let e = StorageError::not_found("message", "msg-42");
        assert_eq!(e.to_string(), "not found: message msg-42");

        let e = StorageError::backend("lance", "table missing");
        assert!(e.to_string().contains("lance"));
        assert!(e.to_string().contains("table missing"));

        let e = StorageError::conflict("session", "s-7", "stale version");
        assert!(e.to_string().contains("s-7"));
        assert!(e.to_string().contains("stale version"));
    }

    #[test]
    fn rides_anyhow_and_roundtrips_via_downcast() {
        let e: anyhow::Error = StorageError::not_found("plan", "p-1").into();
        let typed = e
            .downcast_ref::<StorageError>()
            .expect("typed variant preserved through anyhow");
        assert!(matches!(typed, StorageError::NotFound { kind: "plan", .. }));
    }
}
