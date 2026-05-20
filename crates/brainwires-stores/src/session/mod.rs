#![deny(missing_docs)]
//! Pluggable session-persistence for the Brainwires Agent Framework.
//!
//! See the crate-level README for an overview. The [`SessionStore`] trait is
//! the single extension point; [`InMemorySessionStore`] is the default impl
//! used by tests and ephemeral sessions, and `SqliteSessionStore` (behind the
//! crate's `sqlite` feature) provides disk-backed persistence.

/// Interactive session-broker surface (live registry, spawn/list/send/history).
pub mod broker;
mod error;
mod in_memory;
#[cfg(feature = "sqlite")]
mod sqlite;
mod types;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

pub use brainwires_core::Message;
pub use broker::{SessionBroker, SessionMessage, SessionSummary, SpawnRequest, SpawnedSession};
pub use error::SessionError;
pub use in_memory::InMemorySessionStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSessionStore;
pub use types::{SessionId, SessionRecord};

/// Pagination window passed to [`SessionStore::list_paginated`].
///
/// `offset` rows are skipped from the start of the listing; `limit`, when
/// `Some`, caps how many rows are returned. `Default` is `{ offset: 0, limit: None }`,
/// which is equivalent to the unbounded [`SessionStore::list`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListOptions {
    /// Number of records to skip from the start of the listing.
    pub offset: usize,
    /// Maximum number of records to return. `None` means no cap.
    pub limit: Option<usize>,
}

impl ListOptions {
    /// Convenience constructor.
    pub fn new(offset: usize, limit: Option<usize>) -> Self {
        Self { offset, limit }
    }
}

/// Trait implemented by every session-persistence backend.
///
/// Implementations must be cheap to share via `Arc` and safe to call
/// concurrently from any async context.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Load a session's full transcript. Returns `Ok(None)` when the id is
    /// not known — callers should treat this as "fresh session".
    async fn load(&self, id: &SessionId) -> Result<Option<Vec<Message>>>;

    /// Overwrite a session's full transcript. Creating the session if
    /// it didn't already exist.
    ///
    /// Implementations should treat the provided slice as the authoritative
    /// state and persist it atomically — a crash mid-write must leave the
    /// store with either the old or new transcript, never a partial one.
    async fn save(&self, id: &SessionId, messages: &[Message]) -> Result<()>;

    /// Enumerate every session the store knows about, newest-last. Returns
    /// metadata only — use [`Self::load`] to read message content.
    async fn list(&self) -> Result<Vec<SessionRecord>>;

    /// Enumerate sessions with `offset` / `limit` pagination. The default
    /// implementation calls [`Self::list`] and slices in memory; backends that
    /// can push the window down to storage (e.g. SQLite `LIMIT/OFFSET`) should
    /// override this to avoid loading the full set.
    async fn list_paginated(&self, opts: ListOptions) -> Result<Vec<SessionRecord>> {
        let all = self.list().await?;
        let start = opts.offset.min(all.len());
        let end = match opts.limit {
            Some(limit) => start.saturating_add(limit).min(all.len()),
            None => all.len(),
        };
        Ok(all[start..end].to_vec())
    }

    /// Remove a session. Deleting an unknown id is a no-op (not an error).
    async fn delete(&self, id: &SessionId) -> Result<()>;
}

/// Convenience alias used by downstream crates that hold the store behind
/// an `Arc`.
pub type ArcSessionStore = Arc<dyn SessionStore>;
