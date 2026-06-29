//! Concrete `AnalyticsSink` implementations.
//!
//! Two backends ship in-tree: the always-available `MemoryAnalyticsSink`
//! (ring-buffered, lock-free for reads) and the feature-gated
//! `SqliteAnalyticsSink` (disk-backed, used for the `AnalyticsQuery` APIs).

/// In-memory ring-buffer sink. Always available.
pub mod memory;

/// SQLite-backed durable sink. Requires the `sqlite` feature.
#[cfg(feature = "sqlite")]
pub mod sqlite;
