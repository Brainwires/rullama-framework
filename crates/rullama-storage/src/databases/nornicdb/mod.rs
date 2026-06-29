//! NornicDB vector database backend.
//!
//! Implements [`VectorDatabase`] by delegating to a `NornicTransport`
//! (REST, Bolt, or gRPC; see `transport`).  NornicDB-specific extensions such as cognitive
//! memory tiers, graph relationships, and raw Cypher access are exposed as
//! inherent methods on [`NornicDatabase`].

pub mod transport;

mod database;
mod helpers;
mod types;

mod tests;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use database::NornicDatabase;
pub use types::{CognitiveMemoryTier, NornicConfig, TransportKind};
