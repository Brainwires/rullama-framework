#![deny(missing_docs)]
//! # rullama-knowledge — the knowledge layer
//!
//! Knowledge graphs, behavioral/personal knowledge systems, a brain client for
//! persistent thoughts, and entity extraction for the rullama agent framework.
//!
//! - **BrainClient** — persistent thought storage with semantic search
//! - **Entity/Relationship Graph** — entity types, co-occurrence, impact analysis
//! - **BKS** — Behavioral Knowledge System (shared truths with confidence scoring)
//! - **PKS** — Personal Knowledge System (user-scoped facts)
//! - **Fact Extraction** — automatic categorization and tag extraction
//!
//! The prompting, RAG, spectral, and code-analysis subsystems that once lived
//! here have moved to dedicated crates: adaptive prompting →
//! [`rullama-prompting`](https://docs.rs/rullama-prompting); RAG / hybrid
//! retrieval / spectral / code-analysis →
//! [`rullama-rag`](https://docs.rs/rullama-rag); offline memory consolidation
//! ("dream") → `rullama-stores` (`dream` feature). Depend on those directly.

// Re-export core types
pub use rullama_core;

// ── Knowledge (from rullama-brain) ──────────────────────────────────────

/// Knowledge graph, entities, thoughts, BKS/PKS, brain client.
#[cfg(feature = "knowledge")]
pub mod knowledge;

// Prompting lives in `rullama-prompting` — depend on that directly.

// ── RAG, spectral, code_analysis ──────────────────────────────────────────
// All three live in `rullama-rag`. Spectral and code_analysis travel
// with RAG (no external consumers, only used by `rag::client::*`).
// Depend on `rullama-rag` directly.

// ── Re-exports (knowledge) ─────────────────────────────────────────────────

#[cfg(feature = "knowledge")]
pub use knowledge::brain_client::BrainClient;
#[cfg(feature = "knowledge")]
pub use knowledge::config::{DispositionTrait, MemoryBankConfig};
#[cfg(feature = "knowledge")]
pub use knowledge::entity::{
    ContradictionEvent, ContradictionKind, Entity, EntityStore, EntityStoreStats, EntityType,
    ExtractionResult, Relationship,
};
#[cfg(feature = "knowledge")]
pub use knowledge::relationship_graph::{
    EdgeType, EntityContext, GraphEdge, GraphNode, RelationshipGraph,
};
#[cfg(feature = "knowledge")]
pub use knowledge::thought::{Thought, ThoughtCategory, ThoughtSource};
#[cfg(feature = "knowledge")]
pub use knowledge::types::{
    CaptureThoughtRequest, CaptureThoughtResponse, DeleteThoughtRequest, DeleteThoughtResponse,
    GetThoughtRequest, GetThoughtResponse, ListRecentRequest, ListRecentResponse,
    MemoryStatsRequest, MemoryStatsResponse, SearchKnowledgeRequest, SearchKnowledgeResponse,
    SearchMemoryRequest, SearchMemoryResponse,
};

// Prompting / RAG / spectral / code-analysis live in `rullama-prompting`
// and `rullama-rag` — there are no re-exports here.

/// Prelude for convenient imports.
pub mod prelude {
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::brain_client::BrainClient;
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::entity::{Entity, EntityStore, EntityType};
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::thought::{Thought, ThoughtCategory};
}
