#![deny(missing_docs)]
//! # Brainwires Cognition — Unified Intelligence Layer
//!
//! This crate consolidates three previously separate crates into a single
//! coherent intelligence layer for the Brainwires Agent Framework:
//!
//! ## Knowledge (from brainwires-brain)
//! - **BrainClient** — Persistent thought storage with semantic search
//! - **Entity/Relationship Graph** — Entity types, co-occurrence, impact analysis
//! - **BKS** — Behavioral Knowledge System (shared truths with confidence scoring)
//! - **PKS** — Personal Knowledge System (user-scoped facts)
//! - **Fact Extraction** — Automatic categorization and tag extraction
//!
//! ## Prompting (from brainwires-prompting)
//! - **Techniques** — 15 prompting techniques from the adaptive selection paper
//! - **Clustering** — K-means task clustering by semantic similarity
//! - **Generator** — Dynamic prompt generation with BKS/PKS/SEAL integration
//! - **Learning** — Technique effectiveness tracking and BKS promotion
//! - **Temperature** — Adaptive temperature optimization per cluster
//!
//! ## RAG (from brainwires-rag)
//! - **RagClient** — Core semantic code search with hybrid BM25+vector search
//! - **Embedding** — FastEmbed (all-MiniLM-L6-v2) local embedding generation
//! - **Indexer** — File walking, AST-based chunking for 12 languages
//! - **Git Search** — Semantic search over commit history
//! - **Documents** — PDF, markdown, and plaintext document processing
//!
//! ## Spectral
//! - **SpectralReranker** — MSS-inspired log-det maximization for diverse retrieval
//! - **GraphOps** — Laplacian, Fiedler vector, spectral clustering, sparsification
//! - **Kernel** — Relevance-weighted kernel matrix construction
//! - **Linalg** — Cholesky decomposition and log-determinant computation
//!
//! ## Code Analysis
//! - **RepoMap** — AST-based symbol extraction (definitions, references)
//! - **Relations** — Call graph generation, definition/reference lookup
//! - **Storage** — LanceDB persistence for code relations

// Re-export core types
pub use brainwires_core;

// ── Knowledge (from brainwires-brain) ──────────────────────────────────────

/// Knowledge graph, entities, thoughts, BKS/PKS, brain client.
#[cfg(feature = "knowledge")]
pub mod knowledge;

// Prompting lives in `brainwires-prompting` — depend on that directly.

// ── RAG, spectral, code_analysis ──────────────────────────────────────────
// All three live in `brainwires-rag`. Spectral and code_analysis travel
// with RAG (no external consumers, only used by `rag::client::*`).
// Depend on `brainwires-rag` directly.

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

// Prompting / RAG / spectral / code-analysis live in `brainwires-prompting`
// and `brainwires-rag` — there are no re-exports here.

/// Prelude for convenient imports.
pub mod prelude {
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::brain_client::BrainClient;
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::entity::{Entity, EntityStore, EntityType};
    #[cfg(feature = "knowledge")]
    pub use super::knowledge::thought::{Thought, ThoughtCategory};
}
