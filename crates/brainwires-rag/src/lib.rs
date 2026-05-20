#![deny(missing_docs)]
//! `brainwires-rag` — codebase indexing + hybrid retrieval (vector + BM25)
//! for the Brainwires Agent Framework.
//!
//! Standalone so its dep weight (lancedb, tantivy, git2, tree-sitter +
//! 12 grammars, rmcp, rayon, …) doesn't fall on consumers that only want
//! `brainwires-knowledge` (BKS/PKS/brain client) or
//! `brainwires-prompting`.
//!
//! ## Public surface
//!
//! - [`rag`] — `RagClient`, indexer, hybrid query, embedding wiring,
//!   document chunking, Git history search.
//!
//! Two **internal** subdomains travel with this crate:
//!
//! - [`spectral`] — log-determinant diversity reranking + cross-encoder
//!   reranking. Pure ndarray math; only consumed by `rag::client::reranking`.
//! - [`code_analysis`] — AST-based symbol/definition/reference graphs;
//!   only consumed by `rag::client::code_analysis`.
//!
//! Both are kept in this crate (rather than separate crates) because they
//! have no external consumers and their public API would otherwise need
//! to be invented just for the split.

pub mod code_analysis;
pub mod rag;
pub mod spectral;

// ── Top-level re-exports — match the surface previously exposed at
// `brainwires_knowledge::{RagClient, IndexRequest, ...}` so existing
// consumer imports keep working with a one-line crate-name swap.

pub use rag::client::RagClient;
pub use rag::config::Config;
pub use rag::error::RagError;
pub use rag::types::{
    AdvancedSearchRequest, ClearRequest, ClearResponse, EnsembleRequest, EnsembleResponse,
    FindDefinitionRequest, FindReferencesRequest, GetCallGraphRequest, GitSearchResult,
    IndexRequest, IndexResponse, IndexingMode, LanguageStats, QueryRequest, QueryResponse,
    SearchGitHistoryRequest, SearchGitHistoryResponse, SearchStrategy, StatisticsRequest,
    StatisticsResponse,
};
pub use rag::types::{FindDefinitionResponse, FindReferencesResponse, GetCallGraphResponse};

pub use spectral::{
    CrossEncoderConfig, CrossEncoderReranker, DiversityReranker, RerankerKind, SpectralReranker,
    SpectralSelectConfig,
};

pub use code_analysis::types::{
    CallEdge, CallGraphNode, Definition, Reference, ReferenceKind, SymbolId, SymbolKind, Visibility,
};
