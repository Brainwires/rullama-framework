//! # RAG — Codebase Indexing and Semantic Search
//!
//! This module provides RAG (Retrieval-Augmented Generation) capabilities for semantic
//! code search across large projects. It combines vector embeddings with BM25 keyword
//! search to enable intelligent code retrieval.
//!
//! ## Architecture
//!
//! - `client::RagClient` — Core library API (indexing, querying, git search)
//! - `embedding` — FastEmbed (all-MiniLM-L6-v2) local embedding generation
//! - `indexer` — File walking, AST-based chunking for 12 languages
//! - `git` — Git history walking and commit chunking
//! - `cache` — Persistent hash cache for incremental updates
//! - `git_cache` — Git commit tracking cache
//! - `config` — Configuration management
//! - `types` — Request/response types with validation
//! - `error` — Domain-specific error types
//!
//! ## External Dependencies (from sibling crates/modules)
//!
//! - Vector database operations → `brainwires_storage::vector_db`
//! - BM25 keyword search → `brainwires_storage::bm25_search`
//! - Path utilities → `brainwires_storage::paths`
//! - Glob utilities → `brainwires_storage::glob_utils`
//! - Code analysis (definitions, references) → `crate::code_analysis`
//! - Spectral reranking → `crate::spectral`

// ── Always-available modules ────────────────────────────────────────────────

/// Domain-specific error types for the RAG system.
pub mod error;

/// Request/response types with validation.
pub mod types;

// ── Core RAG modules ────────────────────────────────────────────────────────

/// Persistent file hash cache for incremental updates.
pub mod cache;

/// Configuration management with environment variable support.
pub mod config;

/// Embedding generation using FastEmbed.
pub mod embedding;

/// Git history walking and commit chunking.
pub mod git;

/// Git commit tracking cache.
pub mod git_cache;

/// File walking, AST parsing, and code chunking.
pub mod indexer;

/// Core library client API (indexing, querying, search, git history).
pub mod client;

/// Document processing, chunking, and hybrid search.
#[cfg(feature = "documents")]
pub mod documents;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use client::RagClient;
pub use config::Config;
pub use error::RagError;
pub use types::{
    AdvancedSearchRequest, ClearRequest, ClearResponse, FindDefinitionRequest,
    FindReferencesRequest, GetCallGraphRequest, GitSearchResult, IndexRequest, IndexResponse,
    IndexingMode, LanguageStats, QueryRequest, QueryResponse, SearchGitHistoryRequest,
    SearchGitHistoryResponse, StatisticsRequest, StatisticsResponse,
};

pub use types::{FindDefinitionResponse, FindReferencesResponse, GetCallGraphResponse};
