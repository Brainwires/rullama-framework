//! Core library client for brainwires-rag
//!
//! This module provides the main client interface for using brainwires-rag
//! as a library in your own Rust applications.
//!
//! # Structure
//!
//! The implementation is split across focused submodules:
//!
//! - `constructor` — constructors, basic utilities, path helpers
//! - `locking` — two-layer index locking (filesystem + in-process broadcast)
//! - `search` — indexing dispatch, semantic search, filtered search, statistics, clear, git history
//! - `ensemble` — multi-strategy ensemble query with Reciprocal Rank Fusion
//! - `reranking` — pluggable diversity/relevance reranking (`spectral` feature)
//! - `code_analysis` — find definition, find references, call graph (`code-analysis` feature)

use crate::code_analysis::HybridRelationsProvider;
use crate::rag::cache::HashCache;
use crate::rag::config::Config;
use crate::rag::embedding::FastEmbedManager;
use crate::rag::git_cache::GitCache;
use crate::rag::indexer::CodeChunker;
use brainwires_storage::databases::VectorDatabase;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// Filesystem locking for cross-process coordination
mod fs_lock;
pub(crate) use fs_lock::FsLockGuard;

// Index locking mechanism (uses fs_lock for cross-process, broadcast for in-process)
mod index_lock;
pub(crate) use index_lock::{IndexLockGuard, IndexLockResult, IndexingOperation};

// ── impl blocks, split by concern ────────────────────────────────────────────

/// Constructor methods and utility helpers.
mod constructor;

/// Two-layer index locking.
mod locking;

/// Indexing dispatch, search, statistics, clear, and git-history search.
mod search;

/// Multi-strategy ensemble query with RRF fusion.
mod ensemble;

/// Pluggable diversity/relevance reranking (requires `spectral`).
mod reranking;

/// Code-navigation: find definition, find references, call graph (requires `code-analysis`).
mod code_analysis;

// ── existing public submodules ────────────────────────────────────────────────

/// Indexing operations (public for MCP server binary).
pub mod indexing;

// Git indexing operations module
pub(crate) mod git_indexing;

#[cfg(test)]
mod tests;

// ── Re-exports (keeps all existing call sites working) ────────────────────────

// All public types from `types` are already glob-imported above via `use crate::rag::types::*`.
// The public API surface of RagClient itself is fully re-exported through the struct below.

/// Main client for interacting with the RAG system
///
/// This client provides a high-level API for indexing codebases and performing
/// semantic searches. It contains all the core functionality and can be used
/// directly as a library or wrapped by the MCP server.
///
/// # Example
///
/// ```ignore
/// use crate::rag::{RagClient, IndexRequest, QueryRequest};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Create client with default configuration
///     let client = RagClient::new().await?;
///
///     // Index a codebase
///     let index_req = IndexRequest {
///         path: "/path/to/code".to_string(),
///         project: Some("my-project".to_string()),
///         include_patterns: vec!["**/*.rs".to_string()],
///         exclude_patterns: vec!["**/target/**".to_string()],
///         max_file_size: 1_048_576,
///     };
///     let response = client.index_codebase(index_req).await?;
///     println!("Indexed {} files", response.files_indexed);
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct RagClient {
    pub(crate) embedding_provider: Arc<FastEmbedManager>,
    pub(crate) vector_db: Arc<dyn VectorDatabase>,
    pub(crate) chunker: Arc<CodeChunker>,
    // Persistent hash cache for incremental updates
    pub(crate) hash_cache: Arc<RwLock<HashCache>>,
    pub(crate) cache_path: PathBuf,
    // Git cache for git history indexing
    pub(crate) git_cache: Arc<RwLock<GitCache>>,
    pub(crate) git_cache_path: PathBuf,
    // Configuration (for accessing batch sizes, timeouts, etc.)
    pub(crate) config: Arc<Config>,
    // In-progress indexing operations (prevents concurrent indexing and allows result sharing)
    pub(crate) indexing_ops: Arc<RwLock<HashMap<String, IndexingOperation>>>,
    // Relations provider for code navigation (find definition, references, call graph)
    pub(crate) relations_provider: Arc<HybridRelationsProvider>,
}
