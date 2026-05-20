//! Unified database traits for the Brainwires storage layer.
//!
//! Two traits define the database capabilities:
//!
//! - [`StorageBackend`](self::StorageBackend) — generic CRUD + vector search for domain stores
//!   (conversations, messages, tasks, plans, etc.)
//! - [`VectorDatabase`](self::VectorDatabase) — RAG-style embedding storage with hybrid search
//!   for the codebase indexing subsystem
//!
//! A single database struct (e.g. `LanceDatabase`, `PostgresDatabase`) can
//! implement **both** traits, sharing one connection pool.

use anyhow::Result;

use super::types::{FieldDef, Filter, Record, ScoredRecord};

// Re-export core types so consumers can use `databases::traits::*`.
pub use brainwires_core::{ChunkMetadata, DatabaseStats, SearchResult};

// ── StorageBackend ──────────────────────────────────────────────────────

/// Backend-agnostic storage operations.
///
/// Domain stores (e.g. `MessageStore` in `brainwires-stores`, etc.)
/// are generic over this trait so they can work with any supported database.
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    /// Ensure a table exists with the given schema.
    ///
    /// Implementations should be idempotent — calling this on an existing table
    /// is a no-op (or verifies compatibility).
    async fn ensure_table(&self, table_name: &str, schema: &[FieldDef]) -> Result<()>;

    /// Insert one or more records into a table.
    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()>;

    /// Query records matching an optional filter.
    ///
    /// Pass `None` for `filter` to return all rows (up to `limit`).
    async fn query(
        &self,
        table_name: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<Record>>;

    /// Delete records matching a filter.
    async fn delete(&self, table_name: &str, filter: &Filter) -> Result<()>;

    /// Count records matching an optional filter.
    async fn count(&self, table_name: &str, filter: Option<&Filter>) -> Result<usize> {
        // Default implementation: count via query.
        Ok(self.query(table_name, filter, None).await?.len())
    }

    /// Vector similarity search.
    ///
    /// Returns up to `limit` records ordered by descending similarity to `vector`.
    /// An optional `filter` narrows the candidates before ranking.
    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredRecord>>;
}

// ── VectorDatabase ──────────────────────────────────────────────────────

/// Trait for vector database operations used by the RAG subsystem.
///
/// Implementations handle connection management, BM25 keyword indexing, and
/// hybrid search fusion internally.
#[async_trait::async_trait]
pub trait VectorDatabase: Send + Sync {
    /// Initialize the database and create collections if needed.
    async fn initialize(&self, dimension: usize) -> Result<()>;

    /// Store embeddings with metadata.
    ///
    /// `root_path` is the normalized root of the indexed project — used for
    /// per-project BM25 isolation.
    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        root_path: &str,
    ) -> Result<usize>;

    /// Search for similar vectors.
    #[allow(clippy::too_many_arguments)]
    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
    ) -> Result<Vec<SearchResult>>;

    /// Search with additional filters (extensions, languages, path patterns).
    #[allow(clippy::too_many_arguments)]
    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>>;

    /// Delete embeddings for a specific file.
    async fn delete_by_file(&self, file_path: &str) -> Result<usize>;

    /// Clear all embeddings.
    async fn clear(&self) -> Result<()>;

    /// Get statistics about the stored data.
    async fn get_statistics(&self) -> Result<DatabaseStats>;

    /// Flush/save changes to disk.
    async fn flush(&self) -> Result<()>;

    /// Count embeddings for a specific root path.
    async fn count_by_root_path(&self, root_path: &str) -> Result<usize>;

    /// Get unique file paths indexed for a specific root path.
    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>>;

    /// Search and return results together with their embedding vectors.
    ///
    /// Used by the spectral diversity reranker which needs the raw embeddings
    /// to compute pairwise similarities. The default implementation delegates
    /// to [`search`](VectorDatabase::search) and returns empty embedding vectors.
    #[allow(clippy::too_many_arguments)]
    async fn search_with_embeddings(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
    ) -> Result<(Vec<SearchResult>, Vec<Vec<f32>>)> {
        let results = self
            .search(
                query_vector,
                query_text,
                limit,
                min_score,
                project,
                root_path,
                hybrid,
            )
            .await?;
        let empty_embeddings = vec![Vec::new(); results.len()];
        Ok((results, empty_embeddings))
    }
}
