//! Vector store abstraction
//!
//! Provides the `VectorStore` trait for pluggable vector database backends.
//! Implementations live in downstream crates (storage with LanceDB, rag with
//! LanceDB/Qdrant) — this trait enables consumers to swap backends without
//! changing application code.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Result from a vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Unique identifier for the stored item.
    pub id: String,
    /// Similarity score (higher = more similar, typically 0.0–1.0).
    pub score: f32,
    /// The text content of the matched chunk.
    pub content: String,
    /// Arbitrary metadata associated with the item.
    pub metadata: serde_json::Value,
}

/// Trait for vector database operations.
///
/// Provides a backend-agnostic interface for storing and searching embeddings.
/// Implementations should handle connection management internally.
///
/// # Example
///
/// ```ignore
/// use rullama_core::{VectorStore, VectorSearchResult};
///
/// async fn search(store: &dyn VectorStore, query_vec: Vec<f32>) -> anyhow::Result<Vec<VectorSearchResult>> {
///     store.search(query_vec, 10, 0.7).await
/// }
/// ```
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Initialize the store (create tables/collections if needed).
    async fn initialize(&self, dimension: usize) -> Result<()>;

    /// Insert embeddings with associated content and metadata.
    ///
    /// Returns the number of items successfully stored.
    async fn upsert(
        &self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        contents: Vec<String>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<usize>;

    /// Search for similar vectors.
    ///
    /// Returns up to `limit` results with score >= `min_score`.
    async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<VectorSearchResult>>;

    /// Delete items by their IDs.
    async fn delete(&self, ids: Vec<String>) -> Result<usize>;

    /// Delete all stored data.
    async fn clear(&self) -> Result<()>;

    /// Get the number of stored items.
    async fn count(&self) -> Result<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify the types compile and serialize correctly
    #[test]
    fn test_search_result_serialization() {
        let result = VectorSearchResult {
            id: "chunk-1".to_string(),
            score: 0.95,
            content: "fn main() {}".to_string(),
            metadata: serde_json::json!({"file": "main.rs", "language": "rust"}),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: VectorSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "chunk-1");
        assert!((deserialized.score - 0.95).abs() < f32::EPSILON);
    }
}
