//! Document BM25 Search
//!
//! Provides per-scope isolation (conversation/project) and document-aware
//! BM25 keyword search using Tantivy.

use anyhow::{Context, Result};
use brainwires_storage::bm25_search::BM25Search;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Result from document BM25 search
#[derive(Debug, Clone)]
pub struct DocumentBM25Result {
    /// Chunk ID (document_id:chunk_index format)
    pub chunk_id: String,
    /// BM25 score
    pub score: f32,
}

/// Document BM25 search manager
///
/// Manages multiple BM25 indices, one per scope (conversation or project).
/// Each scope gets its own isolated Tantivy index for document chunks.
pub struct DocumentBM25Manager {
    /// Base path for BM25 indices
    base_path: PathBuf,
    /// Cached BM25 instances per scope
    indices: RwLock<HashMap<String, Arc<BM25Search>>>,
}

impl DocumentBM25Manager {
    /// Create a new document BM25 manager
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            indices: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a BM25 index for a scope
    fn get_index(&self, scope: &str) -> Result<Arc<BM25Search>> {
        // Check cache first
        {
            let indices = self
                .indices
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to read index cache: {}", e))?;
            if let Some(index) = indices.get(scope) {
                return Ok(Arc::clone(index));
            }
        }

        // Create new index
        let scope_hash = Self::hash_scope(scope);
        let index_path = self
            .base_path
            .join(format!("doc_bm25_{}", &scope_hash[..16]));

        let index = BM25Search::new(&index_path)
            .with_context(|| format!("Failed to create BM25 index for scope: {}", scope))?;

        let index = Arc::new(index);

        // Cache it
        {
            let mut indices = self
                .indices
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to write index cache: {}", e))?;
            indices.insert(scope.to_string(), Arc::clone(&index));
        }

        Ok(index)
    }

    /// Hash a scope ID for filesystem-safe path
    fn hash_scope(scope: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(scope.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Index document chunks for a scope
    ///
    /// # Arguments
    /// * `scope` - Scope identifier (conversation_id or project_id)
    /// * `chunks` - Vec of (chunk_id, content) tuples
    pub fn index_chunks(&self, scope: &str, chunks: Vec<(String, String)>) -> Result<()> {
        let index = self.get_index(scope)?;

        // Convert chunk_id strings to u64 by hashing
        let documents: Vec<(u64, String, String, String)> = chunks
            .into_iter()
            .map(|(chunk_id, content)| {
                let id_hash = Self::hash_chunk_id(&chunk_id);
                // Use chunk_id as both string_id and file_path for document-level BM25
                (id_hash, chunk_id.clone(), content, chunk_id)
            })
            .collect();

        index
            .add_documents(documents)
            .context("Failed to add document chunks to BM25 index")?;

        Ok(())
    }

    /// Search document chunks in a scope
    ///
    /// # Arguments
    /// * `scope` - Scope identifier
    /// * `query` - Search query
    /// * `limit` - Maximum results
    pub fn search(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DocumentBM25Result>> {
        let index = self.get_index(scope)?;

        let results = index
            .search(query, limit)
            .context("Failed to search BM25 index")?;

        Ok(results
            .into_iter()
            .map(|r| DocumentBM25Result {
                chunk_id: format!("bm25:{}", r.id),
                score: r.score,
            })
            .collect())
    }

    /// Search with chunk ID mapping
    ///
    /// # Arguments
    /// * `scope` - Scope identifier
    /// * `query` - Search query
    /// * `limit` - Maximum results
    /// * `chunk_id_map` - Map of hash -> original chunk_id
    pub fn search_with_mapping(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
        chunk_id_map: &HashMap<u64, String>,
    ) -> Result<Vec<DocumentBM25Result>> {
        let index = self.get_index(scope)?;

        let results = index
            .search(query, limit)
            .context("Failed to search BM25 index")?;

        Ok(results
            .into_iter()
            .filter_map(|r| {
                chunk_id_map.get(&r.id).map(|chunk_id| DocumentBM25Result {
                    chunk_id: chunk_id.clone(),
                    score: r.score,
                })
            })
            .collect())
    }

    /// Delete document chunks from a scope
    pub fn delete_chunks(&self, scope: &str, chunk_ids: &[String]) -> Result<()> {
        let index = self.get_index(scope)?;

        for chunk_id in chunk_ids {
            let id_hash = Self::hash_chunk_id(chunk_id);
            index
                .delete_by_id(id_hash)
                .with_context(|| format!("Failed to delete chunk: {}", chunk_id))?;
        }

        Ok(())
    }

    /// Delete all chunks for a document in a scope
    pub fn delete_document(&self, scope: &str, document_id: &str) -> Result<()> {
        let index = self.get_index(scope)?;

        index
            .delete_by_file_path(document_id)
            .context("Failed to delete document from BM25 index")?;

        Ok(())
    }

    /// Clear all chunks for a scope
    pub fn clear_scope(&self, scope: &str) -> Result<()> {
        let index = self.get_index(scope)?;
        index.clear().context("Failed to clear BM25 index")?;
        Ok(())
    }

    /// Hash a chunk_id to u64
    fn hash_chunk_id(chunk_id: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        chunk_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Get statistics for a scope
    pub fn get_stats(&self, scope: &str) -> Result<DocumentBM25Stats> {
        let index = self.get_index(scope)?;
        let stats = index.get_stats()?;
        Ok(DocumentBM25Stats {
            total_chunks: stats.total_documents,
        })
    }
}

/// Statistics about a document BM25 index
#[derive(Debug, Clone)]
pub struct DocumentBM25Stats {
    /// Total number of indexed document chunks.
    pub total_chunks: usize,
}

/// Perform RRF fusion between vector and BM25 document results
///
/// Uses the generic RRF implementation from the bm25_search module.
pub fn document_rrf_fusion(
    vector_results: Vec<(String, f32)>,
    bm25_results: Vec<DocumentBM25Result>,
    limit: usize,
) -> Vec<(String, f32)> {
    use brainwires_storage::bm25_search::reciprocal_rank_fusion_generic;

    let bm25_tuples: Vec<(String, f32)> = bm25_results
        .into_iter()
        .map(|r| (r.chunk_id, r.score))
        .collect();

    reciprocal_rank_fusion_generic([vector_results, bm25_tuples], limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_hash_chunk_id_deterministic() {
        let id1 = DocumentBM25Manager::hash_chunk_id("doc-123:0");
        let id2 = DocumentBM25Manager::hash_chunk_id("doc-123:0");
        let id3 = DocumentBM25Manager::hash_chunk_id("doc-123:1");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_hash_scope() {
        let hash1 = DocumentBM25Manager::hash_scope("conv-123");
        let hash2 = DocumentBM25Manager::hash_scope("conv-123");
        let hash3 = DocumentBM25Manager::hash_scope("conv-456");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA256 hex
    }

    #[test]
    fn test_document_rrf_fusion() {
        let vector_results = vec![
            ("chunk-a".to_string(), 0.9),
            ("chunk-b".to_string(), 0.8),
            ("chunk-c".to_string(), 0.7),
        ];

        let bm25_results = vec![
            DocumentBM25Result {
                chunk_id: "chunk-b".to_string(),
                score: 5.0,
            },
            DocumentBM25Result {
                chunk_id: "chunk-d".to_string(),
                score: 4.0,
            },
            DocumentBM25Result {
                chunk_id: "chunk-a".to_string(),
                score: 3.0,
            },
        ];

        let fused = document_rrf_fusion(vector_results, bm25_results, 5);

        assert!(fused.len() >= 2);

        let chunk_a_score = fused
            .iter()
            .find(|(id, _)| id == "chunk-a")
            .map(|(_, s)| *s);
        let chunk_d_score = fused
            .iter()
            .find(|(id, _)| id == "chunk-d")
            .map(|(_, s)| *s);

        if let (Some(a_score), Some(d_score)) = (chunk_a_score, chunk_d_score) {
            assert!(a_score > d_score);
        }
    }

    #[test]
    fn test_manager_creation() {
        let temp = TempDir::new().unwrap();
        let manager = DocumentBM25Manager::new(temp.path());
        // Smoke test: simply ensure construction succeeded; base_path may or may not exist yet.
        let _ = &manager.base_path;
    }

    #[test]
    fn test_index_and_search() {
        let temp = TempDir::new().unwrap();
        let manager = DocumentBM25Manager::new(temp.path());

        let scope = "test-conversation";
        let chunks = vec![
            (
                "doc-1:0".to_string(),
                "Hello world, this is a test document.".to_string(),
            ),
            (
                "doc-1:1".to_string(),
                "Another chunk with different content about programming.".to_string(),
            ),
        ];

        manager.index_chunks(scope, chunks).unwrap();

        let results = manager.search(scope, "programming", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_clear_scope() {
        let temp = TempDir::new().unwrap();
        let manager = DocumentBM25Manager::new(temp.path());

        let scope = "test-scope";
        let chunks = vec![("doc-1:0".to_string(), "Test content".to_string())];

        manager.index_chunks(scope, chunks).unwrap();

        let stats_before = manager.get_stats(scope).unwrap();
        assert!(stats_before.total_chunks > 0);

        manager.clear_scope(scope).unwrap();

        let stats_after = manager.get_stats(scope).unwrap();
        assert_eq!(stats_after.total_chunks, 0);
    }
}
