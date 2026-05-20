//! ToolEmbedding - Semantic tool discovery via embedding similarity
//!
//! Embeds tool names and descriptions into vectors, then uses cosine
//! similarity to find semantically relevant tools for a given query.

use anyhow::{Context, Result};
use brainwires_rag::rag::embedding::FastEmbedManager;
use std::sync::{Arc, OnceLock};

static EMBED_MANAGER: OnceLock<Arc<FastEmbedManager>> = OnceLock::new();

/// Get or initialize the shared FastEmbedManager.
fn get_embed_manager() -> Result<&'static Arc<FastEmbedManager>> {
    EMBED_MANAGER.get().ok_or(()).or_else(|_| {
        let manager = Arc::new(FastEmbedManager::new()?);
        // Another thread may have initialized it between our check and here;
        // that's fine — just use whichever won the race.
        let _ = EMBED_MANAGER.set(manager.clone());
        Ok::<_, anyhow::Error>(EMBED_MANAGER.get().unwrap())
    })
}

/// Pre-computed embedding index for semantic tool discovery.
///
/// Stores embeddings of tool `"{name}: {description}"` strings and supports
/// cosine-similarity search against user queries.
pub struct ToolEmbeddingIndex {
    /// (tool_name, embedding_vector) pairs
    entries: Vec<ToolEmbeddingEntry>,
    /// Number of tools when the index was built (for staleness detection)
    tool_count: usize,
}

struct ToolEmbeddingEntry {
    name: String,
    embedding: Vec<f32>,
}

impl ToolEmbeddingIndex {
    /// Build an index from tool (name, description) pairs.
    ///
    /// Each tool is embedded as `"{name}: {description}"`.
    /// Returns an empty index if no tools are provided.
    pub fn build(tools: &[(String, String)]) -> Result<Self> {
        if tools.is_empty() {
            return Ok(Self {
                entries: vec![],
                tool_count: 0,
            });
        }

        let manager = get_embed_manager().context("Failed to initialize embedding model")?;

        // Build text representations for embedding
        let texts: Vec<String> = tools
            .iter()
            .map(|(name, desc)| format!("{}: {}", name, desc))
            .collect();

        let embeddings = manager
            .embed_batch(&texts)
            .context("Failed to generate tool embeddings")?;

        let entries = tools
            .iter()
            .zip(embeddings)
            .map(|((name, _), embedding)| ToolEmbeddingEntry {
                name: name.clone(),
                embedding,
            })
            .collect();

        Ok(Self {
            entries,
            tool_count: tools.len(),
        })
    }

    /// Search for tools semantically similar to the query.
    ///
    /// Returns `(tool_name, similarity_score)` pairs sorted by score descending,
    /// filtered by `min_score` and capped at `limit`.
    pub fn search(&self, query: &str, limit: usize, min_score: f32) -> Result<Vec<(String, f32)>> {
        if self.entries.is_empty() {
            return Ok(vec![]);
        }

        let manager = get_embed_manager().context("Failed to get embedding model")?;
        let query_vec = manager.embed(query).context("Failed to embed query")?;
        let query_vec = &query_vec;

        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_vec, &entry.embedding);
                (entry.name.clone(), score)
            })
            .filter(|(_, score)| *score >= min_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// Number of tools in the index (for staleness detection).
    pub fn tool_count(&self) -> usize {
        self.tool_count
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have equal dimensions");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tools() -> Vec<(String, String)> {
        vec![
            (
                "read_file".to_string(),
                "Read the contents of a file from disk".to_string(),
            ),
            (
                "write_file".to_string(),
                "Write content to a file on disk".to_string(),
            ),
            (
                "execute_command".to_string(),
                "Execute a shell command in bash".to_string(),
            ),
            (
                "git_commit".to_string(),
                "Create a git commit with a message".to_string(),
            ),
            (
                "optimize_png".to_string(),
                "Optimize and compress PNG image files".to_string(),
            ),
        ]
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_build_empty() {
        let index = ToolEmbeddingIndex::build(&[]).unwrap();
        assert_eq!(index.tool_count(), 0);
        let results = index.search("anything", 10, 0.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_build_and_search() {
        let tools = sample_tools();
        let index = ToolEmbeddingIndex::build(&tools).unwrap();
        assert_eq!(index.tool_count(), 5);

        // "compress image" should find "optimize_png" (described as "Optimize and compress PNG image files")
        let results = index.search("compress image", 5, 0.0).unwrap();
        assert!(!results.is_empty());
        // The top result should be optimize_png
        assert_eq!(results[0].0, "optimize_png");
    }

    #[test]
    fn test_search_file_operations() {
        let tools = sample_tools();
        let index = ToolEmbeddingIndex::build(&tools).unwrap();

        // "load a document" should find file reading tools
        let results = index.search("load a document", 3, 0.0).unwrap();
        assert!(!results.is_empty());
        // read_file or write_file should be in top results
        let top_names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            top_names.contains(&"read_file") || top_names.contains(&"write_file"),
            "Expected file tools in results, got: {:?}",
            top_names
        );
    }

    #[test]
    fn test_min_score_filtering() {
        let tools = sample_tools();
        let index = ToolEmbeddingIndex::build(&tools).unwrap();

        // With a very high min_score, most results should be filtered out
        let results = index
            .search("random unrelated query xyz", 10, 0.95)
            .unwrap();
        // Very unlikely anything scores above 0.95 for an unrelated query
        assert!(
            results.len() <= 1,
            "Expected few/no results with high min_score, got {}",
            results.len()
        );
    }

    #[test]
    fn test_limit_respected() {
        let tools = sample_tools();
        let index = ToolEmbeddingIndex::build(&tools).unwrap();

        let results = index.search("file", 2, 0.0).unwrap();
        assert!(results.len() <= 2);
    }
}
