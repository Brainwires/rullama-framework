//! Unified embedding abstraction
//!
//! Provides the `EmbeddingProvider` trait for pluggable text embedding backends.
//! Implementations live in downstream crates (storage, rag) — this trait enables
//! consumers to accept any embedding backend without coupling to a specific one.

use anyhow::Result;

/// Trait for text embedding generation.
///
/// Implementations should be thread-safe and reusable across concurrent contexts.
///
/// # Example
///
/// ```ignore
/// use rullama_core::EmbeddingProvider;
///
/// fn search(provider: &dyn EmbeddingProvider, query: &str) -> anyhow::Result<Vec<f32>> {
///     provider.embed(query)
/// }
/// ```
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    ///
    /// Default implementation calls `embed` in a loop. Backends that support
    /// native batching should override this for better performance.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get the dimensionality of the embedding vectors.
    fn dimension(&self) -> usize;

    /// Get the model name (e.g. "all-MiniLM-L6-v2").
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbedding;

    impl EmbeddingProvider for MockEmbedding {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.1, 0.2, 0.3])
        }

        fn dimension(&self) -> usize {
            3
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }
    }

    #[test]
    fn test_embed_single() {
        let provider = MockEmbedding;
        let embedding = provider.embed("test").unwrap();
        assert_eq!(embedding.len(), 3);
    }

    #[test]
    fn test_embed_batch_default() {
        let provider = MockEmbedding;
        let texts = vec!["a".to_string(), "b".to_string()];
        let embeddings = provider.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn test_dimension() {
        let provider = MockEmbedding;
        assert_eq!(provider.dimension(), 3);
    }

    #[test]
    fn test_model_name() {
        let provider = MockEmbedding;
        assert_eq!(provider.model_name(), "mock-model");
    }
}
