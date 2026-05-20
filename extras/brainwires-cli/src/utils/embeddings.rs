//! Embeddings Manager
//!
//! Provides a simple async wrapper around the storage::embeddings::CachedEmbeddingProvider.
//! This module exists for API compatibility with code that expects an async interface.
//!
//! For direct synchronous access, use `crate::storage::embeddings::CachedEmbeddingProvider`.

use anyhow::Result;
use std::sync::Arc;

use crate::storage::embeddings::CachedEmbeddingProvider;

/// Async embeddings manager that wraps the synchronous CachedEmbeddingProvider
///
/// This provides an async API for contexts that need it, while delegating
/// to the underlying FastEmbed-based implementation with LRU caching.
pub struct EmbeddingsManager {
    provider: Arc<CachedEmbeddingProvider>,
}

impl EmbeddingsManager {
    /// Create a new embeddings manager
    ///
    /// Initializes the FastEmbed model (all-MiniLM-L6-v2, 384 dimensions).
    /// Returns an error if the model fails to load.
    pub fn new() -> Result<Self> {
        let provider = CachedEmbeddingProvider::new()?;
        Ok(Self {
            provider: Arc::new(provider),
        })
    }

    /// Create an embeddings manager from an existing provider
    pub fn from_provider(provider: Arc<CachedEmbeddingProvider>) -> Self {
        Self { provider }
    }

    /// Generate an embedding for text (async wrapper)
    ///
    /// Uses the cached embedding if available, otherwise computes it.
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // The underlying embed_cached is synchronous but fast (uses LRU cache)
        // We wrap it in an async function for API compatibility
        self.provider.embed_cached(text)
    }

    /// Generate embeddings for multiple texts (async wrapper)
    pub async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.provider.embed_batch(texts)
    }

    /// Get the embedding dimension (384 for all-MiniLM-L6-v2)
    pub fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    /// Get the number of cached embeddings
    pub fn cache_len(&self) -> usize {
        self.provider.cache_len()
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        self.provider.clear_cache();
    }

    /// Get the underlying provider for direct access
    pub fn provider(&self) -> &Arc<CachedEmbeddingProvider> {
        &self.provider
    }
}

impl Clone for EmbeddingsManager {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires FastEmbed model files on disk — run manually after downloading the model"]
    fn test_embeddings_manager_new() {
        let manager = EmbeddingsManager::new().unwrap();
        assert_eq!(manager.dimension(), 384);
    }

    #[tokio::test]
    #[ignore = "requires FastEmbed model files on disk — run manually after downloading the model"]
    async fn test_generate_embedding() {
        let manager = EmbeddingsManager::new().unwrap();
        let result = manager.generate_embedding("test text").await;
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 dimension
    }

    #[tokio::test]
    #[ignore = "requires FastEmbed model files on disk — run manually after downloading the model"]
    async fn test_generate_embeddings_batch() {
        let manager = EmbeddingsManager::new().unwrap();
        let texts = vec!["first message".to_string(), "second message".to_string()];
        let embeddings = manager.generate_embeddings(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[tokio::test]
    #[ignore = "requires FastEmbed model files on disk — run manually after downloading the model"]
    async fn test_caching() {
        let manager = EmbeddingsManager::new().unwrap();

        // First call caches
        let _emb1 = manager.generate_embedding("cached query").await.unwrap();
        assert_eq!(manager.cache_len(), 1);

        // Second call hits cache
        let _emb2 = manager.generate_embedding("cached query").await.unwrap();
        assert_eq!(manager.cache_len(), 1);

        // Different query adds to cache
        let _emb3 = manager.generate_embedding("different query").await.unwrap();
        assert_eq!(manager.cache_len(), 2);

        // Clear cache
        manager.clear_cache();
        assert_eq!(manager.cache_len(), 0);
    }

    #[test]
    #[ignore = "requires FastEmbed model files on disk — run manually after downloading the model"]
    fn test_clone() {
        let manager = EmbeddingsManager::new().unwrap();
        let cloned = manager.clone();
        assert_eq!(manager.dimension(), cloned.dimension());
    }
}
