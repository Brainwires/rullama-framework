//! Embedding Provider
//!
//! Provides text embeddings using FastEmbed with LRU caching.
//!
//! This module is the canonical owner of embedding infrastructure in the framework:
//!
//! - **FastEmbedManager** - Low-level wrapper around the fastembed crate (ONNX model)
//! - **CachedEmbeddingProvider** - LRU-cached wrapper that reduces latency for repeated queries
//!
//! Both implement the `brainwires_core::EmbeddingProvider` trait.

use anyhow::{Context, Result};
pub use brainwires_core::EmbeddingProvider;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

/// Default cache size for embeddings (1000 entries)
const DEFAULT_CACHE_SIZE: usize = 1000;
const EMBEDDING_DIM_MINILM: usize = 384;
const EMBEDDING_DIM_BGE_BASE: usize = 768;

// ── FastEmbedManager ────────────────────────────────────────────────────────

/// FastEmbed-based embedding provider using ONNX models.
///
/// The underlying ONNX model is loaded **lazily** on the first `embed` /
/// `embed_batch` call — construction is cheap and does not touch the network.
/// Once loaded, the model is cached for the lifetime of the manager.
///
/// Uses RwLock for safe interior mutability since fastembed's `embed()` requires `&mut self`.
/// Default model is all-MiniLM-L6-v2 (384 dimensions).
pub struct FastEmbedManager {
    model: OnceLock<RwLock<TextEmbedding>>,
    model_enum: EmbeddingModel,
    cache_dir: PathBuf,
    dimension: usize,
    model_name: String,
}

impl FastEmbedManager {
    /// Create a new FastEmbedManager with the default model (all-MiniLM-L6-v2).
    ///
    /// This is cheap — the ONNX model is not loaded until the first embed call.
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create a new FastEmbedManager from a model name string.
    ///
    /// This is cheap — the ONNX model is not loaded until the first embed call.
    pub fn from_model_name(model_name: &str) -> Result<Self> {
        let model = match model_name {
            "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
            "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            _ => {
                tracing::warn!(
                    "Unknown model '{}', falling back to all-MiniLM-L6-v2",
                    model_name
                );
                EmbeddingModel::AllMiniLML6V2
            }
        };
        Self::with_model(model)
    }

    /// Create a new FastEmbedManager with a specific model.
    ///
    /// This is cheap — the ONNX model is not loaded until the first embed call.
    pub fn with_model(model: EmbeddingModel) -> Result<Self> {
        let (dimension, name) = match model {
            EmbeddingModel::AllMiniLML6V2 => (EMBEDDING_DIM_MINILM, "all-MiniLM-L6-v2"),
            EmbeddingModel::AllMiniLML12V2 => (EMBEDDING_DIM_MINILM, "all-MiniLM-L12-v2"),
            EmbeddingModel::BGEBaseENV15 => (EMBEDDING_DIM_BGE_BASE, "BAAI/bge-base-en-v1.5"),
            EmbeddingModel::BGESmallENV15 => (EMBEDDING_DIM_MINILM, "BAAI/bge-small-en-v1.5"),
            _ => (EMBEDDING_DIM_MINILM, "all-MiniLM-L6-v2"),
        };

        let cache_dir = brainwires_core::paths::PlatformPaths::default_fastembed_cache_path();
        let _ = std::fs::create_dir_all(&cache_dir);

        Ok(Self {
            model: OnceLock::new(),
            model_enum: model,
            cache_dir,
            dimension,
            model_name: name.to_string(),
        })
    }

    /// Lazily initialize the ONNX model on first use.
    fn get_model(&self) -> Result<&RwLock<TextEmbedding>> {
        if let Some(m) = self.model.get() {
            return Ok(m);
        }
        tracing::info!("Initializing FastEmbed model: {:?}", self.model_enum);
        let mut options = InitOptions::default();
        options.model_name = self.model_enum.clone();
        options.show_download_progress = true;
        options.cache_dir = self.cache_dir.clone();
        let embedding_model =
            TextEmbedding::try_new(options).context("Failed to initialize FastEmbed model")?;
        // If another thread won the init race, use the winner's value.
        let _ = self.model.set(RwLock::new(embedding_model));
        Ok(self.model.get().expect("model just set"))
    }

    /// Generate embeddings for a batch of texts (raw, no caching).
    ///
    /// This is the low-level batch method. Prefer using `CachedEmbeddingProvider`
    /// for repeated queries.
    pub fn embed_batch_vec(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!("Generating embeddings for {} texts", texts.len());

        let model = self.get_model()?;
        let mut model = model.write().unwrap_or_else(|poisoned| {
            tracing::warn!("FastEmbed model lock was poisoned, recovering...");
            poisoned.into_inner()
        });

        let embeddings = model
            .embed(texts, None)
            .context("Failed to generate embeddings")?;

        Ok(embeddings)
    }

    /// Generate an embedding for a single text (inherent method for convenience).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_vec(vec![text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }

    /// Generate embeddings for a batch of texts (inherent method for convenience).
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_vec(texts.to_vec())
    }

    /// Get the dimensionality of the embedding vectors.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl EmbeddingProvider for FastEmbedManager {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch_vec(vec![text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_vec(texts.to_vec())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl Default for FastEmbedManager {
    fn default() -> Self {
        Self::new().expect("Failed to initialize default FastEmbed model")
    }
}

// ── CachedEmbeddingProvider ─────────────────────────────────────────────────

/// LRU-cached embedding provider for generating text embeddings.
///
/// Wraps `FastEmbedManager` and adds an LRU cache for memoizing query embeddings
/// to reduce latency in agent loops that often repeat similar queries.
pub struct CachedEmbeddingProvider {
    inner: Arc<FastEmbedManager>,
    cache: RwLock<LruCache<u64, Vec<f32>>>,
}

impl CachedEmbeddingProvider {
    /// Create a new cached embedding provider with the default model
    pub fn new() -> Result<Self> {
        let inner = FastEmbedManager::new().context("Failed to create embedding provider")?;

        Ok(Self {
            inner: Arc::new(inner),
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("DEFAULT_CACHE_SIZE is non-zero"),
            )),
        })
    }

    /// Create a cached wrapper around an existing FastEmbedManager
    pub fn with_manager(manager: Arc<FastEmbedManager>) -> Self {
        Self {
            inner: manager,
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("DEFAULT_CACHE_SIZE is non-zero"),
            )),
        }
    }

    /// Hash text to a cache key
    fn hash_text(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate an embedding with caching
    ///
    /// Checks the LRU cache first; if not found, generates the embedding
    /// and stores it in the cache.
    pub fn embed_cached(&self, text: &str) -> Result<Vec<f32>> {
        let cache_key = Self::hash_text(text);

        // Check cache first (read lock)
        if let Ok(cache) = self.cache.read()
            && let Some(embedding) = cache.peek(&cache_key)
        {
            return Ok(embedding.clone());
        }

        // Generate embedding
        let embedding = self.inner.embed(text)?;

        // Store in cache (write lock)
        if let Ok(mut cache) = self.cache.write() {
            cache.put(cache_key, embedding.clone());
        }

        Ok(embedding)
    }

    /// Get the number of cached embeddings
    pub fn cache_len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get a reference to the underlying FastEmbedManager
    pub fn inner(&self) -> &Arc<FastEmbedManager> {
        &self.inner
    }

    /// Generate an embedding for a single text (inherent method for convenience).
    ///
    /// This delegates to the `EmbeddingProvider` trait implementation, making
    /// the method available without requiring the trait to be in scope.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_cached(text)
    }

    /// Generate embeddings for a batch of texts (inherent method for convenience).
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch_vec(texts.to_vec())
    }

    /// Get the dimensionality of the embedding vectors (inherent method for convenience).
    pub fn dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Get the model name (inherent method for convenience).
    pub fn model_name(&self) -> &str {
        &self.inner.model_name
    }
}

impl EmbeddingProvider for CachedEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_cached(text)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}

impl Clone for CachedEmbeddingProvider {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("DEFAULT_CACHE_SIZE is non-zero"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FastEmbedManager tests ──────────────────────────────────────────

    #[test]
    fn test_fastembed_creation() {
        let manager = FastEmbedManager::new().unwrap();
        assert_eq!(manager.dimension(), 384);
        assert_eq!(manager.model_name(), "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_fastembed_embed_single() {
        let manager = FastEmbedManager::new().unwrap();
        let embedding = manager.embed("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_fastembed_embed_batch() {
        let manager = FastEmbedManager::new().unwrap();
        let texts = vec![
            "fn main() { println!(\"Hello, world!\"); }".to_string(),
            "pub struct Vector { x: f32, y: f32 }".to_string(),
        ];

        let embeddings = manager.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[test]
    fn test_fastembed_empty_batch() {
        let manager = FastEmbedManager::new().unwrap();
        let embeddings = manager.embed_batch_vec(vec![]).unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    fn test_fastembed_default() {
        let manager = FastEmbedManager::default();
        assert_eq!(manager.dimension(), 384);
    }

    #[test]
    fn test_fastembed_from_model_name() {
        let manager = FastEmbedManager::from_model_name("all-MiniLM-L6-v2").unwrap();
        assert_eq!(manager.dimension(), 384);
    }

    #[test]
    fn test_fastembed_unknown_model_fallback() {
        let manager = FastEmbedManager::from_model_name("unknown-model").unwrap();
        assert_eq!(manager.dimension(), 384);
        assert_eq!(manager.model_name(), "all-MiniLM-L6-v2");
    }

    // ── CachedEmbeddingProvider tests ───────────────────────────────────

    #[test]
    fn test_cached_provider_creation() {
        let provider = CachedEmbeddingProvider::new().unwrap();
        assert_eq!(provider.dimension(), 384);
    }

    #[test]
    fn test_cached_provider_embed_single() {
        let provider = CachedEmbeddingProvider::new().unwrap();
        let embedding = provider.embed("Hello, world!").unwrap();

        assert_eq!(embedding.len(), 384);

        // Verify it's normalized (approximately)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cached_provider_embed_batch() {
        let provider = CachedEmbeddingProvider::new().unwrap();
        let texts = vec![
            "First message".to_string(),
            "Second message".to_string(),
            "Third message".to_string(),
        ];

        let embeddings = provider.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
        assert_eq!(embeddings[2].len(), 384);
    }

    #[test]
    fn test_cached_provider_clone() {
        let provider = CachedEmbeddingProvider::new().unwrap();
        let cloned = provider.clone();

        assert_eq!(provider.dimension(), cloned.dimension());
    }

    #[test]
    fn test_cached_provider_caching() {
        let provider = CachedEmbeddingProvider::new().unwrap();

        // First call should compute and cache
        let embedding1 = provider.embed_cached("test query").unwrap();
        assert_eq!(provider.cache_len(), 1);

        // Second call should return cached value
        let embedding2 = provider.embed_cached("test query").unwrap();
        assert_eq!(provider.cache_len(), 1); // Still 1, not 2

        // Embeddings should be identical
        assert_eq!(embedding1, embedding2);

        // Different query should add to cache
        let _embedding3 = provider.embed_cached("different query").unwrap();
        assert_eq!(provider.cache_len(), 2);

        // Clear cache
        provider.clear_cache();
        assert_eq!(provider.cache_len(), 0);
    }
}
