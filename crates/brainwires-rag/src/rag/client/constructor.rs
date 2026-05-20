//! Constructor methods and basic utility methods for [`RagClient`].

use super::RagClient;
use crate::code_analysis::HybridRelationsProvider;
use crate::rag::cache::HashCache;
use crate::rag::config::Config;
use crate::rag::embedding::FastEmbedManager;
use crate::rag::git_cache::GitCache;
use crate::rag::indexer::CodeChunker;
use crate::rag::indexer::FileInfo;
use crate::rag::indexer::detect_language;
use brainwires_storage::databases::VectorDatabase;

#[cfg(feature = "qdrant-backend")]
use brainwires_storage::databases::QdrantDatabase;

#[cfg(not(feature = "qdrant-backend"))]
use brainwires_storage::databases::LanceDatabase;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

impl RagClient {
    /// Create a new RAG client with default configuration
    ///
    /// This will initialize the embedding model, vector database, and load
    /// any existing caches from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration cannot be loaded
    /// - Embedding model cannot be initialized
    /// - Vector database cannot be initialized
    pub async fn new() -> Result<Self> {
        let config = Config::new().context("Failed to load configuration")?;
        Self::with_config(config).await
    }

    /// Create a new RAG client with custom configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use crate::rag::{RagClient, Config};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let mut config = Config::default();
    ///     config.embedding.model_name = "BAAI/bge-small-en-v1.5".to_string();
    ///
    ///     let client = RagClient::with_config(config).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn with_config(config: Config) -> Result<Self> {
        tracing::info!("Initializing RAG client with configuration");
        tracing::debug!("Vector DB backend: {}", config.vector_db.backend);
        tracing::debug!("Embedding model: {}", config.embedding.model_name);
        tracing::debug!("Chunk size: {}", config.indexing.chunk_size);

        // Initialize embedding provider with configured model
        let embedding_provider = Arc::new(
            FastEmbedManager::from_model_name(&config.embedding.model_name)
                .context("Failed to initialize embedding provider")?,
        );

        // Initialize the appropriate vector database backend
        #[cfg(feature = "qdrant-backend")]
        let vector_db: Arc<dyn VectorDatabase> = {
            tracing::info!(
                "Using Qdrant vector database backend at {}",
                config.vector_db.qdrant_url
            );
            Arc::new(
                QdrantDatabase::with_url(&config.vector_db.qdrant_url)
                    .await
                    .context("Failed to initialize Qdrant vector database")?,
            ) as Arc<dyn VectorDatabase>
        };

        #[cfg(not(feature = "qdrant-backend"))]
        let vector_db: Arc<dyn VectorDatabase> = {
            tracing::info!(
                "Using LanceDB vector database backend at {}",
                config.vector_db.lancedb_path.display()
            );
            Arc::new(
                LanceDatabase::new(config.vector_db.lancedb_path.to_string_lossy().into_owned())
                    .await
                    .context("Failed to initialize LanceDB vector database")?,
            ) as Arc<dyn VectorDatabase>
        };

        // Initialize the database with the embedding dimension
        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        // Create chunker with configured chunk size
        let chunker = Arc::new(CodeChunker::default_strategy());

        // Load persistent hash cache
        let cache_path = config.cache.hash_cache_path.clone();
        let hash_cache = HashCache::load(&cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load hash cache: {}, starting fresh", e);
            HashCache::default()
        });

        tracing::info!("Using hash cache file: {:?}", cache_path);

        // Load persistent git cache
        let git_cache_path = config.cache.git_cache_path.clone();
        let git_cache = GitCache::load(&git_cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load git cache: {}, starting fresh", e);
            GitCache::default()
        });

        tracing::info!("Using git cache file: {:?}", git_cache_path);

        // Initialize relations provider for code navigation
        let relations_provider = Arc::new(
            HybridRelationsProvider::new().context("Failed to initialize relations provider")?,
        );

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            git_cache: Arc::new(RwLock::new(git_cache)),
            git_cache_path,
            config: Arc::new(config),
            indexing_ops: Arc::new(RwLock::new(HashMap::new())),
            relations_provider,
        })
    }

    /// Create a RAG client with an externally-provided vector database.
    ///
    /// This enables callers to share a database connection across subsystems
    /// instead of creating a new one internally.
    pub async fn with_vector_db(
        vector_db: Arc<dyn VectorDatabase>,
        config: Config,
    ) -> Result<Self> {
        tracing::info!("Initializing RAG client with externally-provided vector database");

        // Initialize embedding provider with configured model
        let embedding_provider = Arc::new(
            FastEmbedManager::from_model_name(&config.embedding.model_name)
                .context("Failed to initialize embedding provider")?,
        );

        // Initialize the database with the embedding dimension
        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        // Create chunker with configured chunk size
        let chunker = Arc::new(CodeChunker::default_strategy());

        // Load persistent hash cache
        let cache_path = config.cache.hash_cache_path.clone();
        let hash_cache = HashCache::load(&cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load hash cache: {}, starting fresh", e);
            HashCache::default()
        });

        // Load persistent git cache
        let git_cache_path = config.cache.git_cache_path.clone();
        let git_cache = GitCache::load(&git_cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load git cache: {}, starting fresh", e);
            GitCache::default()
        });

        // Initialize relations provider for code navigation
        let relations_provider = Arc::new(
            HybridRelationsProvider::new().context("Failed to initialize relations provider")?,
        );

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            git_cache: Arc::new(RwLock::new(git_cache)),
            git_cache_path,
            config: Arc::new(config),
            indexing_ops: Arc::new(RwLock::new(HashMap::new())),
            relations_provider,
        })
    }

    /// Create a new client with custom database path (for testing)
    #[cfg(test)]
    pub async fn new_with_db_path(db_path: &str, cache_path: PathBuf) -> Result<Self> {
        // Create a test config with custom paths
        let mut config = Config::default();
        config.vector_db.lancedb_path = PathBuf::from(db_path);
        config.cache.hash_cache_path = cache_path.clone();
        config.cache.git_cache_path = cache_path.parent().unwrap().join("git_cache.json");

        Self::with_config(config).await
    }

    /// Create FileInfo from a file path for relations analysis
    pub(crate) fn create_file_info(
        &self,
        file_path: &str,
        project: Option<String>,
    ) -> Result<FileInfo> {
        use std::path::Path;

        let path = Path::new(file_path);
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("Failed to canonicalize path: {}", file_path))?;

        let content = std::fs::read_to_string(&canonical)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        let extension = canonical
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string());

        let language = extension.as_ref().and_then(|ext| detect_language(ext));

        // Compute file hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Determine root path (parent directory)
        let root_path = canonical
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/".to_string());

        let relative_path = canonical
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.to_string());

        Ok(FileInfo {
            path: canonical,
            relative_path,
            root_path,
            project,
            extension,
            language,
            content,
            hash,
        })
    }

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    pub fn normalize_path(path: &str) -> Result<String> {
        let path_buf = PathBuf::from(path);
        let canonical = std::fs::canonicalize(&path_buf)
            .with_context(|| format!("Failed to canonicalize path: {}", path))?;
        Ok(canonical.to_string_lossy().to_string())
    }

    /// Check if a specific path's index is dirty (incomplete/corrupted)
    ///
    /// Returns true if the path is marked as dirty, meaning a previous indexing
    /// operation was interrupted and the data may be inconsistent.
    pub async fn is_index_dirty(&self, path: &str) -> bool {
        if let Ok(normalized) = Self::normalize_path(path) {
            let cache = self.hash_cache.read().await;
            cache.is_dirty(&normalized)
        } else {
            false
        }
    }

    /// Check if any indexed paths are dirty
    ///
    /// Returns a list of paths that have dirty indexes.
    pub async fn get_dirty_paths(&self) -> Vec<String> {
        let cache = self.hash_cache.read().await;
        cache.get_dirty_roots().keys().cloned().collect()
    }

    /// Check if searching on a specific path should be blocked due to dirty state
    ///
    /// Returns an error if the path is dirty, otherwise Ok(())
    pub(crate) async fn check_path_not_dirty(&self, path: Option<&str>) -> Result<()> {
        if let Some(p) = path
            && self.is_index_dirty(p).await
        {
            anyhow::bail!(
                "Index for '{}' is dirty (previous indexing was interrupted). \
                    Please re-run index_codebase to rebuild the index before querying.",
                p
            );
        }
        Ok(())
    }

    /// Get the configuration used by this client
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the embedding dimension used by this client
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_provider.dimension()
    }
}
