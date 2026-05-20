//! Core [`LanceDatabase`] struct and helper methods.

use anyhow::{Context, Result};
use arrow_array::{FixedSizeListArray, RecordBatch, StringArray, UInt32Array, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use lancedb::Table;
use lancedb::connection::Connection;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::bm25_search::{BM25Search, RrfScorer, SearchScorer};
use crate::databases::traits::ChunkMetadata;

/// Default table name for RAG embeddings.
pub(super) const RAG_TABLE_NAME: &str = "code_embeddings";

/// Unified LanceDB database backend.
///
/// Holds a single `lancedb::Connection` and implements both
/// [`StorageBackend`](crate::databases::traits::StorageBackend) (for domain stores) and
/// [`VectorDatabase`](crate::databases::traits::VectorDatabase) (for RAG).
///
/// # Example
///
/// ```ignore
/// let db = Arc::new(LanceDatabase::new("/path/to/db").await?);
///
/// // Use as StorageBackend
/// let messages = MessageStore::new(db.clone(), embeddings);
///
/// // Use as VectorDatabase
/// db.initialize(384).await?;
/// db.store_embeddings(embeddings, metadata, contents, root_path).await?;
/// ```
pub struct LanceDatabase {
    pub(super) connection: Connection,
    pub(super) db_path: String,
    /// RAG table name (default: "code_embeddings").
    pub(super) rag_table_name: String,
    /// Per-project BM25 search indexes for keyword matching.
    pub(super) bm25_indexes: Arc<RwLock<HashMap<String, BM25Search>>>,
    /// Pluggable search scorer for hybrid result fusion (default: RRF).
    pub(super) scorer: Arc<dyn SearchScorer>,
}

impl LanceDatabase {
    /// Create a new LanceDB database at the given path.
    ///
    /// The path can be a local directory. Parent directories are created
    /// automatically.
    pub async fn new(db_path: impl Into<String>) -> Result<Self> {
        let db_path = db_path.into();

        if let Some(parent) = std::path::Path::new(&db_path).parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        let connection = lancedb::connect(&db_path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        Ok(Self {
            connection,
            db_path,
            rag_table_name: RAG_TABLE_NAME.to_string(),
            bm25_indexes: Arc::new(RwLock::new(HashMap::new())),
            scorer: Arc::new(RrfScorer),
        })
    }

    /// Create with the platform default LanceDB path.
    pub async fn with_default_path() -> Result<Self> {
        let db_path = Self::default_lancedb_path();
        Self::new(db_path).await
    }

    /// Set a custom search scorer for hybrid result fusion.
    pub fn with_scorer(mut self, scorer: Arc<dyn SearchScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// Get the underlying LanceDB connection (for legacy code).
    pub fn connection(&self) -> &Connection {
        &self.connection
    }

    /// Get the database path.
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    /// Report backend capabilities.
    pub fn capabilities(&self) -> crate::databases::BackendCapabilities {
        crate::databases::BackendCapabilities {
            vector_search: true,
        }
    }

    /// Get default database path.
    pub fn default_lancedb_path() -> String {
        brainwires_core::paths::PlatformPaths::default_lancedb_path()
            .to_string_lossy()
            .to_string()
    }

    // ── VectorDatabase helpers ──────────────────────────────────────────

    pub(super) fn hash_root_path(root_path: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(root_path.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)[..16].to_string()
    }

    pub(super) fn bm25_path_for_root(&self, root_path: &str) -> String {
        let hash = Self::hash_root_path(root_path);
        format!("{}/bm25_{}", self.db_path, hash)
    }

    pub(super) fn get_or_create_bm25(&self, root_path: &str) -> Result<()> {
        let hash = Self::hash_root_path(root_path);

        {
            let indexes = self.bm25_indexes.read().map_err(|e| {
                anyhow::anyhow!("Failed to acquire read lock on BM25 indexes: {}", e)
            })?;
            if indexes.contains_key(&hash) {
                return Ok(());
            }
        }

        let mut indexes = self
            .bm25_indexes
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock on BM25 indexes: {}", e))?;

        if indexes.contains_key(&hash) {
            return Ok(());
        }

        let bm25_path = self.bm25_path_for_root(root_path);
        tracing::info!(
            "Creating BM25 index for root path '{}' at: {}",
            root_path,
            bm25_path
        );

        let bm25_index = BM25Search::new(&bm25_path)
            .with_context(|| format!("Failed to initialize BM25 index for root: {}", root_path))?;

        indexes.insert(hash, bm25_index);
        Ok(())
    }

    pub(super) fn create_rag_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("root_path", DataType::Utf8, true),
            Field::new("start_line", DataType::UInt32, false),
            Field::new("end_line", DataType::UInt32, false),
            Field::new("language", DataType::Utf8, false),
            Field::new("extension", DataType::Utf8, false),
            Field::new("file_hash", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("project", DataType::Utf8, true),
        ]))
    }

    pub(super) async fn get_rag_table(&self) -> Result<Table> {
        self.connection
            .open_table(&self.rag_table_name)
            .execute()
            .await
            .context("Failed to open RAG table")
    }

    pub(super) fn create_rag_record_batch(
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        schema: Arc<Schema>,
    ) -> Result<RecordBatch> {
        let num_rows = embeddings.len();
        let dimension = embeddings[0].len();

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings
                .into_iter()
                .map(|v| Some(v.into_iter().map(Some))),
            dimension as i32,
        );

        let id_array = StringArray::from(
            (0..num_rows)
                .map(|i| format!("{}:{}", metadata[i].file_path, metadata[i].start_line))
                .collect::<Vec<_>>(),
        );
        let file_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_path.as_str())
                .collect::<Vec<_>>(),
        );
        let root_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.root_path.as_deref())
                .collect::<Vec<_>>(),
        );
        let start_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.start_line as u32)
                .collect::<Vec<_>>(),
        );
        let end_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.end_line as u32)
                .collect::<Vec<_>>(),
        );
        let language_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.language.as_deref().unwrap_or("Unknown"))
                .collect::<Vec<_>>(),
        );
        let extension_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.extension.as_deref().unwrap_or(""))
                .collect::<Vec<_>>(),
        );
        let file_hash_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_hash.as_str())
                .collect::<Vec<_>>(),
        );
        let indexed_at_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.indexed_at.to_string())
                .collect::<Vec<_>>(),
        );
        let content_array =
            StringArray::from(contents.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let project_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.project.as_deref())
                .collect::<Vec<_>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vector_array),
                Arc::new(id_array),
                Arc::new(file_path_array),
                Arc::new(root_path_array),
                Arc::new(start_line_array),
                Arc::new(end_line_array),
                Arc::new(language_array),
                Arc::new(extension_array),
                Arc::new(file_hash_array),
                Arc::new(indexed_at_array),
                Arc::new(content_array),
                Arc::new(project_array),
            ],
        )
        .context("Failed to create RecordBatch")
    }
}
