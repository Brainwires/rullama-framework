//! Image Analysis Store
//!
//! Provides storage and retrieval for analyzed images with embeddings.
//! Images are stored with their LLM-generated analysis for semantic search.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};
use brainwires_storage::embeddings::CachedEmbeddingProvider;
use brainwires_storage::image_types::{
    ImageFormat, ImageMetadata, ImageSearchRequest, ImageSearchResult, ImageStorage,
};

const TABLE_NAME: &str = "images";

// ── Schema ──────────────────────────────────────────────────────────────

fn table_schema(dimension: usize) -> Vec<FieldDef> {
    vec![
        FieldDef::required("vector", FieldType::Vector(dimension)),
        FieldDef::required("image_id", FieldType::Utf8),
        FieldDef::optional("message_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::optional("file_name", FieldType::Utf8),
        FieldDef::required("format", FieldType::Utf8),
        FieldDef::required("mime_type", FieldType::Utf8),
        FieldDef::optional("width", FieldType::UInt32),
        FieldDef::optional("height", FieldType::UInt32),
        FieldDef::required("file_size_bytes", FieldType::UInt64),
        FieldDef::required("file_hash", FieldType::Utf8),
        FieldDef::required("analysis", FieldType::Utf8),
        FieldDef::optional("extracted_text", FieldType::Utf8),
        FieldDef::required("tags", FieldType::Utf8), // JSON-encoded Vec<String>
        FieldDef::required("storage_type", FieldType::Utf8),
        FieldDef::required("storage_value", FieldType::Utf8),
        FieldDef::required("created_at", FieldType::Int64),
    ]
}

// ── Record conversion helpers ───────────────────────────────────────────

fn to_record(m: &ImageMetadata, storage: &ImageStorage, embedding: Vec<f32>) -> Record {
    let tags_json = serde_json::to_string(&m.tags).unwrap_or_else(|_| "[]".to_string());

    vec![
        ("vector".into(), FieldValue::Vector(embedding)),
        (
            "image_id".into(),
            FieldValue::Utf8(Some(m.image_id.clone())),
        ),
        ("message_id".into(), FieldValue::Utf8(m.message_id.clone())),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(m.conversation_id.clone())),
        ),
        ("file_name".into(), FieldValue::Utf8(m.file_name.clone())),
        (
            "format".into(),
            FieldValue::Utf8(Some(m.format.as_str().to_string())),
        ),
        (
            "mime_type".into(),
            FieldValue::Utf8(Some(m.mime_type.clone())),
        ),
        ("width".into(), FieldValue::UInt32(m.width)),
        ("height".into(), FieldValue::UInt32(m.height)),
        (
            "file_size_bytes".into(),
            FieldValue::UInt64(Some(m.file_size_bytes)),
        ),
        (
            "file_hash".into(),
            FieldValue::Utf8(Some(m.file_hash.clone())),
        ),
        (
            "analysis".into(),
            FieldValue::Utf8(Some(m.analysis.clone())),
        ),
        (
            "extracted_text".into(),
            FieldValue::Utf8(m.extracted_text.clone()),
        ),
        ("tags".into(), FieldValue::Utf8(Some(tags_json))),
        (
            "storage_type".into(),
            FieldValue::Utf8(Some(storage.storage_type().to_string())),
        ),
        (
            "storage_value".into(),
            FieldValue::Utf8(Some(storage.value().to_string())),
        ),
        ("created_at".into(), FieldValue::Int64(Some(m.created_at))),
    ]
}

fn from_record(r: &Record) -> Result<ImageMetadata> {
    let image_id = record_get(r, "image_id")
        .and_then(|v| v.as_str())
        .context("missing image_id")?
        .to_string();

    let message_id = record_get(r, "message_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let conversation_id = record_get(r, "conversation_id")
        .and_then(|v| v.as_str())
        .context("missing conversation_id")?
        .to_string();

    let file_name = record_get(r, "file_name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let format_str = record_get(r, "format")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let format: ImageFormat = format_str.parse().unwrap_or(ImageFormat::Unknown);

    let mime_type = record_get(r, "mime_type")
        .and_then(|v| v.as_str())
        .unwrap_or("application/octet-stream")
        .to_string();

    let width = record_get(r, "width").and_then(|v| match v {
        FieldValue::UInt32(Some(n)) => Some(*n).filter(|&n| n > 0),
        _ => None,
    });

    let height = record_get(r, "height").and_then(|v| match v {
        FieldValue::UInt32(Some(n)) => Some(*n).filter(|&n| n > 0),
        _ => None,
    });

    let file_size_bytes = record_get(r, "file_size_bytes")
        .and_then(|v| match v {
            FieldValue::UInt64(Some(n)) => Some(*n),
            _ => None,
        })
        .unwrap_or(0);

    let file_hash = record_get(r, "file_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let analysis = record_get(r, "analysis")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let extracted_text = record_get(r, "extracted_text")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let tags_json = record_get(r, "tags")
        .and_then(|v| v.as_str())
        .unwrap_or("[]");
    let tags: Vec<String> = serde_json::from_str(tags_json).unwrap_or_default();

    let created_at = record_get(r, "created_at")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    Ok(ImageMetadata {
        image_id,
        message_id,
        conversation_id,
        file_name,
        format,
        mime_type,
        width,
        height,
        file_size_bytes,
        file_hash,
        analysis,
        extracted_text,
        tags,
        created_at,
    })
}

fn storage_from_record(r: &Record) -> Option<ImageStorage> {
    let storage_type = record_get(r, "storage_type").and_then(|v| v.as_str())?;
    let storage_value = record_get(r, "storage_value")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Some(match storage_type {
        "base64" => ImageStorage::Base64(storage_value),
        "file" => ImageStorage::FilePath(storage_value),
        "url" => ImageStorage::Url(storage_value),
        _ => ImageStorage::Base64(storage_value),
    })
}

// ── ImageStore ──────────────────────────────────────────────────────────

/// Store for analyzed images with semantic search
pub struct ImageStore<B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase> {
    backend: Arc<B>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl<B: StorageBackend> ImageStore<B> {
    /// Create a new image store
    pub fn new(backend: Arc<B>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self {
            backend,
            embeddings,
        }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        let dimension = self.embeddings.dimension();
        self.backend
            .ensure_table(TABLE_NAME, &table_schema(dimension))
            .await
    }

    /// Compute SHA256 hash of image bytes
    pub fn compute_hash(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    /// Store an analyzed image
    ///
    /// # Arguments
    /// * `metadata` - Image metadata including analysis
    /// * `storage` - How to store the image data (base64, file path, or URL)
    pub async fn store(
        &self,
        metadata: ImageMetadata,
        storage: ImageStorage,
    ) -> Result<ImageMetadata> {
        // Generate embedding from searchable text (analysis + OCR + tags)
        let searchable_text = metadata.searchable_text();
        let embedding = self.embeddings.embed(&searchable_text)?;

        let record = to_record(&metadata, &storage, embedding);

        self.backend
            .insert(TABLE_NAME, vec![record])
            .await
            .context("Failed to store image")?;

        Ok(metadata)
    }

    /// Store image with analysis from bytes
    ///
    /// # Arguments
    /// * `bytes` - Raw image bytes
    /// * `analysis` - LLM-generated analysis
    /// * `conversation_id` - Conversation to associate with
    /// * `format` - Image format
    pub async fn store_from_bytes(
        &self,
        bytes: &[u8],
        analysis: String,
        conversation_id: String,
        format: ImageFormat,
    ) -> Result<ImageMetadata> {
        let file_hash = Self::compute_hash(bytes);

        // Check for duplicate
        if let Some(existing) = self.get_by_hash(&file_hash).await? {
            return Ok(existing);
        }

        let image_id = format!("img_{}", uuid::Uuid::new_v4());
        let metadata = ImageMetadata::new(
            image_id,
            conversation_id,
            format,
            bytes.len() as u64,
            file_hash,
            analysis,
        );

        let storage = ImageStorage::from_bytes(bytes);
        self.store(metadata, storage).await
    }

    /// Get image by hash (for deduplication)
    pub async fn get_by_hash(&self, file_hash: &str) -> Result<Option<ImageMetadata>> {
        let filter = Filter::Eq(
            "file_hash".into(),
            FieldValue::Utf8(Some(file_hash.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), Some(1))
            .await
            .context("Failed to query images by hash")?;

        match records.first() {
            Some(r) => Ok(Some(from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Get image by ID
    pub async fn get(&self, image_id: &str) -> Result<Option<ImageMetadata>> {
        let filter = Filter::Eq(
            "image_id".into(),
            FieldValue::Utf8(Some(image_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), Some(1))
            .await
            .context("Failed to query image by ID")?;

        match records.first() {
            Some(r) => Ok(Some(from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Search images using semantic search on analysis text
    pub async fn search(&self, request: ImageSearchRequest) -> Result<Vec<ImageSearchResult>> {
        // Generate query embedding
        let query_embedding = self.embeddings.embed(&request.query)?;

        // Build filter
        let mut filters = Vec::new();

        if let Some(ref conv_id) = request.conversation_id {
            filters.push(Filter::Eq(
                "conversation_id".into(),
                FieldValue::Utf8(Some(conv_id.clone())),
            ));
        }

        if let Some(format) = request.format {
            filters.push(Filter::Eq(
                "format".into(),
                FieldValue::Utf8(Some(format.as_str().to_string())),
            ));
        }

        let filter = if filters.is_empty() {
            None
        } else if filters.len() == 1 {
            Some(filters.remove(0))
        } else {
            Some(Filter::And(filters))
        };

        // Execute vector search
        let scored_records = self
            .backend
            .vector_search(
                TABLE_NAME,
                "vector",
                query_embedding,
                request.limit,
                filter.as_ref(),
            )
            .await
            .context("Failed to execute image search")?;

        let mut search_results = Vec::new();

        for scored in &scored_records {
            if scored.score < request.min_score {
                continue;
            }

            let metadata = from_record(&scored.record)?;
            search_results.push(ImageSearchResult::from_metadata(metadata, scored.score));
        }

        // Sort by score descending
        search_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(search_results)
    }

    /// List images by conversation
    pub async fn list_by_conversation(&self, conversation_id: &str) -> Result<Vec<ImageMetadata>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), None)
            .await
            .context("Failed to list images by conversation")?;

        let mut images: Vec<ImageMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by created_at descending
        images.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(images)
    }

    /// List images by message
    pub async fn list_by_message(&self, message_id: &str) -> Result<Vec<ImageMetadata>> {
        let filter = Filter::Eq(
            "message_id".into(),
            FieldValue::Utf8(Some(message_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), None)
            .await
            .context("Failed to list images by message")?;

        let images: Vec<ImageMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        Ok(images)
    }

    /// Delete an image
    pub async fn delete(&self, image_id: &str) -> Result<bool> {
        let filter = Filter::Eq(
            "image_id".into(),
            FieldValue::Utf8(Some(image_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete image")?;

        Ok(true)
    }

    /// Delete all images for a conversation
    pub async fn delete_by_conversation(&self, conversation_id: &str) -> Result<usize> {
        let images = self.list_by_conversation(conversation_id).await?;
        let count = images.len();

        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete images by conversation")?;

        Ok(count)
    }

    /// Get image data (base64 or path)
    pub async fn get_image_data(&self, image_id: &str) -> Result<Option<ImageStorage>> {
        let filter = Filter::Eq(
            "image_id".into(),
            FieldValue::Utf8(Some(image_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), Some(1))
            .await
            .context("Failed to query image data")?;

        match records.first() {
            Some(r) => Ok(storage_from_record(r)),
            None => Ok(None),
        }
    }

    /// Count images in a conversation
    pub async fn count_by_conversation(&self, conversation_id: &str) -> Result<usize> {
        let images = self.list_by_conversation(conversation_id).await?;
        Ok(images.len())
    }
}
