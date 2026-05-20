use anyhow::{Context, Result};
use std::sync::Arc;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};

const TABLE_NAME: &str = "messages";

/// Metadata for a message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MessageMetadata {
    /// Unique message identifier.
    pub message_id: String,
    /// Conversation this message belongs to.
    pub conversation_id: String,
    /// Message role (e.g., "user", "assistant").
    pub role: String,
    /// Message text content.
    pub content: String,
    /// Token count estimate.
    pub token_count: Option<i32>,
    /// Model that generated this message.
    pub model_id: Option<String>,
    /// Image references as JSON array string.
    pub images: Option<String>, // JSON array as string
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Optional Unix timestamp after which this entry should be evicted.
    ///
    /// `None` means no expiry (the entry persists indefinitely).  Use
    /// [`MessageStore::delete_expired`] to perform bulk eviction, or call
    /// `TieredMemory::evict_expired` (in `brainwires-memory`) for tier-aware cleanup.
    pub expires_at: Option<i64>,
}

/// Return the backend-agnostic table schema for messages.
fn table_schema(embedding_dim: usize) -> Vec<FieldDef> {
    vec![
        FieldDef::required("vector", FieldType::Vector(embedding_dim)),
        FieldDef::required("message_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::required("role", FieldType::Utf8),
        FieldDef::required("content", FieldType::Utf8),
        FieldDef::optional("token_count", FieldType::Int32),
        FieldDef::optional("model_id", FieldType::Utf8),
        FieldDef::optional("images", FieldType::Utf8),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::optional("expires_at", FieldType::Int64),
    ]
}

/// Arrow `Schema` for the messages table (LanceDatabase compatibility).
pub fn messages_schema(embedding_dim: usize) -> Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field, Schema};

    Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
        Field::new("message_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("token_count", DataType::Int32, true),
        Field::new("model_id", DataType::Utf8, true),
        Field::new("images", DataType::Utf8, true),
        Field::new("created_at", DataType::Int64, false),
        Field::new("expires_at", DataType::Int64, true),
    ]))
}

fn to_record(m: &MessageMetadata, embedding: Vec<f32>) -> Record {
    vec![
        ("vector".into(), FieldValue::Vector(embedding)),
        (
            "message_id".into(),
            FieldValue::Utf8(Some(m.message_id.clone())),
        ),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(m.conversation_id.clone())),
        ),
        ("role".into(), FieldValue::Utf8(Some(m.role.clone()))),
        ("content".into(), FieldValue::Utf8(Some(m.content.clone()))),
        ("token_count".into(), FieldValue::Int32(m.token_count)),
        ("model_id".into(), FieldValue::Utf8(m.model_id.clone())),
        ("images".into(), FieldValue::Utf8(m.images.clone())),
        ("created_at".into(), FieldValue::Int64(Some(m.created_at))),
        ("expires_at".into(), FieldValue::Int64(m.expires_at)),
    ]
}

fn from_record(r: &Record) -> Result<MessageMetadata> {
    Ok(MessageMetadata {
        message_id: record_get(r, "message_id")
            .and_then(|v| v.as_str())
            .context("missing message_id")?
            .to_string(),
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        role: record_get(r, "role")
            .and_then(|v| v.as_str())
            .context("missing role")?
            .to_string(),
        content: record_get(r, "content")
            .and_then(|v| v.as_str())
            .context("missing content")?
            .to_string(),
        token_count: record_get(r, "token_count").and_then(|v| v.as_i32()),
        model_id: record_get(r, "model_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        images: record_get(r, "images")
            .and_then(|v| v.as_str())
            .map(String::from),
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        expires_at: record_get(r, "expires_at").and_then(|v| v.as_i64()),
    })
}

/// Store for managing messages with semantic search
pub struct MessageStore<B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase> {
    backend: Arc<B>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl<B: StorageBackend> MessageStore<B> {
    /// Create a new message store
    pub fn new(backend: Arc<B>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self {
            backend,
            embeddings,
        }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend
            .ensure_table(TABLE_NAME, &table_schema(self.embeddings.dimension()))
            .await
    }

    /// Add a message to the store
    pub async fn add(&self, message: MessageMetadata) -> Result<()> {
        // Generate embedding for the content
        let embedding = self.embeddings.embed(&message.content)?;
        let record = to_record(&message, embedding);

        self.backend
            .insert(TABLE_NAME, vec![record])
            .await
            .context("Failed to add message")?;

        Ok(())
    }

    /// Add multiple messages in batch
    pub async fn add_batch(&self, messages: Vec<MessageMetadata>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Generate embeddings for all messages
        let contents: Vec<String> = messages.iter().map(|m| m.content.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&contents)?;

        let records: Vec<Record> = messages
            .iter()
            .zip(embeddings.into_iter())
            .map(|(m, emb)| to_record(m, emb))
            .collect();

        self.backend
            .insert(TABLE_NAME, records)
            .await
            .context("Failed to add messages")?;

        Ok(())
    }

    /// Get a single message by ID
    pub async fn get(&self, message_id: &str) -> Result<Option<MessageMetadata>> {
        let filter = Filter::Eq(
            "message_id".into(),
            FieldValue::Utf8(Some(message_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), Some(1))
            .await?;

        match records.first() {
            Some(r) => Ok(Some(from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Get messages for a conversation
    pub async fn get_by_conversation(&self, conversation_id: &str) -> Result<Vec<MessageMetadata>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;

        records.iter().map(from_record).collect()
    }

    /// Search messages by semantic similarity
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(MessageMetadata, f32)>> {
        self.search_with_filter(query, limit, min_score, None).await
    }

    /// Search messages within a specific conversation by semantic similarity
    pub async fn search_conversation(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(MessageMetadata, f32)>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.search_with_filter(query, limit, min_score, Some(filter))
            .await
    }

    /// Search messages with optional filter by semantic similarity
    async fn search_with_filter(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
        filter: Option<Filter>,
    ) -> Result<Vec<(MessageMetadata, f32)>> {
        // Generate query embedding (use cached version for repeated queries)
        let query_embedding = self.embeddings.embed_cached(query)?;

        let scored = self
            .backend
            .vector_search(
                TABLE_NAME,
                "vector",
                query_embedding,
                limit,
                filter.as_ref(),
            )
            .await?;

        let mut messages_with_scores = Vec::new();

        for sr in scored {
            if sr.score >= min_score {
                let message = from_record(&sr.record)?;
                messages_with_scores.push((message, sr.score));
            }
        }

        Ok(messages_with_scores)
    }

    /// Delete all messages for a conversation
    pub async fn delete_by_conversation(&self, conversation_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend.delete(TABLE_NAME, &filter).await?;
        Ok(())
    }

    /// Delete a specific message
    pub async fn delete(&self, message_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "message_id".into(),
            FieldValue::Utf8(Some(message_id.to_string())),
        );
        self.backend.delete(TABLE_NAME, &filter).await?;
        Ok(())
    }

    /// Delete all messages whose `expires_at` timestamp is in the past.
    ///
    /// Returns the number of rows deleted.  Rows with `expires_at = NULL`
    /// (no TTL) are never touched.
    ///
    /// Call this at agent run completion or on a periodic background schedule
    /// to enforce session-tier TTL policies.
    pub async fn delete_expired(&self) -> Result<usize> {
        use chrono::Utc;
        let now = Utc::now().timestamp();

        let filter = Filter::And(vec![
            Filter::NotNull("expires_at".into()),
            Filter::Lte("expires_at".into(), FieldValue::Int64(Some(now))),
        ]);

        let count = self.backend.count(TABLE_NAME, Some(&filter)).await?;
        if count > 0 {
            self.backend.delete(TABLE_NAME, &filter).await?;
        }
        Ok(count)
    }
}
