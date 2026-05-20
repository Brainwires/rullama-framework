//! Persistent storage for warm tier message summaries
//!
//! Uses a [`StorageBackend`](brainwires_storage::StorageBackend) for persistence with semantic search capability.

use anyhow::{Context, Result};
use std::sync::Arc;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord, StorageBackend, record_get,
};

use super::tier_types::MessageSummary;

const TABLE_NAME: &str = "summaries";

// ── Schema ──────────────────────────────────────────────────────────────

/// Return the backend-agnostic field definitions for the summaries table.
pub fn summaries_field_defs(embedding_dim: usize) -> Vec<FieldDef> {
    vec![
        FieldDef::required("summary_id", FieldType::Utf8),
        FieldDef::required("original_message_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::required("role", FieldType::Utf8),
        FieldDef::required("summary", FieldType::Utf8),
        FieldDef::required("key_entities", FieldType::Utf8), // JSON array
        FieldDef::required("vector", FieldType::Vector(embedding_dim)),
        FieldDef::required("created_at", FieldType::Int64),
    ]
}

/// Arrow schema for the summaries table, used by `LanceDatabase` table creation.
pub fn summaries_schema(embedding_dim: usize) -> std::sync::Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field};

    std::sync::Arc::new(arrow_schema::Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
        Field::new("summary_id", DataType::Utf8, false),
        Field::new("original_message_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("summary", DataType::Utf8, false),
        Field::new("key_entities", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

// ── Record conversion helpers ───────────────────────────────────────────

fn to_record(summary: &MessageSummary, embedding: Vec<f32>) -> Record {
    let key_entities_json =
        serde_json::to_string(&summary.key_entities).unwrap_or_else(|_| "[]".to_string());

    vec![
        (
            "summary_id".into(),
            FieldValue::Utf8(Some(summary.summary_id.clone())),
        ),
        (
            "original_message_id".into(),
            FieldValue::Utf8(Some(summary.original_message_id.clone())),
        ),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(summary.conversation_id.clone())),
        ),
        ("role".into(), FieldValue::Utf8(Some(summary.role.clone()))),
        (
            "summary".into(),
            FieldValue::Utf8(Some(summary.summary.clone())),
        ),
        (
            "key_entities".into(),
            FieldValue::Utf8(Some(key_entities_json)),
        ),
        ("vector".into(), FieldValue::Vector(embedding)),
        (
            "created_at".into(),
            FieldValue::Int64(Some(summary.created_at)),
        ),
    ]
}

fn from_record(r: &Record) -> Result<MessageSummary> {
    let key_entities: Vec<String> = record_get(r, "key_entities")
        .and_then(|v| v.as_str())
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default();

    Ok(MessageSummary {
        summary_id: record_get(r, "summary_id")
            .and_then(|v| v.as_str())
            .context("missing summary_id")?
            .to_string(),
        original_message_id: record_get(r, "original_message_id")
            .and_then(|v| v.as_str())
            .context("missing original_message_id")?
            .to_string(),
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        role: record_get(r, "role")
            .and_then(|v| v.as_str())
            .context("missing role")?
            .to_string(),
        summary: record_get(r, "summary")
            .and_then(|v| v.as_str())
            .context("missing summary")?
            .to_string(),
        key_entities,
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
    })
}

// ── SummaryStore ────────────────────────────────────────────────────────

/// Store for warm tier message summaries with semantic search
pub struct SummaryStore<B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase> {
    backend: Arc<B>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl<B: StorageBackend> SummaryStore<B> {
    /// Create a new summary store
    pub fn new(backend: Arc<B>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self {
            backend,
            embeddings,
        }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        let dim = self.embeddings.dimension();
        self.backend
            .ensure_table(TABLE_NAME, &summaries_field_defs(dim))
            .await
    }

    /// Add a summary to the store
    pub async fn add(&self, summary: MessageSummary) -> Result<()> {
        let embedding = self.embeddings.embed(&summary.summary)?;
        let record = to_record(&summary, embedding);

        self.backend
            .insert(TABLE_NAME, vec![record])
            .await
            .context("Failed to add summary")
    }

    /// Add multiple summaries in batch
    pub async fn add_batch(&self, summaries: Vec<MessageSummary>) -> Result<()> {
        if summaries.is_empty() {
            return Ok(());
        }

        let contents: Vec<String> = summaries.iter().map(|s| s.summary.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&contents)?;

        let records: Vec<Record> = summaries
            .iter()
            .zip(embeddings.into_iter())
            .map(|(s, emb)| to_record(s, emb))
            .collect();

        self.backend
            .insert(TABLE_NAME, records)
            .await
            .context("Failed to add summaries")
    }

    /// Get a summary by ID
    pub async fn get(&self, summary_id: &str) -> Result<Option<MessageSummary>> {
        let filter = Filter::Eq(
            "summary_id".into(),
            FieldValue::Utf8(Some(summary_id.to_string())),
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

    /// Get all summaries for a conversation
    pub async fn get_by_conversation(&self, conversation_id: &str) -> Result<Vec<MessageSummary>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;

        records.iter().map(from_record).collect()
    }

    /// Search summaries by semantic similarity
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(MessageSummary, f32)>> {
        self.search_with_filter(query, limit, min_score, None).await
    }

    /// Search summaries within a specific conversation
    pub async fn search_conversation(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(MessageSummary, f32)>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.search_with_filter(query, limit, min_score, Some(filter))
            .await
    }

    /// Search summaries with optional filter
    async fn search_with_filter(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
        filter: Option<Filter>,
    ) -> Result<Vec<(MessageSummary, f32)>> {
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

        scored_records_to_summaries(&scored, min_score)
    }

    /// Delete a summary by ID
    pub async fn delete(&self, summary_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "summary_id".into(),
            FieldValue::Utf8(Some(summary_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete summary")
    }

    /// Get count of summaries
    pub async fn count(&self) -> Result<usize> {
        self.backend.count(TABLE_NAME, None).await
    }

    /// Get oldest summaries (for demotion to cold tier)
    pub async fn get_oldest(&self, limit: usize) -> Result<Vec<MessageSummary>> {
        let records = self.backend.query(TABLE_NAME, None, None).await?;

        let mut summaries: Vec<MessageSummary> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by created_at ascending (oldest first)
        summaries.sort_by_key(|s| s.created_at);
        summaries.truncate(limit);

        Ok(summaries)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn scored_records_to_summaries(
    scored: &[ScoredRecord],
    min_score: f32,
) -> Result<Vec<(MessageSummary, f32)>> {
    let mut results = Vec::new();
    for sr in scored {
        if sr.score >= min_score {
            let summary = from_record(&sr.record)?;
            results.push((summary, sr.score));
        }
    }
    Ok(results)
}
