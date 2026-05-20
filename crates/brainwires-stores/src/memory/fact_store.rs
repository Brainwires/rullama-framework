//! Persistent storage for cold tier key facts
//!
//! Uses a [`StorageBackend`](brainwires_storage::StorageBackend) for persistence with semantic search capability.

use anyhow::{Context, Result};
use std::sync::Arc;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord, StorageBackend, record_get,
};

use super::tier_types::{FactType, KeyFact};

const TABLE_NAME: &str = "facts";

// ── Schema ──────────────────────────────────────────────────────────────

/// Return the backend-agnostic field definitions for the facts table.
pub fn facts_field_defs(embedding_dim: usize) -> Vec<FieldDef> {
    vec![
        FieldDef::required("fact_id", FieldType::Utf8),
        FieldDef::required("original_message_ids", FieldType::Utf8), // JSON array
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::required("fact", FieldType::Utf8),
        FieldDef::required("fact_type", FieldType::Utf8),
        FieldDef::required("vector", FieldType::Vector(embedding_dim)),
        FieldDef::required("created_at", FieldType::Int64),
    ]
}

/// Arrow schema for the facts table, used by `LanceDatabase` table creation.
pub fn facts_schema(embedding_dim: usize) -> std::sync::Arc<arrow_schema::Schema> {
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
        Field::new("fact_id", DataType::Utf8, false),
        Field::new("original_message_ids", DataType::Utf8, false), // JSON array
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("fact", DataType::Utf8, false),
        Field::new("fact_type", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

// ── Record conversion helpers ───────────────────────────────────────────

fn to_record(fact: &KeyFact, embedding: Vec<f32>) -> Record {
    let original_message_ids_json =
        serde_json::to_string(&fact.original_message_ids).unwrap_or_else(|_| "[]".to_string());

    vec![
        (
            "fact_id".into(),
            FieldValue::Utf8(Some(fact.fact_id.clone())),
        ),
        (
            "original_message_ids".into(),
            FieldValue::Utf8(Some(original_message_ids_json)),
        ),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(fact.conversation_id.clone())),
        ),
        ("fact".into(), FieldValue::Utf8(Some(fact.fact.clone()))),
        (
            "fact_type".into(),
            FieldValue::Utf8(Some(fact_type_to_string(fact.fact_type).to_string())),
        ),
        ("vector".into(), FieldValue::Vector(embedding)),
        (
            "created_at".into(),
            FieldValue::Int64(Some(fact.created_at)),
        ),
    ]
}

fn from_record(r: &Record) -> Result<KeyFact> {
    let original_message_ids: Vec<String> = record_get(r, "original_message_ids")
        .and_then(|v| v.as_str())
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default();

    let fact_type = record_get(r, "fact_type")
        .and_then(|v| v.as_str())
        .map(string_to_fact_type)
        .unwrap_or(FactType::Other);

    Ok(KeyFact {
        fact_id: record_get(r, "fact_id")
            .and_then(|v| v.as_str())
            .context("missing fact_id")?
            .to_string(),
        original_message_ids,
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        fact: record_get(r, "fact")
            .and_then(|v| v.as_str())
            .context("missing fact")?
            .to_string(),
        fact_type,
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
    })
}

/// Convert fact type to string for storage
fn fact_type_to_string(fact_type: FactType) -> &'static str {
    match fact_type {
        FactType::Decision => "decision",
        FactType::Definition => "definition",
        FactType::Requirement => "requirement",
        FactType::CodeChange => "code_change",
        FactType::Configuration => "configuration",
        FactType::Other => "other",
    }
}

/// Convert string to fact type
fn string_to_fact_type(s: &str) -> FactType {
    match s {
        "decision" => FactType::Decision,
        "definition" => FactType::Definition,
        "requirement" => FactType::Requirement,
        "code_change" => FactType::CodeChange,
        "configuration" => FactType::Configuration,
        _ => FactType::Other,
    }
}

// ── FactStore ───────────────────────────────────────────────────────────

/// Store for cold tier key facts with semantic search
pub struct FactStore<B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase> {
    backend: Arc<B>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl<B: StorageBackend> FactStore<B> {
    /// Create a new fact store
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
            .ensure_table(TABLE_NAME, &facts_field_defs(dim))
            .await
    }

    /// Add a fact to the store
    pub async fn add(&self, fact: KeyFact) -> Result<()> {
        let embedding = self.embeddings.embed(&fact.fact)?;
        let record = to_record(&fact, embedding);

        self.backend
            .insert(TABLE_NAME, vec![record])
            .await
            .context("Failed to add fact")
    }

    /// Add multiple facts in batch
    pub async fn add_batch(&self, facts: Vec<KeyFact>) -> Result<()> {
        if facts.is_empty() {
            return Ok(());
        }

        let contents: Vec<String> = facts.iter().map(|f| f.fact.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&contents)?;

        let records: Vec<Record> = facts
            .iter()
            .zip(embeddings.into_iter())
            .map(|(f, emb)| to_record(f, emb))
            .collect();

        self.backend
            .insert(TABLE_NAME, records)
            .await
            .context("Failed to add facts")
    }

    /// Get a fact by ID
    pub async fn get(&self, fact_id: &str) -> Result<Option<KeyFact>> {
        let filter = Filter::Eq(
            "fact_id".into(),
            FieldValue::Utf8(Some(fact_id.to_string())),
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

    /// Get all facts for a conversation
    pub async fn get_by_conversation(&self, conversation_id: &str) -> Result<Vec<KeyFact>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;

        records.iter().map(from_record).collect()
    }

    /// Search facts by semantic similarity
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(KeyFact, f32)>> {
        self.search_with_filter(query, limit, min_score, None).await
    }

    /// Search facts within a specific conversation
    pub async fn search_conversation(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(KeyFact, f32)>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.search_with_filter(query, limit, min_score, Some(filter))
            .await
    }

    /// Search facts with optional filter
    async fn search_with_filter(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
        filter: Option<Filter>,
    ) -> Result<Vec<(KeyFact, f32)>> {
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

        scored_records_to_facts(&scored, min_score)
    }

    /// Delete a fact by ID
    pub async fn delete(&self, fact_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "fact_id".into(),
            FieldValue::Utf8(Some(fact_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete fact")
    }

    /// Get count of facts
    pub async fn count(&self) -> Result<usize> {
        self.backend.count(TABLE_NAME, None).await
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn scored_records_to_facts(scored: &[ScoredRecord], min_score: f32) -> Result<Vec<(KeyFact, f32)>> {
    let mut results = Vec::new();
    for sr in scored {
        if sr.score >= min_score {
            let fact = from_record(&sr.record)?;
            results.push((fact, sr.score));
        }
    }
    Ok(results)
}
