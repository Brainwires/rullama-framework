//! Mental-model tier — synthesised agent beliefs about patterns.
//!
//! The mental-model tier sits below Cold in the memory hierarchy.  It holds
//! high-level, synthesised beliefs that were explicitly derived from a set of
//! source facts rather than from individual conversation messages.
//!
//! Entries are written explicitly via
//! `TieredMemory::synthesize_mental_model` (in `brainwires-memory`); they
//! are never populated automatically.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};

const TABLE_NAME: &str = "mental_models";

// ── Types ────────────────────────────────────────────────────────────────────

/// The kind of pattern a mental model encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    /// Observed behavioural patterns (how people or systems act).
    Behavioral,
    /// Structural / architectural relationships.
    Structural,
    /// Cause-and-effect relationships.
    Causal,
    /// Step-by-step procedural knowledge.
    Procedural,
}

impl ModelType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Behavioral => "behavioral",
            Self::Structural => "structural",
            Self::Causal => "causal",
            Self::Procedural => "procedural",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "behavioral" => Self::Behavioral,
            "structural" => Self::Structural,
            "causal" => Self::Causal,
            "procedural" => Self::Procedural,
            _ => Self::Behavioral,
        }
    }
}

/// A synthesised mental model stored in the lowest memory tier.
#[derive(Debug, Clone)]
pub struct MentalModel {
    /// Unique identifier (UUID).
    pub model_id: String,
    /// IDs of the cold-tier facts this model was synthesised from.
    pub source_fact_ids: Vec<String>,
    /// Conversation context this model belongs to.
    pub conversation_id: String,
    /// The synthesised belief text.
    pub model_text: String,
    /// What kind of pattern this encodes.
    pub model_type: ModelType,
    /// Confidence in this model (0.0–1.0).
    pub confidence: f32,
    /// Number of supporting facts.
    pub evidence_count: u32,
    /// Unix timestamp of creation.
    pub created_at: i64,
}

impl MentalModel {
    /// Create a new mental model with the given text.
    pub fn new(
        model_text: String,
        model_type: ModelType,
        conversation_id: String,
        source_fact_ids: Vec<String>,
    ) -> Self {
        Self {
            model_id: Uuid::new_v4().to_string(),
            source_fact_ids,
            conversation_id,
            model_text,
            model_type,
            confidence: 0.5,
            evidence_count: 0,
            created_at: chrono::Utc::now().timestamp(),
        }
    }
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Persistent storage for the mental-model tier.
pub struct MentalModelStore {
    backend: Arc<dyn StorageBackend>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl MentalModelStore {
    /// Create a new store backed by the given [`StorageBackend`].
    pub fn new(backend: Arc<dyn StorageBackend>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self {
            backend,
            embeddings,
        }
    }

    /// Ensure the `mental_models` table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend
            .ensure_table(
                TABLE_NAME,
                &[
                    FieldDef::required("vector", FieldType::Vector(self.embeddings.dimension())),
                    FieldDef::required("model_id", FieldType::Utf8),
                    FieldDef::required("source_fact_ids", FieldType::Utf8), // JSON
                    FieldDef::required("conversation_id", FieldType::Utf8),
                    FieldDef::required("model_text", FieldType::Utf8),
                    FieldDef::required("model_type", FieldType::Utf8),
                    FieldDef::required("confidence", FieldType::Float32),
                    FieldDef::required("evidence_count", FieldType::Int64),
                    FieldDef::required("created_at", FieldType::Int64),
                ],
            )
            .await
            .context("Failed to create mental_models table")?;
        Ok(())
    }

    /// Persist a new mental model.
    pub async fn add(&self, model: MentalModel) -> Result<()> {
        let embedding = self.embeddings.embed(&model.model_text)?;
        let record = to_record(&model, embedding);
        self.backend
            .insert(TABLE_NAME, vec![record])
            .await
            .context("Failed to insert mental model")?;
        Ok(())
    }

    /// Semantic search over mental models.
    ///
    /// Returns `(model, score)` pairs sorted by descending similarity.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<(MentalModel, f32)>> {
        let embedding = self.embeddings.embed_cached(query)?;
        let scored = self
            .backend
            .vector_search(TABLE_NAME, "vector", embedding, limit, None)
            .await?;
        scored
            .into_iter()
            .map(|sr| {
                let model = from_record(&sr.record)?;
                Ok((model, sr.score))
            })
            .collect()
    }

    /// Delete a mental model by ID.
    pub async fn delete(&self, model_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "model_id".into(),
            FieldValue::Utf8(Some(model_id.to_string())),
        );
        self.backend.delete(TABLE_NAME, &filter).await?;
        Ok(())
    }

    /// Count all stored mental models.
    pub async fn count(&self) -> Result<usize> {
        self.backend.count(TABLE_NAME, None).await
    }
}

// ── Record helpers ────────────────────────────────────────────────────────────

fn to_record(model: &MentalModel, embedding: Vec<f32>) -> Record {
    let source_ids_json =
        serde_json::to_string(&model.source_fact_ids).unwrap_or_else(|_| "[]".into());

    vec![
        ("vector".into(), FieldValue::Vector(embedding)),
        (
            "model_id".into(),
            FieldValue::Utf8(Some(model.model_id.clone())),
        ),
        (
            "source_fact_ids".into(),
            FieldValue::Utf8(Some(source_ids_json)),
        ),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(model.conversation_id.clone())),
        ),
        (
            "model_text".into(),
            FieldValue::Utf8(Some(model.model_text.clone())),
        ),
        (
            "model_type".into(),
            FieldValue::Utf8(Some(model.model_type.as_str().to_string())),
        ),
        (
            "confidence".into(),
            FieldValue::Float32(Some(model.confidence)),
        ),
        (
            "evidence_count".into(),
            FieldValue::Int64(Some(model.evidence_count as i64)),
        ),
        (
            "created_at".into(),
            FieldValue::Int64(Some(model.created_at)),
        ),
    ]
}

fn from_record(record: &Record) -> Result<MentalModel> {
    let model_id = record_get(record, "model_id")
        .and_then(|v| v.as_str())
        .context("Missing model_id")?
        .to_string();
    let source_ids_str = record_get(record, "source_fact_ids")
        .and_then(|v| v.as_str())
        .unwrap_or("[]");
    let source_fact_ids: Vec<String> = serde_json::from_str(source_ids_str).unwrap_or_default();
    let conversation_id = record_get(record, "conversation_id")
        .and_then(|v| v.as_str())
        .context("Missing conversation_id")?
        .to_string();
    let model_text = record_get(record, "model_text")
        .and_then(|v| v.as_str())
        .context("Missing model_text")?
        .to_string();
    let model_type = record_get(record, "model_type")
        .and_then(|v| v.as_str())
        .map(ModelType::parse)
        .unwrap_or(ModelType::Behavioral);
    let confidence = record_get(record, "confidence")
        .and_then(|v| v.as_f32())
        .unwrap_or(0.5);
    let evidence_count = record_get(record, "evidence_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as u32;
    let created_at = record_get(record, "created_at")
        .and_then(|v| v.as_i64())
        .context("Missing created_at")?;

    Ok(MentalModel {
        model_id,
        source_fact_ids,
        conversation_id,
        model_text,
        model_type,
        confidence,
        evidence_count,
        created_at,
    })
}
