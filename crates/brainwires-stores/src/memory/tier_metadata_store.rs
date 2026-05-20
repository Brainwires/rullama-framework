//! Persistent storage for tier metadata
//!
//! Tracks which tier each message is in and access patterns for tier promotion/demotion decisions.

use std::collections::HashMap;

use anyhow::{Context, Result};
use std::sync::Arc;

use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};

use super::tier_types::{MemoryAuthority, MemoryTier, TierMetadata};

const TABLE_NAME: &str = "tier_metadata";

fn table_schema() -> Vec<FieldDef> {
    vec![
        FieldDef::required("message_id", FieldType::Utf8),
        FieldDef::required("tier", FieldType::Utf8),
        FieldDef::required("importance", FieldType::Float32),
        FieldDef::required("last_accessed", FieldType::Int64),
        FieldDef::required("access_count", FieldType::Int32),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::required("authority", FieldType::Utf8),
    ]
}

fn to_record(m: &TierMetadata) -> Record {
    vec![
        (
            "message_id".into(),
            FieldValue::Utf8(Some(m.message_id.clone())),
        ),
        (
            "tier".into(),
            FieldValue::Utf8(Some(tier_to_string(m.tier).to_string())),
        ),
        ("importance".into(), FieldValue::Float32(Some(m.importance))),
        (
            "last_accessed".into(),
            FieldValue::Int64(Some(m.last_accessed)),
        ),
        (
            "access_count".into(),
            FieldValue::Int32(Some(m.access_count as i32)),
        ),
        ("created_at".into(), FieldValue::Int64(Some(m.created_at))),
        (
            "authority".into(),
            FieldValue::Utf8(Some(m.authority.as_str().to_string())),
        ),
    ]
}

fn from_record(r: &Record) -> Result<TierMetadata> {
    let authority = record_get(r, "authority")
        .and_then(|v| v.as_str())
        .map(MemoryAuthority::parse)
        .unwrap_or_default();

    Ok(TierMetadata {
        message_id: record_get(r, "message_id")
            .and_then(|v| v.as_str())
            .context("missing message_id")?
            .to_string(),
        tier: record_get(r, "tier")
            .and_then(|v| v.as_str())
            .map(string_to_tier)
            .unwrap_or(MemoryTier::Hot),
        importance: record_get(r, "importance")
            .and_then(|v| v.as_f32())
            .context("missing importance")?,
        last_accessed: record_get(r, "last_accessed")
            .and_then(|v| v.as_i64())
            .context("missing last_accessed")?,
        access_count: record_get(r, "access_count")
            .and_then(|v| v.as_i32())
            .context("missing access_count")? as u32,
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        authority,
    })
}

fn tier_to_string(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::Hot => "hot",
        MemoryTier::Warm => "warm",
        MemoryTier::Cold => "cold",
        MemoryTier::MentalModel => "mental_model",
    }
}

fn string_to_tier(s: &str) -> MemoryTier {
    match s {
        "hot" => MemoryTier::Hot,
        "warm" => MemoryTier::Warm,
        "cold" => MemoryTier::Cold,
        _ => MemoryTier::Hot,
    }
}

/// Store for tier metadata
pub struct TierMetadataStore<
    B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase,
> {
    backend: Arc<B>,
}

impl<B: StorageBackend> TierMetadataStore<B> {
    /// Create a new tier metadata store
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend.ensure_table(TABLE_NAME, &table_schema()).await
    }

    /// Arrow schema for the tier metadata table, used by `LanceDatabase` table creation.
    pub fn tier_metadata_schema() -> Arc<arrow_schema::Schema> {
        Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("message_id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("tier", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("importance", arrow_schema::DataType::Float32, false),
            arrow_schema::Field::new("last_accessed", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("access_count", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("created_at", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("authority", arrow_schema::DataType::Utf8, false),
        ]))
    }

    /// Add tier metadata
    pub async fn add(&self, metadata: TierMetadata) -> Result<()> {
        self.backend
            .insert(TABLE_NAME, vec![to_record(&metadata)])
            .await
            .context("Failed to add tier metadata")
    }

    /// Add multiple metadata entries in batch
    pub async fn add_batch(&self, metadata: Vec<TierMetadata>) -> Result<()> {
        if metadata.is_empty() {
            return Ok(());
        }
        let records: Vec<Record> = metadata.iter().map(to_record).collect();
        self.backend
            .insert(TABLE_NAME, records)
            .await
            .context("Failed to add tier metadata batch")
    }

    /// Fetch metadata for a set of message IDs in a single query.
    pub async fn get_many(&self, message_ids: &[&str]) -> Result<HashMap<String, TierMetadata>> {
        if message_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let vals: Vec<FieldValue> = message_ids
            .iter()
            .map(|id| FieldValue::Utf8(Some(id.to_string())))
            .collect();
        let filter = Filter::In("message_id".into(), vals);

        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;
        let entries: Vec<TierMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        Ok(entries
            .into_iter()
            .map(|m| (m.message_id.clone(), m))
            .collect())
    }

    /// Get metadata by message ID
    pub async fn get(&self, message_id: &str) -> Result<Option<TierMetadata>> {
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

    /// Get all metadata
    pub async fn get_all(&self) -> Result<Vec<TierMetadata>> {
        let records = self.backend.query(TABLE_NAME, None, None).await?;
        records.iter().map(from_record).collect()
    }

    /// Get metadata by tier
    pub async fn get_by_tier(&self, tier: MemoryTier) -> Result<Vec<TierMetadata>> {
        let filter = Filter::Eq(
            "tier".into(),
            FieldValue::Utf8(Some(tier_to_string(tier).to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;
        records.iter().map(from_record).collect()
    }

    /// Update metadata (delete old and insert new)
    pub async fn update(&self, metadata: TierMetadata) -> Result<()> {
        self.delete(&metadata.message_id).await?;
        self.add(metadata).await
    }

    /// Delete metadata by message ID
    pub async fn delete(&self, message_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "message_id".into(),
            FieldValue::Utf8(Some(message_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete tier metadata")
    }

    /// Get count of metadata entries
    pub async fn count(&self) -> Result<usize> {
        self.backend.count(TABLE_NAME, None).await
    }

    /// Get count by tier
    pub async fn count_by_tier(&self, tier: MemoryTier) -> Result<usize> {
        let filter = Filter::Eq(
            "tier".into(),
            FieldValue::Utf8(Some(tier_to_string(tier).to_string())),
        );
        self.backend.count(TABLE_NAME, Some(&filter)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = table_schema();
        assert_eq!(
            schema.len(),
            7,
            "Schema must have 7 fields including authority"
        );
    }

    #[test]
    fn test_tier_conversion() {
        assert_eq!(tier_to_string(MemoryTier::Hot), "hot");
        assert_eq!(tier_to_string(MemoryTier::Warm), "warm");
        assert_eq!(tier_to_string(MemoryTier::Cold), "cold");

        assert_eq!(string_to_tier("hot"), MemoryTier::Hot);
        assert_eq!(string_to_tier("warm"), MemoryTier::Warm);
        assert_eq!(string_to_tier("cold"), MemoryTier::Cold);
        assert_eq!(string_to_tier("unknown"), MemoryTier::Hot);
    }

    #[test]
    fn test_tier_metadata_has_default_authority() {
        let meta = TierMetadata::new("m-1".to_string(), 0.5);
        assert_eq!(meta.authority, MemoryAuthority::Session);
    }

    #[test]
    fn test_tier_metadata_with_canonical_authority() {
        let meta = TierMetadata::with_authority("m-2".to_string(), 0.9, MemoryAuthority::Canonical);
        assert_eq!(meta.authority, MemoryAuthority::Canonical);
    }
}
