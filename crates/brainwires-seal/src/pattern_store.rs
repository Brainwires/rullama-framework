//! SEAL Pattern Store
//!
//! Provides persistent storage for learned query patterns used by the SEAL
//! (Self-Evolving Agentic Learning) system. Patterns are stored in LanceDB
//! with embeddings for semantic similarity matching.

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::LanceDatabase;

use super::learning::QueryPattern;
use super::query_core::QuestionType;

/// LanceDB extension trait: SEAL-patterns table management.
///
/// Lives next to [`PatternStore`] because the SEAL patterns table is its
/// only consumer; previously declared in the CLI's `crate::storage`
/// aggregator. Implemented for [`LanceDatabase`] below.
pub trait LanceDatabaseExt {
    /// Ensure the SEAL patterns table exists, creating it if missing.
    fn ensure_seal_patterns_table(
        &self,
        embedding_dim: usize,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Open the SEAL patterns table.
    fn seal_patterns_table(
        &self,
    ) -> impl std::future::Future<Output = Result<lancedb::Table>> + Send;

    /// Schema for the SEAL patterns table at the given embedding dimension.
    fn seal_patterns_schema(dimension: usize) -> Arc<Schema>;
}

impl LanceDatabaseExt for LanceDatabase {
    async fn ensure_seal_patterns_table(&self, embedding_dim: usize) -> Result<()> {
        let table_name = "seal_patterns";
        let table_names = self.connection().table_names().execute().await?;

        if table_names.contains(&table_name.to_string()) {
            return Ok(());
        }

        let schema = Self::seal_patterns_schema(embedding_dim);
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema.clone());

        self.connection()
            .create_table(
                table_name,
                Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>,
            )
            .execute()
            .await
            .context("Failed to create seal_patterns table")?;

        Ok(())
    }

    async fn seal_patterns_table(&self) -> Result<lancedb::Table> {
        self.connection()
            .open_table("seal_patterns")
            .execute()
            .await
            .context("Failed to open seal_patterns table")
    }

    fn seal_patterns_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("pattern_id", DataType::Utf8, false),
            Field::new("question_type", DataType::Utf8, false),
            Field::new("template", DataType::Utf8, false),
            Field::new("entity_types", DataType::Utf8, false),
            Field::new("success_count", DataType::Int32, false),
            Field::new("failure_count", DataType::Int32, false),
            Field::new("avg_results", DataType::Float32, false),
            Field::new("last_used", DataType::Int64, false),
            Field::new("created_at", DataType::Int64, false),
        ]))
    }
}

/// Metadata for a persisted SEAL pattern.
#[derive(Debug, Clone)]
pub struct PatternMetadata {
    /// Stable identifier for the pattern row.
    pub pattern_id: String,
    /// Serialized [`super::query_core::QuestionType`] (tag string).
    pub question_type: String,
    /// Pattern template string (S-expression-like form).
    pub template: String,
    /// Entity-type tags this pattern was derived from.
    pub entity_types: Vec<String>,
    /// Times this pattern produced a useful result.
    pub success_count: u32,
    /// Times this pattern failed to produce a useful result.
    pub failure_count: u32,
    /// Average number of results returned when this pattern was used.
    pub avg_results: f32,
    /// Last-used timestamp (Unix seconds).
    pub last_used: i64,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
}

impl PatternMetadata {
    /// Calculate pattern reliability score
    pub fn reliability(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.0;
        }
        self.success_count as f32 / total as f32
    }
}

/// Store for SEAL learned patterns with semantic search
pub struct PatternStore {
    client: Arc<LanceDatabase>,
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl PatternStore {
    /// Create a new pattern store
    pub fn new(client: Arc<LanceDatabase>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self { client, embeddings }
    }

    /// Save a new pattern or update an existing one
    pub async fn save_pattern(&self, pattern: &QueryPattern, template: &str) -> Result<()> {
        // Generate embedding from template
        let embedding = self.embeddings.embed(template)?;

        // Get current timestamp
        let now = chrono::Utc::now().timestamp();

        // Check if pattern exists
        if self.get_pattern(&pattern.id).await?.is_some() {
            // Update existing pattern
            self.update_pattern(pattern, now).await
        } else {
            // Insert new pattern
            self.insert_pattern(pattern, template, &embedding, now)
                .await
        }
    }

    /// Insert a new pattern
    async fn insert_pattern(
        &self,
        pattern: &QueryPattern,
        template: &str,
        embedding: &[f32],
        timestamp: i64,
    ) -> Result<()> {
        let table = self.client.seal_patterns_table().await?;
        let dimension = self.embeddings.dimension();

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("pattern_id", DataType::Utf8, false),
            Field::new("question_type", DataType::Utf8, false),
            Field::new("template", DataType::Utf8, false),
            Field::new("entity_types", DataType::Utf8, false),
            Field::new("success_count", DataType::Int32, false),
            Field::new("failure_count", DataType::Int32, false),
            Field::new("avg_results", DataType::Float32, false),
            Field::new("last_used", DataType::Int64, false),
            Field::new("created_at", DataType::Int64, false),
        ]));

        // Create embedding array
        let embedding_array = Float32Array::from(embedding.to_vec());
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = FixedSizeListArray::new(
            vector_field,
            dimension as i32,
            Arc::new(embedding_array),
            None,
        );

        let pattern_id = StringArray::from(vec![pattern.id.as_str()]);
        let question_type =
            StringArray::from(vec![format!("{:?}", pattern.question_type).as_str()]);
        let template_arr = StringArray::from(vec![template]);
        let entity_types = StringArray::from(vec![serde_json::to_string(&pattern.required_types)?]);
        let success_count = Int32Array::from(vec![pattern.success_count as i32]);
        let failure_count = Int32Array::from(vec![pattern.failure_count as i32]);
        let avg_results = Float32Array::from(vec![pattern.avg_results]);
        let last_used = Int64Array::from(vec![timestamp]);
        let created_at = Int64Array::from(vec![timestamp]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(vectors) as ArrayRef,
                Arc::new(pattern_id),
                Arc::new(question_type),
                Arc::new(template_arr),
                Arc::new(entity_types),
                Arc::new(success_count),
                Arc::new(failure_count),
                Arc::new(avg_results),
                Arc::new(last_used),
                Arc::new(created_at),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to insert pattern")?;

        Ok(())
    }

    /// Update an existing pattern's statistics
    async fn update_pattern(&self, pattern: &QueryPattern, timestamp: i64) -> Result<()> {
        let table = self.client.seal_patterns_table().await?;

        // LanceDB doesn't support direct updates, so we need to delete and re-insert
        // Delete existing record
        table
            .delete(&format!("pattern_id = '{}'", pattern.id))
            .await
            .context("Failed to delete old pattern")?;

        // Re-insert with updated stats
        let embedding = self.embeddings.embed(&pattern.template)?;
        self.insert_pattern(pattern, &pattern.template, &embedding, timestamp)
            .await
    }

    /// Get a pattern by ID
    pub async fn get_pattern(&self, pattern_id: &str) -> Result<Option<PatternMetadata>> {
        let table = self.client.seal_patterns_table().await?;

        let filter = format!("pattern_id = '{}'", pattern_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query pattern")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        if batches.is_empty() {
            return Ok(None);
        }

        let batch = &batches[0];
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        Ok(Some(self.batch_to_metadata(batch, 0)?))
    }

    /// Search for similar patterns using semantic similarity
    pub async fn search_similar(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(PatternMetadata, f32)>> {
        let embedding = self.embeddings.embed(query)?;
        let table = self.client.seal_patterns_table().await?;

        let stream = table
            .vector_search(embedding)
            .context("Failed to create vector search")?
            .limit(limit)
            .execute()
            .await
            .context("Failed to execute vector search")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut patterns = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                // Get distance score (convert to similarity)
                let distance_col = batch
                    .column_by_name("_distance")
                    .context("Missing distance column")?;
                let distance = distance_col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid distance type")?
                    .value(i);

                // Convert L2 distance to similarity score (0-1)
                let similarity = 1.0 / (1.0 + distance);

                if similarity >= min_score {
                    let metadata = self.batch_to_metadata(batch, i)?;
                    patterns.push((metadata, similarity));
                }
            }
        }

        // Sort by similarity descending
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(patterns)
    }

    /// Get patterns by question type
    pub async fn get_by_question_type(
        &self,
        question_type: &QuestionType,
    ) -> Result<Vec<PatternMetadata>> {
        let table = self.client.seal_patterns_table().await?;
        let type_str = format!("{:?}", question_type);

        let filter = format!("question_type = '{}'", type_str);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query patterns by type")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut patterns = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                patterns.push(self.batch_to_metadata(batch, i)?);
            }
        }

        Ok(patterns)
    }

    /// Get high-reliability patterns (for learning context)
    pub async fn get_reliable_patterns(
        &self,
        min_reliability: f32,
    ) -> Result<Vec<PatternMetadata>> {
        let table = self.client.seal_patterns_table().await?;

        let stream = table
            .query()
            .execute()
            .await
            .context("Failed to query all patterns")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut patterns = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                let metadata = self.batch_to_metadata(batch, i)?;
                if metadata.reliability() >= min_reliability {
                    patterns.push(metadata);
                }
            }
        }

        // Sort by reliability descending
        patterns.sort_by(|a, b| {
            b.reliability()
                .partial_cmp(&a.reliability())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(patterns)
    }

    /// Delete patterns with low reliability (cleanup)
    pub async fn prune_low_reliability(
        &self,
        min_reliability: f32,
        min_uses: u32,
    ) -> Result<usize> {
        let table = self.client.seal_patterns_table().await?;

        // Get all patterns
        let stream = table
            .query()
            .execute()
            .await
            .context("Failed to query patterns for pruning")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut to_delete = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                let metadata = self.batch_to_metadata(batch, i)?;
                let total_uses = metadata.success_count + metadata.failure_count;

                // Only prune patterns that have been used enough times
                if total_uses >= min_uses && metadata.reliability() < min_reliability {
                    to_delete.push(metadata.pattern_id);
                }
            }
        }

        // Delete low-reliability patterns
        for pattern_id in &to_delete {
            table
                .delete(&format!("pattern_id = '{}'", pattern_id))
                .await
                .context("Failed to delete low-reliability pattern")?;
        }

        Ok(to_delete.len())
    }

    /// Get pattern count
    pub async fn count(&self) -> Result<usize> {
        let table = self.client.seal_patterns_table().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }

    /// Convert a record batch row to PatternMetadata
    fn batch_to_metadata(&self, batch: &RecordBatch, row: usize) -> Result<PatternMetadata> {
        let pattern_id = batch
            .column_by_name("pattern_id")
            .context("Missing pattern_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid pattern_id type")?
            .value(row)
            .to_string();

        let question_type = batch
            .column_by_name("question_type")
            .context("Missing question_type")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid question_type type")?
            .value(row)
            .to_string();

        let template = batch
            .column_by_name("template")
            .context("Missing template")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid template type")?
            .value(row)
            .to_string();

        let entity_types_json = batch
            .column_by_name("entity_types")
            .context("Missing entity_types")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid entity_types type")?
            .value(row);

        let entity_types: Vec<String> = serde_json::from_str(entity_types_json).unwrap_or_default();

        let success_count = batch
            .column_by_name("success_count")
            .context("Missing success_count")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("Invalid success_count type")?
            .value(row) as u32;

        let failure_count = batch
            .column_by_name("failure_count")
            .context("Missing failure_count")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("Invalid failure_count type")?
            .value(row) as u32;

        let avg_results = batch
            .column_by_name("avg_results")
            .context("Missing avg_results")?
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Invalid avg_results type")?
            .value(row);

        let last_used = batch
            .column_by_name("last_used")
            .context("Missing last_used")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("Invalid last_used type")?
            .value(row);

        let created_at = batch
            .column_by_name("created_at")
            .context("Missing created_at")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("Invalid created_at type")?
            .value(row);

        Ok(PatternMetadata {
            pattern_id,
            question_type,
            template,
            entity_types,
            success_count,
            failure_count,
            avg_results,
            last_used,
            created_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::graph::EntityType;
    use brainwires_storage::databases::VectorDatabase;
    use tempfile::TempDir;

    async fn create_test_store() -> (PatternStore, TempDir) {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.lance");

        let client = Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap());
        client.initialize(384).await.unwrap();
        client.ensure_seal_patterns_table(384).await.unwrap();

        let embeddings = Arc::new(CachedEmbeddingProvider::new().unwrap());
        let store = PatternStore::new(client, embeddings);

        (store, temp)
    }

    fn create_test_pattern() -> QueryPattern {
        QueryPattern {
            id: "test-pattern-1".to_string(),
            question_type: QuestionType::Dependency,
            template: "What uses {entity}?".to_string(),
            required_types: vec![EntityType::File],
            success_count: 5,
            failure_count: 1,
            avg_results: 3.5,
            created_at: chrono::Utc::now().timestamp(),
            last_used_at: chrono::Utc::now().timestamp(),
        }
    }

    #[tokio::test]
    async fn test_save_and_get_pattern() {
        if std::env::var("TEST_EMBED_NETWORK").ok().as_deref() != Some("1") {
            eprintln!("skipping: set TEST_EMBED_NETWORK=1 to run (needs FastEmbed model)");
            return;
        }
        let (store, _temp) = create_test_store().await;
        let pattern = create_test_pattern();

        // Save pattern
        store
            .save_pattern(&pattern, "What uses {entity}?")
            .await
            .unwrap();

        // Retrieve pattern
        let retrieved = store.get_pattern(&pattern.id).await.unwrap();
        assert!(retrieved.is_some());

        let metadata = retrieved.unwrap();
        assert_eq!(metadata.pattern_id, pattern.id);
        assert_eq!(metadata.success_count, pattern.success_count);
        assert_eq!(metadata.failure_count, pattern.failure_count);
    }

    #[tokio::test]
    async fn test_reliability_calculation() {
        let metadata = PatternMetadata {
            pattern_id: "test".to_string(),
            question_type: "Definition".to_string(),
            template: "What is {entity}?".to_string(),
            entity_types: vec!["File".to_string()],
            success_count: 8,
            failure_count: 2,
            avg_results: 1.0,
            last_used: 0,
            created_at: 0,
        };

        assert!((metadata.reliability() - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_count_patterns() {
        if std::env::var("TEST_EMBED_NETWORK").ok().as_deref() != Some("1") {
            eprintln!("skipping: set TEST_EMBED_NETWORK=1 to run (needs FastEmbed model)");
            return;
        }
        let (store, _temp) = create_test_store().await;

        // Initially empty
        assert_eq!(store.count().await.unwrap(), 0);

        // Add a pattern
        let pattern = create_test_pattern();
        store
            .save_pattern(&pattern, "What uses {entity}?")
            .await
            .unwrap();

        // Should have one pattern
        assert_eq!(store.count().await.unwrap(), 1);
    }
}
