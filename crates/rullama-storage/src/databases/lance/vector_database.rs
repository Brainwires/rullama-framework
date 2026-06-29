//! [`VectorDatabase`] implementation for [`LanceDatabase`].

use anyhow::{Context, Result};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray, UInt32Array,
};
use futures::stream::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashMap;

use crate::databases::traits::{ChunkMetadata, DatabaseStats, SearchResult, VectorDatabase};
use crate::databases::types::{FieldValue, Filter};
use crate::glob_utils;

use super::arrow_convert::filter_to_sql;
use super::database::LanceDatabase;

#[async_trait::async_trait]
impl VectorDatabase for LanceDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        tracing::info!(
            "Initializing LanceDB with dimension {} at {}",
            dimension,
            self.db_path
        );

        let table_names = self
            .connection
            .table_names()
            .execute()
            .await
            .context("Failed to list tables")?;

        if table_names.contains(&self.rag_table_name) {
            tracing::info!("Table '{}' already exists", self.rag_table_name);
            return Ok(());
        }

        let schema = Self::create_rag_schema(dimension);
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![empty_batch].into_iter().map(Ok),
            schema.clone(),
        ));

        self.connection
            .create_table(&self.rag_table_name, batches)
            .execute()
            .await
            .context("Failed to create table")?;

        tracing::info!("Created table '{}'", self.rag_table_name);
        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        root_path: &str,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let dimension = embeddings[0].len();
        let schema = Self::create_rag_schema(dimension);

        let table = self.get_rag_table().await?;
        let current_count = table.count_rows(None).await.unwrap_or(0) as u64;

        let batch = Self::create_rag_record_batch(
            embeddings,
            metadata.clone(),
            contents.clone(),
            schema.clone(),
        )?;
        let count = batch.num_rows();

        let batches: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![batch].into_iter().map(Ok),
            schema,
        ));

        table
            .add(batches)
            .execute()
            .await
            .context("Failed to add records to table")?;

        self.get_or_create_bm25(root_path)?;

        let bm25_docs: Vec<_> = (0..count)
            .map(|i| {
                let id = current_count + i as u64;
                let string_id = format!("{}:{}", metadata[i].file_path, metadata[i].start_line);
                (
                    id,
                    string_id,
                    contents[i].clone(),
                    metadata[i].file_path.clone(),
                )
            })
            .collect();

        let hash = Self::hash_root_path(root_path);
        let bm25_indexes = self
            .bm25_indexes
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

        if let Some(bm25) = bm25_indexes.get(&hash) {
            bm25.add_documents(bm25_docs)
                .context("Failed to add documents to BM25 index")?;
        }
        drop(bm25_indexes);

        tracing::info!(
            "Stored {} embeddings with BM25 indexing for root: {}",
            count,
            root_path
        );
        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_rag_table().await?;

        if hybrid {
            // Vector and BM25 use separate limits.  Vector uses a 3× multiplier
            // (semantic proximity decays quickly with rank so fewer are needed).
            // BM25 uses a 10× multiplier with a 50-result floor so that rare
            // exact-match terms (e.g. proper names) are not prematurely cut off
            // before RRF fusion — BM25-only hits already score ~half of
            // vector+BM25 hits in RRF, so we need more of them in the candidate
            // pool to keep all occurrences above the final limit cutoff.
            let vector_search_limit = limit * 3;
            let bm25_search_limit = (limit * 10).max(50);

            let query = table
                .vector_search(query_vector)
                .context("Failed to create vector search")?
                .limit(vector_search_limit);

            let stream = if let Some(ref project_name) = project {
                query
                    .only_if(filter_to_sql(&Filter::Eq(
                        "project".into(),
                        FieldValue::Utf8(Some(project_name.clone())),
                    )))
                    .execute()
                    .await
                    .context("Failed to execute search")?
            } else {
                query.execute().await.context("Failed to execute search")?
            };

            let results: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .context("Failed to collect search results")?;

            let mut vector_results = Vec::new();
            let mut original_scores: HashMap<String, (f32, Option<f32>)> = HashMap::new();
            // Pre-build lookup: string_id → (batch_index, row_index) for post-fusion resolution
            let mut id_to_location: HashMap<String, (usize, usize)> = HashMap::new();

            for (batch_idx, batch) in results.iter().enumerate() {
                let distance_array = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?;

                let id_array = batch
                    .column_by_name("id")
                    .context("Missing id column in vector results")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid id column type")?;

                for i in 0..batch.num_rows() {
                    let distance = distance_array.value(i);
                    let score = 1.0 / (1.0 + distance);
                    let string_id = id_array.value(i).to_string();
                    vector_results.push((string_id.clone(), score));
                    original_scores.insert(string_id.clone(), (score, None));
                    id_to_location.insert(string_id, (batch_idx, i));
                }
            }

            let bm25_indexes = self
                .bm25_indexes
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

            let mut all_bm25_results = Vec::new();
            for (root_hash, bm25) in bm25_indexes.iter() {
                tracing::debug!("Searching BM25 index for root hash: {}", root_hash);
                let bm25_results = bm25
                    .search(query_text, bm25_search_limit)
                    .context("Failed to search BM25 index")?;

                for result in &bm25_results {
                    original_scores
                        .entry(result.string_id.clone())
                        .and_modify(|e| e.1 = Some(result.score))
                        .or_insert((0.0, Some(result.score)));
                }

                all_bm25_results.extend(bm25_results);
            }
            drop(bm25_indexes);

            // Use a wider internal RRF limit so BM25-only hits are not squeezed
            // out by vector+BM25 hits that score ~2× higher in RRF.
            // The caller's limit is enforced at the end of the result-building loop.
            let rrf_limit = (limit * 2).max(20);
            let combined = self
                .scorer
                .fuse(vector_results, all_bm25_results, rrf_limit);

            let mut search_results = Vec::new();

            for (string_id, combined_score) in combined {
                let Some(&(batch_idx, idx)) = id_to_location.get(&string_id) else {
                    // BM25-only hit not in vector results — cannot materialize
                    // without a separate LanceDB lookup (acceptable trade-off for now)
                    tracing::debug!(
                        "BM25-only hit '{}' not in vector result batches — skipping",
                        string_id
                    );
                    continue;
                };

                let batch = &results[batch_idx];

                let file_path_array = batch
                    .column_by_name("file_path")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let root_path_array = batch
                    .column_by_name("root_path")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let start_line_array = batch
                    .column_by_name("start_line")
                    .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
                let end_line_array = batch
                    .column_by_name("end_line")
                    .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
                let language_array = batch
                    .column_by_name("language")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let content_array = batch
                    .column_by_name("content")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let project_array = batch
                    .column_by_name("project")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let indexed_at_array = batch
                    .column_by_name("indexed_at")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());

                if let (
                    Some(fp),
                    Some(rp),
                    Some(sl),
                    Some(el),
                    Some(lang),
                    Some(cont),
                    Some(proj),
                ) = (
                    file_path_array,
                    root_path_array,
                    start_line_array,
                    end_line_array,
                    language_array,
                    content_array,
                    project_array,
                ) {
                    let (vector_score, keyword_score) = original_scores
                        .get(&string_id)
                        .copied()
                        .unwrap_or((0.0, None));

                    let passes_filter =
                        vector_score >= min_score || keyword_score.is_some_and(|k| k >= min_score);

                    if passes_filter {
                        let result_root_path = if rp.is_null(idx) {
                            None
                        } else {
                            Some(rp.value(idx).to_string())
                        };

                        if let Some(ref filter_path) = root_path
                            && result_root_path.as_ref() != Some(filter_path)
                        {
                            continue;
                        }

                        search_results.push(SearchResult {
                            score: combined_score,
                            vector_score,
                            keyword_score,
                            file_path: fp.value(idx).to_string(),
                            root_path: result_root_path,
                            start_line: sl.value(idx) as usize,
                            end_line: el.value(idx) as usize,
                            language: lang.value(idx).to_string(),
                            content: cont.value(idx).to_string(),
                            project: if proj.is_null(idx) {
                                None
                            } else {
                                Some(proj.value(idx).to_string())
                            },
                            indexed_at: indexed_at_array
                                .and_then(|ia| ia.value(idx).parse::<i64>().ok())
                                .unwrap_or(0),
                        });
                    }
                }
            }

            // Enforce caller's limit after the wider RRF pass
            search_results.truncate(limit);

            Ok(search_results)
        } else {
            // Pure vector search
            let query = table
                .vector_search(query_vector)
                .context("Failed to create vector search")?
                .limit(limit);

            let stream = if let Some(ref project_name) = project {
                query
                    .only_if(filter_to_sql(&Filter::Eq(
                        "project".into(),
                        FieldValue::Utf8(Some(project_name.clone())),
                    )))
                    .execute()
                    .await
                    .context("Failed to execute search")?
            } else {
                query.execute().await.context("Failed to execute search")?
            };

            let results: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .context("Failed to collect search results")?;

            let mut search_results = Vec::new();

            for batch in results {
                let file_path_array = batch
                    .column_by_name("file_path")
                    .context("Missing file_path column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid file_path type")?;

                let root_path_array = batch
                    .column_by_name("root_path")
                    .context("Missing root_path column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid root_path type")?;

                let start_line_array = batch
                    .column_by_name("start_line")
                    .context("Missing start_line column")?
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .context("Invalid start_line type")?;

                let end_line_array = batch
                    .column_by_name("end_line")
                    .context("Missing end_line column")?
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .context("Invalid end_line type")?;

                let language_array = batch
                    .column_by_name("language")
                    .context("Missing language column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid language type")?;

                let content_array = batch
                    .column_by_name("content")
                    .context("Missing content column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid content type")?;

                let project_array = batch
                    .column_by_name("project")
                    .context("Missing project column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid project type")?;

                let distance_array = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?;

                let indexed_at_array = batch
                    .column_by_name("indexed_at")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());

                for i in 0..batch.num_rows() {
                    let distance = distance_array.value(i);
                    let score = 1.0 / (1.0 + distance);

                    if score >= min_score {
                        let result_root_path = if root_path_array.is_null(i) {
                            None
                        } else {
                            Some(root_path_array.value(i).to_string())
                        };

                        if let Some(ref filter_path) = root_path
                            && result_root_path.as_ref() != Some(filter_path)
                        {
                            continue;
                        }

                        search_results.push(SearchResult {
                            score,
                            vector_score: score,
                            keyword_score: None,
                            file_path: file_path_array.value(i).to_string(),
                            root_path: result_root_path,
                            start_line: start_line_array.value(i) as usize,
                            end_line: end_line_array.value(i) as usize,
                            language: language_array.value(i).to_string(),
                            content: content_array.value(i).to_string(),
                            project: if project_array.is_null(i) {
                                None
                            } else {
                                Some(project_array.value(i).to_string())
                            },
                            indexed_at: indexed_at_array
                                .and_then(|ia| ia.value(i).parse::<i64>().ok())
                                .unwrap_or(0),
                        });
                    }
                }
            }

            Ok(search_results)
        }
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        let search_limit = limit * 3;

        let mut results = self
            .search(
                query_vector,
                query_text,
                search_limit,
                min_score,
                project,
                root_path,
                hybrid,
            )
            .await?;

        results.retain(|result| {
            if !file_extensions.is_empty() {
                let has_extension = file_extensions
                    .iter()
                    .any(|ext| result.file_path.ends_with(&format!(".{}", ext)));
                if !has_extension {
                    return false;
                }
            }

            if !languages.is_empty() && !languages.contains(&result.language) {
                return false;
            }

            if !path_patterns.is_empty()
                && !glob_utils::matches_any_pattern(&result.file_path, &path_patterns)
            {
                return false;
            }

            true
        });

        results.truncate(limit);
        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        {
            let bm25_indexes = self
                .bm25_indexes
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

            for (root_hash, bm25) in bm25_indexes.iter() {
                bm25.delete_by_file_path(file_path)
                    .context("Failed to delete from BM25 index")?;
                tracing::debug!(
                    "Deleted BM25 entries for file: {} in index: {}",
                    file_path,
                    root_hash
                );
            }
        }

        let table = self.get_rag_table().await?;
        let filter = filter_to_sql(&Filter::Eq(
            "file_path".into(),
            FieldValue::Utf8(Some(file_path.to_string())),
        ));
        table
            .delete(&filter)
            .await
            .context("Failed to delete records")?;

        tracing::info!("Deleted embeddings for file: {}", file_path);
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        self.connection
            .drop_table(&self.rag_table_name, &[])
            .await
            .context("Failed to drop table")?;

        let bm25_indexes = self
            .bm25_indexes
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

        for (root_hash, bm25) in bm25_indexes.iter() {
            bm25.clear().context("Failed to clear BM25 index")?;
            tracing::info!("Cleared BM25 index for root hash: {}", root_hash);
        }
        drop(bm25_indexes);

        tracing::info!("Cleared all embeddings and all per-project BM25 indexes");
        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let table = self.get_rag_table().await?;

        let count_result = table
            .count_rows(None)
            .await
            .context("Failed to count rows")?;

        let stream = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "language".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query languages")?;

        let query_result: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect language data")?;

        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for batch in query_result {
            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            for i in 0..batch.num_rows() {
                let language = language_array.value(i);
                *language_counts.entry(language.to_string()).or_insert(0) += 1;
            }
        }

        let mut language_breakdown: Vec<(String, usize)> = language_counts.into_iter().collect();
        language_breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(DatabaseStats {
            total_points: count_result,
            total_vectors: count_result,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let table = self.get_rag_table().await?;
        let filter = filter_to_sql(&Filter::Eq(
            "root_path".into(),
            FieldValue::Utf8(Some(root_path.to_string())),
        ));
        let count = table
            .count_rows(Some(filter))
            .await
            .context("Failed to count rows by root path")?;
        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let table = self.get_rag_table().await?;
        let filter = filter_to_sql(&Filter::Eq(
            "root_path".into(),
            FieldValue::Utf8(Some(root_path.to_string())),
        ));
        let stream = table
            .query()
            .only_if(filter)
            .select(lancedb::query::Select::Columns(vec![
                "file_path".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query indexed files")?;

        let results: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect file paths")?;

        let mut file_paths = std::collections::HashSet::new();
        for batch in results {
            let file_path_array = batch
                .column_by_name("file_path")
                .context("Missing file_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid file_path type")?;

            for i in 0..batch.num_rows() {
                file_paths.insert(file_path_array.value(i).to_string());
            }
        }

        Ok(file_paths.into_iter().collect())
    }

    async fn search_with_embeddings(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        hybrid: bool,
    ) -> Result<(Vec<SearchResult>, Vec<Vec<f32>>)> {
        let results = self
            .search(
                query_vector,
                query_text,
                limit,
                min_score,
                project,
                root_path,
                hybrid,
            )
            .await?;

        if results.is_empty() {
            return Ok((results, Vec::new()));
        }

        let table = self.get_rag_table().await?;
        let mut embeddings = Vec::with_capacity(results.len());

        for result in &results {
            let filter = filter_to_sql(&Filter::And(vec![
                Filter::Eq(
                    "file_path".into(),
                    FieldValue::Utf8(Some(result.file_path.clone())),
                ),
                Filter::Eq(
                    "start_line".into(),
                    FieldValue::UInt32(Some(result.start_line as u32)),
                ),
            ]));
            let stream = table
                .query()
                .only_if(filter)
                .select(lancedb::query::Select::Columns(vec!["vector".to_string()]))
                .limit(1)
                .execute()
                .await
                .context("Failed to query embedding vector")?;

            let batches: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .context("Failed to collect embedding vector")?;

            let mut found = false;
            for batch in &batches {
                if batch.num_rows() > 0
                    && let Some(vector_col) = batch.column_by_name("vector")
                    && let Some(fsl) = vector_col.as_any().downcast_ref::<FixedSizeListArray>()
                {
                    let values = fsl
                        .value(0)
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .map(|a| a.values().to_vec())
                        .unwrap_or_default();
                    embeddings.push(values);
                    found = true;
                    break;
                }
            }
            if !found {
                embeddings.push(Vec::new());
            }
        }

        Ok((results, embeddings))
    }
}
