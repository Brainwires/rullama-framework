//! Milvus vector database backend using the REST API v2.
//!
//! Connects to a running Milvus instance via HTTP and implements the
//! [`VectorDatabase`] trait for code-embedding storage and hybrid search.
//! All communication uses the Milvus RESTful API v2 endpoints through
//! `reqwest`.
//!
//! # Feature flag
//!
//! Requires `milvus-backend`.

use crate::databases::bm25_helpers::{self, SharedIdfStats};
use crate::databases::traits::VectorDatabase;
use crate::glob_utils;
use anyhow::{Context, Result};
use brainwires_core::{ChunkMetadata, DatabaseStats, SearchResult};
use serde_json::{Value, json};

const DEFAULT_URL: &str = "http://localhost:19530";
const DEFAULT_COLLECTION: &str = "code_embeddings";

/// Maximum number of entities per insert batch (Milvus limit).
const INSERT_BATCH_SIZE: usize = 1000;

/// Maximum number of entities returned by a single query request.
const QUERY_LIMIT: usize = 16384;

/// Milvus-backed vector database for code embeddings.
///
/// Uses the Milvus REST API v2 for all operations. Requires a running Milvus
/// instance (standalone or cluster) reachable at the configured URL.
pub struct MilvusDatabase {
    client: reqwest::Client,
    base_url: String,
    collection_name: String,
    idf_stats: SharedIdfStats,
}

impl MilvusDatabase {
    /// Create a new Milvus client with default local configuration.
    ///
    /// Connects to `http://localhost:19530` with collection `code_embeddings`.
    pub fn new() -> Self {
        Self::with_config(DEFAULT_URL, DEFAULT_COLLECTION)
    }

    /// Create a new Milvus client with a custom URL.
    ///
    /// Uses the default collection name `code_embeddings`.
    pub fn with_url(url: &str) -> Self {
        Self::with_config(url, DEFAULT_COLLECTION)
    }

    /// Create a new Milvus client with full configuration.
    pub fn with_config(url: &str, collection: &str) -> Self {
        tracing::info!(
            "Creating Milvus client: url={}, collection={}",
            url,
            collection
        );

        Self {
            client: reqwest::Client::new(),
            base_url: url.trim_end_matches('/').to_string(),
            collection_name: collection.to_string(),
            idf_stats: bm25_helpers::new_shared_idf_stats(),
        }
    }

    /// Get the default Milvus URL (public for CLI version info).
    pub fn default_url() -> &'static str {
        DEFAULT_URL
    }

    // ── REST helpers ────────────────────────────────────────────────────

    /// POST a JSON body to a Milvus REST v2 endpoint and return the parsed
    /// response.
    ///
    /// Returns an error if the HTTP request fails, the response is not valid
    /// JSON, or the response contains a non-zero error code.
    async fn api_post(&self, path: &str, body: Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        tracing::debug!("Milvus POST {} body={}", path, body);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("HTTP POST to {} failed", url))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .context("Failed to read Milvus response body")?;

        let parsed: Value = serde_json::from_str(&text)
            .with_context(|| format!("Milvus returned non-JSON (HTTP {}): {}", status, text))?;

        // Milvus REST v2 uses a `code` field — 0 (or 200) means success.
        if let Some(code) = parsed.get("code").and_then(|c| c.as_i64())
            && code != 0
            && code != 200
        {
            let message = parsed
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!(
                "Milvus API error on {}: code={}, message={}",
                path,
                code,
                message
            );
        }

        Ok(parsed)
    }

    /// Escape a string value for use in a Milvus filter expression.
    pub(crate) fn escape_filter_value(value: &str) -> String {
        value.replace('\\', "\\\\").replace('"', "\\\"")
    }

    // ── IDF refresh ─────────────────────────────────────────────────────

    /// Refresh IDF statistics by scanning existing documents in the
    /// collection.
    async fn refresh_idf_stats(&self) -> Result<()> {
        tracing::info!("Refreshing IDF statistics from Milvus...");

        let body = json!({
            "collectionName": self.collection_name,
            "filter": "",
            "outputFields": ["content"],
            "limit": QUERY_LIMIT
        });

        let resp = self.api_post("/v2/vectordb/entities/query", body).await;

        let documents: Vec<String> = match resp {
            Ok(val) => val
                .get("data")
                .and_then(|d| d.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            item.get("content")
                                .and_then(|c| c.as_str())
                                .map(String::from)
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(e) => {
                tracing::warn!("Failed to fetch documents for IDF refresh: {}", e);
                return Ok(());
            }
        };

        tracing::info!("Refreshing IDF stats from {} documents", documents.len());
        bm25_helpers::update_idf_stats(&self.idf_stats, &documents).await;

        Ok(())
    }

    /// Check whether the configured collection already exists.
    async fn collection_exists(&self) -> Result<bool> {
        let body = json!({ "collectionName": self.collection_name });
        let resp = self
            .api_post("/v2/vectordb/collections/has", body)
            .await
            .context("Failed to check if Milvus collection exists")?;

        Ok(resp
            .get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool())
            .unwrap_or(false))
    }

    /// Build a Milvus filter expression string from optional filter
    /// parameters.
    fn build_filter_expr(
        &self,
        project: &Option<String>,
        root_path: &Option<String>,
        file_extensions: &[String],
        languages: &[String],
    ) -> String {
        let mut clauses: Vec<String> = Vec::new();

        if let Some(proj) = project {
            clauses.push(format!(
                "project == \"{}\"",
                Self::escape_filter_value(proj)
            ));
        }

        if let Some(rp) = root_path {
            clauses.push(format!(
                "root_path == \"{}\"",
                Self::escape_filter_value(rp)
            ));
        }

        if !file_extensions.is_empty() {
            let items: Vec<String> = file_extensions
                .iter()
                .map(|e| format!("\"{}\"", Self::escape_filter_value(e)))
                .collect();
            clauses.push(format!("extension in [{}]", items.join(", ")));
        }

        if !languages.is_empty() {
            let items: Vec<String> = languages
                .iter()
                .map(|l| format!("\"{}\"", Self::escape_filter_value(l)))
                .collect();
            clauses.push(format!("language in [{}]", items.join(", ")));
        }

        clauses.join(" and ")
    }
}

// ── VectorDatabase implementation ───────────────────────────────────────

#[async_trait::async_trait]
impl VectorDatabase for MilvusDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        if self.collection_exists().await? {
            tracing::info!(
                "Milvus collection '{}' already exists",
                self.collection_name
            );
            // Ensure the collection is loaded.
            let load_body = json!({ "collectionName": self.collection_name });
            self.api_post("/v2/vectordb/collections/load", load_body)
                .await
                .context("Failed to load existing Milvus collection")?;
            return Ok(());
        }

        tracing::info!(
            "Creating Milvus collection '{}' with dimension {}",
            self.collection_name,
            dimension
        );

        let create_body = json!({
            "collectionName": self.collection_name,
            "schema": {
                "autoId": true,
                "enableDynamicField": true,
                "fields": [
                    {
                        "fieldName": "id",
                        "dataType": "Int64",
                        "isPrimary": true,
                        "autoID": true
                    },
                    {
                        "fieldName": "embedding",
                        "dataType": "FloatVector",
                        "elementTypeParams": { "dim": dimension }
                    },
                    {
                        "fieldName": "file_path",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 2048 }
                    },
                    {
                        "fieldName": "root_path",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 2048 }
                    },
                    {
                        "fieldName": "project",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 512 }
                    },
                    {
                        "fieldName": "start_line",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "end_line",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "language",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 128 }
                    },
                    {
                        "fieldName": "extension",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 32 }
                    },
                    {
                        "fieldName": "file_hash",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 128 }
                    },
                    {
                        "fieldName": "indexed_at",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "content",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": 65535 }
                    }
                ]
            },
            "indexParams": [
                {
                    "fieldName": "embedding",
                    "indexName": "embedding_index",
                    "metricType": "COSINE"
                }
            ]
        });

        self.api_post("/v2/vectordb/collections/create", create_body)
            .await
            .context("Failed to create Milvus collection")?;

        // Load the collection into memory so it is queryable.
        let load_body = json!({ "collectionName": self.collection_name });
        self.api_post("/v2/vectordb/collections/load", load_body)
            .await
            .context("Failed to load Milvus collection after creation")?;

        tracing::info!(
            "Milvus collection '{}' created and loaded",
            self.collection_name
        );

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

        let total = embeddings.len();
        tracing::debug!("Storing {} embeddings in Milvus", total);

        let mut inserted: usize = 0;

        // Build all data rows first, then batch-insert.
        let rows: Vec<Value> = embeddings
            .into_iter()
            .zip(metadata)
            .zip(contents)
            .map(|((emb, meta), content)| {
                json!({
                    "embedding": emb,
                    "file_path": meta.file_path,
                    "root_path": meta.root_path.as_deref().unwrap_or(root_path),
                    "project": meta.project.as_deref().unwrap_or(""),
                    "start_line": meta.start_line as i64,
                    "end_line": meta.end_line as i64,
                    "language": meta.language.as_deref().unwrap_or("Unknown"),
                    "extension": meta.extension.as_deref().unwrap_or(""),
                    "file_hash": meta.file_hash,
                    "indexed_at": meta.indexed_at,
                    "content": content
                })
            })
            .collect();

        for chunk in rows.chunks(INSERT_BATCH_SIZE) {
            let body = json!({
                "collectionName": self.collection_name,
                "data": chunk
            });

            let resp = self
                .api_post("/v2/vectordb/entities/insert", body)
                .await
                .context("Failed to insert entities into Milvus")?;

            let batch_count = resp
                .get("data")
                .and_then(|d| d.get("insertCount"))
                .and_then(|c| c.as_u64())
                .unwrap_or(chunk.len() as u64);

            inserted += batch_count as usize;
        }

        tracing::debug!("Inserted {} entities into Milvus", inserted);

        // Refresh IDF statistics after adding new documents.
        if let Err(e) = self.refresh_idf_stats().await {
            tracing::warn!("Failed to refresh IDF stats after indexing: {}", e);
        }

        Ok(inserted)
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
        self.search_filtered(
            query_vector,
            query_text,
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            vec![],
            vec![],
            vec![],
        )
        .await
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
        tracing::debug!(
            "Milvus search: limit={}, min_score={}, project={:?}, root_path={:?}, \
             hybrid={}, ext={:?}, lang={:?}, path={:?}",
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            file_extensions,
            languages,
            path_patterns,
        );

        let filter_expr =
            self.build_filter_expr(&project, &root_path, &file_extensions, &languages);

        let mut body = json!({
            "collectionName": self.collection_name,
            "data": [query_vector],
            "annsField": "embedding",
            "limit": limit,
            "outputFields": [
                "file_path",
                "root_path",
                "project",
                "start_line",
                "end_line",
                "language",
                "extension",
                "indexed_at",
                "content"
            ]
        });

        if !filter_expr.is_empty() {
            body["filter"] = Value::String(filter_expr);
        }

        let resp = self
            .api_post("/v2/vectordb/entities/search", body)
            .await
            .context("Failed to search Milvus collection")?;

        let data = resp
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();

        let mut results: Vec<SearchResult> = Vec::with_capacity(data.len());

        for item in &data {
            // Milvus COSINE metric returns `distance` in [0, 2] where 0 =
            // identical.  Convert to a similarity score in [0, 1].
            let distance = item.get("distance").and_then(|d| d.as_f64()).unwrap_or(1.0) as f32;
            let vector_score = 1.0 - distance;

            let content = match item.get("content").and_then(|c| c.as_str()) {
                Some(c) => c.to_string(),
                None => continue,
            };

            let (final_score, keyword_score) = if hybrid {
                let kw_score =
                    bm25_helpers::calculate_bm25_score(&self.idf_stats, query_text, &content).await;
                (
                    bm25_helpers::combine_scores(vector_score, kw_score),
                    Some(kw_score),
                )
            } else {
                (vector_score, None)
            };

            // Apply min_score filter after hybrid combination.
            if final_score < min_score {
                continue;
            }

            let file_path = match item.get("file_path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => continue,
            };

            let result_root_path = item
                .get("root_path")
                .and_then(|v| v.as_str())
                .map(String::from);

            let result_project = item
                .get("project")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from);

            let start_line = item.get("start_line").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

            let end_line = item.get("end_line").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

            let language = item
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();

            let indexed_at = item.get("indexed_at").and_then(|v| v.as_i64()).unwrap_or(0);

            results.push(SearchResult {
                file_path,
                root_path: result_root_path,
                content,
                score: final_score,
                vector_score,
                keyword_score,
                start_line,
                end_line,
                language,
                project: result_project,
                indexed_at,
            });
        }

        // Re-sort by combined score when using hybrid search.
        if hybrid {
            results.sort_by(|a, b| b.score.total_cmp(&a.score));
        }

        // Post-filter by path patterns using glob matching.
        if !path_patterns.is_empty() {
            results.retain(|r| glob_utils::matches_any_pattern(&r.file_path, &path_patterns));
        }

        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        tracing::debug!("Deleting embeddings for file: {}", file_path);

        let filter = format!("file_path == \"{}\"", Self::escape_filter_value(file_path));

        let body = json!({
            "collectionName": self.collection_name,
            "filter": filter
        });

        self.api_post("/v2/vectordb/entities/delete", body)
            .await
            .context("Failed to delete entities from Milvus")?;

        // Milvus REST API does not reliably report deleted count.
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        tracing::info!("Dropping Milvus collection '{}'", self.collection_name);

        let body = json!({ "collectionName": self.collection_name });
        self.api_post("/v2/vectordb/collections/drop", body)
            .await
            .context("Failed to drop Milvus collection")?;

        // Clear IDF stats.
        let mut stats = self.idf_stats.write().await;
        stats.total_docs = 0;
        stats.doc_frequencies.clear();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let body = json!({ "collectionName": self.collection_name });
        let resp = self
            .api_post("/v2/vectordb/collections/describe", body)
            .await
            .context("Failed to describe Milvus collection")?;

        let row_count = resp
            .get("data")
            .and_then(|d| d.get("rowCount"))
            .and_then(|r| r.as_str())
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| {
                resp.get("data")
                    .and_then(|d| d.get("rowCount"))
                    .and_then(|r| r.as_u64())
                    .map(|n| n as usize)
            })
            .unwrap_or(0);

        // Language breakdown is not directly available from the Milvus
        // describe endpoint; return an empty breakdown.
        Ok(DatabaseStats {
            total_points: row_count,
            total_vectors: row_count,
            language_breakdown: vec![],
        })
    }

    async fn flush(&self) -> Result<()> {
        // The Milvus REST API v2 does not expose a flush endpoint.
        // Data is persisted automatically after insert operations.
        tracing::debug!("Milvus flush is a no-op via REST API v2");
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let filter = format!("root_path == \"{}\"", Self::escape_filter_value(root_path));

        let body = json!({
            "collectionName": self.collection_name,
            "filter": filter,
            "outputFields": ["id"],
            "limit": QUERY_LIMIT
        });

        let resp = self
            .api_post("/v2/vectordb/entities/query", body)
            .await
            .context("Failed to query Milvus for count by root path")?;

        let count = resp
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);

        if count >= QUERY_LIMIT {
            tracing::warn!(
                "count_by_root_path hit query limit ({}); actual count may be higher",
                QUERY_LIMIT
            );
        }

        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let filter = format!("root_path == \"{}\"", Self::escape_filter_value(root_path));

        let body = json!({
            "collectionName": self.collection_name,
            "filter": filter,
            "outputFields": ["file_path"],
            "limit": QUERY_LIMIT
        });

        let resp = self
            .api_post("/v2/vectordb/entities/query", body)
            .await
            .context("Failed to query Milvus for indexed files")?;

        let data = resp
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();

        let mut unique_paths = std::collections::HashSet::new();
        for item in &data {
            if let Some(path) = item.get("file_path").and_then(|v| v.as_str()) {
                unique_paths.insert(path.to_string());
            }
        }

        if data.len() >= QUERY_LIMIT {
            tracing::warn!(
                "get_indexed_files hit query limit ({}); results may be incomplete",
                QUERY_LIMIT
            );
        }

        Ok(unique_paths.into_iter().collect())
    }
}

impl Default for MilvusDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata(file_path: &str, start: usize, end: usize) -> ChunkMetadata {
        ChunkMetadata {
            root_path: Some("/test/root".to_string()),
            file_path: file_path.to_string(),
            project: Some("test-project".to_string()),
            start_line: start,
            end_line: end,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "test_hash".to_string(),
            indexed_at: 1234567890,
        }
    }

    #[test]
    fn test_escape_filter_value() {
        assert_eq!(MilvusDatabase::escape_filter_value("hello"), "hello");
        assert_eq!(
            MilvusDatabase::escape_filter_value(r#"say "hi""#),
            r#"say \"hi\""#
        );
        assert_eq!(
            MilvusDatabase::escape_filter_value(r"back\slash"),
            r"back\\slash"
        );
    }

    #[tokio::test]
    #[ignore] // Requires running Milvus server on localhost:19530
    async fn test_milvus_lifecycle() {
        let db = MilvusDatabase::new();
        db.initialize(384).await.unwrap();

        // Store
        let embeddings = vec![vec![0.1f32; 384], vec![0.2f32; 384]];
        let metadata = vec![
            test_metadata("test1.rs", 1, 10),
            test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];
        let count = db
            .store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Search
        let results = db
            .search(vec![0.1f32; 384], "main", 10, 0.0, None, None, false)
            .await
            .unwrap();
        assert!(!results.is_empty());

        // Stats
        let stats = db.get_statistics().await.unwrap();
        assert!(stats.total_points >= 2);

        // Clear
        db.clear().await.unwrap();
    }
}
