//! Weaviate vector database backend for code embeddings.
//!
//! Connects to a running Weaviate instance via its REST and GraphQL APIs
//! (`/v1/schema`, `/v1/batch/objects`, `/v1/graphql`). Hybrid search uses
//! Weaviate's native `hybrid` operator (vector + BM25 fusion server-side),
//! with a client-side [`SharedIdfStats`] kept as a fallback only.

use crate::databases::bm25_helpers::{self, SharedIdfStats};
use crate::databases::traits::{ChunkMetadata, DatabaseStats, SearchResult, VectorDatabase};
use crate::glob_utils;
use anyhow::{Context, Result};
use serde_json::{Value, json};

const DEFAULT_CLASS_NAME: &str = "CodeEmbedding";

/// Weaviate-backed vector database for code embeddings.
///
/// Communicates with Weaviate over HTTP using the REST v1 API and GraphQL.
/// Hybrid search leverages Weaviate's native `hybrid` operator which fuses
/// vector similarity with BM25 keyword scoring on the server side.
pub struct WeaviateDatabase {
    client: reqwest::Client,
    base_url: String,
    /// Weaviate class name (PascalCase).
    class_name: String,
    /// Client-side IDF statistics — used as fallback only; native hybrid is
    /// preferred.
    idf_stats: SharedIdfStats,
}

impl WeaviateDatabase {
    /// Create a new Weaviate client pointing at `localhost:8080` with the
    /// default class name `CodeEmbedding`.
    pub fn new() -> Self {
        Self::with_url("http://localhost:8080")
    }

    /// Create a new Weaviate client with a custom URL and the default class
    /// name `CodeEmbedding`.
    pub fn with_url(url: &str) -> Self {
        Self::with_config(url, DEFAULT_CLASS_NAME)
    }

    /// Create a new Weaviate client with a custom URL and class name.
    pub fn with_config(url: &str, class_name: &str) -> Self {
        tracing::info!(
            "Creating Weaviate client at {} with class '{}'",
            url,
            class_name
        );
        Self {
            client: reqwest::Client::new(),
            base_url: url.trim_end_matches('/').to_string(),
            class_name: class_name.to_string(),
            idf_stats: bm25_helpers::new_shared_idf_stats(),
        }
    }

    /// Get the default Weaviate URL (public for CLI version info).
    pub fn default_url() -> String {
        "http://localhost:8080".to_string()
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Check whether the class already exists in the Weaviate schema.
    async fn class_exists(&self) -> Result<bool> {
        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to check Weaviate schema")?;

        Ok(resp.status().is_success())
    }

    /// Generate a deterministic UUID (v5-style) from file path and line range
    /// so that repeated indexing of the same chunk produces the same ID.
    pub(crate) fn deterministic_uuid(
        file_path: &str,
        start_line: usize,
        end_line: usize,
    ) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}:{}", file_path, start_line, end_line).as_bytes());
        let hash = hasher.finalize();
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]),
            u16::from_be_bytes([hash[4], hash[5]]),
            (u16::from_be_bytes([hash[6], hash[7]]) & 0x0FFF) | 0x5000,
            (u16::from_be_bytes([hash[8], hash[9]]) & 0x3FFF) | 0x8000,
            u64::from_be_bytes([
                0, 0, hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]
            ]),
        )
    }

    /// Build a Weaviate `where` filter object from the optional query
    /// parameters.
    fn build_where_filter(
        &self,
        project: &Option<String>,
        root_path: &Option<String>,
        file_extensions: &[String],
        languages: &[String],
    ) -> Option<Value> {
        let mut operands: Vec<Value> = Vec::new();

        if let Some(proj) = project {
            operands.push(json!({
                "path": ["project"],
                "operator": "Equal",
                "valueText": proj,
            }));
        }

        if let Some(rp) = root_path {
            operands.push(json!({
                "path": ["root_path"],
                "operator": "Equal",
                "valueText": rp,
            }));
        }

        if !file_extensions.is_empty() {
            operands.push(json!({
                "path": ["extension"],
                "operator": "ContainsAny",
                "valueTextArray": file_extensions,
            }));
        }

        if !languages.is_empty() {
            operands.push(json!({
                "path": ["language"],
                "operator": "ContainsAny",
                "valueTextArray": languages,
            }));
        }

        match operands.len() {
            0 => None,
            1 => Some(operands.into_iter().next().unwrap()),
            _ => Some(json!({
                "operator": "And",
                "operands": operands,
            })),
        }
    }

    /// Build the GraphQL fields list used for Get queries.
    fn result_fields() -> &'static str {
        "file_path root_path content project start_line end_line language extension indexed_at _additional { score }"
    }

    /// Execute a GraphQL query against Weaviate and return the parsed JSON
    /// response body.
    async fn graphql(&self, query: &str) -> Result<Value> {
        let url = format!("{}/v1/graphql", self.base_url);
        let body = json!({ "query": query });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Weaviate GraphQL request failed")?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .context("Failed to read Weaviate GraphQL response body")?;

        if !status.is_success() {
            anyhow::bail!(
                "Weaviate GraphQL returned HTTP {}: {}",
                status.as_u16(),
                text
            );
        }

        let parsed: Value =
            serde_json::from_str(&text).context("Failed to parse Weaviate GraphQL response")?;

        // Surface GraphQL-level errors.
        if let Some(errors) = parsed.get("errors")
            && errors.is_array()
            && !errors.as_array().unwrap().is_empty()
        {
            tracing::warn!("Weaviate GraphQL errors: {}", errors);
        }

        Ok(parsed)
    }

    /// Parse a single GraphQL result object into a [`SearchResult`].
    fn parse_result(obj: &Value) -> Option<SearchResult> {
        let file_path = obj.get("file_path")?.as_str()?.to_string();
        let content = obj.get("content")?.as_str()?.to_string();
        let start_line = obj.get("start_line")?.as_u64()? as usize;
        let end_line = obj.get("end_line")?.as_u64()? as usize;

        let language = obj
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let project = obj
            .get("project")
            .and_then(|v| v.as_str())
            .map(String::from);

        let root_path = obj
            .get("root_path")
            .and_then(|v| v.as_str())
            .map(String::from);

        let indexed_at = obj.get("indexed_at").and_then(|v| v.as_i64()).unwrap_or(0);

        let score = obj
            .get("_additional")
            .and_then(|a| a.get("score"))
            .and_then(|s| s.as_str())
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);

        Some(SearchResult {
            file_path,
            root_path,
            content,
            score,
            vector_score: score,
            keyword_score: None,
            start_line,
            end_line,
            language,
            project,
            indexed_at,
        })
    }
}

// ── VectorDatabase trait ─────────────────────────────────────────────────

#[async_trait::async_trait]
impl VectorDatabase for WeaviateDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        if self.class_exists().await? {
            tracing::info!(
                "Weaviate class '{}' already exists, skipping creation",
                self.class_name
            );
            return Ok(());
        }

        tracing::info!(
            "Creating Weaviate class '{}' with dimension {}",
            self.class_name,
            dimension
        );

        let schema = json!({
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexConfig": {
                "distance": "cosine"
            },
            "properties": [
                { "name": "file_path",  "dataType": ["text"] },
                { "name": "root_path",  "dataType": ["text"] },
                { "name": "project",    "dataType": ["text"] },
                { "name": "start_line", "dataType": ["int"]  },
                { "name": "end_line",   "dataType": ["int"]  },
                { "name": "language",   "dataType": ["text"] },
                { "name": "extension",  "dataType": ["text"] },
                { "name": "file_hash",  "dataType": ["text"] },
                { "name": "indexed_at", "dataType": ["int"]  },
                { "name": "content",    "dataType": ["text"] },
            ]
        });

        let url = format!("{}/v1/schema", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&schema)
            .send()
            .await
            .context("Failed to create Weaviate class")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Weaviate class creation returned HTTP {}: {}",
                status.as_u16(),
                body
            );
        }

        tracing::info!("Weaviate class '{}' created successfully", self.class_name);
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
        tracing::debug!(
            "Storing {} embeddings into Weaviate class '{}'",
            total,
            self.class_name
        );

        // Build objects
        let objects: Vec<Value> = embeddings
            .into_iter()
            .zip(metadata)
            .zip(contents)
            .map(|((embedding, meta), content)| {
                let uuid =
                    Self::deterministic_uuid(&meta.file_path, meta.start_line, meta.end_line);

                json!({
                    "id": uuid,
                    "class": self.class_name,
                    "properties": {
                        "file_path":  meta.file_path,
                        "root_path":  meta.root_path.as_deref().unwrap_or(root_path),
                        "project":    meta.project.as_deref().unwrap_or(""),
                        "start_line": meta.start_line as i64,
                        "end_line":   meta.end_line as i64,
                        "language":   meta.language.as_deref().unwrap_or("Unknown"),
                        "extension":  meta.extension.as_deref().unwrap_or(""),
                        "file_hash":  meta.file_hash,
                        "indexed_at": meta.indexed_at,
                        "content":    content,
                    },
                    "vector": embedding,
                })
            })
            .collect();

        // Batch in chunks of 100
        let batch_url = format!("{}/v1/batch/objects", self.base_url);
        let mut stored = 0usize;

        for chunk in objects.chunks(100) {
            let body = json!({ "objects": chunk });

            let resp = self
                .client
                .post(&batch_url)
                .json(&body)
                .send()
                .await
                .context("Weaviate batch insert failed")?;

            let status = resp.status();
            if !status.is_success() {
                let err_body = resp.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Weaviate batch insert returned HTTP {}: {}",
                    status.as_u16(),
                    err_body
                );
            }

            // Parse response to count successes (Weaviate returns per-object
            // status but we trust the batch on a 2xx).
            stored += chunk.len();
            tracing::debug!("Batch stored {}/{} objects", stored, total);
        }

        // Update client-side IDF stats (fallback).
        let contents_for_idf: Vec<String> = objects
            .iter()
            .filter_map(|o| {
                o.get("properties")
                    .and_then(|p| p.get("content"))
                    .and_then(|c| c.as_str())
                    .map(String::from)
            })
            .collect();
        if !contents_for_idf.is_empty() {
            bm25_helpers::update_idf_stats(&self.idf_stats, &contents_for_idf).await;
        }

        tracing::info!("Stored {} embeddings in Weaviate", stored);
        Ok(stored)
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
            "Weaviate search: limit={}, min_score={}, project={:?}, root_path={:?}, \
             hybrid={}, ext={:?}, lang={:?}, path={:?}",
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            file_extensions,
            languages,
            path_patterns
        );

        // ── Build the where clause ──────────────────────────────────────
        let where_filter =
            self.build_where_filter(&project, &root_path, &file_extensions, &languages);
        let where_clause = match where_filter {
            Some(f) => format!(", where: {}", serde_json::to_string(&f).unwrap()),
            None => String::new(),
        };

        // ── Build the search operator ───────────────────────────────────
        let vector_str = format!(
            "[{}]",
            query_vector
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let search_operator = if hybrid {
            // Escape the query text for GraphQL string literal.
            let escaped_query = query_text.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "hybrid: {{ query: \"{}\", vector: {}, alpha: 0.7 }}",
                escaped_query, vector_str
            )
        } else {
            format!("nearVector: {{ vector: {} }}", vector_str)
        };

        let fields = Self::result_fields();

        let gql = format!(
            "{{ Get {{ {}({}, limit: {}{}) {{ {} }} }} }}",
            self.class_name, search_operator, limit, where_clause, fields
        );

        let response = self.graphql(&gql).await?;

        // ── Parse results ───────────────────────────────────────────────
        let empty_vec = vec![];
        let items = response
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(&self.class_name))
            .and_then(|c| c.as_array())
            .unwrap_or(&empty_vec);

        let mut results: Vec<SearchResult> = items
            .iter()
            .filter_map(Self::parse_result)
            .filter(|r| r.score >= min_score)
            .collect();

        // Post-filter by path patterns using glob matching.
        if !path_patterns.is_empty() {
            results.retain(|r| glob_utils::matches_any_pattern(&r.file_path, &path_patterns));
        }

        // Sort descending by score.
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        tracing::debug!("Weaviate search returned {} results", results.len());
        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        tracing::debug!("Deleting Weaviate objects for file: {}", file_path);

        let url = format!("{}/v1/batch/objects/delete", self.base_url);
        let body = json!({
            "match": {
                "class": self.class_name,
                "where": {
                    "path": ["file_path"],
                    "operator": "Equal",
                    "valueText": file_path,
                }
            }
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Weaviate batch delete failed")?;

        let status = resp.status();
        if !status.is_success() {
            let err_body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Weaviate batch delete returned HTTP {}: {}",
                status.as_u16(),
                err_body
            );
        }

        // Weaviate batch delete does not reliably report the exact count of
        // deleted objects, so we return 0.
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        tracing::info!(
            "Clearing Weaviate class '{}' (deleting schema)",
            self.class_name
        );

        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);
        let resp = self
            .client
            .delete(&url)
            .send()
            .await
            .context("Failed to delete Weaviate class")?;

        let status = resp.status();
        // 404 is fine — the class was already gone.
        if !status.is_success() && status.as_u16() != 404 {
            let err_body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Weaviate class deletion returned HTTP {}: {}",
                status.as_u16(),
                err_body
            );
        }

        // Reset client-side IDF stats.
        let mut stats = self.idf_stats.write().await;
        stats.total_docs = 0;
        stats.doc_frequencies.clear();

        tracing::info!("Weaviate class '{}' deleted", self.class_name);
        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let gql = format!(
            "{{ Aggregate {{ {} {{ meta {{ count }} }} }} }}",
            self.class_name
        );

        let response = self.graphql(&gql).await?;

        let count = response
            .get("data")
            .and_then(|d| d.get("Aggregate"))
            .and_then(|a| a.get(&self.class_name))
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|obj| obj.get("meta"))
            .and_then(|m| m.get("count"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        Ok(DatabaseStats {
            total_points: count,
            total_vectors: count,
            language_breakdown: vec![],
        })
    }

    async fn flush(&self) -> Result<()> {
        // Weaviate persists automatically; no explicit flush required.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let escaped = root_path.replace('\\', "\\\\").replace('"', "\\\"");
        let where_filter = json!({
            "path": ["root_path"],
            "operator": "Equal",
            "valueText": escaped,
        });
        let where_str = serde_json::to_string(&where_filter).unwrap();

        let gql = format!(
            "{{ Aggregate {{ {}(where: {}) {{ meta {{ count }} }} }} }}",
            self.class_name, where_str
        );

        let response = self.graphql(&gql).await?;

        let count = response
            .get("data")
            .and_then(|d| d.get("Aggregate"))
            .and_then(|a| a.get(&self.class_name))
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|obj| obj.get("meta"))
            .and_then(|m| m.get("count"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let escaped = root_path.replace('\\', "\\\\").replace('"', "\\\"");
        let where_filter = json!({
            "path": ["root_path"],
            "operator": "Equal",
            "valueText": escaped,
        });
        let where_str = serde_json::to_string(&where_filter).unwrap();

        let gql = format!(
            "{{ Get {{ {}(where: {}, limit: 10000) {{ file_path }} }} }}",
            self.class_name, where_str
        );

        let response = self.graphql(&gql).await?;

        let empty_vec = vec![];
        let items = response
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(&self.class_name))
            .and_then(|c| c.as_array())
            .unwrap_or(&empty_vec);

        let mut file_paths = std::collections::HashSet::new();
        for item in items {
            if let Some(fp) = item.get("file_path").and_then(|v| v.as_str()) {
                file_paths.insert(fp.to_string());
            }
        }

        Ok(file_paths.into_iter().collect())
    }
}

impl Default for WeaviateDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ChunkMetadata;

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

    #[tokio::test]
    #[ignore] // Requires running Weaviate server on localhost:8080
    async fn test_weaviate_lifecycle() {
        let db = WeaviateDatabase::new();
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

        // Delete
        db.delete_by_file("test1.rs").await.unwrap();

        // Clear
        db.clear().await.unwrap();
    }

    #[test]
    fn test_deterministic_uuid() {
        let uuid1 = WeaviateDatabase::deterministic_uuid("file.rs", 1, 10);
        let uuid2 = WeaviateDatabase::deterministic_uuid("file.rs", 1, 10);
        let uuid3 = WeaviateDatabase::deterministic_uuid("other.rs", 1, 10);
        assert_eq!(uuid1, uuid2); // Same inputs = same UUID
        assert_ne!(uuid1, uuid3); // Different inputs = different UUID
        // Check UUID format (8-4-4-4-12)
        assert_eq!(uuid1.len(), 36);
        assert_eq!(uuid1.chars().filter(|c| *c == '-').count(), 4);
    }
}
