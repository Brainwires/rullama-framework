//! Pinecone managed cloud vector database backend.
//!
//! [`PineconeDatabase`] implements [`VectorDatabase`] for use with Pinecone's
//! cloud-hosted vector search service.
//!
//! # Feature flag
//!
//! Requires `pinecone-backend`.
//!
//! # Configuration
//!
//! Requires a Pinecone API key and an index host URL. The index must already
//! exist in the Pinecone console — this backend does **not** create indexes
//! automatically.
//!
//! ```ignore
//! let db = PineconeDatabase::new(
//!     "https://my-index-abc1234.svc.us-east-1-aws.pinecone.io",
//!     "pcsk_...",
//!     "my-namespace",
//! );
//! db.initialize(384).await?;
//! ```

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

use crate::databases::traits::{ChunkMetadata, DatabaseStats, SearchResult, VectorDatabase};

/// Pinecone REST API vector database client.
///
/// Implements [`VectorDatabase`] only (no
/// [`StorageBackend`](crate::databases::traits::StorageBackend) — Pinecone is a
/// pure vector store with no relational query support).
pub struct PineconeDatabase {
    /// Base URL of the Pinecone index (e.g. `https://my-index-abc.svc.pinecone.io`).
    index_host: String,
    /// Pinecone API key.
    api_key: String,
    /// Namespace to use within the index (for multi-project isolation).
    namespace: String,
    /// HTTP client shared across requests.
    client: Client,
    /// Embedding dimension (set during `initialize`).
    dimension: RwLock<Option<usize>>,
}

impl PineconeDatabase {
    /// Create a new Pinecone database client.
    ///
    /// - `index_host` — the full URL of the Pinecone index
    /// - `api_key` — Pinecone API key
    /// - `namespace` — namespace within the index (use `""` for default)
    pub fn new(
        index_host: impl Into<String>,
        api_key: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Self {
        let index_host = index_host.into().trim_end_matches('/').to_string();
        Self {
            index_host,
            api_key: api_key.into(),
            namespace: namespace.into(),
            client: Client::new(),
            dimension: RwLock::new(None),
        }
    }

    /// Build a full URL for a Pinecone REST API endpoint.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.index_host, path)
    }

    /// Build metadata filter JSON for Pinecone queries.
    fn build_metadata_filter(
        &self,
        project: Option<&str>,
        root_path: Option<&str>,
        file_extensions: &[String],
        languages: &[String],
        path_patterns: &[String],
    ) -> Option<serde_json::Value> {
        let mut conditions = Vec::new();

        if let Some(p) = project {
            conditions.push(serde_json::json!({ "project": { "$eq": p } }));
        }
        if let Some(rp) = root_path {
            conditions.push(serde_json::json!({ "root_path": { "$eq": rp } }));
        }
        if !file_extensions.is_empty() {
            conditions.push(serde_json::json!({ "extension": { "$in": file_extensions } }));
        }
        if !languages.is_empty() {
            conditions.push(serde_json::json!({ "language": { "$in": languages } }));
        }
        // Path patterns are matched client-side since Pinecone doesn't support
        // regex/glob in metadata filters. We request extra results and post-filter.
        let _ = path_patterns;

        if conditions.is_empty() {
            None
        } else if conditions.len() == 1 {
            Some(conditions.into_iter().next().unwrap())
        } else {
            Some(serde_json::json!({ "$and": conditions }))
        }
    }

    /// Convert Pinecone query results into `SearchResult` values.
    fn matches_to_results(
        &self,
        matches: Vec<PineconeMatch>,
        min_score: f32,
        path_patterns: &[String],
    ) -> Vec<SearchResult> {
        matches
            .into_iter()
            .filter(|m| m.score >= min_score)
            .filter_map(|m| {
                let meta = m.metadata.as_ref()?;
                let file_path = meta.get("file_path")?.as_str()?.to_string();

                // Client-side path pattern filtering
                if !path_patterns.is_empty() {
                    let matches_pattern = path_patterns.iter().any(|p| file_path.contains(p));
                    if !matches_pattern {
                        return None;
                    }
                }

                let content = meta
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let root_path = meta
                    .get("root_path")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let start_line =
                    meta.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let end_line = meta.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let language = meta
                    .get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let project = meta
                    .get("project")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let indexed_at = meta.get("indexed_at").and_then(|v| v.as_i64()).unwrap_or(0);

                Some(SearchResult {
                    file_path,
                    root_path,
                    content,
                    score: m.score,
                    vector_score: m.score,
                    keyword_score: None,
                    start_line,
                    end_line,
                    language,
                    project,
                    indexed_at,
                })
            })
            .collect()
    }
}

// ── Pinecone REST API types ──────────────────────────────────────────────

#[derive(Serialize)]
struct UpsertRequest {
    vectors: Vec<PineconeVector>,
    namespace: String,
}

#[derive(Serialize)]
struct PineconeVector {
    id: String,
    values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct QueryRequest {
    vector: Vec<f32>,
    top_k: usize,
    namespace: String,
    include_metadata: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct QueryResponse {
    matches: Vec<PineconeMatch>,
}

#[derive(Deserialize)]
struct PineconeMatch {
    #[allow(dead_code)]
    id: String,
    score: f32,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct DeleteRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<serde_json::Value>,
    namespace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    delete_all: Option<bool>,
}

#[derive(Deserialize)]
struct DescribeIndexStatsResponse {
    namespaces: Option<HashMap<String, NamespaceStats>>,
    #[allow(dead_code)]
    dimension: Option<usize>,
    #[allow(dead_code)]
    total_vector_count: Option<usize>,
}

#[derive(Deserialize)]
struct NamespaceStats {
    vector_count: usize,
}

#[derive(Deserialize)]
struct ListResponse {
    vectors: Option<Vec<ListVector>>,
    #[allow(dead_code)]
    pagination: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ListVector {
    id: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FetchResponse {
    vectors: Option<HashMap<String, FetchedVector>>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FetchedVector {
    metadata: Option<serde_json::Value>,
}

// ── VectorDatabase implementation ────────────────────────────────────────

#[async_trait::async_trait]
impl VectorDatabase for PineconeDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        // Store the dimension. We don't create the index — Pinecone indexes
        // are managed via the Pinecone console / API separately.
        {
            let mut dim = self.dimension.write().map_err(|e| anyhow::anyhow!("{e}"))?;
            *dim = Some(dimension);
        }

        // Verify connectivity by fetching index stats.
        let resp = self
            .client
            .post(self.url("/describe_index_stats"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({}))
            .send()
            .await
            .context("Failed to connect to Pinecone index")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone health check failed (HTTP {status}): {body}");
        }

        tracing::info!(dimension, host = %self.index_host, "Pinecone database initialized");
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
        if embeddings.len() != metadata.len() || embeddings.len() != contents.len() {
            bail!(
                "Mismatched lengths: {} embeddings, {} metadata, {} contents",
                embeddings.len(),
                metadata.len(),
                contents.len()
            );
        }

        let total = embeddings.len();
        // Pinecone recommends batches of up to 100 vectors.
        const BATCH_SIZE: usize = 100;
        let mut stored = 0;

        for batch_start in (0..total).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(total);
            let mut vectors = Vec::with_capacity(batch_end - batch_start);

            for i in batch_start..batch_end {
                let meta = &metadata[i];
                let id = format!("{}:{}:{}", root_path, meta.file_path, meta.start_line);

                let metadata_json = serde_json::json!({
                    "file_path": meta.file_path,
                    "root_path": root_path,
                    "project": meta.project,
                    "start_line": meta.start_line,
                    "end_line": meta.end_line,
                    "language": meta.language,
                    "extension": meta.extension,
                    "file_hash": meta.file_hash,
                    "indexed_at": meta.indexed_at,
                    "content": contents[i],
                });

                vectors.push(PineconeVector {
                    id,
                    values: embeddings[i].clone(),
                    metadata: Some(metadata_json),
                });
            }

            let request = UpsertRequest {
                vectors,
                namespace: self.namespace.clone(),
            };

            let resp = self
                .client
                .post(self.url("/vectors/upsert"))
                .header("Api-Key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .context("Pinecone upsert request failed")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("Pinecone upsert failed (HTTP {status}): {body}");
            }

            stored += batch_end - batch_start;
            tracing::debug!(stored, total, "Pinecone upsert progress");
        }

        tracing::info!(stored, "Stored embeddings in Pinecone");
        Ok(stored)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        _query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        _hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        let filter =
            self.build_metadata_filter(project.as_deref(), root_path.as_deref(), &[], &[], &[]);

        let request = QueryRequest {
            vector: query_vector,
            top_k: limit,
            namespace: self.namespace.clone(),
            include_metadata: true,
            filter,
        };

        let resp = self
            .client
            .post(self.url("/query"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Pinecone query request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone query failed (HTTP {status}): {body}");
        }

        let query_resp: QueryResponse = resp
            .json()
            .await
            .context("Failed to parse Pinecone query response")?;
        Ok(self.matches_to_results(query_resp.matches, min_score, &[]))
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        _query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
        _hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        // Request extra results when post-filtering by path pattern.
        let extra = if path_patterns.is_empty() { 1 } else { 3 };

        let filter = self.build_metadata_filter(
            project.as_deref(),
            root_path.as_deref(),
            &file_extensions,
            &languages,
            &path_patterns,
        );

        let request = QueryRequest {
            vector: query_vector,
            top_k: limit * extra,
            namespace: self.namespace.clone(),
            include_metadata: true,
            filter,
        };

        let resp = self
            .client
            .post(self.url("/query"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Pinecone filtered query request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone filtered query failed (HTTP {status}): {body}");
        }

        let query_resp: QueryResponse = resp
            .json()
            .await
            .context("Failed to parse Pinecone query response")?;
        let mut results = self.matches_to_results(query_resp.matches, min_score, &path_patterns);
        results.truncate(limit);
        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        // Pinecone supports deletion by metadata filter.
        let request = DeleteRequest {
            ids: None,
            filter: Some(serde_json::json!({ "file_path": { "$eq": file_path } })),
            namespace: self.namespace.clone(),
            delete_all: None,
        };

        let resp = self
            .client
            .post(self.url("/vectors/delete"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Pinecone delete request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone delete failed (HTTP {status}): {body}");
        }

        // Pinecone delete doesn't return a count — return 0 as a placeholder.
        tracing::debug!(file_path, "Deleted vectors for file from Pinecone");
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        let request = DeleteRequest {
            ids: None,
            filter: None,
            namespace: self.namespace.clone(),
            delete_all: Some(true),
        };

        let resp = self
            .client
            .post(self.url("/vectors/delete"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Pinecone clear request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone clear failed (HTTP {status}): {body}");
        }

        tracing::info!("Cleared all vectors from Pinecone namespace");
        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let resp = self
            .client
            .post(self.url("/describe_index_stats"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({}))
            .send()
            .await
            .context("Pinecone describe_index_stats failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone describe_index_stats failed (HTTP {status}): {body}");
        }

        let stats: DescribeIndexStatsResponse = resp
            .json()
            .await
            .context("Failed to parse Pinecone stats")?;

        let total_vectors = stats
            .namespaces
            .as_ref()
            .and_then(|ns| ns.get(&self.namespace))
            .map(|n| n.vector_count)
            .unwrap_or(0);

        Ok(DatabaseStats {
            total_points: total_vectors,
            total_vectors,
            // Pinecone doesn't expose per-language breakdowns natively.
            language_breakdown: Vec::new(),
        })
    }

    async fn flush(&self) -> Result<()> {
        // Pinecone is a managed service — writes are durable immediately.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        // Use describe_index_stats with a filter to approximate the count.
        // Pinecone's describe_index_stats supports a filter parameter.
        let resp = self
            .client
            .post(self.url("/describe_index_stats"))
            .header("Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "filter": { "root_path": { "$eq": root_path } }
            }))
            .send()
            .await
            .context("Pinecone count_by_root_path failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Pinecone count_by_root_path failed (HTTP {status}): {body}");
        }

        let stats: DescribeIndexStatsResponse = resp
            .json()
            .await
            .context("Failed to parse Pinecone stats")?;

        let count = stats
            .namespaces
            .as_ref()
            .and_then(|ns| ns.get(&self.namespace))
            .map(|n| n.vector_count)
            .unwrap_or(0);

        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        // Pinecone doesn't have a native "list unique metadata values" API.
        // We list vector IDs (which encode root_path:file_path:line) and extract
        // unique file paths from the IDs.
        let prefix = format!("{}:", root_path);

        let resp = self
            .client
            .get(self.url("/vectors/list"))
            .header("Api-Key", &self.api_key)
            .query(&[
                ("namespace", self.namespace.as_str()),
                ("prefix", prefix.as_str()),
                ("limit", "10000"),
            ])
            .send()
            .await
            .context("Pinecone list vectors failed")?;

        if !resp.status().is_success() {
            // If the list endpoint is unavailable (older Pinecone plans), fall back
            // to returning an empty list rather than failing hard.
            tracing::warn!(
                "Pinecone list endpoint returned non-success; returning empty file list"
            );
            return Ok(Vec::new());
        }

        let list: ListResponse = resp
            .json()
            .await
            .context("Failed to parse Pinecone list response")?;

        let mut files: Vec<String> = list
            .vectors
            .unwrap_or_default()
            .into_iter()
            .filter_map(|v| {
                // ID format: "root_path:file_path:start_line"
                let rest = v.id.strip_prefix(&prefix)?;
                let file_path = rest.rsplit_once(':')?.0;
                Some(file_path.to_string())
            })
            .collect();

        files.sort();
        files.dedup();
        Ok(files)
    }
}
