//! Internal transport abstraction layer for NornicDB.
//!
//! Defines the `NornicTransport` trait (internal to `nornicdb`) and three
//! concrete implementations:
//!
//! | Transport       | Feature gate        | Protocol          |
//! |-----------------|---------------------|-------------------|
//! | `RestTransport` | `nornicdb-backend`  | HTTP / REST       |
//! | `BoltTransport` | `nornicdb-bolt`     | Neo4j Bolt binary |
//! | `GrpcTransport` | `nornicdb-grpc`     | gRPC (Qdrant)     |
//!
//! `NornicDatabase` (in `mod.rs`) holds an `Arc<dyn NornicTransport>` and
//! delegates all wire-level communication through this trait.

use anyhow::{Context, Result};
use serde_json::{Value, json};

// ── Trait ───────────────────────────────────────────────────────────────

/// Wire-level transport abstraction for NornicDB.
///
/// Every method maps to a logical operation that may be implemented over
/// different protocols.  Implementations must be `Send + Sync` so the
/// trait object can live inside an `Arc`.
#[async_trait::async_trait]
pub(crate) trait NornicTransport: Send + Sync {
    /// Verify that the remote server is reachable and healthy.
    async fn health_check(&self) -> Result<bool>;

    /// Execute an arbitrary Cypher query with the given JSON parameters.
    async fn execute_cypher(&self, query: &str, params: Value) -> Result<Vec<Value>>;

    /// Combined vector + keyword (BM25) search.
    async fn hybrid_search(
        &self,
        query_text: &str,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        node_label: &str,
        filters: Value,
    ) -> Result<Vec<Value>>;

    /// Pure vector similarity search.
    async fn vector_search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        node_label: &str,
        filters: Value,
    ) -> Result<Vec<Value>>;

    /// Batch-upsert nodes.  Returns the number of nodes written.
    async fn store_nodes(&self, nodes: Vec<Value>, node_label: &str) -> Result<usize>;

    /// Delete nodes matching `{property}: value` on the given label.
    async fn delete_nodes(&self, node_label: &str, property: &str, value: &str) -> Result<usize>;

    /// Count nodes matching `{property}: value` on the given label.
    async fn count_nodes(&self, node_label: &str, property: &str, value: &str) -> Result<usize>;

    /// Return the distinct values of `property` on nodes whose
    /// `filter_prop` equals `filter_val`.
    async fn distinct_property(
        &self,
        node_label: &str,
        property: &str,
        filter_prop: &str,
        filter_val: &str,
    ) -> Result<Vec<String>>;

    /// Human-readable name of the transport (for diagnostics / logging).
    #[allow(dead_code)]
    fn transport_name(&self) -> &'static str;
}

// ════════════════════════════════════════════════════════════════════════
//  REST Transport  (always available when `nornicdb-backend` is enabled)
// ════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use tokio::sync::RwLock;

/// REST transport backed by `reqwest`.
///
/// Talks to a NornicDB server over its HTTP API.  Authentication is
/// performed lazily via [`authenticate`] which stores a JWT for
/// subsequent requests.
pub(crate) struct RestTransport {
    client: reqwest::Client,
    base_url: String,
    database: String,
    auth_token: Arc<RwLock<Option<String>>>,
}

impl RestTransport {
    /// Create a new REST transport targeting `base_url`.
    ///
    /// No network call is made until the first method invocation.
    pub(crate) async fn new(base_url: &str, database: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client for NornicDB REST transport")?;

        tracing::info!(
            "REST transport created for {} (database: {})",
            base_url,
            database
        );

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            database: database.to_string(),
            auth_token: Arc::new(RwLock::new(None)),
        })
    }

    /// Authenticate against the NornicDB server and store the resulting
    /// JWT for future requests.
    pub(crate) async fn authenticate(&self, username: &str, password: &str) -> Result<()> {
        let url = format!("{}/auth/token", self.base_url);
        let body = json!({
            "username": username,
            "password": password,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to POST /auth/token")?;

        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .context("Failed to parse authentication response")?;

        if !status.is_success() {
            anyhow::bail!(
                "Authentication failed (HTTP {}): {}",
                status,
                payload.get("error").unwrap_or(&Value::Null)
            );
        }

        let token = payload
            .get("token")
            .and_then(Value::as_str)
            .context("Authentication response missing 'token' field")?
            .to_string();

        *self.auth_token.write().await = Some(token);
        tracing::info!("REST transport authenticated successfully");
        Ok(())
    }

    /// Build a request with the Bearer token attached (if available).
    fn build_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        // We cannot await inside a non-async fn, so we use try_read to
        // opportunistically attach the token.  The token changes very
        // rarely (only on authenticate()) so this is safe.
        let builder = self.client.request(method, &url);
        if let Ok(guard) = self.auth_token.try_read()
            && let Some(ref token) = *guard
        {
            return builder.bearer_auth(token);
        }
        builder
    }
}

#[async_trait::async_trait]
impl NornicTransport for RestTransport {
    async fn health_check(&self) -> Result<bool> {
        let resp = self
            .build_request(reqwest::Method::GET, "/health")
            .send()
            .await;

        match resp {
            Ok(r) => Ok(r.status().is_success()),
            Err(e) => {
                tracing::warn!("NornicDB health check failed: {}", e);
                Ok(false)
            }
        }
    }

    async fn execute_cypher(&self, query: &str, params: Value) -> Result<Vec<Value>> {
        let path = format!("/db/{}/tx/commit", self.database);
        let body = json!({
            "statements": [{
                "statement": query,
                "parameters": params,
            }]
        });

        let resp = self
            .build_request(reqwest::Method::POST, &path)
            .json(&body)
            .send()
            .await
            .context("Failed to execute Cypher query via REST")?;

        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .context("Failed to parse Cypher response")?;

        // Check for errors in the response envelope.
        if let Some(errors) = payload.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            anyhow::bail!("Cypher error(s): {}", serde_json::to_string(errors)?);
        }

        if !status.is_success() {
            anyhow::bail!("Cypher query failed (HTTP {}): {}", status, payload);
        }

        // Extract rows: results[0].data[].row
        let rows = payload
            .get("results")
            .and_then(|r| r.get(0))
            .and_then(|r| r.get("data"))
            .and_then(Value::as_array)
            .map(|data| {
                data.iter()
                    .filter_map(|entry| entry.get("row").cloned())
                    .collect::<Vec<Value>>()
            })
            .unwrap_or_default();

        Ok(rows)
    }

    async fn hybrid_search(
        &self,
        query_text: &str,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        node_label: &str,
        filters: Value,
    ) -> Result<Vec<Value>> {
        let body = json!({
            "query": query_text,
            "embedding": query_vector,
            "limit": limit,
            "min_score": min_score,
            "labels": [node_label],
            "filters": filters,
        });

        let resp = self
            .build_request(reqwest::Method::POST, "/nornicdb/search")
            .json(&body)
            .send()
            .await
            .context("Failed to execute hybrid search via REST")?;

        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .context("Failed to parse hybrid search response")?;

        if !status.is_success() {
            anyhow::bail!("Hybrid search failed (HTTP {}): {}", status, payload);
        }

        Self::map_search_results(&payload)
    }

    async fn vector_search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        node_label: &str,
        filters: Value,
    ) -> Result<Vec<Value>> {
        let body = json!({
            "vector": query_vector,
            "limit": limit,
            "min_score": min_score,
            "labels": [node_label],
            "filters": filters,
        });

        let resp = self
            .build_request(reqwest::Method::POST, "/nornicdb/similar")
            .json(&body)
            .send()
            .await
            .context("Failed to execute vector search via REST")?;

        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .context("Failed to parse vector search response")?;

        if !status.is_success() {
            anyhow::bail!("Vector search failed (HTTP {}): {}", status, payload);
        }

        Self::map_search_results(&payload)
    }

    async fn store_nodes(&self, nodes: Vec<Value>, node_label: &str) -> Result<usize> {
        const BATCH_SIZE: usize = 500;
        let mut total = 0usize;

        for chunk in nodes.chunks(BATCH_SIZE) {
            let batch: Vec<Value> = chunk.to_vec();
            let cypher = format!(
                "UNWIND $batch AS item \
                 MERGE (n:{node_label} {{file_path: item.file_path, start_line: item.start_line}}) \
                 SET n += item"
            );
            let params = json!({ "batch": batch });
            self.execute_cypher(&cypher, params)
                .await
                .context("Failed to store node batch via REST")?;
            total += chunk.len();
        }

        tracing::debug!("Stored {} nodes (label: {})", total, node_label);
        Ok(total)
    }

    async fn delete_nodes(&self, node_label: &str, property: &str, value: &str) -> Result<usize> {
        let cypher = format!(
            "MATCH (n:{node_label} {{{property}: $value}}) \
             WITH n, count(n) AS cnt \
             DETACH DELETE n \
             RETURN cnt"
        );
        let params = json!({ "value": value });
        let rows = self
            .execute_cypher(&cypher, params)
            .await
            .context("Failed to delete nodes via REST")?;

        let count = rows
            .first()
            .and_then(|r| r.get(0))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        tracing::debug!(
            "Deleted {} nodes (label: {}, {}: {})",
            count,
            node_label,
            property,
            value
        );
        Ok(count)
    }

    async fn count_nodes(&self, node_label: &str, property: &str, value: &str) -> Result<usize> {
        let cypher =
            format!("MATCH (n:{node_label}) WHERE n.{property} = $value RETURN count(n) AS cnt");
        let params = json!({ "value": value });
        let rows = self
            .execute_cypher(&cypher, params)
            .await
            .context("Failed to count nodes via REST")?;

        let count = rows
            .first()
            .and_then(|r| r.get(0))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        Ok(count)
    }

    async fn distinct_property(
        &self,
        node_label: &str,
        property: &str,
        filter_prop: &str,
        filter_val: &str,
    ) -> Result<Vec<String>> {
        let cypher = format!(
            "MATCH (n:{node_label}) WHERE n.{filter_prop} = $filter_val \
             RETURN DISTINCT n.{property} AS val"
        );
        let params = json!({ "filter_val": filter_val });
        let rows = self
            .execute_cypher(&cypher, params)
            .await
            .context("Failed to query distinct property via REST")?;

        let values = rows
            .iter()
            .filter_map(|r| r.get(0).and_then(Value::as_str).map(String::from))
            .collect();

        Ok(values)
    }

    fn transport_name(&self) -> &'static str {
        "REST"
    }
}

impl RestTransport {
    /// Map the server-side search response into a normalised `Vec<Value>`.
    fn map_search_results(payload: &Value) -> Result<Vec<Value>> {
        let results = payload
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mapped: Vec<Value> = results
            .into_iter()
            .map(|item| {
                json!({
                    "file_path":     item.get("file_path").cloned().unwrap_or(Value::Null),
                    "content":       item.get("content").cloned().unwrap_or(Value::Null),
                    "score":         item.get("score").cloned().unwrap_or(Value::Null),
                    "vector_score":  item.get("vector_score").cloned().unwrap_or(Value::Null),
                    "keyword_score": item.get("keyword_score").cloned().unwrap_or(Value::Null),
                    "start_line":    item.get("start_line").cloned().unwrap_or(Value::Null),
                    "end_line":      item.get("end_line").cloned().unwrap_or(Value::Null),
                    "language":      item.get("language").cloned().unwrap_or(Value::Null),
                    "project":       item.get("project").cloned().unwrap_or(Value::Null),
                    "root_path":     item.get("root_path").cloned().unwrap_or(Value::Null),
                    "indexed_at":    item.get("indexed_at").cloned().unwrap_or(Value::Null),
                })
            })
            .collect();

        Ok(mapped)
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Bolt Transport  (requires `nornicdb-bolt` feature)
// ════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nornicdb-bolt")]
mod bolt {
    use super::*;
    use neo4rs::Graph;

    /// Bolt (binary) transport using the `neo4rs` driver.
    ///
    /// Connects to a Neo4j-compatible server over the Bolt protocol,
    /// which is more efficient than REST for high-throughput workloads.
    pub(crate) struct BoltTransport {
        graph: Graph,
    }

    impl BoltTransport {
        /// Open a Bolt connection.
        pub(crate) async fn new(url: &str, username: &str, password: &str) -> Result<Self> {
            tracing::info!("Connecting to NornicDB via Bolt at {}", url);
            let graph = Graph::new(url, username, password)
                .context("Failed to connect to NornicDB via Bolt")?;
            Ok(Self { graph })
        }

        /// Convert a `serde_json::Value` object into a `neo4rs::Query`
        /// with all parameters bound.
        fn bind_params(query_str: &str, params: &Value) -> neo4rs::Query {
            let mut q = neo4rs::query(query_str);
            if let Some(obj) = params.as_object() {
                for (key, val) in obj {
                    match val {
                        Value::String(s) => q = q.param(key.as_str(), s.clone()),
                        Value::Number(n) if n.is_i64() => {
                            q = q.param(key.as_str(), n.as_i64().unwrap());
                        }
                        Value::Number(n) if n.is_f64() => {
                            q = q.param(key.as_str(), n.as_f64().unwrap());
                        }
                        Value::Bool(b) => q = q.param(key.as_str(), *b),
                        other => {
                            // Fall back to stringified JSON.
                            q = q.param(key.as_str(), other.to_string());
                        }
                    }
                }
            }
            q
        }

        /// Execute a query and collect all result rows into `Vec<Value>`.
        async fn run_query(&self, query_str: &str, params: &Value) -> Result<Vec<Value>> {
            let q = Self::bind_params(query_str, params);
            let mut result = self
                .graph
                .execute(q)
                .await
                .context("Bolt query execution failed")?;

            let mut rows: Vec<Value> = Vec::new();
            while let Some(row) = result.next().await? {
                // Attempt to pull known column names.  neo4rs rows are
                // positional; we serialise each column as a JSON value.
                let mut row_values: Vec<Value> = Vec::new();
                // neo4rs exposes columns via `row.get::<T>(name)`.
                // We try common column names and fall back to index-based
                // extraction. Since we control the queries, we know the
                // expected column names.
                if let Ok(val) = row.get::<String>("cnt") {
                    row_values.push(Value::String(val));
                } else if let Ok(val) = row.get::<i64>("cnt") {
                    row_values.push(json!(val));
                } else if let Ok(val) = row.get::<String>("val") {
                    row_values.push(Value::String(val));
                } else {
                    // Generic fallback: try to get the first column as a string.
                    if let Ok(val) = row.get::<String>("n") {
                        row_values.push(Value::String(val));
                    }
                }

                if !row_values.is_empty() {
                    rows.push(Value::Array(row_values));
                }
            }

            Ok(rows)
        }
    }

    #[async_trait::async_trait]
    impl NornicTransport for BoltTransport {
        async fn health_check(&self) -> Result<bool> {
            let q = neo4rs::query("RETURN 1 AS ping");
            match self.graph.execute(q).await {
                Ok(_) => Ok(true),
                Err(e) => {
                    tracing::warn!("Bolt health check failed: {}", e);
                    Ok(false)
                }
            }
        }

        async fn execute_cypher(&self, query: &str, params: Value) -> Result<Vec<Value>> {
            self.run_query(query, &params).await
        }

        async fn hybrid_search(
            &self,
            _query_text: &str,
            query_vector: Vec<f32>,
            limit: usize,
            min_score: f32,
            node_label: &str,
            filters: Value,
        ) -> Result<Vec<Value>> {
            // Bolt does not expose a server-side BM25 endpoint.
            // Fall back to pure vector search.
            tracing::debug!(
                "Bolt transport: hybrid_search delegates to vector_search (no BM25 via Bolt)"
            );
            self.vector_search(query_vector, limit, min_score, node_label, filters)
                .await
        }

        async fn vector_search(
            &self,
            query_vector: Vec<f32>,
            limit: usize,
            min_score: f32,
            _node_label: &str,
            _filters: Value,
        ) -> Result<Vec<Value>> {
            let cypher = "CALL db.index.vector.queryNodes($index_name, $limit, $query_vector) \
                          YIELD node, score \
                          WHERE score >= $min_score \
                          RETURN node, score";

            let params = json!({
                "index_name": "code_embedding_index",
                "limit": limit,
                "query_vector": query_vector,
                "min_score": min_score,
            });

            let rows = self.run_query(cypher, &params).await?;

            // Map Bolt rows into the canonical search-result shape.
            let results: Vec<Value> = rows
                .into_iter()
                .map(|row| {
                    // row is expected to be [node, score]
                    let node = row.get(0).cloned().unwrap_or(Value::Null);
                    let score = row.get(1).cloned().unwrap_or(Value::Null);
                    json!({
                        "file_path":     node.get("file_path").cloned().unwrap_or(Value::Null),
                        "content":       node.get("content").cloned().unwrap_or(Value::Null),
                        "score":         score,
                        "vector_score":  score,
                        "keyword_score": Value::Null,
                        "start_line":    node.get("start_line").cloned().unwrap_or(Value::Null),
                        "end_line":      node.get("end_line").cloned().unwrap_or(Value::Null),
                        "language":      node.get("language").cloned().unwrap_or(Value::Null),
                        "project":       node.get("project").cloned().unwrap_or(Value::Null),
                        "root_path":     node.get("root_path").cloned().unwrap_or(Value::Null),
                        "indexed_at":    node.get("indexed_at").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect();

            Ok(results)
        }

        async fn store_nodes(&self, nodes: Vec<Value>, node_label: &str) -> Result<usize> {
            const BATCH_SIZE: usize = 500;
            let mut total = 0usize;

            for chunk in nodes.chunks(BATCH_SIZE) {
                let batch: Vec<Value> = chunk.to_vec();
                let cypher = format!(
                    "UNWIND $batch AS item \
                     MERGE (n:{node_label} {{file_path: item.file_path, start_line: item.start_line}}) \
                     SET n += item"
                );
                let params = json!({ "batch": batch });
                self.run_query(&cypher, &params)
                    .await
                    .context("Failed to store node batch via Bolt")?;
                total += chunk.len();
            }

            tracing::debug!("Bolt: stored {} nodes (label: {})", total, node_label);
            Ok(total)
        }

        async fn delete_nodes(
            &self,
            node_label: &str,
            property: &str,
            value: &str,
        ) -> Result<usize> {
            let cypher = format!(
                "MATCH (n:{node_label} {{{property}: $value}}) \
                 WITH n, count(n) AS cnt \
                 DETACH DELETE n \
                 RETURN cnt"
            );
            let params = json!({ "value": value });
            let rows = self
                .run_query(&cypher, &params)
                .await
                .context("Failed to delete nodes via Bolt")?;

            let count = rows
                .first()
                .and_then(|r| r.get(0))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;

            Ok(count)
        }

        async fn count_nodes(
            &self,
            node_label: &str,
            property: &str,
            value: &str,
        ) -> Result<usize> {
            let cypher = format!(
                "MATCH (n:{node_label}) WHERE n.{property} = $value RETURN count(n) AS cnt"
            );
            let params = json!({ "value": value });
            let rows = self
                .run_query(&cypher, &params)
                .await
                .context("Failed to count nodes via Bolt")?;

            let count = rows
                .first()
                .and_then(|r| r.get(0))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;

            Ok(count)
        }

        async fn distinct_property(
            &self,
            node_label: &str,
            property: &str,
            filter_prop: &str,
            filter_val: &str,
        ) -> Result<Vec<String>> {
            let cypher = format!(
                "MATCH (n:{node_label}) WHERE n.{filter_prop} = $filter_val \
                 RETURN DISTINCT n.{property} AS val"
            );
            let params = json!({ "filter_val": filter_val });
            let rows = self
                .run_query(&cypher, &params)
                .await
                .context("Failed to query distinct property via Bolt")?;

            let values = rows
                .iter()
                .filter_map(|r| r.get(0).and_then(Value::as_str).map(String::from))
                .collect();

            Ok(values)
        }

        fn transport_name(&self) -> &'static str {
            "Bolt"
        }
    }
}

#[cfg(feature = "nornicdb-bolt")]
pub(crate) use bolt::BoltTransport;

// ════════════════════════════════════════════════════════════════════════
//  gRPC Transport  (requires `nornicdb-grpc` feature)
// ════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nornicdb-grpc")]
mod grpc {
    use super::*;
    use qdrant_client::qdrant::{
        Condition, CountPointsBuilder, DeletePointsBuilder, Filter, PointStruct,
        ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    };
    use qdrant_client::{Payload, Qdrant};
    use std::collections::HashSet;

    const COLLECTION_NAME: &str = "code_embeddings";

    /// gRPC transport targeting a Qdrant-compatible endpoint.
    ///
    /// Cypher operations are **not** supported over gRPC; only vector
    /// operations are available.
    pub(crate) struct GrpcTransport {
        client: Qdrant,
    }

    impl GrpcTransport {
        /// Connect to a Qdrant-compatible gRPC endpoint.
        pub(crate) async fn new(url: &str) -> Result<Self> {
            tracing::info!("Connecting to NornicDB via gRPC at {}", url);
            let client = Qdrant::from_url(url)
                .build()
                .context("Failed to build gRPC client for NornicDB")?;
            Ok(Self { client })
        }

        /// Convert a scored Qdrant point into the canonical search-result
        /// JSON shape.
        fn scored_point_to_value(point: &qdrant_client::qdrant::ScoredPoint) -> Value {
            let p = &point.payload;
            let get_str = |key: &str| -> Value {
                p.get(key)
                    .and_then(|v| v.as_str())
                    .map(|s| Value::String(s.to_string()))
                    .unwrap_or(Value::Null)
            };
            let get_int = |key: &str| -> Value {
                p.get(key)
                    .and_then(|v| v.as_integer())
                    .map(|n| json!(n))
                    .unwrap_or(Value::Null)
            };

            json!({
                "file_path":     get_str("file_path"),
                "content":       get_str("content"),
                "score":         point.score,
                "vector_score":  point.score,
                "keyword_score": Value::Null,
                "start_line":    get_int("start_line"),
                "end_line":      get_int("end_line"),
                "language":      get_str("language"),
                "project":       get_str("project"),
                "root_path":     get_str("root_path"),
                "indexed_at":    get_int("indexed_at"),
            })
        }

        /// Build a Qdrant `Filter` from a serde_json filters object.
        ///
        /// Supported keys: `project`, `root_path`, `file_path`.  All are
        /// treated as exact-match string conditions.
        fn build_filter(filters: &Value) -> Option<Filter> {
            let obj = filters.as_object()?;
            let mut conditions: Vec<Condition> = Vec::new();

            for key in &["project", "root_path", "file_path"] {
                if let Some(val) = obj.get(*key).and_then(Value::as_str) {
                    conditions.push(Condition::matches(*key, val.to_string()));
                }
            }

            if conditions.is_empty() {
                None
            } else {
                Some(Filter::must(conditions))
            }
        }
    }

    #[async_trait::async_trait]
    impl NornicTransport for GrpcTransport {
        async fn health_check(&self) -> Result<bool> {
            match self.client.list_collections().await {
                Ok(_) => Ok(true),
                Err(e) => {
                    tracing::warn!("gRPC health check failed: {}", e);
                    Ok(false)
                }
            }
        }

        async fn execute_cypher(&self, _query: &str, _params: Value) -> Result<Vec<Value>> {
            Err(anyhow::anyhow!(
                "Cypher queries are not supported over gRPC. Use REST or Bolt transport."
            ))
        }

        async fn hybrid_search(
            &self,
            _query_text: &str,
            query_vector: Vec<f32>,
            limit: usize,
            min_score: f32,
            node_label: &str,
            filters: Value,
        ) -> Result<Vec<Value>> {
            tracing::debug!(
                "gRPC transport: hybrid_search delegates to vector_search (no BM25 via gRPC)"
            );
            self.vector_search(query_vector, limit, min_score, node_label, filters)
                .await
        }

        async fn vector_search(
            &self,
            query_vector: Vec<f32>,
            limit: usize,
            min_score: f32,
            _node_label: &str,
            filters: Value,
        ) -> Result<Vec<Value>> {
            let mut builder = SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit as u64)
                .score_threshold(min_score)
                .with_payload(true);

            if let Some(filter) = Self::build_filter(&filters) {
                builder = builder.filter(filter);
            }

            let search_result = self
                .client
                .search_points(builder)
                .await
                .context("gRPC vector search failed")?;

            let results: Vec<Value> = search_result
                .result
                .iter()
                .map(Self::scored_point_to_value)
                .collect();

            Ok(results)
        }

        async fn store_nodes(&self, nodes: Vec<Value>, _node_label: &str) -> Result<usize> {
            if nodes.is_empty() {
                return Ok(0);
            }

            let count = nodes.len();
            let points: Vec<PointStruct> = nodes
                .into_iter()
                .enumerate()
                .filter_map(|(idx, node)| {
                    // Extract the embedding vector; skip nodes without one.
                    let embedding: Vec<f32> =
                        node.get("embedding").and_then(Value::as_array).map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })?;

                    let payload: Payload = node
                        .clone()
                        .try_into()
                        .expect("JSON object always converts to Payload");

                    Some(PointStruct::new(idx as u64, embedding, payload))
                })
                .collect();

            let written = points.len();
            self.client
                .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
                .await
                .context("gRPC upsert_points failed")?;

            tracing::debug!("gRPC: stored {}/{} nodes", written, count);
            Ok(written)
        }

        async fn delete_nodes(
            &self,
            _node_label: &str,
            property: &str,
            value: &str,
        ) -> Result<usize> {
            let filter = Filter::must([Condition::matches(property, value.to_string())]);

            self.client
                .delete_points(DeletePointsBuilder::new(COLLECTION_NAME).points(filter))
                .await
                .context("gRPC delete_points failed")?;

            // Qdrant does not return a deleted count directly.
            tracing::debug!("gRPC: deleted nodes where {} = {}", property, value);
            Ok(0)
        }

        async fn count_nodes(
            &self,
            _node_label: &str,
            property: &str,
            value: &str,
        ) -> Result<usize> {
            let filter = Filter::must([Condition::matches(property, value.to_string())]);

            let result = self
                .client
                .count(CountPointsBuilder::new(COLLECTION_NAME).filter(filter))
                .await
                .context("gRPC count_points failed")?;

            let count = result.result.map(|r| r.count).unwrap_or(0) as usize;
            Ok(count)
        }

        async fn distinct_property(
            &self,
            _node_label: &str,
            property: &str,
            filter_prop: &str,
            filter_val: &str,
        ) -> Result<Vec<String>> {
            let filter = Filter::must([Condition::matches(filter_prop, filter_val.to_string())]);

            let mut distinct: HashSet<String> = HashSet::new();
            let mut offset: Option<qdrant_client::qdrant::PointId> = None;

            loop {
                let mut builder = ScrollPointsBuilder::new(COLLECTION_NAME)
                    .filter(filter.clone())
                    .with_payload(true)
                    .limit(500);

                if let Some(ref point_id) = offset {
                    builder = builder.offset(point_id.clone());
                }

                let scroll_result = self
                    .client
                    .scroll(builder)
                    .await
                    .context("gRPC scroll failed during distinct_property")?;

                if scroll_result.result.is_empty() {
                    break;
                }

                for point in &scroll_result.result {
                    if let Some(val) = point.payload.get(property).and_then(|v| v.as_str()) {
                        distinct.insert(val.to_string());
                    }
                }

                offset = scroll_result.next_page_offset;
                if offset.is_none() {
                    break;
                }
            }

            Ok(distinct.into_iter().collect())
        }

        fn transport_name(&self) -> &'static str {
            "gRPC"
        }
    }
}

#[cfg(feature = "nornicdb-grpc")]
pub(crate) use grpc::GrpcTransport;
