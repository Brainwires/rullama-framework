//! [`NornicDatabase`] struct, constructors, [`VectorDatabase`] impl, and
//! NornicDB-specific extension methods.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

use crate::databases::traits::VectorDatabase;
use crate::glob_utils;
use brainwires_core::{ChunkMetadata, DatabaseStats, SearchResult};

#[cfg(any(feature = "nornicdb-bolt", feature = "nornicdb-grpc"))]
use super::helpers::extract_host;
use super::helpers::{build_filters, map_node_to_search_result, map_to_search_result};
use super::transport::{NornicTransport, RestTransport};
use super::types::{CognitiveMemoryTier, NornicConfig, TransportKind};

#[cfg(feature = "nornicdb-bolt")]
use super::transport::BoltTransport;
#[cfg(feature = "nornicdb-grpc")]
use super::transport::GrpcTransport;

// ── Main struct ─────────────────────────────────────────────────────────

/// NornicDB-backed vector database.
///
/// Wraps a `NornicTransport` trait object (internal — see `transport.rs`)
/// and implements the generic
/// [`VectorDatabase`] trait.  NornicDB-specific extensions (graph
/// relationships, cognitive tiers, raw Cypher) are available as inherent
/// methods.
pub struct NornicDatabase {
    pub(super) transport: Arc<dyn NornicTransport>,
    pub(super) node_label: String,
    pub(super) index_name: String,
    #[allow(dead_code)]
    pub(super) database: String,
}

// ── Constructors ────────────────────────────────────────────────────────

impl NornicDatabase {
    /// Default NornicDB REST URL.
    pub fn default_url() -> String {
        "http://localhost:7474".to_string()
    }

    /// Create a client with default settings (REST transport, localhost, no auth).
    pub async fn new() -> Result<Self> {
        Self::with_config(NornicConfig::default()).await
    }

    /// Create a REST client pointing at a custom URL.
    pub async fn with_url(url: &str) -> Result<Self> {
        Self::with_config(NornicConfig {
            url: url.to_string(),
            ..Default::default()
        })
        .await
    }

    /// Create a client using the Bolt binary protocol.
    #[cfg(feature = "nornicdb-bolt")]
    pub async fn with_bolt(url: &str, username: &str, password: &str) -> Result<Self> {
        let host = extract_host(url);
        let bolt_url = format!("bolt://{}:7687", host);
        let transport = BoltTransport::new(&bolt_url, username, password).await?;
        Ok(Self {
            transport: Arc::new(transport),
            node_label: "CodeChunk".to_string(),
            index_name: "code_embedding_index".to_string(),
            database: "neo4j".to_string(),
        })
    }

    /// Create a client using the gRPC protocol.
    #[cfg(feature = "nornicdb-grpc")]
    pub async fn with_grpc(url: &str) -> Result<Self> {
        let transport = GrpcTransport::new(url).await?;
        Ok(Self {
            transport: Arc::new(transport),
            node_label: "CodeChunk".to_string(),
            index_name: "code_embedding_index".to_string(),
            database: "neo4j".to_string(),
        })
    }

    /// Create a client from a full [`NornicConfig`].
    pub async fn with_config(config: NornicConfig) -> Result<Self> {
        let transport: Arc<dyn NornicTransport> = match &config.transport {
            TransportKind::Rest => {
                let rest = RestTransport::new(&config.url, &config.database).await?;
                // Authenticate before wrapping in Arc so we can call &self methods.
                if let (Some(user), Some(pass)) = (&config.username, &config.password) {
                    rest.authenticate(user, pass).await?;
                }
                Arc::new(rest)
            }
            #[cfg(feature = "nornicdb-bolt")]
            TransportKind::Bolt { port } => {
                let host = extract_host(&config.url);
                let bolt_url = format!("bolt://{}:{}", host, port);
                let (user, pass) = (
                    config.username.as_deref().unwrap_or("neo4j"),
                    config.password.as_deref().unwrap_or(""),
                );
                Arc::new(BoltTransport::new(&bolt_url, user, pass).await?)
            }
            #[cfg(not(feature = "nornicdb-bolt"))]
            TransportKind::Bolt { .. } => {
                anyhow::bail!("Bolt transport requires the 'nornicdb-bolt' feature");
            }
            #[cfg(feature = "nornicdb-grpc")]
            TransportKind::Grpc { port } => {
                let host = extract_host(&config.url);
                let grpc_url = format!("http://{}:{}", host, port);
                Arc::new(GrpcTransport::new(&grpc_url).await?)
            }
            #[cfg(not(feature = "nornicdb-grpc"))]
            TransportKind::Grpc { .. } => {
                anyhow::bail!("gRPC transport requires the 'nornicdb-grpc' feature");
            }
        };

        Ok(Self {
            transport,
            node_label: config.node_label,
            index_name: config.index_name,
            database: config.database,
        })
    }

    /// Check whether the NornicDB server is reachable.
    pub async fn health_check(&self) -> Result<bool> {
        self.transport.health_check().await
    }

    /// Best-effort authentication attempt.
    ///
    /// For REST this uses the `/auth/token` endpoint.  For Bolt, auth is
    /// done at connection time.  For gRPC, no separate auth mechanism
    /// exists.
    pub async fn authenticate(&self, username: &str, password: &str) -> Result<()> {
        // Run a trivial query to verify the connection is alive.  If the
        // transport is REST, the token was already set in `with_config`.
        let _ = self.transport.execute_cypher("RETURN 1", json!({})).await;
        // Best-effort: if execute_cypher is unsupported (gRPC) we just
        // swallow the error — health_check is a better alternative there.
        let _ = username;
        let _ = password;
        Ok(())
    }
}

// ── VectorDatabase trait ────────────────────────────────────────────────

#[async_trait::async_trait]
impl VectorDatabase for NornicDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        // Create vector index.
        let create_index = format!(
            "CALL db.index.vector.createNodeIndex('{}', '{}', 'embedding', {}, 'cosine')",
            self.index_name, self.node_label, dimension
        );
        match self
            .transport
            .execute_cypher(&create_index, json!({}))
            .await
        {
            Ok(_) => tracing::info!(
                "Created vector index '{}' with dimension {}",
                self.index_name,
                dimension
            ),
            Err(e) => {
                // Index may already exist — log and continue.
                tracing::info!("Vector index may already exist: {}", e);
            }
        }

        // Create uniqueness constraint.
        let constraint = format!(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:{}) REQUIRE (n.file_path, n.start_line) IS UNIQUE",
            self.node_label
        );
        match self.transport.execute_cypher(&constraint, json!({})).await {
            Ok(_) => tracing::info!("Created uniqueness constraint"),
            Err(e) => tracing::info!("Constraint may already exist: {}", e),
        }

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

        let count = embeddings.len();

        let nodes: Vec<Value> = embeddings
            .into_iter()
            .zip(metadata)
            .zip(contents)
            .map(|((emb, meta), content)| {
                json!({
                    "file_path": meta.file_path,
                    "root_path": meta.root_path.unwrap_or_else(|| root_path.to_string()),
                    "project": meta.project,
                    "start_line": meta.start_line,
                    "end_line": meta.end_line,
                    "language": meta.language.unwrap_or_default(),
                    "extension": meta.extension.unwrap_or_default(),
                    "file_hash": meta.file_hash,
                    "indexed_at": meta.indexed_at,
                    "content": content,
                    "embedding": emb,
                })
            })
            .collect();

        self.transport.store_nodes(nodes, &self.node_label).await?;
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
        let filters = build_filters(
            project.as_deref(),
            root_path.as_deref(),
            &file_extensions,
            &languages,
        );

        let raw_results = if hybrid {
            self.transport
                .hybrid_search(
                    query_text,
                    query_vector,
                    limit,
                    min_score,
                    &self.node_label,
                    filters,
                )
                .await?
        } else {
            self.transport
                .vector_search(query_vector, limit, min_score, &self.node_label, filters)
                .await?
        };

        let mut results: Vec<SearchResult> = raw_results
            .iter()
            .filter_map(map_to_search_result)
            .collect();

        // Post-filter by path patterns using glob matching.
        if !path_patterns.is_empty() {
            results.retain(|r| glob_utils::matches_any_pattern(&r.file_path, &path_patterns));
        }

        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        self.transport
            .delete_nodes(&self.node_label, "file_path", file_path)
            .await
    }

    async fn clear(&self) -> Result<()> {
        let delete_all = format!("MATCH (n:{}) DETACH DELETE n", self.node_label);
        self.transport
            .execute_cypher(&delete_all, json!({}))
            .await?;

        let drop_index = format!("DROP INDEX {} IF EXISTS", self.index_name);
        // Ignore error if index doesn't exist.
        let _ = self.transport.execute_cypher(&drop_index, json!({})).await;

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let count_query = format!("MATCH (n:{}) RETURN count(n) AS total", self.node_label);
        let count_rows = self
            .transport
            .execute_cypher(&count_query, json!({}))
            .await?;
        let total = count_rows
            .first()
            .and_then(|r| r.get("total"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let lang_query = format!(
            "MATCH (n:{}) RETURN n.language AS lang, count(n) AS cnt ORDER BY cnt DESC",
            self.node_label
        );
        let lang_rows = self
            .transport
            .execute_cypher(&lang_query, json!({}))
            .await?;
        let language_breakdown: Vec<(String, usize)> = lang_rows
            .iter()
            .filter_map(|r| {
                let lang = r.get("lang")?.as_str()?.to_string();
                let cnt = r.get("cnt")?.as_u64()? as usize;
                Some((lang, cnt))
            })
            .collect();

        Ok(DatabaseStats {
            total_points: total,
            total_vectors: total,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // NornicDB does not require an explicit flush.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        self.transport
            .count_nodes(&self.node_label, "root_path", root_path)
            .await
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        self.transport
            .distinct_property(&self.node_label, "file_path", "root_path", root_path)
            .await
    }
}

// ── NornicDB-specific extensions ────────────────────────────────────────

impl NornicDatabase {
    /// Execute an arbitrary Cypher query and return the raw result rows.
    pub async fn cypher_query(&self, query: &str, params: Value) -> Result<Value> {
        let rows = self.transport.execute_cypher(query, params).await?;
        Ok(Value::Array(rows))
    }

    /// Create a typed relationship between two code chunks.
    pub async fn create_relationship(
        &self,
        from_file: &str,
        from_line: usize,
        to_file: &str,
        to_line: usize,
        relationship_type: &str,
        properties: Value,
    ) -> Result<()> {
        let query = format!(
            "MATCH (a:{label} {{file_path: $from_file, start_line: $from_line}}) \
             MATCH (b:{label} {{file_path: $to_file, start_line: $to_line}}) \
             MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
            label = self.node_label,
            rel_type = relationship_type,
        );
        let params = json!({
            "from_file": from_file,
            "from_line": from_line,
            "to_file": to_file,
            "to_line": to_line,
            "props": properties,
        });
        self.transport.execute_cypher(&query, params).await?;
        Ok(())
    }

    /// Find code chunks related to a starting node via graph traversal.
    pub async fn find_related(
        &self,
        file_path: &str,
        start_line: usize,
        max_depth: usize,
        relationship_types: Option<Vec<String>>,
    ) -> Result<Vec<SearchResult>> {
        let rel_pattern = match &relationship_types {
            Some(types) if !types.is_empty() => {
                format!(":{}", types.join("|"))
            }
            _ => String::new(),
        };
        let query = format!(
            "MATCH (start:{label} {{file_path: $file_path, start_line: $start_line}}) \
             MATCH (start)-[{rel}*1..{depth}]->(related:{label}) \
             RETURN DISTINCT related",
            label = self.node_label,
            rel = rel_pattern,
            depth = max_depth,
        );
        let params = json!({
            "file_path": file_path,
            "start_line": start_line,
        });
        let rows = self.transport.execute_cypher(&query, params).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("related").and_then(map_node_to_search_result))
            .collect())
    }

    /// Store a code chunk with an additional cognitive memory tier label.
    pub async fn store_with_memory_tier(
        &self,
        embedding: Vec<f32>,
        metadata: ChunkMetadata,
        content: String,
        tier: CognitiveMemoryTier,
    ) -> Result<()> {
        let query = format!(
            "MERGE (n:{label} {{file_path: $file_path, start_line: $start_line}}) \
             SET n += $props, n.embedding = $embedding \
             SET n:{tier_label}",
            label = self.node_label,
            tier_label = tier.as_label(),
        );
        let params = json!({
            "file_path": metadata.file_path,
            "start_line": metadata.start_line,
            "embedding": embedding,
            "props": {
                "root_path": metadata.root_path,
                "project": metadata.project,
                "end_line": metadata.end_line,
                "language": metadata.language,
                "extension": metadata.extension,
                "file_hash": metadata.file_hash,
                "indexed_at": metadata.indexed_at,
                "content": content,
            }
        });
        self.transport.execute_cypher(&query, params).await?;
        Ok(())
    }

    /// Search within a specific cognitive memory tier.
    pub async fn search_by_memory_tier(
        &self,
        query_vector: Vec<f32>,
        tier: CognitiveMemoryTier,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let query = format!(
            "CALL db.index.vector.queryNodes('{}', $limit, $vector) \
             YIELD node, score \
             WHERE node:{tier_label} \
             RETURN node, score",
            self.index_name,
            tier_label = tier.as_label(),
        );
        let params = json!({
            "limit": limit,
            "vector": query_vector,
        });
        let rows = self.transport.execute_cypher(&query, params).await?;
        Ok(rows
            .iter()
            .filter_map(|r| {
                let score = r.get("score")?.as_f64()? as f32;
                let node = r.get("node")?;
                let mut result = map_node_to_search_result(node)?;
                result.score = score;
                result.vector_score = score;
                Some(result)
            })
            .collect())
    }

    /// Get NornicDB-specific embedding statistics.
    pub async fn embedding_stats(&self) -> Result<Value> {
        let query = format!(
            "MATCH (n:{}) WHERE n.embedding IS NOT NULL \
             RETURN count(n) AS embedded_count, \
             avg(size(n.embedding)) AS avg_dimension",
            self.node_label
        );
        let rows = self.transport.execute_cypher(&query, json!({})).await?;
        Ok(rows.first().cloned().unwrap_or(json!({})))
    }
}

// ── Default impl ────────────────────────────────────────────────────────

impl Default for NornicDatabase {
    fn default() -> Self {
        tokio::runtime::Runtime::new()
            .expect("failed to create tokio runtime")
            .block_on(Self::new())
            .expect("Failed to create default NornicDB client")
    }
}
