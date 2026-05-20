//! SurrealDB backend for [`StorageBackend`] and [`VectorDatabase`] traits.
//!
//! Implements both generic CRUD operations and RAG-style embedding storage
//! using SurrealDB's native MTREE vector indexing with KNN search.
//!
//! # Requirements
//!
//! * A running SurrealDB server (v2.x+).
//! * The `surrealdb-backend` Cargo feature enabled on `brainwires-storage`.
//!
//! # Example
//!
//! ```rust,no_run
//! use brainwires_storage::databases::surrealdb::SurrealDatabase;
//! use brainwires_storage::databases::traits::VectorDatabase;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let db = SurrealDatabase::new("ws://localhost:8000", "brainwires", "storage").await?;
//! db.initialize(384).await?;
//! # Ok(())
//! # }
//! ```

use crate::databases::bm25_helpers::{self, SharedIdfStats};
use crate::databases::capabilities::BackendCapabilities;
use crate::databases::traits::{
    ChunkMetadata, DatabaseStats, SearchResult, StorageBackend, VectorDatabase,
};
use crate::databases::types::{FieldDef, FieldType, FieldValue, Filter, Record, ScoredRecord};
use crate::glob_utils;
use anyhow::{Context, Result};
use serde_json::json;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;

const DEFAULT_TABLE: &str = "code_embeddings";
const DEFAULT_URL: &str = "ws://localhost:8000";

/// SurrealDB backed vector database for code embeddings.
///
/// Uses MTREE indexing for approximate nearest-neighbour search with cosine
/// distance and client-side BM25 scoring for hybrid (vector + keyword) queries.
pub struct SurrealDatabase {
    db: Surreal<Any>,
    idf_stats: SharedIdfStats,
}

impl SurrealDatabase {
    /// Create a new client connected to a SurrealDB instance with default
    /// root credentials (`root` / `root`).
    ///
    /// Connects to the given URL and selects the given namespace and database.
    pub async fn new(url: &str, namespace: &str, database: &str) -> Result<Self> {
        Self::with_config(url, namespace, database, "root", "root").await
    }

    /// Create a new client with explicit credentials.
    pub async fn with_config(
        url: &str,
        namespace: &str,
        database: &str,
        username: &str,
        password: &str,
    ) -> Result<Self> {
        tracing::info!("Connecting to SurrealDB at {}", url);

        let db = surrealdb::engine::any::connect(url)
            .await
            .with_context(|| format!("Failed to connect to SurrealDB at {url}"))?;

        db.signin(surrealdb::opt::auth::Root { username, password })
            .await
            .context("Failed to sign in to SurrealDB")?;

        db.use_ns(namespace)
            .use_db(database)
            .await
            .context("Failed to select SurrealDB namespace/database")?;

        let instance = Self {
            db,
            idf_stats: bm25_helpers::new_shared_idf_stats(),
        };

        // Seed IDF stats from any existing rows.
        if let Err(e) = instance.refresh_idf_stats().await {
            tracing::warn!("Failed to initialize IDF stats: {}", e);
        }

        Ok(instance)
    }

    /// Return the default connection URL.
    pub fn default_url() -> String {
        DEFAULT_URL.to_string()
    }

    /// Return the backend capabilities.
    pub fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            vector_search: true,
        }
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Refresh IDF statistics by scanning all stored content.
    async fn refresh_idf_stats(&self) -> Result<()> {
        tracing::debug!("Refreshing IDF statistics from table '{}'", DEFAULT_TABLE);

        let mut result = self
            .db
            .query(format!("SELECT content FROM {DEFAULT_TABLE}"))
            .await
            .context("Failed to query content for IDF refresh")?;

        let rows: Vec<serde_json::Value> = match result.take(0) {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!("IDF refresh skipped (table may not exist): {}", e);
                return Ok(());
            }
        };

        let documents: Vec<String> = rows
            .iter()
            .filter_map(|row| {
                row.get("content")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();

        tracing::info!("Refreshing IDF stats from {} documents", documents.len());
        bm25_helpers::update_idf_stats(&self.idf_stats, &documents).await;

        Ok(())
    }

    /// Execute the core filtered search logic shared by `search` and
    /// `search_filtered`.
    #[allow(clippy::too_many_arguments)]
    async fn do_search(
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
            "Searching table '{}': limit={}, min_score={}, project={:?}, root_path={:?}, \
             hybrid={}, ext={:?}, lang={:?}, path={:?}",
            DEFAULT_TABLE,
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            file_extensions,
            languages,
            path_patterns,
        );

        // Build the WHERE clause dynamically.
        let mut conditions = Vec::new();
        if project.is_some() {
            conditions.push("project = $project".to_string());
        }
        if root_path.is_some() {
            conditions.push("root_path = $root_path".to_string());
        }
        if !file_extensions.is_empty() {
            conditions.push("extension IN $extensions".to_string());
        }
        if !languages.is_empty() {
            conditions.push("language IN $languages".to_string());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" AND {}", conditions.join(" AND "))
        };

        let query = format!(
            "SELECT *, vector::similarity::cosine(embedding, $query_vec) AS vector_score \
             FROM {table} \
             WHERE embedding <|{limit}|> $query_vec{where_clause} \
             ORDER BY vector_score DESC",
            table = DEFAULT_TABLE,
            limit = limit,
            where_clause = where_clause,
        );

        let mut stmt = self.db.query(&query).bind(("query_vec", query_vector));

        if let Some(p) = project.clone() {
            stmt = stmt.bind(("project", p));
        }
        if let Some(rp) = root_path.clone() {
            stmt = stmt.bind(("root_path", rp));
        }
        if !file_extensions.is_empty() {
            stmt = stmt.bind(("extensions", file_extensions.clone()));
        }
        if !languages.is_empty() {
            stmt = stmt.bind(("languages", languages.clone()));
        }

        let mut result = stmt.await.context("Failed to execute search query")?;

        let rows: Vec<serde_json::Value> =
            result.take(0).context("Failed to take search results")?;

        let mut results: Vec<SearchResult> = Vec::with_capacity(rows.len());

        for row in &rows {
            let vector_score = row
                .get("vector_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            if vector_score < min_score {
                continue;
            }

            let file_path = row
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let result_root_path = row
                .get("root_path")
                .and_then(|v| v.as_str())
                .map(String::from);
            let result_project = row
                .get("project")
                .and_then(|v| v.as_str())
                .map(String::from);
            let start_line = row.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let end_line = row.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let language = row
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();
            let indexed_at = row.get("indexed_at").and_then(|v| v.as_i64()).unwrap_or(0);
            let content = row
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

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

        // Post-filter by glob path patterns.
        if !path_patterns.is_empty() {
            results.retain(|r| glob_utils::matches_any_pattern(&r.file_path, &path_patterns));
        }

        Ok(results)
    }
}

// ── VectorDatabase trait implementation ──────────────────────────────────

#[async_trait::async_trait]
impl VectorDatabase for SurrealDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        tracing::info!(
            "Initializing SurrealDB table '{}' with vector dimension {}",
            DEFAULT_TABLE,
            dimension
        );

        let ddl = format!(
            r#"
            DEFINE TABLE IF NOT EXISTS {table} SCHEMAFULL;
            DEFINE FIELD embedding    ON {table} TYPE array<float, {dim}>;
            DEFINE FIELD file_path    ON {table} TYPE string;
            DEFINE FIELD root_path    ON {table} TYPE option<string>;
            DEFINE FIELD project      ON {table} TYPE option<string>;
            DEFINE FIELD start_line   ON {table} TYPE int;
            DEFINE FIELD end_line     ON {table} TYPE int;
            DEFINE FIELD language     ON {table} TYPE option<string>;
            DEFINE FIELD extension    ON {table} TYPE option<string>;
            DEFINE FIELD file_hash    ON {table} TYPE string;
            DEFINE FIELD indexed_at   ON {table} TYPE int;
            DEFINE FIELD content      ON {table} TYPE string;
            DEFINE INDEX idx_{table}_embedding ON {table} FIELDS embedding MTREE DIMENSION {dim} DIST COSINE TYPE F32;
            DEFINE INDEX idx_{table}_file_path ON {table} FIELDS file_path;
            DEFINE INDEX idx_{table}_root_path ON {table} FIELDS root_path;
            DEFINE INDEX idx_{table}_project   ON {table} FIELDS project;
            "#,
            table = DEFAULT_TABLE,
            dim = dimension,
        );

        self.db
            .query(&ddl)
            .await
            .context("Failed to initialize SurrealDB embeddings table")?;

        tracing::info!("SurrealDB table '{}' initialized", DEFAULT_TABLE);
        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        _root_path: &str,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let count = embeddings.len();
        tracing::debug!("Storing {} embeddings in '{}'", count, DEFAULT_TABLE);

        // Build a batch query with BEGIN TRANSACTION / COMMIT.
        let mut batch = String::from("BEGIN TRANSACTION;\n");

        for ((embedding, meta), content) in embeddings.into_iter().zip(metadata).zip(contents) {
            let record = json!({
                "embedding": embedding,
                "file_path": meta.file_path,
                "root_path": meta.root_path,
                "project": meta.project,
                "start_line": meta.start_line as i64,
                "end_line": meta.end_line as i64,
                "language": meta.language,
                "extension": meta.extension,
                "file_hash": meta.file_hash,
                "indexed_at": meta.indexed_at,
                "content": content,
            });

            // Escape the JSON for embedding in SurrealQL.
            let record_str =
                serde_json::to_string(&record).context("Failed to serialize embedding record")?;
            batch.push_str(&format!(
                "CREATE {table} CONTENT {record};\n",
                table = DEFAULT_TABLE,
                record = record_str,
            ));
        }

        batch.push_str("COMMIT TRANSACTION;\n");

        self.db
            .query(&batch)
            .await
            .context("Failed to store embeddings batch")?;

        tracing::info!("Stored {} embeddings in '{}'", count, DEFAULT_TABLE);

        // Refresh IDF statistics after adding new documents.
        if let Err(e) = self.refresh_idf_stats().await {
            tracing::warn!("Failed to refresh IDF stats after indexing: {}", e);
        }

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
        self.do_search(
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
        self.do_search(
            query_vector,
            query_text,
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            file_extensions,
            languages,
            path_patterns,
        )
        .await
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        tracing::debug!("Deleting embeddings for file: {}", file_path);

        let mut result = self
            .db
            .query(format!(
                "DELETE FROM {table} WHERE file_path = $path",
                table = DEFAULT_TABLE,
            ))
            .bind(("path", file_path.to_string()))
            .await
            .context("Failed to delete embeddings by file path")?;

        // SurrealDB DELETE returns the deleted records; count them.
        let deleted: Vec<serde_json::Value> = result.take(0).unwrap_or_default();
        let count = deleted.len();

        tracing::info!("Deleted {} rows for file '{}'", count, file_path);
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        tracing::info!("Clearing all embeddings from table '{}'", DEFAULT_TABLE);

        self.db
            .query(format!("DELETE FROM {DEFAULT_TABLE}"))
            .await
            .context("Failed to clear embeddings table")?;

        // Clear IDF stats.
        let mut stats = self.idf_stats.write().await;
        stats.total_docs = 0;
        stats.doc_frequencies.clear();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        tracing::debug!("Fetching statistics for table '{}'", DEFAULT_TABLE);

        // Total row count.
        let mut result = self
            .db
            .query(format!(
                "SELECT count() AS total FROM {table} GROUP ALL",
                table = DEFAULT_TABLE,
            ))
            .await
            .context("Failed to count rows")?;

        let count_rows: Vec<serde_json::Value> = result.take(0).unwrap_or_default();
        let total = count_rows
            .first()
            .and_then(|r| r.get("total"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Per-language breakdown.
        let mut lang_result = self
            .db
            .query(format!(
                "SELECT language, count() AS lang_count FROM {table} GROUP BY language",
                table = DEFAULT_TABLE,
            ))
            .await
            .context("Failed to fetch language breakdown")?;

        let lang_rows: Vec<serde_json::Value> = lang_result.take(0).unwrap_or_default();
        let language_breakdown: Vec<(String, usize)> = lang_rows
            .iter()
            .map(|row| {
                let lang = row
                    .get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();
                let cnt = row.get("lang_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                (lang, cnt)
            })
            .collect();

        Ok(DatabaseStats {
            total_points: total,
            total_vectors: total,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // SurrealDB persists transactionally — no explicit flush needed.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let mut result = self
            .db
            .query(format!(
                "SELECT count() AS total FROM {table} WHERE root_path = $rp GROUP ALL",
                table = DEFAULT_TABLE,
            ))
            .bind(("rp", root_path.to_string()))
            .await
            .context("Failed to count rows by root_path")?;

        let rows: Vec<serde_json::Value> = result.take(0).unwrap_or_default();
        let count = rows
            .first()
            .and_then(|r| r.get("total"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let mut result = self
            .db
            .query(format!(
                "SELECT file_path FROM {table} WHERE root_path = $rp GROUP BY file_path",
                table = DEFAULT_TABLE,
            ))
            .bind(("rp", root_path.to_string()))
            .await
            .context("Failed to fetch indexed files")?;

        let rows: Vec<serde_json::Value> = result.take(0).unwrap_or_default();
        let files: Vec<String> = rows
            .iter()
            .filter_map(|row| {
                row.get("file_path")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();

        Ok(files)
    }
}

// ── StorageBackend trait implementation ───────────────────────────────

/// Convert a [`FieldValue`] to a [`serde_json::Value`] for SurrealDB queries.
fn field_value_to_json(val: &FieldValue) -> serde_json::Value {
    match val {
        FieldValue::Utf8(Some(s)) => json!(s),
        FieldValue::Utf8(None) => serde_json::Value::Null,
        FieldValue::Int32(Some(v)) => json!(*v),
        FieldValue::Int32(None) => serde_json::Value::Null,
        FieldValue::Int64(Some(v)) => json!(*v),
        FieldValue::Int64(None) => serde_json::Value::Null,
        FieldValue::UInt32(Some(v)) => json!(*v),
        FieldValue::UInt32(None) => serde_json::Value::Null,
        FieldValue::UInt64(Some(v)) => json!(*v),
        FieldValue::UInt64(None) => serde_json::Value::Null,
        FieldValue::Float32(Some(v)) => json!(*v),
        FieldValue::Float32(None) => serde_json::Value::Null,
        FieldValue::Float64(Some(v)) => json!(*v),
        FieldValue::Float64(None) => serde_json::Value::Null,
        FieldValue::Boolean(Some(v)) => json!(*v),
        FieldValue::Boolean(None) => serde_json::Value::Null,
        FieldValue::Vector(v) => json!(v),
    }
}

/// Convert a SurrealQL type name for a [`FieldType`].
fn field_type_to_surrealql(ft: &FieldType) -> String {
    match ft {
        FieldType::Utf8 => "string".to_string(),
        FieldType::Int32 | FieldType::Int64 | FieldType::UInt32 | FieldType::UInt64 => {
            "int".to_string()
        }
        FieldType::Float32 | FieldType::Float64 => "float".to_string(),
        FieldType::Boolean => "bool".to_string(),
        FieldType::Vector(n) => format!("array<float, {n}>"),
    }
}

/// Convert a [`Filter`] tree into a SurrealQL WHERE clause fragment
/// with named bind parameters (`$p0`, `$p1`, ...).
///
/// Returns `(sql_fragment, bindings)` where bindings is a vec of
/// `(param_name, json_value)` pairs.
fn filter_to_surrealql(
    filter: &Filter,
    param_offset: &mut usize,
) -> (String, Vec<(String, serde_json::Value)>) {
    match filter {
        Filter::Eq(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} = ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::Ne(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} != ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::Lt(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} < ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::Lte(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} <= ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::Gt(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} > ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::Gte(col, val) => {
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            (
                format!("{col} >= ${name}"),
                vec![(name, field_value_to_json(val))],
            )
        }
        Filter::NotNull(col) => (format!("{col} IS NOT NULL"), vec![]),
        Filter::IsNull(col) => (format!("{col} IS NULL"), vec![]),
        Filter::In(col, values) => {
            if values.is_empty() {
                return ("false".to_string(), vec![]);
            }
            let name = format!("p{}", *param_offset);
            *param_offset += 1;
            let json_arr: Vec<serde_json::Value> = values.iter().map(field_value_to_json).collect();
            (
                format!("{col} IN ${name}"),
                vec![(name, serde_json::Value::Array(json_arr))],
            )
        }
        Filter::And(filters) => {
            if filters.is_empty() {
                return ("true".to_string(), vec![]);
            }
            let mut parts = Vec::new();
            let mut all_bindings = Vec::new();
            for f in filters {
                let (sql, bindings) = filter_to_surrealql(f, param_offset);
                parts.push(sql);
                all_bindings.extend(bindings);
            }
            (format!("({})", parts.join(" AND ")), all_bindings)
        }
        Filter::Or(filters) => {
            if filters.is_empty() {
                return ("false".to_string(), vec![]);
            }
            let mut parts = Vec::new();
            let mut all_bindings = Vec::new();
            for f in filters {
                let (sql, bindings) = filter_to_surrealql(f, param_offset);
                parts.push(sql);
                all_bindings.extend(bindings);
            }
            (format!("({})", parts.join(" OR ")), all_bindings)
        }
        Filter::Raw(raw) => (raw.clone(), vec![]),
    }
}

/// Parse a JSON row from SurrealDB into a [`Record`].
///
/// Attempts to infer field types from JSON value types. Vectors are stored
/// as JSON arrays of numbers.
fn json_row_to_record(row: &serde_json::Value) -> Record {
    let obj = match row.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    let mut record = Vec::with_capacity(obj.len());
    for (key, val) in obj {
        // Skip SurrealDB internal `id` field (record link like `table:ID`).
        if key == "id" {
            // Still include it but as a string.
            let id_str = match val {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(_) => {
                    // SurrealDB record IDs can be objects like { "tb": "x", "id": { ... } }
                    serde_json::to_string(val).ok()
                }
                _ => Some(val.to_string()),
            };
            record.push((key.clone(), FieldValue::Utf8(id_str)));
            continue;
        }

        let field_value = match val {
            serde_json::Value::Null => FieldValue::Utf8(None),
            serde_json::Value::Bool(b) => FieldValue::Boolean(Some(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    FieldValue::Int64(Some(i))
                } else if let Some(f) = n.as_f64() {
                    FieldValue::Float64(Some(f))
                } else {
                    FieldValue::Utf8(Some(n.to_string()))
                }
            }
            serde_json::Value::String(s) => FieldValue::Utf8(Some(s.clone())),
            serde_json::Value::Array(arr) => {
                // Try to interpret as a float vector.
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                if floats.len() == arr.len() && !arr.is_empty() {
                    FieldValue::Vector(floats)
                } else {
                    // Fall back to JSON string representation.
                    FieldValue::Utf8(serde_json::to_string(val).ok())
                }
            }
            serde_json::Value::Object(_) => FieldValue::Utf8(serde_json::to_string(val).ok()),
        };
        record.push((key.clone(), field_value));
    }

    record
}

/// Bind a vec of `(name, json_value)` pairs to a SurrealDB query builder.
///
/// Because the SurrealDB SDK's `.bind()` method is generic and consumes
/// self, we need to chain bindings in a loop. This macro-like helper
/// builds the full query string with embedded JSON literals for the bind
/// values, since `.bind()` with dynamic names requires a workaround.
///
/// We use `.bind(("name", value))` chaining.
async fn execute_with_bindings(
    db: &Surreal<Any>,
    query: &str,
    bindings: Vec<(String, serde_json::Value)>,
) -> Result<Vec<serde_json::Value>> {
    let mut stmt = db.query(query);
    for (name, value) in bindings {
        stmt = stmt.bind((name, value));
    }
    let mut result = stmt.await.context("Failed to execute SurrealQL query")?;
    let rows: Vec<serde_json::Value> = result.take(0).unwrap_or_default();
    Ok(rows)
}

/// Execute a statement that does not return meaningful rows (e.g. DELETE, CREATE).
async fn execute_void_with_bindings(
    db: &Surreal<Any>,
    query: &str,
    bindings: Vec<(String, serde_json::Value)>,
) -> Result<()> {
    let mut stmt = db.query(query);
    for (name, value) in bindings {
        stmt = stmt.bind((name, value));
    }
    stmt.await.context("Failed to execute SurrealQL query")?;
    Ok(())
}

#[async_trait::async_trait]
impl StorageBackend for SurrealDatabase {
    async fn ensure_table(&self, table_name: &str, schema: &[FieldDef]) -> Result<()> {
        let mut ddl = format!("DEFINE TABLE IF NOT EXISTS {table_name} SCHEMAFULL;\n");

        for field in schema {
            let surreal_type = field_type_to_surrealql(&field.field_type);
            let type_expr = if field.nullable {
                format!("option<{surreal_type}>")
            } else {
                surreal_type.clone()
            };
            ddl.push_str(&format!(
                "DEFINE FIELD {name} ON {table_name} TYPE {type_expr};\n",
                name = field.name,
            ));

            // Create MTREE index for vector fields.
            if let FieldType::Vector(dim) = field.field_type {
                ddl.push_str(&format!(
                    "DEFINE INDEX idx_{table_name}_{name} ON {table_name} FIELDS {name} MTREE DIMENSION {dim} DIST COSINE TYPE F32;\n",
                    name = field.name,
                ));
            }
        }

        self.db
            .query(&ddl)
            .await
            .with_context(|| format!("Failed to create table '{table_name}'"))?;

        Ok(())
    }

    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut batch = String::from("BEGIN TRANSACTION;\n");

        for record in &records {
            let mut obj = serde_json::Map::new();
            for (name, value) in record {
                obj.insert(name.clone(), field_value_to_json(value));
            }
            let record_json = serde_json::to_string(&serde_json::Value::Object(obj))
                .context("Failed to serialize record")?;
            batch.push_str(&format!("CREATE {table_name} CONTENT {record_json};\n"));
        }

        batch.push_str("COMMIT TRANSACTION;\n");

        self.db
            .query(&batch)
            .await
            .with_context(|| format!("Failed to insert into '{table_name}'"))?;

        Ok(())
    }

    async fn query(
        &self,
        table_name: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<Record>> {
        let mut query = format!("SELECT * FROM {table_name}");
        let mut bindings = Vec::new();

        if let Some(f) = filter {
            let mut offset = 0usize;
            let (where_sql, where_bindings) = filter_to_surrealql(f, &mut offset);
            query.push_str(&format!(" WHERE {where_sql}"));
            bindings = where_bindings;
        }

        if let Some(n) = limit {
            query.push_str(&format!(" LIMIT {n}"));
        }

        let rows = execute_with_bindings(&self.db, &query, bindings).await?;
        Ok(rows.iter().map(json_row_to_record).collect())
    }

    async fn delete(&self, table_name: &str, filter: &Filter) -> Result<()> {
        let mut offset = 0usize;
        let (where_sql, bindings) = filter_to_surrealql(filter, &mut offset);
        let query = format!("DELETE FROM {table_name} WHERE {where_sql}");

        execute_void_with_bindings(&self.db, &query, bindings).await?;
        Ok(())
    }

    async fn count(&self, table_name: &str, filter: Option<&Filter>) -> Result<usize> {
        let mut query = format!("SELECT count() AS total FROM {table_name}");
        let mut bindings = Vec::new();

        if let Some(f) = filter {
            let mut offset = 0usize;
            let (where_sql, where_bindings) = filter_to_surrealql(f, &mut offset);
            query.push_str(&format!(" WHERE {where_sql}"));
            bindings = where_bindings;
        }

        query.push_str(" GROUP ALL");

        let rows = execute_with_bindings(&self.db, &query, bindings).await?;
        let count = rows
            .first()
            .and_then(|r| r.get("total"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        Ok(count)
    }

    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredRecord>> {
        let mut bindings: Vec<(String, serde_json::Value)> =
            vec![("query_vec".to_string(), json!(vector))];

        let mut where_extra = String::new();
        if let Some(f) = filter {
            let mut offset = 0usize;
            let (where_sql, filter_bindings) = filter_to_surrealql(f, &mut offset);
            where_extra = format!(" AND {where_sql}");
            bindings.extend(filter_bindings);
        }

        let query = format!(
            "SELECT *, vector::similarity::cosine({vec_col}, $query_vec) AS __score \
             FROM {table} \
             WHERE {vec_col} <|{limit}|> $query_vec{where_extra} \
             ORDER BY __score DESC",
            vec_col = vector_column,
            table = table_name,
            limit = limit,
            where_extra = where_extra,
        );

        let rows = execute_with_bindings(&self.db, &query, bindings).await?;

        let mut results = Vec::with_capacity(rows.len());
        for row in &rows {
            let score = row.get("__score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

            // Build record, skipping the synthetic __score column.
            let mut record_row = json_row_to_record(row);
            record_row.retain(|(name, _)| name != "__score");

            results.push(ScoredRecord {
                record: record_row,
                score,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_url() {
        assert_eq!(SurrealDatabase::default_url(), "ws://localhost:8000");
    }

    #[test]
    fn test_capabilities() {
        let caps = SurrealDatabase::capabilities();
        assert!(caps.vector_search);
    }

    #[test]
    fn test_field_value_to_json() {
        assert_eq!(
            field_value_to_json(&FieldValue::Utf8(Some("hello".into()))),
            json!("hello")
        );
        assert_eq!(
            field_value_to_json(&FieldValue::Utf8(None)),
            serde_json::Value::Null
        );
        assert_eq!(field_value_to_json(&FieldValue::Int32(Some(42))), json!(42));
        assert_eq!(
            field_value_to_json(&FieldValue::Boolean(Some(true))),
            json!(true)
        );
        assert_eq!(
            field_value_to_json(&FieldValue::Vector(vec![1.0, 2.0, 3.0])),
            json!([1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn test_field_type_to_surrealql() {
        assert_eq!(field_type_to_surrealql(&FieldType::Utf8), "string");
        assert_eq!(field_type_to_surrealql(&FieldType::Int32), "int");
        assert_eq!(field_type_to_surrealql(&FieldType::Int64), "int");
        assert_eq!(field_type_to_surrealql(&FieldType::Float32), "float");
        assert_eq!(field_type_to_surrealql(&FieldType::Boolean), "bool");
        assert_eq!(
            field_type_to_surrealql(&FieldType::Vector(384)),
            "array<float, 384>"
        );
    }

    #[test]
    fn test_filter_to_surrealql_eq() {
        let filter = Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into())));
        let mut offset = 0;
        let (sql, bindings) = filter_to_surrealql(&filter, &mut offset);
        assert_eq!(sql, "name = $p0");
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, "p0");
        assert_eq!(bindings[0].1, json!("Alice"));
    }

    #[test]
    fn test_filter_to_surrealql_and() {
        let filter = Filter::And(vec![
            Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into()))),
            Filter::Gt("age".into(), FieldValue::Int32(Some(21))),
        ]);
        let mut offset = 0;
        let (sql, bindings) = filter_to_surrealql(&filter, &mut offset);
        assert_eq!(sql, "(name = $p0 AND age > $p1)");
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_filter_to_surrealql_or() {
        let filter = Filter::Or(vec![
            Filter::Eq("status".into(), FieldValue::Utf8(Some("active".into()))),
            Filter::Eq("status".into(), FieldValue::Utf8(Some("pending".into()))),
        ]);
        let mut offset = 0;
        let (sql, bindings) = filter_to_surrealql(&filter, &mut offset);
        assert_eq!(sql, "(status = $p0 OR status = $p1)");
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_filter_to_surrealql_null_checks() {
        let mut offset = 0;
        let (sql, bindings) = filter_to_surrealql(&Filter::IsNull("email".into()), &mut offset);
        assert_eq!(sql, "email IS NULL");
        assert!(bindings.is_empty());

        let (sql, bindings) = filter_to_surrealql(&Filter::NotNull("email".into()), &mut offset);
        assert_eq!(sql, "email IS NOT NULL");
        assert!(bindings.is_empty());
    }

    #[test]
    fn test_filter_to_surrealql_in() {
        let filter = Filter::In(
            "id".into(),
            vec![
                FieldValue::Int64(Some(1)),
                FieldValue::Int64(Some(2)),
                FieldValue::Int64(Some(3)),
            ],
        );
        let mut offset = 0;
        let (sql, bindings) = filter_to_surrealql(&filter, &mut offset);
        assert_eq!(sql, "id IN $p0");
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].1, json!([1, 2, 3]));
    }

    #[test]
    fn test_filter_to_surrealql_empty_and_or() {
        let mut offset = 0;
        let (sql, _) = filter_to_surrealql(&Filter::And(vec![]), &mut offset);
        assert_eq!(sql, "true");

        let (sql, _) = filter_to_surrealql(&Filter::Or(vec![]), &mut offset);
        assert_eq!(sql, "false");
    }

    #[test]
    fn test_json_row_to_record() {
        let row = json!({
            "id": "code_embeddings:abc123",
            "name": "test",
            "count": 42,
            "active": true,
            "score": 0.95
        });
        let record = json_row_to_record(&row);
        assert!(!record.is_empty());

        // Check that id is included as a string.
        let id_field = record.iter().find(|(n, _)| n == "id");
        assert!(id_field.is_some());
    }
}
