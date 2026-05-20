//! PostgreSQL + pgvector backend for the [`VectorDatabase`] trait.
//!
//! This module provides a PostgreSQL-backed vector database implementation
//! using the [pgvector](https://github.com/pgvector/pgvector) extension for
//! approximate nearest-neighbour search and
//! [tokio-postgres](https://docs.rs/tokio-postgres) with
//! [deadpool-postgres](https://docs.rs/deadpool-postgres) for async connection
//! pooling.
//!
//! # Requirements
//!
//! * A running PostgreSQL server with the `vector` extension installed.
//! * The `postgres-backend` Cargo feature enabled on `brainwires-storage`.
//!
//! # Example
//!
//! ```rust,no_run
//! use brainwires_storage::databases::postgres::PostgresDatabase;
//! use brainwires_storage::databases::traits::VectorDatabase;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let db = PostgresDatabase::new().await?;
//! db.initialize(384).await?;
//! # Ok(())
//! # }
//! ```

use crate::databases::bm25_helpers::{self, SharedIdfStats};
use crate::databases::sql::{self, postgres::PostgresDialect};
use crate::databases::traits::{
    ChunkMetadata, DatabaseStats, SearchResult, StorageBackend, VectorDatabase,
};
use crate::databases::types::{FieldDef, FieldValue, Filter, Record, ScoredRecord};
use crate::glob_utils;
use anyhow::{Context, Result};
use deadpool_postgres::{Config, Pool, Runtime};
use pgvector::Vector;
use tokio_postgres::types::ToSql;

const DEFAULT_TABLE: &str = "code_embeddings";
const DEFAULT_URL: &str = "postgresql://localhost:5432/brainwires";

/// PostgreSQL + pgvector backed vector database for code embeddings.
///
/// Uses HNSW indexing for fast approximate nearest-neighbour search and
/// client-side BM25 scoring for hybrid (vector + keyword) queries.
pub struct PostgresDatabase {
    pool: Pool,
    table_name: String,
    idf_stats: SharedIdfStats,
}

impl PostgresDatabase {
    /// Create a new client connected to the default local PostgreSQL instance.
    ///
    /// Connects to [`DEFAULT_URL`] (`postgresql://localhost:5432/brainwires`)
    /// and uses the default table name `code_embeddings`.
    pub async fn new() -> Result<Self> {
        Self::with_url(DEFAULT_URL).await
    }

    /// Create a new client with a custom connection string.
    pub async fn with_url(url: &str) -> Result<Self> {
        tracing::info!("Connecting to PostgreSQL at {}", url);

        let mut cfg = Config::new();
        cfg.url = Some(url.to_string());
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), tokio_postgres::NoTls)
            .context("Failed to create PostgreSQL connection pool")?;

        // Verify connectivity by grabbing a connection.
        let _conn = pool
            .get()
            .await
            .context("Failed to connect to PostgreSQL")?;

        Self::with_pool(pool, DEFAULT_TABLE).await
    }

    /// Create a new client from an existing connection pool.
    ///
    /// This is useful when the caller already manages a pool or wants to
    /// share it across subsystems.
    pub async fn with_pool(pool: Pool, table_name: &str) -> Result<Self> {
        let db = Self {
            pool,
            table_name: table_name.to_string(),
            idf_stats: bm25_helpers::new_shared_idf_stats(),
        };

        // Seed IDF stats from any existing rows.
        if let Err(e) = db.refresh_idf_stats().await {
            tracing::warn!("Failed to initialize IDF stats: {}", e);
        }

        Ok(db)
    }

    /// Return the default connection URL.
    pub fn default_url() -> String {
        DEFAULT_URL.to_string()
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Refresh IDF statistics by scanning all stored content.
    async fn refresh_idf_stats(&self) -> Result<()> {
        tracing::debug!("Refreshing IDF statistics from table '{}'", self.table_name);

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let query = format!("SELECT content FROM {}", self.table_name);
        let rows = match client.query(&*query, &[]).await {
            Ok(rows) => rows,
            Err(e) => {
                // Table may not exist yet — that is fine.
                tracing::debug!("IDF refresh skipped (table may not exist): {}", e);
                return Ok(());
            }
        };

        let documents: Vec<String> = rows
            .iter()
            .filter_map(|row| row.try_get::<_, String>("content").ok())
            .collect();

        tracing::info!("Refreshing IDF stats from {} documents", documents.len());
        bm25_helpers::update_idf_stats(&self.idf_stats, &documents).await;

        Ok(())
    }

    /// Execute the core filtered search logic shared by `search` and
    /// `search_filtered`.
    // reason: this is the union of two public API surfaces (`search` +
    // `search_filtered`). Bundling args into a struct would just shuffle them
    // to the call sites without simplifying anything.
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
            self.table_name,
            limit,
            min_score,
            project,
            root_path,
            hybrid,
            file_extensions,
            languages,
            path_patterns,
        );

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let pg_vector = Vector::from(query_vector);

        let query = format!(
            r#"
            SELECT
                file_path,
                root_path,
                project,
                start_line,
                end_line,
                language,
                extension,
                indexed_at,
                content,
                1.0 - (embedding <=> $1::vector) AS vector_score
            FROM {table}
            WHERE 1=1
              AND ($2::text IS NULL OR project = $2)
              AND ($3::text IS NULL OR root_path = $3)
              AND (cardinality($4::text[]) = 0 OR extension = ANY($4))
              AND (cardinality($5::text[]) = 0 OR language = ANY($5))
            ORDER BY embedding <=> $1::vector
            LIMIT $6
            "#,
            table = self.table_name,
        );

        let limit_i64 = limit as i64;

        let rows = client
            .query(
                &*query,
                &[
                    &pg_vector,
                    &project.as_deref(),
                    &root_path.as_deref(),
                    &file_extensions,
                    &languages,
                    &limit_i64,
                ],
            )
            .await
            .context("Failed to execute search query")?;

        let mut results: Vec<SearchResult> = Vec::with_capacity(rows.len());

        for row in &rows {
            let vector_score: f64 = row.try_get("vector_score").unwrap_or(0.0);
            let vector_score = vector_score as f32;

            // Skip results below the minimum score threshold.
            if vector_score < min_score {
                continue;
            }

            let file_path: String = row
                .try_get("file_path")
                .context("Missing file_path column")?;
            let result_root_path: Option<String> = row.try_get("root_path").ok();
            let result_project: Option<String> = row.try_get("project").ok();
            let start_line: i32 = row.try_get("start_line").unwrap_or(0);
            let end_line: i32 = row.try_get("end_line").unwrap_or(0);
            let language: String = row
                .try_get("language")
                .unwrap_or_else(|_| "Unknown".to_string());
            let indexed_at: i64 = row.try_get("indexed_at").unwrap_or(0);
            let content: String = row.try_get("content").unwrap_or_default();

            // Calculate keyword score if hybrid search is enabled.
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
                start_line: start_line as usize,
                end_line: end_line as usize,
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
impl VectorDatabase for PostgresDatabase {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        tracing::info!(
            "Initializing PostgreSQL table '{}' with vector dimension {}",
            self.table_name,
            dimension
        );

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        // Enable the pgvector extension.
        client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .context("Failed to create vector extension")?;

        // Create the embeddings table.
        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {table} (
                id          BIGSERIAL PRIMARY KEY,
                embedding   vector({dim}),
                file_path   TEXT    NOT NULL,
                root_path   TEXT,
                project     TEXT,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                language    TEXT,
                extension   TEXT,
                file_hash   TEXT    NOT NULL,
                indexed_at  BIGINT  NOT NULL,
                content     TEXT    NOT NULL
            )
            "#,
            table = self.table_name,
            dim = dimension,
        );
        client
            .execute(&*create_table, &[])
            .await
            .context("Failed to create embeddings table")?;

        // Create B-tree indexes for common filter columns.
        let idx_file_path = format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_file_path ON {table} (file_path)",
            table = self.table_name,
        );
        client
            .execute(&*idx_file_path, &[])
            .await
            .context("Failed to create file_path index")?;

        let idx_root_path = format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_root_path ON {table} (root_path)",
            table = self.table_name,
        );
        client
            .execute(&*idx_root_path, &[])
            .await
            .context("Failed to create root_path index")?;

        let idx_project = format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_project ON {table} (project)",
            table = self.table_name,
        );
        client
            .execute(&*idx_project, &[])
            .await
            .context("Failed to create project index")?;

        // HNSW index works on empty tables (unlike IVFFlat which requires data).
        let idx_embedding = format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_embedding ON {table} \
             USING hnsw (embedding vector_cosine_ops)",
            table = self.table_name,
        );
        client
            .execute(&*idx_embedding, &[])
            .await
            .context("Failed to create HNSW embedding index")?;

        tracing::info!("PostgreSQL table '{}' initialized", self.table_name);
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
        tracing::debug!("Storing {} embeddings in '{}'", count, self.table_name);

        let mut client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let insert_sql = format!(
            r#"
            INSERT INTO {table}
                (embedding, file_path, root_path, project,
                 start_line, end_line, language, extension,
                 file_hash, indexed_at, content)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
            table = self.table_name,
        );

        let tx = client
            .transaction()
            .await
            .context("Failed to begin transaction")?;

        for ((embedding, meta), content) in embeddings.into_iter().zip(metadata).zip(contents) {
            let pg_vector = Vector::from(embedding);
            let start_line = meta.start_line as i32;
            let end_line = meta.end_line as i32;

            tx.execute(
                &*insert_sql,
                &[
                    &pg_vector,
                    &meta.file_path,
                    &meta.root_path.as_deref(),
                    &meta.project.as_deref(),
                    &start_line,
                    &end_line,
                    &meta.language.as_deref(),
                    &meta.extension.as_deref(),
                    &meta.file_hash,
                    &meta.indexed_at,
                    &content,
                ],
            )
            .await
            .context("Failed to insert embedding row")?;
        }

        tx.commit().await.context("Failed to commit transaction")?;

        tracing::info!("Stored {} embeddings in '{}'", count, self.table_name);

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

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let query = format!("DELETE FROM {} WHERE file_path = $1", self.table_name);

        let deleted = client
            .execute(&*query, &[&file_path])
            .await
            .context("Failed to delete embeddings by file path")?;

        tracing::info!("Deleted {} rows for file '{}'", deleted, file_path);

        Ok(deleted as usize)
    }

    async fn clear(&self) -> Result<()> {
        tracing::info!("Clearing all embeddings from table '{}'", self.table_name);

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let query = format!("TRUNCATE {}", self.table_name);
        client
            .execute(&*query, &[])
            .await
            .context("Failed to truncate embeddings table")?;

        // Clear IDF stats.
        let mut stats = self.idf_stats.write().await;
        stats.total_docs = 0;
        stats.doc_frequencies.clear();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        tracing::debug!("Fetching statistics for table '{}'", self.table_name);

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        // Total row count.
        let count_query = format!("SELECT COUNT(*) AS total FROM {}", self.table_name);
        let row = client
            .query_one(&*count_query, &[])
            .await
            .context("Failed to count rows")?;
        let total: i64 = row.try_get("total").unwrap_or(0);

        // Per-language breakdown.
        let lang_query = format!(
            "SELECT language, COUNT(*) AS lang_count FROM {} GROUP BY language",
            self.table_name,
        );
        let lang_rows = client
            .query(&*lang_query, &[])
            .await
            .context("Failed to fetch language breakdown")?;

        let language_breakdown: Vec<(String, usize)> = lang_rows
            .iter()
            .map(|row| {
                let lang: String = row
                    .try_get("language")
                    .unwrap_or_else(|_| "Unknown".to_string());
                let cnt: i64 = row.try_get("lang_count").unwrap_or(0);
                (lang, cnt as usize)
            })
            .collect();

        Ok(DatabaseStats {
            total_points: total as usize,
            total_vectors: total as usize,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // PostgreSQL persists transactionally — no explicit flush needed.
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let query = format!(
            "SELECT COUNT(*) AS cnt FROM {} WHERE root_path = $1",
            self.table_name,
        );

        let row = client
            .query_one(&*query, &[&root_path])
            .await
            .context("Failed to count rows by root_path")?;
        let count: i64 = row.try_get("cnt").unwrap_or(0);

        Ok(count as usize)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let query = format!(
            "SELECT DISTINCT file_path FROM {} WHERE root_path = $1",
            self.table_name,
        );

        let rows = client
            .query(&*query, &[&root_path])
            .await
            .context("Failed to fetch indexed files")?;

        let files: Vec<String> = rows
            .iter()
            .filter_map(|row| row.try_get("file_path").ok())
            .collect();

        Ok(files)
    }
}

// ── StorageBackend trait implementation ───────────────────────────────

/// Convert a [`FieldValue`] slice into boxed `ToSql` parameters for `tokio_postgres`.
fn field_values_to_params(values: &[FieldValue]) -> Vec<Box<dyn ToSql + Sync + Send>> {
    values
        .iter()
        .map(|v| -> Box<dyn ToSql + Sync + Send> {
            match v {
                FieldValue::Utf8(opt) => Box::new(opt.clone()),
                FieldValue::Int32(opt) => Box::new(*opt),
                FieldValue::Int64(opt) => Box::new(*opt),
                FieldValue::UInt32(opt) => Box::new(opt.map(|u| u as i32)),
                FieldValue::UInt64(opt) => Box::new(opt.map(|u| u as i64)),
                FieldValue::Float32(opt) => Box::new(*opt),
                FieldValue::Float64(opt) => Box::new(*opt),
                FieldValue::Boolean(opt) => Box::new(*opt),
                FieldValue::Vector(v) => Box::new(Vector::from(v.clone())),
            }
        })
        .collect()
}

/// Build `&[&dyn ToSql]` references from the boxed parameter list.
fn params_as_refs(params: &[Box<dyn ToSql + Sync + Send>]) -> Vec<&(dyn ToSql + Sync)> {
    params
        .iter()
        .map(|p| -> &(dyn ToSql + Sync) { p.as_ref() })
        .collect()
}

/// Parse a single `tokio_postgres::Row` into a [`Record`] using column type metadata.
fn row_to_record(row: &tokio_postgres::Row) -> Record {
    use tokio_postgres::types::Type;

    let mut record = Vec::with_capacity(row.columns().len());
    for (i, col) in row.columns().iter().enumerate() {
        let name = col.name().to_string();
        let val = match *col.type_() {
            Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => {
                FieldValue::Utf8(row.try_get::<_, String>(i).ok())
            }
            Type::INT4 => FieldValue::Int32(row.try_get::<_, i32>(i).ok()),
            Type::INT8 => FieldValue::Int64(row.try_get::<_, i64>(i).ok()),
            Type::INT2 => FieldValue::Int32(row.try_get::<_, i16>(i).ok().map(|v| v as i32)),
            Type::FLOAT4 => FieldValue::Float32(row.try_get::<_, f32>(i).ok()),
            Type::FLOAT8 => FieldValue::Float64(row.try_get::<_, f64>(i).ok()),
            Type::BOOL => FieldValue::Boolean(row.try_get::<_, bool>(i).ok()),
            _ => {
                // For pgvector columns and any other unknown type, try to
                // read as a pgvector Vector first, then fall back to string.
                if let Ok(v) = row.try_get::<_, Vector>(i) {
                    FieldValue::Vector(v.to_vec())
                } else {
                    FieldValue::Utf8(row.try_get::<_, String>(i).ok())
                }
            }
        };
        record.push((name, val));
    }
    record
}

#[async_trait::async_trait]
impl StorageBackend for PostgresDatabase {
    async fn ensure_table(&self, table_name: &str, schema: &[FieldDef]) -> Result<()> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        // Enable pgvector extension if schema contains a vector column.
        let has_vector = schema
            .iter()
            .any(|f| matches!(f.field_type, crate::databases::types::FieldType::Vector(_)));
        if has_vector {
            client
                .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
                .await
                .context("Failed to create vector extension")?;
        }

        let ddl = sql::build_create_table(table_name, schema, &PostgresDialect);
        client
            .execute(&*ddl, &[])
            .await
            .with_context(|| format!("Failed to create table '{table_name}'"))?;

        Ok(())
    }

    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        // Extract column names from the first record.
        let col_names: Vec<&str> = records[0].iter().map(|(name, _)| name.as_str()).collect();

        // Build rows of FieldValues aligned with col_names.
        let rows: Vec<Vec<FieldValue>> = records
            .iter()
            .map(|rec| rec.iter().map(|(_, v)| v.clone()).collect())
            .collect();

        let (sql, values) = sql::build_insert(table_name, &col_names, &rows, &PostgresDialect);
        let boxed = field_values_to_params(&values);
        let refs = params_as_refs(&boxed);

        client
            .execute(&*sql, &refs)
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
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let (sql, values) = sql::build_select(table_name, filter, limit, &PostgresDialect);
        let boxed = field_values_to_params(&values);
        let refs = params_as_refs(&boxed);

        let rows = client
            .query(&*sql, &refs)
            .await
            .with_context(|| format!("Failed to query '{table_name}'"))?;

        Ok(rows.iter().map(row_to_record).collect())
    }

    async fn delete(&self, table_name: &str, filter: &Filter) -> Result<()> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let (sql, values) = sql::build_delete(table_name, filter, &PostgresDialect);
        let boxed = field_values_to_params(&values);
        let refs = params_as_refs(&boxed);

        client
            .execute(&*sql, &refs)
            .await
            .with_context(|| format!("Failed to delete from '{table_name}'"))?;

        Ok(())
    }

    async fn count(&self, table_name: &str, filter: Option<&Filter>) -> Result<usize> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let (sql, values) = sql::build_count(table_name, filter, &PostgresDialect);
        let boxed = field_values_to_params(&values);
        let refs = params_as_refs(&boxed);

        let row = client
            .query_one(&*sql, &refs)
            .await
            .with_context(|| format!("Failed to count rows in '{table_name}'"))?;

        let count: i64 = row.try_get(0).unwrap_or(0);
        Ok(count as usize)
    }

    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredRecord>> {
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get connection from pool")?;

        let pg_vector = Vector::from(vector);
        let limit_i64 = limit as i64;

        // Build the query with optional filter.
        // Parameter layout: $1 = vector, then filter params, then limit.
        let (where_clause, filter_values) = if let Some(f) = filter {
            let (sql, vals) = sql::filter_to_sql(f, &PostgresDialect, 2);
            (format!("WHERE {}", sql), vals)
        } else {
            (String::new(), vec![])
        };

        let limit_param_idx = 2 + filter_values.len();
        let quoted_col = format!("\"{}\"", vector_column);

        let sql = format!(
            "SELECT *, 1.0 - ({col} <=> $1::vector) AS __score \
             FROM \"{table}\" {where} \
             ORDER BY {col} <=> $1::vector \
             LIMIT ${limit_idx}",
            col = quoted_col,
            table = table_name,
            where = where_clause,
            limit_idx = limit_param_idx,
        );

        // Build params: vector, filter values, limit.
        let mut all_values: Vec<Box<dyn ToSql + Sync + Send>> = Vec::new();
        all_values.push(Box::new(pg_vector));
        all_values.extend(field_values_to_params(&filter_values));
        all_values.push(Box::new(limit_i64));

        let refs = params_as_refs(&all_values);

        let rows = client
            .query(&*sql, &refs)
            .await
            .with_context(|| format!("Failed vector search on '{table_name}'.'{vector_column}'"))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in &rows {
            let score: f64 = row.try_get("__score").unwrap_or(0.0);

            // Build the record, skipping the synthetic __score column.
            let mut record = Vec::new();
            for (i, col) in row.columns().iter().enumerate() {
                if col.name() == "__score" {
                    continue;
                }
                let name = col.name().to_string();
                let val = {
                    use tokio_postgres::types::Type;
                    match *col.type_() {
                        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => {
                            FieldValue::Utf8(row.try_get::<_, String>(i).ok())
                        }
                        Type::INT4 => FieldValue::Int32(row.try_get::<_, i32>(i).ok()),
                        Type::INT8 => FieldValue::Int64(row.try_get::<_, i64>(i).ok()),
                        Type::INT2 => {
                            FieldValue::Int32(row.try_get::<_, i16>(i).ok().map(|v| v as i32))
                        }
                        Type::FLOAT4 => FieldValue::Float32(row.try_get::<_, f32>(i).ok()),
                        Type::FLOAT8 => FieldValue::Float64(row.try_get::<_, f64>(i).ok()),
                        Type::BOOL => FieldValue::Boolean(row.try_get::<_, bool>(i).ok()),
                        _ => {
                            if let Ok(v) = row.try_get::<_, Vector>(i) {
                                FieldValue::Vector(v.to_vec())
                            } else {
                                FieldValue::Utf8(row.try_get::<_, String>(i).ok())
                            }
                        }
                    }
                };
                record.push((name, val));
            }

            results.push(ScoredRecord {
                record,
                score: score as f32,
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
        assert_eq!(
            PostgresDatabase::default_url(),
            "postgresql://localhost:5432/brainwires"
        );
    }
}
