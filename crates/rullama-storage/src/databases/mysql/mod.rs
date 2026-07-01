//! MySQL / MariaDB backend for the [`StorageBackend`] trait.
//!
//! Implements generic CRUD operations using [`mysql_async`] with connection
//! pooling.  Vector search is performed client-side via cosine similarity
//! since MySQL has no native vector type.
//!
//! # Requirements
//!
//! * A running MySQL or MariaDB server.
//! * The `mysql-backend` Cargo feature enabled on `rullama-storage`.

use crate::databases::capabilities::BackendCapabilities;
use crate::databases::sql::{self, mysql::MySqlDialect};
use crate::databases::traits::StorageBackend;
use crate::databases::types::{FieldValue, Record, ScoredRecord};
use anyhow::{Context, Result};
use mysql_async::prelude::*;

const DEFAULT_URL: &str = "mysql://localhost:3306/rullama";

/// MySQL / MariaDB backed storage database.
///
/// Uses [`mysql_async`] connection pooling for async operations.  Vector
/// columns are stored as JSON arrays and similarity search is performed
/// client-side via cosine similarity.
pub struct MySqlDatabase {
    pool: mysql_async::Pool,
}

impl MySqlDatabase {
    /// Create a new client connected to the given MySQL URL.
    ///
    /// Verifies connectivity by executing a ping before returning.
    pub async fn new(url: &str) -> Result<Self> {
        tracing::info!("Connecting to MySQL at {}", url);

        let pool = mysql_async::Pool::new(url);

        // Verify connectivity.
        let conn = pool
            .get_conn()
            .await
            .context("Failed to connect to MySQL")?;
        conn.disconnect()
            .await
            .context("Failed to disconnect MySQL verification connection")?;

        Ok(Self { pool })
    }

    /// Return the default connection URL.
    pub fn default_url() -> String {
        DEFAULT_URL.to_string()
    }

    /// Return the capability set for this backend.
    pub fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            vector_search: false,
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert a [`FieldValue`] to a [`mysql_async::Value`] suitable for binding.
fn field_value_to_mysql(val: &FieldValue) -> mysql_async::Value {
    match val {
        FieldValue::Utf8(Some(s)) => mysql_async::Value::from(s.as_str()),
        FieldValue::Utf8(None) => mysql_async::Value::NULL,
        FieldValue::Int32(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::Int32(None) => mysql_async::Value::NULL,
        FieldValue::Int64(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::Int64(None) => mysql_async::Value::NULL,
        FieldValue::UInt32(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::UInt32(None) => mysql_async::Value::NULL,
        FieldValue::UInt64(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::UInt64(None) => mysql_async::Value::NULL,
        FieldValue::Float32(Some(v)) => mysql_async::Value::from(*v as f64),
        FieldValue::Float32(None) => mysql_async::Value::NULL,
        FieldValue::Float64(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::Float64(None) => mysql_async::Value::NULL,
        FieldValue::Boolean(Some(v)) => mysql_async::Value::from(*v),
        FieldValue::Boolean(None) => mysql_async::Value::NULL,
        FieldValue::Vector(v) => {
            // Store as JSON array string.
            let json = serde_json::to_string(v).unwrap_or_default();
            mysql_async::Value::from(json)
        }
    }
}

/// Convert bind parameters from [`FieldValue`] into [`mysql_async::Params`].
fn to_params(values: &[FieldValue]) -> mysql_async::Params {
    if values.is_empty() {
        mysql_async::Params::Empty
    } else {
        mysql_async::Params::Positional(values.iter().map(field_value_to_mysql).collect())
    }
}

/// Parse a single [`mysql_async::Row`] into a [`Record`] using column metadata.
fn row_to_record(row: &mysql_async::Row) -> Record {
    let columns = row.columns_ref();
    let mut record = Vec::with_capacity(columns.len());

    for (i, col) in columns.iter().enumerate() {
        let name = col.name_str().to_string();
        let value = match col.column_type() {
            // String-like types
            mysql_async::consts::ColumnType::MYSQL_TYPE_VARCHAR
            | mysql_async::consts::ColumnType::MYSQL_TYPE_VAR_STRING
            | mysql_async::consts::ColumnType::MYSQL_TYPE_STRING
            | mysql_async::consts::ColumnType::MYSQL_TYPE_BLOB
            | mysql_async::consts::ColumnType::MYSQL_TYPE_TINY_BLOB
            | mysql_async::consts::ColumnType::MYSQL_TYPE_MEDIUM_BLOB
            | mysql_async::consts::ColumnType::MYSQL_TYPE_LONG_BLOB => {
                let v: Option<String> = row.get(i);
                FieldValue::Utf8(v)
            }
            // JSON — could be a vector or a plain string
            mysql_async::consts::ColumnType::MYSQL_TYPE_JSON => {
                let v: Option<String> = row.get(i);
                // Try to parse as a float vector first.
                if let Some(ref s) = v {
                    if let Ok(vec) = serde_json::from_str::<Vec<f32>>(s) {
                        FieldValue::Vector(vec)
                    } else {
                        FieldValue::Utf8(v)
                    }
                } else {
                    FieldValue::Vector(vec![])
                }
            }
            // Integer types
            mysql_async::consts::ColumnType::MYSQL_TYPE_TINY
            | mysql_async::consts::ColumnType::MYSQL_TYPE_SHORT
            | mysql_async::consts::ColumnType::MYSQL_TYPE_INT24
            | mysql_async::consts::ColumnType::MYSQL_TYPE_LONG => {
                if col
                    .flags()
                    .contains(mysql_async::consts::ColumnFlags::UNSIGNED_FLAG)
                {
                    let v: Option<u32> = row.get(i);
                    FieldValue::UInt32(v)
                } else {
                    let v: Option<i32> = row.get(i);
                    FieldValue::Int32(v)
                }
            }
            mysql_async::consts::ColumnType::MYSQL_TYPE_LONGLONG => {
                if col
                    .flags()
                    .contains(mysql_async::consts::ColumnFlags::UNSIGNED_FLAG)
                {
                    let v: Option<u64> = row.get(i);
                    FieldValue::UInt64(v)
                } else {
                    let v: Option<i64> = row.get(i);
                    FieldValue::Int64(v)
                }
            }
            // Float types
            mysql_async::consts::ColumnType::MYSQL_TYPE_FLOAT => {
                let v: Option<f32> = row.get(i);
                FieldValue::Float32(v)
            }
            mysql_async::consts::ColumnType::MYSQL_TYPE_DOUBLE
            | mysql_async::consts::ColumnType::MYSQL_TYPE_DECIMAL
            | mysql_async::consts::ColumnType::MYSQL_TYPE_NEWDECIMAL => {
                let v: Option<f64> = row.get(i);
                FieldValue::Float64(v)
            }
            // Boolean (MySQL BOOLEAN is TINYINT(1), handled above as TINY,
            // but BIT type can also represent booleans)
            mysql_async::consts::ColumnType::MYSQL_TYPE_BIT => {
                let v: Option<bool> = row.get(i);
                FieldValue::Boolean(v)
            }
            // Fallback: try to read as string.
            _ => {
                let v: Option<String> = row.get(i);
                FieldValue::Utf8(v)
            }
        };

        record.push((name, value));
    }

    record
}

/// Compute cosine similarity between two float slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── StorageBackend implementation ────────────────────────────────────────

#[async_trait::async_trait]
impl StorageBackend for MySqlDatabase {
    async fn ensure_table(
        &self,
        table_name: &str,
        schema: &[crate::databases::types::FieldDef],
    ) -> Result<()> {
        let ddl = sql::build_create_table(table_name, schema, &MySqlDialect);
        tracing::debug!("ensure_table: {}", ddl);

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        conn.exec_drop(&ddl, mysql_async::Params::Empty)
            .await
            .with_context(|| format!("Failed to create table `{}`", table_name))?;

        Ok(())
    }

    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Extract column names from the first record.
        let column_names: Vec<&str> = records[0].iter().map(|(n, _)| n.as_str()).collect();

        // Extract values in row-major order.
        let value_rows: Vec<Vec<FieldValue>> = records
            .iter()
            .map(|rec| rec.iter().map(|(_, v)| v.clone()).collect())
            .collect();

        let (sql, bind_values) =
            sql::build_insert(table_name, &column_names, &value_rows, &MySqlDialect);

        tracing::debug!("insert: {} ({} params)", sql, bind_values.len());

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        conn.exec_drop(&sql, to_params(&bind_values))
            .await
            .with_context(|| format!("Failed to insert into `{}`", table_name))?;

        Ok(())
    }

    async fn query(
        &self,
        table_name: &str,
        filter: Option<&crate::databases::types::Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<Record>> {
        let (sql, bind_values) = sql::build_select(table_name, filter, limit, &MySqlDialect);

        tracing::debug!("query: {} ({} params)", sql, bind_values.len());

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        let rows: Vec<mysql_async::Row> = conn
            .exec(&sql, to_params(&bind_values))
            .await
            .with_context(|| format!("Failed to query `{}`", table_name))?;

        Ok(rows.iter().map(row_to_record).collect())
    }

    async fn delete(
        &self,
        table_name: &str,
        filter: &crate::databases::types::Filter,
    ) -> Result<()> {
        let (sql, bind_values) = sql::build_delete(table_name, filter, &MySqlDialect);

        tracing::debug!("delete: {} ({} params)", sql, bind_values.len());

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        conn.exec_drop(&sql, to_params(&bind_values))
            .await
            .with_context(|| format!("Failed to delete from `{}`", table_name))?;

        Ok(())
    }

    async fn count(
        &self,
        table_name: &str,
        filter: Option<&crate::databases::types::Filter>,
    ) -> Result<usize> {
        let (sql, bind_values) = sql::build_count(table_name, filter, &MySqlDialect);

        tracing::debug!("count: {} ({} params)", sql, bind_values.len());

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        let result: Option<u64> = conn
            .exec_first(&sql, to_params(&bind_values))
            .await
            .with_context(|| format!("Failed to count rows in `{}`", table_name))?;

        Ok(result.unwrap_or(0) as usize)
    }

    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&crate::databases::types::Filter>,
    ) -> Result<Vec<ScoredRecord>> {
        // MySQL has no native vector search. Fetch all matching rows and
        // compute cosine similarity client-side.
        let (mut sql, bind_values) = if let Some(f) = filter {
            let (where_sql, vals) = sql::filter_to_sql(f, &MySqlDialect, 1);
            (
                format!(
                    "SELECT * FROM {} WHERE {}",
                    MySqlDialect.quote_ident(table_name),
                    where_sql
                ),
                vals,
            )
        } else {
            (
                format!("SELECT * FROM {}", MySqlDialect.quote_ident(table_name)),
                vec![],
            )
        };

        // Only fetch rows that have a non-null vector column.
        if bind_values.is_empty() {
            sql.push_str(&format!(
                " WHERE {} IS NOT NULL",
                MySqlDialect.quote_ident(vector_column)
            ));
        } else {
            sql.push_str(&format!(
                " AND {} IS NOT NULL",
                MySqlDialect.quote_ident(vector_column)
            ));
        }

        tracing::debug!("vector_search: {} ({} params)", sql, bind_values.len());

        let mut conn = self
            .pool
            .get_conn()
            .await
            .context("Failed to get MySQL connection")?;

        let rows: Vec<mysql_async::Row> = conn
            .exec(&sql, to_params(&bind_values))
            .await
            .with_context(|| format!("Failed to query `{}` for vector search", table_name))?;

        let mut scored: Vec<ScoredRecord> = Vec::with_capacity(rows.len());

        for row in &rows {
            let record = row_to_record(row);

            // Extract the vector from the record.
            let row_vector: Option<Vec<f32>> = record
                .iter()
                .find(|(name, _)| name == vector_column)
                .and_then(|(_, val)| match val {
                    FieldValue::Vector(v) if !v.is_empty() => Some(v.clone()),
                    _ => None,
                });

            if let Some(row_vec) = row_vector {
                let score = cosine_similarity(&vector, &row_vec);
                scored.push(ScoredRecord { record, score });
            }
        }

        // Sort descending by score and truncate to limit.
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(limit);

        Ok(scored)
    }
}

use sql::SqlDialect;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_url() {
        assert_eq!(
            MySqlDatabase::default_url(),
            "mysql://localhost:3306/rullama"
        );
    }

    #[test]
    fn test_capabilities() {
        let caps = MySqlDatabase::capabilities();
        assert!(!caps.vector_search);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_len() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_field_value_to_mysql_utf8() {
        let val = FieldValue::Utf8(Some("hello".into()));
        let mysql_val = field_value_to_mysql(&val);
        assert_ne!(mysql_val, mysql_async::Value::NULL);
    }

    #[test]
    fn test_field_value_to_mysql_null() {
        let val = FieldValue::Utf8(None);
        let mysql_val = field_value_to_mysql(&val);
        assert_eq!(mysql_val, mysql_async::Value::NULL);
    }

    #[test]
    fn test_field_value_to_mysql_vector() {
        let val = FieldValue::Vector(vec![1.0, 2.0, 3.0]);
        let mysql_val = field_value_to_mysql(&val);
        // Should be a JSON string representation.
        match mysql_val {
            mysql_async::Value::Bytes(b) => {
                let s = String::from_utf8(b).unwrap();
                assert_eq!(s, "[1.0,2.0,3.0]");
            }
            _ => panic!("Expected Bytes variant for vector"),
        }
    }

    #[test]
    fn test_to_params_empty() {
        let params = to_params(&[]);
        assert_eq!(params, mysql_async::Params::Empty);
    }
}
