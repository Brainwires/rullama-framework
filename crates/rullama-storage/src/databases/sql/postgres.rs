//! PostgreSQL dialect for the shared SQL generation layer.
//!
//! Uses `$N` positional placeholders, double-quote identifier quoting,
//! and pgvector's `vector(N)` type for embedding columns.

use crate::databases::types::FieldType;

use super::SqlDialect;

/// PostgreSQL + pgvector SQL dialect.
pub struct PostgresDialect;

impl SqlDialect for PostgresDialect {
    fn map_type(&self, field_type: &FieldType) -> String {
        match field_type {
            FieldType::Utf8 => "TEXT".to_string(),
            FieldType::Int32 => "INTEGER".to_string(),
            FieldType::Int64 => "BIGINT".to_string(),
            FieldType::UInt32 => "INTEGER".to_string(),
            FieldType::UInt64 => "BIGINT".to_string(),
            FieldType::Float32 => "REAL".to_string(),
            FieldType::Float64 => "DOUBLE PRECISION".to_string(),
            FieldType::Boolean => "BOOLEAN".to_string(),
            FieldType::Vector(n) => format!("vector({})", n),
        }
    }

    fn placeholder(&self, n: usize) -> String {
        format!("${}", n)
    }

    fn quote_ident(&self, ident: &str) -> String {
        format!("\"{}\"", ident)
    }
}
