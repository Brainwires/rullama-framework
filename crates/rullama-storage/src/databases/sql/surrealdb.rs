//! SurrealDB dialect for the shared SQL generation layer.
//!
//! SurrealQL uses `$N` positional placeholders and double-quote identifier
//! quoting, similar to PostgreSQL.  SurrealDB supports a native `array<float>`
//! type for vector columns and `float` / `int` / `bool` / `string` for
//! scalar types.

use crate::databases::types::FieldType;

use super::SqlDialect;

/// SurrealDB (SurrealQL) dialect.
pub struct SurrealDialect;

impl SqlDialect for SurrealDialect {
    fn map_type(&self, field_type: &FieldType) -> String {
        match field_type {
            FieldType::Utf8 => "string".to_string(),
            FieldType::Int32 => "int".to_string(),
            FieldType::Int64 => "int".to_string(),
            FieldType::UInt32 => "int".to_string(),
            FieldType::UInt64 => "int".to_string(),
            FieldType::Float32 => "float".to_string(),
            FieldType::Float64 => "float".to_string(),
            FieldType::Boolean => "bool".to_string(),
            // SurrealDB supports native vector storage via array<float, N>.
            FieldType::Vector(n) => format!("array<float, {}>", n),
        }
    }

    fn placeholder(&self, n: usize) -> String {
        format!("${}", n)
    }

    fn quote_ident(&self, ident: &str) -> String {
        format!("\"{}\"", ident)
    }
}
