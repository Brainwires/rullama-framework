//! MySQL / MariaDB dialect for the shared SQL generation layer.
//!
//! Uses positional `?` placeholders, backtick identifier quoting,
//! and `JSON` for vector columns (MySQL has no native vector type).

use crate::databases::types::FieldType;

use super::SqlDialect;

/// MySQL / MariaDB SQL dialect.
pub struct MySqlDialect;

impl SqlDialect for MySqlDialect {
    fn map_type(&self, field_type: &FieldType) -> String {
        match field_type {
            FieldType::Utf8 => "TEXT".to_string(),
            FieldType::Int32 => "INT".to_string(),
            FieldType::Int64 => "BIGINT".to_string(),
            FieldType::UInt32 => "INT UNSIGNED".to_string(),
            FieldType::UInt64 => "BIGINT UNSIGNED".to_string(),
            FieldType::Float32 => "FLOAT".to_string(),
            FieldType::Float64 => "DOUBLE".to_string(),
            FieldType::Boolean => "BOOLEAN".to_string(),
            // MySQL has no native vector type; store as JSON array.
            FieldType::Vector(_) => "JSON".to_string(),
        }
    }

    fn placeholder(&self, _n: usize) -> String {
        "?".to_string()
    }

    fn quote_ident(&self, ident: &str) -> String {
        format!("`{}`", ident)
    }
}
