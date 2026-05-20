//! Schema, record, and filter types for the unified database layer.
//!
//! These types define a backend-agnostic data model that both
//! [`StorageBackend`](super::traits::StorageBackend) and
//! [`VectorDatabase`](super::traits::VectorDatabase) implementations use.

// ── Schema types ────────────────────────────────────────────────────────

/// Definition of a single field within a table schema.
#[derive(Debug, Clone)]
pub struct FieldDef {
    /// Column name.
    pub name: String,
    /// Data type.
    pub field_type: FieldType,
    /// Whether `NULL` / `None` values are permitted.
    pub nullable: bool,
}

impl FieldDef {
    /// Shorthand constructor for a non-nullable field.
    pub fn required(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            nullable: false,
        }
    }

    /// Shorthand constructor for a nullable field.
    pub fn optional(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            nullable: true,
        }
    }
}

/// Supported column data types.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// UTF-8 string.
    Utf8,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit unsigned integer.
    UInt64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit floating point.
    Float64,
    /// Boolean.
    Boolean,
    /// Fixed-size float vector with the given dimension (for embeddings).
    Vector(usize),
}

// ── Record types ────────────────────────────────────────────────────────

/// A single typed column value.
#[derive(Debug, Clone)]
pub enum FieldValue {
    /// UTF-8 string (nullable).
    Utf8(Option<String>),
    /// 32-bit signed integer (nullable).
    Int32(Option<i32>),
    /// 64-bit signed integer (nullable).
    Int64(Option<i64>),
    /// 32-bit unsigned integer (nullable).
    UInt32(Option<u32>),
    /// 64-bit unsigned integer (nullable).
    UInt64(Option<u64>),
    /// 32-bit floating point (nullable).
    Float32(Option<f32>),
    /// 64-bit floating point (nullable).
    Float64(Option<f64>),
    /// Boolean (nullable).
    Boolean(Option<bool>),
    /// Dense float vector (for embeddings). Empty vec means NULL.
    Vector(Vec<f32>),
}

impl FieldValue {
    /// Return the value as a string reference, if it is `Utf8(Some(_))`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FieldValue::Utf8(Some(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the value as an `i64`, if it is `Int64(Some(_))`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FieldValue::Int64(Some(v)) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as an `i32`, if it is `Int32(Some(_))`.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            FieldValue::Int32(Some(v)) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as an `f32`, if it is `Float32(Some(_))`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            FieldValue::Float32(Some(v)) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as an `f64`, if it is `Float64(Some(_))`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::Float64(Some(v)) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as a `bool`, if it is `Boolean(Some(_))`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FieldValue::Boolean(Some(v)) => Some(*v),
            _ => None,
        }
    }

    /// Return the value as a float vector reference.
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            FieldValue::Vector(v) if !v.is_empty() => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// A generic row: ordered list of `(column_name, value)` pairs.
pub type Record = Vec<(String, FieldValue)>;

/// Helper to look up a field in a [`Record`] by name.
pub fn record_get<'a>(record: &'a Record, name: &str) -> Option<&'a FieldValue> {
    record.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// A record returned from a vector similarity search, with a relevance score.
#[derive(Debug, Clone)]
pub struct ScoredRecord {
    /// The matched row.
    pub record: Record,
    /// Similarity score (higher is better, typically 0.0–1.0).
    pub score: f32,
}

// ── Filter types ────────────────────────────────────────────────────────

/// Structured query filter that backends translate into their native syntax.
///
/// Use [`Filter::Raw`] as an escape hatch for backend-specific expressions.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Column equals value.
    Eq(String, FieldValue),
    /// Column does not equal value.
    Ne(String, FieldValue),
    /// Column is less than value.
    Lt(String, FieldValue),
    /// Column is less than or equal to value.
    Lte(String, FieldValue),
    /// Column is greater than value.
    Gt(String, FieldValue),
    /// Column is greater than or equal to value.
    Gte(String, FieldValue),
    /// Column is NOT NULL.
    NotNull(String),
    /// Column IS NULL.
    IsNull(String),
    /// Column value is in the given list.
    In(String, Vec<FieldValue>),
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// At least one sub-filter must match.
    Or(Vec<Filter>),
    /// Raw backend-specific filter string (escape hatch).
    Raw(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- FieldDef ---

    #[test]
    fn field_def_required_is_not_nullable() {
        let f = FieldDef::required("name", FieldType::Utf8);
        assert_eq!(f.name, "name");
        assert!(!f.nullable);
        assert_eq!(f.field_type, FieldType::Utf8);
    }

    #[test]
    fn field_def_optional_is_nullable() {
        let f = FieldDef::optional("score", FieldType::Float32);
        assert_eq!(f.name, "score");
        assert!(f.nullable);
        assert_eq!(f.field_type, FieldType::Float32);
    }

    // --- FieldValue accessors ---

    #[test]
    fn field_value_as_str_returns_some_when_present() {
        let v = FieldValue::Utf8(Some("hello".to_string()));
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn field_value_as_str_returns_none_when_null() {
        assert_eq!(FieldValue::Utf8(None).as_str(), None);
    }

    #[test]
    fn field_value_as_str_returns_none_for_wrong_type() {
        assert_eq!(FieldValue::Int32(Some(1)).as_str(), None);
    }

    #[test]
    fn field_value_as_i64_returns_some_when_present() {
        let v = FieldValue::Int64(Some(42));
        assert_eq!(v.as_i64(), Some(42));
    }

    #[test]
    fn field_value_as_i64_returns_none_when_null() {
        assert_eq!(FieldValue::Int64(None).as_i64(), None);
    }

    #[test]
    fn field_value_as_i32_works() {
        assert_eq!(FieldValue::Int32(Some(-5)).as_i32(), Some(-5));
        assert_eq!(FieldValue::Int32(None).as_i32(), None);
    }

    #[test]
    fn field_value_as_f32_works() {
        let v = FieldValue::Float32(Some(2.5));
        assert!((v.as_f32().unwrap() - 2.5f32).abs() < 1e-5);
        assert_eq!(FieldValue::Float32(None).as_f32(), None);
    }

    #[test]
    fn field_value_as_bool_works() {
        assert_eq!(FieldValue::Boolean(Some(true)).as_bool(), Some(true));
        assert_eq!(FieldValue::Boolean(None).as_bool(), None);
    }

    #[test]
    fn field_value_as_vector_returns_slice_when_non_empty() {
        let v = FieldValue::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.as_vector(), Some([1.0f32, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn field_value_as_vector_returns_none_for_empty() {
        let v = FieldValue::Vector(vec![]);
        assert_eq!(v.as_vector(), None);
    }

    // --- record_get ---

    #[test]
    fn record_get_finds_field_by_name() {
        let record: Record = vec![
            ("id".to_string(), FieldValue::Int64(Some(1))),
            (
                "name".to_string(),
                FieldValue::Utf8(Some("Alice".to_string())),
            ),
        ];
        let v = record_get(&record, "name").unwrap();
        assert_eq!(v.as_str(), Some("Alice"));
    }

    #[test]
    fn record_get_returns_none_for_missing_field() {
        let record: Record = vec![("id".to_string(), FieldValue::Int64(Some(1)))];
        assert!(record_get(&record, "missing").is_none());
    }

    // --- ScoredRecord ---

    #[test]
    fn scored_record_holds_score_and_record() {
        let scored = ScoredRecord {
            record: vec![("id".to_string(), FieldValue::Int64(Some(5)))],
            score: 0.95,
        };
        assert_eq!(scored.score, 0.95);
        assert_eq!(record_get(&scored.record, "id").unwrap().as_i64(), Some(5));
    }
}
