//! Arrow ↔ Record conversion helpers for LanceDB.
//!
//! These functions translate between the generic [`Record`](super::super::types::Record)
//! / [`FieldDef`](super::super::types::FieldDef) types and Apache Arrow's
//! `RecordBatch` / `Schema` types that LanceDB operates on.

use anyhow::Result;
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::databases::types::{FieldDef, FieldType, FieldValue, Filter, Record};

/// Convert [`FieldDef`] slice to an Arrow [`Schema`].
pub fn field_defs_to_schema(defs: &[FieldDef]) -> Schema {
    let fields: Vec<Field> = defs
        .iter()
        .map(|d| {
            let dt = match &d.field_type {
                FieldType::Utf8 => DataType::Utf8,
                FieldType::Int32 => DataType::Int32,
                FieldType::Int64 => DataType::Int64,
                FieldType::UInt32 => DataType::UInt32,
                FieldType::UInt64 => DataType::UInt64,
                FieldType::Float32 => DataType::Float32,
                FieldType::Float64 => DataType::Float64,
                FieldType::Boolean => DataType::Boolean,
                FieldType::Vector(dim) => DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    *dim as i32,
                ),
            };
            Field::new(&d.name, dt, d.nullable)
        })
        .collect();
    Schema::new(fields)
}

/// Convert a batch of [`Record`]s to an Arrow [`RecordBatch`].
///
/// All records must have the same columns in the same order.
pub fn records_to_batch(records: &[Record]) -> Result<RecordBatch> {
    if records.is_empty() {
        anyhow::bail!("Cannot create RecordBatch from zero records");
    }

    let first = &records[0];
    let num_rows = records.len();

    let mut fields = Vec::with_capacity(first.len());
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(first.len());

    for (col_idx, (name, sample)) in first.iter().enumerate() {
        match sample {
            FieldValue::Utf8(_) => {
                let values: Vec<Option<&str>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Utf8(v) => v.as_deref(),
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Utf8, nullable));
                columns.push(Arc::new(StringArray::from(values)));
            }
            FieldValue::Int32(_) => {
                let values: Vec<Option<i32>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Int32(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Int32, nullable));
                columns.push(Arc::new(Int32Array::from(values)));
            }
            FieldValue::Int64(_) => {
                let values: Vec<Option<i64>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Int64(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Int64, nullable));
                columns.push(Arc::new(Int64Array::from(values)));
            }
            FieldValue::UInt32(_) => {
                let values: Vec<Option<u32>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::UInt32(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::UInt32, nullable));
                columns.push(Arc::new(UInt32Array::from(values)));
            }
            FieldValue::UInt64(_) => {
                let values: Vec<Option<u64>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::UInt64(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::UInt64, nullable));
                columns.push(Arc::new(UInt64Array::from(values)));
            }
            FieldValue::Float32(_) => {
                let values: Vec<Option<f32>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Float32(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Float32, nullable));
                columns.push(Arc::new(Float32Array::from(values)));
            }
            FieldValue::Float64(_) => {
                let values: Vec<Option<f64>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Float64(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Float64, nullable));
                columns.push(Arc::new(Float64Array::from(values)));
            }
            FieldValue::Boolean(_) => {
                let values: Vec<Option<bool>> = records
                    .iter()
                    .map(|r| match &r[col_idx].1 {
                        FieldValue::Boolean(v) => *v,
                        _ => None,
                    })
                    .collect();
                let nullable = values.iter().any(|v| v.is_none());
                fields.push(Field::new(name, DataType::Boolean, nullable));
                columns.push(Arc::new(BooleanArray::from(values)));
            }
            FieldValue::Vector(sample_vec) => {
                let dim = sample_vec.len() as i32;
                let mut flat_values: Vec<f32> = Vec::with_capacity(num_rows * dim as usize);
                for r in records {
                    match &r[col_idx].1 {
                        FieldValue::Vector(v) => flat_values.extend_from_slice(v),
                        _ => flat_values.extend(std::iter::repeat_n(0.0f32, dim as usize)),
                    }
                }
                let values_array = Float32Array::from(flat_values);
                let list_field = Arc::new(Field::new("item", DataType::Float32, true));
                let list = FixedSizeListArray::try_new(
                    list_field.clone(),
                    dim,
                    Arc::new(values_array),
                    None,
                )?;
                fields.push(Field::new(
                    name,
                    DataType::FixedSizeList(list_field, dim),
                    false,
                ));
                columns.push(Arc::new(list));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Extract all rows from a [`RecordBatch`] into [`Record`]s.
pub fn batch_to_records(batch: &RecordBatch, out: &mut Vec<Record>) -> Result<()> {
    for row in 0..batch.num_rows() {
        let mut record = Vec::with_capacity(batch.num_columns());
        for (col_idx, field) in batch.schema().fields().iter().enumerate() {
            let val = extract_field_value(batch, col_idx, row, field)?;
            record.push((field.name().clone(), val));
        }
        out.push(record);
    }
    Ok(())
}

/// Extract a single [`FieldValue`] from a batch cell.
pub fn extract_field_value(
    batch: &RecordBatch,
    col_idx: usize,
    row: usize,
    field: &Field,
) -> Result<FieldValue> {
    let col = batch.column(col_idx);

    Ok(match field.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            if arr.is_null(row) {
                FieldValue::Utf8(None)
            } else {
                FieldValue::Utf8(Some(arr.value(row).to_string()))
            }
        }
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::Int32(None)
            } else {
                FieldValue::Int32(Some(arr.value(row)))
            }
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::Int64(None)
            } else {
                FieldValue::Int64(Some(arr.value(row)))
            }
        }
        DataType::UInt32 => {
            let arr = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::UInt32(None)
            } else {
                FieldValue::UInt32(Some(arr.value(row)))
            }
        }
        DataType::UInt64 => {
            let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::UInt64(None)
            } else {
                FieldValue::UInt64(Some(arr.value(row)))
            }
        }
        DataType::Float32 => {
            let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::Float32(None)
            } else {
                FieldValue::Float32(Some(arr.value(row)))
            }
        }
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
            if arr.is_null(row) {
                FieldValue::Float64(None)
            } else {
                FieldValue::Float64(Some(arr.value(row)))
            }
        }
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            if arr.is_null(row) {
                FieldValue::Boolean(None)
            } else {
                FieldValue::Boolean(Some(arr.value(row)))
            }
        }
        DataType::FixedSizeList(_, _dim) => {
            let arr = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            if arr.is_null(row) {
                FieldValue::Vector(Vec::new())
            } else {
                let inner = arr.value(row);
                let floats = inner.as_any().downcast_ref::<Float32Array>().unwrap();
                FieldValue::Vector(floats.values().to_vec())
            }
        }
        other => {
            tracing::warn!("Unsupported Arrow data type {other:?}, reading as Utf8");
            FieldValue::Utf8(Some(format!("{:?}", col)))
        }
    })
}

/// Escape a column name for use in a LanceDB SQL-like filter.
///
/// Wraps the name in backticks and escapes any backticks within the name,
/// preventing column names from being misinterpreted as SQL keywords.
fn quote_col(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Convert a [`Filter`] to a LanceDB SQL-like filter string.
///
/// Column names are backtick-quoted to prevent injection when they coincide
/// with SQL reserved words. String values are single-quote–escaped.
///
/// # Security note
/// [`Filter::Raw`] is inherently unsafe — it accepts arbitrary SQL and cannot
/// be escaped. It is deprecated and emits a warning at runtime. Callers should
/// migrate to the typed `Filter` variants.
pub fn filter_to_sql(filter: &Filter) -> String {
    match filter {
        Filter::Eq(col, val) => format!("{} = {}", quote_col(col), value_to_sql(val)),
        Filter::Ne(col, val) => format!("{} != {}", quote_col(col), value_to_sql(val)),
        Filter::Lt(col, val) => format!("{} < {}", quote_col(col), value_to_sql(val)),
        Filter::Lte(col, val) => format!("{} <= {}", quote_col(col), value_to_sql(val)),
        Filter::Gt(col, val) => format!("{} > {}", quote_col(col), value_to_sql(val)),
        Filter::Gte(col, val) => format!("{} >= {}", quote_col(col), value_to_sql(val)),
        Filter::NotNull(col) => format!("{} IS NOT NULL", quote_col(col)),
        Filter::IsNull(col) => format!("{} IS NULL", quote_col(col)),
        Filter::In(col, vals) => {
            let items: Vec<String> = vals.iter().map(value_to_sql).collect();
            format!("{} IN ({})", quote_col(col), items.join(", "))
        }
        Filter::And(parts) => {
            let clauses: Vec<String> = parts.iter().map(filter_to_sql).collect();
            format!("({})", clauses.join(" AND "))
        }
        Filter::Or(parts) => {
            let clauses: Vec<String> = parts.iter().map(filter_to_sql).collect();
            format!("({})", clauses.join(" OR "))
        }
        Filter::Raw(s) => {
            // Filter::Raw is an explicit escape hatch for backend-specific
            // SQL expressions. Callers are responsible for ensuring the
            // contents are safe (no untrusted user input).
            s.clone()
        }
    }
}

/// Convert a [`FieldValue`] to a SQL literal for LanceDB filters.
pub fn value_to_sql(val: &FieldValue) -> String {
    match val {
        FieldValue::Utf8(Some(s)) => format!("'{}'", s.replace('\'', "''")),
        FieldValue::Utf8(None) => "NULL".to_string(),
        FieldValue::Int32(Some(v)) => v.to_string(),
        FieldValue::Int32(None) => "NULL".to_string(),
        FieldValue::Int64(Some(v)) => v.to_string(),
        FieldValue::Int64(None) => "NULL".to_string(),
        FieldValue::UInt32(Some(v)) => v.to_string(),
        FieldValue::UInt32(None) => "NULL".to_string(),
        FieldValue::UInt64(Some(v)) => v.to_string(),
        FieldValue::UInt64(None) => "NULL".to_string(),
        FieldValue::Float32(Some(v)) => v.to_string(),
        FieldValue::Float32(None) => "NULL".to_string(),
        FieldValue::Float64(Some(v)) => v.to_string(),
        FieldValue::Float64(None) => "NULL".to_string(),
        FieldValue::Boolean(Some(v)) => if *v { "TRUE" } else { "FALSE" }.to_string(),
        FieldValue::Boolean(None) => "NULL".to_string(),
        FieldValue::Vector(_) => "NULL".to_string(),
    }
}
