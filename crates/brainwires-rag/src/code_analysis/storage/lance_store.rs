//! LanceDB-based storage for code relationships.

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{RelationsStats, RelationsStore};
use crate::code_analysis::types::{
    CallEdge, Definition, Reference, ReferenceKind, SymbolId, SymbolKind, Visibility,
};

/// Table name for definitions
const DEFINITIONS_TABLE: &str = "definitions";
/// Table name for references
const REFERENCES_TABLE: &str = "references";
/// Table name for call edges
const CALL_EDGES_TABLE: &str = "call_edges";

/// LanceDB-based relations store.
///
/// Stores definitions and references in separate LanceDB tables for efficient querying.
pub struct LanceRelationsStore {
    /// Path to the database directory
    db_path: PathBuf,
    /// Database connection (lazy initialized)
    db: Arc<RwLock<Option<lancedb::Connection>>>,
}

/// Arrow schema for the definitions table
fn definitions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("symbol_id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("symbol_name", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("start_col", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("end_col", DataType::UInt32, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("doc_comment", DataType::Utf8, true),
        Field::new("visibility", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("root_path", DataType::Utf8, true),
        Field::new("project", DataType::Utf8, true),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

/// Arrow schema for the references table
fn references_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("start_col", DataType::UInt32, false),
        Field::new("end_col", DataType::UInt32, false),
        Field::new("target_symbol_id", DataType::Utf8, false),
        Field::new("reference_kind", DataType::Utf8, false),
        Field::new("root_path", DataType::Utf8, true),
        Field::new("project", DataType::Utf8, true),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

impl LanceRelationsStore {
    /// Create a new LanceDB relations store
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        // Ensure directory exists
        tokio::fs::create_dir_all(&db_path)
            .await
            .context("Failed to create relations database directory")?;

        Ok(Self {
            db_path,
            db: Arc::new(RwLock::new(None)),
        })
    }

    /// Get or create the database connection
    async fn get_connection(&self) -> Result<lancedb::Connection> {
        let mut db_guard = self.db.write().await;

        if let Some(ref db) = *db_guard {
            return Ok(db.clone());
        }

        let db = lancedb::connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        *db_guard = Some(db.clone());
        Ok(db)
    }

    /// Ensure definitions table exists
    async fn ensure_definitions_table(&self) -> Result<()> {
        let db = self.get_connection().await?;
        let table_names = db.table_names().execute().await?;

        if table_names.contains(&DEFINITIONS_TABLE.to_string()) {
            return Ok(());
        }

        let schema = definitions_schema();
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

        db.create_table(
            DEFINITIONS_TABLE,
            Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>,
        )
        .execute()
        .await
        .context("Failed to create definitions table")?;

        Ok(())
    }

    /// Ensure references table exists
    async fn ensure_references_table(&self) -> Result<()> {
        let db = self.get_connection().await?;
        let table_names = db.table_names().execute().await?;

        if table_names.contains(&REFERENCES_TABLE.to_string()) {
            return Ok(());
        }

        let schema = references_schema();
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

        db.create_table(
            REFERENCES_TABLE,
            Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>,
        )
        .execute()
        .await
        .context("Failed to create references table")?;

        Ok(())
    }

    /// Helper: check if a table exists
    async fn table_exists(&self, table_name: &str) -> Result<bool> {
        let db = self.get_connection().await?;
        let table_names = db.table_names().execute().await?;
        Ok(table_names.contains(&table_name.to_string()))
    }

    /// Helper: get string value from a RecordBatch column
    fn get_string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
        Ok(batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {column}"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| format!("Invalid type for column: {column}"))?
            .value(row)
            .to_string())
    }

    /// Helper: get optional string value (empty string becomes None)
    fn get_optional_string(
        batch: &RecordBatch,
        column: &str,
        row: usize,
    ) -> Result<Option<String>> {
        let arr = batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {column}"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| format!("Invalid type for column: {column}"))?;

        if arr.is_null(row) {
            return Ok(None);
        }
        let val = arr.value(row);
        if val.is_empty() {
            Ok(None)
        } else {
            Ok(Some(val.to_string()))
        }
    }

    /// Helper: get u32 value from a RecordBatch column
    fn get_u32_value(batch: &RecordBatch, column: &str, row: usize) -> Result<u32> {
        Ok(batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {column}"))?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .with_context(|| format!("Invalid type for column: {column}"))?
            .value(row))
    }

    /// Helper: get i64 value from a RecordBatch column
    fn get_i64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<i64> {
        Ok(batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {column}"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .with_context(|| format!("Invalid type for column: {column}"))?
            .value(row))
    }

    /// Parse a SymbolKind from its display name
    fn parse_symbol_kind(s: &str) -> SymbolKind {
        match s {
            "function" => SymbolKind::Function,
            "method" => SymbolKind::Method,
            "class" => SymbolKind::Class,
            "struct" => SymbolKind::Struct,
            "interface" => SymbolKind::Interface,
            "trait" => SymbolKind::Trait,
            "enum" => SymbolKind::Enum,
            "module" => SymbolKind::Module,
            "variable" => SymbolKind::Variable,
            "constant" => SymbolKind::Constant,
            "parameter" => SymbolKind::Parameter,
            "field" => SymbolKind::Field,
            "import" => SymbolKind::Import,
            "export" => SymbolKind::Export,
            "enum variant" => SymbolKind::EnumVariant,
            "type alias" => SymbolKind::TypeAlias,
            _ => SymbolKind::Unknown,
        }
    }

    /// Parse a Visibility from its string representation
    fn parse_visibility(s: &str) -> Visibility {
        match s {
            "public" => Visibility::Public,
            "private" => Visibility::Private,
            "protected" => Visibility::Protected,
            "internal" => Visibility::Internal,
            _ => Visibility::Private,
        }
    }

    /// Parse a ReferenceKind from its string representation
    fn parse_reference_kind(s: &str) -> ReferenceKind {
        match s {
            "call" => ReferenceKind::Call,
            "read" => ReferenceKind::Read,
            "write" => ReferenceKind::Write,
            "import" => ReferenceKind::Import,
            "type_reference" => ReferenceKind::TypeReference,
            "inheritance" => ReferenceKind::Inheritance,
            "instantiation" => ReferenceKind::Instantiation,
            _ => ReferenceKind::Unknown,
        }
    }

    /// Convert a RecordBatch row into a Definition
    fn row_to_definition(batch: &RecordBatch, row: usize) -> Result<Definition> {
        let file_path = Self::get_string_value(batch, "file_path", row)?;
        let symbol_name = Self::get_string_value(batch, "symbol_name", row)?;
        let kind_str = Self::get_string_value(batch, "kind", row)?;
        let start_line = Self::get_u32_value(batch, "start_line", row)? as usize;
        let start_col = Self::get_u32_value(batch, "start_col", row)? as usize;
        let end_line = Self::get_u32_value(batch, "end_line", row)? as usize;
        let end_col = Self::get_u32_value(batch, "end_col", row)? as usize;
        let signature = Self::get_string_value(batch, "signature", row)?;
        let doc_comment = Self::get_optional_string(batch, "doc_comment", row)?;
        let visibility_str = Self::get_string_value(batch, "visibility", row)?;
        let parent_id = Self::get_optional_string(batch, "parent_id", row)?;
        let root_path = Self::get_optional_string(batch, "root_path", row)?;
        let project = Self::get_optional_string(batch, "project", row)?;
        let indexed_at = Self::get_i64_value(batch, "indexed_at", row)?;

        Ok(Definition {
            symbol_id: SymbolId {
                file_path,
                name: symbol_name,
                kind: Self::parse_symbol_kind(&kind_str),
                start_line,
                start_col,
            },
            root_path,
            project,
            end_line,
            end_col,
            signature,
            doc_comment,
            visibility: Self::parse_visibility(&visibility_str),
            parent_id,
            indexed_at,
        })
    }

    /// Convert a RecordBatch row into a Reference
    fn row_to_reference(batch: &RecordBatch, row: usize) -> Result<Reference> {
        let file_path = Self::get_string_value(batch, "file_path", row)?;
        let start_line = Self::get_u32_value(batch, "start_line", row)? as usize;
        let end_line = Self::get_u32_value(batch, "end_line", row)? as usize;
        let start_col = Self::get_u32_value(batch, "start_col", row)? as usize;
        let end_col = Self::get_u32_value(batch, "end_col", row)? as usize;
        let target_symbol_id = Self::get_string_value(batch, "target_symbol_id", row)?;
        let kind_str = Self::get_string_value(batch, "reference_kind", row)?;
        let root_path = Self::get_optional_string(batch, "root_path", row)?;
        let project = Self::get_optional_string(batch, "project", row)?;
        let indexed_at = Self::get_i64_value(batch, "indexed_at", row)?;

        Ok(Reference {
            file_path,
            root_path,
            project,
            start_line,
            end_line,
            start_col,
            end_col,
            target_symbol_id,
            reference_kind: Self::parse_reference_kind(&kind_str),
            indexed_at,
        })
    }

    /// Convert a RecordBatch row into a CallEdge
    fn row_to_call_edge(batch: &RecordBatch, row: usize) -> Result<CallEdge> {
        Ok(CallEdge {
            caller_id: Self::get_string_value(batch, "caller_id", row)?,
            callee_id: Self::get_string_value(batch, "callee_id", row)?,
            call_site_file: Self::get_string_value(batch, "call_site_file", row)?,
            call_site_line: Self::get_u32_value(batch, "call_site_line", row)? as usize,
            call_site_col: Self::get_u32_value(batch, "call_site_col", row)? as usize,
        })
    }

    /// Escape a string value for use in a LanceDB SQL filter
    fn escape_filter_value(value: &str) -> String {
        value.replace('\'', "''")
    }
}

#[async_trait]
impl RelationsStore for LanceRelationsStore {
    async fn store_definitions(
        &self,
        definitions: Vec<Definition>,
        _root_path: &str,
    ) -> Result<usize> {
        if definitions.is_empty() {
            return Ok(0);
        }

        self.ensure_definitions_table().await?;

        let count = definitions.len();
        let schema = definitions_schema();

        // Build column arrays
        let symbol_ids: Vec<String> = definitions
            .iter()
            .map(|d| d.symbol_id.to_storage_id())
            .collect();
        let file_paths: Vec<&str> = definitions
            .iter()
            .map(|d| d.symbol_id.file_path.as_str())
            .collect();
        let symbol_names: Vec<&str> = definitions
            .iter()
            .map(|d| d.symbol_id.name.as_str())
            .collect();
        let kinds: Vec<String> = definitions
            .iter()
            .map(|d| d.symbol_id.kind.display_name().to_string())
            .collect();
        let start_lines: Vec<u32> = definitions
            .iter()
            .map(|d| d.symbol_id.start_line as u32)
            .collect();
        let start_cols: Vec<u32> = definitions
            .iter()
            .map(|d| d.symbol_id.start_col as u32)
            .collect();
        let end_lines: Vec<u32> = definitions.iter().map(|d| d.end_line as u32).collect();
        let end_cols: Vec<u32> = definitions.iter().map(|d| d.end_col as u32).collect();
        let signatures: Vec<&str> = definitions.iter().map(|d| d.signature.as_str()).collect();
        let doc_comments: Vec<Option<&str>> = definitions
            .iter()
            .map(|d| d.doc_comment.as_deref())
            .collect();
        let visibilities: Vec<String> = definitions
            .iter()
            .map(|d| format!("{:?}", d.visibility).to_lowercase())
            .collect();
        let parent_ids: Vec<Option<&str>> =
            definitions.iter().map(|d| d.parent_id.as_deref()).collect();
        let root_paths: Vec<Option<&str>> =
            definitions.iter().map(|d| d.root_path.as_deref()).collect();
        let projects: Vec<Option<&str>> =
            definitions.iter().map(|d| d.project.as_deref()).collect();
        let indexed_ats: Vec<i64> = definitions.iter().map(|d| d.indexed_at).collect();

        let symbol_id_refs: Vec<&str> = symbol_ids.iter().map(|s| s.as_str()).collect();
        let kind_refs: Vec<&str> = kinds.iter().map(|s| s.as_str()).collect();
        let visibility_refs: Vec<&str> = visibilities.iter().map(|s| s.as_str()).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(symbol_id_refs)) as ArrayRef,
                Arc::new(StringArray::from(file_paths)) as ArrayRef,
                Arc::new(StringArray::from(symbol_names)) as ArrayRef,
                Arc::new(StringArray::from(kind_refs)) as ArrayRef,
                Arc::new(UInt32Array::from(start_lines)) as ArrayRef,
                Arc::new(UInt32Array::from(start_cols)) as ArrayRef,
                Arc::new(UInt32Array::from(end_lines)) as ArrayRef,
                Arc::new(UInt32Array::from(end_cols)) as ArrayRef,
                Arc::new(StringArray::from(signatures)) as ArrayRef,
                Arc::new(StringArray::from(doc_comments)) as ArrayRef,
                Arc::new(StringArray::from(visibility_refs)) as ArrayRef,
                Arc::new(StringArray::from(parent_ids)) as ArrayRef,
                Arc::new(StringArray::from(root_paths)) as ArrayRef,
                Arc::new(StringArray::from(projects)) as ArrayRef,
                Arc::new(Int64Array::from(indexed_ats)) as ArrayRef,
            ],
        )
        .context("Failed to create definitions RecordBatch")?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let db = self.get_connection().await?;
        let table = db
            .open_table(DEFINITIONS_TABLE)
            .execute()
            .await
            .context("Failed to open definitions table")?;

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to add definitions to LanceDB")?;

        tracing::debug!("Stored {} definitions", count);
        Ok(count)
    }

    async fn store_references(
        &self,
        references: Vec<Reference>,
        _root_path: &str,
    ) -> Result<usize> {
        if references.is_empty() {
            return Ok(0);
        }

        self.ensure_references_table().await?;

        let count = references.len();
        let schema = references_schema();

        let file_paths: Vec<&str> = references.iter().map(|r| r.file_path.as_str()).collect();
        let start_lines: Vec<u32> = references.iter().map(|r| r.start_line as u32).collect();
        let end_lines: Vec<u32> = references.iter().map(|r| r.end_line as u32).collect();
        let start_cols: Vec<u32> = references.iter().map(|r| r.start_col as u32).collect();
        let end_cols: Vec<u32> = references.iter().map(|r| r.end_col as u32).collect();
        let target_ids: Vec<&str> = references
            .iter()
            .map(|r| r.target_symbol_id.as_str())
            .collect();
        let ref_kinds: Vec<String> = references
            .iter()
            .map(|r| {
                serde_json::to_value(r.reference_kind)
                    .ok()
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown".to_string())
            })
            .collect();
        let root_paths: Vec<Option<&str>> =
            references.iter().map(|r| r.root_path.as_deref()).collect();
        let projects: Vec<Option<&str>> = references.iter().map(|r| r.project.as_deref()).collect();
        let indexed_ats: Vec<i64> = references.iter().map(|r| r.indexed_at).collect();

        let ref_kind_refs: Vec<&str> = ref_kinds.iter().map(|s| s.as_str()).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(file_paths)) as ArrayRef,
                Arc::new(UInt32Array::from(start_lines)) as ArrayRef,
                Arc::new(UInt32Array::from(end_lines)) as ArrayRef,
                Arc::new(UInt32Array::from(start_cols)) as ArrayRef,
                Arc::new(UInt32Array::from(end_cols)) as ArrayRef,
                Arc::new(StringArray::from(target_ids)) as ArrayRef,
                Arc::new(StringArray::from(ref_kind_refs)) as ArrayRef,
                Arc::new(StringArray::from(root_paths)) as ArrayRef,
                Arc::new(StringArray::from(projects)) as ArrayRef,
                Arc::new(Int64Array::from(indexed_ats)) as ArrayRef,
            ],
        )
        .context("Failed to create references RecordBatch")?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let db = self.get_connection().await?;
        let table = db
            .open_table(REFERENCES_TABLE)
            .execute()
            .await
            .context("Failed to open references table")?;

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to add references to LanceDB")?;

        tracing::debug!("Stored {} references", count);
        Ok(count)
    }

    async fn find_definition_at(
        &self,
        file_path: &str,
        line: usize,
        _column: usize,
    ) -> Result<Option<Definition>> {
        if !self.table_exists(DEFINITIONS_TABLE).await? {
            return Ok(None);
        }

        let db = self.get_connection().await?;
        let table = db
            .open_table(DEFINITIONS_TABLE)
            .execute()
            .await
            .context("Failed to open definitions table")?;

        let escaped_path = Self::escape_filter_value(file_path);
        let filter = format!(
            "file_path = '{}' AND start_line <= {} AND end_line >= {}",
            escaped_path, line as u32, line as u32
        );

        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query definitions")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        // Find the most specific (smallest span) definition containing the position
        let mut best: Option<Definition> = None;
        let mut best_span: usize = usize::MAX;

        for batch in &batches {
            for i in 0..batch.num_rows() {
                let def = Self::row_to_definition(batch, i)?;
                let span = def.end_line.saturating_sub(def.symbol_id.start_line);
                if span < best_span {
                    best_span = span;
                    best = Some(def);
                }
            }
        }

        Ok(best)
    }

    async fn find_definitions_by_name(&self, name: &str) -> Result<Vec<Definition>> {
        if !self.table_exists(DEFINITIONS_TABLE).await? {
            return Ok(Vec::new());
        }

        let db = self.get_connection().await?;
        let table = db
            .open_table(DEFINITIONS_TABLE)
            .execute()
            .await
            .context("Failed to open definitions table")?;

        let escaped_name = Self::escape_filter_value(name);
        let filter = format!("symbol_name = '{}'", escaped_name);

        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query definitions by name")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                results.push(Self::row_to_definition(batch, i)?);
            }
        }

        Ok(results)
    }

    async fn find_references(&self, target_symbol_id: &str) -> Result<Vec<Reference>> {
        if !self.table_exists(REFERENCES_TABLE).await? {
            return Ok(Vec::new());
        }

        let db = self.get_connection().await?;
        let table = db
            .open_table(REFERENCES_TABLE)
            .execute()
            .await
            .context("Failed to open references table")?;

        let escaped_id = Self::escape_filter_value(target_symbol_id);
        let filter = format!("target_symbol_id = '{}'", escaped_id);

        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query references")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                results.push(Self::row_to_reference(batch, i)?);
            }
        }

        Ok(results)
    }

    async fn get_callers(&self, symbol_id: &str) -> Result<Vec<CallEdge>> {
        if !self.table_exists(CALL_EDGES_TABLE).await? {
            return Ok(Vec::new());
        }

        let db = self.get_connection().await?;
        let table = db
            .open_table(CALL_EDGES_TABLE)
            .execute()
            .await
            .context("Failed to open call_edges table")?;

        let escaped_id = Self::escape_filter_value(symbol_id);
        let filter = format!("callee_id = '{}'", escaped_id);

        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query callers")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                results.push(Self::row_to_call_edge(batch, i)?);
            }
        }

        Ok(results)
    }

    async fn get_callees(&self, symbol_id: &str) -> Result<Vec<CallEdge>> {
        if !self.table_exists(CALL_EDGES_TABLE).await? {
            return Ok(Vec::new());
        }

        let db = self.get_connection().await?;
        let table = db
            .open_table(CALL_EDGES_TABLE)
            .execute()
            .await
            .context("Failed to open call_edges table")?;

        let escaped_id = Self::escape_filter_value(symbol_id);
        let filter = format!("caller_id = '{}'", escaped_id);

        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query callees")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                results.push(Self::row_to_call_edge(batch, i)?);
            }
        }

        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        let escaped_path = Self::escape_filter_value(file_path);
        let filter = format!("file_path = '{}'", escaped_path);
        let mut deleted = 0;

        // Delete from definitions table
        if self.table_exists(DEFINITIONS_TABLE).await? {
            let db = self.get_connection().await?;
            let table = db
                .open_table(DEFINITIONS_TABLE)
                .execute()
                .await
                .context("Failed to open definitions table")?;

            // Count rows before deletion
            let stream = table
                .query()
                .only_if(&filter)
                .execute()
                .await
                .context("Failed to query definitions for deletion count")?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let def_count: usize = batches.iter().map(|b| b.num_rows()).sum();
            deleted += def_count;

            table
                .delete(&filter)
                .await
                .context("Failed to delete definitions by file")?;
        }

        // Delete from references table
        if self.table_exists(REFERENCES_TABLE).await? {
            let db = self.get_connection().await?;
            let table = db
                .open_table(REFERENCES_TABLE)
                .execute()
                .await
                .context("Failed to open references table")?;

            let stream = table
                .query()
                .only_if(&filter)
                .execute()
                .await
                .context("Failed to query references for deletion count")?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let ref_count: usize = batches.iter().map(|b| b.num_rows()).sum();
            deleted += ref_count;

            table
                .delete(&filter)
                .await
                .context("Failed to delete references by file")?;
        }

        tracing::debug!("Deleted {} records for file {}", deleted, file_path);
        Ok(deleted)
    }

    async fn clear(&self) -> Result<()> {
        let db = self.get_connection().await?;

        for table_name in [DEFINITIONS_TABLE, REFERENCES_TABLE, CALL_EDGES_TABLE] {
            if self.table_exists(table_name).await? {
                db.drop_table(table_name, &[])
                    .await
                    .with_context(|| format!("Failed to drop table {table_name}"))?;
            }
        }

        tracing::debug!("Cleared all relations tables");
        Ok(())
    }

    async fn get_stats(&self) -> Result<RelationsStats> {
        let mut stats = RelationsStats::default();

        // Count definitions
        if self.table_exists(DEFINITIONS_TABLE).await? {
            let db = self.get_connection().await?;
            let table = db
                .open_table(DEFINITIONS_TABLE)
                .execute()
                .await
                .context("Failed to open definitions table")?;

            let stream = table
                .query()
                .execute()
                .await
                .context("Failed to query definitions for stats")?;

            let batches: Vec<RecordBatch> = stream.try_collect().await?;

            let mut unique_files = HashSet::new();
            for batch in &batches {
                stats.definition_count += batch.num_rows();

                // Collect unique file paths
                if let Some(col) = batch.column_by_name("file_path")
                    && let Some(arr) = col.as_any().downcast_ref::<StringArray>()
                {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            unique_files.insert(arr.value(i).to_string());
                        }
                    }
                }
            }
            stats.files_with_definitions = unique_files.len();
        }

        // Count references
        if self.table_exists(REFERENCES_TABLE).await? {
            let db = self.get_connection().await?;
            let table = db
                .open_table(REFERENCES_TABLE)
                .execute()
                .await
                .context("Failed to open references table")?;

            let stream = table
                .query()
                .execute()
                .await
                .context("Failed to query references for stats")?;

            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            for batch in &batches {
                stats.reference_count += batch.num_rows();
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_store_creation() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.definition_count, 0);
    }

    #[tokio::test]
    async fn test_store_empty_definitions() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let count = store.store_definitions(Vec::new(), "/test").await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_lance_store_and_query_definitions() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let definitions = vec![
            Definition {
                symbol_id: SymbolId::new("src/main.rs", "main", SymbolKind::Function, 1, 0),
                root_path: Some("/project".to_string()),
                project: Some("test".to_string()),
                end_line: 10,
                end_col: 1,
                signature: "fn main()".to_string(),
                doc_comment: Some("Entry point".to_string()),
                visibility: Visibility::Public,
                parent_id: None,
                indexed_at: 12345,
            },
            Definition {
                symbol_id: SymbolId::new("src/lib.rs", "helper", SymbolKind::Function, 5, 0),
                root_path: Some("/project".to_string()),
                project: Some("test".to_string()),
                end_line: 15,
                end_col: 1,
                signature: "fn helper() -> bool".to_string(),
                doc_comment: None,
                visibility: Visibility::Private,
                parent_id: None,
                indexed_at: 12345,
            },
        ];

        let count = store
            .store_definitions(definitions, "/project")
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Query by name
        let found = store.find_definitions_by_name("main").await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].symbol_id.name, "main");
        assert_eq!(found[0].signature, "fn main()");
        assert_eq!(found[0].doc_comment, Some("Entry point".to_string()));

        // Query at location
        let at_line = store.find_definition_at("src/main.rs", 5, 0).await.unwrap();
        assert!(at_line.is_some());
        assert_eq!(at_line.unwrap().symbol_id.name, "main");

        // Stats
        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.definition_count, 2);
        assert_eq!(stats.files_with_definitions, 2);
    }

    #[tokio::test]
    async fn test_lance_store_and_query_references() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let references = vec![Reference {
            file_path: "src/consumer.rs".to_string(),
            root_path: None,
            project: None,
            start_line: 25,
            end_line: 25,
            start_col: 10,
            end_col: 20,
            target_symbol_id: "src/main.rs:main:1:0".to_string(),
            reference_kind: ReferenceKind::Call,
            indexed_at: 12345,
        }];

        let count = store
            .store_references(references, "/project")
            .await
            .unwrap();
        assert_eq!(count, 1);

        let found = store.find_references("src/main.rs:main:1:0").await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].file_path, "src/consumer.rs");
        assert_eq!(found[0].reference_kind, ReferenceKind::Call);

        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.reference_count, 1);
    }

    #[tokio::test]
    async fn test_lance_delete_by_file() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let definitions = vec![
            Definition {
                symbol_id: SymbolId::new("src/main.rs", "main", SymbolKind::Function, 1, 0),
                root_path: None,
                project: None,
                end_line: 10,
                end_col: 1,
                signature: "fn main()".to_string(),
                doc_comment: None,
                visibility: Visibility::Public,
                parent_id: None,
                indexed_at: 12345,
            },
            Definition {
                symbol_id: SymbolId::new("src/lib.rs", "helper", SymbolKind::Function, 5, 0),
                root_path: None,
                project: None,
                end_line: 15,
                end_col: 1,
                signature: "fn helper()".to_string(),
                doc_comment: None,
                visibility: Visibility::Private,
                parent_id: None,
                indexed_at: 12345,
            },
        ];

        store
            .store_definitions(definitions, "/project")
            .await
            .unwrap();

        let deleted = store.delete_by_file("src/main.rs").await.unwrap();
        assert_eq!(deleted, 1);

        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.definition_count, 1);
    }

    #[tokio::test]
    async fn test_lance_clear() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let definitions = vec![Definition {
            symbol_id: SymbolId::new("src/main.rs", "main", SymbolKind::Function, 1, 0),
            root_path: None,
            project: None,
            end_line: 10,
            end_col: 1,
            signature: "fn main()".to_string(),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
            indexed_at: 12345,
        }];

        store
            .store_definitions(definitions, "/project")
            .await
            .unwrap();

        store.clear().await.unwrap();

        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.definition_count, 0);
    }
}
