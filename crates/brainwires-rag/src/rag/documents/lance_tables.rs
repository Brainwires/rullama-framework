//! LanceDB table management for document storage
//!
//! Defines schemas and provides table creation/access helpers for
//! document chunks and metadata tables in LanceDB.

use anyhow::{Context, Result};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lancedb::{Connection, Table};
use std::sync::Arc;

/// Table name for document chunks (with embeddings)
pub const DOCUMENTS_TABLE: &str = "documents";

/// Table name for document metadata
pub const DOCUMENT_METADATA_TABLE: &str = "document_metadata";

/// Schema for document chunks table (with vector embeddings)
pub fn documents_schema(embedding_dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        // Vector field for semantic search
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
        // Chunk identification
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        // Scope filters
        Field::new("conversation_id", DataType::Utf8, true),
        Field::new("project_id", DataType::Utf8, true),
        // Document info
        Field::new("file_name", DataType::Utf8, false),
        Field::new("file_type", DataType::Utf8, false),
        // Chunk content
        Field::new("content", DataType::Utf8, false),
        // Position info
        Field::new("start_offset", DataType::UInt32, false),
        Field::new("end_offset", DataType::UInt32, false),
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("total_chunks", DataType::UInt32, false),
        // Optional metadata
        Field::new("section", DataType::Utf8, true),
        Field::new("page_number", DataType::UInt32, true),
        // Integrity
        Field::new("file_hash", DataType::Utf8, false),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

/// Schema for document metadata table
pub fn document_metadata_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("document_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, true),
        Field::new("project_id", DataType::Utf8, true),
        Field::new("file_name", DataType::Utf8, false),
        Field::new("file_type", DataType::Utf8, false),
        Field::new("file_size_bytes", DataType::UInt64, false),
        Field::new("chunk_count", DataType::UInt32, false),
        Field::new("file_hash", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, true),
        Field::new("page_count", DataType::UInt32, true),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

/// Ensure the documents (chunks) table exists in the database
pub async fn ensure_documents_table(connection: &Connection, embedding_dim: usize) -> Result<()> {
    let table_names = connection.table_names().execute().await?;

    if table_names.contains(&DOCUMENTS_TABLE.to_string()) {
        return Ok(());
    }

    let schema = documents_schema(embedding_dim);
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

    connection
        .create_table(
            DOCUMENTS_TABLE,
            Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>,
        )
        .execute()
        .await
        .context("Failed to create documents table")?;

    Ok(())
}

/// Ensure the document metadata table exists in the database
pub async fn ensure_document_metadata_table(connection: &Connection) -> Result<()> {
    let table_names = connection.table_names().execute().await?;

    if table_names.contains(&DOCUMENT_METADATA_TABLE.to_string()) {
        return Ok(());
    }

    let schema = document_metadata_schema();
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

    connection
        .create_table(
            DOCUMENT_METADATA_TABLE,
            Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>,
        )
        .execute()
        .await
        .context("Failed to create document_metadata table")?;

    Ok(())
}

/// Open the documents (chunks) table
pub async fn open_documents_table(connection: &Connection) -> Result<Table> {
    connection
        .open_table(DOCUMENTS_TABLE)
        .execute()
        .await
        .context("Failed to open documents table")
}

/// Open the document metadata table
pub async fn open_document_metadata_table(connection: &Connection) -> Result<Table> {
    connection
        .open_table(DOCUMENT_METADATA_TABLE)
        .execute()
        .await
        .context("Failed to open document_metadata table")
}
