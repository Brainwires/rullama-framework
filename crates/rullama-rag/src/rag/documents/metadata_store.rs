//! Document Metadata Store
//!
//! Stores and retrieves document-level metadata (not chunks) in LanceDB.
//! Used to track which documents have been indexed and their properties.

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;

use super::lance_tables;
use super::types::{DocumentMetadata, DocumentType};

/// Store for document metadata
pub struct DocumentMetadataStore {
    connection: Arc<Connection>,
}

impl DocumentMetadataStore {
    /// Create a new document metadata store
    pub fn new(connection: Arc<Connection>) -> Self {
        Self { connection }
    }

    /// Save document metadata
    pub async fn save(&self, metadata: &DocumentMetadata) -> Result<()> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;
        let schema = lance_tables::document_metadata_schema();

        // Check if document already exists
        if self.get(&metadata.document_id).await?.is_some() {
            // Delete existing record first
            table
                .delete(&format!("document_id = '{}'", metadata.document_id))
                .await
                .context("Failed to delete existing document metadata")?;
        }

        // Create record batch
        let batch = self.metadata_to_batch(metadata, &schema)?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to save document metadata")?;

        Ok(())
    }

    /// Get document metadata by ID
    pub async fn get(&self, document_id: &str) -> Result<Option<DocumentMetadata>> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        let filter = format!("document_id = '{}'", document_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query document metadata")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        if batches.is_empty() {
            return Ok(None);
        }

        let batch = &batches[0];
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        Ok(Some(self.batch_to_metadata(batch, 0)?))
    }

    /// Get document by file hash (to detect duplicates)
    pub async fn get_by_hash(&self, file_hash: &str) -> Result<Option<DocumentMetadata>> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        let filter = format!("file_hash = '{}'", file_hash);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query document by hash")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        if batches.is_empty() {
            return Ok(None);
        }

        let batch = &batches[0];
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        Ok(Some(self.batch_to_metadata(batch, 0)?))
    }

    /// List documents for a conversation
    pub async fn list_by_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<DocumentMetadata>> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        let filter = format!("conversation_id = '{}'", conversation_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to list documents by conversation")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut documents = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                documents.push(self.batch_to_metadata(batch, i)?);
            }
        }

        // Sort by created_at descending
        documents.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(documents)
    }

    /// List documents for a project
    pub async fn list_by_project(&self, project_id: &str) -> Result<Vec<DocumentMetadata>> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        let filter = format!("project_id = '{}'", project_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to list documents by project")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut documents = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                documents.push(self.batch_to_metadata(batch, i)?);
            }
        }

        documents.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(documents)
    }

    /// List all documents
    pub async fn list_all(&self) -> Result<Vec<DocumentMetadata>> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        let stream = table
            .query()
            .execute()
            .await
            .context("Failed to list all documents")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut documents = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                documents.push(self.batch_to_metadata(batch, i)?);
            }
        }

        documents.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(documents)
    }

    /// Delete document metadata
    pub async fn delete(&self, document_id: &str) -> Result<bool> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;

        // Check if exists
        if self.get(document_id).await?.is_none() {
            return Ok(false);
        }

        table
            .delete(&format!("document_id = '{}'", document_id))
            .await
            .context("Failed to delete document metadata")?;

        Ok(true)
    }

    /// Count all documents
    pub async fn count(&self) -> Result<usize> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }

    /// Count documents for a conversation
    pub async fn count_by_conversation(&self, conversation_id: &str) -> Result<usize> {
        let table = lance_tables::open_document_metadata_table(&self.connection).await?;
        let filter = format!("conversation_id = '{}'", conversation_id);
        let count = table.count_rows(Some(filter)).await?;
        Ok(count)
    }

    /// Convert DocumentMetadata to RecordBatch
    fn metadata_to_batch(
        &self,
        metadata: &DocumentMetadata,
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch> {
        let document_id = StringArray::from(vec![metadata.document_id.as_str()]);
        let conversation_id =
            StringArray::from(vec![metadata.conversation_id.as_deref().unwrap_or("")]);
        let project_id = StringArray::from(vec![metadata.project_id.as_deref().unwrap_or("")]);
        let file_name = StringArray::from(vec![metadata.file_name.as_str()]);
        let file_type = StringArray::from(vec![format!("{:?}", metadata.file_type).as_str()]);
        let file_size_bytes = UInt64Array::from(vec![metadata.file_size_bytes]);
        let chunk_count = UInt32Array::from(vec![metadata.chunk_count]);
        let file_hash = StringArray::from(vec![metadata.file_hash.as_str()]);
        let title = StringArray::from(vec![metadata.title.as_deref().unwrap_or("")]);
        let page_count = UInt32Array::from(vec![metadata.page_count.unwrap_or(0)]);
        let created_at = Int64Array::from(vec![metadata.created_at]);

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(document_id) as ArrayRef,
                Arc::new(conversation_id),
                Arc::new(project_id),
                Arc::new(file_name),
                Arc::new(file_type),
                Arc::new(file_size_bytes),
                Arc::new(chunk_count),
                Arc::new(file_hash),
                Arc::new(title),
                Arc::new(page_count),
                Arc::new(created_at),
            ],
        )
        .context("Failed to create record batch for document metadata")
    }

    /// Convert RecordBatch row to DocumentMetadata
    fn batch_to_metadata(&self, batch: &RecordBatch, row: usize) -> Result<DocumentMetadata> {
        let document_id = batch
            .column_by_name("document_id")
            .context("Missing document_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid document_id type")?
            .value(row)
            .to_string();

        let conversation_id_str = batch
            .column_by_name("conversation_id")
            .context("Missing conversation_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid conversation_id type")?
            .value(row);
        let conversation_id = if conversation_id_str.is_empty() {
            None
        } else {
            Some(conversation_id_str.to_string())
        };

        let project_id_str = batch
            .column_by_name("project_id")
            .context("Missing project_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid project_id type")?
            .value(row);
        let project_id = if project_id_str.is_empty() {
            None
        } else {
            Some(project_id_str.to_string())
        };

        let file_name = batch
            .column_by_name("file_name")
            .context("Missing file_name")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid file_name type")?
            .value(row)
            .to_string();

        let file_type_str = batch
            .column_by_name("file_type")
            .context("Missing file_type")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid file_type type")?
            .value(row);
        let file_type = match file_type_str {
            "Pdf" => DocumentType::Pdf,
            "Markdown" => DocumentType::Markdown,
            "PlainText" => DocumentType::PlainText,
            "Docx" => DocumentType::Docx,
            _ => DocumentType::Unknown,
        };

        let file_size_bytes = batch
            .column_by_name("file_size_bytes")
            .context("Missing file_size_bytes")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("Invalid file_size_bytes type")?
            .value(row);

        let chunk_count = batch
            .column_by_name("chunk_count")
            .context("Missing chunk_count")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Invalid chunk_count type")?
            .value(row);

        let file_hash = batch
            .column_by_name("file_hash")
            .context("Missing file_hash")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid file_hash type")?
            .value(row)
            .to_string();

        let title_str = batch
            .column_by_name("title")
            .context("Missing title")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid title type")?
            .value(row);
        let title = if title_str.is_empty() {
            None
        } else {
            Some(title_str.to_string())
        };

        let page_count_val = batch
            .column_by_name("page_count")
            .context("Missing page_count")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Invalid page_count type")?
            .value(row);
        let page_count = if page_count_val == 0 {
            None
        } else {
            Some(page_count_val)
        };

        let created_at = batch
            .column_by_name("created_at")
            .context("Missing created_at")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("Invalid created_at type")?
            .value(row);

        Ok(DocumentMetadata {
            document_id,
            conversation_id,
            project_id,
            file_name,
            file_type,
            file_size_bytes,
            chunk_count,
            file_hash,
            title,
            page_count,
            created_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> (DocumentMetadataStore, TempDir) {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.lance");

        let connection = Arc::new(
            lancedb::connect(db_path.to_str().unwrap())
                .execute()
                .await
                .unwrap(),
        );
        lance_tables::ensure_document_metadata_table(&connection)
            .await
            .unwrap();

        let store = DocumentMetadataStore::new(connection);
        (store, temp)
    }

    fn create_test_metadata() -> DocumentMetadata {
        DocumentMetadata::new(
            "doc-123".to_string(),
            "test.pdf".to_string(),
            DocumentType::Pdf,
            1024,
            "abc123hash".to_string(),
        )
        .with_conversation("conv-456".to_string())
        .with_chunk_count(5)
        .with_title("Test Document".to_string())
    }

    #[tokio::test]
    async fn test_save_and_get() {
        let (store, _temp) = create_test_store().await;
        let metadata = create_test_metadata();

        store.save(&metadata).await.unwrap();

        let retrieved = store.get(&metadata.document_id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.document_id, metadata.document_id);
        assert_eq!(retrieved.file_name, metadata.file_name);
        assert_eq!(retrieved.file_type, DocumentType::Pdf);
        assert_eq!(retrieved.chunk_count, 5);
        assert_eq!(retrieved.title, Some("Test Document".to_string()));
    }

    #[tokio::test]
    async fn test_get_by_hash() {
        let (store, _temp) = create_test_store().await;
        let metadata = create_test_metadata();

        store.save(&metadata).await.unwrap();

        let retrieved = store.get_by_hash(&metadata.file_hash).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().document_id, metadata.document_id);
    }

    #[tokio::test]
    async fn test_delete() {
        let (store, _temp) = create_test_store().await;
        let metadata = create_test_metadata();

        store.save(&metadata).await.unwrap();
        assert!(store.get(&metadata.document_id).await.unwrap().is_some());

        let deleted = store.delete(&metadata.document_id).await.unwrap();
        assert!(deleted);

        assert!(store.get(&metadata.document_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_count() {
        let (store, _temp) = create_test_store().await;

        assert_eq!(store.count().await.unwrap(), 0);

        let metadata = create_test_metadata();
        store.save(&metadata).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 1);
    }
}
