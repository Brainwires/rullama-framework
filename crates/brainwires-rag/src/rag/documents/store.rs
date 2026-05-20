//! Document Store with Hybrid Search
//!
//! Provides storage and retrieval for document chunks with support for
//! hybrid search (vector + BM25) using Reciprocal Rank Fusion (RRF).

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray, UInt32Array,
};
use brainwires_core::EmbeddingProvider;
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::bm25::{DocumentBM25Manager, document_rrf_fusion};
use super::chunker::DocumentChunker;
use super::lance_tables;
use super::metadata_store::DocumentMetadataStore;
use super::processor::DocumentProcessor;
use super::types::{
    ChunkerConfig, DocumentChunk, DocumentMetadata, DocumentSearchRequest, DocumentSearchResult,
    DocumentType,
};

/// Main document store with hybrid search capabilities
pub struct DocumentStore {
    connection: Arc<Connection>,
    embeddings: Arc<dyn EmbeddingProvider>,
    bm25_manager: DocumentBM25Manager,
    metadata_store: DocumentMetadataStore,
    chunker: DocumentChunker,
}

impl DocumentStore {
    /// Create a new document store
    pub fn new(
        connection: Arc<Connection>,
        embeddings: Arc<dyn EmbeddingProvider>,
        bm25_base_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            metadata_store: DocumentMetadataStore::new(Arc::clone(&connection)),
            connection,
            embeddings,
            bm25_manager: DocumentBM25Manager::new(bm25_base_path),
            chunker: DocumentChunker::new(),
        }
    }

    /// Create a new document store with custom chunker config
    pub fn with_chunker_config(
        connection: Arc<Connection>,
        embeddings: Arc<dyn EmbeddingProvider>,
        bm25_base_path: impl Into<std::path::PathBuf>,
        chunker_config: ChunkerConfig,
    ) -> Self {
        Self {
            metadata_store: DocumentMetadataStore::new(Arc::clone(&connection)),
            connection,
            embeddings,
            bm25_manager: DocumentBM25Manager::new(bm25_base_path),
            chunker: DocumentChunker::with_config(chunker_config),
        }
    }

    /// Ensure the required tables exist in the database
    pub async fn ensure_tables(&self) -> Result<()> {
        let dim = self.embeddings.dimension();
        lance_tables::ensure_documents_table(&self.connection, dim).await?;
        lance_tables::ensure_document_metadata_table(&self.connection).await?;
        Ok(())
    }

    /// Index a document from a file path
    pub async fn index_file(
        &self,
        file_path: &Path,
        scope: DocumentScope,
    ) -> Result<DocumentMetadata> {
        // Read and extract text
        let bytes = std::fs::read(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let file_type = DocumentType::from_path(file_path);

        self.index_bytes(&bytes, &file_name, file_type, scope).await
    }

    /// Index a document from bytes
    pub async fn index_bytes(
        &self,
        bytes: &[u8],
        file_name: &str,
        file_type: DocumentType,
        scope: DocumentScope,
    ) -> Result<DocumentMetadata> {
        // Check for duplicate by hash
        let file_hash = DocumentProcessor::compute_hash(bytes);
        if let Some(existing) = self.metadata_store.get_by_hash(&file_hash).await? {
            // Document already indexed
            return Ok(existing);
        }

        // Extract text content
        let extracted = DocumentProcessor::extract_from_bytes(bytes, file_type)?;

        if extracted.is_empty() {
            anyhow::bail!("Extracted document is empty");
        }

        // Generate document ID
        let document_id = uuid::Uuid::new_v4().to_string();

        // Create metadata
        let mut metadata = DocumentMetadata::new(
            document_id.clone(),
            file_name.to_string(),
            file_type,
            bytes.len() as u64,
            file_hash,
        );

        if let Some(title) = extracted.title {
            metadata = metadata.with_title(title);
        }

        if let Some(page_count) = extracted.page_count {
            metadata = metadata.with_page_count(page_count as u32);
        }

        // Apply scope
        let scope_id = match &scope {
            DocumentScope::Conversation(id) => {
                metadata = metadata.with_conversation(id.clone());
                id.clone()
            }
            DocumentScope::Project(id) => {
                metadata = metadata.with_project(id.clone());
                id.clone()
            }
            DocumentScope::Global => "global".to_string(),
        };

        // Chunk the document
        let chunks = self.chunker.chunk(&document_id, &extracted.content);

        if chunks.is_empty() {
            anyhow::bail!("Document produced no chunks");
        }

        metadata = metadata.with_chunk_count(chunks.len() as u32);

        // Index chunks in LanceDB with embeddings
        self.index_chunks_to_lance(&chunks, &metadata, &scope)
            .await?;

        // Index chunks in BM25
        let bm25_chunks: Vec<(String, String)> = chunks
            .iter()
            .map(|c| (c.chunk_id.clone(), c.content.clone()))
            .collect();
        self.bm25_manager.index_chunks(&scope_id, bm25_chunks)?;

        // Save metadata
        self.metadata_store.save(&metadata).await?;

        Ok(metadata)
    }

    /// Index chunks to LanceDB
    async fn index_chunks_to_lance(
        &self,
        chunks: &[DocumentChunk],
        metadata: &DocumentMetadata,
        _scope: &DocumentScope,
    ) -> Result<()> {
        let table = lance_tables::open_documents_table(&self.connection).await?;
        let dimension = self.embeddings.dimension();
        let schema = lance_tables::documents_schema(dimension);

        // Generate embeddings for all chunks
        let contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&contents)?;

        let now = chrono::Utc::now().timestamp();

        // Build arrays for the batch
        let mut all_embeddings: Vec<f32> = Vec::with_capacity(chunks.len() * dimension);
        let mut chunk_ids: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut document_ids: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut conversation_ids: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut project_ids: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut file_names: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut file_types: Vec<String> = Vec::with_capacity(chunks.len());
        let mut contents_arr: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut start_offsets: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut end_offsets: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut chunk_indices: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut total_chunks_arr: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut sections: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut page_numbers: Vec<u32> = Vec::with_capacity(chunks.len());
        let mut file_hashes: Vec<&str> = Vec::with_capacity(chunks.len());
        let mut indexed_ats: Vec<i64> = Vec::with_capacity(chunks.len());

        let conv_id = metadata.conversation_id.as_deref().unwrap_or("");
        let proj_id = metadata.project_id.as_deref().unwrap_or("");
        let file_type_str = format!("{:?}", metadata.file_type);

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            all_embeddings.extend(embedding);
            chunk_ids.push(&chunk.chunk_id);
            document_ids.push(&chunk.document_id);
            conversation_ids.push(conv_id);
            project_ids.push(proj_id);
            file_names.push(&metadata.file_name);
            file_types.push(file_type_str.clone());
            contents_arr.push(&chunk.content);
            start_offsets.push(chunk.start_offset as u32);
            end_offsets.push(chunk.end_offset as u32);
            chunk_indices.push(chunk.chunk_index);
            total_chunks_arr.push(chunk.total_chunks);
            sections.push(chunk.section.as_deref().unwrap_or(""));
            page_numbers.push(chunk.page_number.unwrap_or(0));
            file_hashes.push(&metadata.file_hash);
            indexed_ats.push(now);
        }

        // Create embedding array
        let embedding_array = Float32Array::from(all_embeddings);
        let vector_field = Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            true,
        ));
        let vectors = FixedSizeListArray::new(
            vector_field,
            dimension as i32,
            Arc::new(embedding_array),
            None,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(vectors) as ArrayRef,
                Arc::new(StringArray::from(chunk_ids)),
                Arc::new(StringArray::from(document_ids)),
                Arc::new(StringArray::from(conversation_ids)),
                Arc::new(StringArray::from(project_ids)),
                Arc::new(StringArray::from(file_names)),
                Arc::new(StringArray::from(file_types)),
                Arc::new(StringArray::from(contents_arr)),
                Arc::new(UInt32Array::from(start_offsets)),
                Arc::new(UInt32Array::from(end_offsets)),
                Arc::new(UInt32Array::from(chunk_indices)),
                Arc::new(UInt32Array::from(total_chunks_arr)),
                Arc::new(StringArray::from(sections)),
                Arc::new(UInt32Array::from(page_numbers)),
                Arc::new(StringArray::from(file_hashes)),
                Arc::new(Int64Array::from(indexed_ats)),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to add document chunks to LanceDB")?;

        Ok(())
    }

    /// Search documents with hybrid search (vector + BM25)
    pub async fn search(
        &self,
        request: DocumentSearchRequest,
    ) -> Result<Vec<DocumentSearchResult>> {
        let scope_id = request
            .conversation_id
            .clone()
            .or(request.project_id.clone())
            .unwrap_or_else(|| "global".to_string());

        if request.hybrid {
            self.hybrid_search(&request, &scope_id).await
        } else {
            self.vector_search(&request).await
        }
    }

    /// Perform vector-only search
    async fn vector_search(
        &self,
        request: &DocumentSearchRequest,
    ) -> Result<Vec<DocumentSearchResult>> {
        let embedding = self.embeddings.embed(&request.query)?;
        let table = lance_tables::open_documents_table(&self.connection).await?;

        // Build filter
        let filter = self.build_filter(request);

        let mut query = table
            .vector_search(embedding)
            .context("Failed to create vector search")?;
        query = query.limit(request.limit);

        if let Some(filter) = filter {
            query = query.only_if(filter);
        }

        let stream = query
            .execute()
            .await
            .context("Failed to execute vector search")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                // Get distance (lower is better)
                let distance = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?
                    .value(i);

                // Convert distance to similarity (0-1)
                let similarity = 1.0 / (1.0 + distance);

                if similarity >= request.min_score {
                    let result = self.batch_to_search_result(batch, i, similarity)?;
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Perform hybrid search with RRF fusion
    async fn hybrid_search(
        &self,
        request: &DocumentSearchRequest,
        scope_id: &str,
    ) -> Result<Vec<DocumentSearchResult>> {
        // BM25 pre-fetch uses a large multiplier so that rare terms (e.g.
        // proper names) return all matching chunks before RRF fusion.  A 10×
        // multiplier with a 50-result floor means a default limit=10 request
        // still retrieves up to 100 BM25 candidates before ranking.
        let bm25_prefetch = (request.limit * 10).max(50);

        // Run vector and BM25 searches in parallel
        let vector_future = self.vector_search(request);
        let bm25_results = self
            .bm25_manager
            .search(scope_id, &request.query, bm25_prefetch)?;

        let vector_results = vector_future.await?;

        // Convert vector results to (chunk_id, score) for RRF
        let vector_for_rrf: Vec<(String, f32)> = vector_results
            .iter()
            .map(|r| (r.chunk_id.clone(), r.vector_score))
            .collect();

        // Fuse with a wider internal limit so that BM25-only hits (which score
        // ~half of vector+BM25 hits in RRF due to missing the vector contribution)
        // are not squeezed below the cutoff.  The final sort + truncate enforces
        // the caller's requested limit.
        let rrf_internal_limit = (request.limit * 2).max(20);
        let fused = document_rrf_fusion(vector_for_rrf, bm25_results, rrf_internal_limit);

        // Build final results with combined scores
        let mut results = Vec::new();
        let chunk_id_to_result: HashMap<String, DocumentSearchResult> = vector_results
            .into_iter()
            .map(|r| (r.chunk_id.clone(), r))
            .collect();

        for (chunk_id, combined_score) in fused {
            if let Some(mut result) = chunk_id_to_result.get(&chunk_id).cloned() {
                result.score = combined_score;
                results.push(result);
            } else {
                // Result came from BM25 only - need to fetch from LanceDB
                if let Ok(Some(result)) = self.get_chunk_by_id(&chunk_id).await {
                    let doc_id = result.document_id.clone();
                    let mut search_result = DocumentSearchResult {
                        chunk_id: result.chunk_id,
                        document_id: result.document_id,
                        file_name: String::new(),
                        content: result.content,
                        score: combined_score,
                        vector_score: 0.0,
                        keyword_score: Some(1.0),
                        chunk_index: result.chunk_index,
                        total_chunks: result.total_chunks,
                        section: result.section,
                        page_number: result.page_number,
                    };

                    if let Ok(Some(meta)) = self.metadata_store.get(&doc_id).await {
                        search_result.file_name = meta.file_name;
                    }

                    results.push(search_result);
                }
            }
        }

        // Sort by combined score and enforce the caller's limit
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(request.limit);

        Ok(results)
    }

    /// Get a chunk by ID from LanceDB
    async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<DocumentChunk>> {
        let table = lance_tables::open_documents_table(&self.connection).await?;

        let filter = format!("chunk_id = '{}'", chunk_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query chunk by ID")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        if batches.is_empty() {
            return Ok(None);
        }

        let batch = &batches[0];
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        let chunk_id = self.get_string_value(batch, "chunk_id", 0)?;
        let document_id = self.get_string_value(batch, "document_id", 0)?;
        let content = self.get_string_value(batch, "content", 0)?;
        let start_offset = self.get_u32_value(batch, "start_offset", 0)? as usize;
        let end_offset = self.get_u32_value(batch, "end_offset", 0)? as usize;
        let chunk_index = self.get_u32_value(batch, "chunk_index", 0)?;
        let total_chunks = self.get_u32_value(batch, "total_chunks", 0)?;

        let section_str = self.get_string_value(batch, "section", 0)?;
        let section = if section_str.is_empty() {
            None
        } else {
            Some(section_str)
        };

        let page_num = self.get_u32_value(batch, "page_number", 0)?;
        let page_number = if page_num == 0 { None } else { Some(page_num) };

        Ok(Some(DocumentChunk {
            chunk_id,
            document_id,
            content,
            start_offset,
            end_offset,
            chunk_index,
            total_chunks,
            page_number,
            section,
        }))
    }

    /// Build filter string for LanceDB query
    fn build_filter(&self, request: &DocumentSearchRequest) -> Option<String> {
        let mut filters = Vec::new();

        if let Some(ref conv_id) = request.conversation_id {
            filters.push(format!("conversation_id = '{}'", conv_id));
        }

        if let Some(ref proj_id) = request.project_id {
            filters.push(format!("project_id = '{}'", proj_id));
        }

        if let Some(ref file_type) = request.file_type {
            filters.push(format!("file_type = '{:?}'", file_type));
        }

        if filters.is_empty() {
            None
        } else {
            Some(filters.join(" AND "))
        }
    }

    /// Convert a batch row to DocumentSearchResult
    fn batch_to_search_result(
        &self,
        batch: &RecordBatch,
        row: usize,
        score: f32,
    ) -> Result<DocumentSearchResult> {
        let chunk_id = self.get_string_value(batch, "chunk_id", row)?;
        let document_id = self.get_string_value(batch, "document_id", row)?;
        let file_name = self.get_string_value(batch, "file_name", row)?;
        let content = self.get_string_value(batch, "content", row)?;
        let chunk_index = self.get_u32_value(batch, "chunk_index", row)?;
        let total_chunks = self.get_u32_value(batch, "total_chunks", row)?;

        let section_str = self.get_string_value(batch, "section", row)?;
        let section = if section_str.is_empty() {
            None
        } else {
            Some(section_str)
        };

        let page_num = self.get_u32_value(batch, "page_number", row)?;
        let page_number = if page_num == 0 { None } else { Some(page_num) };

        Ok(DocumentSearchResult {
            chunk_id,
            document_id,
            file_name,
            content,
            score,
            vector_score: score,
            keyword_score: None,
            chunk_index,
            total_chunks,
            section,
            page_number,
        })
    }

    /// Helper to get string value from batch
    fn get_string_value(&self, batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
        Ok(batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {}", column))?
            .as_any()
            .downcast_ref::<StringArray>()
            .with_context(|| format!("Invalid type for column: {}", column))?
            .value(row)
            .to_string())
    }

    /// Helper to get u32 value from batch
    fn get_u32_value(&self, batch: &RecordBatch, column: &str, row: usize) -> Result<u32> {
        Ok(batch
            .column_by_name(column)
            .with_context(|| format!("Missing column: {}", column))?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .with_context(|| format!("Invalid type for column: {}", column))?
            .value(row))
    }

    /// Delete a document and all its chunks
    pub async fn delete_document(&self, document_id: &str) -> Result<bool> {
        let metadata = match self.metadata_store.get(document_id).await? {
            Some(m) => m,
            None => return Ok(false),
        };

        let scope_id = metadata
            .conversation_id
            .clone()
            .or(metadata.project_id.clone())
            .unwrap_or_else(|| "global".to_string());

        // Delete from LanceDB
        let table = lance_tables::open_documents_table(&self.connection).await?;
        table
            .delete(&format!("document_id = '{}'", document_id))
            .await
            .context("Failed to delete document chunks from LanceDB")?;

        // Delete from BM25
        self.bm25_manager.delete_document(&scope_id, document_id)?;

        // Delete metadata
        self.metadata_store.delete(document_id).await?;

        Ok(true)
    }

    /// List documents for a conversation
    pub async fn list_by_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<DocumentMetadata>> {
        self.metadata_store
            .list_by_conversation(conversation_id)
            .await
    }

    /// List documents for a project
    pub async fn list_by_project(&self, project_id: &str) -> Result<Vec<DocumentMetadata>> {
        self.metadata_store.list_by_project(project_id).await
    }

    /// Get document metadata by ID
    pub async fn get_metadata(&self, document_id: &str) -> Result<Option<DocumentMetadata>> {
        self.metadata_store.get(document_id).await
    }

    /// Get all chunks for a document
    pub async fn get_document_chunks(&self, document_id: &str) -> Result<Vec<DocumentChunk>> {
        let table = lance_tables::open_documents_table(&self.connection).await?;

        let filter = format!("document_id = '{}'", document_id);
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .context("Failed to query document chunks")?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut chunks = Vec::new();
        for batch in &batches {
            for i in 0..batch.num_rows() {
                let chunk_id = self.get_string_value(batch, "chunk_id", i)?;
                let document_id = self.get_string_value(batch, "document_id", i)?;
                let content = self.get_string_value(batch, "content", i)?;
                let start_offset = self.get_u32_value(batch, "start_offset", i)? as usize;
                let end_offset = self.get_u32_value(batch, "end_offset", i)? as usize;
                let chunk_index = self.get_u32_value(batch, "chunk_index", i)?;
                let total_chunks = self.get_u32_value(batch, "total_chunks", i)?;

                let section_str = self.get_string_value(batch, "section", i)?;
                let section = if section_str.is_empty() {
                    None
                } else {
                    Some(section_str)
                };

                let page_num = self.get_u32_value(batch, "page_number", i)?;
                let page_number = if page_num == 0 { None } else { Some(page_num) };

                chunks.push(DocumentChunk {
                    chunk_id,
                    document_id,
                    content,
                    start_offset,
                    end_offset,
                    chunk_index,
                    total_chunks,
                    page_number,
                    section,
                });
            }
        }

        // Sort by chunk index
        chunks.sort_by_key(|c| c.chunk_index);

        Ok(chunks)
    }

    /// Count total documents
    pub async fn count(&self) -> Result<usize> {
        self.metadata_store.count().await
    }
}

/// Scope for document storage
#[derive(Debug, Clone)]
pub enum DocumentScope {
    /// Document belongs to a specific conversation
    Conversation(String),
    /// Document belongs to a specific project
    Project(String),
    /// Document is globally accessible
    Global,
}
