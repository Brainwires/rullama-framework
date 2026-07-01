//! Document storage types for large document support
//!
//! Provides core types for document metadata, chunks, and search operations.
//! Documents are stored separately from messages with their own chunking
//! strategy optimized for natural language (paragraph/sentence-based)
//! rather than code (AST-based).

use serde::{Deserialize, Serialize};
use std::path::Path;

// Default chunk configuration
const DEFAULT_TARGET_CHUNK_SIZE: usize = 1500;
const DEFAULT_MAX_CHUNK_SIZE: usize = 2500;
const DEFAULT_MIN_CHUNK_SIZE: usize = 100;
const DEFAULT_OVERLAP_SIZE: usize = 200;

// Small document chunk configuration
const SMALL_TARGET_CHUNK_SIZE: usize = 800;
const SMALL_MAX_CHUNK_SIZE: usize = 1200;
const SMALL_MIN_CHUNK_SIZE: usize = 50;
const SMALL_OVERLAP_SIZE: usize = 100;

// Large document chunk configuration
const LARGE_TARGET_CHUNK_SIZE: usize = 2000;
const LARGE_MAX_CHUNK_SIZE: usize = 3500;
const LARGE_MIN_CHUNK_SIZE: usize = 200;
const LARGE_OVERLAP_SIZE: usize = 300;

/// Document type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentType {
    /// PDF documents
    Pdf,
    /// Markdown files
    Markdown,
    /// Plain text files
    PlainText,
    /// Microsoft Word documents (.docx)
    Docx,
    /// Unknown or unsupported format
    Unknown,
}

impl DocumentType {
    /// Detect document type from file path
    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Detect document type from extension string
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "pdf" => Self::Pdf,
            "md" | "markdown" => Self::Markdown,
            "txt" | "text" => Self::PlainText,
            "docx" => Self::Docx,
            _ => Self::Unknown,
        }
    }

    /// Get file extension for this document type
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Markdown => "md",
            Self::PlainText => "txt",
            Self::Docx => "docx",
            Self::Unknown => "",
        }
    }

    /// Get MIME type for this document type
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Pdf => "application/pdf",
            Self::Markdown => "text/markdown",
            Self::PlainText => "text/plain",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Unknown => "application/octet-stream",
        }
    }

    /// Check if this document type is supported for text extraction
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf => write!(f, "PDF"),
            Self::Markdown => write!(f, "Markdown"),
            Self::PlainText => write!(f, "Plain Text"),
            Self::Docx => write!(f, "DOCX"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Metadata for a stored document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Unique document identifier
    pub document_id: String,
    /// Optional conversation binding
    pub conversation_id: Option<String>,
    /// Optional project binding
    pub project_id: Option<String>,
    /// Original file name
    pub file_name: String,
    /// Detected document type
    pub file_type: DocumentType,
    /// File size in bytes
    pub file_size_bytes: u64,
    /// Number of chunks created
    pub chunk_count: u32,
    /// SHA256 hash of file content
    pub file_hash: String,
    /// Unix timestamp when indexed
    pub created_at: i64,
    /// Optional page count (for PDFs)
    pub page_count: Option<u32>,
    /// Optional title extracted from document
    pub title: Option<String>,
}

impl DocumentMetadata {
    /// Create a new document metadata instance
    pub fn new(
        document_id: String,
        file_name: String,
        file_type: DocumentType,
        file_size_bytes: u64,
        file_hash: String,
    ) -> Self {
        Self {
            document_id,
            conversation_id: None,
            project_id: None,
            file_name,
            file_type,
            file_size_bytes,
            chunk_count: 0,
            file_hash,
            created_at: chrono::Utc::now().timestamp(),
            page_count: None,
            title: None,
        }
    }

    /// Set the conversation binding
    pub fn with_conversation(mut self, conversation_id: String) -> Self {
        self.conversation_id = Some(conversation_id);
        self
    }

    /// Set the project binding
    pub fn with_project(mut self, project_id: String) -> Self {
        self.project_id = Some(project_id);
        self
    }

    /// Set the chunk count
    pub fn with_chunk_count(mut self, count: u32) -> Self {
        self.chunk_count = count;
        self
    }

    /// Set page count (for PDFs)
    pub fn with_page_count(mut self, count: u32) -> Self {
        self.page_count = Some(count);
        self
    }

    /// Set document title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }
}

/// A chunk of document content with position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique chunk identifier (document_id:chunk_index)
    pub chunk_id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk text content
    pub content: String,
    /// Start byte offset in original document
    pub start_offset: usize,
    /// End byte offset in original document
    pub end_offset: usize,
    /// Zero-based chunk index
    pub chunk_index: u32,
    /// Total chunks in document
    pub total_chunks: u32,
    /// Optional page number (for PDFs)
    pub page_number: Option<u32>,
    /// Optional section header this chunk belongs to
    pub section: Option<String>,
}

impl DocumentChunk {
    /// Create a new document chunk
    pub fn new(
        document_id: String,
        content: String,
        start_offset: usize,
        end_offset: usize,
        chunk_index: u32,
        total_chunks: u32,
    ) -> Self {
        let chunk_id = format!("{}:{}", document_id, chunk_index);
        Self {
            chunk_id,
            document_id,
            content,
            start_offset,
            end_offset,
            chunk_index,
            total_chunks,
            page_number: None,
            section: None,
        }
    }

    /// Set page number
    pub fn with_page(mut self, page: u32) -> Self {
        self.page_number = Some(page);
        self
    }

    /// Set section header
    pub fn with_section(mut self, section: String) -> Self {
        self.section = Some(section);
        self
    }

    /// Get the length of this chunk in bytes
    pub fn len(&self) -> usize {
        self.end_offset - self.start_offset
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Configuration for document chunking
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Target chunk size in characters
    pub target_chunk_size: usize,
    /// Maximum chunk size (hard limit)
    pub max_chunk_size: usize,
    /// Minimum chunk size (avoid tiny chunks)
    pub min_chunk_size: usize,
    /// Overlap between chunks for context continuity
    pub overlap_size: usize,
    /// Whether to respect markdown headers as chunk boundaries
    pub respect_headers: bool,
    /// Whether to respect paragraph boundaries
    pub respect_paragraphs: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            target_chunk_size: DEFAULT_TARGET_CHUNK_SIZE,
            max_chunk_size: DEFAULT_MAX_CHUNK_SIZE,
            min_chunk_size: DEFAULT_MIN_CHUNK_SIZE,
            overlap_size: DEFAULT_OVERLAP_SIZE,
            respect_headers: true,
            respect_paragraphs: true,
        }
    }
}

impl ChunkerConfig {
    /// Create config for small documents
    pub fn small() -> Self {
        Self {
            target_chunk_size: SMALL_TARGET_CHUNK_SIZE,
            max_chunk_size: SMALL_MAX_CHUNK_SIZE,
            min_chunk_size: SMALL_MIN_CHUNK_SIZE,
            overlap_size: SMALL_OVERLAP_SIZE,
            ..Default::default()
        }
    }

    /// Create config for large documents
    pub fn large() -> Self {
        Self {
            target_chunk_size: LARGE_TARGET_CHUNK_SIZE,
            max_chunk_size: LARGE_MAX_CHUNK_SIZE,
            min_chunk_size: LARGE_MIN_CHUNK_SIZE,
            overlap_size: LARGE_OVERLAP_SIZE,
            ..Default::default()
        }
    }
}

/// Request for document search
#[derive(Debug, Clone)]
pub struct DocumentSearchRequest {
    /// Search query text
    pub query: String,
    /// Optional conversation filter
    pub conversation_id: Option<String>,
    /// Optional project filter
    pub project_id: Option<String>,
    /// Maximum results to return
    pub limit: usize,
    /// Minimum similarity score (0.0-1.0)
    pub min_score: f32,
    /// Enable hybrid search (vector + BM25)
    pub hybrid: bool,
    /// Optional document type filter
    pub file_type: Option<DocumentType>,
}

impl DocumentSearchRequest {
    /// Create a new search request with default settings
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            conversation_id: None,
            project_id: None,
            limit: 10,
            min_score: 0.5,
            hybrid: true,
            file_type: None,
        }
    }

    /// Filter by conversation
    pub fn with_conversation(mut self, conversation_id: String) -> Self {
        self.conversation_id = Some(conversation_id);
        self
    }

    /// Filter by project
    pub fn with_project(mut self, project_id: String) -> Self {
        self.project_id = Some(project_id);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set minimum score
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Enable or disable hybrid search
    pub fn with_hybrid(mut self, hybrid: bool) -> Self {
        self.hybrid = hybrid;
        self
    }

    /// Filter by document type
    pub fn with_file_type(mut self, file_type: DocumentType) -> Self {
        self.file_type = Some(file_type);
        self
    }
}

/// A single search result
#[derive(Debug, Clone)]
pub struct DocumentSearchResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Parent document ID
    pub document_id: String,
    /// Original file name
    pub file_name: String,
    /// Chunk content
    pub content: String,
    /// Combined RRF score (for hybrid) or vector score
    pub score: f32,
    /// Pure vector similarity score
    pub vector_score: f32,
    /// BM25 keyword score (if hybrid search)
    pub keyword_score: Option<f32>,
    /// Chunk index in document
    pub chunk_index: u32,
    /// Total chunks in document
    pub total_chunks: u32,
    /// Optional section header
    pub section: Option<String>,
    /// Optional page number
    pub page_number: Option<u32>,
}

impl DocumentSearchResult {
    /// Create from chunk with vector score
    pub fn from_chunk(chunk: &DocumentChunk, file_name: String, vector_score: f32) -> Self {
        Self {
            chunk_id: chunk.chunk_id.clone(),
            document_id: chunk.document_id.clone(),
            file_name,
            content: chunk.content.clone(),
            score: vector_score,
            vector_score,
            keyword_score: None,
            chunk_index: chunk.chunk_index,
            total_chunks: chunk.total_chunks,
            section: chunk.section.clone(),
            page_number: chunk.page_number,
        }
    }

    /// Set the combined score (after RRF fusion)
    pub fn with_combined_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Set the keyword score
    pub fn with_keyword_score(mut self, score: f32) -> Self {
        self.keyword_score = Some(score);
        self
    }
}

/// Result of document text extraction
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    /// Extracted text content
    pub content: String,
    /// Detected document type
    pub file_type: DocumentType,
    /// Page count (if applicable)
    pub page_count: Option<usize>,
    /// Extracted title (if found)
    pub title: Option<String>,
    /// Any warnings during extraction
    pub warnings: Vec<String>,
}

impl ExtractedDocument {
    /// Create a new extracted document
    pub fn new(content: String, file_type: DocumentType) -> Self {
        Self {
            content,
            file_type,
            page_count: None,
            title: None,
            warnings: Vec::new(),
        }
    }

    /// Set page count
    pub fn with_page_count(mut self, count: usize) -> Self {
        self.page_count = Some(count);
        self
    }

    /// Set title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Check if content is empty
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }

    /// Get content length in bytes
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_document_type_from_path() {
        assert_eq!(
            DocumentType::from_path(&PathBuf::from("test.pdf")),
            DocumentType::Pdf
        );
        assert_eq!(
            DocumentType::from_path(&PathBuf::from("README.md")),
            DocumentType::Markdown
        );
        assert_eq!(
            DocumentType::from_path(&PathBuf::from("notes.txt")),
            DocumentType::PlainText
        );
        assert_eq!(
            DocumentType::from_path(&PathBuf::from("doc.docx")),
            DocumentType::Docx
        );
        assert_eq!(
            DocumentType::from_path(&PathBuf::from("file.xyz")),
            DocumentType::Unknown
        );
    }

    #[test]
    fn test_document_type_from_extension() {
        assert_eq!(DocumentType::from_extension("PDF"), DocumentType::Pdf);
        assert_eq!(
            DocumentType::from_extension("markdown"),
            DocumentType::Markdown
        );
        assert_eq!(DocumentType::from_extension("TXT"), DocumentType::PlainText);
    }

    #[test]
    fn test_document_type_mime_types() {
        assert_eq!(DocumentType::Pdf.mime_type(), "application/pdf");
        assert_eq!(DocumentType::Markdown.mime_type(), "text/markdown");
        assert_eq!(DocumentType::PlainText.mime_type(), "text/plain");
    }

    #[test]
    fn test_document_type_is_supported() {
        assert!(DocumentType::Pdf.is_supported());
        assert!(DocumentType::Markdown.is_supported());
        assert!(DocumentType::PlainText.is_supported());
        assert!(DocumentType::Docx.is_supported());
        assert!(!DocumentType::Unknown.is_supported());
    }

    #[test]
    fn test_document_metadata_builder() {
        let meta = DocumentMetadata::new(
            "doc-123".to_string(),
            "test.pdf".to_string(),
            DocumentType::Pdf,
            1024,
            "abc123".to_string(),
        )
        .with_conversation("conv-456".to_string())
        .with_project("proj-789".to_string())
        .with_chunk_count(10)
        .with_page_count(5)
        .with_title("Test Document".to_string());

        assert_eq!(meta.document_id, "doc-123");
        assert_eq!(meta.conversation_id, Some("conv-456".to_string()));
        assert_eq!(meta.project_id, Some("proj-789".to_string()));
        assert_eq!(meta.chunk_count, 10);
        assert_eq!(meta.page_count, Some(5));
        assert_eq!(meta.title, Some("Test Document".to_string()));
    }

    #[test]
    fn test_document_chunk_creation() {
        let chunk = DocumentChunk::new(
            "doc-123".to_string(),
            "Hello world".to_string(),
            0,
            11,
            0,
            5,
        );

        assert_eq!(chunk.chunk_id, "doc-123:0");
        assert_eq!(chunk.len(), 11);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_search_request_builder() {
        let request = DocumentSearchRequest::new("test query")
            .with_conversation("conv-123".to_string())
            .with_limit(20)
            .with_min_score(0.7)
            .with_hybrid(false)
            .with_file_type(DocumentType::Pdf);

        assert_eq!(request.query, "test query");
        assert_eq!(request.conversation_id, Some("conv-123".to_string()));
        assert_eq!(request.limit, 20);
        assert_eq!(request.min_score, 0.7);
        assert!(!request.hybrid);
        assert_eq!(request.file_type, Some(DocumentType::Pdf));
    }

    #[test]
    fn test_extracted_document() {
        let doc = ExtractedDocument::new("Hello world".to_string(), DocumentType::PlainText)
            .with_page_count(1)
            .with_title("Test".to_string())
            .with_warning("Some warning".to_string());

        assert_eq!(doc.len(), 11);
        assert!(!doc.is_empty());
        assert_eq!(doc.page_count, Some(1));
        assert_eq!(doc.title, Some("Test".to_string()));
        assert_eq!(doc.warnings.len(), 1);
    }
}
