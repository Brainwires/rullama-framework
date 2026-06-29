//! Shared search types used across the RAG, vector DB, and spectral modules.
//!
//! These types live in core because they are needed by both `rullama-storage`
//! (the vector DB layer) and `rullama-knowledge` (the RAG / indexer layer).

use serde::{Deserialize, Serialize};

/// A single search result from vector or hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SearchResult {
    /// File path relative to the indexed root.
    pub file_path: String,
    /// Absolute path to the indexed root directory.
    #[serde(default)]
    pub root_path: Option<String>,
    /// The code chunk content.
    pub content: String,
    /// Combined similarity score (0.0 to 1.0).
    pub score: f32,
    /// Vector similarity score (0.0 to 1.0).
    pub vector_score: f32,
    /// Keyword match score (0.0 to 1.0) — only present in hybrid search.
    pub keyword_score: Option<f32>,
    /// Starting line number in the file.
    pub start_line: usize,
    /// Ending line number in the file.
    pub end_line: usize,
    /// Programming language detected.
    pub language: String,
    /// Optional project name for multi-project support.
    pub project: Option<String>,
    /// Timestamp when the chunk was indexed (Unix epoch seconds).
    /// For git commits this equals the commit date.
    #[serde(default)]
    pub indexed_at: i64,
}

/// Metadata stored with each code chunk in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// File path relative to indexed root.
    pub file_path: String,
    /// Absolute path to the indexed root directory.
    #[serde(default)]
    pub root_path: Option<String>,
    /// Project name (for multi-project support).
    pub project: Option<String>,
    /// Starting line number.
    pub start_line: usize,
    /// Ending line number.
    pub end_line: usize,
    /// Programming language.
    pub language: Option<String>,
    /// File extension.
    pub extension: Option<String>,
    /// SHA256 hash of the file content.
    pub file_hash: String,
    /// Timestamp when indexed.
    pub indexed_at: i64,
}

/// Statistics about the vector database contents.
#[derive(Debug, Clone, Default)]
pub struct DatabaseStats {
    /// Total number of stored points.
    pub total_points: usize,
    /// Total number of vectors.
    pub total_vectors: usize,
    /// Breakdown of indexed chunks by programming language.
    pub language_breakdown: Vec<(String, usize)>,
}
