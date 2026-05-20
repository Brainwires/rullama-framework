use serde::{Deserialize, Serialize};

/// Request to get statistics about the index
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StatisticsRequest {}

/// Statistics about the indexed codebase
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StatisticsResponse {
    /// Total number of indexed files
    pub total_files: usize,
    /// Total number of code chunks
    pub total_chunks: usize,
    /// Total number of embeddings
    pub total_embeddings: usize,
    /// Size of the vector database in bytes
    pub database_size_bytes: u64,
    /// Breakdown by programming language
    pub language_breakdown: Vec<LanguageStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
/// Statistics for a single programming language in the index.
pub struct LanguageStats {
    /// Language name.
    pub language: String,
    /// Number of indexed files for this language.
    pub file_count: usize,
    /// Number of code chunks for this language.
    pub chunk_count: usize,
}

/// Request to clear the index
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClearRequest {}

/// Response from clear operation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClearResponse {
    /// Whether the operation was successful
    pub success: bool,
    /// Optional message
    pub message: String,
}
