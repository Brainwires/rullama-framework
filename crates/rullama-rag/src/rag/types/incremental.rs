use serde::{Deserialize, Serialize};

/// Request for incremental update
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IncrementalUpdateRequest {
    /// Path to the codebase directory
    pub path: String,
    /// Optional project name
    #[serde(default)]
    pub project: Option<String>,
    /// Optional glob patterns to include
    #[serde(default)]
    pub include_patterns: Vec<String>,
    /// Optional glob patterns to exclude
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

/// Response from incremental update
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IncrementalUpdateResponse {
    /// Number of files added
    pub files_added: usize,
    /// Number of files updated
    pub files_updated: usize,
    /// Number of files removed
    pub files_removed: usize,
    /// Number of chunks created/updated
    pub chunks_modified: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}
