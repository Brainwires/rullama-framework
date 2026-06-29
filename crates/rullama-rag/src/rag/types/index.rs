use serde::{Deserialize, Serialize};

/// Maximum allowed length for project names.
pub(super) const PROJECT_NAME_MAX_LENGTH: usize = 256;

/// Request to index a codebase
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IndexRequest {
    /// Path to the codebase directory to index
    pub path: String,
    /// Optional project name (for multi-project support)
    #[serde(default)]
    pub project: Option<String>,
    /// Optional glob patterns to include (e.g., ["**/*.rs", "**/*.toml"])
    #[serde(default)]
    pub include_patterns: Vec<String>,
    /// Optional glob patterns to exclude (e.g., ["**/target/**", "**/node_modules/**"])
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
    /// Maximum file size in bytes to index (default: 1MB)
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,
}

/// Default maximum file size for indexing (1 MB).
pub fn default_max_file_size() -> usize {
    1_048_576 // 1MB
}

/// Indexing mode used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum IndexingMode {
    /// Full indexing (all files)
    Full,
    /// Incremental update (only changed files)
    Incremental,
}

/// Response from indexing operation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IndexResponse {
    /// Indexing mode used (full or incremental)
    pub mode: IndexingMode,
    /// Number of files successfully indexed
    pub files_indexed: usize,
    /// Number of code chunks created
    pub chunks_created: usize,
    /// Number of embeddings generated
    pub embeddings_generated: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Any errors encountered (non-fatal)
    #[serde(default)]
    pub errors: Vec<String>,
    /// Number of files updated (incremental mode only)
    #[serde(default)]
    pub files_updated: usize,
    /// Number of files removed (incremental mode only)
    #[serde(default)]
    pub files_removed: usize,
}

/// Input validation for IndexRequest
impl IndexRequest {
    /// Validate the index request
    pub fn validate(&self) -> Result<(), String> {
        // Validate path exists and is a directory
        let path = std::path::Path::new(&self.path);
        if !path.exists() {
            return Err(format!("Path does not exist: {}", self.path));
        }
        if !path.is_dir() {
            return Err(format!("Path is not a directory: {}", self.path));
        }

        // Canonicalize to prevent path traversal attacks
        let _canonical = path
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize path: {}", e))?;

        // Validate max_file_size is reasonable (max 100MB)
        const MAX_FILE_SIZE_LIMIT: usize = 100_000_000; // 100MB
        if self.max_file_size > MAX_FILE_SIZE_LIMIT {
            return Err(format!(
                "max_file_size too large: {} bytes (max: {} bytes)",
                self.max_file_size, MAX_FILE_SIZE_LIMIT
            ));
        }

        // Validate project name if provided
        if let Some(ref project) = self.project {
            if project.is_empty() {
                return Err("project name cannot be empty".to_string());
            }
            if project.len() > PROJECT_NAME_MAX_LENGTH {
                return Err(format!(
                    "project name too long (max {} characters)",
                    PROJECT_NAME_MAX_LENGTH
                ));
            }
        }

        Ok(())
    }
}
