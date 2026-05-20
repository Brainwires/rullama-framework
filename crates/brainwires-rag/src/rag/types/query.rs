use serde::{Deserialize, Serialize};

// Re-export shared types from core
pub use brainwires_core::SearchResult;

use super::index::PROJECT_NAME_MAX_LENGTH;

/// Default value for hybrid search (enabled).
pub fn default_hybrid() -> bool {
    true
}

/// Default result limit.
pub fn default_limit() -> usize {
    10
}

/// Default minimum similarity score.
pub fn default_min_score() -> f32 {
    0.7
}

/// Request to query the codebase
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct QueryRequest {
    /// The question or search query
    pub query: String,
    /// Optional path to filter by specific indexed codebase
    #[serde(default)]
    pub path: Option<String>,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Number of results to return (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score (0.0 to 1.0, default: 0.7)
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Enable hybrid search (vector + keyword) - default: true
    #[serde(default = "default_hybrid")]
    pub hybrid: bool,
}

/// Response from query operation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct QueryResponse {
    /// List of search results, ordered by relevance
    pub results: Vec<SearchResult>,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// The actual threshold used (may be lower than requested if adaptive search kicked in)
    #[serde(default)]
    pub threshold_used: f32,
    /// Whether the threshold was automatically lowered to find results
    #[serde(default)]
    pub threshold_lowered: bool,
}

impl QueryRequest {
    /// Validate the query request
    pub fn validate(&self) -> Result<(), String> {
        // Validate query is not empty
        if self.query.trim().is_empty() {
            return Err("query cannot be empty".to_string());
        }

        // Validate query length is reasonable (max 10KB)
        const MAX_QUERY_LENGTH: usize = 10_240; // 10KB
        if self.query.len() > MAX_QUERY_LENGTH {
            return Err(format!(
                "query too long: {} bytes (max: {} bytes)",
                self.query.len(),
                MAX_QUERY_LENGTH
            ));
        }

        // Validate min_score is in valid range [0.0, 1.0]
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(format!(
                "min_score must be between 0.0 and 1.0, got: {}",
                self.min_score
            ));
        }

        // Validate limit is reasonable (max 1000)
        const MAX_LIMIT: usize = 1000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
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
