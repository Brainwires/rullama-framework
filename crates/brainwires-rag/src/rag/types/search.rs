use serde::{Deserialize, Serialize};

use super::index::PROJECT_NAME_MAX_LENGTH;
use super::query::{QueryRequest, default_limit, default_min_score};

/// Request to search with file type filters
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AdvancedSearchRequest {
    /// The search query
    pub query: String,
    /// Optional path to filter by specific indexed codebase
    #[serde(default)]
    pub path: Option<String>,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Number of results to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Filter by file extensions (e.g., ["rs", "toml"])
    #[serde(default)]
    pub file_extensions: Vec<String>,
    /// Filter by programming languages
    #[serde(default)]
    pub languages: Vec<String>,
    /// Filter by file path patterns (glob)
    #[serde(default)]
    pub path_patterns: Vec<String>,
}

impl AdvancedSearchRequest {
    /// Validate the advanced search request
    pub fn validate(&self) -> Result<(), String> {
        // Reuse QueryRequest validation logic
        let query_req = QueryRequest {
            query: self.query.clone(),
            path: None,
            project: self.project.clone(),
            limit: self.limit,
            min_score: self.min_score,
            hybrid: true,
        };
        query_req.validate()?;

        // Additional validation for file extensions
        for ext in &self.file_extensions {
            if ext.is_empty() {
                return Err("file extension cannot be empty".to_string());
            }
            if ext.len() > 20 {
                return Err(format!(
                    "file extension too long: {} (max 20 characters)",
                    ext
                ));
            }
        }

        // Validate languages
        for lang in &self.languages {
            if lang.is_empty() {
                return Err("language name cannot be empty".to_string());
            }
            if lang.len() > 50 {
                return Err(format!(
                    "language name too long: {} (max 50 characters)",
                    lang
                ));
            }
        }

        Ok(())
    }
}

/// Default git path for search.
pub fn default_git_path() -> String {
    ".".to_string()
}

/// Default maximum number of commits to search.
pub fn default_max_commits() -> usize {
    10
}

/// Request to search git history
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchGitHistoryRequest {
    /// The search query
    pub query: String,
    /// Path to the codebase (will discover git repo)
    #[serde(default = "default_git_path")]
    pub path: String,
    /// Optional project name
    #[serde(default)]
    pub project: Option<String>,
    /// Optional branch name (default: current branch)
    #[serde(default)]
    pub branch: Option<String>,
    /// Maximum number of commits to index/search (default: 10)
    #[serde(default = "default_max_commits")]
    pub max_commits: usize,
    /// Number of results to return (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score (0.0 to 1.0, default: 0.7)
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Filter by commit author (optional regex pattern)
    #[serde(default)]
    pub author: Option<String>,
    /// Filter by commits since this date (ISO 8601 or Unix timestamp)
    #[serde(default)]
    pub since: Option<String>,
    /// Filter by commits until this date (ISO 8601 or Unix timestamp)
    #[serde(default)]
    pub until: Option<String>,
    /// Filter by file path pattern (optional regex)
    #[serde(default)]
    pub file_pattern: Option<String>,
}

impl SearchGitHistoryRequest {
    /// Validate the git history search request
    pub fn validate(&self) -> Result<(), String> {
        // Validate query
        if self.query.trim().is_empty() {
            return Err("query cannot be empty".to_string());
        }

        const MAX_QUERY_LENGTH: usize = 10_240; // 10KB
        if self.query.len() > MAX_QUERY_LENGTH {
            return Err(format!(
                "query too long: {} bytes (max: {} bytes)",
                self.query.len(),
                MAX_QUERY_LENGTH
            ));
        }

        // Validate path
        let path = std::path::Path::new(&self.path);
        if !path.exists() {
            return Err(format!("Path does not exist: {}", self.path));
        }

        // Validate min_score range
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(format!(
                "min_score must be between 0.0 and 1.0, got: {}",
                self.min_score
            ));
        }

        // Validate limit
        const MAX_LIMIT: usize = 1000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
            ));
        }

        // Validate max_commits
        const MAX_COMMITS_LIMIT: usize = 10000;
        if self.max_commits > MAX_COMMITS_LIMIT {
            return Err(format!(
                "max_commits too large: {} (max: {})",
                self.max_commits, MAX_COMMITS_LIMIT
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

/// A single git search result
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GitSearchResult {
    /// Git commit hash (SHA)
    pub commit_hash: String,
    /// Commit message
    pub commit_message: String,
    /// Author name
    pub author: String,
    /// Author email
    pub author_email: String,
    /// Commit date (Unix timestamp)
    pub commit_date: i64,
    /// Combined similarity score (0.0 to 1.0)
    pub score: f32,
    /// Vector similarity score
    pub vector_score: f32,
    /// Keyword match score (if hybrid search enabled)
    pub keyword_score: Option<f32>,
    /// Files changed in this commit
    pub files_changed: Vec<String>,
    /// Diff snippet (first ~500 characters)
    pub diff_snippet: String,
}

/// Response from git history search
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchGitHistoryResponse {
    /// List of matching commits, ordered by relevance
    pub results: Vec<GitSearchResult>,
    /// Number of commits indexed during this search
    pub commits_indexed: usize,
    /// Total commits in cache for this repo
    pub total_cached_commits: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}
