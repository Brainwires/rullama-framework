use serde::{Deserialize, Serialize};

use super::query::{SearchResult, default_limit, default_min_score};

/// Search strategies available for the ensemble query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    /// Semantic (vector) search — finds conceptually similar code.
    Semantic,
    /// Keyword (BM25) search — finds exact term matches.
    Keyword,
    /// Git history search — finds matching commits, messages, and diffs.
    GitHistory,
    /// Code navigation search — finds definitions/references via AST relations.
    /// Only available when the `code-analysis` feature is enabled.
    CodeNavigation,
}

/// Request for the parallel multi-strategy ensemble query.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EnsembleRequest {
    /// The search query.
    pub query: String,
    /// Optional path to filter by specific indexed codebase.
    #[serde(default)]
    pub path: Option<String>,
    /// Optional project name to filter results.
    #[serde(default)]
    pub project: Option<String>,
    /// Maximum results to return after fusion.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score per strategy (default: 0.7).
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Strategies to fan out across.  An empty list means "all available".
    #[serde(default)]
    pub strategies: Vec<SearchStrategy>,
    /// File extensions to restrict results to (e.g., `["rs", "toml"]`).
    #[serde(default)]
    pub file_extensions: Vec<String>,
    /// Programming languages to restrict results to.
    #[serde(default)]
    pub languages: Vec<String>,
    /// If `true` and the `spectral` feature is enabled, apply spectral
    /// diversity reranking as a final pass on the fused result set.
    #[serde(default)]
    pub spectral_rerank: bool,
}

/// Response from the ensemble multi-strategy query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResponse {
    /// Merged, deduplicated, and RRF-fused results.
    pub results: Vec<SearchResult>,
    /// Total wall-clock time in milliseconds.
    pub duration_ms: u64,
    /// Names of the strategies that ran (and didn't error).
    pub strategies_used: Vec<String>,
    /// Number of raw results returned by each strategy before fusion.
    pub per_strategy_counts: std::collections::HashMap<String, usize>,
}
