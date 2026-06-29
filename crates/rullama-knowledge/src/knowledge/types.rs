use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── capture_thought ──────────────────────────────────────────────────────

/// Request to capture a new thought.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CaptureThoughtRequest {
    /// The thought text to capture
    pub content: String,
    /// Category: decision, person, insight, meeting_note, idea, action_item, reference, general.
    /// Auto-detected if omitted.
    #[serde(default)]
    pub category: Option<String>,
    /// User-provided tags
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Importance score 0.0–1.0 (default: 0.5)
    #[serde(default)]
    pub importance: Option<f32>,
    /// Source identifier (default: "manual")
    #[serde(default)]
    pub source: Option<String>,
    /// Optional tenant/owner ID for per-owner scoping. `None` = unscoped.
    #[serde(default)]
    pub owner_id: Option<String>,
}

/// Response after capturing a thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureThoughtResponse {
    /// UUID of the captured thought.
    pub id: String,
    /// Detected or specified category.
    pub category: String,
    /// Auto-extracted and user-provided tags.
    pub tags: Vec<String>,
    /// Importance score.
    pub importance: f32,
    /// Number of facts extracted from the thought.
    pub facts_extracted: usize,
    /// IDs of existing thoughts that corroborate this one.
    pub corroborations: Vec<String>,
    /// IDs of existing thoughts that contradict this one.
    pub contradictions: Vec<String>,
    /// Confidence assigned to this thought after evidence check (0.0–1.0).
    pub confidence: f32,
}

/// Result of checking a thought against existing evidence.
#[derive(Debug, Clone, Default)]
pub struct EvidenceCheckResult {
    /// IDs of thoughts that corroborate the new thought (score ≥ corroboration threshold).
    pub corroborations: Vec<String>,
    /// IDs of thoughts that contradict the new thought (similar but negation-divergent).
    pub contradictions: Vec<String>,
}

// ── search_memory ────────────────────────────────────────────────────────

/// Request to search memory.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryRequest {
    /// Natural language search query
    pub query: String,
    /// Max results (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score (default: 0.6)
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Filter by ThoughtCategory
    #[serde(default)]
    pub category: Option<String>,
    /// Which stores to search: "thoughts", "facts". Default: all.
    #[serde(default)]
    pub sources: Option<Vec<String>>,
    /// Optional tenant/owner ID for per-owner scoping. `None` = unscoped.
    #[serde(default)]
    pub owner_id: Option<String>,
}

/// Response from a memory search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMemoryResponse {
    /// Matching results.
    pub results: Vec<MemorySearchResult>,
    /// Total number of results.
    pub total: usize,
}

/// A single memory search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    /// The matched content text.
    pub content: String,
    /// Similarity score.
    pub score: f32,
    /// Source store (e.g. "thoughts", "facts").
    pub source: String,
    /// Thought UUID if from thoughts store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_id: Option<String>,
    /// Category of the matched item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Tags of the matched item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Unix timestamp of creation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
}

// ── list_recent ──────────────────────────────────────────────────────────

/// Request to list recent thoughts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListRecentRequest {
    /// Max results (default: 20)
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    /// Filter by category
    #[serde(default)]
    pub category: Option<String>,
    /// ISO 8601 timestamp (default: 7 days ago)
    #[serde(default)]
    pub since: Option<String>,
    /// Optional tenant/owner ID for per-owner scoping. `None` = unscoped.
    #[serde(default)]
    pub owner_id: Option<String>,
}

/// Response from listing recent thoughts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRecentResponse {
    /// Recent thought summaries.
    pub thoughts: Vec<ThoughtSummary>,
    /// Total count.
    pub total: usize,
}

/// Summary of a thought for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtSummary {
    /// Thought UUID.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Category name.
    pub category: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Importance score.
    pub importance: f32,
    /// Unix timestamp of creation.
    pub created_at: i64,
}

// ── get_thought ──────────────────────────────────────────────────────────

/// Request to get a single thought by ID.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetThoughtRequest {
    /// Thought UUID
    pub id: String,
    /// Optional tenant/owner ID for per-owner scoping. `None` = unscoped.
    #[serde(default)]
    pub owner_id: Option<String>,
}

/// Response containing a full thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetThoughtResponse {
    /// Thought UUID.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Category name.
    pub category: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Source identifier.
    pub source: String,
    /// Importance score.
    pub importance: f32,
    /// Unix timestamp of creation.
    pub created_at: i64,
    /// Unix timestamp of last update.
    pub updated_at: i64,
}

// ── search_knowledge ─────────────────────────────────────────────────────

/// Request to search the knowledge store (PKS/BKS).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchKnowledgeRequest {
    /// Context to match against
    pub query: String,
    /// "personal" (PKS), "behavioral" (BKS), or "all" (default)
    #[serde(default)]
    pub source: Option<String>,
    /// PKS/BKS category filter
    #[serde(default)]
    pub category: Option<String>,
    /// Minimum confidence (default: 0.5)
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    /// Max results (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Response from a knowledge search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchKnowledgeResponse {
    /// Matching knowledge results.
    pub results: Vec<KnowledgeResult>,
    /// Total count.
    pub total: usize,
}

/// A single knowledge search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    /// Knowledge source ("personal" or "behavioral").
    pub source: String,
    /// Knowledge category.
    pub category: String,
    /// Fact/truth key or pattern.
    pub key: String,
    /// Fact/truth value or rule.
    pub value: String,
    /// Confidence score.
    pub confidence: f32,
    /// Optional additional context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

// ── memory_stats ─────────────────────────────────────────────────────────

// No request params needed — but we still define an empty struct for the macro.
/// Request for memory statistics (no parameters).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryStatsRequest {}

/// Response with memory statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatsResponse {
    /// Thought store statistics.
    pub thoughts: ThoughtStats,
    /// Personal Knowledge Store statistics.
    pub pks: PksStats,
    /// Behavioral Knowledge Store statistics.
    pub bks: BksStats,
}

/// Statistics about stored thoughts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtStats {
    /// Total number of thoughts.
    pub total: usize,
    /// Counts by category.
    pub by_category: std::collections::HashMap<String, usize>,
    /// Thoughts created in the last 24 hours.
    pub recent_24h: usize,
    /// Thoughts created in the last 7 days.
    pub recent_7d: usize,
    /// Thoughts created in the last 30 days.
    pub recent_30d: usize,
    /// Most-used tags with counts.
    pub top_tags: Vec<(String, usize)>,
}

/// Personal Knowledge Store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PksStats {
    /// Total number of personal facts.
    pub total_facts: u32,
    /// Counts by category.
    pub by_category: std::collections::HashMap<String, u32>,
    /// Average confidence score.
    pub avg_confidence: f32,
}

/// Behavioral Knowledge Store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BksStats {
    /// Total number of behavioral truths.
    pub total_truths: u32,
    /// Counts by category.
    pub by_category: std::collections::HashMap<String, u32>,
}

// ── delete_thought ───────────────────────────────────────────────────────

/// Request to delete a thought.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeleteThoughtRequest {
    /// Thought UUID to delete
    pub id: String,
    /// Optional tenant/owner ID for per-owner scoping. `None` = unscoped.
    #[serde(default)]
    pub owner_id: Option<String>,
}

/// Response after deleting a thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteThoughtResponse {
    /// Whether the thought was successfully deleted.
    pub deleted: bool,
    /// UUID of the deleted thought.
    pub id: String,
}

// ── defaults ─────────────────────────────────────────────────────────────

fn default_limit() -> usize {
    10
}

fn default_list_limit() -> usize {
    20
}

fn default_min_score() -> f32 {
    0.6
}

fn default_min_confidence() -> f32 {
    0.5
}
