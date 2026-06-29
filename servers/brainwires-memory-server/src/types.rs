//! Wire types for the Mem0-compatible REST API.
//!
//! The service is backed by `brainwires-knowledge` (LanceDB-backed thought
//! store) with per-`user_id` tenant isolation. Legacy Mem0 request fields that
//! the knowledge layer does not model (`agent_id`, `session_id`) are accepted
//! for wire-compatibility but are not used for storage filtering.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Memory record ─────────────────────────────────────────────────────────────

/// A persisted memory record (Mem0-compatible response shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique memory ID.
    pub id: Uuid,
    /// Memory content (plain text).
    pub memory: String,
    /// Owner user ID (tenant).
    pub user_id: String,
    /// Accepted on input for Mem0 compatibility; not currently persisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Accepted on input for Mem0 compatibility; not currently persisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Arbitrary metadata (currently passthrough-only).
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Category tag (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
}

// ── Request bodies ────────────────────────────────────────────────────────────

/// `POST /v1/memories` — add one or more memories.
#[derive(Debug, Deserialize)]
pub struct AddMemoryRequest {
    /// Message list used to extract memories (Mem0 v1 format: role+content pairs).
    /// If `memory` is provided directly, messages are ignored.
    #[serde(default)]
    pub messages: Vec<MessageItem>,
    /// Direct memory text (brainwires extension — bypasses extraction).
    pub memory: Option<String>,
    /// Owner user ID. Required for tenant isolation.
    pub user_id: String,
    /// Optional agent scope (accepted but not persisted).
    pub agent_id: Option<String>,
    /// Optional session scope (accepted but not persisted).
    pub session_id: Option<String>,
    /// Arbitrary metadata to attach (currently unused).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A role+content pair (mirrors OpenAI chat message format).
#[derive(Debug, Deserialize)]
pub struct MessageItem {
    /// `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// Message content.
    pub content: String,
}

/// Response to `POST /v1/memories`.
#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    /// Created memories (one entry per extracted memory).
    pub results: Vec<MemoryResult>,
}

/// Single item in an add-memories response.
#[derive(Debug, Serialize)]
pub struct MemoryResult {
    /// The stored memory text.
    pub memory: String,
    /// `"add"`, `"update"`, or `"delete"`.
    pub event: String,
    /// Assigned memory ID.
    pub id: Uuid,
}

/// `GET /v1/memories` query parameters.
#[derive(Debug, Deserialize)]
pub struct ListMemoriesQuery {
    /// Filter by user ID (required for tenant isolation).
    pub user_id: Option<String>,
    /// Filter by agent ID (accepted but not applied).
    pub agent_id: Option<String>,
    /// Filter by session ID (accepted but not applied).
    pub session_id: Option<String>,
    /// Pagination: page number (1-based, default 1).
    #[serde(default = "default_page")]
    pub page: u32,
    /// Pagination: page size (default 50).
    #[serde(default = "default_page_size")]
    pub page_size: u32,
}

fn default_page() -> u32 {
    1
}
fn default_page_size() -> u32 {
    50
}

/// Response to `GET /v1/memories`.
#[derive(Debug, Serialize)]
pub struct ListMemoriesResponse {
    /// Returned memories.
    pub results: Vec<Memory>,
    /// Total count (matches the filtered list size before pagination).
    pub total: u64,
    /// Page returned.
    pub page: u32,
    /// Page size used.
    pub page_size: u32,
}

/// `POST /v1/memories/search` request body.
#[derive(Debug, Deserialize)]
pub struct SearchMemoriesRequest {
    /// Search query (vector similarity search via the knowledge backend).
    pub query: String,
    /// Filter by user ID (required for tenant isolation).
    pub user_id: Option<String>,
    /// Filter by agent ID (accepted but not applied).
    pub agent_id: Option<String>,
    /// Maximum results to return (default 10).
    #[serde(default = "default_search_limit")]
    pub limit: u32,
}

fn default_search_limit() -> u32 {
    10
}

/// Response to `POST /v1/memories/search`.
#[derive(Debug, Serialize)]
pub struct SearchMemoriesResponse {
    /// Matched memories, ordered by relevance.
    pub results: Vec<SearchResult>,
}

/// A single search hit.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    /// The matching memory.
    #[serde(flatten)]
    pub memory: Memory,
    /// Relevance score (0–1).
    pub score: f64,
}

/// `PATCH /v1/memories/{id}` request body.
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    /// New memory content.
    pub memory: String,
    /// Owner user ID (required for tenant isolation).
    pub user_id: String,
}

/// Generic success/message response.
#[derive(Debug, Serialize)]
pub struct MessageResponse {
    /// Human-readable message.
    pub message: String,
}
