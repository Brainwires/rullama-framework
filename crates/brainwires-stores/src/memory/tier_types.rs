//! Pure schema types shared by the tier stores.
//!
//! These are referenced by `MessageStore`, `SummaryStore`, `FactStore`,
//! `MentalModelStore`, and `TierMetadataStore`, plus by the orchestration
//! layer in the `brainwires-memory` crate. They live here (in the schema
//! crate) so the schema stores can be used standalone without pulling
//! `brainwires-memory` and so `brainwires-memory` doesn't have to re-export
//! schema types it doesn't own.

use chrono::Utc;

/// Seconds in one hour. Used by [`TierMetadata::retention_score`].
pub const SECS_PER_HOUR: f32 = 3600.0;

/// Default weight on similarity in the multi-factor retention score.
pub const SIMILARITY_WEIGHT: f32 = 0.50;

/// Default weight on recency in the multi-factor retention score.
pub const RECENCY_WEIGHT: f32 = 0.30;

/// Default weight on importance in the multi-factor retention score.
pub const IMPORTANCE_WEIGHT: f32 = 0.20;

/// Trust level of a memory entry's origin.
///
/// Controls which code paths are allowed to write long-lived `Canonical`
/// entries into the memory store. The `brainwires-memory` crate gates
/// canonical writes behind a capability token (`CanonicalWriteToken`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum MemoryAuthority {
    /// Transient — may be discarded between runs without notice.
    Ephemeral,
    /// Default for agent messages — persists for the duration of a session.
    #[default]
    Session,
    /// Long-lived, authoritative knowledge.
    Canonical,
}

impl MemoryAuthority {
    /// Display string used as the stored column value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ephemeral => "ephemeral",
            Self::Session => "session",
            Self::Canonical => "canonical",
        }
    }

    /// Parse from a stored string.
    pub fn parse(s: &str) -> Self {
        match s {
            "ephemeral" => Self::Ephemeral,
            "canonical" => Self::Canonical,
            _ => Self::Session,
        }
    }
}

/// Memory tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// Full messages — highest fidelity.
    Hot,
    /// Compressed summaries — medium fidelity.
    Warm,
    /// Key facts only — lowest fidelity but most compressed.
    Cold,
    /// Synthesised agent beliefs about patterns — the deepest tier.
    /// Entries are written explicitly; never populated automatically.
    MentalModel,
}

impl MemoryTier {
    /// Get the next cooler tier.
    pub fn demote(&self) -> Option<MemoryTier> {
        match self {
            MemoryTier::Hot => Some(MemoryTier::Warm),
            MemoryTier::Warm => Some(MemoryTier::Cold),
            MemoryTier::Cold => Some(MemoryTier::MentalModel),
            MemoryTier::MentalModel => None,
        }
    }

    /// Get the next hotter tier.
    pub fn promote(&self) -> Option<MemoryTier> {
        match self {
            MemoryTier::Hot => None,
            MemoryTier::Warm => Some(MemoryTier::Hot),
            MemoryTier::Cold => Some(MemoryTier::Warm),
            MemoryTier::MentalModel => Some(MemoryTier::Cold),
        }
    }
}

/// Metadata tracking for tiered storage.
#[derive(Debug, Clone)]
pub struct TierMetadata {
    /// Message identifier.
    pub message_id: String,
    /// Current memory tier.
    pub tier: MemoryTier,
    /// Importance score (0.0–1.0).
    pub importance: f32,
    /// Last access timestamp (Unix seconds).
    pub last_accessed: i64,
    /// Number of times accessed.
    pub access_count: u32,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Authority level of this memory entry. Defaults to `Session`.
    pub authority: MemoryAuthority,
}

impl TierMetadata {
    /// Create new tier metadata with the given importance score.
    pub fn new(message_id: String, importance: f32) -> Self {
        let now = Utc::now().timestamp();
        Self {
            message_id,
            tier: MemoryTier::Hot,
            importance,
            last_accessed: now,
            access_count: 0,
            created_at: now,
            authority: MemoryAuthority::Session,
        }
    }

    /// Create metadata with explicit authority level.
    pub fn with_authority(message_id: String, importance: f32, authority: MemoryAuthority) -> Self {
        Self {
            authority,
            ..Self::new(message_id, importance)
        }
    }

    /// Record an access.
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now().timestamp();
        self.access_count += 1;
    }

    /// Calculate a score for demotion priority (lower = demote first).
    pub fn retention_score(&self) -> f32 {
        let age_hours = (Utc::now().timestamp() - self.last_accessed) as f32 / SECS_PER_HOUR;
        let recency_factor = (-0.01 * age_hours).exp();
        let access_factor = (self.access_count as f32).ln_1p() * 0.1;

        self.importance * SIMILARITY_WEIGHT
            + recency_factor * RECENCY_WEIGHT
            + access_factor * IMPORTANCE_WEIGHT
    }
}

/// Summary of a message for warm-tier storage.
#[derive(Debug, Clone)]
pub struct MessageSummary {
    /// Unique summary identifier.
    pub summary_id: String,
    /// Original message that was summarized.
    pub original_message_id: String,
    /// Conversation this summary belongs to.
    pub conversation_id: String,
    /// Role of the original message.
    pub role: String,
    /// Summarized text.
    pub summary: String,
    /// Key entities mentioned in the message.
    pub key_entities: Vec<String>,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
}

/// Key fact extracted from messages for cold-tier storage.
#[derive(Debug, Clone)]
pub struct KeyFact {
    /// Unique fact identifier.
    pub fact_id: String,
    /// Messages this fact was extracted from.
    pub original_message_ids: Vec<String>,
    /// Conversation this fact belongs to.
    pub conversation_id: String,
    /// The fact text.
    pub fact: String,
    /// Category of the fact.
    pub fact_type: FactType,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
}

/// Type of key fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactType {
    /// A decision that was made.
    Decision,
    /// A definition or concept.
    Definition,
    /// A requirement or constraint.
    Requirement,
    /// A code change or modification.
    CodeChange,
    /// A configuration setting.
    Configuration,
    /// Other type of fact.
    Other,
}
