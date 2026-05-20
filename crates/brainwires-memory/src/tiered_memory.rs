//! Tiered Memory Storage System
//!
//! Implements a three-tier memory hierarchy for conversation storage:
//! - **Hot**: Full messages - recent, important, or recently accessed
//! - **Warm**: Compressed summaries - older messages that may be needed
//! - **Cold**: Ultra-compressed key facts - archival storage
//!
//! Messages flow from hot → warm → cold based on age and importance,
//! and can be promoted back up when accessed.
//!
//! ## Persistence
//!
//! All tiers are backed by LanceDB for persistence:
//! - Hot tier: MessageStore (messages table)
//! - Warm tier: SummaryStore (summaries table)
//! - Cold tier: FactStore (facts table)
//! - Metadata: TierMetadataStore (tier_metadata table)

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use uuid::Uuid;

use brainwires_storage::CachedEmbeddingProvider;
use brainwires_storage::databases::{LanceDatabase, StorageBackend};

use brainwires_stores::{
    FactStore, FactType, KeyFact, MemoryAuthority, MemoryTier, MentalModel, MentalModelStore,
    MessageMetadata, MessageStore, MessageSummary, ModelType, SummaryStore, TierMetadata,
    TierMetadataStore,
};
// Weight constants (SIMILARITY_WEIGHT, RECENCY_WEIGHT, IMPORTANCE_WEIGHT)
// are defined locally below — they live in brainwires-stores::tier_types
// for the schema stores' use, and are duplicated here intentionally to
// keep the orchestration crate self-contained.

const SIMILARITY_WEIGHT: f32 = 0.50;
const RECENCY_WEIGHT: f32 = 0.30;
const IMPORTANCE_WEIGHT: f32 = 0.20;
const DEFAULT_HOT_RETENTION_HOURS: u64 = 24;
const DEFAULT_WARM_RETENTION_HOURS: u64 = 168;
const DEFAULT_HOT_IMPORTANCE_THRESHOLD: f32 = 0.3;
const DEFAULT_WARM_IMPORTANCE_THRESHOLD: f32 = 0.1;
const DEFAULT_MAX_HOT_MESSAGES: usize = 1000;
const DEFAULT_MAX_WARM_SUMMARIES: usize = 5000;
const FAST_DECAY_RATE: f32 = 0.05;

/// Temporal keywords that imply a recency-sensitive query.
const TEMPORAL_KEYWORDS: &[&str] = &[
    "recent",
    "recently",
    "latest",
    "last",
    "current",
    "currently",
    "today",
    "yesterday",
    "this week",
    "now",
    "just",
    "new",
    "newest",
];

/// Detect whether a query is temporally sensitive.
///
/// Returns a score in `[0.0, 1.0]` based on keyword density: each matching
/// keyword from [`TEMPORAL_KEYWORDS`] contributes 1 hit; score is clamped at
/// `hits / 3.0` to avoid saturating on very long queries.
fn detect_temporal_query(query: &str) -> f32 {
    let lower = query.to_lowercase();
    let hits = TEMPORAL_KEYWORDS
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();
    (hits as f32 / 3.0).min(1.0)
}

// MemoryAuthority, MemoryTier, TierMetadata, MessageSummary, KeyFact,
// FactType — moved to brainwires-stores::tier_types (used by both the
// schema stores and this orchestration layer).

/// Capability token that unlocks writes to the `Canonical` memory authority tier.
///
/// The constructor is intentionally `pub(crate)` — external crates obtain one
/// only through designated authorisation entry points (e.g. a CLI-layer
/// function or a privileged agent config). This ensures that ordinary agent
/// tool calls cannot silently promote their output to canonical authority.
///
/// ## Example
/// ```ignore
/// // Inside crate only:
/// let token = CanonicalWriteToken::new();
/// tiered_memory.add_canonical_message(message, 0.9, token).await?;
/// ```
#[derive(Debug)]
pub struct CanonicalWriteToken(());

impl CanonicalWriteToken {
    /// Create a new token. Only callable within this crate.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self(())
    }
}

/// Combined retrieval score that blends similarity, recency, and stored importance.
///
/// Weights: similarity × 0.50 + recency × 0.30 + importance × 0.20.
#[derive(Debug, Clone)]
pub struct MultiFactorScore {
    /// Raw cosine/dot-product similarity from the embedding search (0–1).
    pub similarity: f32,
    /// Recency factor: `exp(−0.01 × hours_since_last_access)`.  1.0 = just
    /// accessed, approaches 0 for very old entries.
    pub recency: f32,
    /// Stored importance score (0–1) from [`TierMetadata::importance`].
    pub importance: f32,
    /// Weighted combined score used for ranking.
    pub combined: f32,
}

impl MultiFactorScore {
    /// Compute the combined score from its components using default weights.
    pub fn compute(similarity: f32, recency: f32, importance: f32) -> Self {
        Self::compute_with_weights(
            similarity,
            recency,
            importance,
            SIMILARITY_WEIGHT,
            RECENCY_WEIGHT,
            IMPORTANCE_WEIGHT,
        )
    }

    /// Compute the combined score using caller-supplied weights.
    ///
    /// Weights need not sum to 1.0 — `combined` is the raw dot product and is
    /// clamped to `[0.0, 1.0]` for consistency.
    pub fn compute_with_weights(
        similarity: f32,
        recency: f32,
        importance: f32,
        sim_w: f32,
        rec_w: f32,
        imp_w: f32,
    ) -> Self {
        let combined = (similarity * sim_w + recency * rec_w + importance * imp_w).clamp(0.0, 1.0);
        Self {
            similarity,
            recency,
            importance,
            combined,
        }
    }

    /// Decay rate used for the recency factor (per hour).
    const DECAY_RATE: f32 = 0.01;

    /// Compute the recency factor from `hours_since_last_access`.
    pub fn recency_from_hours(hours_since_access: f32) -> f32 {
        (-Self::DECAY_RATE * hours_since_access).exp()
    }

    /// Compute the recency factor using the fast decay rate (`exp(-0.05 × h)`).
    pub fn recency_from_hours_fast(hours_since_access: f32) -> f32 {
        (-FAST_DECAY_RATE * hours_since_access).exp()
    }
}

/// Result from adaptive search across tiers
#[derive(Debug, Clone)]
pub struct TieredSearchResult {
    /// The content text.
    pub content: String,
    /// Raw similarity score returned by the vector store (0-1).
    pub score: f32,
    /// Memory tier this result came from.
    pub tier: MemoryTier,
    /// Original message identifier.
    pub original_message_id: Option<String>,
    /// Full message metadata if available.
    pub metadata: Option<MessageMetadata>,
    /// Multi-factor score blending similarity, recency, and importance.
    /// Populated by [`TieredMemory::search_adaptive_multi_factor`]; `None` when
    /// returned by the basic [`TieredMemory::search_adaptive`].
    pub multi_factor_score: Option<MultiFactorScore>,
}

/// Configuration for tiered memory behavior
#[derive(Debug, Clone)]
pub struct TieredMemoryConfig {
    /// Hours before considering demotion from hot to warm
    pub hot_retention_hours: u64,
    /// Hours before considering demotion from warm to cold
    pub warm_retention_hours: u64,
    /// Minimum importance score to stay in hot tier
    pub hot_importance_threshold: f32,
    /// Minimum importance score to stay in warm tier
    pub warm_importance_threshold: f32,
    /// Maximum messages in hot tier
    pub max_hot_messages: usize,
    /// Maximum summaries in warm tier
    pub max_warm_summaries: usize,
    /// Optional TTL for session-tier messages, in seconds.
    ///
    /// When set, every message added via [`TieredMemory::add_message`] receives
    /// an `expires_at` timestamp of `now + session_ttl_secs`.  Expired entries
    /// are removed by [`TieredMemory::evict_expired`] or lazily during
    /// [`TieredMemory::search_adaptive`].
    ///
    /// `None` (the default) means no TTL — messages persist until explicitly
    /// deleted or demoted.
    pub session_ttl_secs: Option<u64>,
    /// Extra recency weight added when a query is detected as temporally
    /// sensitive (e.g. contains "recent", "latest", "today").
    ///
    /// The additional weight is proportional to the query's temporal score
    /// (`0.0–1.0`).  The three weights are renormalised so they always sum to
    /// `1.0`.  Default: `0.3`.
    pub temporal_boost: f32,
    /// Use a faster recency decay rate (`exp(-0.05 × h)` instead of
    /// `exp(-0.01 × h)`) when the query is temporally sensitive.
    ///
    /// Default: `false`.
    pub fast_decay: bool,
    /// Maximum number of synthesised mental models to retain.
    ///
    /// Default: `500`.
    pub max_mental_models: usize,
}

impl Default for TieredMemoryConfig {
    fn default() -> Self {
        Self {
            hot_retention_hours: DEFAULT_HOT_RETENTION_HOURS,
            warm_retention_hours: DEFAULT_WARM_RETENTION_HOURS,
            hot_importance_threshold: DEFAULT_HOT_IMPORTANCE_THRESHOLD,
            warm_importance_threshold: DEFAULT_WARM_IMPORTANCE_THRESHOLD,
            max_hot_messages: DEFAULT_MAX_HOT_MESSAGES,
            max_warm_summaries: DEFAULT_MAX_WARM_SUMMARIES,
            session_ttl_secs: None,
            temporal_boost: 0.3,
            fast_decay: false,
            max_mental_models: 500,
        }
    }
}

/// Three-tier memory storage system with persistence
pub struct TieredMemory {
    /// Hot tier: Full messages (LanceDB-backed)
    pub hot: Arc<MessageStore>,

    /// Warm tier: Summaries (LanceDB-backed)
    warm: SummaryStore,

    /// Cold tier: Key facts (LanceDB-backed)
    cold: FactStore,

    /// Metadata tracking for all messages (LanceDB-backed)
    tier_metadata: TierMetadataStore,

    /// Mental model tier: synthesised beliefs (LanceDB-backed)
    mental_model: MentalModelStore,

    /// Configuration
    config: TieredMemoryConfig,

    /// Embedding provider for searches
    #[allow(dead_code)]
    embeddings: Arc<CachedEmbeddingProvider>,
}

impl TieredMemory {
    /// Create a new tiered memory system with persistent storage
    pub async fn new(
        hot_store: Arc<MessageStore>,
        db: Arc<LanceDatabase>,
        embeddings: Arc<CachedEmbeddingProvider>,
        config: TieredMemoryConfig,
    ) -> Self {
        let mental_model = MentalModelStore::new(
            Arc::clone(&db) as Arc<dyn StorageBackend>,
            Arc::clone(&embeddings),
        );
        Self {
            hot: hot_store,
            warm: SummaryStore::new(Arc::clone(&db), Arc::clone(&embeddings)),
            cold: FactStore::new(Arc::clone(&db), Arc::clone(&embeddings)),
            tier_metadata: TierMetadataStore::new(db),
            mental_model,
            config,
            embeddings,
        }
    }

    /// Create with default configuration
    pub async fn with_defaults(
        hot_store: Arc<MessageStore>,
        db: Arc<LanceDatabase>,
        embeddings: Arc<CachedEmbeddingProvider>,
    ) -> Self {
        Self::new(hot_store, db, embeddings, TieredMemoryConfig::default()).await
    }

    /// Add a message to the hot tier with `Session` authority.
    ///
    /// If `TieredMemoryConfig::session_ttl_secs` is set, the message will be
    /// assigned an expiry timestamp and will be removed by [`Self::evict_expired`]
    /// after the configured duration.
    pub async fn add_message(
        &mut self,
        mut message: MessageMetadata,
        importance: f32,
    ) -> Result<()> {
        // Apply TTL if configured
        if let Some(ttl_secs) = self.config.session_ttl_secs {
            message.expires_at = Some(Utc::now().timestamp() + ttl_secs as i64);
        }
        let metadata = TierMetadata::new(message.message_id.clone(), importance);
        self.tier_metadata.add(metadata).await?;
        self.hot.add(message).await
    }

    /// Add a message to the hot tier with `Canonical` authority.
    ///
    /// Canonical entries are long-lived and immune to session-TTL eviction.
    /// A [`CanonicalWriteToken`] is required to call this method; obtain one
    /// through an authorised entry point in the CLI layer.
    pub async fn add_canonical_message(
        &mut self,
        message: MessageMetadata,
        importance: f32,
        _token: CanonicalWriteToken,
    ) -> Result<()> {
        // Canonical entries intentionally have no TTL
        let metadata = TierMetadata::with_authority(
            message.message_id.clone(),
            importance,
            MemoryAuthority::Canonical,
        );
        self.tier_metadata.add(metadata).await?;
        self.hot.add(message).await
    }

    /// Delete all hot-tier messages whose TTL has expired.
    ///
    /// Returns the number of entries evicted.  Call this at agent run
    /// completion or on a periodic background schedule.
    ///
    /// Canonical-authority messages are never evicted here regardless of
    /// any `expires_at` value, because they are expected to have `None`.
    pub async fn evict_expired(&self) -> Result<usize> {
        let evicted = self.hot.delete_expired().await?;
        if evicted > 0 {
            tracing::info!(
                evicted,
                "TieredMemory: evicted {} expired message(s)",
                evicted
            );
        }
        Ok(evicted)
    }

    /// Record access to a message (for promotion/retention decisions)
    pub async fn record_access(&mut self, message_id: &str) -> Result<()> {
        if let Some(mut meta) = self.tier_metadata.get(message_id).await? {
            meta.record_access();
            self.tier_metadata.update(meta).await?;
        }
        Ok(())
    }

    /// Search across all tiers with adaptive resolution
    pub async fn search_adaptive(
        &mut self,
        query: &str,
        conversation_id: Option<&str>,
    ) -> Result<Vec<TieredSearchResult>> {
        let mut results = Vec::new();

        // 1. Search hot tier first (full messages)
        let hot_results = if let Some(conv_id) = conversation_id {
            self.hot.search_conversation(conv_id, query, 5, 0.6).await?
        } else {
            self.hot.search(query, 5, 0.6).await?
        };

        for (msg, score) in hot_results {
            // Lazy eviction: skip entries whose TTL has expired
            if let Some(exp) = msg.expires_at
                && exp <= Utc::now().timestamp()
            {
                continue;
            }

            // Record access for retention tracking
            let _ = self.record_access(&msg.message_id).await;

            results.push(TieredSearchResult {
                content: msg.content.clone(),
                score,
                tier: MemoryTier::Hot,
                original_message_id: Some(msg.message_id.clone()),
                metadata: Some(msg),
                multi_factor_score: None,
            });
        }

        // If we have high-confidence hot results, return early
        if results.iter().any(|r| r.score > 0.85) {
            return Ok(results);
        }

        // 2. Search warm tier (summaries)
        let warm_results = if let Some(conv_id) = conversation_id {
            self.warm
                .search_conversation(conv_id, query, 3, 0.5)
                .await?
        } else {
            self.warm.search(query, 3, 0.5).await?
        };

        for (summary, score) in warm_results {
            results.push(TieredSearchResult {
                content: summary.summary.clone(),
                score,
                tier: MemoryTier::Warm,
                original_message_id: Some(summary.original_message_id.clone()),
                metadata: None,
                multi_factor_score: None,
            });
        }

        // 3. If still no good results, search cold tier
        if results.iter().all(|r| r.score < 0.7) {
            let cold_results = if let Some(conv_id) = conversation_id {
                self.cold
                    .search_conversation(conv_id, query, 3, 0.4)
                    .await?
            } else {
                self.cold.search(query, 3, 0.4).await?
            };

            for (fact, score) in cold_results {
                results.push(TieredSearchResult {
                    content: fact.fact.clone(),
                    score,
                    tier: MemoryTier::Cold,
                    original_message_id: fact.original_message_ids.first().cloned(),
                    metadata: None,
                    multi_factor_score: None,
                });
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Search across all tiers and score results using combined similarity,
    /// recency, and importance signals.
    ///
    /// This is the preferred retrieval method for long-horizon agent tasks where
    /// a pure similarity score can surface stale or low-importance results.
    ///
    /// The returned results are sorted by [`MultiFactorScore::combined`]
    /// (descending).  Each result has `multi_factor_score` populated.
    pub async fn search_adaptive_multi_factor(
        &mut self,
        query: &str,
        conversation_id: Option<&str>,
    ) -> Result<Vec<TieredSearchResult>> {
        // Reuse the base search to get similarity-ranked results.
        let mut results = self.search_adaptive(query, conversation_id).await?;

        // Collect message IDs that have associated tier metadata (hot tier).
        let ids: Vec<&str> = results
            .iter()
            .filter_map(|r| r.original_message_id.as_deref())
            .collect();

        let meta_map = self.tier_metadata.get_many(&ids).await.unwrap_or_default();

        let now_secs = chrono::Utc::now().timestamp();

        // Compute temporal sensitivity once for the whole query.
        let temporal_factor = detect_temporal_query(query);
        let use_fast_decay = self.config.fast_decay && temporal_factor > 0.0;

        // Derive per-query weights, renormalised so they sum to 1.0.
        let extra_recency = self.config.temporal_boost * temporal_factor;
        let rec_w = (RECENCY_WEIGHT + extra_recency).min(1.0);
        let remaining = 1.0 - rec_w;
        let sim_share = SIMILARITY_WEIGHT / (SIMILARITY_WEIGHT + IMPORTANCE_WEIGHT);
        let sim_w = sim_share * remaining;
        let imp_w = remaining - sim_w;

        for result in &mut results {
            let similarity = result.score;

            let (recency, importance) = if let Some(id) = &result.original_message_id {
                if let Some(meta) = meta_map.get(id.as_str()) {
                    let hours_since = (now_secs - meta.last_accessed).max(0) as f32 / 3600.0;
                    let rec = if use_fast_decay {
                        MultiFactorScore::recency_from_hours_fast(hours_since)
                    } else {
                        MultiFactorScore::recency_from_hours(hours_since)
                    };
                    (rec, meta.importance)
                } else {
                    (1.0_f32, 0.5_f32) // Fallback: assume fresh + average importance
                }
            } else {
                (1.0_f32, 0.5_f32)
            };

            result.multi_factor_score = Some(MultiFactorScore::compute_with_weights(
                similarity, recency, importance, sim_w, rec_w, imp_w,
            ));
        }

        // Append mental model tier results (up to 5).
        if let Ok(mm_results) = self.search_mental_models(query, 5).await {
            for mut mm in mm_results {
                mm.multi_factor_score = Some(MultiFactorScore::compute_with_weights(
                    mm.score, 1.0, // mental models have no recency — treat as always fresh
                    0.5, // default importance
                    sim_w, rec_w, imp_w,
                ));
                results.push(mm);
            }
        }

        // Re-sort by combined score (highest first).
        results.sort_by(|a, b| {
            let sa = a
                .multi_factor_score
                .as_ref()
                .map_or(a.score, |s| s.combined);
            let sb = b
                .multi_factor_score
                .as_ref()
                .map_or(b.score, |s| s.combined);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Demote a message from hot to warm tier
    pub async fn demote_to_warm(
        &mut self,
        message_id: &str,
        summary: MessageSummary,
    ) -> Result<()> {
        // Update tier metadata
        if let Some(mut meta) = self.tier_metadata.get(message_id).await? {
            meta.tier = MemoryTier::Warm;
            self.tier_metadata.update(meta).await?;
        }

        // Add summary to warm tier
        self.warm.add(summary).await
    }

    /// Demote a summary from warm to cold tier
    pub async fn demote_to_cold(&mut self, summary_id: &str, fact: KeyFact) -> Result<()> {
        // Remove from warm
        self.warm.delete(summary_id).await?;

        // Add to cold
        self.cold.add(fact).await
    }

    /// Promote a message back to hot tier (re-fetch full content)
    pub async fn promote_to_hot(&mut self, message_id: &str) -> Result<Option<MessageMetadata>> {
        // Update metadata
        if let Some(mut meta) = self.tier_metadata.get(message_id).await? {
            meta.tier = MemoryTier::Hot;
            meta.record_access();
            self.tier_metadata.update(meta).await?;
        }

        // The message should still be in the hot store (we don't delete on demotion)
        // Just update access tracking
        Ok(None)
    }

    /// Get messages that should be considered for demotion
    pub async fn get_demotion_candidates(
        &self,
        tier: MemoryTier,
        count: usize,
    ) -> Result<Vec<String>> {
        let all_metadata = self.tier_metadata.get_by_tier(tier).await?;

        let mut candidates: Vec<_> = all_metadata
            .into_iter()
            .map(|m| (m.message_id.clone(), m.retention_score()))
            .collect();

        // Sort by retention score (lowest first = demote first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect())
    }

    /// Get statistics about tier distribution
    pub async fn get_stats(&self) -> Result<TieredMemoryStats> {
        let hot_count = self.tier_metadata.count_by_tier(MemoryTier::Hot).await?;
        let warm_count = self.warm.count().await?;
        let cold_count = self.cold.count().await?;
        let mental_model_count = self.mental_model.count().await.unwrap_or(0);
        let total_tracked = self.tier_metadata.count().await?;

        Ok(TieredMemoryStats {
            hot_count,
            warm_count,
            cold_count,
            mental_model_count,
            total_tracked,
        })
    }

    /// Fallback summarization without LLM
    pub fn fallback_summarize(&self, content: &str) -> String {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() <= 75 {
            content.to_string()
        } else {
            format!("{}...", words[..75].join(" "))
        }
    }

    /// Create a fallback fact from a summary
    pub fn fallback_fact(&self, summary: &MessageSummary) -> KeyFact {
        KeyFact {
            fact_id: Uuid::new_v4().to_string(),
            original_message_ids: vec![summary.original_message_id.clone()],
            conversation_id: summary.conversation_id.clone(),
            fact: summary.summary.clone(),
            fact_type: FactType::Other,
            created_at: Utc::now().timestamp(),
        }
    }

    // ── Mental model tier ────────────────────────────────────────────────────

    /// Synthesise and store a new mental model from a set of source fact IDs.
    ///
    /// Returns the new model's ID.
    ///
    /// The table is created on first call; subsequent calls reuse it.
    pub async fn synthesize_mental_model(
        &mut self,
        fact_ids: &[String],
        model_text: String,
        model_type: ModelType,
        conversation_id: String,
    ) -> Result<String> {
        self.mental_model.ensure_table().await?;

        let mut model =
            MentalModel::new(model_text, model_type, conversation_id, fact_ids.to_vec());
        model.evidence_count = fact_ids.len() as u32;
        let id = model.model_id.clone();
        self.mental_model.add(model).await?;
        Ok(id)
    }

    /// Semantic search over the mental model tier.
    pub async fn search_mental_models(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<TieredSearchResult>> {
        let raw = self.mental_model.search(query, limit).await?;
        Ok(raw
            .into_iter()
            .map(|(model, score)| TieredSearchResult {
                content: model.model_text.clone(),
                score,
                tier: MemoryTier::MentalModel,
                original_message_id: model.source_fact_ids.first().cloned(),
                metadata: None,
                multi_factor_score: None,
            })
            .collect())
    }
}

/// Statistics about tiered memory usage
#[derive(Debug, Clone)]
pub struct TieredMemoryStats {
    /// Number of entries in the hot tier.
    pub hot_count: usize,
    /// Number of entries in the warm tier.
    pub warm_count: usize,
    /// Number of entries in the cold tier.
    pub cold_count: usize,
    /// Number of synthesised mental models.
    pub mental_model_count: usize,
    /// Total tracked entries across all tiers.
    pub total_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MultiFactorScore ───────────────────────────────────────────────────

    #[test]
    fn test_multi_factor_score_weights_sum_to_one() {
        // weights: 0.50 + 0.30 + 0.20 = 1.0
        let score = MultiFactorScore::compute(1.0, 1.0, 1.0);
        assert!(
            (score.combined - 1.0).abs() < 1e-6,
            "all-one inputs should yield combined=1"
        );
    }

    #[test]
    fn test_multi_factor_score_zero_inputs() {
        let score = MultiFactorScore::compute(0.0, 0.0, 0.0);
        assert_eq!(score.combined, 0.0);
    }

    #[test]
    fn test_recency_factor_fresh_entry() {
        // An entry accessed 0 hours ago should have recency ≈ 1.0
        let r = MultiFactorScore::recency_from_hours(0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recency_factor_decays_over_time() {
        let r_now = MultiFactorScore::recency_from_hours(0.0);
        let r_day = MultiFactorScore::recency_from_hours(24.0);
        let r_week = MultiFactorScore::recency_from_hours(168.0);
        assert!(
            r_now > r_day,
            "fresh entry must score higher than 1-day-old"
        );
        assert!(
            r_day > r_week,
            "1-day-old must score higher than 1-week-old"
        );
        assert!(r_week > 0.0, "recency factor must remain positive");
    }

    #[test]
    fn test_high_similarity_low_recency_can_be_beaten_by_balanced_entry() {
        // High similarity but stale (1 week old, no importance)
        let stale =
            MultiFactorScore::compute(0.95, MultiFactorScore::recency_from_hours(168.0), 0.0);
        // Moderate similarity but recent and important
        let fresh = MultiFactorScore::compute(0.70, MultiFactorScore::recency_from_hours(1.0), 0.9);
        // The balanced entry should edge ahead
        assert!(
            fresh.combined > stale.combined,
            "fresh important entry ({:.3}) should beat stale high-similarity entry ({:.3})",
            fresh.combined,
            stale.combined
        );
    }

    // ── Tier demotion / promotion ─────────────────────────────────────────

    #[test]
    fn test_tier_demotion() {
        assert_eq!(MemoryTier::Hot.demote(), Some(MemoryTier::Warm));
        assert_eq!(MemoryTier::Warm.demote(), Some(MemoryTier::Cold));
        assert_eq!(MemoryTier::Cold.demote(), Some(MemoryTier::MentalModel));
        assert_eq!(MemoryTier::MentalModel.demote(), None);
    }

    #[test]
    fn test_tier_promotion() {
        assert_eq!(MemoryTier::Hot.promote(), None);
        assert_eq!(MemoryTier::Warm.promote(), Some(MemoryTier::Hot));
        assert_eq!(MemoryTier::Cold.promote(), Some(MemoryTier::Warm));
        assert_eq!(MemoryTier::MentalModel.promote(), Some(MemoryTier::Cold));
    }

    #[test]
    fn test_tier_metadata_retention_score() {
        let mut meta = TierMetadata::new("test-1".to_string(), 0.8);

        // High importance should give higher score
        let score1 = meta.retention_score();
        assert!(score1 > 0.0);

        // Recording access should maintain or increase score
        meta.record_access();
        let score2 = meta.retention_score();
        assert!(score2 >= score1 * 0.9); // Allow some variance due to time
    }

    #[test]
    fn test_default_config() {
        let config = TieredMemoryConfig::default();
        assert_eq!(config.hot_retention_hours, 24);
        assert_eq!(config.warm_retention_hours, 168);
        assert!(config.hot_importance_threshold > 0.0);
        assert!(config.session_ttl_secs.is_none());
    }

    #[test]
    fn test_config_with_session_ttl() {
        let config = TieredMemoryConfig {
            session_ttl_secs: Some(3600),
            ..TieredMemoryConfig::default()
        };
        assert_eq!(config.session_ttl_secs, Some(3600));
    }

    // ── MemoryAuthority ───────────────────────────────────────────────────

    #[test]
    fn test_memory_authority_default() {
        assert_eq!(MemoryAuthority::default(), MemoryAuthority::Session);
    }

    #[test]
    fn test_memory_authority_round_trip() {
        for auth in [
            MemoryAuthority::Ephemeral,
            MemoryAuthority::Session,
            MemoryAuthority::Canonical,
        ] {
            assert_eq!(MemoryAuthority::parse(auth.as_str()), auth);
        }
    }

    #[test]
    fn test_memory_authority_unknown_defaults_to_session() {
        assert_eq!(MemoryAuthority::parse("bogus"), MemoryAuthority::Session);
    }

    #[test]
    fn test_tier_metadata_default_authority() {
        let meta = TierMetadata::new("m-1".to_string(), 0.5);
        assert_eq!(meta.authority, MemoryAuthority::Session);
    }

    #[test]
    fn test_tier_metadata_with_authority() {
        let meta = TierMetadata::with_authority("m-2".to_string(), 0.9, MemoryAuthority::Canonical);
        assert_eq!(meta.authority, MemoryAuthority::Canonical);
        assert_eq!(meta.importance, 0.9);
    }

    #[test]
    fn test_canonical_write_token_is_crate_private() {
        // CanonicalWriteToken::new() is pub(crate) — this test being inside
        // the same crate confirms we can construct it; external crates cannot.
        let _token = CanonicalWriteToken::new();
    }

    // ── Feature 2: Temporal scoring ─────────────────────────────────────

    #[test]
    fn test_detect_temporal_query_empty() {
        assert_eq!(detect_temporal_query(""), 0.0);
    }

    #[test]
    fn test_detect_temporal_query_no_keywords() {
        assert_eq!(detect_temporal_query("how does authentication work?"), 0.0);
    }

    #[test]
    fn test_detect_temporal_query_single_keyword() {
        let score = detect_temporal_query("what is the latest approach?");
        assert!(score > 0.0, "expected score > 0 for 'latest'");
    }

    #[test]
    fn test_detect_temporal_query_dense() {
        let score = detect_temporal_query("what was the latest change today?");
        assert!(score > 0.0);
    }

    #[test]
    fn test_detect_temporal_query_max_clamp() {
        // Five keywords — result must be <= 1.0
        let score = detect_temporal_query("recent latest current today now new");
        assert!(score <= 1.0, "score must not exceed 1.0");
        assert!(score > 0.0);
    }

    #[test]
    fn test_compute_with_weights_sum_normalised() {
        // Weights sum to 1.0 — combined should equal the weighted sum, clamped.
        let sim_w = 0.4_f32;
        let rec_w = 0.4_f32;
        let imp_w = 0.2_f32;
        let score = MultiFactorScore::compute_with_weights(0.8, 0.9, 0.6, sim_w, rec_w, imp_w);
        let expected = (0.8 * sim_w + 0.9 * rec_w + 0.6 * imp_w).clamp(0.0, 1.0);
        assert!((score.combined - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compute_with_weights_matches_compute_for_default_weights() {
        let a = MultiFactorScore::compute(0.7, 0.8, 0.5);
        let b = MultiFactorScore::compute_with_weights(
            0.7,
            0.8,
            0.5,
            SIMILARITY_WEIGHT,
            RECENCY_WEIGHT,
            IMPORTANCE_WEIGHT,
        );
        assert!((a.combined - b.combined).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_config_defaults() {
        let cfg = TieredMemoryConfig::default();
        assert_eq!(cfg.temporal_boost, 0.3);
        assert!(!cfg.fast_decay);
    }

    #[test]
    fn test_fast_decay_rate_higher_than_normal() {
        // Fast decay should make old items score lower than normal decay.
        let hours = 48.0_f32;
        let normal = MultiFactorScore::recency_from_hours(hours);
        let fast = MultiFactorScore::recency_from_hours_fast(hours);
        assert!(
            fast < normal,
            "fast decay should produce lower recency for old items"
        );
    }
}
