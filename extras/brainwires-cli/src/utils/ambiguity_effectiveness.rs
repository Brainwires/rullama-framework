//! AT-CoT Ambiguity Type Effectiveness Tracking
//!
//! Tracks which ambiguity type combinations lead to successful task completion
//! and promotes effective patterns to BKS for collective learning.

use crate::types::question::AmbiguityType;
use anyhow::Result;
use brainwires::knowledge::bks_pks::personal::{
    PersonalFact, PersonalFactCategory, PersonalFactSource, PersonalKnowledgeCache,
};
use brainwires::knowledge::bks_pks::{
    BehavioralKnowledgeCache, BehavioralTruth, TruthCategory, TruthSource,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Tracks effectiveness of ambiguity type predictions
///
/// Uses EMA (Exponential Moving Average) statistics to track success rates,
/// iteration counts, and quality scores for different ambiguity type combinations.
/// Promotes effective patterns to BKS when reliability thresholds are met.
pub struct AmbiguityEffectivenessTracker {
    /// BKS cache for shared learning
    bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,

    /// PKS cache for user preferences
    pks_cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,

    /// Local statistics (in-memory)
    local_stats: HashMap<TypeCombination, TypeStats>,

    /// EMA alpha for statistics updates (default: 0.3)
    ema_alpha: f32,

    /// Promotion threshold for BKS (default: 0.8 = 80%)
    promotion_threshold: f32,

    /// Minimum uses before promotion (default: 5)
    min_uses: u32,
}

/// Combination of ambiguity types used together
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TypeCombination {
    /// Ambiguity types, sorted for consistent hashing
    types: Vec<AmbiguityType>,
}

/// Statistics for a type combination
#[derive(Debug, Clone)]
pub(crate) struct TypeStats {
    /// Number of successful task completions
    success_count: u32,

    /// Number of failed task completions
    failure_count: u32,

    /// EMA of iterations used (lower is better)
    avg_iterations: f32,

    /// EMA of quality score (0.0-1.0, higher is better)
    avg_quality_score: f32,

    /// Whether this combination has been promoted to BKS
    promoted_to_bks: bool,
}

#[allow(dead_code)]
impl TypeStats {
    /// Calculate success rate as a fraction (0.0-1.0)
    pub(crate) fn success_rate(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.0;
        }
        self.success_count as f32 / total as f32
    }

    /// Total number of uses (successes + failures)
    pub(crate) fn total_uses(&self) -> u32 {
        self.success_count + self.failure_count
    }
}

impl AmbiguityEffectivenessTracker {
    /// Create a new effectiveness tracker
    pub fn new(
        bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,
        pks_cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,
    ) -> Self {
        Self {
            bks_cache,
            pks_cache,
            local_stats: HashMap::new(),
            ema_alpha: 0.3,
            promotion_threshold: 0.8,
            min_uses: 5,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,
        pks_cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,
        ema_alpha: f32,
        promotion_threshold: f32,
        min_uses: u32,
    ) -> Self {
        Self {
            bks_cache,
            pks_cache,
            local_stats: HashMap::new(),
            ema_alpha,
            promotion_threshold,
            min_uses,
        }
    }

    /// Record outcome of using specific ambiguity types
    ///
    /// # Arguments
    /// * `types` - Ambiguity types predicted for this query
    /// * `task_description` - Description of the task (for BKS context)
    /// * `success` - Whether the task completed successfully
    /// * `iterations` - Number of iterations used
    /// * `quality_score` - Quality score (0.0-1.0, higher is better)
    pub async fn record_outcome(
        &mut self,
        types: &[AmbiguityType],
        task_description: &str,
        success: bool,
        iterations: u32,
        quality_score: f32,
    ) -> Result<()> {
        if types.is_empty() {
            warn!("Cannot record outcome for empty ambiguity types");
            return Ok(());
        }

        let combo = TypeCombination::from_types(types);

        // Update local stats with EMA
        let stats = self.local_stats.entry(combo.clone()).or_insert(TypeStats {
            success_count: 0,
            failure_count: 0,
            avg_iterations: iterations as f32,
            avg_quality_score: quality_score,
            promoted_to_bks: false,
        });

        if success {
            stats.success_count += 1;
        } else {
            stats.failure_count += 1;
        }

        // Update EMA statistics
        let alpha = self.ema_alpha;
        stats.avg_iterations = alpha * (iterations as f32) + (1.0 - alpha) * stats.avg_iterations;
        stats.avg_quality_score = alpha * quality_score + (1.0 - alpha) * stats.avg_quality_score;

        let total_uses = stats.success_count + stats.failure_count;
        let reliability = stats.success_count as f32 / total_uses as f32;

        debug!(
            "Recorded AT-CoT outcome: {:?}, success={}, total_uses={}, reliability={:.1}%",
            combo.types,
            success,
            total_uses,
            reliability * 100.0
        );

        // Clone stats for BKS/PKS updates (avoids borrow issues)
        let stats_clone = stats.clone();
        let should_promote = !stats.promoted_to_bks;

        // Drop the mutable borrow
        let _ = stats;

        // Check for BKS promotion (80% success, 5+ uses)
        if should_promote {
            self.check_and_promote_cloned(&combo, task_description, stats_clone.clone())
                .await?;
        }

        // Update PKS user preferences
        self.update_user_preferences_cloned(&combo, stats_clone)
            .await?;

        Ok(())
    }

    /// Check if type combination should be promoted to BKS (using cloned stats)
    async fn check_and_promote_cloned(
        &mut self,
        combo: &TypeCombination,
        task_description: &str,
        stats: TypeStats,
    ) -> Result<()> {
        let total_uses = stats.success_count + stats.failure_count;
        let reliability = stats.success_count as f32 / total_uses as f32;

        if reliability >= self.promotion_threshold && total_uses >= self.min_uses {
            // Promote to BKS
            if let Some(ref bks_cache) = self.bks_cache {
                let truth = BehavioralTruth::new(
                    TruthCategory::ClarifyingQuestions,
                    extract_context_pattern(task_description).to_string(),
                    format!(
                        "For queries requiring {:?} ambiguity type disambiguation, AT-CoT achieves {:.1}% success rate",
                        combo.types,
                        reliability * 100.0
                    ),
                    format!(
                        "Learned from {} task completions with avg {} iterations and {:.2} quality score",
                        total_uses,
                        stats.avg_iterations.round(),
                        stats.avg_quality_score
                    ),
                    TruthSource::SuccessPattern,
                    None,
                );

                let mut bks: tokio::sync::MutexGuard<'_, BehavioralKnowledgeCache> =
                    bks_cache.lock().await;
                bks.queue_submission(truth)?;

                // Mark as promoted in the actual stats
                if let Some(actual_stats) = self.local_stats.get_mut(combo) {
                    actual_stats.promoted_to_bks = true;
                }

                info!(
                    "Promoted AT-CoT ambiguity type combination to BKS: {:?} ({:.1}% success, {} uses)",
                    combo.types,
                    reliability * 100.0,
                    total_uses
                );
            }
        }

        Ok(())
    }

    /// Update PKS with user's preferred ambiguity types (using cloned stats)
    async fn update_user_preferences_cloned(
        &mut self,
        combo: &TypeCombination,
        stats: TypeStats,
    ) -> Result<()> {
        let total_uses = stats.success_count + stats.failure_count;
        let reliability = stats.success_count as f32 / total_uses as f32;

        // Store in PKS if reliability is high (local only, not synced)
        if reliability >= 0.7
            && total_uses >= 3
            && let Some(ref pks_cache) = self.pks_cache
        {
            let fact = PersonalFact::new(
                PersonalFactCategory::AmbiguityTypePreference,
                format!("preferred_ambiguity_types:{}", combo.to_key()),
                format!("{:?}", combo.types),
                Some(format!(
                    "{:.1}% success rate over {} uses",
                    reliability * 100.0,
                    total_uses
                )),
                PersonalFactSource::InferredFromBehavior,
                true, // local only
            );

            let mut pks = pks_cache.lock().await;
            pks.queue_submission(fact)?;

            debug!(
                "Updated PKS with ambiguity type preference: {:?} ({:.1}% success)",
                combo.types,
                reliability * 100.0
            );
        }

        Ok(())
    }

    /// Get statistics for a specific type combination
    #[cfg(test)]
    pub(crate) fn get_stats(&self, types: &[AmbiguityType]) -> Option<&TypeStats> {
        let combo = TypeCombination::from_types(types);
        self.local_stats.get(&combo)
    }

    /// Get all tracked type combinations
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn get_all_combinations(&self) -> Vec<(Vec<AmbiguityType>, TypeStats)> {
        self.local_stats
            .iter()
            .map(|(combo, stats)| (combo.types.clone(), stats.clone()))
            .collect()
    }
}

impl TypeCombination {
    /// Create from a slice of ambiguity types (sorted for consistency)
    fn from_types(types: &[AmbiguityType]) -> Self {
        let mut sorted_types = types.to_vec();
        sorted_types.sort_by_key(|t| format!("{:?}", t));
        Self {
            types: sorted_types,
        }
    }

    /// Get a stable string key for this combination
    fn to_key(&self) -> String {
        self.types
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// Extract a context pattern from task description for BKS
fn extract_context_pattern(task_description: &str) -> &str {
    // Simple heuristic: take first 100 chars or until first newline
    let pattern = task_description.lines().next().unwrap_or(task_description);
    if pattern.len() > 100 {
        &pattern[..100]
    } else {
        pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_combination_from_types() {
        let types = vec![AmbiguityType::Specify, AmbiguityType::Semantic];
        let combo = TypeCombination::from_types(&types);

        // Should be sorted: Semantic comes before Specify alphabetically
        assert_eq!(combo.types.len(), 2);
        assert_eq!(combo.types[0], AmbiguityType::Semantic);
        assert_eq!(combo.types[1], AmbiguityType::Specify);
    }

    #[test]
    fn test_type_combination_key() {
        let types = vec![AmbiguityType::Semantic, AmbiguityType::Specify];
        let combo = TypeCombination::from_types(&types);

        let key = combo.to_key();
        assert_eq!(key, "Semantic+Specify");
    }

    #[test]
    fn test_type_stats_success_rate() {
        let stats = TypeStats {
            success_count: 8,
            failure_count: 2,
            avg_iterations: 10.0,
            avg_quality_score: 0.85,
            promoted_to_bks: false,
        };

        assert_eq!(stats.success_rate(), 0.8);
        assert_eq!(stats.total_uses(), 10);
    }

    #[tokio::test]
    async fn test_record_outcome_updates_stats() {
        let mut tracker = AmbiguityEffectivenessTracker::new(None, None);

        let types = vec![AmbiguityType::Semantic];
        tracker
            .record_outcome(&types, "test task", true, 10, 0.9)
            .await
            .unwrap();

        let stats = tracker.get_stats(&types).unwrap();
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.avg_iterations, 10.0);
        assert_eq!(stats.avg_quality_score, 0.9);
    }

    #[tokio::test]
    async fn test_ema_statistics() {
        let mut tracker = AmbiguityEffectivenessTracker::new(None, None);

        let types = vec![AmbiguityType::Semantic];

        // First outcome: 10 iterations, 0.9 quality
        tracker
            .record_outcome(&types, "test", true, 10, 0.9)
            .await
            .unwrap();

        // Second outcome: 20 iterations, 0.7 quality
        tracker
            .record_outcome(&types, "test", true, 20, 0.7)
            .await
            .unwrap();

        let stats = tracker.get_stats(&types).unwrap();

        // EMA with alpha=0.3:
        // avg_iterations = 0.3 * 20 + 0.7 * 10 = 13.0
        // avg_quality = 0.3 * 0.7 + 0.7 * 0.9 = 0.84
        assert!((stats.avg_iterations - 13.0).abs() < 0.01);
        assert!((stats.avg_quality_score - 0.84).abs() < 0.01);
    }

    #[test]
    fn test_extract_context_pattern() {
        assert_eq!(
            extract_context_pattern("Implement a cache"),
            "Implement a cache"
        );

        assert_eq!(
            extract_context_pattern("Implement a cache\nwith LRU eviction"),
            "Implement a cache"
        );

        let long_text = "a".repeat(150);
        assert_eq!(extract_context_pattern(&long_text).len(), 100);
    }
}
