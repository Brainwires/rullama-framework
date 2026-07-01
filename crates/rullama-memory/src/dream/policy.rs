//! Demotion policy — rules governing when messages are eligible for
//! consolidation into summaries or cold-tier facts.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Policy that controls when messages are demoted from hot storage into
/// summaries (warm) or facts (cold).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemotionPolicy {
    /// Maximum age for hot-tier messages before they become candidates for
    /// summarisation. Default: 24 hours.
    pub hot_max_age: Duration,

    /// Maximum age for warm-tier summaries before they become candidates for
    /// fact extraction. Default: 7 days.
    pub warm_max_age: Duration,

    /// Token budget for the hot tier. When the tier exceeds this budget,
    /// the oldest / least-important messages are consolidated first.
    /// Default: 50 000.
    pub hot_token_budget: usize,

    /// Number of recent messages to *always* keep in the hot tier, regardless
    /// of age or importance. Default: 4.
    pub keep_recent: usize,

    /// Minimum importance score (0.0–1.0) for a message to be retained in the
    /// hot tier even if it exceeds `hot_max_age`. Default: 0.3.
    pub min_importance: f32,
}

impl Default for DemotionPolicy {
    fn default() -> Self {
        Self {
            hot_max_age: Duration::from_secs(24 * 3600),
            warm_max_age: Duration::from_secs(7 * 24 * 3600),
            hot_token_budget: 50_000,
            keep_recent: 4,
            min_importance: 0.3,
        }
    }
}

impl DemotionPolicy {
    /// Decide whether a message should be demoted from the hot tier.
    ///
    /// Returns `true` when the message is old enough *and* its importance is
    /// below the retention threshold *and* the current tier token count is
    /// above the budget.
    pub fn should_demote(
        &self,
        message_age: Duration,
        importance: f32,
        tier_token_count: usize,
    ) -> bool {
        // Never demote if we are under budget
        if tier_token_count <= self.hot_token_budget {
            return false;
        }
        // Never demote if the message is still young
        if message_age < self.hot_max_age {
            return false;
        }
        // Keep messages above the importance threshold
        if importance >= self.min_importance {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let p = DemotionPolicy::default();
        assert_eq!(p.hot_max_age, Duration::from_secs(86400));
        assert_eq!(p.warm_max_age, Duration::from_secs(604800));
        assert_eq!(p.hot_token_budget, 50_000);
        assert_eq!(p.keep_recent, 4);
        assert!((p.min_importance - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_should_demote_under_budget() {
        let p = DemotionPolicy::default();
        // Under budget — never demote
        assert!(!p.should_demote(Duration::from_secs(100_000), 0.0, 1000));
    }

    #[test]
    fn test_should_demote_young_message() {
        let p = DemotionPolicy::default();
        // Over budget but young — do not demote
        assert!(!p.should_demote(Duration::from_secs(3600), 0.0, 100_000));
    }

    #[test]
    fn test_should_demote_important_message() {
        let p = DemotionPolicy::default();
        // Over budget, old, but high importance — do not demote
        assert!(!p.should_demote(Duration::from_secs(100_000), 0.5, 100_000));
    }

    #[test]
    fn test_should_demote_eligible() {
        let p = DemotionPolicy::default();
        // Over budget, old, and low importance — demote
        assert!(p.should_demote(Duration::from_secs(100_000), 0.1, 100_000));
    }

    #[test]
    fn test_should_demote_at_boundary() {
        let p = DemotionPolicy::default();
        // Exactly at importance threshold — should NOT demote (>= check)
        assert!(!p.should_demote(Duration::from_secs(100_000), 0.3, 100_000));
        // Just below threshold — should demote
        assert!(p.should_demote(Duration::from_secs(100_000), 0.29, 100_000));
    }
}
