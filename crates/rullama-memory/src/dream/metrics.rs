//! Dream consolidation metrics and reporting.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::fact_extractor::ExtractedFact;

/// Metrics collected during a single dream consolidation cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamMetrics {
    /// Number of sessions that were scanned.
    pub sessions_processed: usize,
    /// Number of messages that were summarised.
    pub messages_summarized: usize,
    /// Number of durable facts extracted.
    pub facts_extracted: usize,
    /// Total token estimate *before* consolidation.
    pub tokens_before: usize,
    /// Total token estimate *after* consolidation.
    pub tokens_after: usize,
    /// Number of contradictions detected between old and new facts.
    pub contradictions_found: usize,
    /// Wall-clock duration of the consolidation cycle.
    #[serde(with = "humantime_serde_compat")]
    pub duration: Duration,
}

/// Full report returned by [`DreamConsolidator::run_cycle`](super::consolidator::DreamConsolidator::run_cycle).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamReport {
    /// Aggregate metrics for the cycle.
    pub metrics: DreamMetrics,
    /// Summaries that were created during this cycle.
    pub summaries_created: Vec<String>,
    /// Facts that were extracted during this cycle.
    pub facts: Vec<ExtractedFact>,
    /// Non-fatal errors encountered during processing.
    pub errors: Vec<String>,
}

/// Minimal serde helper for `Duration` — stores as fractional seconds.
mod humantime_serde_compat {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_metrics_default() {
        let m = DreamMetrics::default();
        assert_eq!(m.sessions_processed, 0);
        assert_eq!(m.messages_summarized, 0);
        assert_eq!(m.facts_extracted, 0);
        assert_eq!(m.tokens_before, 0);
        assert_eq!(m.tokens_after, 0);
        assert_eq!(m.contradictions_found, 0);
        assert_eq!(m.duration, Duration::ZERO);
    }

    #[test]
    fn test_dream_report_default() {
        let r = DreamReport::default();
        assert!(r.summaries_created.is_empty());
        assert!(r.facts.is_empty());
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_metrics_serde_roundtrip() {
        let m = DreamMetrics {
            sessions_processed: 3,
            messages_summarized: 42,
            facts_extracted: 7,
            tokens_before: 100_000,
            tokens_after: 20_000,
            contradictions_found: 1,
            duration: Duration::from_millis(1234),
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: DreamMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.sessions_processed, 3);
        assert_eq!(m2.messages_summarized, 42);
        assert_eq!(m2.facts_extracted, 7);
        // Duration may have tiny floating-point rounding
        assert!((m2.duration.as_secs_f64() - 1.234).abs() < 0.001);
    }
}
