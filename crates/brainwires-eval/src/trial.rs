//! Evaluation trial results and statistical analysis.
//!
//! A *trial* is one execution of an [`EvaluationCase`](crate::case::EvaluationCase).  Run N trials and
//! summarise with [`EvaluationStats`] which reports the success rate together
//! with a Wilson-score 95 % confidence interval.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Trial result ──────────────────────────────────────────────────────────────

/// Result produced by a single trial run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Sequential index of this trial (0-based).
    pub trial_id: usize,
    /// Whether the trial succeeded.
    pub success: bool,
    /// Wall-clock duration of the trial in milliseconds.
    pub duration_ms: u64,
    /// Error message when `success == false`.
    pub error: Option<String>,
    /// Arbitrary key-value metadata emitted by the case (e.g. iteration count,
    /// token usage, tool names used).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TrialResult {
    /// Create a successful trial result.
    pub fn success(trial_id: usize, duration_ms: u64) -> Self {
        Self {
            trial_id,
            success: true,
            duration_ms,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed trial result.
    pub fn failure(trial_id: usize, duration_ms: u64, error: impl Into<String>) -> Self {
        Self {
            trial_id,
            success: false,
            duration_ms,
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }

    /// Attach an arbitrary metadata value.
    pub fn with_meta(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ── Confidence interval ───────────────────────────────────────────────────────

/// A symmetric 95 % confidence interval around a proportion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceInterval95 {
    /// Lower bound (clipped to 0).
    pub lower: f64,
    /// Upper bound (clipped to 1).
    pub upper: f64,
}

impl ConfidenceInterval95 {
    /// Compute a Wilson-score 95 % confidence interval.
    ///
    /// The Wilson interval is preferred over the naïve Wald interval because it
    /// behaves well at the extremes (p = 0 or p = 1) and for small N.
    ///
    /// Formula: `(p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`
    /// where `z = 1.96` for 95 % confidence.
    pub fn wilson(successes: usize, n: usize) -> Self {
        if n == 0 {
            return Self {
                lower: 0.0,
                upper: 1.0,
            };
        }

        const Z: f64 = 1.96; // 95 % two-tailed
        let p = successes as f64 / n as f64;
        let nf = n as f64;
        let z2 = Z * Z;

        let centre = p + z2 / (2.0 * nf);
        let margin = Z * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt();
        let denom = 1.0 + z2 / nf;

        Self {
            lower: ((centre - margin) / denom).clamp(0.0, 1.0),
            upper: ((centre + margin) / denom).clamp(0.0, 1.0),
        }
    }
}

// ── Summary statistics ────────────────────────────────────────────────────────

/// Aggregate statistics for a set of [`TrialResult`]s from the same case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationStats {
    /// Total number of trials executed.
    pub n_trials: usize,
    /// Number of trials that succeeded.
    pub successes: usize,
    /// `successes / n_trials` (0.0 when n_trials == 0).
    pub success_rate: f64,
    /// Wilson-score 95 % confidence interval around `success_rate`.
    pub confidence_interval_95: ConfidenceInterval95,
    /// Mean trial duration across all trials in milliseconds.
    pub mean_duration_ms: f64,
    /// Median (P50) trial duration in milliseconds.
    pub p50_duration_ms: f64,
    /// 95th-percentile trial duration in milliseconds.
    pub p95_duration_ms: f64,
}

impl EvaluationStats {
    /// Compute statistics from a slice of trial results.
    ///
    /// Returns `None` if `results` is empty.
    pub fn from_trials(results: &[TrialResult]) -> Option<Self> {
        let n = results.len();
        if n == 0 {
            return None;
        }

        let successes = results.iter().filter(|r| r.success).count();
        let success_rate = successes as f64 / n as f64;
        let ci = ConfidenceInterval95::wilson(successes, n);

        let mut durations: Vec<f64> = results.iter().map(|r| r.duration_ms as f64).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_duration_ms = durations.iter().sum::<f64>() / n as f64;
        let p50_duration_ms = percentile(&durations, 50.0);
        let p95_duration_ms = percentile(&durations, 95.0);

        Some(Self {
            n_trials: n,
            successes,
            success_rate,
            confidence_interval_95: ci,
            mean_duration_ms,
            p50_duration_ms,
            p95_duration_ms,
        })
    }
}

/// Compute the p-th percentile of a sorted slice (linear interpolation).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_success_builder() {
        let t = TrialResult::success(0, 42);
        assert!(t.success);
        assert_eq!(t.trial_id, 0);
        assert_eq!(t.duration_ms, 42);
        assert!(t.error.is_none());
    }

    #[test]
    fn test_trial_failure_builder() {
        let t = TrialResult::failure(1, 100, "timeout");
        assert!(!t.success);
        assert_eq!(t.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_trial_with_meta() {
        let t = TrialResult::success(0, 10)
            .with_meta("iterations", serde_json::json!(7))
            .with_meta("model", serde_json::json!("claude-sonnet"));
        assert_eq!(t.metadata["iterations"], serde_json::json!(7));
    }

    #[test]
    fn test_wilson_ci_all_successes() {
        let ci = ConfidenceInterval95::wilson(10, 10);
        assert!(
            ci.lower > 0.7,
            "lower bound should be well above 0 for 10/10"
        );
        assert!((ci.upper - 1.0).abs() < 1e-9, "upper bound should be 1.0");
    }

    #[test]
    fn test_wilson_ci_no_successes() {
        let ci = ConfidenceInterval95::wilson(0, 10);
        assert_eq!(ci.lower, 0.0);
        assert!(ci.upper < 0.3, "upper bound should be low for 0/10");
    }

    #[test]
    fn test_wilson_ci_zero_trials() {
        let ci = ConfidenceInterval95::wilson(0, 0);
        assert_eq!(ci.lower, 0.0);
        assert_eq!(ci.upper, 1.0);
    }

    #[test]
    fn test_wilson_ci_contains_true_rate() {
        // For 70 % true rate with 100 trials the CI must contain 0.70
        let ci = ConfidenceInterval95::wilson(70, 100);
        assert!(ci.lower < 0.70 && ci.upper > 0.70);
    }

    #[test]
    fn test_evaluation_stats_empty() {
        assert!(EvaluationStats::from_trials(&[]).is_none());
    }

    #[test]
    fn test_evaluation_stats_all_success() {
        let trials: Vec<_> = (0..10).map(|i| TrialResult::success(i, 100)).collect();
        let stats = EvaluationStats::from_trials(&trials).unwrap();
        assert_eq!(stats.n_trials, 10);
        assert_eq!(stats.successes, 10);
        assert!((stats.success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluation_stats_mixed() {
        let mut trials: Vec<_> = (0..7).map(|i| TrialResult::success(i, 50)).collect();
        trials.extend((7..10).map(|i| TrialResult::failure(i, 200, "err")));
        let stats = EvaluationStats::from_trials(&trials).unwrap();
        assert_eq!(stats.successes, 7);
        assert!((stats.success_rate - 0.7).abs() < 1e-9);
        assert!(stats.p95_duration_ms >= stats.p50_duration_ms);
        assert!(stats.p50_duration_ms >= stats.mean_duration_ms * 0.5);
    }

    #[test]
    fn test_percentile_single_element() {
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
    }

    #[test]
    fn test_percentile_interpolation() {
        let data = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let p50 = percentile(&data, 50.0);
        assert!((p50 - 20.0).abs() < 1e-9);
    }
}
