//! Evaluation suite — N-trial Monte Carlo runner.
//!
//! [`EvaluationSuite`] runs each registered [`EvaluationCase`] N times,
//! collects [`TrialResult`]s, and computes [`EvaluationStats`] for every case.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::case::EvaluationCase;
use super::trial::{EvaluationStats, TrialResult};

// ── Suite result ──────────────────────────────────────────────────────────────

/// Aggregated results for all cases in a suite run.
#[derive(Debug, Serialize, Deserialize)]
pub struct SuiteResult {
    /// Raw trial results keyed by case name.
    pub case_results: HashMap<String, Vec<TrialResult>>,
    /// Summary statistics keyed by case name.
    pub stats: HashMap<String, EvaluationStats>,
}

impl SuiteResult {
    /// Overall success rate across *all* cases and trials.
    pub fn overall_success_rate(&self) -> f64 {
        let total: usize = self.case_results.values().map(|v| v.len()).sum();
        if total == 0 {
            return 0.0;
        }
        let successes: usize = self
            .case_results
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| r.success)
            .count();
        successes as f64 / total as f64
    }

    /// Returns all cases whose success rate is strictly below `threshold`.
    pub fn failing_cases(&self, threshold: f64) -> Vec<&str> {
        self.stats
            .iter()
            .filter(|(_, s)| s.success_rate < threshold)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// ── Suite configuration ───────────────────────────────────────────────────────

/// Configuration for [`EvaluationSuite`].
#[derive(Debug, Clone)]
pub struct SuiteConfig {
    /// Number of times each case is run.  Minimum 1.
    pub n_trials: usize,
    /// Maximum number of trials to execute concurrently per case.
    /// `1` means sequential execution (deterministic ordering).
    pub max_parallel: usize,
    /// If `true`, a single trial error (not a test failure, but a hard Rust
    /// error) is treated as a test failure rather than propagating to the
    /// caller.
    pub catch_errors_as_failures: bool,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        Self {
            n_trials: 10,
            max_parallel: 1,
            catch_errors_as_failures: true,
        }
    }
}

// ── Suite ─────────────────────────────────────────────────────────────────────

/// N-trial Monte Carlo evaluation runner.
///
/// ## Quick start
/// ```rust,ignore
/// use brainwires_eval::{EvaluationSuite, AlwaysPassCase};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let suite = EvaluationSuite::new(30);
///     let case = Arc::new(AlwaysPassCase::new("smoke"));
///     let results = suite.run_suite(&[case]).await;
///     println!("overall: {:.1}%", results.overall_success_rate() * 100.0);
/// }
/// ```
pub struct EvaluationSuite {
    config: SuiteConfig,
}

impl EvaluationSuite {
    /// Create a suite that runs each case `n_trials` times sequentially.
    pub fn new(n_trials: usize) -> Self {
        Self {
            config: SuiteConfig {
                n_trials: n_trials.max(1),
                ..SuiteConfig::default()
            },
        }
    }

    /// Override the full configuration.
    pub fn with_config(config: SuiteConfig) -> Self {
        Self { config }
    }

    /// Run `n_trials` for a single case and return the raw results.
    ///
    /// Accepts `Arc<dyn EvaluationCase>` so the case can be shared across
    /// parallel async tasks when `max_parallel > 1`.
    pub async fn run_case(&self, case: Arc<dyn EvaluationCase>) -> Vec<TrialResult> {
        let mut results = Vec::with_capacity(self.config.n_trials);

        if self.config.max_parallel <= 1 {
            // Sequential
            for trial_id in 0..self.config.n_trials {
                let result = case.run(trial_id).await;
                results.push(self.resolve(result, trial_id));
            }
        } else {
            // Bounded parallel execution using a semaphore
            use tokio::sync::Semaphore;
            let sem = Arc::new(Semaphore::new(self.config.max_parallel));
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TrialResult>();

            for trial_id in 0..self.config.n_trials {
                let permit = sem.clone().acquire_owned().await.unwrap();
                let tx = tx.clone();
                let case_arc = Arc::clone(&case);
                let catch_errors = self.config.catch_errors_as_failures;

                tokio::spawn(async move {
                    let _permit = permit;
                    let result = case_arc.run(trial_id).await;
                    let trial = match result {
                        Ok(t) => t,
                        Err(e) if catch_errors => TrialResult::failure(trial_id, 0, e.to_string()),
                        Err(e) => {
                            tracing::error!(
                                "Trial {} errored (catch_errors_as_failures=false): {}",
                                trial_id,
                                e
                            );
                            TrialResult::failure(trial_id, 0, format!("Trial errored: {e}"))
                        }
                    };
                    if tx.send(trial).is_err() {
                        tracing::warn!("Trial {} result dropped: receiver closed", trial_id);
                    }
                });
            }
            drop(tx);

            // Drain the channel once all producers have finished.
            while let Some(t) = rx.recv().await {
                results.push(t);
            }

            // Sort by trial_id for deterministic output order.
            results.sort_by_key(|r| r.trial_id);
        }

        results
    }

    /// Run the full suite: execute each case N times and return aggregated results.
    pub async fn run_suite(&self, cases: &[Arc<dyn EvaluationCase>]) -> SuiteResult {
        let mut case_results: HashMap<String, Vec<TrialResult>> = HashMap::new();
        let mut stats: HashMap<String, EvaluationStats> = HashMap::new();

        for case in cases {
            let results = self.run_case(Arc::clone(case)).await;
            let case_stats =
                EvaluationStats::from_trials(&results).expect("case must have at least one trial");
            let name = case.name().to_string();
            tracing::info!(
                case = %name,
                n = case_stats.n_trials,
                success_rate = %format!("{:.1}%", case_stats.success_rate * 100.0),
                ci_low = %format!("{:.3}", case_stats.confidence_interval_95.lower),
                ci_high = %format!("{:.3}", case_stats.confidence_interval_95.upper),
                "EvaluationSuite: case complete"
            );
            case_results.insert(name.clone(), results);
            stats.insert(name, case_stats);
        }

        SuiteResult {
            case_results,
            stats,
        }
    }

    /// Resolve a `Result<TrialResult>` from a case run into a `TrialResult`,
    /// converting errors into failures when `catch_errors_as_failures` is set.
    fn resolve(&self, result: Result<TrialResult>, trial_id: usize) -> TrialResult {
        match result {
            Ok(t) => t,
            Err(e) if self.config.catch_errors_as_failures => {
                TrialResult::failure(trial_id, 0, e.to_string())
            }
            Err(e) => {
                tracing::error!(
                    "Trial {} errored (catch_errors_as_failures=false): {}",
                    trial_id,
                    e
                );
                TrialResult::failure(trial_id, 0, format!("Trial errored: {e}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{AlwaysFailCase, AlwaysPassCase, StochasticCase};

    #[tokio::test]
    async fn test_suite_all_pass() {
        let suite = EvaluationSuite::new(5);
        let case = Arc::new(AlwaysPassCase::new("ok"));
        let result = suite.run_suite(&[case]).await;

        let stats = result.stats.get("ok").unwrap();
        assert_eq!(stats.n_trials, 5);
        assert_eq!(stats.successes, 5);
        assert!((stats.success_rate - 1.0).abs() < 1e-9);
        assert!((result.overall_success_rate() - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_suite_all_fail() {
        let suite = EvaluationSuite::new(3);
        let case = Arc::new(AlwaysFailCase::new("bad", "expected"));
        let result = suite.run_suite(&[case]).await;

        let stats = result.stats.get("bad").unwrap();
        assert_eq!(stats.successes, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[tokio::test]
    async fn test_suite_multiple_cases() {
        let suite = EvaluationSuite::new(10);
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![
            Arc::new(AlwaysPassCase::new("pass")),
            Arc::new(AlwaysFailCase::new("fail", "x")),
        ];
        let result = suite.run_suite(&cases).await;
        assert!(result.stats.contains_key("pass"));
        assert!(result.stats.contains_key("fail"));
        assert!((result.overall_success_rate() - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_suite_n_trials_minimum_one() {
        let suite = EvaluationSuite::new(0); // Should clamp to 1
        let case = Arc::new(AlwaysPassCase::new("x"));
        let result = suite.run_suite(&[case]).await;
        assert_eq!(result.stats["x"].n_trials, 1);
    }

    #[tokio::test]
    async fn test_run_case_returns_correct_count() {
        let suite = EvaluationSuite::new(7);
        let case = Arc::new(AlwaysPassCase::new("seven"));
        let results = suite.run_case(case).await;
        assert_eq!(results.len(), 7);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.trial_id, i);
        }
    }

    #[tokio::test]
    async fn test_failing_cases_filter() {
        let suite = EvaluationSuite::new(10);
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![
            Arc::new(AlwaysPassCase::new("good")),
            Arc::new(StochasticCase::new("flaky", 0.0)), // always fails
        ];
        let result = suite.run_suite(&cases).await;
        let failing = result.failing_cases(0.5);
        assert!(
            failing.contains(&"flaky"),
            "flaky should be in failing list"
        );
        assert!(
            !failing.contains(&"good"),
            "good should not be in failing list"
        );
    }

    #[tokio::test]
    async fn test_confidence_interval_in_suite_result() {
        let suite = EvaluationSuite::new(50);
        let case = Arc::new(StochasticCase::new("ci_test", 0.8));
        let result = suite.run_suite(&[case]).await;
        let stats = &result.stats["ci_test"];
        let ci = stats.confidence_interval_95;
        // With ~40/50 successes the 95 % CI should comfortably contain 0.8
        assert!(ci.lower < 0.85 && ci.upper > 0.65);
    }
}
