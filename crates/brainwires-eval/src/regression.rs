//! Regression testing infrastructure for CI integration.
//!
//! [`RegressionSuite`] compares current [`SuiteResult`] success rates against
//! stored per-category baselines.  If any category drops more than
//! [`RegressionConfig::max_regression`] below its baseline, the check fails —
//! enabling CI pipelines to gate on evaluation regressions automatically.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use brainwires_eval::{
//!     regression::{RegressionSuite, RegressionConfig, CategoryBaseline},
//!     EvaluationSuite, AlwaysPassCase,
//! };
//! use std::sync::Arc;
//!
//! // 1. Run the evaluation suite.
//! let suite = EvaluationSuite::new(30);
//! let cases = vec![Arc::new(AlwaysPassCase::new("smoke_test")) as Arc<_>];
//! let results = suite.run_suite(&cases).await;
//!
//! // 2. Build baselines from current results.
//! let mut reg = RegressionSuite::new();
//! reg.record_baselines(&results);
//!
//! // 3. On the next CI run, compare.
//! let check = reg.check(&results);
//! assert!(check.is_ci_passing());
//! ```

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::suite::SuiteResult;
use super::trial::EvaluationStats;

// ── Baseline ──────────────────────────────────────────────────────────────────

/// Per-category success-rate baseline stored for regression comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryBaseline {
    /// Category label matching [`EvaluationCase::category`](crate::case::EvaluationCase::category).
    pub category: String,
    /// Baseline success rate in [0, 1].
    pub baseline_success_rate: f64,
    /// Unix timestamp (seconds) when this baseline was recorded.
    pub measured_at_unix: i64,
    /// Number of trials used to compute this baseline.
    pub n_trials: usize,
}

impl CategoryBaseline {
    /// Create a new baseline from measured stats.
    pub fn new(category: impl Into<String>, stats: &EvaluationStats) -> Self {
        Self {
            category: category.into(),
            baseline_success_rate: stats.success_rate,
            measured_at_unix: Utc::now().timestamp(),
            n_trials: stats.n_trials,
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the regression checker.
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Maximum tolerated regression below baseline in [0, 1]. Default: 0.05 (5 %).
    pub max_regression: f64,
    /// Minimum number of trials required for a category to be checked.
    /// Categories with fewer trials are skipped (not enough data). Default: 30.
    pub min_trials: usize,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            max_regression: 0.05,
            min_trials: 30,
        }
    }
}

// ── Per-category result ───────────────────────────────────────────────────────

/// Result of a single category's regression check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryRegressionResult {
    /// Category name.
    pub category: String,
    /// Current measured success rate.
    pub current_success_rate: f64,
    /// Baseline success rate this was compared against.
    pub baseline_success_rate: f64,
    /// `baseline - current` (positive means regression, negative means improvement).
    pub regression: f64,
    /// Whether this category passed the regression threshold.
    pub passed: bool,
    /// Human-readable reason when `passed == false`.
    pub reason: Option<String>,
}

// ── Aggregate result ──────────────────────────────────────────────────────────

/// Aggregate result of a full regression check across all categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// `true` when all checked categories passed.
    pub passed: bool,
    /// Per-category breakdown.
    pub category_results: Vec<CategoryRegressionResult>,
}

impl RegressionResult {
    /// Whether all categories passed (suitable for CI gate).
    pub fn is_ci_passing(&self) -> bool {
        self.passed
    }

    /// Categories that failed the regression threshold.
    pub fn failing_categories(&self) -> Vec<&CategoryRegressionResult> {
        self.category_results.iter().filter(|r| !r.passed).collect()
    }

    /// Categories with improvements (negative regression).
    pub fn improved_categories(&self) -> Vec<&CategoryRegressionResult> {
        self.category_results
            .iter()
            .filter(|r| r.regression < 0.0)
            .collect()
    }
}

// ── RegressionSuite ───────────────────────────────────────────────────────────

/// Compares evaluation suite results against stored per-category baselines.
///
/// Fails the check if any category's success rate drops more than
/// [`RegressionConfig::max_regression`] below its baseline.
pub struct RegressionSuite {
    config: RegressionConfig,
    /// Baseline indexed by category name.
    baselines: HashMap<String, CategoryBaseline>,
}

impl Default for RegressionSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionSuite {
    /// Create a new regression suite with default configuration and no baselines.
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
            baselines: HashMap::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
        }
    }

    /// Manually register a baseline for a category.
    pub fn with_baseline(mut self, baseline: CategoryBaseline) -> Self {
        self.baselines.insert(baseline.category.clone(), baseline);
        self
    }

    /// Register a baseline from an [`EvaluationStats`] object.
    pub fn add_baseline(&mut self, category: impl Into<String>, stats: &EvaluationStats) {
        let cat = category.into();
        self.baselines
            .insert(cat.clone(), CategoryBaseline::new(cat, stats));
    }

    /// Record baselines for ALL categories present in `suite_result`.
    ///
    /// Use this to capture the current run as the new baseline.
    pub fn record_baselines(&mut self, suite_result: &SuiteResult) {
        // Aggregate stats by category (not by case name)
        let category_stats = Self::aggregate_by_category(suite_result);
        for (category, stats) in &category_stats {
            self.add_baseline(category.as_str(), stats);
        }
    }

    /// Aggregate per-case stats into per-category stats.
    ///
    /// Combines all trial results across cases sharing the same category.
    fn aggregate_by_category(suite_result: &SuiteResult) -> HashMap<String, EvaluationStats> {
        // We need to look at trial results. Build a mapping:
        // category → [all trial results from that category]
        // NOTE: SuiteResult only has case-level stats. We infer per-category
        // stats by re-aggregating from case_results using trial data.
        let mut category_trials: HashMap<String, Vec<super::trial::TrialResult>> = HashMap::new();

        // For aggregation we need to know which case belongs to which category.
        // SuiteResult stores results keyed by case name. The category is embedded
        // in EvaluationCase but not stored in SuiteResult. As a workaround, we
        // aggregate per case name as a fallback. Callers can register baselines
        // by category directly via `add_baseline`.
        for (case_name, trials) in &suite_result.case_results {
            category_trials
                .entry(case_name.clone())
                .or_default()
                .extend(trials.iter().cloned());
        }

        category_trials
            .into_iter()
            .filter_map(|(cat, trials)| {
                EvaluationStats::from_trials(&trials).map(|stats| (cat, stats))
            })
            .collect()
    }

    /// Serialize baselines to a JSON string.
    pub fn baselines_to_json(&self) -> anyhow::Result<String> {
        let list: Vec<&CategoryBaseline> = self.baselines.values().collect();
        Ok(serde_json::to_string_pretty(&list)?)
    }

    /// Returns `true` when a baseline has been recorded for `category`.
    pub fn has_baseline(&self, category: &str) -> bool {
        self.baselines.contains_key(category)
    }

    /// Retrieve the stored baseline for `category`, or `None` if absent.
    pub fn get_baseline(&self, category: &str) -> Option<&CategoryBaseline> {
        self.baselines.get(category)
    }

    /// Load baselines from a JSON string (produced by [`Self::baselines_to_json`]).
    pub fn load_baselines_from_json(json: &str) -> anyhow::Result<Self> {
        let baselines: Vec<CategoryBaseline> = serde_json::from_str(json)?;
        let mut map = HashMap::new();
        for b in baselines {
            map.insert(b.category.clone(), b);
        }
        Ok(Self {
            config: RegressionConfig::default(),
            baselines: map,
        })
    }

    /// Run the regression check against a completed [`SuiteResult`].
    ///
    /// For each category with a stored baseline:
    /// - Skip if `current_n_trials < min_trials`.
    /// - Fail if `baseline_rate - current_rate > max_regression`.
    pub fn check(&self, suite_result: &SuiteResult) -> RegressionResult {
        let current_stats = Self::aggregate_by_category(suite_result);
        let mut results = Vec::new();
        let mut all_passed = true;

        for (category, baseline) in &self.baselines {
            let Some(current) = current_stats.get(category) else {
                // Category present in baseline but absent from current run — skip.
                continue;
            };

            if current.n_trials < self.config.min_trials {
                // Not enough data — skip.
                continue;
            }

            let regression = baseline.baseline_success_rate - current.success_rate;
            let passed = regression <= self.config.max_regression;
            let reason = if !passed {
                Some(format!(
                    "category '{}' dropped {:.1}% (from {:.1}% to {:.1}%), limit is {:.1}%",
                    category,
                    regression * 100.0,
                    baseline.baseline_success_rate * 100.0,
                    current.success_rate * 100.0,
                    self.config.max_regression * 100.0,
                ))
            } else {
                None
            };

            if !passed {
                all_passed = false;
            }

            results.push(CategoryRegressionResult {
                category: category.clone(),
                current_success_rate: current.success_rate,
                baseline_success_rate: baseline.baseline_success_rate,
                regression,
                passed,
                reason,
            });
        }

        results.sort_by(|a, b| a.category.cmp(&b.category));

        RegressionResult {
            passed: all_passed,
            category_results: results,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialResult;

    fn make_stats(successes: usize, total: usize) -> EvaluationStats {
        let trials: Vec<TrialResult> = (0..total)
            .map(|i| {
                if i < successes {
                    TrialResult::success(i, 10)
                } else {
                    TrialResult::failure(i, 10, "fail")
                }
            })
            .collect();
        EvaluationStats::from_trials(&trials).unwrap()
    }

    #[test]
    fn test_baseline_creation() {
        let stats = make_stats(80, 100);
        let baseline = CategoryBaseline::new("smoke", &stats);
        assert_eq!(baseline.category, "smoke");
        assert!((baseline.baseline_success_rate - 0.8).abs() < 1e-9);
        assert_eq!(baseline.n_trials, 100);
    }

    #[test]
    fn test_check_passes_when_no_regression() {
        let stats = make_stats(80, 100);
        let mut reg = RegressionSuite::new();
        reg.add_baseline("smoke", &stats);

        // Same stats → regression = 0 → passes
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "smoke".to_string(),
                (0..100)
                    .map(|i| {
                        if i < 80 {
                            TrialResult::success(i, 10)
                        } else {
                            TrialResult::failure(i, 10, "fail")
                        }
                    })
                    .collect(),
            )]),
            stats: std::collections::HashMap::from([("smoke".to_string(), stats.clone())]),
        };

        let result = reg.check(&suite_result);
        assert!(result.is_ci_passing(), "no regression should pass");
        assert!(result.failing_categories().is_empty());
    }

    #[test]
    fn test_check_fails_on_regression_above_threshold() {
        let baseline_stats = make_stats(90, 100); // 90 %
        let mut reg = RegressionSuite::new();
        reg.add_baseline("smoke", &baseline_stats);

        // Current: 80 % — drop of 10 %, exceeds default 5 %
        let current_stats = make_stats(80, 100);
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "smoke".to_string(),
                (0..100)
                    .map(|i| {
                        if i < 80 {
                            TrialResult::success(i, 10)
                        } else {
                            TrialResult::failure(i, 10, "fail")
                        }
                    })
                    .collect(),
            )]),
            stats: std::collections::HashMap::from([("smoke".to_string(), current_stats)]),
        };

        let result = reg.check(&suite_result);
        assert!(!result.is_ci_passing(), "10% regression should fail CI");
        assert_eq!(result.failing_categories().len(), 1);
        let failing = &result.failing_categories()[0];
        assert!((failing.regression - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_check_passes_regression_within_threshold() {
        let baseline_stats = make_stats(90, 100); // 90 %
        let config = RegressionConfig {
            max_regression: 0.10, // 10 % allowed
            min_trials: 30,
        };
        let mut reg = RegressionSuite::with_config(config);
        reg.add_baseline("smoke", &baseline_stats);

        // Current: 82 % — drop of 8 %, within 10 %
        let current_stats = make_stats(82, 100);
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "smoke".to_string(),
                (0..100)
                    .map(|i| {
                        if i < 82 {
                            TrialResult::success(i, 10)
                        } else {
                            TrialResult::failure(i, 10, "fail")
                        }
                    })
                    .collect(),
            )]),
            stats: std::collections::HashMap::from([("smoke".to_string(), current_stats)]),
        };

        let result = reg.check(&suite_result);
        assert!(
            result.is_ci_passing(),
            "8% drop within 10% threshold should pass"
        );
    }

    #[test]
    fn test_check_skips_low_trial_count() {
        let baseline_stats = make_stats(90, 100); // 90 %
        let mut reg = RegressionSuite::new(); // min_trials=30
        reg.add_baseline("smoke", &baseline_stats);

        // Only 5 trials — below min_trials
        let current_stats = make_stats(0, 5); // 0 %
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "smoke".to_string(),
                (0..5)
                    .map(|i| TrialResult::failure(i, 10, "fail"))
                    .collect(),
            )]),
            stats: std::collections::HashMap::from([("smoke".to_string(), current_stats)]),
        };

        // Should be skipped due to insufficient trials
        let result = reg.check(&suite_result);
        assert!(result.is_ci_passing(), "low trial count should be skipped");
        assert!(result.category_results.is_empty());
    }

    #[test]
    fn test_json_roundtrip() {
        let stats = make_stats(75, 100);
        let mut reg = RegressionSuite::new();
        reg.add_baseline("smoke", &stats);

        let json = reg.baselines_to_json().unwrap();
        let loaded = RegressionSuite::load_baselines_from_json(&json).unwrap();
        let baseline = loaded.baselines.get("smoke").unwrap();
        assert!((baseline.baseline_success_rate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_record_baselines_from_suite_result() {
        let trials: Vec<TrialResult> = (0..50)
            .map(|i| {
                if i < 40 {
                    TrialResult::success(i, 10)
                } else {
                    TrialResult::failure(i, 10, "fail")
                }
            })
            .collect();
        let stats = EvaluationStats::from_trials(&trials).unwrap();
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "my_case".to_string(),
                trials.clone(),
            )]),
            stats: std::collections::HashMap::from([("my_case".to_string(), stats)]),
        };

        let mut reg = RegressionSuite::new();
        reg.record_baselines(&suite_result);

        assert!(reg.baselines.contains_key("my_case"));
        let b = &reg.baselines["my_case"];
        assert!((b.baseline_success_rate - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_improved_categories() {
        let baseline_stats = make_stats(70, 100); // 70 %
        let mut reg = RegressionSuite::new();
        reg.add_baseline("smoke", &baseline_stats);

        // Current: 90 % — improvement
        let suite_result = SuiteResult {
            case_results: std::collections::HashMap::from([(
                "smoke".to_string(),
                (0..100)
                    .map(|i| {
                        if i < 90 {
                            TrialResult::success(i, 10)
                        } else {
                            TrialResult::failure(i, 10, "fail")
                        }
                    })
                    .collect(),
            )]),
            stats: std::collections::HashMap::from([("smoke".to_string(), make_stats(90, 100))]),
        };

        let result = reg.check(&suite_result);
        assert!(result.is_ci_passing());
        assert_eq!(result.improved_categories().len(), 1);
        assert!(result.improved_categories()[0].regression < 0.0);
    }
}
