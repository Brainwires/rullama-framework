//! Fault classification for eval-driven autonomous self-improvement.
//!
//! [`analyze_suite_for_faults`] inspects a completed [`SuiteResult`] and
//! classifies per-case issues into [`FaultReport`]s sorted by priority.
//! The `suggested_task_description` on each report is ready to pass directly
//! to a `SelfImprovementController` (from `brainwires-autonomy`) as a task
//! description.
//!
//! # Classification precedence (first match wins per case)
//!
//! 1. **Regression** — a baseline exists and `current < baseline − 0.03`.
//! 2. **ConsistentFailure** — `success_rate < consistent_failure_threshold`.
//! 3. **Flaky** — CI width > `flaky_ci_threshold`.
//! 4. **NewCapability** — no baseline recorded yet and `success_rate ≥ 0.8`
//!    (capture it before a future regression goes unnoticed).

use super::regression::RegressionSuite;
use super::suite::SuiteResult;

// ── FaultKind ─────────────────────────────────────────────────────────────────

/// The classification of a detected eval fault.
#[derive(Debug, Clone)]
pub enum FaultKind {
    /// A previously-passing case dropped in success rate below the regression
    /// tolerance.
    Regression {
        /// The baseline success rate before the regression.
        previous_rate: f64,
        /// The current success rate after the regression.
        current_rate: f64,
        /// The absolute drop in success rate (`previous - current`).
        drop: f64,
    },
    /// A new case (no baseline exists) showing stable high success — worth
    /// capturing as a baseline before future regressions sneak in.
    NewCapability {
        /// Human-readable description of the new capability.
        description: String,
    },
    /// Success rate is below the minimum acceptable threshold regardless of
    /// any stored baseline.
    ConsistentFailure {
        /// The observed success rate.
        success_rate: f64,
    },
    /// Wide confidence interval — the case result is highly non-deterministic.
    Flaky {
        /// Mean success rate across trials.
        mean_rate: f64,
        /// Width of the 95% confidence interval.
        ci_width: f64,
    },
}

impl FaultKind {
    /// Scheduling priority (higher = more urgent, max 10).
    ///
    /// | Variant | Priority |
    /// |---------|----------|
    /// | `Regression` | 1–10 scaled by drop % (1 pp → 1 pt, capped at 10) |
    /// | `ConsistentFailure` | 8 |
    /// | `NewCapability` | 5 |
    /// | `Flaky` | 4 |
    pub fn priority(&self) -> u8 {
        match self {
            FaultKind::Regression { drop, .. } => {
                let scaled = (*drop * 100.0).round() as u8;
                scaled.clamp(1, 10)
            }
            FaultKind::ConsistentFailure { .. } => 8,
            FaultKind::NewCapability { .. } => 5,
            FaultKind::Flaky { .. } => 4,
        }
    }

    /// Short label for use in reports and task IDs.
    pub fn label(&self) -> &'static str {
        match self {
            FaultKind::Regression { .. } => "regression",
            FaultKind::NewCapability { .. } => "new_capability",
            FaultKind::ConsistentFailure { .. } => "consistent_failure",
            FaultKind::Flaky { .. } => "flaky",
        }
    }
}

// ── FaultReport ───────────────────────────────────────────────────────────────

/// A classified eval fault ready for self-improvement task generation.
#[derive(Debug, Clone)]
pub struct FaultReport {
    /// Eval case name (matches [`EvaluationCase::name`](crate::case::EvaluationCase::name)).
    pub case_name: String,
    /// Category from [`EvaluationCase::category`](crate::case::EvaluationCase::category), or the case name if unknown.
    pub category: String,
    /// Classification and relevant rates.
    pub fault_kind: FaultKind,
    /// Up to 3 sample error strings from failed trials.
    pub sample_errors: Vec<String>,
    /// Number of failed trials in the current run.
    pub n_failures: usize,
    /// Total trials in the current run.
    pub n_trials: usize,
    /// Human-readable task description for the self-improvement controller.
    pub suggested_task_description: String,
}

impl FaultReport {
    /// Construct a regression fault.
    pub fn regression(
        case_name: impl Into<String>,
        category: impl Into<String>,
        previous_rate: f64,
        current_rate: f64,
        sample_errors: Vec<String>,
        n_failures: usize,
        n_trials: usize,
    ) -> Self {
        let drop = (previous_rate - current_rate).max(0.0);
        let cn = case_name.into();
        let suggested = format!(
            "Fix regression in eval case '{}': success rate dropped from {:.0}% to {:.0}% \
             (drop: {:.0}%). Investigate recent changes and restore reliability.",
            cn,
            previous_rate * 100.0,
            current_rate * 100.0,
            drop * 100.0,
        );
        Self {
            case_name: cn,
            category: category.into(),
            fault_kind: FaultKind::Regression {
                previous_rate,
                current_rate,
                drop,
            },
            sample_errors,
            n_failures,
            n_trials,
            suggested_task_description: suggested,
        }
    }

    /// Construct a new-capability fault (no prior baseline recorded).
    pub fn new_capability(
        case_name: impl Into<String>,
        category: impl Into<String>,
        description: impl Into<String>,
        success_rate: f64,
        n_failures: usize,
        n_trials: usize,
    ) -> Self {
        let cn = case_name.into();
        let desc = description.into();
        let suggested = format!(
            "Record baseline for newly-observed eval case '{}' ({:.0}% success rate). \
             Add documentation and verify the capability is tested consistently.",
            cn,
            success_rate * 100.0,
        );
        Self {
            case_name: cn,
            category: category.into(),
            fault_kind: FaultKind::NewCapability { description: desc },
            sample_errors: Vec::new(),
            n_failures,
            n_trials,
            suggested_task_description: suggested,
        }
    }

    /// Priority derived from the fault kind (delegates to [`FaultKind::priority`]).
    pub fn priority(&self) -> u8 {
        self.fault_kind.priority()
    }
}

// ── analyze_suite_for_faults ──────────────────────────────────────────────────

/// Inspect a [`SuiteResult`] and return classified [`FaultReport`]s.
///
/// **Classification precedence** (first match wins per case):
///
/// 1. **Regression** — baseline exists _and_ `current < baseline − 0.03`.
/// 2. **ConsistentFailure** — `success_rate < consistent_failure_threshold`.
/// 3. **Flaky** — CI width > `flaky_ci_threshold`.
/// 4. **NewCapability** — no baseline recorded yet, `success_rate ≥ 0.8`.
///
/// The returned `Vec` is sorted by [`FaultReport::priority`] descending.
///
/// # Parameters
///
/// | Parameter | Default | Meaning |
/// |-----------|---------|---------|
/// | `consistent_failure_threshold` | 0.2 | Success rates below this are always a fault |
/// | `flaky_ci_threshold` | 0.25 | CI widths above this indicate high variance |
pub fn analyze_suite_for_faults(
    suite_result: &SuiteResult,
    regression_suite: Option<&RegressionSuite>,
    consistent_failure_threshold: f64,
    flaky_ci_threshold: f64,
) -> Vec<FaultReport> {
    let mut reports: Vec<FaultReport> = Vec::new();

    for (case_name, stats) in &suite_result.stats {
        let n_trials = stats.n_trials;
        let n_failures = n_trials - stats.successes;
        let success_rate = stats.success_rate;
        let ci_width = stats.confidence_interval_95.upper - stats.confidence_interval_95.lower;

        // Sample up to 3 error messages from failed trials.
        let sample_errors: Vec<String> = suite_result
            .case_results
            .get(case_name)
            .map(|trials| {
                trials
                    .iter()
                    .filter_map(|t| t.error.clone())
                    .take(3)
                    .collect()
            })
            .unwrap_or_default();

        // Look up any stored baseline for this case.
        let baseline = regression_suite.and_then(|rs| rs.get_baseline(case_name));

        // 1. Regression.
        if let Some(b) = baseline {
            let drop = b.baseline_success_rate - success_rate;
            if drop > 0.03 {
                reports.push(FaultReport::regression(
                    case_name,
                    case_name,
                    b.baseline_success_rate,
                    success_rate,
                    sample_errors,
                    n_failures,
                    n_trials,
                ));
                continue;
            }
        }

        // 2. Consistent failure.
        if success_rate < consistent_failure_threshold {
            let suggested = format!(
                "Fix consistently failing eval case '{}' (success rate: {:.0}%). \
                 Review the implementation and ensure the evaluated functionality \
                 works correctly.",
                case_name,
                success_rate * 100.0,
            );
            reports.push(FaultReport {
                case_name: case_name.clone(),
                category: case_name.clone(),
                fault_kind: FaultKind::ConsistentFailure { success_rate },
                sample_errors,
                n_failures,
                n_trials,
                suggested_task_description: suggested,
            });
            continue;
        }

        // 3. Flaky (wide CI → high variance).
        // Require at least one failure: zero-failure runs can't be flaky by
        // definition, and small-N all-pass runs would otherwise produce
        // spuriously wide Wilson CIs.
        if n_failures > 0 && ci_width > flaky_ci_threshold {
            let suggested = format!(
                "Stabilize flaky eval case '{}' (mean success: {:.0}%, CI width: {:.2}). \
                 Investigate sources of non-determinism and improve consistency.",
                case_name,
                success_rate * 100.0,
                ci_width,
            );
            reports.push(FaultReport {
                case_name: case_name.clone(),
                category: case_name.clone(),
                fault_kind: FaultKind::Flaky {
                    mean_rate: success_rate,
                    ci_width,
                },
                sample_errors,
                n_failures,
                n_trials,
                suggested_task_description: suggested,
            });
            continue;
        }

        // 4. New capability (regression suite present but no baseline for this case).
        if baseline.is_none() && regression_suite.is_some() && success_rate >= 0.8 {
            reports.push(FaultReport::new_capability(
                case_name,
                case_name,
                format!(
                    "New eval case '{}' achieving {:.0}% success — baseline not yet recorded",
                    case_name,
                    success_rate * 100.0,
                ),
                success_rate,
                n_failures,
                n_trials,
            ));
        }
    }

    // Sort by priority descending.
    reports.sort_by_key(|b| std::cmp::Reverse(b.priority()));
    reports
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::RegressionSuite;
    use crate::suite::SuiteResult;
    use crate::trial::{EvaluationStats, TrialResult};
    use std::collections::HashMap;

    fn make_suite_result(case_name: &str, successes: usize, total: usize) -> SuiteResult {
        let trials: Vec<TrialResult> = (0..total)
            .map(|i| {
                if i < successes {
                    TrialResult::success(i, 1)
                } else {
                    TrialResult::failure(i, 1, format!("error_{i}"))
                }
            })
            .collect();
        let stats = EvaluationStats::from_trials(&trials).unwrap();
        SuiteResult {
            case_results: HashMap::from([(case_name.to_string(), trials)]),
            stats: HashMap::from([(case_name.to_string(), stats)]),
        }
    }

    // ── FaultKind priority ─────────────────────────────────────────────────

    #[test]
    fn test_priority_regression_scaled_by_drop() {
        // 5 pp drop → priority 5
        let fk = FaultKind::Regression {
            previous_rate: 0.9,
            current_rate: 0.85,
            drop: 0.05,
        };
        assert_eq!(fk.priority(), 5);
    }

    #[test]
    fn test_priority_regression_capped_at_10() {
        // 25 pp drop → capped at 10
        let fk = FaultKind::Regression {
            previous_rate: 1.0,
            current_rate: 0.75,
            drop: 0.25,
        };
        assert_eq!(fk.priority(), 10);
    }

    #[test]
    fn test_priority_consistent_failure() {
        assert_eq!(
            FaultKind::ConsistentFailure { success_rate: 0.1 }.priority(),
            8
        );
    }

    #[test]
    fn test_priority_new_capability() {
        assert_eq!(
            FaultKind::NewCapability {
                description: "x".into()
            }
            .priority(),
            5
        );
    }

    #[test]
    fn test_priority_flaky() {
        assert_eq!(
            FaultKind::Flaky {
                mean_rate: 0.5,
                ci_width: 0.3
            }
            .priority(),
            4
        );
    }

    // ── FaultReport constructors ───────────────────────────────────────────

    #[test]
    fn test_regression_constructor_sets_fields() {
        let report =
            FaultReport::regression("my_case", "smoke", 0.9, 0.7, vec!["err1".into()], 3, 10);
        assert_eq!(report.case_name, "my_case");
        assert_eq!(report.category, "smoke");
        assert_eq!(report.n_failures, 3);
        assert_eq!(report.n_trials, 10);
        assert!(report.suggested_task_description.contains("my_case"));
        assert!(report.suggested_task_description.contains("regression"));
        match &report.fault_kind {
            FaultKind::Regression {
                drop,
                previous_rate,
                current_rate,
            } => {
                assert!((*drop - 0.2).abs() < 1e-9);
                assert!((*previous_rate - 0.9).abs() < 1e-9);
                assert!((*current_rate - 0.7).abs() < 1e-9);
            }
            _ => panic!("expected Regression variant"),
        }
    }

    #[test]
    fn test_new_capability_constructor() {
        let report = FaultReport::new_capability("new_case", "cat", "desc", 0.85, 1, 10);
        assert_eq!(report.case_name, "new_case");
        assert!(matches!(report.fault_kind, FaultKind::NewCapability { .. }));
        assert!(report.sample_errors.is_empty());
    }

    // ── analyze_suite_for_faults ───────────────────────────────────────────

    #[test]
    fn test_consistent_failure_detected() {
        let result = make_suite_result("bad_case", 1, 20); // 5% success
        let reports = analyze_suite_for_faults(&result, None, 0.2, 0.25);
        assert_eq!(reports.len(), 1);
        assert!(
            matches!(reports[0].fault_kind, FaultKind::ConsistentFailure { .. }),
            "expected ConsistentFailure"
        );
        assert_eq!(reports[0].case_name, "bad_case");
    }

    #[test]
    fn test_regression_detected_when_drop_exceeds_tolerance() {
        let result = make_suite_result("my_case", 7, 10); // 70%
        let mut reg = RegressionSuite::new();
        // Baseline was 90% → 20 pp drop, well above 3 pp tolerance.
        let baseline_trials: Vec<TrialResult> = (0..10)
            .map(|i| {
                if i < 9 {
                    TrialResult::success(i, 1)
                } else {
                    TrialResult::failure(i, 1, "e")
                }
            })
            .collect();
        let baseline_stats = EvaluationStats::from_trials(&baseline_trials).unwrap();
        reg.add_baseline("my_case", &baseline_stats);

        let reports = analyze_suite_for_faults(&result, Some(&reg), 0.2, 0.25);
        assert!(
            reports
                .iter()
                .any(|r| matches!(r.fault_kind, FaultKind::Regression { .. })),
            "expected Regression fault"
        );
    }

    #[test]
    fn test_no_fault_when_within_tolerance() {
        // 88% success, baseline 90% → drop 2 pp ≤ 3 pp tolerance.
        let result = make_suite_result("ok_case", 88, 100);
        let mut reg = RegressionSuite::new();
        let baseline_trials: Vec<TrialResult> = (0..100)
            .map(|i| {
                if i < 90 {
                    TrialResult::success(i, 1)
                } else {
                    TrialResult::failure(i, 1, "e")
                }
            })
            .collect();
        let baseline_stats = EvaluationStats::from_trials(&baseline_trials).unwrap();
        reg.add_baseline("ok_case", &baseline_stats);

        let reports = analyze_suite_for_faults(&result, Some(&reg), 0.2, 0.25);
        assert!(
            reports.is_empty(),
            "2 pp drop within 3 pp tolerance should produce no fault"
        );
    }

    #[test]
    fn test_no_fault_for_passing_case_without_regression_suite() {
        // Use 50 trials at 90% to get a narrow Wilson CI (width ~0.17 < 0.25).
        let result = make_suite_result("good_case", 45, 50);
        let reports = analyze_suite_for_faults(&result, None, 0.2, 0.25);
        assert!(reports.is_empty());
    }

    #[test]
    fn test_new_capability_when_regression_suite_provided_but_no_matching_baseline() {
        // 50 trials at 90% → CI width ~0.17 < 0.25 → not Flaky → reaches NewCapability check.
        let result = make_suite_result("new_case", 45, 50);
        let reg = RegressionSuite::new(); // empty — no baseline for "new_case"
        let reports = analyze_suite_for_faults(&result, Some(&reg), 0.2, 0.25);
        assert!(
            reports
                .iter()
                .any(|r| matches!(r.fault_kind, FaultKind::NewCapability { .. })),
            "should report NewCapability for high-success case with no baseline"
        );
    }

    #[test]
    fn test_results_sorted_by_priority_descending() {
        let mut case_results = HashMap::new();
        let mut stats_map = HashMap::new();

        // consistent_failure case: 10% → priority 8
        let bad: Vec<TrialResult> = (0..10)
            .map(|i| {
                if i < 1 {
                    TrialResult::success(i, 1)
                } else {
                    TrialResult::failure(i, 1, "e")
                }
            })
            .collect();
        stats_map.insert(
            "bad".to_string(),
            EvaluationStats::from_trials(&bad).unwrap(),
        );
        case_results.insert("bad".to_string(), bad);

        // flaky case: 50% with 10 trials — CI width ~ 0.52 > 0.25 → priority 4
        let flaky: Vec<TrialResult> = (0..10)
            .map(|i| {
                if i < 5 {
                    TrialResult::success(i, 1)
                } else {
                    TrialResult::failure(i, 1, "e")
                }
            })
            .collect();
        stats_map.insert(
            "flaky".to_string(),
            EvaluationStats::from_trials(&flaky).unwrap(),
        );
        case_results.insert("flaky".to_string(), flaky);

        let result = SuiteResult {
            case_results,
            stats: stats_map,
        };
        let reports = analyze_suite_for_faults(&result, None, 0.2, 0.25);

        assert!(reports.len() >= 2);
        for i in 0..reports.len() - 1 {
            assert!(
                reports[i].priority() >= reports[i + 1].priority(),
                "reports should be sorted by priority desc"
            );
        }
    }

    #[test]
    fn test_sample_errors_collected() {
        let result = make_suite_result("broken", 0, 5); // all fail
        let reports = analyze_suite_for_faults(&result, None, 0.2, 0.25);
        assert!(!reports.is_empty());
        // Up to 3 sample errors should be present
        assert!(!reports[0].sample_errors.is_empty());
        assert!(reports[0].sample_errors.len() <= 3);
    }
}
