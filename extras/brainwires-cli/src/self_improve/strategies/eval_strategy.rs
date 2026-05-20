//! Eval-driven improvement strategy.
//!
//! [`EvalStrategy`] implements [`ImprovementStrategy`] by running an
//! [`EvaluationSuite`] and converting detected [`FaultReport`]s into
//! [`ImprovementTask`]s.  Attach it to a [`TaskGenerator`] to get
//! eval-driven tasks alongside the standard code-quality strategies.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use brainwires_eval::fault_report::analyze_suite_for_faults;
use brainwires_eval::{EvaluationCase, EvaluationSuite, RegressionSuite, SuiteConfig};

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::self_improve::config::StrategyConfig;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`EvalStrategy`].
#[derive(Debug, Clone)]
pub struct EvalStrategyConfig {
    /// Number of trials per eval case per run.  Default: 10.
    pub n_trials: usize,
    /// Path to load existing regression baselines JSON from.
    /// `None` disables baseline comparison.
    pub baselines_path: Option<String>,
    /// Cases with success rate below this are classified as `ConsistentFailure`.
    /// Default: 0.2.
    pub consistent_failure_threshold: f64,
    /// Cases whose CI width exceeds this are classified as `Flaky`.
    /// Default: 0.25.
    pub flaky_ci_threshold: f64,
    /// Maximum number of improvement tasks to generate from faults.
    /// Default: 5.
    pub max_tasks: usize,
}

impl Default for EvalStrategyConfig {
    fn default() -> Self {
        Self {
            n_trials: 10,
            baselines_path: None,
            consistent_failure_threshold: 0.2,
            flaky_ci_threshold: 0.25,
            max_tasks: 5,
        }
    }
}

// ── EvalStrategy ──────────────────────────────────────────────────────────────

/// An [`ImprovementStrategy`] that runs an eval suite and converts detected
/// faults into improvement tasks.
pub struct EvalStrategy {
    cases: Vec<Arc<dyn EvaluationCase>>,
    eval_config: EvalStrategyConfig,
}

impl EvalStrategy {
    /// Create a new eval strategy with the given cases and configuration.
    pub fn new(cases: Vec<Arc<dyn EvaluationCase>>, eval_config: EvalStrategyConfig) -> Self {
        Self { cases, eval_config }
    }
}

#[async_trait]
impl ImprovementStrategy for EvalStrategy {
    fn name(&self) -> &str {
        "eval_driven"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::EvalDriven
    }

    async fn generate_tasks(
        &self,
        _repo_path: &str,
        _config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let suite = EvaluationSuite::with_config(SuiteConfig {
            n_trials: self.eval_config.n_trials,
            ..SuiteConfig::default()
        });

        let result = suite.run_suite(&self.cases).await;

        // Load baselines if a path is configured.
        let regression_suite = self
            .eval_config
            .baselines_path
            .as_ref()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|json| RegressionSuite::load_baselines_from_json(&json).ok());

        let faults = analyze_suite_for_faults(
            &result,
            regression_suite.as_ref(),
            self.eval_config.consistent_failure_threshold,
            self.eval_config.flaky_ci_threshold,
        );

        let tasks = faults
            .into_iter()
            .take(self.eval_config.max_tasks)
            .enumerate()
            .map(|(i, fault)| ImprovementTask {
                id: format!("eval-{}-{}", fault.fault_kind.label(), i),
                strategy: "eval_driven".to_string(),
                category: ImprovementCategory::EvalDriven,
                description: fault.suggested_task_description.clone(),
                target_files: Vec::new(),
                priority: fault.priority(),
                estimated_diff_lines: 50,
                context: format!(
                    "Eval case: {} | Fault type: {} | Failures: {}/{}",
                    fault.case_name,
                    fault.fault_kind.label(),
                    fault.n_failures,
                    fault.n_trials,
                ),
            })
            .collect();

        Ok(tasks)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_eval::{AlwaysFailCase, AlwaysPassCase};

    #[tokio::test]
    async fn test_generates_tasks_for_consistently_failing_cases() {
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![Arc::new(AlwaysFailCase::new(
            "always_broken",
            "intentional failure",
        ))];
        let config = EvalStrategyConfig {
            n_trials: 5,
            max_tasks: 10,
            ..Default::default()
        };
        let strategy = EvalStrategy::new(cases, config);
        let tasks = strategy
            .generate_tasks(".", &StrategyConfig::default())
            .await
            .unwrap();

        assert!(!tasks.is_empty(), "should generate tasks for failing cases");
        assert!(
            tasks[0].description.contains("always_broken"),
            "task description should mention the case name; got: {}",
            tasks[0].description
        );
        assert_eq!(tasks[0].category, ImprovementCategory::EvalDriven);
        assert_eq!(tasks[0].strategy, "eval_driven");
    }

    #[tokio::test]
    async fn test_no_tasks_for_all_passing_cases_without_regression_suite() {
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![Arc::new(AlwaysPassCase::new("always_ok"))];
        let config = EvalStrategyConfig {
            n_trials: 5,
            baselines_path: None, // no regression suite → no NewCapability fault
            ..Default::default()
        };
        let strategy = EvalStrategy::new(cases, config);
        let tasks = strategy
            .generate_tasks(".", &StrategyConfig::default())
            .await
            .unwrap();

        assert!(
            tasks.is_empty(),
            "no tasks expected for fully passing cases without a regression suite"
        );
    }

    #[tokio::test]
    async fn test_max_tasks_respected() {
        let cases: Vec<Arc<dyn EvaluationCase>> = (0..10)
            .map(|i| {
                Arc::new(AlwaysFailCase::new(format!("case_{i}"), "fail"))
                    as Arc<dyn EvaluationCase>
            })
            .collect();

        let config = EvalStrategyConfig {
            n_trials: 3,
            max_tasks: 3,
            ..Default::default()
        };
        let strategy = EvalStrategy::new(cases, config);
        let tasks = strategy
            .generate_tasks(".", &StrategyConfig::default())
            .await
            .unwrap();

        assert!(tasks.len() <= 3, "max_tasks=3 should cap output at 3 tasks");
    }
}
