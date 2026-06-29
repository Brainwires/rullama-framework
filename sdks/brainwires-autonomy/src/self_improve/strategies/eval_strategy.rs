//! Eval-driven improvement strategy.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use brainwires_eval::fault_report::analyze_suite_for_faults;
use brainwires_eval::{EvaluationCase, EvaluationSuite, RegressionSuite, SuiteConfig};

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Configuration for the eval-driven strategy.
#[derive(Debug, Clone)]
pub struct EvalStrategyConfig {
    /// Number of eval trials to run.
    pub n_trials: usize,
    /// Path to the baselines JSON file.
    pub baselines_path: Option<String>,
    /// Failure rate threshold for consistent failures.
    pub consistent_failure_threshold: f64,
    /// Threshold for flaky CI detection.
    pub flaky_ci_threshold: f64,
    /// Maximum number of tasks to generate.
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

/// Strategy that runs an eval suite and converts detected faults into improvement tasks.
///
/// Uses `brainwires-eval` to run evaluation cases, compare against baselines,
/// and produce tasks targeting consistently failing or regressing cases.
pub struct EvalStrategy {
    cases: Vec<Arc<dyn EvaluationCase>>,
    eval_config: EvalStrategyConfig,
}

impl EvalStrategy {
    /// Create a new eval-driven strategy with the given cases and configuration.
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
