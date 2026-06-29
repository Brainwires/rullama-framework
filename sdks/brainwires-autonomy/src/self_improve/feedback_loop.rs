//! Autonomous eval-driven self-improvement feedback loop.

use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use brainwires_core::Provider;
use brainwires_eval::fault_report::analyze_suite_for_faults;
use brainwires_eval::{EvaluationCase, EvaluationSuite, RegressionSuite, SuiteConfig, SuiteResult};

use super::controller::SelfImprovementController;
use super::strategies::eval_strategy::{EvalStrategy, EvalStrategyConfig};
use crate::config::SelfImprovementConfig;
use crate::metrics::{SessionMetrics, SessionReport};

/// Minimum improvement fraction required to update eval baselines.
const IMPROVEMENT_THRESHOLD: f64 = 0.05;
/// Failure rate threshold for detecting consistent failures.
const CONSISTENT_FAILURE_THRESHOLD: f64 = 0.2;
/// Threshold for flaky CI detection.
const FLAKY_CI_THRESHOLD: f64 = 0.25;

/// Configuration for the feedback loop.
#[derive(Debug, Clone)]
pub struct FeedbackLoopConfig {
    /// Self-improvement configuration.
    pub self_improve: SelfImprovementConfig,
    /// Path to the eval baselines JSON file.
    pub baselines_path: String,
    /// Whether to auto-update baselines after improvement.
    pub auto_update_baselines: bool,
    /// Minimum improvement fraction to update baselines.
    pub improvement_threshold: f64,
    /// Maximum number of feedback rounds.
    pub max_feedback_rounds: u32,
    /// Number of eval trials per round.
    pub n_eval_trials: usize,
    /// Whether to commit updated baselines to Git.
    pub commit_baselines: bool,
    /// Failure rate threshold for consistent failures.
    pub consistent_failure_threshold: f64,
    /// Threshold for flaky CI detection.
    pub flaky_ci_threshold: f64,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            self_improve: SelfImprovementConfig::default(),
            baselines_path: "eval-baselines.json".to_string(),
            auto_update_baselines: true,
            improvement_threshold: IMPROVEMENT_THRESHOLD,
            max_feedback_rounds: 3,
            n_eval_trials: 10,
            commit_baselines: false,
            consistent_failure_threshold: CONSISTENT_FAILURE_THRESHOLD,
            flaky_ci_threshold: FLAKY_CI_THRESHOLD,
        }
    }
}

/// Per-round result summary.
#[derive(Debug, Clone)]
pub struct FeedbackRoundResult {
    /// Round number (1-based).
    pub round: u32,
    /// Number of faults detected before improvement.
    pub faults_before: usize,
    /// Number of faults detected after improvement.
    pub faults_after: usize,
    /// Categories that improved this round.
    pub improved_categories: Vec<String>,
    /// Categories that did not improve.
    pub unimproved_categories: Vec<String>,
    /// Session report for the improvement run.
    pub session_report: SessionReport,
}

/// Aggregate report for a complete feedback loop run.
#[derive(Debug, Clone)]
pub struct FeedbackLoopReport {
    /// Results from each feedback round.
    pub rounds: Vec<FeedbackRoundResult>,
    /// Total duration of the loop in seconds.
    pub total_duration_secs: f64,
    /// Whether the loop converged (zero faults).
    pub converged: bool,
}

impl FeedbackLoopReport {
    /// Render the report as a Markdown document.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Eval-Driven Feedback Loop Report\n\n");
        md.push_str(&format!(
            "**Total duration**: {:.1}s  \n",
            self.total_duration_secs
        ));
        md.push_str(&format!("**Rounds completed**: {}  \n", self.rounds.len()));
        md.push_str(&format!(
            "**Converged**: {}  \n\n",
            if self.converged { "yes" } else { "no" }
        ));

        for round in &self.rounds {
            md.push_str(&format!("---\n\n## Round {}\n\n", round.round));
            md.push_str("| Metric | Value |\n|--------|-------|\n");
            md.push_str(&format!(
                "| Faults before | {} |\n| Faults after | {} |\n",
                round.faults_before, round.faults_after
            ));
            if !round.improved_categories.is_empty() {
                let mut cats = round.improved_categories.clone();
                cats.sort();
                md.push_str(&format!("| Improved | {} |\n", cats.join(", ")));
            }
            if !round.unimproved_categories.is_empty() {
                let mut cats = round.unimproved_categories.clone();
                cats.sort();
                md.push_str(&format!("| Unimproved | {} |\n", cats.join(", ")));
            }
            md.push('\n');
            md.push_str("### Session summary\n\n");
            md.push_str(&round.session_report.to_markdown());
            md.push('\n');
        }

        md
    }
}

/// Orchestrates the eval -> fix -> verify cycle autonomously.
///
/// Each round: runs the eval suite to find faults, uses the self-improvement
/// controller to fix them, re-runs the suite to verify improvement, and
/// optionally updates baselines when scores improve above threshold.
pub struct AutonomousFeedbackLoop {
    config: FeedbackLoopConfig,
    cases: Vec<Arc<dyn EvaluationCase>>,
    provider: Arc<dyn Provider>,
}

impl AutonomousFeedbackLoop {
    /// Create a new autonomous feedback loop.
    pub fn new(
        config: FeedbackLoopConfig,
        cases: Vec<Arc<dyn EvaluationCase>>,
        provider: Arc<dyn Provider>,
    ) -> Self {
        Self {
            config,
            cases,
            provider,
        }
    }

    /// Run the feedback loop, returning a report when complete.
    pub async fn run(&self) -> Result<FeedbackLoopReport> {
        let start = Instant::now();
        let mut rounds = Vec::new();
        let mut converged = false;

        for round in 1..=self.config.max_feedback_rounds {
            tracing::info!(
                "AutonomousFeedbackLoop: starting round {}/{}",
                round,
                self.config.max_feedback_rounds
            );

            let round_result = self.run_round(round).await?;
            let no_faults = round_result.faults_after == 0;
            rounds.push(round_result);

            if no_faults {
                tracing::info!("AutonomousFeedbackLoop: converged after {} round(s)", round);
                converged = true;
                break;
            }
        }

        let report = FeedbackLoopReport {
            rounds,
            total_duration_secs: start.elapsed().as_secs_f64(),
            converged,
        };

        Ok(report)
    }

    async fn run_round(&self, round: u32) -> Result<FeedbackRoundResult> {
        let before = self.run_eval_suite().await?;
        let regression_before = self.load_regression_suite();
        let faults_before = analyze_suite_for_faults(
            &before,
            regression_before.as_ref(),
            self.config.consistent_failure_threshold,
            self.config.flaky_ci_threshold,
        );

        tracing::info!(
            "Round {}: {} fault(s) before improvement",
            round,
            faults_before.len()
        );

        if faults_before.is_empty() {
            let empty_report =
                SessionReport::new(SessionMetrics::new(), std::time::Duration::ZERO, None);
            return Ok(FeedbackRoundResult {
                round,
                faults_before: 0,
                faults_after: 0,
                improved_categories: Vec::new(),
                unimproved_categories: Vec::new(),
                session_report: empty_report,
            });
        }

        let eval_config = EvalStrategyConfig {
            n_trials: self.config.n_eval_trials,
            baselines_path: Some(self.config.baselines_path.clone()),
            consistent_failure_threshold: self.config.consistent_failure_threshold,
            flaky_ci_threshold: self.config.flaky_ci_threshold,
            max_tasks: faults_before.len().min(10),
        };

        let mut improve_config = self.config.self_improve.clone();
        improve_config.max_cycles = (faults_before.len() as u32).min(improve_config.max_cycles);

        let eval_strategy = EvalStrategy::new(self.cases.clone(), eval_config);
        let mut controller = SelfImprovementController::new_with_strategies(
            improve_config,
            self.provider.clone(),
            vec![Box::new(eval_strategy)],
        );

        let session_report = controller.run().await?;

        let after = self.run_eval_suite().await?;
        let regression_after = self.load_regression_suite();
        let faults_after = analyze_suite_for_faults(
            &after,
            regression_after.as_ref(),
            self.config.consistent_failure_threshold,
            self.config.flaky_ci_threshold,
        );

        let names_before: HashSet<String> =
            faults_before.iter().map(|f| f.case_name.clone()).collect();
        let names_after: HashSet<String> =
            faults_after.iter().map(|f| f.case_name.clone()).collect();

        let mut improved_categories: Vec<String> =
            names_before.difference(&names_after).cloned().collect();
        improved_categories.sort();

        let mut unimproved_categories: Vec<String> =
            names_before.intersection(&names_after).cloned().collect();
        unimproved_categories.sort();

        if self.config.auto_update_baselines {
            self.maybe_update_baselines(&before, &after).await;
        }

        Ok(FeedbackRoundResult {
            round,
            faults_before: faults_before.len(),
            faults_after: faults_after.len(),
            improved_categories,
            unimproved_categories,
            session_report,
        })
    }

    async fn run_eval_suite(&self) -> Result<SuiteResult> {
        let suite = EvaluationSuite::with_config(SuiteConfig {
            n_trials: self.config.n_eval_trials,
            ..SuiteConfig::default()
        });
        Ok(suite.run_suite(&self.cases).await)
    }

    fn load_regression_suite(&self) -> Option<RegressionSuite> {
        std::fs::read_to_string(&self.config.baselines_path)
            .ok()
            .and_then(|json| RegressionSuite::load_baselines_from_json(&json).ok())
    }

    async fn maybe_update_baselines(&self, before: &SuiteResult, after: &SuiteResult) {
        let before_rate = before.overall_success_rate();
        let after_rate = after.overall_success_rate();

        if after_rate - before_rate < self.config.improvement_threshold {
            return;
        }

        let mut reg = RegressionSuite::new();
        reg.record_baselines(after);

        match reg.baselines_to_json() {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.config.baselines_path, &json) {
                    tracing::warn!("Failed to write updated baselines: {e}");
                    return;
                }
                tracing::info!(
                    "Baselines updated ({:.1}% -> {:.1}%)",
                    before_rate * 100.0,
                    after_rate * 100.0,
                );
                if self.config.commit_baselines {
                    self.commit_baselines_to_git().await;
                }
            }
            Err(e) => tracing::warn!("Failed to serialize baselines: {e}"),
        }
    }

    async fn commit_baselines_to_git(&self) {
        let add = tokio::process::Command::new("git")
            .args(["add", &self.config.baselines_path])
            .output()
            .await;

        if add.is_ok_and(|o| o.status.success()) {
            let _ = tokio::process::Command::new("git")
                .args([
                    "commit",
                    "-m",
                    "chore(eval): update eval baselines after self-improvement",
                ])
                .output()
                .await;
        }
    }
}
