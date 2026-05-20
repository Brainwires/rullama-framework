//! Autonomous eval-driven self-improvement feedback loop.
//!
//! [`AutonomousFeedbackLoop`] closes the loop between the evaluation framework
//! and the self-improvement controller:
//!
//! 1. Run the eval suite to measure current success rates.
//! 2. Classify faults via [`analyze_suite_for_faults`].
//! 3. If faults exist, run [`SelfImprovementController`] with an
//!    [`EvalStrategy`] to generate and execute targeted fix tasks.
//! 4. Re-run the eval suite to verify improvement.
//! 5. When the overall rate improved by ≥ `improvement_threshold`, update
//!    the JSON baselines file (and optionally commit to git).
//! 6. Repeat until all faults resolve (converged) or `max_feedback_rounds` hit.

use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use brainwires_eval::fault_report::analyze_suite_for_faults;
use brainwires_eval::{EvaluationCase, EvaluationSuite, RegressionSuite, SuiteConfig, SuiteResult};

use super::config::SelfImprovementConfig;
use super::controller::SelfImprovementController;
use super::metrics::{SessionMetrics, SessionReport};
use super::strategies::eval_strategy::{EvalStrategy, EvalStrategyConfig};

// ── Config ────────────────────────────────────────────────────────────────────

/// Full configuration for [`AutonomousFeedbackLoop`].
#[derive(Debug, Clone)]
pub struct FeedbackLoopConfig {
    /// Config passed to [`SelfImprovementController`] each round.
    pub self_improve: SelfImprovementConfig,
    /// Path for loading/saving eval baselines JSON.
    /// Default: `"eval-baselines.json"`.
    pub baselines_path: String,
    /// Automatically save updated baselines when improvement ≥ `improvement_threshold`.
    /// Default: `true`.
    pub auto_update_baselines: bool,
    /// Minimum overall success-rate improvement (0.0–1.0) to trigger a
    /// baseline update.  Default: 0.05 (5 %).
    pub improvement_threshold: f64,
    /// Maximum outer-loop rounds before stopping.  Default: 3.
    pub max_feedback_rounds: u32,
    /// Trials per eval case per run.  Default: 10.
    pub n_eval_trials: usize,
    /// When `true`, commit the updated baselines JSON to git.  Default: false.
    pub commit_baselines: bool,
    /// Passed to [`analyze_suite_for_faults`].  Default: 0.2.
    pub consistent_failure_threshold: f64,
    /// Passed to [`analyze_suite_for_faults`].  Default: 0.25.
    pub flaky_ci_threshold: f64,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            self_improve: SelfImprovementConfig::default(),
            baselines_path: "eval-baselines.json".to_string(),
            auto_update_baselines: true,
            improvement_threshold: 0.05,
            max_feedback_rounds: 3,
            n_eval_trials: 10,
            commit_baselines: false,
            consistent_failure_threshold: 0.2,
            flaky_ci_threshold: 0.25,
        }
    }
}

// ── Per-round result ──────────────────────────────────────────────────────────

/// Result summary for one feedback loop round.
#[derive(Debug, Clone)]
pub struct FeedbackRoundResult {
    /// Round number (1-based).
    pub round: u32,
    /// Number of faults detected *before* the self-improvement run.
    pub faults_before: usize,
    /// Number of faults detected *after* the self-improvement run.
    pub faults_after: usize,
    /// Fault categories (case names) that were resolved in this round.
    pub improved_categories: Vec<String>,
    /// Fault categories (case names) that remain after self-improvement.
    pub unimproved_categories: Vec<String>,
    /// Session report from the self-improvement controller.
    pub session_report: SessionReport,
}

// ── Full loop report ──────────────────────────────────────────────────────────

/// Aggregate report for a complete [`AutonomousFeedbackLoop`] run.
#[derive(Debug, Clone)]
pub struct FeedbackLoopReport {
    /// Per-round summaries.
    pub rounds: Vec<FeedbackRoundResult>,
    /// Wall-clock duration of the entire run in seconds.
    pub total_duration_secs: f64,
    /// `true` if the loop ended with zero faults (fully resolved).
    pub converged: bool,
}

impl FeedbackLoopReport {
    /// Render the report as a GitHub-flavored Markdown string.
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
            if self.converged { "✅ yes" } else { "❌ no" }
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

// ── AutonomousFeedbackLoop ────────────────────────────────────────────────────

/// Orchestrates the eval → fix → verify cycle autonomously.
pub struct AutonomousFeedbackLoop {
    config: FeedbackLoopConfig,
    cases: Vec<Arc<dyn EvaluationCase>>,
}

impl AutonomousFeedbackLoop {
    /// Create a new feedback loop with the given configuration and eval cases.
    pub fn new(config: FeedbackLoopConfig, cases: Vec<Arc<dyn EvaluationCase>>) -> Self {
        Self { config, cases }
    }

    /// Run the autonomous feedback loop and return the full report.
    ///
    /// The Markdown report is also saved to
    /// `test-results/self-improve/feedback-loop-{timestamp}.md`.
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

        self.save_report(&report);
        Ok(report)
    }

    // ── private helpers ───────────────────────────────────────────────────────

    async fn run_round(&self, round: u32) -> Result<FeedbackRoundResult> {
        // 1. Evaluate before.
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

        // Nothing to fix — short-circuit with a converged round.
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

        // 2. Run self-improvement using the eval-driven strategy.
        let eval_config = EvalStrategyConfig {
            n_trials: self.config.n_eval_trials,
            baselines_path: Some(self.config.baselines_path.clone()),
            consistent_failure_threshold: self.config.consistent_failure_threshold,
            flaky_ci_threshold: self.config.flaky_ci_threshold,
            max_tasks: faults_before.len().min(10),
        };

        let mut improve_config = self.config.self_improve.clone();
        // Clamp max_cycles to the number of faults.
        improve_config.max_cycles = (faults_before.len() as u32).min(improve_config.max_cycles);

        let eval_strategy = EvalStrategy::new(self.cases.clone(), eval_config);
        let mut controller = SelfImprovementController::new_with_strategies(
            improve_config,
            vec![Box::new(eval_strategy)],
        );

        let session_report = controller.run().await?;

        // 3. Evaluate after.
        let after = self.run_eval_suite().await?;
        let regression_after = self.load_regression_suite();
        let faults_after = analyze_suite_for_faults(
            &after,
            regression_after.as_ref(),
            self.config.consistent_failure_threshold,
            self.config.flaky_ci_threshold,
        );

        tracing::info!(
            "Round {}: {} fault(s) after improvement ({} resolved)",
            round,
            faults_after.len(),
            faults_before.len().saturating_sub(faults_after.len()),
        );

        // 4. Classify per-category improvement.
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

        // 5. Update baselines if overall rate improved enough.
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
                    "Baselines updated ({:.1}% → {:.1}%)",
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

    fn save_report(&self, report: &FeedbackLoopReport) {
        let dir = "test-results/self-improve";
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!("Failed to create results directory: {e}");
            return;
        }
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let path = format!("{dir}/feedback-loop-{timestamp}.md");
        if let Err(e) = std::fs::write(&path, report.to_markdown()) {
            tracing::warn!("Failed to save feedback loop report to {path}: {e}");
        } else {
            tracing::info!("Feedback loop report saved to {path}");
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_eval::{AlwaysFailCase, AlwaysPassCase};

    fn make_config_dry_run() -> FeedbackLoopConfig {
        FeedbackLoopConfig {
            n_eval_trials: 3,
            max_feedback_rounds: 2,
            auto_update_baselines: false,
            commit_baselines: false,
            self_improve: SelfImprovementConfig {
                dry_run: true,
                max_cycles: 1,
                max_budget: 0.0, // no budget needed for dry run
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_converges_immediately_when_no_faults() {
        // All passing, no regression suite → no faults → should converge in round 1.
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![Arc::new(AlwaysPassCase::new("smoke_ok"))];
        let lp = AutonomousFeedbackLoop::new(make_config_dry_run(), cases);
        let report = lp.run().await.unwrap();

        assert!(report.converged, "should converge when there are no faults");
        assert_eq!(report.rounds.len(), 1);
        assert_eq!(report.rounds[0].faults_before, 0);
        assert_eq!(report.rounds[0].faults_after, 0);
    }

    #[tokio::test]
    async fn test_report_to_markdown_contains_round_info() {
        let cases: Vec<Arc<dyn EvaluationCase>> = vec![Arc::new(AlwaysPassCase::new("pass_case"))];
        let lp = AutonomousFeedbackLoop::new(make_config_dry_run(), cases);
        let report = lp.run().await.unwrap();
        let md = report.to_markdown();

        assert!(md.contains("# Eval-Driven Feedback Loop Report"));
        assert!(md.contains("## Round 1"));
    }

    #[tokio::test]
    async fn test_runs_up_to_max_rounds_without_converging() {
        // Always-failing cases → faults in every round → max_rounds hit, no convergence.
        let cases: Vec<Arc<dyn EvaluationCase>> =
            vec![Arc::new(AlwaysFailCase::new("always_bad", "fail"))];
        let mut config = make_config_dry_run();
        config.max_feedback_rounds = 2;

        let lp = AutonomousFeedbackLoop::new(config, cases);
        let report = lp.run().await.unwrap();

        assert!(!report.converged);
        assert_eq!(report.rounds.len(), 2);
    }
}
