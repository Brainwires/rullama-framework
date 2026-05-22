//! Tier-A feature cases for `brainwires-eval`'s own primitives.
//!
//! Catches regressions in the harness's own foundation.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_eval::{
    AlwaysFailCase, AlwaysPassCase, ConfidenceInterval95, EvaluationCase, EvaluationSuite,
    StochasticCase, TrialResult,
};

use crate::registry::TierACase;

// ── eval.always_pass_case ───────────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::evaluation_framework::always_pass",
        crate_name: "brainwires-eval",
        description: "AlwaysPassCase succeeds every trial; AlwaysFailCase fails every trial",
        factory: || Box::new(AlwaysPassFailCase),
    }
}

struct AlwaysPassFailCase;

#[async_trait]
impl EvaluationCase for AlwaysPassFailCase {
    fn name(&self) -> &str {
        "feature.eval.always_pass_fail"
    }
    fn category(&self) -> &str {
        "feature.eval"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let pass = AlwaysPassCase::new("smoke-pass");
        let fail = AlwaysFailCase::new("smoke-fail", "expected");
        let suite = EvaluationSuite::new(5);
        let r = suite
            .run_suite(&[
                std::sync::Arc::new(pass) as std::sync::Arc<dyn EvaluationCase>,
                std::sync::Arc::new(fail) as std::sync::Arc<dyn EvaluationCase>,
            ])
            .await;
        let pass_rate = r.stats.get("smoke-pass").map(|s| s.success_rate).unwrap_or(0.0);
        let fail_rate = r.stats.get("smoke-fail").map(|s| s.success_rate).unwrap_or(1.0);
        if (pass_rate - 1.0).abs() > 1e-9 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("AlwaysPassCase success_rate={pass_rate} expected 1.0"),
            ));
        }
        if fail_rate.abs() > 1e-9 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("AlwaysFailCase success_rate={fail_rate} expected 0.0"),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── eval.stochastic_case_is_deterministic_per_trial_id ─────────────────────

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::evaluation_framework::stochastic_deterministic",
        crate_name: "brainwires-eval",
        description: "StochasticCase produces the same result for the same trial_id (deterministic seed)",
        factory: || Box::new(StochasticDeterministicCase),
    }
}

struct StochasticDeterministicCase;

#[async_trait]
impl EvaluationCase for StochasticDeterministicCase {
    fn name(&self) -> &str {
        "feature.eval.stochastic_deterministic"
    }
    fn category(&self) -> &str {
        "feature.eval"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let case = StochasticCase::new("flaky", 0.5);
        // Same trial_id → same outcome across two invocations.
        for trial_id in 0..20 {
            let a = case.run(trial_id).await?;
            let b = case.run(trial_id).await?;
            if a.success != b.success {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "StochasticCase non-deterministic for trial_id={trial_id}: \
                         first run success={}, second={}",
                        a.success, b.success
                    ),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── eval.wilson_ci_covers_observed_rate ────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::evaluation_framework::wilson_ci",
        crate_name: "brainwires-eval",
        description: "ConfidenceInterval95::wilson(s, n) brackets the observed success rate s/n",
        factory: || Box::new(WilsonCiCase),
    }
}

struct WilsonCiCase;

#[async_trait]
impl EvaluationCase for WilsonCiCase {
    fn name(&self) -> &str {
        "feature.eval.wilson_ci_brackets_rate"
    }
    fn category(&self) -> &str {
        "feature.eval"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        for (s, n) in [(70_usize, 100_usize), (5, 10), (99, 100), (0, 10), (10, 10)] {
            let observed = s as f64 / n as f64;
            let ci = ConfidenceInterval95::wilson(s, n);
            if !(ci.lower <= observed && observed <= ci.upper) {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "Wilson CI [{:.3}, {:.3}] does not contain observed rate {observed:.3} \
                         (s={s}, n={n})",
                        ci.lower, ci.upper
                    ),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}
