//! Tier-A feature cases for `brainwires-call-policy` (FEATURES.md "Safety").

use anyhow::Result;
use async_trait::async_trait;
use brainwires_call_policy::{BudgetConfig, BudgetGuard};
use brainwires_core::Usage;
use brainwires_eval::{EvaluationCase, TrialResult};

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::call_policy_safety::budget_guard_caps_record_check_reset",
        crate_name: "brainwires-call-policy",
        description: "BudgetGuard: record_usage / record_cost_cents accumulate; check() rejects past caps; reset() zeroes counters",
        factory: || Box::new(BudgetGuardLifecycleCase),
    }
}

struct BudgetGuardLifecycleCase;

#[async_trait]
impl EvaluationCase for BudgetGuardLifecycleCase {
    fn name(&self) -> &str {
        "feature.safety.budget_guard_lifecycle"
    }
    fn category(&self) -> &str {
        "feature.safety"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(200),
            max_usd_cents: Some(500),
            max_rounds: Some(3),
        });
        // Empty guard passes pre-flight.
        if guard.check().is_err() {
            return Ok(TrialResult::failure(
                0,
                0,
                "fresh BudgetGuard rejected pre-flight check",
            ));
        }

        // Push tokens below the cap — still OK.
        guard.record_usage(&Usage::new(50, 50));
        if guard.check().is_err() {
            return Ok(TrialResult::failure(
                0,
                0,
                "guard rejected pre-flight at 100/200 tokens",
            ));
        }

        // Exceed the cap; check must fail.
        guard.record_usage(&Usage::new(100, 100));
        if guard.check().is_ok() {
            return Ok(TrialResult::failure(
                0,
                0,
                "guard allowed pre-flight at 300/200 tokens (over cap)",
            ));
        }

        // record_cost_cents accumulates separately.
        guard.record_cost_cents(600);
        if guard.usd_cents_consumed() != 600 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("expected 600 cents consumed, got {}", guard.usd_cents_consumed()),
            ));
        }

        // reset() zeroes everything.
        guard.reset();
        if guard.tokens_consumed() != 0
            || guard.usd_cents_consumed() != 0
            || guard.rounds_consumed() != 0
        {
            return Ok(TrialResult::failure(
                0,
                0,
                "reset() did not zero all counters",
            ));
        }
        if guard.check().is_err() {
            return Ok(TrialResult::failure(
                0,
                0,
                "after reset, fresh guard rejected pre-flight",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
