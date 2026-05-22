//! Tier-B adversarial cases for `brainwires_call_policy::BudgetProvider`
//! and `CircuitBreakerProvider`.
//!
//! Invariants:
//! - `BudgetProvider::chat` rejects pre-flight when the guard is already
//!   over-budget, WITHOUT invoking the inner provider.
//! - `BudgetProvider::chat` rejects requests whose raw input payload alone
//!   would push consumption past the configured cap.
//! - `BudgetGuard::default()` (no caps) does not block calls — surprises here
//!   would mean an accidental "deny-by-default" budget.
//! - `CircuitBreakerProvider` opens after `failure_threshold` consecutive
//!   failures and rejects further calls without invoking the inner provider.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_call_policy::{
    BudgetConfig, BudgetGuard, BudgetProvider, CircuitBreakerConfig, CircuitBreakerProvider,
};
use brainwires_core::{ChatOptions, Message, Provider};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_test_fixtures::{FailingProvider, ScriptedProvider};

use crate::registry::SecurityCase;

// ── sec.call_policy.budget_precheck_blocks_provider ─────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.budget_precheck_blocks_provider",
        crate_name: "brainwires-call-policy",
        invariant: "BudgetProvider rejects pre-flight without invoking the inner provider when budget is exhausted",
        factory: || Box::new(BudgetPrecheckBlocksProviderCase),
    }
}

struct BudgetPrecheckBlocksProviderCase;

#[async_trait]
impl EvaluationCase for BudgetPrecheckBlocksProviderCase {
    fn name(&self) -> &str {
        "sec.call_policy.budget_precheck_blocks_provider"
    }
    fn category(&self) -> &str {
        "security.call_policy"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(100),
            ..BudgetConfig::default()
        });
        // Pre-consume to push the guard into the exhausted state.
        guard.record_usage(&brainwires_core::Usage::new(60, 60));

        // FailingProvider as inner: if pre-flight is wrong and we ever
        // reach the inner provider, it will surface a recognisable error
        // string. The test then differentiates "blocked-by-budget"
        // (expected) from "inner-was-called" (the security bug).
        let inner: Arc<dyn Provider> = Arc::new(
            FailingProvider::new("inner provider must not be called when budget is exhausted"),
        );
        let budgeted = BudgetProvider::new(inner, guard);
        let result = budgeted
            .chat(&[Message::user("ping")], None, &ChatOptions::default())
            .await;
        match result {
            Err(e) => {
                let msg = format!("{e:#}");
                if msg.contains("inner provider must not be called") {
                    return Ok(TrialResult::failure(
                        0,
                        0,
                        "BudgetProvider invoked inner provider despite exhausted budget",
                    ));
                }
                // Any other error (budget exceeded) is the expected outcome.
                Ok(TrialResult::success(0, 0))
            }
            Ok(_) => Ok(TrialResult::failure(
                0,
                0,
                "BudgetProvider returned Ok despite exhausted budget",
            )),
        }
    }
}

// ── sec.call_policy.budget_rejects_oversized_input ──────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.budget_rejects_oversized_input",
        crate_name: "brainwires-call-policy",
        invariant: "BudgetProvider rejects pre-flight when raw input payload alone exceeds max_tokens",
        factory: || Box::new(BudgetRejectsOversizedInputCase),
    }
}

struct BudgetRejectsOversizedInputCase;

#[async_trait]
impl EvaluationCase for BudgetRejectsOversizedInputCase {
    fn name(&self) -> &str {
        "sec.call_policy.budget_rejects_oversized_input"
    }
    fn category(&self) -> &str {
        "security.call_policy"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // 4-chars-per-token heuristic in budget.rs: 4000 chars ≈ 1000 tokens.
        let huge = "x".repeat(4000);
        let messages = vec![Message::user(huge)];
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(100),
            ..BudgetConfig::default()
        });
        let inner: Arc<dyn Provider> = Arc::new(FailingProvider::new(
            "inner provider must not be reached for oversized input",
        ));
        let budgeted = BudgetProvider::new(inner, guard);
        let result = budgeted.chat(&messages, None, &ChatOptions::default()).await;
        match result {
            Err(e) => {
                let msg = format!("{e:#}");
                if msg.contains("inner provider must not be reached") {
                    return Ok(TrialResult::failure(
                        0,
                        0,
                        "BudgetProvider invoked inner provider for oversized input",
                    ));
                }
                Ok(TrialResult::success(0, 0))
            }
            Ok(_) => Ok(TrialResult::failure(
                0,
                0,
                "BudgetProvider returned Ok for ~1000-token payload with max_tokens=100",
            )),
        }
    }
}

// ── sec.call_policy.budget_unbounded_default ────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.budget_unbounded_default",
        crate_name: "brainwires-call-policy",
        invariant: "BudgetGuard::new(BudgetConfig::default()) does not block calls — opt-in caps only",
        factory: || Box::new(BudgetUnboundedDefaultCase),
    }
}

struct BudgetUnboundedDefaultCase;

#[async_trait]
impl EvaluationCase for BudgetUnboundedDefaultCase {
    fn name(&self) -> &str {
        "sec.call_policy.budget_unbounded_default"
    }
    fn category(&self) -> &str {
        "security.call_policy"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let guard = BudgetGuard::new(BudgetConfig::default());
        // Even after recording large usage, an unbounded guard must not block.
        guard.record_usage(&brainwires_core::Usage::new(1_000_000, 1_000_000));
        guard.record_cost_cents(999_999);
        let inner: Arc<dyn Provider> = Arc::new(ScriptedProvider::always_text("ok", "fine"));
        let budgeted = BudgetProvider::new(inner, guard);
        let r = budgeted
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await?;
        if r.message.text().unwrap_or("") != "fine" {
            return Ok(TrialResult::failure(
                0,
                0,
                "expected inner provider's response under unbounded budget",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.call_policy.circuit_opens_and_blocks ───────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.circuit_opens_and_blocks",
        crate_name: "brainwires-call-policy",
        invariant: "CircuitBreakerProvider trips Open after failure_threshold consecutive errors and rejects further calls without invoking inner",
        factory: || Box::new(CircuitOpensAndBlocksCase),
    }
}

struct CircuitOpensAndBlocksCase;

#[async_trait]
impl EvaluationCase for CircuitOpensAndBlocksCase {
    fn name(&self) -> &str {
        "sec.call_policy.circuit_opens_and_blocks"
    }
    fn category(&self) -> &str {
        "security.call_policy"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let inner: Arc<dyn Provider> = Arc::new(FailingProvider::new("upstream is down"));
        let cb = CircuitBreakerProvider::new(
            inner,
            CircuitBreakerConfig {
                failure_threshold: 3,
                cooldown: std::time::Duration::from_secs(60),
            },
        );
        // Three failing calls should trip the circuit.
        for _ in 0..3 {
            let _ = cb
                .chat(&[Message::user("ping")], None, &ChatOptions::default())
                .await;
        }
        // Replace the inner provider conceptually: we can't swap inner here,
        // so we just confirm that the *next* call returns CircuitOpen quickly
        // without an upstream-down message.
        let result = cb
            .chat(&[Message::user("again")], None, &ChatOptions::default())
            .await;
        let err = match result {
            Err(e) => format!("{e:#}"),
            Ok(_) => {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    "CircuitBreakerProvider returned Ok after circuit should have opened",
                ));
            }
        };
        // Expect a CircuitOpen-style error, NOT the upstream's "upstream is down".
        if err.contains("upstream is down") {
            return Ok(TrialResult::failure(
                0,
                0,
                "circuit did not trip — inner provider still being invoked after threshold",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
