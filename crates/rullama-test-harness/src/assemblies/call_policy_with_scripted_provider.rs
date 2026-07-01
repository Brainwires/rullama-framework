//! Tier-C assembly: `rullama-call-policy` wrapping a `ScriptedProvider`.
//!
//! Proves the most common framework composition path: a `BudgetProvider`
//! enforces caps around any base `Provider`, accumulating real `Usage` from
//! the canned responses without ever invoking the network. Two layers
//! exercised together end-to-end (test-fixtures' scripted responses +
//! call-policy's budget accounting + core's Provider trait + Usage round
//! trip).
//!
//! Deterministic by construction: scripted responses → fixed Usage →
//! known token totals.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{BudgetConfig, BudgetGuard, BudgetProvider};
use rullama_core::{ChatOptions, ChatResponse, Message, Provider, Usage};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_test_fixtures::ScriptedProvider;

pub struct CallPolicyWithScriptedProviderAssembly;

#[async_trait]
impl EvaluationCase for CallPolicyWithScriptedProviderAssembly {
    fn name(&self) -> &str {
        "assembly.call_policy.budget_accumulates_scripted_usage"
    }
    fn category(&self) -> &str {
        "assembly"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // Three canned responses, each declaring 30 prompt + 20 completion
        // tokens. After all three are drained the BudgetGuard should report
        // exactly 3 * 50 = 150 cumulative tokens.
        let canned = ChatResponse {
            message: Message::assistant("ok"),
            usage: Usage::new(30, 20),
            finish_reason: Some("stop".into()),
        };
        let inner: Arc<dyn Provider> = Arc::new(ScriptedProvider::always_response(
            "scripted",
            canned.clone(),
        ));
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(1_000),
            ..BudgetConfig::default()
        });
        let budgeted = BudgetProvider::new(inner, guard.clone());

        for i in 0..3 {
            let r = budgeted
                .chat(&[Message::user("ping")], None, &ChatOptions::default())
                .await?;
            if r.usage.total_tokens != canned.usage.total_tokens {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "call {i}: BudgetProvider returned usage {} tokens, expected {}",
                        r.usage.total_tokens, canned.usage.total_tokens
                    ),
                ));
            }
        }

        let total = guard.tokens_consumed();
        if total != 150 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("expected 150 cumulative tokens, BudgetGuard reports {total}"),
            ));
        }
        let rounds = guard.rounds_consumed();
        if rounds != 3 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("expected 3 rounds, BudgetGuard reports {rounds}"),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

pub fn case() -> Box<dyn EvaluationCase> {
    Box::new(CallPolicyWithScriptedProviderAssembly)
}
