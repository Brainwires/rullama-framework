//! Tier-C assembly: triple-decorator stack
//! `RecordingProvider<BudgetProvider<ScriptedProvider>>`.
//!
//! Exercises the most realistic framework composition: the user's base
//! provider is wrapped in a budget guard for cost safety, then in a
//! recording wrapper for evaluation/replay. Verifies that calls flow
//! through all three layers and that each layer's invariants hold
//! end-to-end.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{BudgetConfig, BudgetGuard, BudgetProvider};
use rullama_core::{ChatOptions, ChatResponse, Message, Provider, Usage};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_test_fixtures::{RecordingProvider, ScriptedProvider};

pub struct RecordingBudgetScriptedAssembly;

#[async_trait]
impl EvaluationCase for RecordingBudgetScriptedAssembly {
    fn name(&self) -> &str {
        "assembly.providers.recording_budget_scripted"
    }
    fn category(&self) -> &str {
        "assembly"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // Base provider: deterministic 25+25-token responses.
        let canned = ChatResponse {
            message: Message::assistant("ok"),
            usage: Usage::new(25, 25),
            finish_reason: Some("stop".into()),
        };
        let scripted = ScriptedProvider::always_response("scripted", canned);

        // Middle layer: budget caps. Large enough to allow all three calls.
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(1_000),
            ..BudgetConfig::default()
        });
        let budgeted = BudgetProvider::new(Arc::new(scripted), guard.clone());

        // Outer layer: recording wrapper around the budget provider.
        let recorder = Arc::new(RecordingProvider::new(budgeted));

        // Drive three calls of varying shape.
        for label in ["hello", "world", "fin"] {
            let r = recorder
                .chat(&[Message::user(label)], None, &ChatOptions::default())
                .await?;
            if r.message.text() != Some("ok") {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "expected 'ok' from inner provider, got {:?}",
                        r.message.text()
                    ),
                ));
            }
        }

        // Layer 1 invariant: recorder captured every call with the right shape.
        if recorder.call_count() != 3 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "RecordingProvider captured {} calls, expected 3",
                    recorder.call_count()
                ),
            ));
        }
        let calls = recorder.calls();
        for (i, call) in calls.iter().enumerate() {
            if call.method != "chat" {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("call {i} method = {:?}, expected \"chat\"", call.method),
                ));
            }
            if call.message_count != 1 {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!(
                        "call {i} message_count = {}, expected 1",
                        call.message_count
                    ),
                ));
            }
        }

        // Layer 2 invariant: budget tracked token usage across all three calls.
        if guard.tokens_consumed() != 150 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "BudgetGuard tokens_consumed = {}, expected 150 (3 * 50)",
                    guard.tokens_consumed()
                ),
            ));
        }
        if guard.rounds_consumed() != 3 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "BudgetGuard rounds_consumed = {}, expected 3",
                    guard.rounds_consumed()
                ),
            ));
        }

        Ok(TrialResult::success(0, 0))
    }
}

pub fn case() -> Box<dyn EvaluationCase> {
    Box::new(RecordingBudgetScriptedAssembly)
}
