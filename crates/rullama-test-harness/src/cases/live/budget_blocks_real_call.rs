//! D.9 — `live.providers.budget_pre_check_blocks_real_call`. Stack:
//!
//!     BudgetProvider → RecordingProvider → real OllamaProvider
//!
//! Initialise the `BudgetGuard` so `max_rounds=1` and then pre-consume
//! that round; the next chat() must be rejected pre-flight by the
//! BudgetProvider and the RecordingProvider must observe zero calls,
//! proving the real Ollama backend is never reached. Verifies the
//! budget invariant against a real backend in the loop.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{BudgetConfig, BudgetGuard, BudgetProvider};
use rullama_core::message::Usage;
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::OllamaProvider;
use rullama_test_fixtures::RecordingProvider;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct BudgetPreCheckBlocksRealCall;

#[async_trait]
impl EvaluationCase for BudgetPreCheckBlocksRealCall {
    fn name(&self) -> &str {
        "live.providers.budget_pre_check_blocks_real_call"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(base) = live_ollama_base() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "RULLAMA_LIVE_OLLAMA_BASE not set",
            ));
        };
        let model = live_ollama_model();
        let started = std::time::Instant::now();

        // The real backend wrapped in a RecordingProvider so we can assert
        // it was never invoked.
        let real = OllamaProvider::new(model.clone(), Some(base));
        let recorder = Arc::new(RecordingProvider::new(real));

        // Budget guard that's already at its `max_rounds` cap.
        let guard = BudgetGuard::new(BudgetConfig {
            max_rounds: Some(1),
            ..Default::default()
        });
        // Pre-consume the only round so the pre-flight check rejects.
        // `record_usage` doesn't bump the round counter; we need
        // `check_and_tick` to do it.
        guard.check_and_tick()?;

        let budgeted: Arc<dyn Provider> =
            Arc::new(BudgetProvider::new(recorder.clone(), guard.clone()));

        let opts = ChatOptions::default().model(model).max_tokens(8);
        let messages = vec![Message::user("Reply hi")];

        let result = budgeted.chat(&messages, None, &opts).await;
        let elapsed = started.elapsed().as_millis() as u64;

        if result.is_ok() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "BudgetProvider should reject pre-flight when max_rounds is exhausted",
            ));
        }
        let err_msg = result.err().unwrap().to_string();
        if !err_msg.contains("budget") && !err_msg.contains("Budget") {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("unexpected error (wanted BudgetExceeded): {err_msg}"),
            ));
        }

        // RecordingProvider should have observed zero chat() calls — the
        // real Ollama backend must never have been reached.
        let real_calls = recorder.calls().len();
        if real_calls != 0 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!(
                    "real provider was invoked {real_calls} times; budget pre-check failed to block"
                ),
            ));
        }

        // Silence the unused-import warning when the `Usage` re-export
        // surface drifts; this keeps the assertion semantics intact.
        let _ = Usage::default();

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("real_provider_calls", real_calls as u64)
            .with_meta("rejected_with", err_msg))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.providers.budget_pre_check_blocks_real_call",
        provider: "ollama",
        description: "exhausted BudgetGuard blocks pre-flight; RecordingProvider confirms the real Ollama backend is never called",
        factory: || Box::new(BudgetPreCheckBlocksRealCall),
    }
}
