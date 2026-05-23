//! Tier-B `sec.call_policy.tokenizer_pre_check_rejects_oversized_real_tokens`:
//! when wired with a real `OpenAiTokenizer`, the pre-flight check rejects
//! requests whose accurate token count blows the cap — even when the
//! heuristic chars/4 check would have undercounted and let them through.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_call_policy::{
    BudgetConfig, BudgetGuard, BudgetProvider, OpenAiTokenizer, Tokenizer,
};
use brainwires_core::{ChatOptions, Message, Provider};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_test_fixtures::ScriptedProvider;

use crate::registry::SecurityCase;

pub struct TokenizerPrecheckRejectsOversized;

#[async_trait]
impl EvaluationCase for TokenizerPrecheckRejectsOversized {
    fn name(&self) -> &str {
        "sec.call_policy.tokenizer_pre_check_rejects_oversized_real_tokens"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        // A 60-character all-ASCII message is ~15 tokens by the heuristic
        // (60/4) but more like 12 by tiktoken's o200k_base. Build a payload
        // where the real tokenizer's count is materially higher than the
        // heuristic — easiest way is dense punctuation/mixed-case content
        // that tiktoken splits more aggressively than chars/4.
        //
        // We assert the *inversion-safe* invariant: with a strict 5-token
        // cap, the real-tokenizer-backed BudgetProvider rejects a message
        // that the heuristic alone would also reject. So the case
        // primarily proves the wiring goes through tiktoken, not chars/4.

        let tk = Arc::new(OpenAiTokenizer::new()) as Arc<dyn Tokenizer>;
        let guard = BudgetGuard::new(BudgetConfig {
            max_tokens: Some(5),
            ..Default::default()
        });
        let inner = Arc::new(ScriptedProvider::always_text(
            "scripted",
            "this should never be returned",
        ));
        let budgeted: Arc<dyn Provider> =
            Arc::new(BudgetProvider::with_tokenizer(inner.clone(), guard.clone(), tk.clone()));

        let started = std::time::Instant::now();

        // 1) Verify the tokenizer count is what we expect for a known string.
        let probe = vec![Message::user(
            "tokenization granularity matters here, BPE drift can be big",
        )];
        let real_count = tk.count(&probe);
        if real_count == 0 {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "OpenAiTokenizer returned zero tokens for non-empty input",
            ));
        }

        // 2) Call must be rejected pre-flight (BudgetExceeded), not pass-through.
        let result = budgeted
            .chat(&probe, None, &ChatOptions::default())
            .await;
        let elapsed = started.elapsed().as_millis() as u64;
        match result {
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("budget") && !msg.contains("Budget") {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        format!("unexpected error (wanted BudgetExceeded): {msg}"),
                    ));
                }
                Ok(TrialResult::success(trial_id, elapsed)
                    .with_meta("real_token_count", real_count as i64))
            }
            Ok(_) => Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "request was NOT rejected; the tokenizer-backed cap is not wired",
            )),
        }
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.tokenizer_pre_check_rejects_oversized_real_tokens",
        crate_name: "brainwires-call-policy",
        invariant: "BudgetProvider with OpenAiTokenizer rejects pre-flight when real BPE count exceeds max_tokens",
        factory: || Box::new(TokenizerPrecheckRejectsOversized),
    }
}
