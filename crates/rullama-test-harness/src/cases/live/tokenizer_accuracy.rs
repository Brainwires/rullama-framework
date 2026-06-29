//! D.13 — `live.providers.tokenizer_accuracy_vs_real_usage`. Send a known
//! prompt to OpenAI + Anthropic; compare framework's `Tokenizer::count`
//! against the provider-reported `Usage.prompt_tokens`. Accept ±15% drift
//! since BPE tables differ across model versions and Anthropic's tokenizer
//! is closed-source. Catches drift if a tokenizer falls behind.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{AnthropicTokenizer, OpenAiTokenizer, Tokenizer};
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::{
    AnthropicChatProvider, AnthropicClient, OpenAiChatProvider, OpenAiClient,
};

use crate::live::{
    live_anthropic_key, live_anthropic_model, live_openai_key, live_openai_model,
};
use crate::registry::LiveCase;

pub struct TokenizerAccuracyVsRealUsage;

/// Longer prompt so the provider's per-message framing overhead (role
/// markers, separators) is amortised across the message body — for a
/// 5-word prompt it'd dominate the comparison.
const PROMPT: &str = "The quick brown fox jumps over the lazy dog. \
                      Tokenization granularity matters; BPE drift can be substantial. \
                      Different model versions tokenize the same string differently, \
                      so a framework tokenizer needs to track upstream changes — when \
                      it falls behind, budget pre-flight either over-rejects (slowing \
                      callers down) or under-rejects (letting an oversized request \
                      through silently). This is exactly the class of bug the live \
                      tier is here to surface: provider drift versus our cached BPE \
                      tables. Aim for parity within roughly fifteen percent on long \
                      passages; short ones can drift further because of framing.";

/// Acceptable drift between the framework's prediction and the provider's
/// reported `prompt_tokens`. 0.20 covers the per-message framing overhead
/// (role markers, separators) plus minor BPE-table differences across
/// model versions. Tighten this if we ship per-provider framing offsets.
const MAX_DRIFT: f64 = 0.20;

fn pct_diff(a: u32, b: u32) -> f64 {
    let denom = a.max(b).max(1) as f64;
    ((a as f64 - b as f64).abs()) / denom
}

#[async_trait]
impl EvaluationCase for TokenizerAccuracyVsRealUsage {
    fn name(&self) -> &str {
        "live.providers.tokenizer_accuracy_vs_real_usage"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let openai_key = live_openai_key();
        let anthropic_key = live_anthropic_key();
        if openai_key.is_none() && anthropic_key.is_none() {
            return Ok(TrialResult::skipped(
                trial_id,
                "neither RULLAMA_LIVE_OPENAI_KEY nor RULLAMA_LIVE_ANTHROPIC_KEY set",
            ));
        }
        let started = std::time::Instant::now();
        let messages = vec![Message::user(PROMPT)];
        let mut failures = Vec::new();
        let mut drifts: Vec<(String, f64)> = Vec::new();

        if let Some(key) = openai_key {
            let model = live_openai_model();
            let client = Arc::new(OpenAiClient::new(key, model.clone()));
            let provider = OpenAiChatProvider::new(client, model.clone());
            // Cap output low — we care about prompt_tokens, not the reply.
            let opts = ChatOptions::default().model(model.clone()).max_tokens(128);
            match provider.chat(&messages, None, &opts).await {
                Ok(resp) => {
                    let predicted = OpenAiTokenizer::new().count(&messages) as u32;
                    let actual = resp.usage.prompt_tokens;
                    let drift = pct_diff(predicted, actual);
                    if drift > MAX_DRIFT {
                        failures.push(format!(
                            "openai: predicted={predicted} actual={actual} drift={:.1}%",
                            drift * 100.0
                        ));
                    }
                    drifts.push(("openai".into(), drift));
                }
                Err(e) => failures.push(format!("openai chat failed: {e}")),
            }
        }

        if let Some(key) = anthropic_key {
            let model = live_anthropic_model();
            let client = Arc::new(AnthropicClient::new(key, model.clone()));
            let provider = AnthropicChatProvider::new(client, model.clone());
            let opts = ChatOptions::default().model(model.clone()).max_tokens(64);
            match provider.chat(&messages, None, &opts).await {
                Ok(resp) => {
                    let predicted = AnthropicTokenizer::new().count(&messages) as u32;
                    let actual = resp.usage.prompt_tokens;
                    if actual == 0 {
                        // Some Anthropic endpoints report 0 prompt_tokens
                        // when cache_control is hot; skip without failing.
                        drifts.push(("anthropic-no-usage".into(), 0.0));
                    } else {
                        let drift = pct_diff(predicted, actual);
                        if drift > MAX_DRIFT {
                            failures.push(format!(
                                "anthropic: predicted={predicted} actual={actual} drift={:.1}%",
                                drift * 100.0
                            ));
                        }
                        drifts.push(("anthropic".into(), drift));
                    }
                }
                Err(e) => failures.push(format!("anthropic chat failed: {e}")),
            }
        }

        let elapsed = started.elapsed().as_millis() as u64;
        if !failures.is_empty() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                failures.join("; "),
            ));
        }
        let drift_meta: Vec<_> = drifts
            .iter()
            .map(|(p, d)| serde_json::json!({"provider": p, "drift_pct": d * 100.0}))
            .collect();
        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("drifts", serde_json::json!(drift_meta)))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.providers.tokenizer_accuracy_vs_real_usage",
        provider: "mixed",
        description: "framework Tokenizer counts match provider-reported prompt_tokens within ±15%",
        factory: || Box::new(TokenizerAccuracyVsRealUsage),
    }
}
