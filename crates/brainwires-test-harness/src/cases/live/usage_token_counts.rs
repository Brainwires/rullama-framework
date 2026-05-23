//! D.7 — `live.providers.usage_token_counts_nonzero`. All three providers
//! return `Usage { total_tokens > 0 }` on a non-empty response. Skipped per
//! provider when the corresponding key is missing — overall case skips only
//! when *all three* keys are absent. Otherwise the case asserts the
//! invariant for every configured provider.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use brainwires_core::{ChatOptions, Message, Provider, Usage};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider::{
    AnthropicChatProvider, AnthropicClient, OllamaProvider, OpenAiChatProvider, OpenAiClient,
};

use crate::live::{
    live_anthropic_key, live_anthropic_model, live_ollama_base, live_ollama_model,
    live_openai_key, live_openai_model,
};
use crate::registry::LiveCase;

pub struct UsageTokenCountsNonzero;

async fn check_one(name: &str, provider: &dyn Provider, opts: &ChatOptions) -> Result<Usage> {
    let messages = vec![Message::user("Reply with one word: hi")];
    let resp = provider
        .chat(&messages, None, opts)
        .await
        .with_context(|| format!("{name} chat call failed"))?;
    Ok(resp.usage)
}

#[async_trait]
impl EvaluationCase for UsageTokenCountsNonzero {
    fn name(&self) -> &str {
        "live.providers.usage_token_counts_nonzero"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let ollama_base = live_ollama_base();
        let openai_key = live_openai_key();
        let anthropic_key = live_anthropic_key();

        if ollama_base.is_none() && openai_key.is_none() && anthropic_key.is_none() {
            return Ok(TrialResult::skipped(
                trial_id,
                "no BRAINWIRES_LIVE_* env vars set for any provider",
            ));
        }

        let started = std::time::Instant::now();
        let mut failures = Vec::new();
        let mut totals: Vec<(String, u32)> = Vec::new();

        if let Some(base) = ollama_base {
            let model = live_ollama_model();
            let provider = OllamaProvider::new(model.clone(), Some(base));
            let opts = ChatOptions::default().model(model).max_tokens(16);
            match check_one("ollama", &provider, &opts).await {
                Ok(u) if u.total_tokens > 0 => totals.push(("ollama".into(), u.total_tokens)),
                Ok(u) => failures.push(format!("ollama: total_tokens={}", u.total_tokens)),
                Err(e) => failures.push(format!("ollama: {e}")),
            }
        }

        if let Some(key) = openai_key {
            let model = live_openai_model();
            let client = Arc::new(OpenAiClient::new(key, model.clone()));
            let provider = OpenAiChatProvider::new(client, model.clone());
            let opts = ChatOptions::default().model(model).max_tokens(128);
            match check_one("openai", &provider, &opts).await {
                Ok(u) if u.total_tokens > 0 => totals.push(("openai".into(), u.total_tokens)),
                Ok(u) => failures.push(format!("openai: total_tokens={}", u.total_tokens)),
                Err(e) => failures.push(format!("openai: {e}")),
            }
        }

        if let Some(key) = anthropic_key {
            let model = live_anthropic_model();
            let client = Arc::new(AnthropicClient::new(key, model.clone()));
            let provider = AnthropicChatProvider::new(client, model.clone());
            let opts = ChatOptions::default().model(model).max_tokens(16);
            match check_one("anthropic", &provider, &opts).await {
                Ok(u) if u.total_tokens > 0 => totals.push(("anthropic".into(), u.total_tokens)),
                Ok(u) => failures.push(format!("anthropic: total_tokens={}", u.total_tokens)),
                Err(e) => failures.push(format!("anthropic: {e}")),
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
        let providers_meta: Vec<_> = totals.iter().map(|(p, _)| p.as_str()).collect();
        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("providers_checked", serde_json::json!(providers_meta)))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.providers.usage_token_counts_nonzero",
        provider: "mixed",
        description: "every configured provider returns Usage.total_tokens > 0",
        factory: || Box::new(UsageTokenCountsNonzero),
    }
}
