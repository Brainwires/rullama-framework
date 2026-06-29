//! D.2 — `live.openai.gpt5_nano_chat_roundtrip`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::{OpenAiChatProvider, OpenAiClient};

use crate::live::{live_openai_key, live_openai_model};
use crate::registry::LiveCase;

pub struct OpenAiChatRoundtrip;

#[async_trait]
impl EvaluationCase for OpenAiChatRoundtrip {
    fn name(&self) -> &str {
        "live.openai.gpt5_nano_chat_roundtrip"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(key) = live_openai_key() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "RULLAMA_LIVE_OPENAI_KEY not set",
            ));
        };
        let model = live_openai_model();
        let started = std::time::Instant::now();
        let client = Arc::new(OpenAiClient::new(key, model.clone()));
        let provider = OpenAiChatProvider::new(client, model.clone());
        // Concrete deterministic prompt — gpt-5-nano sometimes returns
        // empty visible content for vague single-word prompts because
        // the reasoning step decides "no output needed". An explicit
        // sentence-shape ask is far more reliable.
        let messages = vec![Message::user(
            "What is the capital city of France? Reply in exactly one word.",
        )];
        // gpt-5 reasoning models spend tokens on internal chain-of-thought
        // before producing visible output, so a small cap leaves nothing for
        // the visible answer. 1024 keeps cost negligible (well under a
        // cent) while giving the model enough headroom to finish reasoning.
        let opts = ChatOptions::default().model(model).max_tokens(1024);
        let response = provider.chat(&messages, None, &opts).await?;
        let elapsed = started.elapsed().as_millis() as u64;
        let text = response.message.text().unwrap_or("").to_string();
        if text.trim().is_empty() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "OpenAI returned empty text",
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed).with_meta("text_len", text.len()))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.openai.gpt5_nano_chat_roundtrip",
        provider: "openai",
        description: "minimal chat roundtrip against the OpenAI gpt-5-nano model",
        factory: || Box::new(OpenAiChatRoundtrip),
    }
}
