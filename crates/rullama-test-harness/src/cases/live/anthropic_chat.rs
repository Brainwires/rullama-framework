//! D.3 — `live.anthropic.haiku_chat_roundtrip`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::{AnthropicChatProvider, AnthropicClient};

use crate::live::{live_anthropic_key, live_anthropic_model};
use crate::registry::LiveCase;

pub struct AnthropicChatRoundtrip;

#[async_trait]
impl EvaluationCase for AnthropicChatRoundtrip {
    fn name(&self) -> &str {
        "live.anthropic.haiku_chat_roundtrip"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(key) = live_anthropic_key() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "RULLAMA_LIVE_ANTHROPIC_KEY not set",
            ));
        };
        let model = live_anthropic_model();
        let started = std::time::Instant::now();
        let client = Arc::new(AnthropicClient::new(key, model.clone()));
        let provider = AnthropicChatProvider::new(client, model.clone());
        let messages = vec![Message::user("Reply with one word: hi")];
        let opts = ChatOptions::default().model(model).max_tokens(16);
        let response = provider.chat(&messages, None, &opts).await?;
        let elapsed = started.elapsed().as_millis() as u64;
        let text = response.message.text().unwrap_or("").to_string();
        if text.trim().is_empty() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "Anthropic returned empty text",
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed).with_meta("text_len", text.len()))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.anthropic.haiku_chat_roundtrip",
        provider: "anthropic",
        description: "minimal chat roundtrip against the Anthropic claude-haiku-4-5 model",
        factory: || Box::new(AnthropicChatRoundtrip),
    }
}
