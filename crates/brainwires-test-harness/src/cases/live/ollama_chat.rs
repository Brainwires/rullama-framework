//! D.1 — `live.ollama.gemma_chat_roundtrip`: minimal chat roundtrip against
//! a real local Ollama server. Verifies the provider trait contract end-to-end.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{ChatOptions, Message, Provider};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider::OllamaProvider;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct OllamaChatRoundtrip;

#[async_trait]
impl EvaluationCase for OllamaChatRoundtrip {
    fn name(&self) -> &str {
        "live.ollama.gemma_chat_roundtrip"
    }
    fn category(&self) -> &str {
        "live"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let Some(base) = live_ollama_base() else {
            return Ok(TrialResult::skipped(
                trial_id,
                "BRAINWIRES_LIVE_OLLAMA_BASE not set",
            ));
        };
        let model = live_ollama_model();
        let started = std::time::Instant::now();
        let provider = OllamaProvider::new(model.clone(), Some(base));
        let messages = vec![Message::user("Reply with one word: hi")];
        let opts = ChatOptions::default().model(model).max_tokens(16);
        let response = provider.chat(&messages, None, &opts).await?;
        let elapsed = started.elapsed().as_millis() as u64;
        let text = response.message.text().unwrap_or("").to_string();
        if text.trim().is_empty() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "Ollama returned empty text",
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("text_len", text.len())
            .with_meta("model", live_ollama_model()))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.ollama.gemma_chat_roundtrip",
        provider: "ollama",
        description: "minimal chat roundtrip against a local Ollama server",
        factory: || Box::new(OllamaChatRoundtrip),
    }
}
