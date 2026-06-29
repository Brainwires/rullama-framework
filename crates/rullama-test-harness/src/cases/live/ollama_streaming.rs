//! D.4 — `live.ollama.streaming_emits_text_and_done`.

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, Message, Provider, StreamChunk};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::OllamaProvider;
use futures::StreamExt;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct OllamaStreaming;

#[async_trait]
impl EvaluationCase for OllamaStreaming {
    fn name(&self) -> &str {
        "live.ollama.streaming_emits_text_and_done"
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
        let provider = OllamaProvider::new(model.clone(), Some(base));
        let messages = vec![Message::user("Say hello in three words.")];
        let opts = ChatOptions::default().model(model).max_tokens(32);

        let mut stream = provider.stream_chat(&messages, None, &opts);
        let mut text_chunks = 0usize;
        let mut saw_done = false;
        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Text(_) => text_chunks += 1,
                StreamChunk::Done => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }
        let elapsed = started.elapsed().as_millis() as u64;
        if text_chunks == 0 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "no StreamChunk::Text emitted",
            ));
        }
        if !saw_done {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "stream ended without StreamChunk::Done",
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed).with_meta("text_chunks", text_chunks))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.ollama.streaming_emits_text_and_done",
        provider: "ollama",
        description: "streaming completion against Ollama emits ≥1 Text then Done",
        factory: || Box::new(OllamaStreaming),
    }
}
