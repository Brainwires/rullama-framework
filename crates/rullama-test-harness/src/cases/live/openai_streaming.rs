//! D.5 — `live.openai.streaming_emits_text_and_done`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{ChatOptions, Message, Provider, StreamChunk};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::{OpenAiChatProvider, OpenAiClient};
use futures::StreamExt;

use crate::live::{live_openai_key, live_openai_model};
use crate::registry::LiveCase;

pub struct OpenAiStreaming;

#[async_trait]
impl EvaluationCase for OpenAiStreaming {
    fn name(&self) -> &str {
        "live.openai.streaming_emits_text_and_done"
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
        id: "live.openai.streaming_emits_text_and_done",
        provider: "openai",
        description: "streaming completion against OpenAI emits ≥1 Text then Done",
        factory: || Box::new(OpenAiStreaming),
    }
}
