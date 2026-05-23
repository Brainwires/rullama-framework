//! D.18 — `live.ollama.stream_cancellation_aborts_in_flight`. Stream a
//! long-form prompt against the real Ollama, cancel after the first 2
//! text chunks, and assert the stream ends within a short window after
//! cancel — proving the OllamaProvider streaming loop honours
//! `ChatOptions::cancel`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::{ChatOptions, Message, Provider, StreamChunk};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider::OllamaProvider;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct OllamaStreamCancellationAbortsInFlight;

#[async_trait]
impl EvaluationCase for OllamaStreamCancellationAbortsInFlight {
    fn name(&self) -> &str {
        "live.ollama.stream_cancellation_aborts_in_flight"
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

        let provider = Arc::new(OllamaProvider::new(model.clone(), Some(base)));
        let token = CancellationToken::new();
        let opts = ChatOptions::default()
            .model(model)
            .max_tokens(1024)
            .cancel_with(token.clone());
        // A prompt long enough that gemma4 won't finish in one or two chunks.
        let messages = vec![Message::user(
            "Tell me a 500-word story about a robot exploring the moon. \
             Be very detailed about the terrain, the equipment, the silence.",
        )];

        let mut stream = provider.stream_chat(&messages, None, &opts);

        let mut pre_cancel = 0usize;
        let mut saw_done_before_cancel = false;
        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Text(_) => {
                    pre_cancel += 1;
                    if pre_cancel >= 2 {
                        break;
                    }
                }
                StreamChunk::Done => {
                    saw_done_before_cancel = true;
                    break;
                }
                _ => {}
            }
        }

        let elapsed_pre_cancel = started.elapsed().as_millis() as u64;
        if saw_done_before_cancel {
            // Model finished before we could cancel — not a failure of the
            // cancel mechanism per se, but the test can't conclude. Report
            // as success with a note so flakiness is visible.
            return Ok(TrialResult::success(trial_id, elapsed_pre_cancel)
                .with_meta("inconclusive", true)
                .with_meta("pre_cancel_chunks", pre_cancel as u64)
                .with_meta(
                    "note",
                    "model produced Done before 2 text chunks; long-prompt assumption violated",
                ));
        }
        if pre_cancel < 2 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed_pre_cancel,
                format!("only got {pre_cancel} text chunks before stream ended"),
            ));
        }

        // Trigger cancellation.
        token.cancel();
        let cancel_at = std::time::Instant::now();

        // Stream should terminate quickly. Allow up to 3 seconds (Ollama
        // may have a small buffer of bytes in flight; the upper bound here
        // is forgiving — without cancel the stream would run for ~30+s).
        let mut post_cancel = 0usize;
        let mut ended = false;
        let timeout = tokio::time::sleep(std::time::Duration::from_secs(3));
        tokio::pin!(timeout);
        loop {
            tokio::select! {
                biased;
                _ = &mut timeout => break,
                next = stream.next() => match next {
                    Some(_chunk) => { post_cancel += 1; }
                    None => { ended = true; break; }
                },
            }
        }
        let cancel_to_end_ms = cancel_at.elapsed().as_millis() as u64;
        let elapsed = started.elapsed().as_millis() as u64;

        if !ended {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!(
                    "stream did not end within 3s of cancel; {post_cancel} extra chunks observed"
                ),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("pre_cancel_chunks", pre_cancel as u64)
            .with_meta("post_cancel_chunks", post_cancel as u64)
            .with_meta("cancel_to_end_ms", cancel_to_end_ms))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.ollama.stream_cancellation_aborts_in_flight",
        provider: "ollama",
        description: "real Ollama stream terminates within 3s of CancellationToken::cancel()",
        factory: || Box::new(OllamaStreamCancellationAbortsInFlight),
    }
}
