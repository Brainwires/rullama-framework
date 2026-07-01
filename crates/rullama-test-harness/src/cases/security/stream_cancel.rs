//! Tier-B `sec.provider.stream_cancel_aborts_inflight_request`. Build a
//! Provider that emits one `StreamChunk::Text` every 50 ms forever; attach a
//! `CancellationToken` to `ChatOptions::cancel`; cancel after collecting 2
//! chunks; assert no further chunks arrive. Confirms the framework's
//! cooperative-cancellation contract is respected by the streaming
//! select-loop pattern (cancel takes priority over the next chunk).

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use rullama_core::{ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool};
use rullama_eval::{EvaluationCase, TrialResult};
use tokio_util::sync::CancellationToken;

use crate::registry::SecurityCase;

/// Provider that emits a fresh `Text` chunk every 50 ms. `chat()` is
/// stubbed since the case only exercises `stream_chat`. The provider
/// honours `options.cancel`: while waiting for the next chunk it races
/// the sleep against `token.cancelled()` and bails out on cancel.
struct SlowStreamProvider;

#[async_trait]
impl Provider for SlowStreamProvider {
    fn name(&self) -> &str {
        "slow-stream"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> anyhow::Result<ChatResponse> {
        anyhow::bail!("chat() not implemented by SlowStreamProvider")
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, anyhow::Result<StreamChunk>> {
        let cancel = options.cancel.clone();
        Box::pin(async_stream::stream! {
            let mut idx = 0u32;
            loop {
                let sleep = tokio::time::sleep(std::time::Duration::from_millis(50));
                tokio::pin!(sleep);
                if let Some(ref token) = cancel {
                    tokio::select! {
                        biased;
                        _ = token.cancelled() => { break; }
                        _ = &mut sleep => {}
                    }
                } else {
                    sleep.await;
                }
                idx += 1;
                yield Ok(StreamChunk::Text(format!("chunk-{idx}")));
            }
        })
    }
}

pub struct StreamCancelAbortsInflight;

#[async_trait]
impl EvaluationCase for StreamCancelAbortsInflight {
    fn name(&self) -> &str {
        "sec.provider.stream_cancel_aborts_inflight_request"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();
        let provider: Arc<dyn Provider> = Arc::new(SlowStreamProvider);

        let token = CancellationToken::new();
        let opts = ChatOptions::default().cancel_with(token.clone());
        let msgs = vec![Message::user("anything")];

        let mut stream = provider.stream_chat(&msgs, None, &opts);

        // Collect 2 chunks.
        let mut collected = 0usize;
        while let Some(chunk) = stream.next().await {
            let _ = chunk?;
            collected += 1;
            if collected == 2 {
                break;
            }
        }
        if collected != 2 {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("expected 2 pre-cancel chunks, got {collected}"),
            ));
        }

        // Cancel.
        token.cancel();

        // Pull at most a few more times; the stream should end quickly.
        let mut post_cancel = 0usize;
        let mut saw_end = false;
        for _ in 0..6 {
            match tokio::time::timeout(std::time::Duration::from_millis(200), stream.next()).await {
                Ok(Some(_chunk)) => {
                    post_cancel += 1;
                }
                Ok(None) => {
                    saw_end = true;
                    break;
                }
                Err(_) => {
                    // Stream hung — that's a failure: cancel should drive
                    // the stream to end promptly.
                    return Ok(TrialResult::failure(
                        trial_id,
                        started.elapsed().as_millis() as u64,
                        "stream did not end within 200ms of cancel",
                    ));
                }
            }
        }

        let elapsed = started.elapsed().as_millis() as u64;
        if !saw_end {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("stream did not yield None after cancel; got {post_cancel} extra chunks"),
            ));
        }
        // One chunk in flight at cancel time is fine; more than that means
        // the provider isn't checking the token between yields.
        if post_cancel > 1 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("too many post-cancel chunks: {post_cancel} (expected ≤1)"),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("pre_cancel", collected as u64)
            .with_meta("post_cancel", post_cancel as u64))
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.provider.stream_cancel_aborts_inflight_request",
        crate_name: "rullama-core",
        invariant: "ChatOptions::cancel terminates an in-flight stream_chat within one chunk-delay window",
        factory: || Box::new(StreamCancelAbortsInflight),
    }
}
