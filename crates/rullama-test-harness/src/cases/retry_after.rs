//! Tier-A `feature.call_policy.retry_obeys_retry_after_header`. When the
//! upstream provider returns an error with a `Retry-After: N` hint, the
//! `RetryProvider` must wait close to N seconds (not the heuristic
//! exponential backoff). This test uses a synthetic provider that fails
//! once with a Retry-After hint of 1 second and then succeeds — total
//! wall-clock time should be ~1s (not the default ~2s heuristic).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{RetryPolicy, RetryProvider};
use rullama_core::message::Usage;
use rullama_core::{
    ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool,
};
use rullama_eval::{EvaluationCase, TrialResult};
use futures::stream::{self, BoxStream};

use crate::registry::TierACase;

struct OneErrThenOk {
    name: &'static str,
    remaining: AtomicU32,
    err_msg: &'static str,
}

#[async_trait]
impl Provider for OneErrThenOk {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let left = self.remaining.fetch_sub(1, Ordering::Relaxed);
        if left > 0 {
            return Err(anyhow::anyhow!("{}", self.err_msg));
        }
        Ok(ChatResponse {
            message: Message::assistant("ok"),
            usage: Usage::new(4, 2),
            finish_reason: Some("stop".into()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        Box::pin(stream::empty())
    }
}

pub struct RetryObeysRetryAfterHeader;

#[async_trait]
impl EvaluationCase for RetryObeysRetryAfterHeader {
    fn name(&self) -> &str {
        "feature.call_policy.retry_obeys_retry_after_header"
    }
    fn category(&self) -> &str {
        "feature"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let inner: Arc<dyn Provider> = Arc::new(OneErrThenOk {
            name: "rate-limited",
            remaining: AtomicU32::new(1),
            // The retry decorator's `parse_retry_after` finds the hint by
            // string search; this format matches what rullama-provider
            // emits when an Anthropic / OpenAI response carries
            // `Retry-After: 1` in its headers.
            err_msg: "OpenAI API error (429 Too Many Requests): retry-after: 1 — slow down",
        });

        let policy = RetryPolicy {
            max_attempts: 3,
            honor_retry_after: true,
            ..Default::default()
        };
        let retry = RetryProvider::new(inner.clone(), policy);

        let started = std::time::Instant::now();
        let resp = retry
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await?;
        let elapsed = started.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;

        if resp.message.text() != Some("ok") {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed_ms,
                "expected 'ok' response on second attempt",
            ));
        }

        // Retry-After hint was 1 second. The retry decorator should sleep
        // close to that — accept the window [800ms, 1800ms]. Heuristic
        // backoff at attempt=1 is typically 250ms, so values under 500ms
        // mean the hint was ignored; values over 2s mean the backoff
        // doubled on top of the hint (a pre-existing bug class).
        if elapsed_ms < 800 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed_ms,
                format!(
                    "retry happened too fast ({elapsed_ms}ms) — Retry-After hint ignored?"
                ),
            ));
        }
        if elapsed_ms > 1800 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed_ms,
                format!(
                    "retry waited too long ({elapsed_ms}ms) — heuristic backoff stacked on the hint?"
                ),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed_ms)
            .with_meta("retry_wait_ms", elapsed_ms)
            .with_meta("hint_seconds", 1))
    }
}

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::retry_after::RetryObeysRetryAfterHeader",
        crate_name: "rullama-call-policy",
        description: "RetryProvider waits ~Retry-After seconds when honor_retry_after is set",
        factory: || Box::new(RetryObeysRetryAfterHeader),
    }
}
