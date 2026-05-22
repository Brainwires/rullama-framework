//! Tier-B adversarial cases for `brainwires_provider_speech::RateLimiter`.
//!
//! The rate limiter is the only shared cross-provider security primitive in
//! the speech crate — every cloud TTS/STT client (Azure, Cartesia, Deepgram,
//! ElevenLabs, Fish, Google, Murf) wraps its outbound calls in it. Bugs in
//! the bucket invalidate per-provider quota assumptions for ALL of them, so
//! this is the right test target.
//!
//! Invariants:
//! - `RateLimiter::new(0)` advertises zero tokens. The caller's intent is
//!   "block all requests"; the limiter must not silently hand out tokens.
//! - `acquire()` decrements `available_tokens()` by exactly one per call —
//!   no off-by-one in the CAS retry loop.
//! - Concurrent `acquire()` calls never over-allocate beyond `max_tokens`
//!   per burst. (Static budget; once a refill happens further calls can
//!   succeed, but no burst can exceed the configured ceiling.)
//! - `max_requests_per_minute` reports the configured cap unchanged across
//!   the limiter's lifetime — no silent re-tuning.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider_speech::rate_limiter::RateLimiter;

use crate::registry::SecurityCase;

// ── sec.speech.rate_limit_zero_blocks ───────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.speech.rate_limit_zero_blocks",
        crate_name: "brainwires-provider-speech",
        invariant: "RateLimiter::new(0) advertises zero tokens — `0 RPM` must mean `no traffic`, not `unbounded`",
        factory: || Box::new(RateLimitZeroBlocksCase),
    }
}

struct RateLimitZeroBlocksCase;

#[async_trait]
impl EvaluationCase for RateLimitZeroBlocksCase {
    fn name(&self) -> &str {
        "sec.speech.rate_limit_zero_blocks"
    }
    fn category(&self) -> &str {
        "security.provider_speech"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let lim = RateLimiter::new(0);
        if lim.available_tokens() != 0 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "RateLimiter::new(0) reports {} available tokens — expected 0",
                    lim.available_tokens()
                ),
            ));
        }
        if lim.max_requests_per_minute() != 0 {
            return Ok(TrialResult::failure(
                0,
                0,
                "RateLimiter::new(0) silently changed max_requests_per_minute",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.speech.rate_limit_acquire_decrements ────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.speech.rate_limit_acquire_decrements",
        crate_name: "brainwires-provider-speech",
        invariant: "Each acquire() consumes exactly one token — no off-by-one in the CAS retry loop",
        factory: || Box::new(RateLimitAcquireDecrementsCase),
    }
}

struct RateLimitAcquireDecrementsCase;

#[async_trait]
impl EvaluationCase for RateLimitAcquireDecrementsCase {
    fn name(&self) -> &str {
        "sec.speech.rate_limit_acquire_decrements"
    }
    fn category(&self) -> &str {
        "security.provider_speech"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let lim = RateLimiter::new(10);
        let initial = lim.available_tokens();
        for expected in (0..initial).rev() {
            lim.acquire().await;
            let actual = lim.available_tokens();
            if actual != expected {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("after acquire(): expected {expected} tokens, got {actual}"),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.speech.rate_limit_no_burst_overrun ──────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.speech.rate_limit_no_burst_overrun",
        crate_name: "brainwires-provider-speech",
        invariant: "Concurrent acquire() never hands out more tokens than `max_tokens` in a single burst",
        factory: || Box::new(RateLimitNoBurstOverrunCase),
    }
}

struct RateLimitNoBurstOverrunCase;

#[async_trait]
impl EvaluationCase for RateLimitNoBurstOverrunCase {
    fn name(&self) -> &str {
        "sec.speech.rate_limit_no_burst_overrun"
    }
    fn category(&self) -> &str {
        "security.provider_speech"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let lim = Arc::new(RateLimiter::new(5));
        // Spawn 5 tasks that race to drain the bucket. None must block on a
        // refill — the bucket should be exactly empty afterward.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let l = Arc::clone(&lim);
            handles.push(tokio::spawn(async move {
                tokio::time::timeout(std::time::Duration::from_millis(500), l.acquire()).await
            }));
        }
        for h in handles {
            match h.await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    return Ok(TrialResult::failure(
                        0,
                        0,
                        "acquire() timed out within initial burst — over-counting bucket",
                    ));
                }
                Err(e) => return Err(e.into()),
            }
        }
        let remaining = lim.available_tokens();
        if remaining != 0 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "after 5 concurrent acquires on a 5-RPM limiter, {remaining} tokens remain — \
                    the bucket is either over-counted or refilled too quickly"
                ),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.speech.rate_limit_max_is_stable ─────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.speech.rate_limit_max_is_stable",
        crate_name: "brainwires-provider-speech",
        invariant: "max_requests_per_minute() does not change after acquire/refill cycles",
        factory: || Box::new(RateLimitMaxStableCase),
    }
}

struct RateLimitMaxStableCase;

#[async_trait]
impl EvaluationCase for RateLimitMaxStableCase {
    fn name(&self) -> &str {
        "sec.speech.rate_limit_max_is_stable"
    }
    fn category(&self) -> &str {
        "security.provider_speech"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let lim = RateLimiter::new(7);
        let cap_before = lim.max_requests_per_minute();
        for _ in 0..7 {
            lim.acquire().await;
        }
        let cap_after = lim.max_requests_per_minute();
        if cap_before != cap_after {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "max_requests_per_minute drifted from {cap_before} to {cap_after} after draining",
                ),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
