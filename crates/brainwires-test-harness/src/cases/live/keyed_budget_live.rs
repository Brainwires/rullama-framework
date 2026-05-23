//! D.15 — `live.ollama.keyed_budget_isolates_concurrent_users`. Two concurrent
//! `tokio::spawn` tasks hit a real Ollama backend with distinct
//! `KeyedBudgetGuard` keys. Each gets a per-key quota of 3 calls. The
//! invariant: both users land their 3 calls (6 real Ollama calls total),
//! and a 4th call on either side fails pre-flight without invoking the
//! backend a 7th time. Catches per-key counter races that only surface
//! under real network latency.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_call_policy::{BudgetConfig, BudgetProvider, KeyedBudgetGuard};
use brainwires_core::{ChatOptions, Message, Provider};
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_provider::OllamaProvider;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct KeyedBudgetIsolatesConcurrentUsers;

async fn drain_user(
    provider: Arc<dyn Provider>,
    label: &'static str,
    quota: usize,
) -> Result<(usize, bool)> {
    let mut successes = 0usize;
    let msgs = vec![Message::user("Reply hi.")];
    let opts = ChatOptions::default().max_tokens(16);
    for i in 0..quota {
        if provider.chat(&msgs, None, &opts).await.is_ok() {
            successes += 1;
        } else {
            return Err(anyhow::anyhow!("{label} call #{i} failed within quota"));
        }
    }
    // Quota+1th call MUST fail pre-flight.
    let over = provider.chat(&msgs, None, &opts).await;
    let rejected = over.is_err();
    Ok((successes, rejected))
}

#[async_trait]
impl EvaluationCase for KeyedBudgetIsolatesConcurrentUsers {
    fn name(&self) -> &str {
        "live.ollama.keyed_budget_isolates_concurrent_users"
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

        let real = Arc::new(OllamaProvider::new(model.clone(), Some(base)));
        let kbg: KeyedBudgetGuard<String> = KeyedBudgetGuard::new(BudgetConfig {
            max_rounds: Some(3),
            ..Default::default()
        });

        let guard_a = kbg.for_key(&"user_a".to_string()).await;
        let guard_b = kbg.for_key(&"user_b".to_string()).await;
        let provider_a: Arc<dyn Provider> =
            Arc::new(BudgetProvider::new(real.clone(), guard_a.clone()));
        let provider_b: Arc<dyn Provider> =
            Arc::new(BudgetProvider::new(real.clone(), guard_b.clone()));

        let (a, b) = tokio::join!(
            tokio::spawn(drain_user(provider_a.clone(), "user_a", 3)),
            tokio::spawn(drain_user(provider_b.clone(), "user_b", 3)),
        );

        let elapsed = started.elapsed().as_millis() as u64;

        let (a_succ, a_rejected) = a??;
        let (b_succ, b_rejected) = b??;

        if a_succ != 3 || b_succ != 3 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("expected 3+3 successes; got a={a_succ} b={b_succ}"),
            ));
        }
        if !a_rejected || !b_rejected {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!(
                    "4th call should be rejected on both sides; a_rejected={a_rejected} b_rejected={b_rejected}"
                ),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("user_a_rounds", guard_a.rounds_consumed())
            .with_meta("user_b_rounds", guard_b.rounds_consumed())
            .with_meta("total_real_calls", (a_succ + b_succ) as u64))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.ollama.keyed_budget_isolates_concurrent_users",
        provider: "ollama",
        description: "two concurrent users on a real Ollama backend stay within their per-key quotas without interference",
        factory: || Box::new(KeyedBudgetIsolatesConcurrentUsers),
    }
}
