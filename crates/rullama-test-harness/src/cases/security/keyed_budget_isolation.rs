//! Tier-B `sec.call_policy.keyed_budget_isolates_users`: two concurrent
//! callers with distinct keys must not interfere with each other's quota.
//! Exhausting user A's per-key budget must not affect user B; concurrent
//! tick races on the same key must consistently honour the cap.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::{BudgetConfig, BudgetProvider, KeyedBudgetGuard};
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_test_fixtures::ScriptedProvider;

use crate::registry::SecurityCase;

pub struct KeyedBudgetIsolatesUsers;

#[async_trait]
impl EvaluationCase for KeyedBudgetIsolatesUsers {
    fn name(&self) -> &str {
        "sec.call_policy.keyed_budget_isolates_users"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();
        let scripted = Arc::new(ScriptedProvider::always_text("scripted", "ok"));

        let kbg: KeyedBudgetGuard<String> = KeyedBudgetGuard::new(BudgetConfig {
            max_rounds: Some(3),
            ..Default::default()
        });

        let guard_a = kbg.for_key(&"alice".to_string()).await;
        let guard_b = kbg.for_key(&"bob".to_string()).await;

        let provider_a: Arc<dyn Provider> =
            Arc::new(BudgetProvider::new(scripted.clone(), guard_a.clone()));
        let provider_b: Arc<dyn Provider> =
            Arc::new(BudgetProvider::new(scripted.clone(), guard_b.clone()));

        let msgs = vec![Message::user("hi")];
        let opts = ChatOptions::default();

        // 1) Alice exhausts her per-key quota (3 calls).
        for _ in 0..3 {
            provider_a.chat(&msgs, None, &opts).await?;
        }
        // 2) Alice's 4th call must fail pre-flight.
        let alice_fourth = provider_a.chat(&msgs, None, &opts).await;
        if alice_fourth.is_ok() {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "alice's 4th call should be rejected (max_rounds=3) but succeeded",
            ));
        }
        // 3) Bob is unaffected — still has full quota.
        for i in 0..3 {
            if let Err(e) = provider_b.chat(&msgs, None, &opts).await {
                return Ok(TrialResult::failure(
                    trial_id,
                    started.elapsed().as_millis() as u64,
                    format!("bob's call #{} should succeed but failed: {e}", i + 1),
                ));
            }
        }
        // 4) Bob's 4th call must fail too.
        if provider_b.chat(&msgs, None, &opts).await.is_ok() {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                "bob's 4th call should be rejected but succeeded",
            ));
        }

        // 5) Concurrent same-key race: spawn 8 tasks, all hitting the same
        //    new key with a tight cap of 5 calls. Exactly 5 must succeed
        //    and 3 must be rejected. Verifies the atomic counter is honest
        //    under contention (catches a Mutex misuse that allows extra ticks).
        let race_kbg: KeyedBudgetGuard<String> = KeyedBudgetGuard::new(BudgetConfig {
            max_rounds: Some(5),
            ..Default::default()
        });
        let race_guard = race_kbg.for_key(&"raced".to_string()).await;
        let race_provider: Arc<dyn Provider> = Arc::new(BudgetProvider::new(
            scripted.clone(),
            race_guard.clone(),
        ));
        let successes = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let p = race_provider.clone();
            let s = successes.clone();
            handles.push(tokio::spawn(async move {
                let msg = vec![Message::user("x")];
                if p.chat(&msg, None, &ChatOptions::default()).await.is_ok() {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }
        for h in handles {
            h.await?;
        }
        let n = successes.load(Ordering::SeqCst);
        let elapsed = started.elapsed().as_millis() as u64;
        if n != 5 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("race: expected exactly 5/8 successes (cap=5), got {n}"),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("alice_consumed", guard_a.rounds_consumed())
            .with_meta("bob_consumed", guard_b.rounds_consumed())
            .with_meta("race_successes", n as u64))
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.keyed_budget_isolates_users",
        crate_name: "rullama-call-policy",
        invariant: "KeyedBudgetGuard isolates per-key quotas, no leakage across keys, atomic under concurrent same-key ticks",
        factory: || Box::new(KeyedBudgetIsolatesUsers),
    }
}
