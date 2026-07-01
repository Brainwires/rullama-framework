//! Tier-B `sec.call_policy.failover_skips_dead_primary`. A
//! `FailoverProvider` with a dead primary (always returns
//! `connection reset by peer` — a Network-class transient) and a healthy
//! secondary must short-circuit to the secondary and return its response,
//! never invoking it more than once. Also verifies a permanent error
//! (`401 unauthorized`) aborts the chain without consulting the
//! secondary — surfacing the auth/quota issue rather than silently
//! masking it.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::FailoverProvider;
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_test_fixtures::{FailingProvider, RecordingProvider, ScriptedProvider};

use crate::registry::SecurityCase;

pub struct FailoverSkipsDeadPrimary;

#[async_trait]
impl EvaluationCase for FailoverSkipsDeadPrimary {
    fn name(&self) -> &str {
        "sec.call_policy.failover_skips_dead_primary"
    }
    fn category(&self) -> &str {
        "security"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();

        // Branch 1: transient primary failure → secondary succeeds, was called once.
        let primary_dead: Arc<dyn Provider> =
            Arc::new(FailingProvider::new("connection reset by peer"));
        let secondary_scripted = ScriptedProvider::always_text("secondary", "fallback-ok");
        let secondary_recorder = Arc::new(RecordingProvider::new(secondary_scripted));
        let secondary: Arc<dyn Provider> = secondary_recorder.clone();

        let f = FailoverProvider::new(vec![primary_dead, secondary]);
        let resp = f
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await?;
        let text = resp.message.text().unwrap_or("").to_string();
        if text != "fallback-ok" {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("expected secondary response 'fallback-ok'; got {text:?}"),
            ));
        }
        let secondary_calls = secondary_recorder.calls().len();
        if secondary_calls != 1 {
            return Ok(TrialResult::failure(
                trial_id,
                started.elapsed().as_millis() as u64,
                format!("secondary called {secondary_calls} times, want exactly 1"),
            ));
        }

        // Branch 2: permanent primary failure → chain aborts; secondary NEVER called.
        let primary_unauth: Arc<dyn Provider> = Arc::new(FailingProvider::new("401 unauthorized"));
        let secondary2_scripted = ScriptedProvider::always_text("secondary2", "never");
        let secondary2_recorder = Arc::new(RecordingProvider::new(secondary2_scripted));
        let secondary2: Arc<dyn Provider> = secondary2_recorder.clone();

        let f2 = FailoverProvider::new(vec![primary_unauth, secondary2]);
        let result = f2
            .chat(&[Message::user("hi")], None, &ChatOptions::default())
            .await;
        let elapsed = started.elapsed().as_millis() as u64;
        if result.is_ok() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "permanent error should abort the chain but FailoverProvider returned Ok",
            ));
        }
        let calls2 = secondary2_recorder.calls().len();
        if calls2 != 0 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("secondary invoked {calls2} times on permanent primary error; want 0"),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("secondary_called_on_transient", true)
            .with_meta("secondary_called_on_permanent", false))
    }
}

inventory::submit! {
    SecurityCase {
        id: "sec.call_policy.failover_skips_dead_primary",
        crate_name: "rullama-call-policy",
        invariant: "FailoverProvider advances to next provider on transient errors and short-circuits on permanent",
        factory: || Box::new(FailoverSkipsDeadPrimary),
    }
}
