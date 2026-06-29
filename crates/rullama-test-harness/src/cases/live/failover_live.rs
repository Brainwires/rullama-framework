//! D.8 — `live.providers.failover_skips_dead_primary_real`. Pair the
//! framework's `FailoverProvider` with a primary pointing at a deliberately
//! invalid URL (so the request fails with a Network-class transient) and a
//! healthy Ollama secondary; assert the call succeeds and the response
//! comes from the secondary. Verifies the chain works end-to-end against a
//! real backend, not just scripted error fixtures.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_call_policy::FailoverProvider;
use rullama_core::{ChatOptions, Message, Provider};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_provider::OllamaProvider;

use crate::live::{live_ollama_base, live_ollama_model};
use crate::registry::LiveCase;

pub struct FailoverSkipsDeadPrimaryReal;

#[async_trait]
impl EvaluationCase for FailoverSkipsDeadPrimaryReal {
    fn name(&self) -> &str {
        "live.providers.failover_skips_dead_primary_real"
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

        // Primary: points at an unreachable port — every chat() fails with
        // a connection-refused / connection-reset error (Network-class →
        // transient → FailoverProvider advances).
        let dead_primary: Arc<dyn Provider> = Arc::new(OllamaProvider::new(
            model.clone(),
            Some("http://127.0.0.1:1".to_string()),
        ));
        // Secondary: the real local Ollama.
        let real_secondary: Arc<dyn Provider> =
            Arc::new(OllamaProvider::new(model.clone(), Some(base)));

        let chain = FailoverProvider::new(vec![dead_primary, real_secondary]);
        // Use treat_all_errors_as_transient=true so even if Network-class
        // classification slips, the chain still advances. Without it,
        // some kernels surface the connect failure as a string the
        // classifier doesn't recognise.
        let chain = chain.with_treat_all_errors_as_transient(true);

        let msgs = vec![Message::user("Reply with one word: hi")];
        let opts = ChatOptions::default().model(model).max_tokens(16);

        let resp = chain.chat(&msgs, None, &opts).await?;
        let elapsed = started.elapsed().as_millis() as u64;
        let text = resp.message.text().unwrap_or("").to_string();
        if text.trim().is_empty() {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "secondary returned empty text — failover succeeded but no usable reply",
            ));
        }
        Ok(TrialResult::success(trial_id, elapsed).with_meta("text_len", text.len()))
    }
}

inventory::submit! {
    LiveCase {
        id: "live.providers.failover_skips_dead_primary_real",
        provider: "ollama",
        description: "FailoverProvider with bad-URL primary + real Ollama secondary returns the secondary's response",
        factory: || Box::new(FailoverSkipsDeadPrimaryReal),
    }
}
