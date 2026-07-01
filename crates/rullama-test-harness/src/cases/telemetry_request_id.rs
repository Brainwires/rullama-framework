//! Tier-A `feature.telemetry.cost_by_request_id_isolates_calls`: emit
//! `ProviderCall` events under two distinct `request_id`s into the SQLite
//! analytics sink, then prove `AnalyticsQuery::cost_by_request(id)` returns
//! totals scoped to that id alone (no cross-request leakage).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_telemetry::{AnalyticsCollector, AnalyticsEvent, AnalyticsQuery, SqliteAnalyticsSink};

use crate::registry::TierACase;

pub struct CostByRequestIdIsolatesCalls;

fn make_event(request_id: &str, prompt: u32, completion: u32, cost_usd: f64) -> AnalyticsEvent {
    AnalyticsEvent::ProviderCall {
        session_id: None,
        request_id: Some(request_id.to_string()),
        provider: "test".to_string(),
        model: "test-model".to_string(),
        prompt_tokens: prompt,
        completion_tokens: completion,
        duration_ms: 10,
        cost_usd,
        success: true,
        timestamp: chrono::Utc::now(),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        compliance: None,
    }
}

#[async_trait]
impl EvaluationCase for CostByRequestIdIsolatesCalls {
    fn name(&self) -> &str {
        "feature.telemetry.cost_by_request_id_isolates_calls"
    }
    fn category(&self) -> &str {
        "feature"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();
        let tmp = tempfile::tempdir()?;
        let db_path: PathBuf = tmp.path().join("analytics.db");

        let sink = SqliteAnalyticsSink::new_with_path(&db_path)?;
        let collector = AnalyticsCollector::new(vec![
            Box::new(sink) as Box<dyn rullama_telemetry::AnalyticsSink>
        ]);

        // Three events: two share req-A, one is req-B.
        collector.record(make_event("req-a", 100, 50, 0.01));
        collector.record(make_event("req-a", 200, 75, 0.02));
        collector.record(make_event("req-b", 50, 25, 0.005));

        // Allow the sink's background flush to complete. The collector
        // returns immediately from `record`; we need to wait for the
        // SQLite write to settle before querying.
        for _ in 0..40 {
            let q = AnalyticsQuery::new_with_path(&db_path)?;
            if let Some(a) = q.cost_by_request("req-a")?
                && a.call_count == 2
            {
                let b = q.cost_by_request("req-b")?;
                let elapsed = started.elapsed().as_millis() as u64;

                if a.total_prompt_tokens != 300 {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        format!(
                            "req-a prompt tokens: want 300, got {}",
                            a.total_prompt_tokens
                        ),
                    ));
                }
                if a.total_completion_tokens != 125 {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        format!(
                            "req-a completion tokens: want 125, got {}",
                            a.total_completion_tokens
                        ),
                    ));
                }
                if (a.total_cost_usd - 0.03).abs() > 1e-6 {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        format!("req-a cost: want 0.03, got {}", a.total_cost_usd),
                    ));
                }

                let b = b.ok_or_else(|| anyhow::anyhow!("req-b should have one row"))?;
                if b.call_count != 1
                    || b.total_prompt_tokens != 50
                    || b.total_completion_tokens != 25
                {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        format!(
                            "req-b row leaked req-a data: calls={} p={} c={}",
                            b.call_count, b.total_prompt_tokens, b.total_completion_tokens
                        ),
                    ));
                }

                // Unknown id returns None.
                if q.cost_by_request("does-not-exist")?.is_some() {
                    return Ok(TrialResult::failure(
                        trial_id,
                        elapsed,
                        "unknown request_id should return None",
                    ));
                }

                return Ok(TrialResult::success(trial_id, elapsed)
                    .with_meta("req_a_calls", a.call_count)
                    .with_meta("req_b_calls", b.call_count));
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        Ok(TrialResult::failure(
            trial_id,
            started.elapsed().as_millis() as u64,
            "events never landed in sqlite within 2s",
        ))
    }
}

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::telemetry_request_id::CostByRequestIdIsolatesCalls",
        crate_name: "rullama-telemetry",
        description: "cost_by_request(id) returns per-request totals with no cross-request leakage",
        factory: || Box::new(CostByRequestIdIsolatesCalls),
    }
}

// Force this case to be retained by the linker even when nothing in the
// surrounding module references it directly.
#[allow(dead_code)]
pub(crate) fn _force_link() -> Arc<dyn EvaluationCase> {
    Arc::new(CostByRequestIdIsolatesCalls)
}
