//! Empirical evaluation cases for agent allocation scoring.
//!
//! Validates that [`TaskBid::score`] and [`ResourceBid::score`] produce correct
//! relative orderings when each weight factor is varied independently. All cases
//! are deterministic (no I/O, no LLM calls) and complete in <1 ms.
//!
//! ## Cases
//!
//! | Case | Formula validated |
//! |------|------------------|
//! | [`TaskBidScoringCase`] | `0.4*capability + 0.3*availability + 0.3*speed` |
//! | [`ResourceBidScoringCase`] | `0.7*priority_factor + 0.3*bid_factor` |

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use brainwires_agent::contract_net::TaskBid;
use brainwires_agent::market_allocation::ResourceBid;
use brainwires_eval::{EvaluationCase, TrialResult, ndcg_at_k};

// ── Case 1: TaskBid scoring ───────────────────────────────────────────────────

/// Validates `TaskBid::score()` = `0.4 * capability + 0.3 * (1 - load) + 0.3 * speed`
/// where `speed = 1 / (1 + duration_secs / 60)`.
///
/// Three scenarios each isolate one factor while holding the other two constant.
/// A correct implementation must rank bids in strict order of the varying factor.
pub struct TaskBidScoringCase;

fn make_task_bid(capability_score: f32, current_load: f32, duration_secs: u64) -> TaskBid {
    TaskBid {
        agent_id: "eval-agent".to_string(),
        task_id: "eval-task".to_string(),
        capability_score,
        current_load,
        estimated_duration: Duration::from_secs(duration_secs),
        conditions: vec![],
        submitted_at: Instant::now(),
    }
}

#[async_trait]
impl EvaluationCase for TaskBidScoringCase {
    fn name(&self) -> &str {
        "task_bid_scoring"
    }
    fn category(&self) -> &str {
        "agent_allocation"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        // Scenario A — capability dominates (load=0.5, dur=60s fixed)
        // Expected scores: 0.70, 0.58, 0.46, 0.34
        let caps = [1.0f32, 0.7, 0.4, 0.1];
        let scores_a: Vec<f64> = caps
            .iter()
            .map(|&c| make_task_bid(c, 0.5, 60).score() as f64)
            .collect();
        let ndcg_a = ndcg_at_k(&scores_a, &[3, 2, 1, 0], 4);

        // Scenario B — availability dominates (cap=0.5, dur=60s fixed)
        // Expected scores: 0.65, 0.56, 0.47, 0.38
        let loads = [0.0f32, 0.3, 0.6, 0.9];
        let scores_b: Vec<f64> = loads
            .iter()
            .map(|&l| make_task_bid(0.5, l, 60).score() as f64)
            .collect();
        let ndcg_b = ndcg_at_k(&scores_b, &[3, 2, 1, 0], 4);

        // Scenario C — speed dominates (cap=0.5, load=0.5 fixed)
        // Expected scores: 0.65, 0.55, 0.45, 0.40
        let durations = [0u64, 30, 120, 300];
        let scores_c: Vec<f64> = durations
            .iter()
            .map(|&d| make_task_bid(0.5, 0.5, d).score() as f64)
            .collect();
        let ndcg_c = ndcg_at_k(&scores_c, &[3, 2, 1, 0], 4);

        let mean_ndcg = (ndcg_a + ndcg_b + ndcg_c) / 3.0;
        let ms = start.elapsed().as_millis() as u64;

        let threshold = 0.99;
        if ndcg_a >= threshold && ndcg_b >= threshold && ndcg_c >= threshold {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("ndcg_capability", serde_json::json!(ndcg_a))
                .with_meta("ndcg_availability", serde_json::json!(ndcg_b))
                .with_meta("ndcg_speed", serde_json::json!(ndcg_c))
                .with_meta("mean_ndcg", serde_json::json!(mean_ndcg)))
        } else {
            Ok(TrialResult::failure(
                trial_id,
                ms,
                format!(
                    "TaskBid ranking incorrect — \
                     NDCG(capability)={ndcg_a:.4}, \
                     NDCG(availability)={ndcg_b:.4}, \
                     NDCG(speed)={ndcg_c:.4}; \
                     all must be >= {threshold}"
                ),
            )
            .with_meta("mean_ndcg", serde_json::json!(mean_ndcg)))
        }
    }
}

// ── Case 2: ResourceBid scoring ───────────────────────────────────────────────

/// Validates `ResourceBid::score()` = `0.7 * (effective_priority / 10) + 0.3 * min(max_bid / 100, 1)`
/// where `effective_priority = base_priority * urgency_multiplier`.
///
/// Two scenarios: one varying base priority, one varying urgency multiplier.
pub struct ResourceBidScoringCase;

fn make_resource_bid(base_priority: u8, urgency_multiplier: f32, max_bid: u32) -> ResourceBid {
    ResourceBid {
        agent_id: "eval-agent".to_string(),
        resource_id: "eval-resource".to_string(),
        base_priority,
        urgency_multiplier,
        max_bid,
        urgency_reason: "eval".to_string(),
        estimated_duration: Duration::from_secs(60),
        submitted_at: Instant::now(),
    }
}

#[async_trait]
impl EvaluationCase for ResourceBidScoringCase {
    fn name(&self) -> &str {
        "resource_bid_scoring"
    }
    fn category(&self) -> &str {
        "agent_allocation"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        // Scenario A — priority dominates (urgency=1.0, bid=50 fixed)
        // Expected scores: 0.78, 0.57, 0.36, 0.22
        let bases = [9u8, 6, 3, 1];
        let scores_a: Vec<f64> = bases
            .iter()
            .map(|&b| make_resource_bid(b, 1.0, 50).score() as f64)
            .collect();
        let ndcg_a = ndcg_at_k(&scores_a, &[3, 2, 1, 0], 4);

        // Scenario B — urgency multiplier dominates (base=5, bid=50 fixed)
        // Expected scores: 0.85, 0.675, 0.50, 0.325
        let urgencies = [2.0f32, 1.5, 1.0, 0.5];
        let scores_b: Vec<f64> = urgencies
            .iter()
            .map(|&u| make_resource_bid(5, u, 50).score() as f64)
            .collect();
        let ndcg_b = ndcg_at_k(&scores_b, &[3, 2, 1, 0], 4);

        let mean_ndcg = (ndcg_a + ndcg_b) / 2.0;
        let ms = start.elapsed().as_millis() as u64;

        let threshold = 0.99;
        if ndcg_a >= threshold && ndcg_b >= threshold {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("ndcg_priority", serde_json::json!(ndcg_a))
                .with_meta("ndcg_urgency", serde_json::json!(ndcg_b))
                .with_meta("mean_ndcg", serde_json::json!(mean_ndcg)))
        } else {
            Ok(TrialResult::failure(
                trial_id,
                ms,
                format!(
                    "ResourceBid ranking incorrect — \
                     NDCG(priority)={ndcg_a:.4}, \
                     NDCG(urgency)={ndcg_b:.4}; \
                     both must be >= {threshold}"
                ),
            )
            .with_meta("mean_ndcg", serde_json::json!(mean_ndcg)))
        }
    }
}

// ── Suite constructor ─────────────────────────────────────────────────────────

/// Return all agent allocation eval cases ready for use with
/// [`brainwires_eval::EvaluationSuite`] or
/// [`brainwires_autonomy::self_improve::AutonomousFeedbackLoop`].
pub fn agent_scoring_suite() -> Vec<Arc<dyn EvaluationCase>> {
    vec![
        Arc::new(TaskBidScoringCase),
        Arc::new(ResourceBidScoringCase),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_bid_scoring_passes() {
        let case = TaskBidScoringCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "TaskBidScoringCase failed: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn test_resource_bid_scoring_passes() {
        let case = ResourceBidScoringCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "ResourceBidScoringCase failed: {:?}",
            result.error
        );
    }
}
