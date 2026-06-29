//! Empirical evaluation cases for complexity reasoning heuristics.
//!
//! Validates that [`ComplexityScorer::score_heuristic`] produces a correct
//! relative ranking across task descriptions of varying complexity. The case is
//! deterministic (no LLM calls, no I/O) and completes in <1 ms.
//!
//! ## Cases
//!
//! | Case | Formula validated |
//! |------|------------------|
//! | [`ComplexityHeuristicCase`] | `0.3 + Σ keyword_adjustments ± length_bonus` |

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::message::{ChatResponse, Message, StreamChunk};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;
use brainwires_eval::{EvaluationCase, TrialResult, ndcg_at_k};
use brainwires_reasoning::ComplexityScorer;
use futures::stream::BoxStream;

// ── Stub provider ─────────────────────────────────────────────────────────────

/// Minimal provider stub that satisfies `ComplexityScorer::new` without
/// making any network calls. `score_heuristic` never invokes `chat()` or
/// `stream_chat()`, so this stub will never be called.
struct StubProvider;

#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &str {
        "stub"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        unreachable!("StubProvider::chat — score_heuristic does not call the provider")
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        Box::pin(futures::stream::empty())
    }
}

// ── Case: Complexity heuristic ordering ───────────────────────────────────────

/// Validates that `ComplexityScorer::score_heuristic` ranks four task
/// descriptions in order of expected complexity.
///
/// Each description is chosen so that it triggers a distinct set of the
/// keyword-based adjustments, producing well-separated scores:
///
/// | Task | Key keywords | Expected score |
/// |------|-------------|----------------|
/// | T1 — architectural | architecture, distributed, concurrent, async, … | ≈ 1.0 (clamped) |
/// | T2 — moderate refactor | refactor, optimize, performance, multiple, careful | ≈ 0.90 |
/// | T3 — simple fix | (none; 11 words → no length penalty) | ≈ 0.30 |
/// | T4 — trivial | (none; 2 words → −0.1 length penalty) | ≈ 0.20 |
///
/// `success = NDCG@4 >= 0.99`.
pub struct ComplexityHeuristicCase;

#[async_trait]
impl EvaluationCase for ComplexityHeuristicCase {
    fn name(&self) -> &str {
        "complexity_heuristic_ordering"
    }
    fn category(&self) -> &str {
        "reasoning"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        let scorer = ComplexityScorer::new(Arc::new(StubProvider), "stub");

        // T1: triggers architecture(+0.2), distributed(+0.2), concurrent(+0.2),
        //     async(+0.1), security(+0.15), validate(+0.1), multiple(+0.1), design(+0.1)
        //     → 0.3 + 1.15 = 1.45 → clamped to 1.0
        let t1 = "Design a fully distributed concurrent async architecture with security validation across multiple components";

        // T2: triggers refactor(+0.15), optimize(+0.15), performance(+0.1),
        //     multiple(+0.1), careful(+0.1) — 11 words, no length adjustment
        //     → 0.3 + 0.60 = 0.90
        let t2 =
            "Refactor and optimize the performance of multiple components carefully and thoroughly";

        // T3: no complexity keywords — 11 words, no length adjustment → 0.30
        let t3 = "Fix the authentication bug found in the user login module";

        // T4: no complexity keywords — 2 words (<10) → −0.1 → 0.20
        let t4 = "Print hello";

        let tasks = [t1, t2, t3, t4];
        let ground_truth: Vec<usize> = vec![3, 2, 1, 0];

        let scores: Vec<f64> = tasks
            .iter()
            .map(|t| scorer.score_heuristic(t).score as f64)
            .collect();

        let ndcg = ndcg_at_k(&scores, &ground_truth, 4);
        let ms = start.elapsed().as_millis() as u64;

        if ndcg >= 0.99 {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("ndcg", serde_json::json!(ndcg))
                .with_meta("scores", serde_json::json!(scores)))
        } else {
            Ok(TrialResult::failure(
                trial_id,
                ms,
                format!(
                    "ComplexityHeuristic NDCG@4={ndcg:.4} < 0.99 — \
                     heuristic does not correctly order tasks by complexity. \
                     scores={scores:?}"
                ),
            )
            .with_meta("ndcg", serde_json::json!(ndcg))
            .with_meta("scores", serde_json::json!(scores)))
        }
    }
}

// ── Suite constructor ─────────────────────────────────────────────────────────

/// Return all reasoning eval cases ready for use with
/// [`brainwires_eval::EvaluationSuite`] or
/// [`brainwires_autonomy::self_improve::AutonomousFeedbackLoop`].
pub fn reasoning_eval_suite() -> Vec<Arc<dyn EvaluationCase>> {
    vec![Arc::new(ComplexityHeuristicCase)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complexity_heuristic_passes() {
        let case = ComplexityHeuristicCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "ComplexityHeuristicCase failed: {:?}",
            result.error
        );
    }
}
