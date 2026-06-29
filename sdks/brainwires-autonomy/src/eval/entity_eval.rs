//! Empirical evaluation cases for entity importance scoring.
//!
//! These cases validate that [`RelationshipGraph::calculate_importance`]
//! produces correct relative orderings — not just that the formula compiles,
//! but that it ranks entities the way a reasonable agent would expect.
//!
//! All cases are deterministic (no LLM calls, no I/O) and complete in <1 ms.
//!
//! ## Cases
//!
//! | Case | What it validates |
//! |------|------------------|
//! | [`EntityImportanceRankingCase`] | Hub entities rank above peripheral ones |
//! | [`EntitySingleMentionCase`] | ln(1)=0 zero-contribution is compensated by type bonus |
//! | [`EntityTypeBonusCase`] | Type bonus ordering matches the hardcoded priority table |

use std::sync::Arc;

use async_trait::async_trait;
use brainwires_eval::{EvaluationCase, TrialResult, ndcg_at_k};
use brainwires_knowledge::knowledge::entity::{Entity, EntityType};
use brainwires_knowledge::knowledge::relationship_graph::RelationshipGraph;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a minimal `Entity` for eval purposes (no real timestamps needed).
fn make_entity(
    name: &str,
    entity_type: EntityType,
    mention_count: u32,
    n_messages: usize,
) -> Entity {
    let mut e = Entity::new(name.to_string(), entity_type, "msg-0".to_string(), 0);
    // Add extra mentions to reach the desired count.
    for i in 1..mention_count {
        e.add_mention(format!("msg-{i}"), i as i64);
    }
    // Ensure exactly n_messages unique message IDs (may differ from mention_count).
    e.message_ids = (0..n_messages).map(|i| format!("msg-{i}")).collect();
    e
}

// ── Case 1: Hub vs. peripheral ────────────────────────────────────────────────

/// Validates that hub entities (many mentions, many unique messages) score
/// higher than peripheral ones.
///
/// Scenario: 4 entities of the same type (`Concept`) with decreasing mention
/// counts. Expected ordering: hub > mid-a > mid-b > peripheral.
pub struct EntityImportanceRankingCase;

#[async_trait]
impl EvaluationCase for EntityImportanceRankingCase {
    fn name(&self) -> &str {
        "entity_importance_ranking"
    }
    fn category(&self) -> &str {
        "entity_resolution"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        let entities = [
            make_entity("hub", EntityType::Concept, 20, 10),
            make_entity("mid_a", EntityType::Concept, 5, 3),
            make_entity("mid_b", EntityType::Concept, 3, 2),
            make_entity("peripheral", EntityType::Concept, 1, 1),
        ];
        // Ground truth: hub=3, mid_a=2, mid_b=1, peripheral=0
        let ground_truth: Vec<usize> = vec![3, 2, 1, 0];

        let scores: Vec<f64> = entities
            .iter()
            .map(|e| RelationshipGraph::calculate_importance(e) as f64)
            .collect();

        let ndcg = ndcg_at_k(&scores, &ground_truth, 4);
        let ms = start.elapsed().as_millis() as u64;

        if ndcg >= 0.8 {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("ndcg", serde_json::json!(ndcg))
                .with_meta("scores", serde_json::json!(scores)))
        } else {
            Ok(TrialResult::failure(
                trial_id,
                ms,
                format!("NDCG@4={ndcg:.4} < 0.8 — importance formula does not correctly rank hub > peripheral"),
            )
            .with_meta("ndcg", serde_json::json!(ndcg))
            .with_meta("scores", serde_json::json!(scores)))
        }
    }
}

// ── Case 2: Single-mention entity ─────────────────────────────────────────────

/// Validates that a single-mention entity still has non-zero importance.
///
/// `ln(1) = 0`, so the mention-count component contributes nothing for
/// entities seen exactly once. The type bonus and message-spread proxy must
/// compensate. This case documents that guarantee and will catch any future
/// regression where the type bonus is removed.
pub struct EntitySingleMentionCase;

#[async_trait]
impl EvaluationCase for EntitySingleMentionCase {
    fn name(&self) -> &str {
        "entity_single_mention_nonzero"
    }
    fn category(&self) -> &str {
        "entity_resolution"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        // Use File (type bonus = 0.4) and Variable (type bonus = 0.1) — both
        // mention_count=1, 1 message. Even the weakest type must yield > 0.
        let file_entity = make_entity("single_file", EntityType::File, 1, 1);
        let var_entity = make_entity("single_var", EntityType::Variable, 1, 1);

        let file_score = RelationshipGraph::calculate_importance(&file_entity);
        let var_score = RelationshipGraph::calculate_importance(&var_entity);
        let ms = start.elapsed().as_millis() as u64;

        // ln(1)*0.3 = 0; type_bonus varies; recency_proxy = 1*0.05 = 0.05
        // File:     0.0 + 0.40 + 0.05 = 0.45
        // Variable: 0.0 + 0.10 + 0.05 = 0.15
        let detail = format!(
            "file_importance={file_score:.4} (expected ≈0.45), \
             variable_importance={var_score:.4} (expected ≈0.15). \
             Note: mention-count component is 0 for single-mention entities \
             due to ln(1)=0; type bonus compensates."
        );

        if file_score > 0.0 && var_score > 0.0 && file_score > var_score {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("file_importance", serde_json::json!(file_score))
                .with_meta("variable_importance", serde_json::json!(var_score))
                .with_meta("detail", serde_json::json!(detail)))
        } else {
            Ok(TrialResult::failure(trial_id, ms, detail))
        }
    }
}

// ── Case 3: Type bonus ordering ───────────────────────────────────────────────

/// Validates that the type-bonus ordering is respected when all other factors
/// are held constant (mention_count=1, 1 message each).
///
/// Expected ordering mirrors the hardcoded bonuses:
/// File(0.4) > Type(0.35) > Function(0.3) > Error(0.25) > Concept(0.2) > Command(0.15) > Variable(0.1)
pub struct EntityTypeBonusCase;

#[async_trait]
impl EvaluationCase for EntityTypeBonusCase {
    fn name(&self) -> &str {
        "entity_type_bonus_ordering"
    }
    fn category(&self) -> &str {
        "entity_resolution"
    }

    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        let start = std::time::Instant::now();

        // All identical except entity_type. Expected relevance = rank by type bonus descending.
        let types_ordered = [
            EntityType::File,     // bonus 0.40 → relevance 6
            EntityType::Type,     // bonus 0.35 → relevance 5
            EntityType::Function, // bonus 0.30 → relevance 4
            EntityType::Error,    // bonus 0.25 → relevance 3
            EntityType::Concept,  // bonus 0.20 → relevance 2
            EntityType::Command,  // bonus 0.15 → relevance 1
            EntityType::Variable, // bonus 0.10 → relevance 0
        ];
        let ground_truth: Vec<usize> = (0..7).rev().collect(); // [6,5,4,3,2,1,0]

        let scores: Vec<f64> = types_ordered
            .iter()
            .map(|et| {
                let e = make_entity("e", et.clone(), 1, 1);
                RelationshipGraph::calculate_importance(&e) as f64
            })
            .collect();

        let ndcg = ndcg_at_k(&scores, &ground_truth, 7);
        let ms = start.elapsed().as_millis() as u64;

        if ndcg >= 0.95 {
            Ok(TrialResult::success(trial_id, ms)
                .with_meta("ndcg", serde_json::json!(ndcg))
                .with_meta("scores", serde_json::json!(scores)))
        } else {
            Ok(TrialResult::failure(
                trial_id,
                ms,
                format!(
                    "NDCG@7={ndcg:.4} < 0.95 — type bonus ordering is incorrect. scores={scores:?}"
                ),
            )
            .with_meta("ndcg", serde_json::json!(ndcg)))
        }
    }
}

// ── Suite constructor ─────────────────────────────────────────────────────────

/// Return all entity importance eval cases ready for use with
/// [`brainwires_eval::EvaluationSuite`] or
/// [`brainwires_autonomy::self_improve::AutonomousFeedbackLoop`].
pub fn entity_importance_suite() -> Vec<Arc<dyn EvaluationCase>> {
    vec![
        Arc::new(EntityImportanceRankingCase),
        Arc::new(EntitySingleMentionCase),
        Arc::new(EntityTypeBonusCase),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ranking_case_passes() {
        let case = EntityImportanceRankingCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "EntityImportanceRankingCase failed: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn test_single_mention_case_passes() {
        let case = EntitySingleMentionCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "EntitySingleMentionCase failed: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn test_type_bonus_case_passes() {
        let case = EntityTypeBonusCase;
        let result = case.run(0).await.unwrap();
        assert!(
            result.success,
            "EntityTypeBonusCase failed: {:?}",
            result.error
        );
    }
}
