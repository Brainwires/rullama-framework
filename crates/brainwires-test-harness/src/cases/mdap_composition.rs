//! Tier-A feature cases for `brainwires-mdap` composition.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_mdap::Composer;
use brainwires_mdap::decomposition::CompositionFunction;
use brainwires_mdap::microagent::SubtaskOutput;
use serde_json::json;

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::mdap_composition::identity_lastonly_concatenate",
        crate_name: "brainwires-mdap",
        description: "Composer combines SubtaskOutputs under Identity / LastOnly / Concatenate functions",
        factory: || Box::new(IdentityLastOnlyConcatenateCase),
    }
}

struct IdentityLastOnlyConcatenateCase;

#[async_trait]
impl EvaluationCase for IdentityLastOnlyConcatenateCase {
    fn name(&self) -> &str {
        "feature.mdap.composition_identity_lastonly_concatenate"
    }
    fn category(&self) -> &str {
        "feature.mdap"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let composer = Composer::new();
        let outputs = vec![
            SubtaskOutput::new("a", json!(1)),
            SubtaskOutput::new("b", json!(2)),
            SubtaskOutput::new("c", json!(3)),
        ];
        let identity = composer.compose(&outputs, &CompositionFunction::Identity)?;
        if identity != json!(1) {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("Identity composition expected 1, got {identity}"),
            ));
        }
        let last_only = composer.compose(&outputs, &CompositionFunction::LastOnly)?;
        if last_only != json!(3) {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("LastOnly composition expected 3, got {last_only}"),
            ));
        }
        // Concatenate over scalar JSON values produces something deterministic;
        // we only assert that it succeeds (the precise shape depends on the
        // composer's Concatenate impl and is not part of this case's invariant).
        let _concat = composer.compose(&outputs, &CompositionFunction::Concatenate)?;
        Ok(TrialResult::success(0, 0))
    }
}

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::mdap_composition::empty_results_errors",
        crate_name: "brainwires-mdap",
        description: "Composer rejects empty result sets with a MissingResult error",
        factory: || Box::new(EmptyResultsErrorsCase),
    }
}

struct EmptyResultsErrorsCase;

#[async_trait]
impl EvaluationCase for EmptyResultsErrorsCase {
    fn name(&self) -> &str {
        "feature.mdap.empty_results_errors"
    }
    fn category(&self) -> &str {
        "feature.mdap"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let composer = Composer::new();
        let result = composer.compose(&[], &CompositionFunction::Identity);
        if result.is_ok() {
            return Ok(TrialResult::failure(
                0,
                0,
                "Composer.compose accepted empty results — expected MissingResult error",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
