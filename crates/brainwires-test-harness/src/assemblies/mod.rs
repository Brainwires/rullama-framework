//! Tier-C golden-path assemblies.
//!
//! Seven deterministic end-to-end scenarios that prove common feature
//! combinations work together. Listed manually here (not via inventory)
//! because there are only seven and explicit listing is easier to scan.

use std::sync::Arc;

use brainwires_eval::EvaluationCase;

pub mod call_policy_with_scripted_provider;
pub mod recording_budget_scripted;
pub mod skills_sign_verify;

/// Every assembly the harness knows about.
pub fn all() -> Vec<Arc<dyn EvaluationCase>> {
    vec![
        Arc::from(call_policy_with_scripted_provider::case()),
        Arc::from(recording_budget_scripted::case()),
        Arc::from(skills_sign_verify::case()),
    ]
}
