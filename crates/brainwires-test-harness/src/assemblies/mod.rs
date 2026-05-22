//! Tier-C golden-path assemblies.
//!
//! Seven deterministic end-to-end scenarios that prove common feature
//! combinations work together. Listed manually here (not via inventory)
//! because there are only seven and explicit listing is easier to scan.

use std::sync::Arc;

use brainwires_eval::EvaluationCase;

pub mod call_policy_with_scripted_provider;

/// Every assembly the harness knows about.
pub fn all() -> Vec<Arc<dyn EvaluationCase>> {
    vec![Arc::from(call_policy_with_scripted_provider::case())]
}
