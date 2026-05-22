//! Tier-C golden-path assemblies.
//!
//! Seven deterministic end-to-end scenarios that prove common feature
//! combinations work together. Listed manually here (not via inventory)
//! because there are only seven and explicit listing is easier to scan.
//!
//! Skeleton: no assemblies yet. Step 11 populates this list.

use std::sync::Arc;

use brainwires_eval::EvaluationCase;

/// Every assembly the harness knows about. Empty in the skeleton.
pub fn all() -> Vec<Arc<dyn EvaluationCase>> {
    Vec::new()
}
