//! Cross-crate test harness for the Brainwires framework.
//!
//! Three tiers:
//! - **Tier A (feature determinism)** — every FEATURES.md heading has ≥1
//!   deterministic case. Manifest at `tests/feature_inventory.toml` lists
//!   Rust function paths registered via [`registry`].
//! - **Tier B (security adversarial)** — per-invariant adversarial cases
//!   registered via `inventory::submit!` next to each attack.
//! - **Tier C (golden-path assemblies)** — manually-listed integration
//!   scenarios in [`assemblies`].
//!
//! Designed to be a one-way data producer: `tier_*_suite()` returns
//! `Vec<Arc<dyn EvaluationCase>>` that can be fed into either
//! [`brainwires_eval::EvaluationSuite`] directly or
//! `brainwires_autonomy::AutonomousFeedbackLoop` (the latter lives in
//! `extras/` and consumes the harness output without the harness importing
//! the autonomy crate).

use std::sync::Arc;

use brainwires_eval::EvaluationCase;

pub mod assemblies;
pub mod cases;
pub mod manifest;
pub mod registry;

// Re-export the most commonly used items so case authors only need one use.
pub use brainwires_eval::{
    AdversarialTestCase, AdversarialTestType, EvaluationStats, EvaluationSuite, SuiteConfig,
    SuiteResult, TrialResult,
};
pub use brainwires_test_fixtures::{
    FailingProvider, RecordedCall, RecordingProvider, ScriptedProvider, ScriptedResponse,
};

/// Path to the feature-inventory manifest, relative to the workspace root.
pub const MANIFEST_PATH: &str = "crates/brainwires-test-harness/tests/feature_inventory.toml";

/// Tier-A suite: every case the feature-inventory manifest lists. Empty
/// until manifest entries are populated and registered (Steps 9-10).
pub fn tier_a_suite() -> Vec<Arc<dyn EvaluationCase>> {
    Vec::new()
}

/// Tier-B suite: every security adversarial case registered via
/// `inventory::submit!`. Empty until cases land (Steps 5-6).
pub fn tier_b_suite() -> Vec<Arc<dyn EvaluationCase>> {
    registry::all_security_cases()
}

/// Tier-C suite: the 7 golden-path assemblies. Empty until assemblies
/// land (Step 11).
pub fn tier_c_suite() -> Vec<Arc<dyn EvaluationCase>> {
    assemblies::all()
}

/// Convenience: every case across all three tiers.
pub fn all_cases() -> Vec<Arc<dyn EvaluationCase>> {
    let mut v = tier_a_suite();
    v.extend(tier_b_suite());
    v.extend(tier_c_suite());
    v
}
