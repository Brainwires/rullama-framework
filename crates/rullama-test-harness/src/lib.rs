//! Cross-crate test harness for the rullama framework.
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
//! [`rullama_eval::EvaluationSuite`] directly or
//! `rullama_autonomy::AutonomousFeedbackLoop` (the latter lives in
//! `extras/` and consumes the harness output without the harness importing
//! the autonomy crate).

use std::sync::Arc;

use rullama_eval::EvaluationCase;

pub mod assemblies;
pub mod cases;
pub mod live;
pub mod manifest;
pub mod registry;

// Re-export the most commonly used items so case authors only need one use.
pub use rullama_eval::{
    AdversarialTestCase, AdversarialTestType, EvaluationStats, EvaluationSuite, SuiteConfig,
    SuiteResult, TrialResult,
};
pub use rullama_test_fixtures::{
    FailingProvider, RecordedCall, RecordingProvider, ScriptedProvider, ScriptedResponse,
};

/// Path to the feature-inventory manifest, relative to the workspace root.
pub const MANIFEST_PATH: &str = "crates/rullama-test-harness/tests/feature_inventory.toml";

/// Tier-A suite: every case the feature-inventory manifest lists.
/// Loads `tests/feature_inventory.toml` relative to the workspace root,
/// resolves each `required_cases` Rust path via [`registry::lookup_tier_a`],
/// and silently skips entries whose paths aren't yet registered (those
/// surface separately via `cargo xtask test-harness coverage`).
pub fn tier_a_suite() -> Vec<Arc<dyn EvaluationCase>> {
    // Manifest lives in the harness crate's tests/ directory. From a built
    // binary we look it up via CARGO_MANIFEST_DIR (set by cargo when the
    // binary runs); fall back to the workspace-relative path otherwise.
    let manifest_path = std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .map(|p| p.join("tests/feature_inventory.toml"))
        .unwrap_or_else(|_| std::path::PathBuf::from(MANIFEST_PATH));

    let m = match manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in &m.entries {
        for path in &entry.required_cases {
            if let Some(case) = registry::lookup_tier_a(path) {
                out.push(case);
            }
        }
    }
    out
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

/// Tier-D suite: live-provider integration cases. Each case self-skips
/// when its required `RULLAMA_LIVE_*` env vars are absent, so opting
/// in is a matter of exporting the keys before invoking the harness.
///
/// Excluded from [`all_cases`] so the default `cargo xtask test-harness run`
/// stays offline and free. Reach with `--tier=d`.
pub fn tier_d_suite() -> Vec<Arc<dyn EvaluationCase>> {
    registry::all_live_cases()
}

/// Convenience: every Tier A/B/C case (offline + deterministic). Tier-D
/// is opt-in and not included here.
pub fn all_cases() -> Vec<Arc<dyn EvaluationCase>> {
    let mut v = tier_a_suite();
    v.extend(tier_b_suite());
    v.extend(tier_c_suite());
    v
}
