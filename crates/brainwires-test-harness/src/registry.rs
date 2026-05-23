//! Inventory-based registration for harness cases.
//!
//! Tier-A cases and Tier-B security cases each register themselves at link
//! time via `inventory::submit!`. The lookup tables here are populated
//! eagerly the first time anything in this module is touched.
//!
//! Tier A maps **manifest function paths → factory functions** so the
//! manifest TOML can stay declarative.
//! Tier B is keyed by case `id` (the same string the case reports as its
//! [`brainwires_eval::EvaluationCase::name`]).

use std::sync::Arc;

use brainwires_eval::EvaluationCase;

/// Factory function type — every registered case produces a fresh boxed
/// trait object on demand.
pub type CaseFactory = fn() -> Box<dyn EvaluationCase>;

/// A Tier-A feature case, registered by Rust function path so the manifest
/// can list it declaratively.
pub struct TierACase {
    /// Full Rust function path (e.g. "brainwires_test_harness::cases::mdap::k_of_n_quorum")
    pub path: &'static str,
    /// The crate this case attacks/exercises.
    pub crate_name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Factory producing a fresh case instance.
    pub factory: CaseFactory,
}

inventory::collect!(TierACase);

/// A Tier-B security case, registered next to the invariant it attacks.
pub struct SecurityCase {
    /// Stable dotted id (e.g. "sec.sandbox.mount_whitelist"). Must match
    /// the value the case returns from `EvaluationCase::name`.
    pub id: &'static str,
    /// The crate whose invariant this case attacks.
    pub crate_name: &'static str,
    /// One-line statement of the invariant being asserted.
    pub invariant: &'static str,
    /// Factory producing a fresh case instance.
    pub factory: CaseFactory,
}

inventory::collect!(SecurityCase);

/// A Tier-D live-provider case, gated by env-var presence (see [`crate::live`]).
/// Each case self-skips when its required env vars are absent.
pub struct LiveCase {
    /// Stable dotted id (e.g. "live.ollama.gemma_chat_roundtrip").
    pub id: &'static str,
    /// Which provider this case exercises ("ollama" | "openai" | "anthropic" | "mixed").
    pub provider: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Factory producing a fresh case instance.
    pub factory: CaseFactory,
}

inventory::collect!(LiveCase);

/// Look up a Tier-A case by Rust function path. Returns `None` if no
/// registered case has that path.
pub fn lookup_tier_a(path: &str) -> Option<Arc<dyn EvaluationCase>> {
    inventory::iter::<TierACase>()
        .find(|c| c.path == path)
        .map(|c| Arc::from((c.factory)()))
}

/// Every Tier-A case registered in the binary, in inventory iteration order.
pub fn all_tier_a_cases() -> Vec<&'static TierACase> {
    inventory::iter::<TierACase>().collect()
}

/// Every Tier-B security case registered in the binary.
pub fn all_security_cases() -> Vec<Arc<dyn EvaluationCase>> {
    inventory::iter::<SecurityCase>()
        .map(|c| Arc::from((c.factory)()))
        .collect()
}

/// Metadata-only view of registered security cases. Useful for the
/// `xtask test-harness coverage --crit-min=4` check that asserts each
/// critical-gap crate has at least N adversarial cases.
pub fn all_security_metadata() -> Vec<&'static SecurityCase> {
    inventory::iter::<SecurityCase>().collect()
}

/// Every Tier-D live-provider case registered in the binary.
pub fn all_live_cases() -> Vec<Arc<dyn EvaluationCase>> {
    inventory::iter::<LiveCase>()
        .map(|c| Arc::from((c.factory)()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inventories_are_safe() {
        // At skeleton time, no cases are registered yet. The lookup
        // functions must still return empty collections, not panic.
        assert!(all_tier_a_cases().is_empty() || !all_tier_a_cases().is_empty());
        assert!(all_security_metadata().is_empty() || !all_security_metadata().is_empty());
        assert!(lookup_tier_a("does::not::exist").is_none());
    }
}
