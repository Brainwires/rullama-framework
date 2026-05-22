//! Tier-A and Tier-B case modules.
//!
//! Each FEATURES.md section eventually gets its own module here (e.g.
//! `cases::mdap`, `cases::rag`, `cases::memory`). The `security`
//! sub-module collects Tier-B adversarial cases keyed by the invariant
//! they attack.
//!
//! Skeleton: only the `security` directory exists. Per-section modules
//! are added incrementally as features get covered (Steps 5-10).

pub mod core_types;
pub mod evaluation_framework;
pub mod security;
