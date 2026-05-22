//! Tier-B security adversarial cases.
//!
//! One file per invariant (or per group of closely-related invariants on
//! the same module). Each file registers its cases via
//! `inventory::submit! { crate::registry::SecurityCase { ... } }`.
//!
//! Note: `inventory` only picks up symbols from compiled translation units.
//! Each new case file MUST be declared here via `pub mod foo;` even though
//! nothing else in the harness imports it — otherwise the linker drops the
//! whole module and `inventory::iter::<SecurityCase>()` returns nothing.

pub mod call_policy_budget;
pub mod mcp_server_auth;
pub mod network_api_key_format;
pub mod permission_default_deny;
pub mod sandbox_mount_whitelist;
pub mod skills_signature;
pub mod speech_rate_limiter;
pub mod tool_runtime_sanitize;
