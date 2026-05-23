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

pub mod agent_file_locks;
pub mod call_policy_budget;
pub mod empty_features_compile;
pub mod inference_auto_compact;
pub mod keyed_budget_isolation;
pub mod mcp_server_auth;
pub mod network_api_key_format;
pub mod permission_default_deny;
pub mod sandbox_mount_whitelist;
pub mod schema_violation_retry;
pub mod skills_signature;
pub mod speech_rate_limiter;
pub mod stream_cancel;
pub mod tokenizer_precheck;
pub mod tool_runtime_sanitize;
