//! Tier-A and Tier-B case modules.
//!
//! Each FEATURES.md section eventually gets its own module here (e.g.
//! `cases::mdap`, `cases::rag`, `cases::memory`). The `security`
//! sub-module collects Tier-B adversarial cases keyed by the invariant
//! they attack.
//!
//! Skeleton: only the `security` directory exists. Per-section modules
//! are added incrementally as features get covered (Steps 5-10).

pub mod agent_builder;
pub mod anthropic_cache_control;
pub mod call_policy_safety;
pub mod core_types;
pub mod evaluation_framework;
pub mod inference_tool_error_hint;
pub mod inference_turn_report;
pub mod live;
pub mod mcp_and_tools;
pub mod mdap_composition;
pub mod permission_capabilities;
pub mod rag_cited_answer;
pub mod retry_after;
pub mod security;
pub mod telemetry_request_id;
