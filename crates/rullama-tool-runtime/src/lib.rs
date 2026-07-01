#![deny(missing_docs)]
//! `rullama-tool-runtime` — the execution-runtime half of the rullama
//! tool surface. Companion crate to `rullama-tool-builtins` (the concrete
//! tools), re-exported together by the `rullama` facade crate.
//!
//! ## What lives here
//!
//! - [`executor::ToolExecutor`] — the trait every tool dispatcher implements.
//! - [`registry::ToolRegistry`] + [`registry::ToolCategory`] — composable tool
//!   registry and category metadata.
//! - [`error`] — tool-error taxonomy + retry classification.
//! - [`sanitization`] — content-source tagging, injection detection,
//!   sensitive-data redaction.
//! - [`tool_search::ToolSearchTool`] — meta-tool for keyword / regex / (with
//!   `rag` feature) semantic tool discovery.
//! - [`smart_router`] — query-driven category filtering.
//! - [`transaction::TransactionManager`] — idempotency + staging-area
//!   bookkeeping for file-mutating tools (native only).
//! - [`validation::ValidationTool`] — duplicate/syntax/build checks (native
//!   only).
//!
//! ## Feature-gated runtime modules
//!
//! - `orchestrator` (or `orchestrator-wasm`) — [`orchestrator::OrchestratorTool`]
//!   (Rhai script executor).
//! - `oauth` — OAuth 2.0 client, PKCE, pluggable token store.
//! - `openapi` — OpenAPI 3 spec → tool descriptor conversion.
//! - `sandbox` — `sandbox_executor::SandboxedToolExecutor` (wraps any
//!   `ToolExecutor` to route bash/code-exec through `rullama-sandbox`).
//! - `sessions` — `sessions::SessionsTool` (`sessions_list`, `sessions_history`,
//!   `sessions_send`, `sessions_spawn`) over a `rullama-session::SessionBroker`.
//! - `rag` — [`tool_embedding::ToolEmbeddingIndex`] backing `ToolSearchTool`'s
//!   semantic mode.
//!
//! Concrete builtin tools (bash, file_ops, git, web, search, code_exec,
//! interpreters, browser, email, calendar, system, semantic_search) live in
//! `rullama-tool-builtins`. The umbrella `rullama-tools` façade re-exports
//! both crates.

// Re-export the core trait surface so consumers can pull everything off this crate.
pub use rullama_core::{
    CommitResult, IdempotencyRecord, IdempotencyRegistry, StagedWrite, StagingBackend, Tool,
    ToolContext, ToolInputSchema, ToolResult,
};

/// Tool-error taxonomy + retry classification.
pub mod error;
/// `ToolExecutor` trait + pre-hook surface.
pub mod executor;
/// Composable tool registry + category metadata.
pub mod registry;
/// Content-source tagging, injection detection, sensitive-data redaction.
pub mod sanitization;
/// Query-driven category filter (`analyze_query`, `get_smart_tools`).
pub mod smart_router;
/// Meta-tool for keyword / regex / (with `rag`) semantic tool discovery.
pub mod tool_search;

/// Idempotency + staging-area bookkeeping for file-mutating tools.
#[cfg(feature = "native")]
pub mod transaction;
/// Code-quality checks (duplicates, syntax, build).
#[cfg(feature = "native")]
pub mod validation;

/// Rhai-script orchestration tool.
#[cfg(any(feature = "orchestrator", feature = "orchestrator-wasm"))]
pub mod orchestrator;

/// OAuth 2.0 client, PKCE, pluggable token store.
#[cfg(feature = "oauth")]
pub mod oauth;

/// OpenAPI 3 spec → tool descriptor conversion.
#[cfg(feature = "openapi")]
pub mod openapi;

/// Container-sandbox executor wrapper.
#[cfg(feature = "sandbox")]
pub mod sandbox_executor;

/// `SessionsTool` + companion types over `rullama-session::SessionBroker`.
#[cfg(feature = "sessions")]
pub mod sessions;

/// RAG-backed tool-embedding index (powers `ToolSearchTool`'s semantic mode).
#[cfg(feature = "rag")]
pub mod tool_embedding;

// ── Public re-exports ────────────────────────────────────────────────────────

pub use error::{ResourceType, RetryStrategy, ToolErrorCategory, ToolOutcome, classify_error};
pub use executor::{PreHookDecision, ToolExecutor, ToolPreHook};
pub use registry::{ToolCategory, ToolRegistry};
pub use sanitization::{
    contains_pii, contains_sensitive_data, filter_tool_output, is_injection_attempt, redact_pii,
    redact_sensitive_data, sanitize_external_content, wrap_with_content_source,
    wrap_with_content_source_with_pii,
};
pub use smart_router::{
    analyze_messages, analyze_query, get_context_for_analysis, get_smart_tools,
    get_smart_tools_with_mcp, get_tools_for_categories,
};
pub use tool_search::ToolSearchTool;

#[cfg(feature = "native")]
pub use transaction::TransactionManager;
#[cfg(feature = "native")]
pub use validation::{ValidationTool, get_validation_tools};

#[cfg(any(feature = "orchestrator", feature = "orchestrator-wasm"))]
pub use orchestrator::OrchestratorTool;

#[cfg(feature = "openapi")]
pub use openapi::{
    HttpMethod, OpenApiAuth, OpenApiEndpoint, OpenApiParam, OpenApiTool, execute_openapi_tool,
    openapi_to_tools,
};

#[cfg(feature = "sandbox")]
pub use sandbox_executor::SandboxedToolExecutor;

// `SessionBroker` / `SessionId` / `SessionMessage` / `SessionSummary` /
// `SpawnRequest` / `SpawnedSession` live in `rullama-session::broker`. The
// `SessionsTool` here only consumes them — depend on `rullama-session`
// directly.
#[cfg(feature = "sessions")]
pub use sessions::SessionsTool;

#[cfg(feature = "rag")]
pub use tool_embedding::ToolEmbeddingIndex;
