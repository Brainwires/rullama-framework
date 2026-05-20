#![deny(missing_docs)]
//! `brainwires-tool-builtins` — the concrete-tools half of the Brainwires
//! tool surface. Companion crate to `brainwires-tool-runtime` (the executor /
//! registry / framework), unified by the `brainwires-tools` façade.
//!
//! ## Always Available (native feature)
//! - **bash** — Shell command execution with proactive output management
//! - **file_ops** — File read/write/edit/patch/list/search/delete/create_directory
//! - **git** — Git operations (status, diff, log, stage, commit, push, pull, …)
//! - **web** — URL fetching
//! - **search** — Regex-based code search (respects .gitignore)
//! - **default_executor::BuiltinToolExecutor** — `ToolExecutor` impl that
//!   hardcodes dispatch to the above plus the feature-gated builtins below.
//!
//! ## Feature-Gated
//! - **code_exec / interpreters** (`interpreters` feature) — sandboxed
//!   multi-language code execution.
//! - **semantic_search** (`rag` feature) — RAG-powered codebase search.
//! - **email** (`email` feature) — IMAP/SMTP/Gmail-push integration.
//! - **calendar** (`calendar` feature) — Google Calendar / CalDAV.
//! - **browser** (`browser` feature) — headless-browser tooling via the MCP
//!   Thalora subprocess.
//! - **system** (`system` feature) — filesystem-event watching, service
//!   management (absorbed from `brainwires-system`).
//!
//! The runtime trait surface (`ToolExecutor`, `ToolRegistry`, error types,
//! `transaction`, `validation`, optional `orchestrator` / `oauth` /
//! `openapi` / `sandbox_executor` / `sessions`) is **not** exported from this
//! crate — depend on `brainwires-tool-runtime` directly, or use the
//! `brainwires-tools` façade which re-exports both.

mod default_executor;

#[cfg(feature = "native")]
mod bash;
#[cfg(feature = "native")]
mod file_ops;
#[cfg(feature = "native")]
mod git;
#[cfg(feature = "native")]
mod search;
#[cfg(feature = "native")]
mod web;

#[cfg(feature = "interpreters")]
mod code_exec;

#[cfg(feature = "rag")]
mod semantic_search;

#[cfg(feature = "email")]
mod email;

#[cfg(feature = "calendar")]
pub mod calendar;

#[cfg(feature = "browser")]
mod browser;

/// OS-level primitives — filesystem event watching and service management
/// (absorbed from brainwires-system).
#[cfg(feature = "system")]
pub mod system;

/// Sandboxed multi-language code interpreters (absorbed from brainwires-code-interpreters).
#[cfg(feature = "interpreters")]
pub mod interpreters;

// ── Public re-exports ──────────────────────────────────────────────────────

pub use default_executor::BuiltinToolExecutor;

#[cfg(feature = "native")]
pub use bash::BashTool;
#[cfg(feature = "native")]
pub use file_ops::FileOpsTool;
#[cfg(feature = "native")]
pub use git::GitTool;
#[cfg(feature = "native")]
pub use search::SearchTool;
#[cfg(feature = "native")]
pub use web::WebTool;

#[cfg(feature = "interpreters")]
pub use code_exec::CodeExecTool;

#[cfg(feature = "rag")]
pub use semantic_search::SemanticSearchTool;

#[cfg(feature = "email")]
pub use email::{EmailConfig, EmailProvider, EmailSource, EmailTool, gmail_push};

#[cfg(feature = "calendar")]
pub use calendar::CalendarTool;

#[cfg(feature = "browser")]
pub use browser::BrowserTool;

/// Build a [`brainwires_tool_runtime::ToolRegistry`] pre-populated with
/// every concrete builtin gated on by the active feature set.
///
/// Only registers tools owned by **this crate** (plus the runtime's
/// `tool_search` meta-tool and `validation` tools for backward compat).
/// Runtime-only tools like `OrchestratorTool` are not in scope here —
/// the `brainwires-tools` façade composes those on top via its own
/// `registry_with_builtins()` that has visibility into both crates.
///
/// Replaces the historical `ToolRegistry::with_builtins()` constructor
/// that lived in the runtime crate before the runtime/builtins split —
/// the runtime can't reference concrete builtins, so the convenience
/// constructor lives here where it can.
pub fn registry_with_builtins() -> brainwires_tool_runtime::ToolRegistry {
    let mut registry = brainwires_tool_runtime::ToolRegistry::with_runtime_meta_tools();

    #[cfg(feature = "native")]
    {
        registry.register_tools(BashTool::get_tools());
        registry.register_tools(FileOpsTool::get_tools());
        registry.register_tools(GitTool::get_tools());
        registry.register_tools(WebTool::get_tools());
        registry.register_tools(SearchTool::get_tools());
        registry.register_tools(brainwires_tool_runtime::get_validation_tools());
    }

    #[cfg(feature = "interpreters")]
    registry.register_tools(CodeExecTool::get_tools());

    #[cfg(feature = "rag")]
    registry.register_tools(SemanticSearchTool::get_tools());

    registry
}
