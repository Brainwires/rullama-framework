#![deny(missing_docs)]
//! # Brainwires WASM
//!
//! WASM bindings for the Brainwires Agent Framework.
//!
//! This crate provides a JavaScript-friendly API for the WASM-compatible subset of the
//! Brainwires framework, enabling browser-based AI agent applications. All public functions
//! are exposed via `wasm-bindgen` and can be called directly from JavaScript/TypeScript.
//!
//! ## Features
//!
//! - **Core types** (messages, tools, tasks) — always available
//! - **MDAP types and configuration** — always available
//! - **Code interpreters** — enabled with the `interpreters` feature
//! - **Tool orchestrator** — enabled with the `orchestrator` feature; provides a Rhai-based
//!   script engine that can invoke JavaScript tool callbacks from WASM
//!
//! ## JS Usage
//!
//! ```js
//! import init, { version, validate_message, serialize_history } from 'brainwires-wasm';
//!
//! await init();
//! console.log(version()); // e.g. "0.4.1"
//! ```

use wasm_bindgen::prelude::*;

/// Re-export of the [`brainwires_core`] crate for Rust consumers who depend on this
/// WASM crate and need access to core types (`Message`, `Tool`, `Task`, etc.).
pub use brainwires_core;

/// Re-export of MDAP (Multi-Dimensional Adaptive Planning) types — extracted
/// to its own crate in 0.11. Rust consumers get the same module path.
pub use brainwires_mdap as mdap;

/// Re-export of the interpreters module (requires the `interpreters` feature).
///
/// Provides sandboxed code execution capabilities for languages like JavaScript and Python
/// within the WASM environment.
#[cfg(feature = "interpreters")]
pub use brainwires_tool_builtins::interpreters;

/// WASM orchestrator module providing JavaScript-compatible bindings for the tool orchestrator.
///
/// This module is only available when the `orchestrator` feature is enabled and the target
/// is `wasm32` — its closures capture `js_sys::Function` and `Rc<RefCell<…>>`, neither of
/// which is `Send`/`Sync`, so it can't link against `rhai/sync` on native (and there is
/// no JS function to call there anyway). It exposes
/// [`WasmOrchestrator`](wasm_orchestrator::WasmOrchestrator) and
/// [`ExecutionLimits`](wasm_orchestrator::ExecutionLimits) for running Rhai scripts
/// that can call registered JavaScript tool functions.
#[cfg(all(feature = "orchestrator", target_arch = "wasm32"))]
pub mod wasm_orchestrator;

/// Convenience re-exports of the orchestrator types at crate root level.
///
/// - [`WasmExecutionLimits`] — Alias for [`wasm_orchestrator::ExecutionLimits`]
/// - [`WasmOrchestrator`] — The main orchestrator entry point
#[cfg(all(feature = "orchestrator", target_arch = "wasm32"))]
pub use wasm_orchestrator::{ExecutionLimits as WasmExecutionLimits, WasmOrchestrator};

// ── WASM Bindings ────────────────────────────────────────────────────────

/// Returns the crate version string (e.g. `"0.4.1"`).
///
/// This is the version of the `brainwires-wasm` package as declared in `Cargo.toml`.
///
/// # JS Example
///
/// ```js
/// const v = version();
/// console.log(`Brainwires WASM v${v}`);
/// ```
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Validates and normalizes a JSON-encoded message.
///
/// Parses the input JSON string into a [`brainwires_core::Message`] struct and
/// re-serializes it back to JSON. This ensures the message conforms to the expected
/// schema and strips any unknown fields.
///
/// # Parameters
///
/// - `json` — A JSON string representing a single message object.
///
/// # Returns
///
/// The normalized JSON string on success, or an error string describing the
/// validation failure.
///
/// # JS Example
///
/// ```js
/// try {
///     const normalized = validate_message('{"role":"user","content":"Hello"}');
///     console.log(normalized);
/// } catch (e) {
///     console.error("Invalid message:", e);
/// }
/// ```
#[wasm_bindgen]
pub fn validate_message(json: &str) -> Result<String, String> {
    let msg: brainwires_core::Message =
        serde_json::from_str(json).map_err(|e| format!("Invalid message JSON: {e}"))?;
    serde_json::to_string(&msg).map_err(|e| format!("Serialization error: {e}"))
}

/// Validates and normalizes a JSON-encoded tool definition.
///
/// Parses the input JSON string into a [`brainwires_core::Tool`] struct and
/// re-serializes it back to JSON. Use this to verify that a tool definition
/// is well-formed before registering it.
///
/// # Parameters
///
/// - `json` — A JSON string representing a tool definition object.
///
/// # Returns
///
/// The normalized JSON string on success, or an error string describing the
/// validation failure.
///
/// # JS Example
///
/// ```js
/// const toolJson = JSON.stringify({
///     name: "calculator",
///     description: "Performs arithmetic",
///     input_schema: { type: "object", properties: { expr: { type: "string" } } }
/// });
/// const normalized = validate_tool(toolJson);
/// ```
#[wasm_bindgen]
pub fn validate_tool(json: &str) -> Result<String, String> {
    let tool: brainwires_core::Tool =
        serde_json::from_str(json).map_err(|e| format!("Invalid tool JSON: {e}"))?;
    serde_json::to_string(&tool).map_err(|e| format!("Serialization error: {e}"))
}

/// Serializes a conversation history to the stateless protocol format.
///
/// Takes a JSON array of [`brainwires_core::Message`] objects and converts them
/// into the stateless history format used by AI provider APIs. This is useful
/// when you need to send a full conversation context in a single API request.
///
/// # Parameters
///
/// - `messages_json` — A JSON string containing an array of message objects.
///
/// # Returns
///
/// A JSON string in the stateless history format on success, or an error string
/// if the input is malformed.
///
/// # JS Example
///
/// ```js
/// const messages = JSON.stringify([
///     { role: "user", content: "What is 2+2?" },
///     { role: "assistant", content: "4" }
/// ]);
/// const history = serialize_history(messages);
/// // Use `history` in an API request body
/// ```
#[wasm_bindgen]
pub fn serialize_history(messages_json: &str) -> Result<String, String> {
    let messages: Vec<brainwires_core::Message> =
        serde_json::from_str(messages_json).map_err(|e| format!("Invalid messages JSON: {e}"))?;
    let history = brainwires_core::serialize_messages_to_stateless_history(&messages);
    serde_json::to_string(&history).map_err(|e| format!("Serialization error: {e}"))
}
