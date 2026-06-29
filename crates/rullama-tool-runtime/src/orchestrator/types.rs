//! Core types for tool orchestration.
//!
//! This module defines the fundamental data structures used throughout the
//! tool orchestrator:
//!
//! - [`OrchestratorResult`] - The outcome of script execution
//! - [`ToolCall`] - A record of each tool invocation
//! - [`OrchestratorError`] - Error types for various failure modes
//!
//! # Example
//!
//! ```ignore
//! use rullama_tool_runtime::orchestrator::{OrchestratorResult, ToolCall};
//!
//! // Results are typically returned from ToolOrchestrator::execute()
//! let result = OrchestratorResult::success(
//!     "Hello, world!".to_string(),
//!     vec![],
//!     50,
//! );
//!
//! assert!(result.success);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result from executing an orchestration script.
///
/// Contains the execution outcome including the script's output,
/// a log of all tool calls made, timing information, and any error details.
///
/// # Fields
///
/// - `success` - Whether the script completed without errors
/// - `output` - The return value of the script (final expression)
/// - `tool_calls` - Complete log of every tool invocation
/// - `execution_time_ms` - Total wall-clock time for execution
/// - `error` - Error message if execution failed
///
/// # Example
///
/// ```ignore
/// let result = orchestrator.execute(script, limits)?;
///
/// if result.success {
///     println!("Output: {}", result.output);
///     println!("Made {} tool calls in {}ms",
///         result.tool_calls.len(),
///         result.execution_time_ms
///     );
/// } else {
///     eprintln!("Error: {}", result.error.unwrap_or_default());
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorResult {
    /// Whether execution completed successfully
    pub success: bool,
    /// Output from the script (final expression value)
    pub output: String,
    /// All tool calls made during execution
    pub tool_calls: Vec<ToolCall>,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if execution failed
    pub error: Option<String>,
}

impl OrchestratorResult {
    /// Create a successful result.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // String/Vec cannot be used in const fn on stable
    pub fn success(output: String, tool_calls: Vec<ToolCall>, execution_time_ms: u64) -> Self {
        Self {
            success: true,
            output,
            tool_calls,
            execution_time_ms,
            error: None,
        }
    }

    /// Create a failed result.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // String/Vec cannot be used in const fn on stable
    pub fn error(error: String, tool_calls: Vec<ToolCall>, execution_time_ms: u64) -> Self {
        Self {
            success: false,
            output: String::new(),
            tool_calls,
            execution_time_ms,
            error: Some(error),
        }
    }
}

/// Record of a single tool call during script execution.
///
/// Each time a registered tool is invoked from a Rhai script, a `ToolCall`
/// record is created capturing the invocation details. This provides an
/// audit trail and enables debugging of tool orchestration workflows.
///
/// # Fields
///
/// - `tool_name` - The registered name of the tool
/// - `input` - Arguments passed to the tool (serialized as JSON)
/// - `output` - The tool's return value (or error message)
/// - `success` - Whether the tool executed without error
/// - `duration_ms` - How long the tool took to execute
///
/// # Example
///
/// ```ignore
/// // After execution, inspect tool calls
/// for call in &result.tool_calls {
///     println!("Tool: {} took {}ms", call.tool_name, call.duration_ms);
///     if !call.success {
///         println!("  Failed: {}", call.output);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Name of the tool that was called
    pub tool_name: String,
    /// Input passed to the tool (as JSON)
    pub input: serde_json::Value,
    /// Output returned by the tool
    pub output: String,
    /// Whether the call succeeded
    pub success: bool,
    /// Execution time for this call in milliseconds
    pub duration_ms: u64,
}

impl ToolCall {
    /// Create a new tool call record.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // serde_json::Value cannot be used in const fn
    pub fn new(
        tool_name: String,
        input: serde_json::Value,
        output: String,
        success: bool,
        duration_ms: u64,
    ) -> Self {
        Self {
            tool_name,
            input,
            output,
            success,
            duration_ms,
        }
    }
}

/// Errors that can occur during orchestration.
///
/// These error types cover the various failure modes of script execution:
///
/// - **Compilation errors** - Invalid Rhai syntax
/// - **Execution errors** - Runtime errors in the script
/// - **Limit violations** - Operations, tool calls, or time exceeded
/// - **Tool errors** - Problems with tool registration or execution
///
/// # Error Handling
///
/// All errors implement `std::error::Error` and provide human-readable messages.
///
/// ```ignore
/// match orchestrator.execute(script, limits) {
///     Ok(result) => println!("Success: {}", result.output),
///     Err(OrchestratorError::Timeout(ms)) => {
///         eprintln!("Script timed out after {}ms", ms);
///     }
///     Err(OrchestratorError::MaxOperationsExceeded(ops)) => {
///         eprintln!("Script exceeded {} operations (infinite loop?)", ops);
///     }
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
#[derive(Debug, Error)]
pub enum OrchestratorError {
    /// Script failed to compile due to syntax errors.
    #[error("Script compilation failed: {0}")]
    CompilationError(String),

    /// Script execution failed at runtime.
    #[error("Script execution failed: {0}")]
    ExecutionError(String),

    /// Script exceeded the maximum allowed operations.
    ///
    /// This typically indicates an infinite or very long loop.
    /// The contained value is the limit that was exceeded.
    #[error("Script exceeded maximum operations ({0})")]
    MaxOperationsExceeded(u64),

    /// Script made too many tool calls.
    ///
    /// The contained value is the limit that was exceeded.
    #[error("Script exceeded maximum tool calls ({0})")]
    MaxToolCallsExceeded(usize),

    /// Script execution exceeded the time limit.
    ///
    /// Enforced in real-time via Rhai's `on_progress` callback.
    /// The contained value is the timeout in milliseconds.
    #[error("Script execution timed out after {0}ms")]
    Timeout(u64),

    /// Referenced tool was not registered with the orchestrator.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// A registered tool returned an error during execution.
    #[error("Tool execution failed: {0}")]
    ToolError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_result_success() {
        let result = OrchestratorResult::success("output".to_string(), vec![], 100);
        assert!(result.success);
        assert_eq!(result.output, "output");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_orchestrator_result_error() {
        let result = OrchestratorResult::error("failed".to_string(), vec![], 50);
        assert!(!result.success);
        assert_eq!(result.error, Some("failed".to_string()));
    }

    #[test]
    fn test_tool_call_new() {
        let call = ToolCall::new(
            "test_tool".to_string(),
            serde_json::json!({"arg": "value"}),
            "result".to_string(),
            true,
            10,
        );
        assert_eq!(call.tool_name, "test_tool");
        assert!(call.success);
    }
}
