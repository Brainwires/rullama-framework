//! Language-specific executors

#[cfg(feature = "interpreters-rhai")]
pub mod rhai;

#[cfg(feature = "interpreters-lua")]
pub mod lua;

#[cfg(feature = "interpreters-js")]
pub mod javascript;

use super::types::{ExecutionLimits, ExecutionRequest, ExecutionResult};

/// Trait for language executors
#[allow(dead_code)]
pub trait LanguageExecutor {
    /// Execute code and return the result
    fn execute(&self, request: &ExecutionRequest) -> ExecutionResult;

    /// Get the language name
    fn language_name(&self) -> &'static str;

    /// Get the language version
    fn language_version(&self) -> String;
}

/// Helper to create execution limits from request
pub(crate) fn get_limits(request: &ExecutionRequest) -> ExecutionLimits {
    request.limits.clone().unwrap_or_else(|| ExecutionLimits {
        max_timeout_ms: request.timeout_ms,
        max_memory_mb: request.memory_limit_mb,
        ..ExecutionLimits::default()
    })
}

/// Helper to truncate output if too large
pub(crate) fn truncate_output(output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        output.to_string()
    } else {
        let truncated = &output[..max_bytes];
        format!(
            "{}...\n[Output truncated at {} bytes]",
            truncated, max_bytes
        )
    }
}
