//! Execution limits and sandboxing for safe script execution.
//!
//! This module provides [`ExecutionLimits`], a configurable struct that controls
//! resource usage during Rhai script execution. These limits protect against:
//!
//! - **Infinite loops** - via `max_operations`
//! - **Runaway tool calls** - via `max_tool_calls`
//! - **Long-running scripts** - via `timeout_ms` (real-time enforcement)
//! - **Memory exhaustion** - via `max_string_size`, `max_array_size`, `max_map_size`
//!
//! # Preset Profiles
//!
//! Three preset profiles are provided for common use cases:
//!
//! | Profile | Max Ops | Max Tools | Timeout | Use Case |
//! |---------|---------|-----------|---------|----------|
//! | `quick()` | 10,000 | 10 | 5s | Simple scripts |
//! | `default()` | 100,000 | 50 | 30s | General use |
//! | `extended()` | 500,000 | 100 | 120s | Complex orchestration |
//!
//! # Example
//!
//! ```ignore
//! use brainwires_tool_runtime::orchestrator::ExecutionLimits;
//!
//! // Use a preset
//! let limits = ExecutionLimits::quick();
//!
//! // Or customize with the builder pattern
//! let limits = ExecutionLimits::default()
//!     .with_max_operations(50_000)
//!     .with_timeout_ms(10_000);
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Default Limit Constants
// =============================================================================

/// Default maximum Rhai operations before termination (prevents infinite loops)
pub const DEFAULT_MAX_OPERATIONS: u64 = 100_000;

/// Default maximum tool calls per script execution
pub const DEFAULT_MAX_TOOL_CALLS: usize = 50;

/// Default timeout in milliseconds (30 seconds)
pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default maximum string size in bytes (10 MB)
pub const DEFAULT_MAX_STRING_SIZE: usize = 10_000_000;

/// Default maximum array size (number of elements)
pub const DEFAULT_MAX_ARRAY_SIZE: usize = 10_000;

/// Default maximum map size (number of key-value pairs)
pub const DEFAULT_MAX_MAP_SIZE: usize = 1_000;

// =============================================================================
// Quick Profile Constants
// =============================================================================

/// Quick profile: maximum operations (10,000)
pub const QUICK_MAX_OPERATIONS: u64 = 10_000;

/// Quick profile: maximum tool calls (10)
pub const QUICK_MAX_TOOL_CALLS: usize = 10;

/// Quick profile: timeout in milliseconds (5 seconds)
pub const QUICK_TIMEOUT_MS: u64 = 5_000;

// =============================================================================
// Extended Profile Constants
// =============================================================================

/// Extended profile: maximum operations (500,000)
pub const EXTENDED_MAX_OPERATIONS: u64 = 500_000;

/// Extended profile: maximum tool calls (100)
pub const EXTENDED_MAX_TOOL_CALLS: usize = 100;

/// Extended profile: timeout in milliseconds (2 minutes)
pub const EXTENDED_TIMEOUT_MS: u64 = 120_000;

// =============================================================================
// ExecutionLimits
// =============================================================================

/// Limits for safe script execution.
///
/// Configures resource bounds to prevent runaway scripts from consuming
/// excessive CPU, memory, or time. All limits are enforced by the Rhai
/// engine during script execution.
///
/// # Security Note
///
/// The `timeout_ms` limit is enforced in real-time via Rhai's `on_progress`
/// callback, which is checked after every operation. This provides true
/// wall-clock timeout protection, not just operation counting.
///
/// # Serialization
///
/// This struct derives `Serialize` and `Deserialize` for easy configuration
/// storage and transmission (e.g., in JSON config files or API requests).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExecutionLimits {
    /// Maximum number of operations (prevents infinite loops)
    pub max_operations: u64,
    /// Maximum number of tool calls allowed
    pub max_tool_calls: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum string size in bytes
    pub max_string_size: usize,
    /// Maximum array size
    pub max_array_size: usize,
    /// Maximum map size
    pub max_map_size: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_operations: DEFAULT_MAX_OPERATIONS,
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_string_size: DEFAULT_MAX_STRING_SIZE,
            max_array_size: DEFAULT_MAX_ARRAY_SIZE,
            max_map_size: DEFAULT_MAX_MAP_SIZE,
        }
    }
}

impl ExecutionLimits {
    /// Create new limits with all defaults.
    ///
    /// Equivalent to [`ExecutionLimits::default()`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create quick execution limits for simple scripts.
    ///
    /// Suitable for short, trusted scripts that need fast execution:
    /// - 10,000 max operations
    /// - 10 max tool calls
    /// - 5 second timeout
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::quick();
    /// let result = orchestrator.execute("1 + 1", limits)?;
    /// ```
    #[must_use]
    pub fn quick() -> Self {
        Self {
            max_operations: QUICK_MAX_OPERATIONS,
            max_tool_calls: QUICK_MAX_TOOL_CALLS,
            timeout_ms: QUICK_TIMEOUT_MS,
            ..Default::default()
        }
    }

    /// Create extended limits for complex orchestration.
    ///
    /// Suitable for complex multi-tool workflows:
    /// - 500,000 max operations
    /// - 100 max tool calls
    /// - 2 minute timeout
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::extended();
    /// let result = orchestrator.execute(complex_script, limits)?;
    /// ```
    #[must_use]
    pub fn extended() -> Self {
        Self {
            max_operations: EXTENDED_MAX_OPERATIONS,
            max_tool_calls: EXTENDED_MAX_TOOL_CALLS,
            timeout_ms: EXTENDED_TIMEOUT_MS,
            ..Default::default()
        }
    }

    /// Set maximum operations (builder pattern).
    ///
    /// Controls how many Rhai operations (expressions, statements) can execute
    /// before the script is terminated. Prevents infinite loops.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_max_operations(50_000);
    /// ```
    #[must_use]
    pub const fn with_max_operations(mut self, max: u64) -> Self {
        self.max_operations = max;
        self
    }

    /// Set maximum tool calls (builder pattern).
    ///
    /// Limits how many times tools can be invoked from a single script.
    /// Prevents runaway tool usage.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_max_tool_calls(25);
    /// ```
    #[must_use]
    pub const fn with_max_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = max;
        self
    }

    /// Set timeout in milliseconds (builder pattern).
    ///
    /// Enforced in real-time via Rhai's `on_progress` callback, which is
    /// checked after every operation. This provides true wall-clock timeout.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_timeout_ms(10_000); // 10 seconds
    /// ```
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Set maximum string size in bytes (builder pattern).
    ///
    /// Prevents scripts from creating extremely large strings that could
    /// exhaust memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_max_string_size(5_000_000); // 5MB
    /// ```
    #[must_use]
    pub const fn with_max_string_size(mut self, size: usize) -> Self {
        self.max_string_size = size;
        self
    }

    /// Set maximum array size (builder pattern).
    ///
    /// Prevents scripts from creating extremely large arrays that could
    /// exhaust memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_max_array_size(5_000);
    /// ```
    #[must_use]
    pub const fn with_max_array_size(mut self, size: usize) -> Self {
        self.max_array_size = size;
        self
    }

    /// Set maximum map size (builder pattern).
    ///
    /// Prevents scripts from creating extremely large maps/objects that could
    /// exhaust memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = ExecutionLimits::default()
    ///     .with_max_map_size(500);
    /// ```
    #[must_use]
    pub const fn with_max_map_size(mut self, size: usize) -> Self {
        self.max_map_size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_limits() {
        let limits = ExecutionLimits::default();
        assert_eq!(limits.max_operations, DEFAULT_MAX_OPERATIONS);
        assert_eq!(limits.max_tool_calls, DEFAULT_MAX_TOOL_CALLS);
        assert_eq!(limits.timeout_ms, DEFAULT_TIMEOUT_MS);
    }

    #[test]
    fn test_quick_limits() {
        let limits = ExecutionLimits::quick();
        assert_eq!(limits.max_operations, QUICK_MAX_OPERATIONS);
        assert_eq!(limits.max_tool_calls, QUICK_MAX_TOOL_CALLS);
    }

    #[test]
    fn test_extended_limits() {
        let limits = ExecutionLimits::extended();
        assert_eq!(limits.max_operations, EXTENDED_MAX_OPERATIONS);
        assert_eq!(limits.max_tool_calls, EXTENDED_MAX_TOOL_CALLS);
    }

    #[test]
    fn test_builder_pattern() {
        let limits = ExecutionLimits::default()
            .with_max_operations(50_000)
            .with_max_tool_calls(25)
            .with_timeout_ms(10_000);

        assert_eq!(limits.max_operations, 50_000);
        assert_eq!(limits.max_tool_calls, 25);
        assert_eq!(limits.timeout_ms, 10_000);
    }

    #[test]
    fn test_new_equals_default() {
        let new_limits = ExecutionLimits::new();
        let default_limits = ExecutionLimits::default();

        assert_eq!(new_limits.max_operations, default_limits.max_operations);
        assert_eq!(new_limits.max_tool_calls, default_limits.max_tool_calls);
        assert_eq!(new_limits.timeout_ms, default_limits.timeout_ms);
        assert_eq!(new_limits.max_string_size, default_limits.max_string_size);
        assert_eq!(new_limits.max_array_size, default_limits.max_array_size);
        assert_eq!(new_limits.max_map_size, default_limits.max_map_size);
    }

    #[test]
    fn test_with_max_string_size() {
        let limits = ExecutionLimits::default().with_max_string_size(5_000_000);
        assert_eq!(limits.max_string_size, 5_000_000);
        // Other values should remain at defaults
        assert_eq!(limits.max_operations, DEFAULT_MAX_OPERATIONS);
    }

    #[test]
    fn test_with_max_array_size() {
        let limits = ExecutionLimits::default().with_max_array_size(5_000);
        assert_eq!(limits.max_array_size, 5_000);
        // Other values should remain at defaults
        assert_eq!(limits.max_operations, DEFAULT_MAX_OPERATIONS);
    }

    #[test]
    fn test_with_max_map_size() {
        let limits = ExecutionLimits::default().with_max_map_size(500);
        assert_eq!(limits.max_map_size, 500);
        // Other values should remain at defaults
        assert_eq!(limits.max_operations, DEFAULT_MAX_OPERATIONS);
    }

    #[test]
    fn test_full_builder_chain() {
        let limits = ExecutionLimits::new()
            .with_max_operations(200_000)
            .with_max_tool_calls(75)
            .with_timeout_ms(60_000)
            .with_max_string_size(20_000_000)
            .with_max_array_size(20_000)
            .with_max_map_size(2_000);

        assert_eq!(limits.max_operations, 200_000);
        assert_eq!(limits.max_tool_calls, 75);
        assert_eq!(limits.timeout_ms, 60_000);
        assert_eq!(limits.max_string_size, 20_000_000);
        assert_eq!(limits.max_array_size, 20_000);
        assert_eq!(limits.max_map_size, 2_000);
    }
}
