//! Common types for code execution

use serde::{Deserialize, Serialize};

/// Maximum string length for the relaxed execution limits profile (100MB).
const RELAXED_MAX_STRING_LENGTH: usize = 104_857_600;

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Rhai - Native Rust scripting language (fastest, lightweight)
    Rhai,
    /// Lua 5.4 - Small, fast scripting language
    Lua,
    /// JavaScript - ECMAScript via Boa engine
    JavaScript,
}

impl Language {
    /// Get the language name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rhai => "rhai",
            Language::Lua => "lua",
            Language::JavaScript => "javascript",
        }
    }

    /// Parse a language from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rhai" => Some(Language::Rhai),
            "lua" => Some(Language::Lua),
            "javascript" | "js" => Some(Language::JavaScript),
            _ => None,
        }
    }

    /// Get the typical file extension for this language
    pub fn extension(&self) -> &'static str {
        match self {
            Language::Rhai => "rhai",
            Language::Lua => "lua",
            Language::JavaScript => "js",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Request to execute code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    /// Programming language to use
    pub language: Language,

    /// Source code to execute
    pub code: String,

    /// Standard input to provide (optional)
    #[serde(default)]
    pub stdin: Option<String>,

    /// Execution timeout in milliseconds (default: 30000)
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Memory limit in MB (default: 256)
    #[serde(default = "default_memory_mb")]
    pub memory_limit_mb: u32,

    /// Context variables to inject as globals (optional)
    #[serde(default)]
    pub context: Option<serde_json::Value>,

    /// Execution limits profile (optional, overrides individual limits)
    #[serde(default)]
    pub limits: Option<ExecutionLimits>,
}

fn default_timeout_ms() -> u64 {
    30_000
}

fn default_memory_mb() -> u32 {
    256
}

impl Default for ExecutionRequest {
    fn default() -> Self {
        Self {
            language: Language::Rhai,
            code: String::new(),
            stdin: None,
            timeout_ms: default_timeout_ms(),
            memory_limit_mb: default_memory_mb(),
            context: None,
            limits: None,
        }
    }
}

/// Result of code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution completed successfully
    pub success: bool,

    /// Standard output from the program
    pub stdout: String,

    /// Standard error from the program
    pub stderr: String,

    /// Return value of the code (if any), serialized as JSON
    #[serde(default)]
    pub result: Option<serde_json::Value>,

    /// Error message if execution failed
    #[serde(default)]
    pub error: Option<String>,

    /// Execution time in milliseconds
    pub timing_ms: u64,

    /// Memory used in bytes (if available)
    #[serde(default)]
    pub memory_used_bytes: Option<u64>,

    /// Number of operations executed (if tracked)
    #[serde(default)]
    pub operations_count: Option<u64>,
}

impl ExecutionResult {
    /// Create a successful result
    pub fn success(stdout: String, result: Option<serde_json::Value>, timing_ms: u64) -> Self {
        Self {
            success: true,
            stdout,
            stderr: String::new(),
            result,
            error: None,
            timing_ms,
            memory_used_bytes: None,
            operations_count: None,
        }
    }

    /// Create a failed result
    pub fn error(error: String, timing_ms: u64) -> Self {
        Self {
            success: false,
            stdout: String::new(),
            stderr: String::new(),
            result: None,
            error: Some(error),
            timing_ms,
            memory_used_bytes: None,
            operations_count: None,
        }
    }

    /// Create a failed result with captured output
    pub fn error_with_output(
        error: String,
        stdout: String,
        stderr: String,
        timing_ms: u64,
    ) -> Self {
        Self {
            success: false,
            stdout,
            stderr,
            result: None,
            error: Some(error),
            timing_ms,
            memory_used_bytes: None,
            operations_count: None,
        }
    }
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self::error("No execution performed".to_string(), 0)
    }
}

/// Execution limits for sandboxing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLimits {
    /// Maximum execution time in milliseconds
    #[serde(default = "ExecutionLimits::default_timeout_ms")]
    pub max_timeout_ms: u64,

    /// Maximum memory usage in MB
    #[serde(default = "ExecutionLimits::default_memory_mb")]
    pub max_memory_mb: u32,

    /// Maximum output size in bytes
    #[serde(default = "ExecutionLimits::default_output_bytes")]
    pub max_output_bytes: usize,

    /// Maximum number of operations (for loop prevention)
    #[serde(default = "ExecutionLimits::default_operations")]
    pub max_operations: u64,

    /// Maximum call stack depth
    #[serde(default = "ExecutionLimits::default_call_depth")]
    pub max_call_depth: u32,

    /// Maximum string length
    #[serde(default = "ExecutionLimits::default_string_length")]
    pub max_string_length: usize,

    /// Maximum array/list length
    #[serde(default = "ExecutionLimits::default_array_length")]
    pub max_array_length: usize,

    /// Maximum map/dict entries
    #[serde(default = "ExecutionLimits::default_map_size")]
    pub max_map_size: usize,
}

impl ExecutionLimits {
    fn default_timeout_ms() -> u64 {
        30_000
    }
    fn default_memory_mb() -> u32 {
        256
    }
    fn default_output_bytes() -> usize {
        1_048_576 // 1MB
    }
    fn default_operations() -> u64 {
        1_000_000
    }
    fn default_call_depth() -> u32 {
        64
    }
    fn default_string_length() -> usize {
        10_485_760 // 10MB
    }
    fn default_array_length() -> usize {
        100_000
    }
    fn default_map_size() -> usize {
        10_000
    }

    /// Create strict limits for untrusted code
    pub fn strict() -> Self {
        Self {
            max_timeout_ms: 5_000,
            max_memory_mb: 64,
            max_output_bytes: 65_536, // 64KB
            max_operations: 100_000,
            max_call_depth: 32,
            max_string_length: 1_048_576, // 1MB
            max_array_length: 10_000,
            max_map_size: 1_000,
        }
    }

    /// Create relaxed limits for trusted code
    pub fn relaxed() -> Self {
        Self {
            max_timeout_ms: 120_000, // 2 minutes
            max_memory_mb: 512,
            max_output_bytes: 10_485_760, // 10MB
            max_operations: 10_000_000,
            max_call_depth: 128,
            max_string_length: RELAXED_MAX_STRING_LENGTH,
            max_array_length: 1_000_000,
            max_map_size: 100_000,
        }
    }
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_timeout_ms: Self::default_timeout_ms(),
            max_memory_mb: Self::default_memory_mb(),
            max_output_bytes: Self::default_output_bytes(),
            max_operations: Self::default_operations(),
            max_call_depth: Self::default_call_depth(),
            max_string_length: Self::default_string_length(),
            max_array_length: Self::default_array_length(),
            max_map_size: Self::default_map_size(),
        }
    }
}

/// Error types for code execution
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum ExecutionError {
    /// The requested language is not supported or not enabled.
    #[error("Language '{0}' is not supported or not enabled")]
    UnsupportedLanguage(String),

    /// Execution timed out after the given number of milliseconds.
    #[error("Execution timed out after {0}ms")]
    Timeout(u64),

    /// Memory limit exceeded (in MB).
    #[error("Memory limit exceeded: {0}MB")]
    MemoryLimitExceeded(u32),

    /// Operation limit exceeded.
    #[error("Operation limit exceeded: {0} operations")]
    OperationLimitExceeded(u64),

    /// Output exceeded the maximum size (in bytes).
    #[error("Output too large: {0} bytes")]
    OutputTooLarge(usize),

    /// Syntax error in the submitted code.
    #[error("Syntax error: {0}")]
    SyntaxError(String),

    /// Runtime error during execution.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Internal executor error.
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl ExecutionError {
    /// Convert this error into an [`ExecutionResult`] with the given timing.
    pub fn to_result(&self, timing_ms: u64) -> ExecutionResult {
        ExecutionResult::error(self.to_string(), timing_ms)
    }
}

/// Sandbox profile presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SandboxProfile {
    /// Minimal - No I/O, basic math only
    Minimal,
    /// Standard - Console output, JSON, basic stdlib
    #[default]
    Standard,
    /// Extended - More stdlib, regex, datetime
    Extended,
}

impl SandboxProfile {
    /// Get allowed modules for this profile
    pub fn allowed_modules(&self) -> Vec<&'static str> {
        match self {
            SandboxProfile::Minimal => vec!["math"],
            SandboxProfile::Standard => vec!["math", "json", "string", "array", "print"],
            SandboxProfile::Extended => vec![
                "math", "json", "string", "array", "print", "datetime", "regex", "base64",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_parsing() {
        assert_eq!(Language::parse("JAVASCRIPT"), Some(Language::JavaScript));
        assert_eq!(Language::parse("js"), Some(Language::JavaScript));
        assert_eq!(Language::parse("lua"), Some(Language::Lua));
        assert_eq!(Language::parse("rhai"), Some(Language::Rhai));
        assert_eq!(Language::parse("python"), None);
        assert_eq!(Language::parse("unknown"), None);
    }

    #[test]
    fn test_execution_limits_profiles() {
        let strict = ExecutionLimits::strict();
        let relaxed = ExecutionLimits::relaxed();

        assert!(strict.max_timeout_ms < relaxed.max_timeout_ms);
        assert!(strict.max_memory_mb < relaxed.max_memory_mb);
        assert!(strict.max_operations < relaxed.max_operations);
    }

    #[test]
    fn test_execution_result_creation() {
        let success =
            ExecutionResult::success("Hello".to_string(), Some(serde_json::json!(42)), 100);
        assert!(success.success);
        assert_eq!(success.stdout, "Hello");

        let error = ExecutionResult::error("Failed".to_string(), 50);
        assert!(!error.success);
        assert_eq!(error.error, Some("Failed".to_string()));
    }
}
