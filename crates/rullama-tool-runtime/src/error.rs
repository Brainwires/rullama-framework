//! Tool Error Taxonomy and Classification
//!
//! Based on AgentDebug paper (arxiv:2509.25370) - provides error classification
//! for intelligent retry strategies and SEAL learning integration.

use std::time::Duration;

const DEFAULT_MAX_RETRY_ATTEMPTS: u32 = 3;
const EXPONENTIAL_BACKOFF_BASE_SECS: u64 = 2;
const DEFAULT_BACKOFF_BASE_MS: u64 = 500;

/// Error taxonomy based on AgentDebug paper (arxiv:2509.25370)
#[derive(Debug, Clone, PartialEq)]
pub enum ToolErrorCategory {
    /// Transient errors that may succeed on retry (network issues, timeouts).
    Transient {
        /// Error message.
        error: String,
        /// Retry strategy for this error.
        retry_strategy: RetryStrategy,
    },
    /// Input validation errors - need different input parameters.
    InputValidation {
        /// Error message.
        error: String,
        /// Suggested fix for the input.
        suggestion: Option<String>,
    },
    /// External service errors (API limits, service unavailable).
    ExternalService {
        /// Error message.
        error: String,
        /// Name of the external service.
        service: String,
        /// Suggested delay before retry.
        retry_after: Option<Duration>,
    },
    /// Permission/access errors - won't succeed without user action.
    Permission {
        /// Error message.
        error: String,
        /// The permission required to proceed.
        required_permission: String,
    },
    /// Logic errors - indicates model misunderstanding of tool usage.
    Logic {
        /// Error message.
        error: String,
        /// Context in which the logic error occurred.
        context: String,
    },
    /// Resource errors - file not found, memory, disk space.
    Resource {
        /// Error message.
        error: String,
        /// Type of resource involved.
        resource_type: ResourceType,
    },
    /// Unknown/unclassified errors.
    Unknown {
        /// Error message.
        error: String,
    },
}

impl ToolErrorCategory {
    /// Return the category name as a static string.
    pub fn category_name(&self) -> &'static str {
        match self {
            ToolErrorCategory::Transient { .. } => "transient",
            ToolErrorCategory::InputValidation { .. } => "input_validation",
            ToolErrorCategory::ExternalService { .. } => "external_service",
            ToolErrorCategory::Permission { .. } => "permission",
            ToolErrorCategory::Logic { .. } => "logic",
            ToolErrorCategory::Resource { .. } => "resource",
            ToolErrorCategory::Unknown { .. } => "unknown",
        }
    }

    /// Return the error message string.
    pub fn error_message(&self) -> &str {
        match self {
            ToolErrorCategory::Transient { error, .. } => error,
            ToolErrorCategory::InputValidation { error, .. } => error,
            ToolErrorCategory::ExternalService { error, .. } => error,
            ToolErrorCategory::Permission { error, .. } => error,
            ToolErrorCategory::Logic { error, .. } => error,
            ToolErrorCategory::Resource { error, .. } => error,
            ToolErrorCategory::Unknown { error } => error,
        }
    }

    /// Whether this error category is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ToolErrorCategory::Transient { .. } | ToolErrorCategory::ExternalService { .. }
        )
    }

    /// Return the retry strategy for this error.
    pub fn retry_strategy(&self) -> RetryStrategy {
        match self {
            ToolErrorCategory::Transient { retry_strategy, .. } => retry_strategy.clone(),
            ToolErrorCategory::ExternalService { retry_after, .. } => {
                if let Some(delay) = retry_after {
                    RetryStrategy::FixedDelay {
                        delay: *delay,
                        max_attempts: DEFAULT_MAX_RETRY_ATTEMPTS,
                    }
                } else {
                    RetryStrategy::ExponentialBackoff {
                        base: Duration::from_secs(EXPONENTIAL_BACKOFF_BASE_SECS),
                        max_attempts: DEFAULT_MAX_RETRY_ATTEMPTS,
                    }
                }
            }
            _ => RetryStrategy::NoRetry,
        }
    }

    /// Get a suggestion for resolving this error, if available.
    pub fn get_suggestion(&self) -> Option<String> {
        match self {
            ToolErrorCategory::InputValidation { suggestion, .. } => suggestion.clone(),
            ToolErrorCategory::Permission {
                required_permission,
                ..
            } => Some(format!("Requires {} permission", required_permission)),
            ToolErrorCategory::Resource { resource_type, .. } => {
                Some(format!("Resource issue: {:?}", resource_type))
            }
            _ => None,
        }
    }
}

/// Resource types for Resource errors
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// File not found.
    FileNotFound,
    /// Directory not found.
    DirectoryNotFound,
    /// Insufficient disk space.
    DiskSpace,
    /// Insufficient memory.
    Memory,
    /// Process limit reached.
    ProcessLimit,
    /// Other resource type.
    Other(String),
}

/// Retry strategy for transient errors
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Do not retry.
    NoRetry,
    /// Retry immediately up to a maximum number of attempts.
    Immediate {
        /// Maximum number of retry attempts.
        max_attempts: u32,
    },
    /// Retry with a fixed delay between attempts.
    FixedDelay {
        /// Delay between retries.
        delay: Duration,
        /// Maximum number of retry attempts.
        max_attempts: u32,
    },
    /// Retry with exponential backoff.
    ExponentialBackoff {
        /// Base duration for backoff calculation.
        base: Duration,
        /// Maximum number of retry attempts.
        max_attempts: u32,
    },
}

impl RetryStrategy {
    /// Compute the delay for a given retry attempt, or `None` if exhausted.
    pub fn delay_for_attempt(&self, attempt: u32) -> Option<Duration> {
        match self {
            RetryStrategy::NoRetry => None,
            RetryStrategy::Immediate { max_attempts } => {
                if attempt < *max_attempts {
                    Some(Duration::ZERO)
                } else {
                    None
                }
            }
            RetryStrategy::FixedDelay {
                delay,
                max_attempts,
            } => {
                if attempt < *max_attempts {
                    Some(*delay)
                } else {
                    None
                }
            }
            RetryStrategy::ExponentialBackoff { base, max_attempts } => {
                if attempt < *max_attempts {
                    Some(*base * 2u32.pow(attempt))
                } else {
                    None
                }
            }
        }
    }

    /// Return the maximum number of retry attempts.
    pub fn max_attempts(&self) -> u32 {
        match self {
            RetryStrategy::NoRetry => 0,
            RetryStrategy::Immediate { max_attempts } => *max_attempts,
            RetryStrategy::FixedDelay { max_attempts, .. } => *max_attempts,
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => *max_attempts,
        }
    }
}

impl Default for RetryStrategy {
    fn default() -> Self {
        RetryStrategy::ExponentialBackoff {
            base: Duration::from_millis(DEFAULT_BACKOFF_BASE_MS),
            max_attempts: DEFAULT_MAX_RETRY_ATTEMPTS,
        }
    }
}

struct ErrorPattern {
    keywords: &'static [&'static str],
    category_builder: fn(&str) -> ToolErrorCategory,
}

const ERROR_PATTERNS: &[ErrorPattern] = &[
    ErrorPattern {
        keywords: &[
            "connection refused",
            "connection reset",
            "connection timed out",
        ],
        category_builder: |e| ToolErrorCategory::Transient {
            error: e.to_string(),
            retry_strategy: RetryStrategy::ExponentialBackoff {
                base: Duration::from_secs(1),
                max_attempts: 3,
            },
        },
    },
    ErrorPattern {
        keywords: &["timeout", "timed out", "deadline exceeded"],
        category_builder: |e| ToolErrorCategory::Transient {
            error: e.to_string(),
            retry_strategy: RetryStrategy::ExponentialBackoff {
                base: Duration::from_secs(2),
                max_attempts: 3,
            },
        },
    },
    ErrorPattern {
        keywords: &["network", "dns", "host unreachable", "no route"],
        category_builder: |e| ToolErrorCategory::Transient {
            error: e.to_string(),
            retry_strategy: RetryStrategy::ExponentialBackoff {
                base: Duration::from_secs(1),
                max_attempts: 3,
            },
        },
    },
    ErrorPattern {
        keywords: &["rate limit", "too many requests", "429", "quota exceeded"],
        category_builder: |e| ToolErrorCategory::ExternalService {
            error: e.to_string(),
            service: "API".to_string(),
            retry_after: Some(Duration::from_secs(5)),
        },
    },
    ErrorPattern {
        keywords: &["service unavailable", "503", "502", "bad gateway"],
        category_builder: |e| ToolErrorCategory::ExternalService {
            error: e.to_string(),
            service: "external".to_string(),
            retry_after: Some(Duration::from_secs(3)),
        },
    },
    ErrorPattern {
        keywords: &["internal server error", "500"],
        category_builder: |e| ToolErrorCategory::ExternalService {
            error: e.to_string(),
            service: "external".to_string(),
            retry_after: Some(Duration::from_secs(2)),
        },
    },
    ErrorPattern {
        keywords: &["permission denied", "access denied", "forbidden", "403"],
        category_builder: |e| ToolErrorCategory::Permission {
            error: e.to_string(),
            required_permission: "access".to_string(),
        },
    },
    ErrorPattern {
        keywords: &["unauthorized", "401", "authentication"],
        category_builder: |e| ToolErrorCategory::Permission {
            error: e.to_string(),
            required_permission: "authentication".to_string(),
        },
    },
    ErrorPattern {
        keywords: &["read-only", "cannot write", "not writable"],
        category_builder: |e| ToolErrorCategory::Permission {
            error: e.to_string(),
            required_permission: "write".to_string(),
        },
    },
    ErrorPattern {
        keywords: &[
            "no such file",
            "file not found",
            "cannot find",
            "does not exist",
        ],
        category_builder: |e| ToolErrorCategory::Resource {
            error: e.to_string(),
            resource_type: ResourceType::FileNotFound,
        },
    },
    ErrorPattern {
        keywords: &["not a directory", "is a directory", "directory not found"],
        category_builder: |e| ToolErrorCategory::Resource {
            error: e.to_string(),
            resource_type: ResourceType::DirectoryNotFound,
        },
    },
    ErrorPattern {
        keywords: &["no space left", "disk full", "quota"],
        category_builder: |e| ToolErrorCategory::Resource {
            error: e.to_string(),
            resource_type: ResourceType::DiskSpace,
        },
    },
    ErrorPattern {
        keywords: &["out of memory", "cannot allocate", "memory"],
        category_builder: |e| ToolErrorCategory::Resource {
            error: e.to_string(),
            resource_type: ResourceType::Memory,
        },
    },
    ErrorPattern {
        keywords: &["invalid argument", "invalid parameter", "invalid input"],
        category_builder: |e| ToolErrorCategory::InputValidation {
            error: e.to_string(),
            suggestion: Some("Check the input parameters".to_string()),
        },
    },
    ErrorPattern {
        keywords: &["missing required", "required field", "missing argument"],
        category_builder: |e| ToolErrorCategory::InputValidation {
            error: e.to_string(),
            suggestion: Some("Provide all required parameters".to_string()),
        },
    },
    ErrorPattern {
        keywords: &["invalid path", "bad path", "malformed"],
        category_builder: |e| ToolErrorCategory::InputValidation {
            error: e.to_string(),
            suggestion: Some("Check the path format".to_string()),
        },
    },
    ErrorPattern {
        keywords: &["type error", "expected", "invalid type"],
        category_builder: |e| ToolErrorCategory::InputValidation {
            error: e.to_string(),
            suggestion: Some("Check parameter types".to_string()),
        },
    },
];

/// Classify an error from a tool result
pub fn classify_error(tool_name: &str, error: &str) -> ToolErrorCategory {
    let error_lower = error.to_lowercase();
    for pattern in ERROR_PATTERNS {
        if pattern.keywords.iter().any(|kw| error_lower.contains(kw)) {
            return (pattern.category_builder)(error);
        }
    }
    match tool_name {
        "bash" | "Bash" | "execute_command" => classify_bash_error(error),
        "read_file" | "ReadFile" | "Read" | "write_file" | "WriteFile" | "Write" => {
            classify_file_error(error)
        }
        "web_search" | "WebSearch" | "web_fetch" | "WebFetch" | "fetch_url" => {
            classify_web_error(error)
        }
        _ => ToolErrorCategory::Unknown {
            error: error.to_string(),
        },
    }
}

fn classify_bash_error(error: &str) -> ToolErrorCategory {
    let error_lower = error.to_lowercase();
    if error_lower.contains("command not found") {
        ToolErrorCategory::InputValidation {
            error: error.to_string(),
            suggestion: Some(
                "Command does not exist. Check spelling or install the program.".to_string(),
            ),
        }
    } else if error_lower.contains("exit code") || error_lower.contains("failed with") {
        ToolErrorCategory::Logic {
            error: error.to_string(),
            context: "bash_execution".to_string(),
        }
    } else {
        ToolErrorCategory::Unknown {
            error: error.to_string(),
        }
    }
}

fn classify_file_error(error: &str) -> ToolErrorCategory {
    let error_lower = error.to_lowercase();
    if error_lower.contains("binary") || error_lower.contains("not valid utf-8") {
        ToolErrorCategory::InputValidation {
            error: error.to_string(),
            suggestion: Some("File is binary or not valid text.".to_string()),
        }
    } else if error_lower.contains("too large") {
        ToolErrorCategory::Resource {
            error: error.to_string(),
            resource_type: ResourceType::Memory,
        }
    } else {
        ToolErrorCategory::Unknown {
            error: error.to_string(),
        }
    }
}

fn classify_web_error(error: &str) -> ToolErrorCategory {
    let error_lower = error.to_lowercase();
    if error_lower.contains("ssl") || error_lower.contains("certificate") {
        ToolErrorCategory::ExternalService {
            error: error.to_string(),
            service: "SSL/TLS".to_string(),
            retry_after: None,
        }
    } else if error_lower.contains("redirect") {
        ToolErrorCategory::InputValidation {
            error: error.to_string(),
            suggestion: Some("URL redirected. Follow the redirect or use the new URL.".to_string()),
        }
    } else {
        ToolErrorCategory::Unknown {
            error: error.to_string(),
        }
    }
}

/// Outcome of a tool execution (for SEAL learning)
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// Name of the tool that was executed.
    pub tool_name: String,
    /// Whether execution succeeded.
    pub success: bool,
    /// Number of retries performed.
    pub retries: u32,
    /// Error category if the tool failed.
    pub error_category: Option<ToolErrorCategory>,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

impl ToolOutcome {
    /// Create a successful tool outcome.
    pub fn success(tool_name: &str, retries: u32, execution_time_ms: u64) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            success: true,
            retries,
            error_category: None,
            execution_time_ms,
        }
    }
    /// Create a failed tool outcome.
    pub fn failure(
        tool_name: &str,
        retries: u32,
        error_category: ToolErrorCategory,
        execution_time_ms: u64,
    ) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            success: false,
            retries,
            error_category: Some(error_category),
            execution_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_transient_errors() {
        let cat = classify_error("bash", "Connection refused");
        assert!(matches!(cat, ToolErrorCategory::Transient { .. }));
        assert!(cat.is_retryable());
    }

    #[test]
    fn test_classify_permission_errors() {
        let cat = classify_error("write_file", "Permission denied");
        assert!(matches!(cat, ToolErrorCategory::Permission { .. }));
        assert!(!cat.is_retryable());
    }

    #[test]
    fn test_classify_resource_errors() {
        let cat = classify_error("read_file", "No such file or directory");
        assert!(matches!(
            cat,
            ToolErrorCategory::Resource {
                resource_type: ResourceType::FileNotFound,
                ..
            }
        ));
    }

    #[test]
    fn test_retry_strategy_delay() {
        let strategy = RetryStrategy::ExponentialBackoff {
            base: Duration::from_millis(100),
            max_attempts: 3,
        };
        assert_eq!(
            strategy.delay_for_attempt(0),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            strategy.delay_for_attempt(1),
            Some(Duration::from_millis(200))
        );
        assert_eq!(
            strategy.delay_for_attempt(2),
            Some(Duration::from_millis(400))
        );
        assert_eq!(strategy.delay_for_attempt(3), None);
    }
}
