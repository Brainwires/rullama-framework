//! MDAP (Massively Decomposed Agentic Processes) Error Types
//!
//! Provides domain-specific error types for the MDAP framework implementation,
//! based on the MAKER paper's error handling requirements.

use std::collections::HashMap;
use thiserror::Error;

/// Main error type for the MDAP system
#[derive(Error, Debug)]
pub enum MdapError {
    /// Voting error.
    #[error("Voting error: {0}")]
    Voting(#[from] VotingError),

    /// Red-flag validation error.
    #[error("Red-flag validation error: {0}")]
    RedFlag(#[from] RedFlagError),

    /// Decomposition error.
    #[error("Decomposition error: {0}")]
    Decomposition(#[from] DecompositionError),

    /// Microagent execution error.
    #[error("Microagent error: {0}")]
    Microagent(#[from] MicroagentError),

    /// Composition error.
    #[error("Composition error: {0}")]
    Composition(#[from] CompositionError),

    /// Scaling law calculation error.
    #[error("Scaling error: {0}")]
    Scaling(#[from] ScalingError),

    /// AI provider error.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(#[from] MdapConfigError),

    /// I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Semaphore acquire error.
    #[error("Semaphore acquire error: {0}")]
    Semaphore(String),

    /// Task join error.
    #[error("Task join error: {0}")]
    TaskJoin(String),

    /// Catch-all error.
    #[error("{0}")]
    Other(String),

    /// Tool recursion limit reached.
    #[error("Tool recursion limit reached: depth {depth} >= max {max_depth}")]
    ToolRecursionLimit {
        /// Current recursion depth.
        depth: u32,
        /// Maximum allowed depth.
        max_depth: u32,
    },

    /// Tool execution failed.
    #[error("Tool execution failed: {tool} - {reason}")]
    ToolExecutionFailed {
        /// Tool name.
        tool: String,
        /// Failure reason.
        reason: String,
    },

    /// Tool not allowed for this microagent.
    #[error("Tool not allowed for microagent: {tool} (category: {category})")]
    ToolNotAllowed {
        /// Tool name.
        tool: String,
        /// Tool category.
        category: String,
    },

    /// Tool intent parsing failed.
    #[error("Tool intent parsing failed: {0}")]
    ToolIntentParseFailed(String),

    /// General configuration error.
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Errors related to the first-to-ahead-by-k voting system (Algorithm 2)
#[derive(Error, Debug)]
pub enum VotingError {
    /// Maximum samples exceeded without reaching consensus.
    #[error("Maximum samples exceeded: {samples} samples taken, no consensus reached")]
    MaxSamplesExceeded {
        /// Number of samples taken.
        samples: u32,
        /// Vote tally per response hash.
        votes: HashMap<String, u32>,
    },

    /// All samples were red-flagged as invalid.
    #[error("All samples were red-flagged: {red_flagged}/{total} samples invalid")]
    AllSamplesRedFlagged {
        /// Number of red-flagged samples.
        red_flagged: u32,
        /// Total number of samples.
        total: u32,
    },

    /// Voting was cancelled.
    #[error("Voting cancelled")]
    Cancelled,

    /// No valid responses received.
    #[error("No valid responses received after {attempts} attempts")]
    NoValidResponses {
        /// Number of attempts made.
        attempts: u32,
    },

    /// Sampler returned an error.
    #[error("Sampler returned error: {0}")]
    SamplerError(String),

    /// Unable to hash a response for comparison.
    #[error("Vote comparison failed: unable to hash response")]
    HashError,

    /// Invalid k value (must be >= 1).
    #[error("Invalid k value: k must be >= 1, got {0}")]
    InvalidK(u32),

    /// Parallel execution error.
    #[error("Parallel execution error: {0}")]
    ParallelError(String),
}

/// Errors related to red-flag validation (Algorithm 3)
#[derive(Error, Debug)]
pub enum RedFlagError {
    /// Response exceeds token limit.
    #[error("Response too long: {tokens} tokens exceeds limit of {limit}")]
    ResponseTooLong {
        /// Actual token count.
        tokens: u32,
        /// Maximum allowed tokens.
        limit: u32,
    },

    /// Response format does not match expected format.
    #[error("Invalid format: expected {expected}, got {got}")]
    InvalidFormat {
        /// Expected format.
        expected: String,
        /// Actual format received.
        got: String,
    },

    /// Self-correction pattern detected in response.
    #[error("Self-correction detected: '{pattern}' indicates model confusion")]
    SelfCorrectionDetected {
        /// Detected self-correction pattern.
        pattern: String,
    },

    /// Confused reasoning detected in response.
    #[error("Confused reasoning detected: '{pattern}'")]
    ConfusedReasoning {
        /// Detected confusion pattern.
        pattern: String,
    },

    /// Parse error in response.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Response is empty.
    #[error("Empty response")]
    EmptyResponse,

    /// Invalid JSON structure in response.
    #[error("Invalid JSON structure: {0}")]
    InvalidJson(String),

    /// Missing required field in response.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Validation pattern error.
    #[error("Validation pattern error: {0}")]
    PatternError(String),
}

/// Errors related to task decomposition
#[derive(Error, Debug)]
pub enum DecompositionError {
    /// Maximum decomposition depth exceeded.
    #[error("Maximum decomposition depth exceeded: {depth} > {max_depth}")]
    MaxDepthExceeded {
        /// Current depth.
        depth: u32,
        /// Maximum allowed depth.
        max_depth: u32,
    },

    /// Task cannot be decomposed further.
    #[error("Task cannot be decomposed further: {0}")]
    CannotDecompose(String),

    /// Circular dependency detected in subtasks.
    #[error("Circular dependency detected in subtasks: {0}")]
    CircularDependency(String),

    /// Invalid subtask dependency.
    #[error("Invalid subtask dependency: '{subtask}' depends on non-existent '{dependency}'")]
    InvalidDependency {
        /// Subtask with the invalid dependency.
        subtask: String,
        /// Non-existent dependency name.
        dependency: String,
    },

    /// Decomposition voting failed.
    #[error("Decomposition voting failed: {0}")]
    VotingFailed(String),

    /// Empty decomposition result.
    #[error("Empty decomposition result for task: {0}")]
    EmptyResult(String),

    /// Invalid decomposition strategy.
    #[error("Invalid decomposition strategy: {0}")]
    InvalidStrategy(String),

    /// Discriminator error.
    #[error("Discriminator error: {0}")]
    DiscriminatorError(String),
}

/// Errors related to microagent execution
#[derive(Error, Debug)]
pub enum MicroagentError {
    /// Subtask execution failed.
    #[error("Subtask execution failed: {subtask_id} - {reason}")]
    ExecutionFailed {
        /// Subtask identifier.
        subtask_id: String,
        /// Failure reason.
        reason: String,
    },

    /// Subtask timed out.
    #[error("Subtask timeout after {timeout_ms}ms: {subtask_id}")]
    Timeout {
        /// Subtask identifier.
        subtask_id: String,
        /// Timeout duration in milliseconds.
        timeout_ms: u64,
    },

    /// Invalid input for subtask.
    #[error("Invalid input state for subtask '{subtask_id}': {reason}")]
    InvalidInput {
        /// Subtask identifier.
        subtask_id: String,
        /// Reason for invalid input.
        reason: String,
    },

    /// Output parsing failed for subtask.
    #[error("Output parsing failed for subtask '{subtask_id}': {reason}")]
    OutputParseFailed {
        /// Subtask identifier.
        subtask_id: String,
        /// Parsing failure reason.
        reason: String,
    },

    /// Provider communication error.
    #[error("Provider communication error: {0}")]
    ProviderError(String),

    /// Context too large for microagent.
    #[error("Context too large for microagent: {size} tokens > {limit} limit")]
    ContextTooLarge {
        /// Actual context size in tokens.
        size: u32,
        /// Maximum token limit.
        limit: u32,
    },

    /// Missing dependency result.
    #[error("Missing dependency result: subtask '{subtask_id}' requires '{dependency}'")]
    MissingDependency {
        /// Subtask identifier.
        subtask_id: String,
        /// Missing dependency name.
        dependency: String,
    },
}

/// Errors related to result composition
#[derive(Error, Debug)]
pub enum CompositionError {
    /// Missing subtask result.
    #[error("Missing subtask result: {0}")]
    MissingResult(String),

    /// Incompatible result types for composition.
    #[error("Incompatible result types: cannot compose {type_a} with {type_b}")]
    IncompatibleTypes {
        /// First type.
        type_a: String,
        /// Second type.
        type_b: String,
    },

    /// Composition function not found.
    #[error("Composition function '{function}' not found")]
    FunctionNotFound {
        /// Function name.
        function: String,
    },

    /// Composition execution failed.
    #[error("Composition execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid composition order.
    #[error("Invalid composition order: {0}")]
    InvalidOrder(String),

    /// Result validation failed.
    #[error("Result validation failed: {0}")]
    ValidationFailed(String),
}

/// Errors related to scaling law calculations
#[derive(Error, Debug)]
pub enum ScalingError {
    /// Invalid success probability value.
    #[error("Invalid success probability: {0} must be in range (0.5, 1.0)")]
    InvalidSuccessProbability(f64),

    /// Invalid target probability value.
    #[error("Invalid target probability: {0} must be in range (0.0, 1.0)")]
    InvalidTargetProbability(f64),

    /// Invalid step count.
    #[error("Invalid step count: must be > 0, got {0}")]
    InvalidStepCount(u64),

    /// Voting cannot converge at this success rate.
    #[error("Voting cannot converge: per-step success rate {p} <= 0.5")]
    VotingCannotConverge {
        /// Per-step success probability.
        p: f64,
    },

    /// Cost estimation failed.
    #[error("Cost estimation failed: {0}")]
    CostEstimationFailed(String),

    /// Numerical overflow in calculation.
    #[error("Numerical overflow in calculation: {0}")]
    NumericalOverflow(String),
}

/// Errors related to MDAP configuration
#[derive(Error, Debug)]
pub enum MdapConfigError {
    /// Invalid k value.
    #[error("Invalid k value: must be >= 1, got {0}")]
    InvalidK(u32),

    /// Invalid target success rate.
    #[error("Invalid target success rate: must be in (0.0, 1.0), got {0}")]
    InvalidTargetSuccessRate(f64),

    /// Invalid parallel samples count.
    #[error("Invalid parallel samples: must be 1-4, got {0}")]
    InvalidParallelSamples(u32),

    /// Invalid max samples per subtask.
    #[error("Invalid max samples per subtask: must be > 0, got {0}")]
    InvalidMaxSamples(u32),

    /// Invalid max response tokens.
    #[error("Invalid max response tokens: must be > 0, got {0}")]
    InvalidMaxTokens(u32),

    /// Invalid max decomposition depth.
    #[error("Invalid decomposition max depth: must be > 0, got {0}")]
    InvalidMaxDepth(u32),

    /// Configuration file not found.
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    /// Configuration parse error.
    #[error("Configuration parse error: {0}")]
    ParseError(String),
}

// Conversion from anyhow::Error to MdapError
impl From<anyhow::Error> for MdapError {
    fn from(err: anyhow::Error) -> Self {
        MdapError::Other(format!("{:#}", err))
    }
}

// Conversion from tokio semaphore acquire error
impl From<tokio::sync::AcquireError> for MdapError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        MdapError::Semaphore(err.to_string())
    }
}

// Conversion from tokio join error
impl From<tokio::task::JoinError> for MdapError {
    fn from(err: tokio::task::JoinError) -> Self {
        MdapError::TaskJoin(err.to_string())
    }
}

// Helper methods for MdapError
impl MdapError {
    /// Create a new error from a string message
    pub fn other(msg: impl Into<String>) -> Self {
        MdapError::Other(msg.into())
    }

    /// Create a provider error
    pub fn provider(msg: impl Into<String>) -> Self {
        MdapError::Provider(msg.into())
    }

    /// Convert to a user-facing error string
    pub fn to_user_string(&self) -> String {
        format!("{}", self)
    }

    /// Check if this is a user/configuration error vs system/runtime error
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            MdapError::Config(_)
                | MdapError::Scaling(ScalingError::InvalidSuccessProbability(_))
                | MdapError::Scaling(ScalingError::InvalidTargetProbability(_))
        )
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            MdapError::Provider(_)
                | MdapError::Semaphore(_)
                | MdapError::Voting(VotingError::SamplerError(_))
                | MdapError::Microagent(MicroagentError::ProviderError(_))
                | MdapError::ToolExecutionFailed { .. }
        )
    }

    /// Check if this is a tool-related error
    pub fn is_tool_error(&self) -> bool {
        matches!(
            self,
            MdapError::ToolRecursionLimit { .. }
                | MdapError::ToolExecutionFailed { .. }
                | MdapError::ToolNotAllowed { .. }
                | MdapError::ToolIntentParseFailed(_)
        )
    }

    /// Check if this error indicates the voting process should be restarted
    pub fn should_restart_voting(&self) -> bool {
        matches!(
            self,
            MdapError::RedFlag(RedFlagError::ResponseTooLong { .. })
                | MdapError::RedFlag(RedFlagError::SelfCorrectionDetected { .. })
                | MdapError::RedFlag(RedFlagError::ConfusedReasoning { .. })
        )
    }

    /// Check if this error is a red-flag (should discard and resample)
    pub fn is_red_flag(&self) -> bool {
        matches!(self, MdapError::RedFlag(_))
    }
}

/// Result type alias for MDAP operations
pub type MdapResult<T> = Result<T, MdapError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_error_display() {
        let mut votes = HashMap::new();
        votes.insert("option_a".to_string(), 3);
        votes.insert("option_b".to_string(), 2);

        let err = VotingError::MaxSamplesExceeded { samples: 50, votes };
        assert!(
            err.to_string()
                .contains("Maximum samples exceeded: 50 samples taken")
        );
    }

    #[test]
    fn test_red_flag_error_display() {
        let err = RedFlagError::ResponseTooLong {
            tokens: 800,
            limit: 750,
        };
        assert_eq!(
            err.to_string(),
            "Response too long: 800 tokens exceeds limit of 750"
        );
    }

    #[test]
    fn test_self_correction_error() {
        let err = RedFlagError::SelfCorrectionDetected {
            pattern: "Wait,".to_string(),
        };
        assert!(err.to_string().contains("Wait,"));
        assert!(err.to_string().contains("model confusion"));
    }

    #[test]
    fn test_decomposition_error() {
        let err = DecompositionError::MaxDepthExceeded {
            depth: 15,
            max_depth: 10,
        };
        assert_eq!(
            err.to_string(),
            "Maximum decomposition depth exceeded: 15 > 10"
        );
    }

    #[test]
    fn test_microagent_error() {
        let err = MicroagentError::Timeout {
            subtask_id: "task_001".to_string(),
            timeout_ms: 5000,
        };
        assert!(err.to_string().contains("task_001"));
        assert!(err.to_string().contains("5000ms"));
    }

    #[test]
    fn test_scaling_error() {
        let err = ScalingError::VotingCannotConverge { p: 0.45 };
        assert!(err.to_string().contains("0.45"));
        assert!(err.to_string().contains("<= 0.5"));
    }

    #[test]
    fn test_config_error() {
        let err = MdapConfigError::InvalidParallelSamples(8);
        assert_eq!(
            err.to_string(),
            "Invalid parallel samples: must be 1-4, got 8"
        );
    }

    #[test]
    fn test_mdap_error_from_voting() {
        let voting_err = VotingError::Cancelled;
        let mdap_err: MdapError = voting_err.into();
        assert!(matches!(mdap_err, MdapError::Voting(_)));
    }

    #[test]
    fn test_mdap_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("test error");
        let mdap_err: MdapError = anyhow_err.into();
        assert!(matches!(mdap_err, MdapError::Other(_)));
    }

    #[test]
    fn test_is_user_error() {
        let user_err = MdapError::Config(MdapConfigError::InvalidK(0));
        assert!(user_err.is_user_error());

        let system_err = MdapError::Provider("connection failed".to_string());
        assert!(!system_err.is_user_error());
    }

    #[test]
    fn test_is_retryable() {
        let retryable = MdapError::Provider("timeout".to_string());
        assert!(retryable.is_retryable());

        let not_retryable = MdapError::Config(MdapConfigError::InvalidK(0));
        assert!(!not_retryable.is_retryable());
    }

    #[test]
    fn test_is_red_flag() {
        let red_flag = MdapError::RedFlag(RedFlagError::EmptyResponse);
        assert!(red_flag.is_red_flag());

        let not_red_flag = MdapError::Voting(VotingError::Cancelled);
        assert!(!not_red_flag.is_red_flag());
    }

    #[test]
    fn test_should_restart_voting() {
        let should_restart = MdapError::RedFlag(RedFlagError::SelfCorrectionDetected {
            pattern: "Actually,".to_string(),
        });
        assert!(should_restart.should_restart_voting());

        let should_not_restart = MdapError::Voting(VotingError::MaxSamplesExceeded {
            samples: 50,
            votes: HashMap::new(),
        });
        assert!(!should_not_restart.should_restart_voting());
    }

    #[test]
    fn test_error_chain() {
        let red_flag_err = RedFlagError::InvalidFormat {
            expected: "JSON".to_string(),
            got: "plain text".to_string(),
        };
        let mdap_err: MdapError = red_flag_err.into();
        assert!(matches!(mdap_err, MdapError::RedFlag(_)));
        assert!(mdap_err.to_string().contains("Invalid format"));
    }

    #[test]
    fn test_composition_error() {
        let err = CompositionError::IncompatibleTypes {
            type_a: "String".to_string(),
            type_b: "Number".to_string(),
        };
        assert!(err.to_string().contains("String"));
        assert!(err.to_string().contains("Number"));
    }

    #[test]
    fn test_circular_dependency() {
        let err = DecompositionError::CircularDependency("A -> B -> C -> A".to_string());
        assert!(err.to_string().contains("Circular dependency"));
    }

    #[test]
    fn test_invalid_dependency() {
        let err = DecompositionError::InvalidDependency {
            subtask: "task_b".to_string(),
            dependency: "task_unknown".to_string(),
        };
        assert!(err.to_string().contains("task_b"));
        assert!(err.to_string().contains("task_unknown"));
    }
}
