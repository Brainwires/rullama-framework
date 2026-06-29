//! Error types for the autonomy crate.

use thiserror::Error;

/// Top-level error type for autonomous operations.
#[derive(Error, Debug)]
pub enum AutonomyError {
    /// Safety constraint triggered a stop.
    #[error("Safety stop: {0}")]
    SafetyStop(String),

    /// Budget limit exceeded.
    #[error("Budget exceeded: ${0:.2}")]
    BudgetExceeded(f64),

    /// Circuit breaker tripped after consecutive failures.
    #[error("Circuit breaker tripped after {0} consecutive failures")]
    CircuitBreakerTripped(u32),

    /// Diff size limit exceeded.
    #[error("Diff limit exceeded: {0} lines")]
    DiffLimitExceeded(u32),

    /// Maximum cycle count reached.
    #[error("Cycle limit reached: {0}")]
    CycleLimitReached(u32),

    /// Git operation error.
    #[error("Git error: {0}")]
    GitError(String),

    /// Forge (GitHub/GitLab) operation error.
    #[error("Forge error: {0}")]
    ForgeError(String),

    /// Webhook delivery or parsing error.
    #[error("Webhook error: {0}")]
    WebhookError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Agent execution error.
    #[error("Agent error: {0}")]
    AgentError(String),

    /// Investigation/analysis error.
    #[error("Investigation error: {0}")]
    InvestigationError(String),

    /// Merge policy violation.
    #[error("Merge policy error: {0}")]
    MergePolicyError(String),

    /// Crash recovery error.
    #[error("Crash recovery error: {0}")]
    CrashRecoveryError(String),

    /// GPIO hardware access error.
    #[error("GPIO error: {0}")]
    GpioError(String),

    /// Scheduler error.
    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    /// File system reactor error.
    #[error("Reactor error: {0}")]
    ReactorError(String),

    /// System service management error.
    #[error("Service error: {0}")]
    ServiceError(String),

    /// Other unclassified error.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Convenience result alias.
pub type AutonomyResult<T> = Result<T, AutonomyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_safety_stop() {
        let err = AutonomyError::SafetyStop("test reason".to_string());
        assert_eq!(err.to_string(), "Safety stop: test reason");
    }

    #[test]
    fn display_budget_exceeded() {
        let err = AutonomyError::BudgetExceeded(12.5);
        assert_eq!(err.to_string(), "Budget exceeded: $12.50");
    }

    #[test]
    fn display_circuit_breaker_tripped() {
        let err = AutonomyError::CircuitBreakerTripped(5);
        assert_eq!(
            err.to_string(),
            "Circuit breaker tripped after 5 consecutive failures"
        );
    }

    #[test]
    fn display_diff_limit_exceeded() {
        let err = AutonomyError::DiffLimitExceeded(300);
        assert_eq!(err.to_string(), "Diff limit exceeded: 300 lines");
    }

    #[test]
    fn display_remaining_variants() {
        assert_eq!(
            AutonomyError::CycleLimitReached(10).to_string(),
            "Cycle limit reached: 10"
        );
        assert_eq!(
            AutonomyError::GitError("bad ref".to_string()).to_string(),
            "Git error: bad ref"
        );
        assert_eq!(
            AutonomyError::ForgeError("404".to_string()).to_string(),
            "Forge error: 404"
        );
        assert_eq!(
            AutonomyError::WebhookError("bad sig".to_string()).to_string(),
            "Webhook error: bad sig"
        );
        assert_eq!(
            AutonomyError::ConfigError("missing".to_string()).to_string(),
            "Configuration error: missing"
        );
        assert_eq!(
            AutonomyError::AgentError("timeout".to_string()).to_string(),
            "Agent error: timeout"
        );
        assert_eq!(
            AutonomyError::InvestigationError("parse".to_string()).to_string(),
            "Investigation error: parse"
        );
        assert_eq!(
            AutonomyError::MergePolicyError("blocked".to_string()).to_string(),
            "Merge policy error: blocked"
        );
    }

    #[test]
    fn display_new_error_variants() {
        assert_eq!(
            AutonomyError::CrashRecoveryError("meta-crash".to_string()).to_string(),
            "Crash recovery error: meta-crash"
        );
        assert_eq!(
            AutonomyError::GpioError("pin busy".to_string()).to_string(),
            "GPIO error: pin busy"
        );
        assert_eq!(
            AutonomyError::SchedulerError("bad cron".to_string()).to_string(),
            "Scheduler error: bad cron"
        );
        assert_eq!(
            AutonomyError::ReactorError("watch failed".to_string()).to_string(),
            "Reactor error: watch failed"
        );
        assert_eq!(
            AutonomyError::ServiceError("denied".to_string()).to_string(),
            "Service error: denied"
        );
    }
}
