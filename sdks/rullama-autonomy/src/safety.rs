//! Safety mechanisms for autonomous operations.
//!
//! Provides circuit breakers, budget tracking, approval policies, and
//! dead man's switches to prevent runaway autonomous operations.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::{SafetyConfig, SelfImprovementConfig};

// ── Safety stop reasons ─────────────────────────────────────────────────────

/// Reason an autonomous operation was stopped by a safety mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyStop {
    /// Budget limit was exceeded.
    BudgetExceeded(
        /// Total cost incurred.
        f64,
    ),
    /// Maximum cycle count reached.
    CycleLimitReached(
        /// Number of cycles completed.
        u32,
    ),
    /// Circuit breaker tripped after consecutive failures.
    CircuitBreakerTripped(
        /// Number of consecutive failures.
        u32,
    ),
    /// Total diff line limit exceeded.
    DiffLimitExceeded(
        /// Number of diff lines.
        u32,
    ),
    /// Dead man's switch heartbeat timed out.
    HeartbeatTimeout,
    /// Daily operation limit reached.
    DailyLimitReached(
        /// Number of daily operations completed.
        u32,
    ),
    /// Operation was explicitly rejected.
    OperationRejected(
        /// Rejection reason.
        String,
    ),
}

impl fmt::Display for SafetyStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetyStop::BudgetExceeded(cost) => write!(f, "Budget exceeded: ${cost:.2}"),
            SafetyStop::CycleLimitReached(cycles) => write!(f, "Cycle limit reached: {cycles}"),
            SafetyStop::CircuitBreakerTripped(failures) => {
                write!(
                    f,
                    "Circuit breaker tripped after {failures} consecutive failures"
                )
            }
            SafetyStop::DiffLimitExceeded(lines) => {
                write!(f, "Total diff limit exceeded: {lines} lines")
            }
            SafetyStop::HeartbeatTimeout => write!(f, "Dead man's switch: heartbeat timeout"),
            SafetyStop::DailyLimitReached(count) => {
                write!(f, "Daily operation limit reached: {count}")
            }
            SafetyStop::OperationRejected(reason) => {
                write!(f, "Operation rejected: {reason}")
            }
        }
    }
}

// ── Autonomous operation types ──────────────────────────────────────────────

/// Describes an autonomous operation that may require approval.
#[derive(Debug, Clone)]
pub enum AutonomousOperation {
    /// Start a self-improvement session.
    StartImprovement {
        /// Name of the improvement strategy.
        strategy: String,
        /// Estimated cost in USD.
        estimated_cost: f64,
    },
    /// Commit code changes.
    CommitChanges {
        /// Number of changed lines.
        diff_lines: u32,
        /// List of modified files.
        files: Vec<String>,
    },
    /// Create a pull request.
    CreatePullRequest {
        /// Branch name for the PR.
        branch: String,
        /// Title of the pull request.
        title: String,
    },
    /// Merge a pull request.
    MergePullRequest {
        /// Pull request identifier.
        pr_id: String,
        /// Confidence score for the merge.
        confidence: f64,
    },
    /// Spawn a new agent.
    SpawnAgent {
        /// Description of the agent's task.
        description: String,
    },
    /// Restart a running agent.
    RestartAgent {
        /// Identifier of the agent to restart.
        agent_id: String,
        /// Reason for the restart.
        reason: String,
    },
    /// Start a model training job.
    StartTrainingJob {
        /// Training provider name.
        provider: String,
        /// Number of training examples.
        dataset_size: usize,
    },
    /// Access a GPIO pin on the host system.
    GpioAccess {
        /// GPIO chip number.
        chip: u32,
        /// GPIO line number.
        line: u32,
        /// Direction (input/output).
        direction: String,
        /// Agent requesting access.
        agent_id: String,
    },
    /// Attempt crash recovery for a self-improvement session.
    CrashRecovery {
        /// Unique crash identifier.
        crash_id: String,
        /// Strategy to apply for recovery.
        fix_strategy: String,
    },
    /// Execute a scheduled autonomous task.
    ScheduledTask {
        /// Name of the scheduled task.
        name: String,
        /// Type of task being scheduled.
        task_type: String,
    },
    /// React to a file system change.
    ReactToFileChange {
        /// Path of the changed file.
        path: String,
        /// Type of change detected.
        event_type: String,
    },
    /// Manage a system service (systemd, docker, process).
    ManageService {
        /// Name of the service.
        name: String,
        /// Operation to perform (start, stop, restart, etc.).
        operation: String,
        /// Type of service (systemd, docker, process).
        service_type: String,
    },
}

impl fmt::Display for AutonomousOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartImprovement { strategy, .. } => write!(f, "start improvement: {strategy}"),
            Self::CommitChanges { diff_lines, .. } => {
                write!(f, "commit {diff_lines} changed lines")
            }
            Self::CreatePullRequest { title, .. } => write!(f, "create PR: {title}"),
            Self::MergePullRequest { pr_id, .. } => write!(f, "merge PR: {pr_id}"),
            Self::SpawnAgent { description } => write!(f, "spawn agent: {description}"),
            Self::RestartAgent { agent_id, .. } => write!(f, "restart agent: {agent_id}"),
            Self::StartTrainingJob { provider, .. } => write!(f, "training job: {provider}"),
            Self::GpioAccess {
                chip,
                line,
                direction,
                ..
            } => write!(f, "GPIO access: chip{chip}/line{line} ({direction})"),
            Self::CrashRecovery {
                crash_id,
                fix_strategy,
            } => {
                write!(f, "crash recovery {crash_id}: {fix_strategy}")
            }
            Self::ScheduledTask { name, task_type } => {
                write!(f, "scheduled task: {name} ({task_type})")
            }
            Self::ReactToFileChange { path, event_type } => {
                write!(f, "react to {event_type}: {path}")
            }
            Self::ManageService {
                name,
                operation,
                service_type,
            } => write!(f, "{operation} {service_type} service: {name}"),
        }
    }
}

// ── Approval policy ─────────────────────────────────────────────────────────

/// Gate for autonomous operations — implementations decide whether to allow
/// or reject an operation before it executes.
///
/// Used by the safety guard, pipeline, and supervisor to enforce approval
/// requirements on sensitive operations (PR creation, merging, agent restarts).
#[async_trait]
pub trait ApprovalPolicy: Send + Sync {
    /// Check whether the given operation is allowed.
    async fn check(&self, op: &AutonomousOperation) -> Result<(), SafetyStop>;
}

/// Default policy that approves everything (for testing / trusted environments).
pub struct AlwaysApprove;

#[async_trait]
impl ApprovalPolicy for AlwaysApprove {
    async fn check(&self, _op: &AutonomousOperation) -> Result<(), SafetyStop> {
        Ok(())
    }
}

/// Policy that rejects all autonomous operations (require manual execution).
pub struct AlwaysReject;

#[async_trait]
impl ApprovalPolicy for AlwaysReject {
    async fn check(&self, op: &AutonomousOperation) -> Result<(), SafetyStop> {
        Err(SafetyStop::OperationRejected(format!(
            "Manual approval required for: {op}"
        )))
    }
}

// ── Circuit breaker ─────────────────────────────────────────────────────────

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Normal operation — failures are counted.
    Closed,
    /// Tripped — all operations rejected until cooldown expires.
    Open,
    /// Cooldown expired — allow one probe operation.
    HalfOpen,
}

/// Circuit breaker that trips after consecutive failures and resets after a cooldown period.
///
/// Follows the closed -> open -> half-open pattern: closed allows operations,
/// open rejects them, and half-open permits a single probe after cooldown.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    consecutive_failures: u32,
    threshold: u32,
    cooldown_secs: u64,
    last_failure: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given failure threshold and cooldown.
    pub fn new(threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            consecutive_failures: 0,
            threshold,
            cooldown_secs,
            last_failure: None,
        }
    }

    /// Check if the circuit breaker allows operations.
    pub fn check(&mut self) -> Result<(), SafetyStop> {
        match self.state {
            CircuitBreakerState::Closed => Ok(()),
            CircuitBreakerState::Open => {
                // Check if cooldown has expired
                if let Some(last) = self.last_failure
                    && last.elapsed().as_secs() >= self.cooldown_secs
                {
                    self.state = CircuitBreakerState::HalfOpen;
                    tracing::info!("Circuit breaker entering half-open state");
                    return Ok(());
                }
                Err(SafetyStop::CircuitBreakerTripped(self.consecutive_failures))
            }
            CircuitBreakerState::HalfOpen => {
                // Allow one probe operation
                Ok(())
            }
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = CircuitBreakerState::Closed;
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());

        if self.consecutive_failures >= self.threshold {
            self.state = CircuitBreakerState::Open;
            tracing::warn!(
                "Circuit breaker tripped after {} consecutive failures",
                self.consecutive_failures
            );
        }
    }

    /// Return the current circuit breaker state.
    pub fn state(&self) -> CircuitBreakerState {
        self.state
    }

    /// Return the number of consecutive failures recorded.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }
}

// ── Budget tracker ──────────────────────────────────────────────────────────

/// Tracks spending and enforces budget limits.
///
/// Enforces three constraints: total cumulative cost, per-operation cost ceiling,
/// and a daily operation count limit that resets at midnight UTC.
#[derive(Debug)]
pub struct BudgetTracker {
    total_cost: f64,
    max_total: f64,
    max_per_op: f64,
    daily_operations: u32,
    max_daily_operations: u32,
    day_start: DateTime<Utc>,
}

impl BudgetTracker {
    /// Create a new budget tracker with the given limits.
    pub fn new(max_total: f64, max_per_op: f64, max_daily_operations: u32) -> Self {
        Self {
            total_cost: 0.0,
            max_total,
            max_per_op,
            daily_operations: 0,
            max_daily_operations,
            day_start: Utc::now(),
        }
    }

    /// Check if the budget allows an operation with the estimated cost.
    pub fn check(&mut self, estimated_cost: f64) -> Result<(), SafetyStop> {
        self.maybe_reset_daily();

        if self.total_cost + estimated_cost > self.max_total {
            return Err(SafetyStop::BudgetExceeded(self.total_cost));
        }
        if estimated_cost > self.max_per_op {
            return Err(SafetyStop::BudgetExceeded(estimated_cost));
        }
        if self.daily_operations >= self.max_daily_operations {
            return Err(SafetyStop::DailyLimitReached(self.daily_operations));
        }
        Ok(())
    }

    /// Record an operation's actual cost.
    pub fn record_cost(&mut self, cost: f64) {
        self.total_cost += cost;
        self.daily_operations += 1;
    }

    /// Return the total cost incurred so far.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Return the number of operations performed today.
    pub fn daily_operations(&self) -> u32 {
        self.daily_operations
    }

    fn maybe_reset_daily(&mut self) {
        let now = Utc::now();
        if now.date_naive() != self.day_start.date_naive() {
            self.daily_operations = 0;
            self.day_start = now;
        }
    }
}

// ── Dead man's switch ───────────────────────────────────────────────────────

/// Dead man's switch that trips if no heartbeat is received within the timeout.
#[derive(Debug)]
pub struct DeadManSwitch {
    last_heartbeat: Instant,
    timeout_secs: u64,
}

impl DeadManSwitch {
    /// Create a new dead man's switch with the given timeout in seconds.
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            last_heartbeat: Instant::now(),
            timeout_secs,
        }
    }

    /// Send a heartbeat to keep the switch alive.
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Check if the switch has timed out.
    pub fn check(&self) -> Result<(), SafetyStop> {
        if self.last_heartbeat.elapsed().as_secs() >= self.timeout_secs {
            Err(SafetyStop::HeartbeatTimeout)
        } else {
            Ok(())
        }
    }

    /// Return the number of seconds since the last heartbeat.
    pub fn elapsed_secs(&self) -> u64 {
        self.last_heartbeat.elapsed().as_secs()
    }
}

// ── SafetyGuard (composite) ─────────────────────────────────────────────────

/// Composite safety guard combining circuit breaker, budget tracker, diff limits,
/// and dead man's switch.
pub struct SafetyGuard {
    circuit_breaker: CircuitBreaker,
    budget: BudgetTracker,
    dead_man: DeadManSwitch,
    total_diff_lines: u32,
    max_total_diff: u32,
    cycles_completed: u32,
    max_cycles: u32,
    approval_policy: Arc<dyn ApprovalPolicy>,
}

impl SafetyGuard {
    /// Create from a full SafetyConfig.
    pub fn from_config(config: &SafetyConfig, max_cycles: u32) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_cooldown_secs,
            ),
            budget: BudgetTracker::new(
                config.max_total_cost,
                config.max_per_operation_cost,
                config.max_daily_operations,
            ),
            dead_man: DeadManSwitch::new(config.heartbeat_timeout_secs),
            total_diff_lines: 0,
            max_total_diff: config.max_total_diff,
            cycles_completed: 0,
            max_cycles,
            approval_policy: Arc::new(AlwaysApprove),
        }
    }

    /// Create from the simpler SelfImprovementConfig (backwards-compatible).
    pub fn new(config: &SelfImprovementConfig) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(config.circuit_breaker_threshold, 300),
            budget: BudgetTracker::new(config.max_budget, config.max_budget, 1000),
            dead_man: DeadManSwitch::new(1800),
            total_diff_lines: 0,
            max_total_diff: config.max_total_diff,
            cycles_completed: 0,
            max_cycles: config.max_cycles,
            approval_policy: Arc::new(AlwaysApprove),
        }
    }

    /// Set the approval policy.
    pub fn with_approval_policy(mut self, policy: Arc<dyn ApprovalPolicy>) -> Self {
        self.approval_policy = policy;
        self
    }

    /// Check all safety constraints before continuing.
    pub fn check_can_continue(&mut self) -> Result<(), SafetyStop> {
        self.circuit_breaker.check()?;
        self.budget.check(0.0)?;
        self.dead_man.check()?;

        if self.cycles_completed >= self.max_cycles {
            return Err(SafetyStop::CycleLimitReached(self.cycles_completed));
        }
        if self.total_diff_lines >= self.max_total_diff {
            return Err(SafetyStop::DiffLimitExceeded(self.total_diff_lines));
        }
        Ok(())
    }

    /// Check approval policy for an operation.
    pub async fn check_approval(&self, op: &AutonomousOperation) -> Result<(), SafetyStop> {
        self.approval_policy.check(op).await
    }

    /// Record a successful cycle.
    pub fn record_success(&mut self, diff_lines: u32) {
        self.circuit_breaker.record_success();
        self.total_diff_lines += diff_lines;
        self.cycles_completed += 1;
    }

    /// Record a failed cycle.
    pub fn record_failure(&mut self) {
        self.circuit_breaker.record_failure();
        self.cycles_completed += 1;
    }

    /// Record cost for budget tracking.
    pub fn record_cost(&mut self, cost: f64) {
        self.budget.record_cost(cost);
    }

    /// Send heartbeat to dead man's switch.
    pub fn heartbeat(&mut self) {
        self.dead_man.heartbeat();
    }

    /// Return the number of cycles completed.
    pub fn cycles_completed(&self) -> u32 {
        self.cycles_completed
    }

    /// Return the total cost tracked by the budget.
    pub fn total_cost(&self) -> f64 {
        self.budget.total_cost()
    }

    /// Return the total number of diff lines accumulated.
    pub fn total_diff_lines(&self) -> u32 {
        self.total_diff_lines
    }

    /// Return the current circuit breaker state.
    pub fn circuit_breaker_state(&self) -> CircuitBreakerState {
        self.circuit_breaker.state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_trips() {
        let mut cb = CircuitBreaker::new(3, 60);
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(cb.check().is_err());
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_budget_tracker() {
        let mut bt = BudgetTracker::new(10.0, 5.0, 100);
        assert!(bt.check(3.0).is_ok());
        bt.record_cost(3.0);
        assert!(bt.check(3.0).is_ok());
        bt.record_cost(3.0);
        // Now at $6, trying to add $5 would exceed $10
        assert!(bt.check(5.0).is_err());
    }

    #[test]
    fn test_dead_man_switch() {
        let dms = DeadManSwitch::new(1800);
        assert!(dms.check().is_ok());
    }

    #[test]
    fn test_safety_guard_cycle_limit() {
        let config = SelfImprovementConfig {
            max_cycles: 2,
            ..Default::default()
        };
        let mut guard = SafetyGuard::new(&config);
        assert!(guard.check_can_continue().is_ok());
        guard.record_success(10);
        assert!(guard.check_can_continue().is_ok());
        guard.record_success(10);
        assert!(guard.check_can_continue().is_err());
    }

    // ── Approval policy tests ───────────────────────────────────────────

    fn commit_op() -> AutonomousOperation {
        AutonomousOperation::CommitChanges {
            diff_lines: 42,
            files: vec!["src/lib.rs".to_string()],
        }
    }

    fn merge_op() -> AutonomousOperation {
        AutonomousOperation::MergePullRequest {
            pr_id: "PR-7".to_string(),
            confidence: 0.92,
        }
    }

    #[tokio::test]
    async fn approval_required_blocks_without_approval() {
        // Build a guard whose approval policy is "always reject" — the analogue
        // of "require explicit approval". Any op should be blocked.
        let config = SelfImprovementConfig::default();
        let guard = SafetyGuard::new(&config).with_approval_policy(Arc::new(AlwaysReject));
        let result = guard.check_approval(&commit_op()).await;
        assert!(
            result.is_err(),
            "AlwaysReject should block an op without explicit approval"
        );
        match result.unwrap_err() {
            SafetyStop::OperationRejected(msg) => {
                assert!(
                    msg.contains("Manual approval"),
                    "rejection message should mention manual approval, got: {msg}"
                );
            }
            other => panic!("expected OperationRejected, got {other:?}"),
        }
    }

    /// An approval policy that grants exactly one pre-approved op (by display label)
    /// and rejects everything else — exercises the "approve op A, try op B" case.
    struct PreApprovedOnly {
        approved_label: String,
    }

    #[async_trait]
    impl ApprovalPolicy for PreApprovedOnly {
        async fn check(&self, op: &AutonomousOperation) -> Result<(), SafetyStop> {
            if op.to_string() == self.approved_label {
                Ok(())
            } else {
                Err(SafetyStop::OperationRejected(format!(
                    "not pre-approved: {op}"
                )))
            }
        }
    }

    #[tokio::test]
    async fn approval_denied_for_non_preapproved_op() {
        let config = SelfImprovementConfig::default();
        let pre_approved = commit_op();
        let guard = SafetyGuard::new(&config).with_approval_policy(Arc::new(PreApprovedOnly {
            approved_label: pre_approved.to_string(),
        }));

        // Pre-approved op passes.
        assert!(
            guard.check_approval(&pre_approved).await.is_ok(),
            "pre-approved op should be allowed"
        );
        // Different op is rejected.
        assert!(
            guard.check_approval(&merge_op()).await.is_err(),
            "non-preapproved op should be rejected"
        );
    }

    #[tokio::test]
    async fn automatic_policy_allows_safe_ops() {
        // AlwaysApprove is the default "auto-approve safe" policy.
        let config = SelfImprovementConfig::default();
        let guard = SafetyGuard::new(&config);
        assert!(
            guard.check_approval(&commit_op()).await.is_ok(),
            "AlwaysApprove should allow safe ops"
        );
        assert!(
            guard.check_approval(&merge_op()).await.is_ok(),
            "AlwaysApprove should allow all ops"
        );
    }

    // ── Operation guard behavior ────────────────────────────────────────

    #[test]
    fn circuit_breaker_blocks_after_trip() {
        // A tripped circuit breaker causes check_can_continue to fail until cooldown.
        let config = SelfImprovementConfig {
            max_cycles: 100,
            circuit_breaker_threshold: 2,
            ..Default::default()
        };
        let mut guard = SafetyGuard::new(&config);
        guard.record_failure();
        guard.record_failure();
        assert_eq!(
            guard.circuit_breaker_state(),
            CircuitBreakerState::Open,
            "breaker should trip after threshold failures"
        );
        assert!(matches!(
            guard.check_can_continue(),
            Err(SafetyStop::CircuitBreakerTripped(_))
        ));
    }

    #[test]
    fn diff_limit_blocks_once_exceeded() {
        let config = SelfImprovementConfig {
            max_cycles: 100,
            max_total_diff: 50,
            ..Default::default()
        };
        let mut guard = SafetyGuard::new(&config);
        guard.record_success(60);
        let result = guard.check_can_continue();
        assert!(
            matches!(result, Err(SafetyStop::DiffLimitExceeded(_))),
            "expected DiffLimitExceeded, got {result:?}"
        );
    }
}
