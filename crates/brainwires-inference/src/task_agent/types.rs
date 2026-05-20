//! Types for TaskAgent: configuration, status, result, and failure categories.

use std::path::PathBuf;

use crate::validation_loop::ValidationConfig;
use brainwires_agent::execution_graph::{ExecutionGraph, RunTelemetry};

/// Tool names whose results originate from external / untrusted sources and
/// must be sanitised before injection into the conversation history.
pub(super) const EXTERNAL_CONTENT_TOOLS: &[&str] = &[
    "fetch_url",
    "web_fetch",
    "web_search",
    "context_recall",
    "semantic_search",
];

pub(super) const DEFAULT_LOOP_DETECTION_WINDOW: usize = 5;
pub(super) const DEFAULT_MAX_ITERATIONS: u32 = 100;

/// Configuration for stuck-agent (loop) detection.
#[derive(Debug, Clone)]
pub struct LoopDetectionConfig {
    /// Consecutive identical tool-name calls that trigger abort. Default: 5.
    pub window_size: usize,
    /// Whether loop detection is active. Default: true.
    pub enabled: bool,
}

impl Default for LoopDetectionConfig {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_LOOP_DETECTION_WINDOW,
            enabled: true,
        }
    }
}

/// Runtime status of a task agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskAgentStatus {
    /// Agent is idle, waiting to be started.
    Idle,
    /// Agent is actively working on something.
    Working(String),
    /// Agent is blocked waiting for a file lock.
    WaitingForLock(String),
    /// Agent execution is paused.
    Paused(String),
    /// Agent is replanning after detecting goal drift or failure.
    Replanning(String),
    /// Agent completed the task successfully.
    Completed(String),
    /// Agent failed to complete the task.
    Failed(String),
}

impl std::fmt::Display for TaskAgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskAgentStatus::Idle => write!(f, "Idle"),
            TaskAgentStatus::Working(desc) => write!(f, "Working: {}", desc),
            TaskAgentStatus::WaitingForLock(path) => write!(f, "Waiting for lock: {}", path),
            TaskAgentStatus::Paused(reason) => write!(f, "Paused: {}", reason),
            TaskAgentStatus::Replanning(reason) => write!(f, "Replanning: {}", reason),
            TaskAgentStatus::Completed(summary) => write!(f, "Completed: {}", summary),
            TaskAgentStatus::Failed(error) => write!(f, "Failed: {}", error),
        }
    }
}

/// Classification of why an agent run failed.
///
/// Always `Some` when [`TaskAgentResult::success`] is `false`, always `None`
/// on success.  Enables trend queries and dashboards over failure modes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FailureCategory {
    /// Agent exhausted the allowed iteration count.
    IterationLimitExceeded,
    /// Cumulative token usage exceeded [`TaskAgentConfig::max_total_tokens`].
    TokenBudgetExceeded,
    /// Cumulative cost exceeded [`TaskAgentConfig::max_cost_usd`].
    CostBudgetExceeded,
    /// Wall-clock timeout exceeded [`TaskAgentConfig::timeout_secs`].
    WallClockTimeout,
    /// Loop detection fired — agent was calling the same tool repeatedly.
    LoopDetected,
    /// Replan cycle count exceeded [`TaskAgentConfig::max_replan_attempts`].
    MaxReplanAttemptsExceeded,
    /// File scope whitelist violation (reserved for future hard-stop policy).
    FileScopeViolation,
    /// Validation checks failed and could not be resolved within the
    /// iteration budget.
    ValidationFailed,
    /// An unexpected tool execution error caused abort.
    ToolExecutionError,
    /// Failure cause could not be determined.
    Unknown,
    /// Plan budget check failed before execution started — task was rejected
    /// before any side effects occurred.
    PlanBudgetExceeded,
}

/// Result of a completed task agent execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskAgentResult {
    /// The agent's unique ID.
    pub agent_id: String,
    /// The task ID that was executed.
    pub task_id: String,
    /// Whether the task completed successfully.
    pub success: bool,
    /// Completion summary or error description.
    pub summary: String,
    /// Number of provider call iterations used.
    pub iterations: u32,
    /// Number of replan cycles during execution.
    pub replan_count: u32,
    /// True when any budget ceiling caused the stop.
    pub budget_exhausted: bool,
    /// Last meaningful assistant message when stopped early, if any.
    pub partial_output: Option<String>,
    /// Cumulative tokens consumed across all provider calls.
    pub total_tokens_used: u64,
    /// Estimated cost in USD ($0.000003/token conservative estimate).
    pub total_cost_usd: f64,
    /// True when wall-clock timeout caused the stop.
    pub timed_out: bool,
    /// Why the agent failed. `None` on success, always `Some` on failure.
    pub failure_category: Option<FailureCategory>,
    /// Full execution trace (DAG of provider-call steps + tool call records).
    pub execution_graph: ExecutionGraph,
    /// Structured telemetry summary derived from the execution graph.
    pub telemetry: RunTelemetry,
    /// Pre-execution plan produced before the task loop started, if
    /// [`TaskAgentConfig::plan_budget`] was configured.  `None` when planning
    /// was not requested or when the plan could not be parsed.
    pub pre_execution_plan: Option<brainwires_core::SerializablePlan>,
}

/// Configuration for a task agent.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TaskAgentConfig {
    /// Maximum provider call iterations before the agent is forced to fail.
    ///
    /// Default: 100 (high default to avoid artificial limits on complex tasks).
    pub max_iterations: u32,

    /// Override the system prompt.
    ///
    /// When `None`, [`crate::system_prompts::reasoning_agent_prompt`] is used.
    pub system_prompt: Option<String>,

    /// Temperature for AI calls (0.0 – 1.0).
    pub temperature: f32,

    /// Maximum tokens for a single AI response.
    pub max_tokens: u32,

    /// Quality checks to run before accepting completion.
    ///
    /// Set to `None` to disable validation entirely (useful in tests).
    pub validation_config: Option<ValidationConfig>,

    /// Loop detection settings. `None` disables. Default: 5-call window, enabled.
    pub loop_detection: Option<LoopDetectionConfig>,

    /// Inject goal-reminder every N iterations. `None` disables. Default: Some(10).
    pub goal_revalidation_interval: Option<u32>,

    /// Abort after this many REPLAN cycles. Default: 3.
    pub max_replan_attempts: u32,

    /// Abort when cumulative tokens reach this ceiling. Default: None.
    pub max_total_tokens: Option<u64>,

    /// Abort when cumulative cost (USD) reaches this ceiling. Default: None.
    pub max_cost_usd: Option<f64>,

    /// Wall-clock timeout for the entire execute() call, in seconds. Default: None.
    pub timeout_secs: Option<u64>,

    /// Per-agent file scope whitelist.
    ///
    /// When `Some`, the agent receives a scope-violation error for any file
    /// operation targeting a path that is not prefixed by at least one entry
    /// in this list.  When `None`, file access is unrestricted.
    ///
    /// Uses [`Path::starts_with`](std::path::Path::starts_with) for prefix matching, which is
    /// component-aware: `"/src"` allows `"/src/main.rs"` but denies
    /// `"/src_extra/file.txt"`.
    pub allowed_files: Option<Vec<PathBuf>>,

    /// Optional pre-execution budget check.
    ///
    /// When `Some`, the agent asks the provider to produce a structured JSON
    /// plan before starting execution. The plan is validated against the budget
    /// constraints; if any constraint is exceeded the run fails immediately
    /// with [`FailureCategory::PlanBudgetExceeded`] before any file or tool
    /// side-effects occur.
    ///
    /// Set to `None` (the default) to skip the planning phase entirely.
    pub plan_budget: Option<brainwires_core::PlanBudget>,

    /// Context budget in tokens.
    ///
    /// When the estimated conversation token count exceeds this value,
    /// the [`on_context_pressure`][crate::agent_hooks::AgentLifecycleHooks::on_context_pressure]
    /// hook is called so the consumer can summarize or evict messages.
    ///
    /// Only effective when lifecycle hooks are set on the [`AgentContext`](crate::context::AgentContext).
    /// Default: `None` (no context pressure callbacks).
    pub context_budget_tokens: Option<u64>,

    /// Optional analytics collector.
    ///
    /// When `Some`, emits [`brainwires_telemetry::AnalyticsEvent::AgentRun`] after
    /// each run completes (success or failure). Feature-gated by `analytics`.
    #[cfg(feature = "telemetry")]
    pub analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,

    /// Optional billing hook.
    ///
    /// When `Some`, emits a [`brainwires_telemetry::UsageEvent`] at every
    /// cost-accrual point during the run: once per provider call (tokens) and
    /// once per tool call. Feature-gated by `billing`.
    ///
    /// The hook is fail-open — errors are logged but never abort the agent run.
    #[cfg(feature = "telemetry")]
    pub billing_hook: Option<crate::task_agent::BillingHookRef>,
}

impl Default for TaskAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096,
            validation_config: Some(ValidationConfig::default()),
            loop_detection: Some(LoopDetectionConfig::default()),
            goal_revalidation_interval: Some(10),
            max_replan_attempts: 3,
            max_total_tokens: None,
            max_cost_usd: None,
            timeout_secs: None,
            allowed_files: None,
            plan_budget: None,
            context_budget_tokens: None,
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
            #[cfg(feature = "telemetry")]
            billing_hook: None,
        }
    }
}
