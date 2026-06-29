//! Configuration types for autonomous operations.

use serde::{Deserialize, Serialize};

/// Default dead man's switch heartbeat timeout in seconds (30 minutes).
const DEFAULT_HEARTBEAT_TIMEOUT_SECS: u64 = 1800;

/// Top-level configuration for the autonomy subsystem.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutonomyConfig {
    /// Self-improvement session configuration.
    #[serde(default)]
    pub self_improve: SelfImprovementConfig,
    /// Safety and budget limits.
    #[serde(default)]
    pub safety: SafetyConfig,
    /// Git workflow configuration.
    #[serde(default)]
    pub git_workflow: GitWorkflowConfig,
    /// Crash recovery configuration.
    #[serde(default)]
    pub crash_recovery: CrashRecoveryConfig,
    /// GPIO hardware access configuration.
    ///
    /// Only available with the `gpio` feature; uses
    /// [`brainwires_hardware::gpio::config::GpioConfig`] (which the GPIO
    /// pin manager / safety policy `from_config` expects).
    #[cfg(feature = "gpio")]
    #[serde(default)]
    pub gpio: GpioConfig,
}

/// Configuration for self-improvement sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfImprovementConfig {
    /// Maximum improvement cycles to run.
    pub max_cycles: u32,
    /// Maximum total cost in USD.
    pub max_budget: f64,
    /// If true, generate tasks but don't execute them.
    pub dry_run: bool,
    /// Enabled strategy names (empty = all).
    pub strategies: Vec<String>,
    /// Max iterations per agent task.
    pub agent_iterations: u32,
    /// Max diff lines per single task.
    pub max_diff_per_task: u32,
    /// Max total diff lines across entire session.
    pub max_total_diff: u32,
    /// Create PRs for committed changes.
    pub create_prs: bool,
    /// Git branch prefix for improvement branches.
    pub branch_prefix: String,
    /// Override model for agent tasks.
    pub model: Option<String>,
    /// Override provider.
    pub provider: Option<String>,
    /// Consecutive failures before circuit breaker trips.
    pub circuit_breaker_threshold: u32,
}

impl Default for SelfImprovementConfig {
    fn default() -> Self {
        Self {
            max_cycles: 10,
            max_budget: 10.0,
            dry_run: false,
            strategies: Vec::new(),
            agent_iterations: 25,
            max_diff_per_task: 200,
            max_total_diff: 1000,
            create_prs: false,
            branch_prefix: "self-improve/".to_string(),
            model: None,
            provider: None,
            circuit_breaker_threshold: 3,
        }
    }
}

impl SelfImprovementConfig {
    /// Check if a given strategy name is enabled (empty list = all enabled).
    pub fn is_strategy_enabled(&self, name: &str) -> bool {
        self.strategies.is_empty() || self.strategies.iter().any(|s| s == name)
    }
}

/// Per-strategy configuration passed to strategy task generators during scanning.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Path to the repository root.
    pub repo_path: String,
    /// Maximum tasks to generate per strategy.
    pub max_tasks_per_strategy: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            repo_path: ".".to_string(),
            max_tasks_per_strategy: 5,
        }
    }
}

/// Safety and budget configuration for autonomous operations.
///
/// Controls cost limits, operation quotas, circuit breaker behavior, and
/// file path restrictions that apply across all autonomous features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Maximum total cost in USD across all operations.
    pub max_total_cost: f64,
    /// Maximum cost per single operation.
    pub max_per_operation_cost: f64,
    /// Maximum daily operations.
    pub max_daily_operations: u32,
    /// Consecutive failure threshold for circuit breaker.
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker cooldown in seconds.
    pub circuit_breaker_cooldown_secs: u64,
    /// Max diff lines per task.
    pub max_diff_per_task: u32,
    /// Max total diff lines per session.
    pub max_total_diff: u32,
    /// Max concurrent agents.
    pub max_concurrent_agents: u32,
    /// Dead man's switch heartbeat timeout in seconds.
    pub heartbeat_timeout_secs: u64,
    /// Allowed path globs for file modifications.
    pub allowed_paths: Vec<String>,
    /// Forbidden path globs (takes precedence over allowed).
    pub forbidden_paths: Vec<String>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_total_cost: 50.0,
            max_per_operation_cost: 5.0,
            max_daily_operations: 100,
            circuit_breaker_threshold: 3,
            circuit_breaker_cooldown_secs: 300,
            max_diff_per_task: 200,
            max_total_diff: 1000,
            max_concurrent_agents: 5,
            heartbeat_timeout_secs: DEFAULT_HEARTBEAT_TIMEOUT_SECS,
            allowed_paths: Vec::new(),
            forbidden_paths: Vec::new(),
        }
    }
}

/// Git workflow pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitWorkflowConfig {
    /// Branch prefix for autonomous fix branches.
    pub branch_prefix: String,
    /// Whether to auto-merge PRs when policy allows.
    pub auto_merge: bool,
    /// Default merge method.
    pub merge_method: String,
    /// Minimum investigation confidence to proceed with fix.
    pub min_confidence: f64,
    /// Webhook server configuration.
    #[serde(default)]
    pub webhook: WebhookConfig,
}

impl Default for GitWorkflowConfig {
    fn default() -> Self {
        Self {
            branch_prefix: "autonomy/".to_string(),
            auto_merge: false,
            merge_method: "squash".to_string(),
            min_confidence: 0.7,
            webhook: WebhookConfig::default(),
        }
    }
}

/// Webhook server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Listen address.
    pub listen_addr: String,
    /// Listen port.
    pub port: u16,
    /// Webhook secret for HMAC verification.
    pub secret: Option<String>,
    /// Directory for webhook event logs.
    #[serde(default = "default_webhook_log_dir")]
    pub log_dir: String,
    /// Number of days to keep webhook logs.
    #[serde(default = "default_webhook_keep_days")]
    pub keep_days: u32,
    /// Per-repository webhook configuration.
    #[serde(default)]
    pub repos: std::collections::HashMap<String, WebhookRepoConfig>,
}

fn default_webhook_log_dir() -> String {
    dirs::home_dir()
        .map(|h| {
            h.join(".brainwires")
                .join("webhook-logs")
                .to_string_lossy()
                .to_string()
        })
        .unwrap_or_else(|| "/tmp/brainwires/webhook-logs".to_string())
}

fn default_webhook_keep_days() -> u32 {
    30
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 3000,
            secret: None,
            log_dir: default_webhook_log_dir(),
            keep_days: default_webhook_keep_days(),
            repos: std::collections::HashMap::new(),
        }
    }
}

/// Per-repository webhook configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookRepoConfig {
    /// Which events to handle (e.g., "issues", "push", "pull_request").
    #[serde(default)]
    pub events: Vec<String>,
    /// Whether to automatically investigate issues.
    #[serde(default)]
    pub auto_investigate: bool,
    /// Whether to automatically apply fixes.
    #[serde(default)]
    pub auto_fix: bool,
    /// Whether to automatically merge PRs when policy allows.
    #[serde(default)]
    pub auto_merge: bool,
    /// Only handle issues with these labels (empty = all).
    #[serde(default)]
    pub labels_filter: Vec<String>,
    /// Commands to run after processing an event.
    #[serde(default)]
    pub post_commands: Vec<CommandConfig>,
}

impl Default for WebhookRepoConfig {
    fn default() -> Self {
        Self {
            events: vec!["issues".to_string()],
            auto_investigate: true,
            auto_fix: false,
            auto_merge: false,
            labels_filter: Vec::new(),
            post_commands: Vec::new(),
        }
    }
}

/// Command to execute with variable interpolation support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandConfig {
    /// Command to run.
    pub cmd: String,
    /// Arguments for the command.
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory (supports variables like `${REPO_NAME}`).
    #[serde(default)]
    pub working_dir: Option<String>,
}

/// Crash recovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashRecoveryConfig {
    /// Maximum fix attempts before giving up.
    pub max_fix_attempts: u32,
    /// Path to crash recovery state file.
    pub state_file: String,
    /// Whether crash recovery is enabled.
    pub enabled: bool,
}

impl Default for CrashRecoveryConfig {
    fn default() -> Self {
        Self {
            max_fix_attempts: 3,
            state_file: dirs::home_dir()
                .map(|h| {
                    h.join(".brainwires")
                        .join("crash-recovery.json")
                        .to_string_lossy()
                        .to_string()
                })
                .unwrap_or_else(|| "/tmp/brainwires/crash-recovery.json".to_string()),
            enabled: true,
        }
    }
}

/// GPIO hardware access configuration — re-exported from
/// [`brainwires_hardware::gpio::config::GpioConfig`].
///
/// Previously `brainwires-autonomy` declared its own divergent
/// `GpioConfig` struct here, which (despite identical fields) was not
/// accepted by [`brainwires_hardware::gpio::GpioPinManager::from_config`]
/// or [`brainwires_hardware::gpio::GpioSafetyPolicy::from_config`]. The
/// re-export ensures `AutonomyConfig.gpio` is the same type the hardware
/// crate's pin manager and safety policy operate on.
#[cfg(feature = "gpio")]
pub use brainwires_hardware::gpio::config::GpioConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autonomy_config_default_succeeds() {
        let config = AutonomyConfig::default();
        // Verify field types are accessible
        let _si: &SelfImprovementConfig = &config.self_improve;
        let _safety: &SafetyConfig = &config.safety;
        let _git: &GitWorkflowConfig = &config.git_workflow;
    }

    #[test]
    fn self_improvement_config_default_has_sensible_values() {
        let config = SelfImprovementConfig::default();
        assert_eq!(config.max_cycles, 10);
        assert!((config.max_budget - 10.0).abs() < f64::EPSILON);
        assert!(!config.dry_run);
        assert!(config.strategies.is_empty());
        assert_eq!(config.agent_iterations, 25);
        assert_eq!(config.circuit_breaker_threshold, 3);
        assert_eq!(config.branch_prefix, "self-improve/");
        assert!(config.model.is_none());
        assert!(config.provider.is_none());
    }

    #[test]
    fn safety_config_default_has_sensible_values() {
        let config = SafetyConfig::default();
        assert!((config.max_total_cost - 50.0).abs() < f64::EPSILON);
        assert!((config.max_per_operation_cost - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.max_daily_operations, 100);
        assert_eq!(config.circuit_breaker_threshold, 3);
        assert_eq!(config.circuit_breaker_cooldown_secs, 300);
        assert_eq!(config.max_concurrent_agents, 5);
        assert_eq!(config.heartbeat_timeout_secs, 1800);
        assert!(config.allowed_paths.is_empty());
        assert!(config.forbidden_paths.is_empty());
    }

    #[test]
    fn git_workflow_config_default_has_sensible_values() {
        let config = GitWorkflowConfig::default();
        assert_eq!(config.branch_prefix, "autonomy/");
        assert!(!config.auto_merge);
        assert_eq!(config.merge_method, "squash");
        assert!((config.min_confidence - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.webhook.port, 3000);
    }

    #[test]
    fn serde_roundtrip_autonomy_config() {
        let config = AutonomyConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: AutonomyConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            deserialized.self_improve.max_cycles,
            config.self_improve.max_cycles
        );
        assert_eq!(
            deserialized.safety.max_total_cost,
            config.safety.max_total_cost
        );
        assert_eq!(
            deserialized.git_workflow.branch_prefix,
            config.git_workflow.branch_prefix
        );
    }

    #[test]
    fn is_strategy_enabled_empty_list_enables_all() {
        let config = SelfImprovementConfig::default();
        assert!(config.is_strategy_enabled("clippy"));
        assert!(config.is_strategy_enabled("anything"));
    }

    #[test]
    fn is_strategy_enabled_specific_list() {
        let config = SelfImprovementConfig {
            strategies: vec!["clippy".to_string(), "todo".to_string()],
            ..Default::default()
        };
        assert!(config.is_strategy_enabled("clippy"));
        assert!(config.is_strategy_enabled("todo"));
        assert!(!config.is_strategy_enabled("dead_code"));
    }
}
