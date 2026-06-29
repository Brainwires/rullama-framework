//! Agent health monitoring and degradation detection.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// --- HealthMonitorConfig default threshold constants ---

/// Default max milliseconds per iteration before flagging as slow (30 seconds).
const DEFAULT_SLOW_ITERATION_THRESHOLD_MS: u64 = 30_000;
/// Default max error rate (0.0-1.0) before flagging (30%).
const DEFAULT_ERROR_RATE_THRESHOLD: f64 = 0.3;
/// Default seconds without heartbeat before marking as unknown (2 minutes).
const DEFAULT_HEARTBEAT_TIMEOUT_SECS: u64 = 120;
/// Default seconds without activity before marking as stalled (5 minutes).
const DEFAULT_STALL_TIMEOUT_SECS: u64 = 300;

/// Health status of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Agent is operating normally.
    Healthy,
    /// Agent is experiencing minor issues but still functional.
    Degraded,
    /// Agent is unresponsive or critically failed.
    Unhealthy,
    /// Agent status is unknown (no recent heartbeat).
    Unknown,
}

/// Signal indicating a specific type of degradation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationSignal {
    /// Agent is taking longer than expected per iteration.
    SlowIterations {
        /// Average milliseconds per iteration.
        avg_ms: u64,
        /// Threshold in milliseconds.
        threshold_ms: u64,
    },
    /// Agent is consuming excessive tokens.
    HighTokenUsage {
        /// Tokens consumed so far.
        tokens_used: u64,
        /// Token budget.
        budget: u64,
    },
    /// Agent is stuck in a loop (repeating similar actions).
    LoopDetected {
        /// The action being repeated.
        repeated_action: String,
        /// Number of repetitions detected.
        count: u32,
    },
    /// Agent has not sent a heartbeat recently.
    MissedHeartbeat {
        /// Seconds since the last heartbeat.
        last_seen_secs_ago: u64,
    },
    /// Agent's error rate is above threshold.
    HighErrorRate {
        /// Current error rate.
        error_rate: f64,
        /// Maximum acceptable error rate.
        threshold: f64,
    },
    /// Agent is making no progress (no file changes, no tool calls).
    Stalled {
        /// Seconds since last activity.
        idle_secs: u64,
    },
}

/// Performance metrics for an agent.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Number of iterations completed.
    pub iterations: u32,
    /// Total tokens consumed.
    pub total_tokens: u64,
    /// Total cost in USD.
    pub total_cost: f64,
    /// Number of errors encountered.
    pub errors: u32,
    /// Number of tool calls made.
    pub tool_calls: u32,
    /// Number of files modified.
    pub files_modified: u32,
    /// Timestamp of the last activity.
    pub last_activity: Option<Instant>,
}

impl PerformanceMetrics {
    /// Compute the error rate as errors divided by tool calls.
    pub fn error_rate(&self) -> f64 {
        if self.tool_calls == 0 {
            0.0
        } else {
            self.errors as f64 / self.tool_calls as f64
        }
    }

    /// Compute the average number of tokens used per iteration.
    pub fn avg_tokens_per_iteration(&self) -> u64 {
        if self.iterations == 0 {
            0
        } else {
            self.total_tokens / self.iterations as u64
        }
    }
}

/// Monitors agent health by tracking performance metrics and detecting degradation.
///
/// Checks for missed heartbeats, high error rates, and stalled agents,
/// producing [`DegradationSignal`]s that the supervisor can act on.
pub struct HealthMonitor {
    agents: HashMap<String, AgentHealth>,
    config: HealthMonitorConfig,
}

struct AgentHealth {
    status: HealthStatus,
    metrics: PerformanceMetrics,
    signals: Vec<DegradationSignal>,
    last_heartbeat: Instant,
}

/// Configuration for health monitoring thresholds.
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Max milliseconds per iteration before flagging as slow.
    pub slow_iteration_threshold_ms: u64,
    /// Max error rate (0.0-1.0) before flagging.
    pub error_rate_threshold: f64,
    /// Seconds without heartbeat before marking as unknown.
    pub heartbeat_timeout_secs: u64,
    /// Seconds without activity before marking as stalled.
    pub stall_timeout_secs: u64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            slow_iteration_threshold_ms: DEFAULT_SLOW_ITERATION_THRESHOLD_MS,
            error_rate_threshold: DEFAULT_ERROR_RATE_THRESHOLD,
            heartbeat_timeout_secs: DEFAULT_HEARTBEAT_TIMEOUT_SECS,
            stall_timeout_secs: DEFAULT_STALL_TIMEOUT_SECS,
        }
    }
}

impl HealthMonitor {
    /// Create a new health monitor with the given configuration.
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
        }
    }

    /// Register an agent for monitoring.
    pub fn register(&mut self, agent_id: &str) {
        self.agents.insert(
            agent_id.to_string(),
            AgentHealth {
                status: HealthStatus::Healthy,
                metrics: PerformanceMetrics::default(),
                signals: Vec::new(),
                last_heartbeat: Instant::now(),
            },
        );
    }

    /// Record a heartbeat from an agent.
    pub fn heartbeat(&mut self, agent_id: &str) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.last_heartbeat = Instant::now();
            agent.metrics.last_activity = Some(Instant::now());
        }
    }

    /// Update metrics for an agent.
    pub fn update_metrics(&mut self, agent_id: &str, metrics: PerformanceMetrics) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.metrics = metrics;
        }
    }

    /// Evaluate all agents and return those with degradation signals.
    pub fn evaluate_all(&mut self) -> Vec<(String, HealthStatus, Vec<DegradationSignal>)> {
        let mut results = Vec::new();
        let config = self.config.clone();

        for (id, agent) in &mut self.agents {
            let mut signals = Vec::new();

            // Check heartbeat
            let secs_since_heartbeat = agent.last_heartbeat.elapsed().as_secs();
            if secs_since_heartbeat > config.heartbeat_timeout_secs {
                signals.push(DegradationSignal::MissedHeartbeat {
                    last_seen_secs_ago: secs_since_heartbeat,
                });
            }

            // Check error rate
            let error_rate = agent.metrics.error_rate();
            if error_rate > config.error_rate_threshold && agent.metrics.tool_calls > 5 {
                signals.push(DegradationSignal::HighErrorRate {
                    error_rate,
                    threshold: config.error_rate_threshold,
                });
            }

            // Check stall
            if let Some(last) = agent.metrics.last_activity {
                let idle_secs = last.elapsed().as_secs();
                if idle_secs > config.stall_timeout_secs {
                    signals.push(DegradationSignal::Stalled { idle_secs });
                }
            }

            // Determine status
            agent.status = if signals.is_empty() {
                HealthStatus::Healthy
            } else if signals
                .iter()
                .any(|s| matches!(s, DegradationSignal::MissedHeartbeat { .. }))
            {
                HealthStatus::Unknown
            } else {
                HealthStatus::Degraded
            };

            agent.signals = signals.clone();

            if agent.status != HealthStatus::Healthy {
                results.push((id.clone(), agent.status, signals));
            }
        }

        results
    }

    /// Get the current status of a specific agent.
    pub fn status(&self, agent_id: &str) -> HealthStatus {
        self.agents
            .get(agent_id)
            .map(|a| a.status)
            .unwrap_or(HealthStatus::Unknown)
    }

    /// Remove an agent from monitoring.
    pub fn unregister(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_rate_zero_when_no_tool_calls() {
        let m = PerformanceMetrics::default();
        assert!((m.error_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn error_rate_correct_value() {
        let m = PerformanceMetrics {
            errors: 3,
            tool_calls: 10,
            ..Default::default()
        };
        assert!((m.error_rate() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn avg_tokens_per_iteration_zero_when_no_iterations() {
        let m = PerformanceMetrics::default();
        assert_eq!(m.avg_tokens_per_iteration(), 0);
    }

    #[test]
    fn avg_tokens_per_iteration_correct_value() {
        let m = PerformanceMetrics {
            iterations: 4,
            total_tokens: 1000,
            ..Default::default()
        };
        assert_eq!(m.avg_tokens_per_iteration(), 250);
    }

    #[test]
    fn health_monitor_register_and_status() {
        let mut monitor = HealthMonitor::new(HealthMonitorConfig::default());
        assert_eq!(monitor.status("agent-1"), HealthStatus::Unknown);

        monitor.register("agent-1");
        assert_eq!(monitor.status("agent-1"), HealthStatus::Healthy);

        monitor.unregister("agent-1");
        assert_eq!(monitor.status("agent-1"), HealthStatus::Unknown);
    }
}
