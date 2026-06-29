//! Agent supervisor — monitors health and restarts degraded agents.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::health::{HealthMonitor, HealthMonitorConfig, HealthStatus};
use super::hibernate::AgentLifecycleManager;
use crate::safety::{ApprovalPolicy, AutonomousOperation, SafetyStop};

/// Policy for how the supervisor responds to degraded agents.
#[derive(Debug, Clone)]
pub enum SupervisorAction {
    /// Do nothing, just log.
    Ignore,
    /// Restart the agent.
    Restart,
    /// Shut down the agent.
    Shutdown,
    /// Escalate to human (log warning + stop autonomous operations).
    Escalate,
    /// Invoke crash recovery: diagnose the failure, apply a fix, rebuild, and resume.
    CrashRecover,
}

/// Configuration for the supervisor.
#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// How often to check agent health (seconds).
    pub check_interval_secs: u64,
    /// Maximum restart attempts per agent before escalating.
    pub max_restarts: u32,
    /// Action to take when an agent is degraded.
    pub on_degraded: SupervisorAction,
    /// Action to take when an agent is unhealthy/unknown.
    pub on_unhealthy: SupervisorAction,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 30,
            max_restarts: 3,
            on_degraded: SupervisorAction::Ignore,
            on_unhealthy: SupervisorAction::Restart,
        }
    }
}

/// Supervisor that monitors agent health and takes corrective action.
///
/// Periodically evaluates all registered agents, applying the configured
/// policy (ignore, restart, shutdown, escalate, or crash-recover) based
/// on each agent's health status.
pub struct AgentSupervisor {
    config: SupervisorConfig,
    health: Arc<RwLock<HealthMonitor>>,
    lifecycle: Arc<dyn AgentLifecycleManager>,
    approval: Arc<dyn ApprovalPolicy>,
    restart_counts: HashMap<String, u32>,
}

impl AgentSupervisor {
    /// Create a new supervisor with the given configuration, lifecycle manager, and approval policy.
    pub fn new(
        config: SupervisorConfig,
        lifecycle: Arc<dyn AgentLifecycleManager>,
        approval: Arc<dyn ApprovalPolicy>,
    ) -> Self {
        Self {
            health: Arc::new(RwLock::new(HealthMonitor::new(
                HealthMonitorConfig::default(),
            ))),
            config,
            lifecycle,
            approval,
            restart_counts: HashMap::new(),
        }
    }

    /// Get a reference to the shared health monitor.
    pub fn health_monitor(&self) -> Arc<RwLock<HealthMonitor>> {
        self.health.clone()
    }

    /// Run one supervision cycle: evaluate health and take action.
    pub async fn check_cycle(&mut self) -> Vec<SupervisorEvent> {
        let mut events = Vec::new();
        let degraded = self.health.write().await.evaluate_all();

        for (agent_id, status, signals) in degraded {
            let action = match status {
                HealthStatus::Degraded => &self.config.on_degraded,
                HealthStatus::Unhealthy | HealthStatus::Unknown => &self.config.on_unhealthy,
                HealthStatus::Healthy => continue,
            };

            match action {
                SupervisorAction::Ignore => {
                    tracing::debug!("Supervisor: ignoring degraded agent {agent_id}");
                }
                SupervisorAction::Restart => {
                    let count = self.restart_counts.entry(agent_id.clone()).or_insert(0);
                    if *count >= self.config.max_restarts {
                        tracing::warn!(
                            "Supervisor: agent {agent_id} exceeded max restarts ({}), escalating",
                            self.config.max_restarts
                        );
                        events.push(SupervisorEvent::Escalated {
                            agent_id: agent_id.clone(),
                            reason: "max restarts exceeded".to_string(),
                        });
                        continue;
                    }

                    let op = AutonomousOperation::RestartAgent {
                        agent_id: agent_id.clone(),
                        reason: format!("health status: {status:?}, signals: {signals:?}"),
                    };

                    match self.approval.check(&op).await {
                        Ok(()) => match self.lifecycle.shutdown(&agent_id).await {
                            Ok(()) => {
                                *count += 1;
                                tracing::info!(
                                    "Supervisor: restarted agent {agent_id} (attempt {count})"
                                );
                                events.push(SupervisorEvent::Restarted {
                                    agent_id: agent_id.clone(),
                                    attempt: *count,
                                });
                            }
                            Err(e) => {
                                tracing::error!("Supervisor: failed to restart {agent_id}: {e}");
                                events.push(SupervisorEvent::RestartFailed {
                                    agent_id: agent_id.clone(),
                                    error: e.to_string(),
                                });
                            }
                        },
                        Err(SafetyStop::OperationRejected(reason)) => {
                            tracing::warn!("Supervisor: restart of {agent_id} rejected: {reason}");
                            events.push(SupervisorEvent::Escalated {
                                agent_id: agent_id.clone(),
                                reason,
                            });
                        }
                        Err(stop) => {
                            tracing::warn!("Supervisor: safety stop for {agent_id}: {stop}");
                        }
                    }
                }
                SupervisorAction::Shutdown => {
                    if let Err(e) = self.lifecycle.shutdown(&agent_id).await {
                        tracing::error!("Supervisor: failed to shut down {agent_id}: {e}");
                    } else {
                        events.push(SupervisorEvent::ShutDown {
                            agent_id: agent_id.clone(),
                        });
                    }
                }
                SupervisorAction::CrashRecover => {
                    tracing::info!("Supervisor: initiating crash recovery for agent {agent_id}");
                    events.push(SupervisorEvent::CrashRecoveryStarted {
                        agent_id: agent_id.clone(),
                        crash_id: uuid::Uuid::new_v4().to_string(),
                    });
                    // Actual crash recovery is handled by the CrashHandler
                    // which must be wired in by the host application. The supervisor
                    // emits the event so the orchestrator can invoke CrashHandler.
                }
                SupervisorAction::Escalate => {
                    events.push(SupervisorEvent::Escalated {
                        agent_id: agent_id.clone(),
                        reason: format!("status: {status:?}"),
                    });
                }
            }
        }

        events
    }

    /// Run the supervisor loop until cancelled.
    pub async fn run(&mut self, cancel: tokio::sync::watch::Receiver<bool>) {
        let interval = std::time::Duration::from_secs(self.config.check_interval_secs);
        loop {
            if *cancel.borrow() {
                break;
            }
            let events = self.check_cycle().await;
            for event in &events {
                tracing::info!("Supervisor event: {event:?}");
            }
            tokio::time::sleep(interval).await;
        }
    }
}

/// Events produced by the supervisor.
#[derive(Debug, Clone)]
pub enum SupervisorEvent {
    /// An agent was successfully restarted.
    Restarted {
        /// Identifier of the restarted agent.
        agent_id: String,
        /// Restart attempt number.
        attempt: u32,
    },
    /// An agent restart attempt failed.
    RestartFailed {
        /// Identifier of the agent.
        agent_id: String,
        /// Error message from the failed restart.
        error: String,
    },
    /// An agent was shut down.
    ShutDown {
        /// Identifier of the shut-down agent.
        agent_id: String,
    },
    /// An issue was escalated for human attention.
    Escalated {
        /// Identifier of the agent.
        agent_id: String,
        /// Reason for escalation.
        reason: String,
    },
    /// Crash recovery was initiated for an agent.
    CrashRecoveryStarted {
        /// Identifier of the agent.
        agent_id: String,
        /// Crash identifier.
        crash_id: String,
    },
    /// Crash recovery completed successfully.
    CrashRecovered {
        /// Identifier of the agent.
        agent_id: String,
        /// Summary of the fix applied.
        fix_summary: String,
    },
    /// Crash recovery failed.
    CrashRecoveryFailed {
        /// Identifier of the agent.
        agent_id: String,
        /// Error message.
        error: String,
    },
}
