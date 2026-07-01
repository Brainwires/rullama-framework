//! Heartbeat collector for agent discovery and status monitoring
//!
//! Collects information about all running agents and detects changes
//! for broadcasting to the remote backend.
//!
//! All functions take `sessions_dir: &Path` instead of depending on
//! CLI-specific path resolution. The CLI passes `PlatformPaths::sessions_dir()`
//! at call sites.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::protocol::{AgentEventType, RemoteAgentInfo};
use crate::ipc::discovery::{cleanup_stale_sockets, list_agent_sessions_with_metadata};
use crate::ipc::protocol::AgentMetadata;

/// Data collected during a heartbeat
#[derive(Debug, Clone)]
pub struct HeartbeatData {
    /// Current state of all agents
    pub agents: Vec<RemoteAgentInfo>,
    /// System CPU load (0.0 to 1.0)
    pub system_load: f32,
    /// Hostname of the machine
    pub hostname: String,
    /// Operating system
    pub os: String,
    /// CLI version
    pub version: String,
}

/// Agent state change event
#[derive(Debug, Clone)]
pub struct AgentEvent {
    /// Type of event
    pub event_type: AgentEventType,
    /// Agent session ID
    pub agent_id: String,
    /// Additional event data
    pub data: serde_json::Value,
}

/// Collects heartbeat data and detects agent changes
pub struct HeartbeatCollector {
    /// Last known state of agents (session_id -> info)
    last_agents: HashMap<String, RemoteAgentInfo>,
    /// Sessions directory for agent discovery
    sessions_dir: PathBuf,
    /// Version string (injected, not read from env!)
    version: String,
}

impl HeartbeatCollector {
    /// Create a new heartbeat collector
    ///
    /// # Arguments
    /// * `sessions_dir` - Directory containing agent session files
    /// * `version` - CLI version string (injected to avoid env! dependency)
    pub fn new(sessions_dir: PathBuf, version: String) -> Self {
        Self {
            last_agents: HashMap::new(),
            sessions_dir,
            version,
        }
    }

    /// Collect current state of all agents
    ///
    /// This also cleans up stale socket files from dead sessions to ensure
    /// only actually running agents are reported.
    pub async fn collect(&mut self) -> Result<HeartbeatData> {
        // Clean up stale sockets first to avoid reporting dead sessions
        if let Err(e) = cleanup_stale_sockets(&self.sessions_dir).await {
            tracing::warn!("Failed to cleanup stale sockets: {}", e);
        }

        let metadata_list =
            list_agent_sessions_with_metadata(&self.sessions_dir).unwrap_or_default();

        let agents: Vec<RemoteAgentInfo> = metadata_list
            .into_iter()
            .map(RemoteAgentInfo::from)
            .collect();

        // Update last known state
        self.last_agents = agents
            .iter()
            .map(|a| (a.session_id.clone(), a.clone()))
            .collect();

        Ok(HeartbeatData {
            agents,
            system_load: get_system_load(),
            hostname: gethostname::gethostname().to_string_lossy().to_string(),
            os: std::env::consts::OS.to_string(),
            version: self.version.clone(),
        })
    }

    /// Detect changes since last collection
    ///
    /// Returns a list of agent events representing what changed.
    pub fn detect_changes(&mut self) -> Result<Vec<AgentEvent>> {
        let current_metadata =
            list_agent_sessions_with_metadata(&self.sessions_dir).unwrap_or_default();
        let current_agents: HashMap<String, RemoteAgentInfo> = current_metadata
            .into_iter()
            .map(|m| {
                let info = RemoteAgentInfo::from(m);
                (info.session_id.clone(), info)
            })
            .collect();

        let mut events = Vec::new();

        // Check for new agents (spawned)
        for (session_id, agent) in &current_agents {
            if !self.last_agents.contains_key(session_id) {
                events.push(AgentEvent {
                    event_type: AgentEventType::Spawned,
                    agent_id: session_id.clone(),
                    data: serde_json::to_value(agent).unwrap_or_default(),
                });
            }
        }

        // Check for removed agents (exited)
        for session_id in self.last_agents.keys() {
            if !current_agents.contains_key(session_id) {
                events.push(AgentEvent {
                    event_type: AgentEventType::Exited,
                    agent_id: session_id.clone(),
                    data: serde_json::json!({}),
                });
            }
        }

        // Check for state changes in existing agents
        for (session_id, current) in &current_agents {
            if let Some(previous) = self.last_agents.get(session_id) {
                // Check busy state change
                if current.is_busy != previous.is_busy {
                    events.push(AgentEvent {
                        event_type: if current.is_busy {
                            AgentEventType::Busy
                        } else {
                            AgentEventType::Idle
                        },
                        agent_id: session_id.clone(),
                        data: serde_json::json!({
                            "is_busy": current.is_busy,
                            "status": current.status,
                        }),
                    });
                }

                // Check for other state changes (message count, status)
                if current.message_count != previous.message_count
                    || current.status != previous.status
                {
                    events.push(AgentEvent {
                        event_type: AgentEventType::StateChanged,
                        agent_id: session_id.clone(),
                        data: serde_json::json!({
                            "message_count": current.message_count,
                            "status": current.status,
                            "previous_message_count": previous.message_count,
                            "previous_status": previous.status,
                        }),
                    });
                }
            }
        }

        // Update last known state
        self.last_agents = current_agents;

        Ok(events)
    }

    /// Get the current list of agents without updating state
    pub fn get_current_agents(&self) -> Vec<RemoteAgentInfo> {
        self.last_agents.values().cloned().collect()
    }

    /// Check if any agents are currently tracked
    pub fn has_agents(&self) -> bool {
        !self.last_agents.is_empty()
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.last_agents.len()
    }

    /// Get the sessions directory
    pub fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }
}

/// Convert from IPC AgentMetadata to Remote AgentInfo
impl From<AgentMetadata> for RemoteAgentInfo {
    fn from(meta: AgentMetadata) -> Self {
        Self {
            session_id: meta.session_id,
            model: meta.model,
            is_busy: meta.is_busy,
            parent_id: meta.parent_agent_id,
            working_directory: meta.working_directory,
            message_count: 0, // Not tracked in AgentMetadata
            last_activity: meta.last_activity,
            status: if meta.is_busy {
                "busy".to_string()
            } else {
                "idle".to_string()
            },
            name: meta.spawn_reason,
        }
    }
}

/// Get current system CPU load
///
/// Returns a value between 0.0 and 1.0.
/// Falls back to 0.0 if load cannot be determined.
fn get_system_load() -> f32 {
    // Try to read from /proc/loadavg on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/loadavg")
            && let Some(first) = contents.split_whitespace().next()
            && let Ok(load) = first.parse::<f32>()
        {
            // Normalize by number of CPUs
            let num_cpus = std::thread::available_parallelism()
                .map(|p| p.get() as f32)
                .unwrap_or(1.0);
            return (load / num_cpus).min(1.0);
        }
    }

    // Try sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        // macOS load average would require different syscalls
        // For now, return 0.0 as placeholder
    }

    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_collector_new() {
        let temp_dir = tempfile::tempdir().unwrap();
        let collector =
            HeartbeatCollector::new(temp_dir.path().to_path_buf(), "0.1.0-test".to_string());
        assert!(!collector.has_agents());
        assert_eq!(collector.agent_count(), 0);
    }

    #[test]
    fn test_remote_agent_info_from_metadata() {
        let metadata = AgentMetadata::new(
            "test-session".to_string(),
            "claude-3-5-sonnet".to_string(),
            "/home/user/project".to_string(),
        );

        let info = RemoteAgentInfo::from(metadata);

        assert_eq!(info.session_id, "test-session");
        assert_eq!(info.model, "claude-3-5-sonnet");
        assert_eq!(info.working_directory, "/home/user/project");
        assert!(!info.is_busy);
        assert_eq!(info.status, "idle");
    }

    #[test]
    fn test_remote_agent_info_busy_status() {
        let mut metadata = AgentMetadata::new(
            "busy-session".to_string(),
            "gpt-4".to_string(),
            "/tmp".to_string(),
        );
        metadata.is_busy = true;

        let info = RemoteAgentInfo::from(metadata);

        assert!(info.is_busy);
        assert_eq!(info.status, "busy");
    }

    #[test]
    fn test_system_load() {
        let load = get_system_load();
        assert!(load >= 0.0);
        assert!(load <= 1.0);
    }
}
