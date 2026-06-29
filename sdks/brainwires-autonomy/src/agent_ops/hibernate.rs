//! Agent hibernation data types.
//!
//! These types represent the manifest and session metadata for hibernated agents.
//! The actual hibernate/resume process management is left to the CLI or host application
//! which implements the [`AgentLifecycleManager`] trait.

use serde::{Deserialize, Serialize};

/// Hibernate manifest — list of sessions to restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HibernateManifest {
    /// Timestamp when hibernation occurred (Unix epoch seconds).
    pub hibernated_at: i64,
    /// List of hibernated sessions.
    pub sessions: Vec<HibernatedSession>,
    /// Version for forward compatibility.
    pub version: u32,
}

impl HibernateManifest {
    /// Current manifest version.
    pub const VERSION: u32 = 1;

    /// Create a new manifest from a list of sessions.
    pub fn new(sessions: Vec<HibernatedSession>) -> Self {
        Self {
            hibernated_at: chrono::Utc::now().timestamp(),
            sessions,
            version: Self::VERSION,
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Sort sessions so parents come before children (for proper resumption order).
    pub fn sorted_by_hierarchy(&self) -> Vec<&HibernatedSession> {
        let mut sorted: Vec<&HibernatedSession> = self.sessions.iter().collect();
        sorted.sort_by(|a, b| match (&a.parent_agent_id, &b.parent_agent_id) {
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        });
        sorted
    }
}

/// A hibernated session's metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HibernatedSession {
    /// Session ID.
    pub session_id: String,
    /// Model used.
    pub model: String,
    /// Working directory.
    pub working_directory: String,
    /// Parent agent ID (for hierarchy restoration).
    pub parent_agent_id: Option<String>,
    /// Reason for spawning (if child agent).
    pub spawn_reason: Option<String>,
    /// Whether the agent was busy when hibernated.
    pub was_busy: bool,
}

/// Trait for managing agent lifecycle (hibernate/resume).
///
/// Implementations handle the actual process management, IPC, and session
/// token handling which are host-specific.
#[async_trait::async_trait]
pub trait AgentLifecycleManager: Send + Sync {
    /// Hibernate an agent, saving its state for later resumption.
    async fn hibernate(&self, agent_id: &str) -> anyhow::Result<HibernatedSession>;
    /// Resume a previously hibernated agent.
    async fn resume(&self, session: &HibernatedSession) -> anyhow::Result<String>;
    /// Check if an agent is currently alive.
    async fn is_alive(&self, agent_id: &str) -> bool;
    /// Gracefully shut down an agent.
    async fn shutdown(&self, agent_id: &str) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_serialization() {
        let session = HibernatedSession {
            session_id: "test-123".to_string(),
            model: "gpt-4".to_string(),
            working_directory: "/home/user/project".to_string(),
            parent_agent_id: None,
            spawn_reason: None,
            was_busy: false,
        };

        let manifest = HibernateManifest::new(vec![session]);
        let json = manifest.to_json().unwrap();
        let parsed = HibernateManifest::from_json(&json).unwrap();
        assert_eq!(parsed.sessions.len(), 1);
        assert_eq!(parsed.sessions[0].session_id, "test-123");
        assert_eq!(parsed.version, HibernateManifest::VERSION);
    }

    #[test]
    fn test_hierarchy_sort() {
        let parent = HibernatedSession {
            session_id: "parent".to_string(),
            model: "m".to_string(),
            working_directory: "/".to_string(),
            parent_agent_id: None,
            spawn_reason: None,
            was_busy: false,
        };
        let child = HibernatedSession {
            session_id: "child".to_string(),
            model: "m".to_string(),
            working_directory: "/".to_string(),
            parent_agent_id: Some("parent".to_string()),
            spawn_reason: Some("subtask".to_string()),
            was_busy: true,
        };

        let manifest = HibernateManifest::new(vec![child, parent]);
        let sorted = manifest.sorted_by_hierarchy();
        assert_eq!(sorted[0].session_id, "parent");
        assert_eq!(sorted[1].session_id, "child");
    }
}
