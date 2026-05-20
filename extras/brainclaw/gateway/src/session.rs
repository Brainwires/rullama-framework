//! Session management for mapping channel users to agent sessions.

use std::time::Duration;

use chrono::Utc;
use dashmap::DashMap;
use uuid::Uuid;

use brainwires_network::channels::identity::{ChannelSession, ChannelUser};

/// Manages the mapping of (platform, platform_user_id) to agent sessions.
pub struct SessionManager {
    /// Maps (platform, platform_user_id) -> session.
    sessions: DashMap<(String, String), ChannelSession>,
}

impl SessionManager {
    /// Create a new empty session manager.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Get an existing session or create a new one for the given user.
    pub fn get_or_create_session(&self, user: &ChannelUser) -> ChannelSession {
        let key = (user.platform.clone(), user.platform_user_id.clone());

        // Try to get existing session and update last_activity
        if let Some(mut entry) = self.sessions.get_mut(&key) {
            entry.last_activity = Utc::now();
            return entry.clone();
        }

        // Create new session
        let now = Utc::now();
        let session_id = Uuid::new_v4();
        let session = ChannelSession {
            id: session_id,
            channel_user: user.clone(),
            agent_session_id: session_id.to_string(),
            created_at: now,
            last_activity: now,
        };

        self.sessions.insert(key, session.clone());
        session
    }

    /// Get an existing session for a platform user.
    pub fn get_session(&self, platform: &str, user_id: &str) -> Option<ChannelSession> {
        let key = (platform.to_string(), user_id.to_string());
        self.sessions.get(&key).map(|entry| entry.clone())
    }

    /// Remove a session for a platform user.
    pub fn remove_session(&self, platform: &str, user_id: &str) {
        let key = (platform.to_string(), user_id.to_string());
        self.sessions.remove(&key);
    }

    /// List all active sessions.
    pub fn list_sessions(&self) -> Vec<ChannelSession> {
        self.sessions
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Remove sessions that have been inactive longer than the given timeout.
    pub fn cleanup_expired(&self, timeout: Duration) {
        let cutoff =
            Utc::now() - chrono::Duration::from_std(timeout).unwrap_or(chrono::Duration::hours(1));
        self.sessions
            .retain(|_key, session| session.last_activity > cutoff);
    }

    /// Return the number of active sessions.
    pub fn count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_user(platform: &str, user_id: &str) -> ChannelUser {
        ChannelUser {
            platform: platform.to_string(),
            platform_user_id: user_id.to_string(),
            display_name: format!("User {user_id}"),
            username: None,
            avatar_url: None,
        }
    }

    #[test]
    fn get_or_create_creates_new_session() {
        let mgr = SessionManager::new();
        let user = test_user("discord", "12345");

        let session = mgr.get_or_create_session(&user);
        assert_eq!(session.channel_user.platform, "discord");
        assert_eq!(session.channel_user.platform_user_id, "12345");
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn get_or_create_returns_existing_session() {
        let mgr = SessionManager::new();
        let user = test_user("discord", "12345");

        let session1 = mgr.get_or_create_session(&user);
        let session2 = mgr.get_or_create_session(&user);

        assert_eq!(session1.id, session2.id);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn get_session_returns_none_for_missing() {
        let mgr = SessionManager::new();
        assert!(mgr.get_session("discord", "99999").is_none());
    }

    #[test]
    fn get_session_returns_existing() {
        let mgr = SessionManager::new();
        let user = test_user("telegram", "42");
        let created = mgr.get_or_create_session(&user);

        let found = mgr.get_session("telegram", "42").unwrap();
        assert_eq!(found.id, created.id);
    }

    #[test]
    fn remove_session_works() {
        let mgr = SessionManager::new();
        let user = test_user("slack", "100");
        mgr.get_or_create_session(&user);
        assert_eq!(mgr.count(), 1);

        mgr.remove_session("slack", "100");
        assert_eq!(mgr.count(), 0);
        assert!(mgr.get_session("slack", "100").is_none());
    }

    #[test]
    fn list_sessions_returns_all() {
        let mgr = SessionManager::new();
        mgr.get_or_create_session(&test_user("discord", "1"));
        mgr.get_or_create_session(&test_user("telegram", "2"));
        mgr.get_or_create_session(&test_user("slack", "3"));

        let sessions = mgr.list_sessions();
        assert_eq!(sessions.len(), 3);
    }

    #[test]
    fn cleanup_expired_removes_old_sessions() {
        let mgr = SessionManager::new();
        let user = test_user("discord", "old-user");

        // Create session, then manually backdate it
        mgr.get_or_create_session(&user);
        {
            let key = ("discord".to_string(), "old-user".to_string());
            let mut entry = mgr.sessions.get_mut(&key).unwrap();
            entry.last_activity = Utc::now() - chrono::Duration::hours(2);
        }

        // Also create a fresh session
        mgr.get_or_create_session(&test_user("discord", "new-user"));

        assert_eq!(mgr.count(), 2);

        // Clean up sessions older than 1 hour
        mgr.cleanup_expired(Duration::from_secs(3600));

        assert_eq!(mgr.count(), 1);
        assert!(mgr.get_session("discord", "old-user").is_none());
        assert!(mgr.get_session("discord", "new-user").is_some());
    }
}
