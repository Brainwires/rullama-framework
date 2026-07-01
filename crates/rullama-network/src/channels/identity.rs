//! User and session identity types for channel communication.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A user on a messaging platform.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelUser {
    /// The platform this user belongs to (e.g., "discord", "telegram").
    pub platform: String,
    /// The user's unique identifier on the platform.
    pub platform_user_id: String,
    /// The user's display name.
    pub display_name: String,
    /// The user's username, if available.
    pub username: Option<String>,
    /// URL to the user's avatar, if available.
    pub avatar_url: Option<String>,
}

/// A session linking a channel user to an agent session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelSession {
    /// Unique session identifier.
    pub id: Uuid,
    /// The channel user associated with this session.
    pub channel_user: ChannelUser,
    /// The agent session identifier this channel session is linked to.
    pub agent_session_id: String,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// When the session was last active.
    pub last_activity: DateTime<Utc>,
}

/// Identifies a conversation on a specific platform and channel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ConversationId {
    /// The platform (e.g., "discord", "telegram", "slack").
    pub platform: String,
    /// The channel identifier on the platform.
    pub channel_id: String,
    /// The server/workspace identifier, if applicable.
    pub server_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_user_serde_roundtrip() {
        let user = ChannelUser {
            platform: "discord".to_string(),
            platform_user_id: "123456".to_string(),
            display_name: "Alice".to_string(),
            username: Some("alice#1234".to_string()),
            avatar_url: None,
        };
        let json = serde_json::to_string(&user).unwrap();
        let deserialized: ChannelUser = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.display_name, "Alice");
        assert_eq!(deserialized.platform_user_id, "123456");
    }

    #[test]
    fn channel_session_serde_roundtrip() {
        let session = ChannelSession {
            id: Uuid::new_v4(),
            channel_user: ChannelUser {
                platform: "telegram".to_string(),
                platform_user_id: "789".to_string(),
                display_name: "Bob".to_string(),
                username: None,
                avatar_url: None,
            },
            agent_session_id: "agent-sess-001".to_string(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
        };
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChannelSession = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.agent_session_id, "agent-sess-001");
    }

    #[test]
    fn conversation_id_serde_roundtrip() {
        let conv = ConversationId {
            platform: "slack".to_string(),
            channel_id: "C01234".to_string(),
            server_id: Some("T56789".to_string()),
        };
        let json = serde_json::to_string(&conv).unwrap();
        let deserialized: ConversationId = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, conv);
    }
}
