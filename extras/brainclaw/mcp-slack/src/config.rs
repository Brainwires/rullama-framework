//! Configuration types for the Slack channel adapter.

use serde::{Deserialize, Serialize};

/// Configuration for the Slack channel adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    /// Slack bot token (xoxb-...) for Web API calls (required).
    pub slack_bot_token: String,
    /// Slack app-level token (xapp-...) for Socket Mode (required).
    pub slack_app_token: String,
    /// WebSocket URL for the brainwires-gateway.
    pub gateway_url: String,
    /// Optional authentication token for the gateway handshake.
    pub gateway_token: Option<String>,
    /// In public/private channels, only respond when the bot is @mentioned.
    /// DMs (channel IDs starting with "D") always respond.
    /// Default: false (backward compatible).
    #[serde(default)]
    pub group_mention_required: bool,
    /// The bot's Slack user ID (e.g. "U0123456789") used to detect @mentions
    /// in channel messages.  When set, `<@BOT_USER_ID>` must appear in the
    /// message text for a response to be sent.
    #[serde(default)]
    pub bot_user_id: Option<String>,
    /// Additional keyword patterns (case-insensitive substring match) that
    /// trigger a response in channel messages even without an @mention.
    /// Only used when `group_mention_required = true`.
    #[serde(default)]
    pub mention_patterns: Vec<String>,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            slack_bot_token: String::new(),
            slack_app_token: String::new(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            group_mention_required: false,
            bot_user_id: None,
            mention_patterns: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_gateway_url() {
        let config = SlackConfig::default();
        assert_eq!(config.gateway_url, "ws://127.0.0.1:18789/ws");
        assert!(config.slack_bot_token.is_empty());
        assert!(config.slack_app_token.is_empty());
        assert!(config.gateway_token.is_none());
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = SlackConfig {
            slack_bot_token: "xoxb-test-token".to_string(),
            slack_app_token: "xapp-test-token".to_string(),
            gateway_url: "ws://localhost:9999/ws".to_string(),
            gateway_token: Some("gw-secret".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SlackConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.slack_bot_token, "xoxb-test-token");
        assert_eq!(parsed.slack_app_token, "xapp-test-token");
        assert_eq!(parsed.gateway_url, "ws://localhost:9999/ws");
        assert_eq!(parsed.gateway_token.as_deref(), Some("gw-secret"));
    }

    #[test]
    fn config_from_env_pattern() {
        let config = SlackConfig {
            slack_bot_token: "xoxb-12345".to_string(),
            slack_app_token: "xapp-67890".to_string(),
            ..Default::default()
        };
        assert_eq!(config.slack_bot_token, "xoxb-12345");
        assert_eq!(config.slack_app_token, "xapp-67890");
        assert_eq!(config.gateway_url, "ws://127.0.0.1:18789/ws");
    }
}
