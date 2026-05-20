//! Configuration for the Signal channel adapter.
//!
//! BrainClaw connects to Signal via the `signal-cli-rest-api` daemon
//! (see <https://github.com/bbernhard/signal-cli-rest-api>).  Start it with:
//!
//! ```text
//! signal-cli -a +1234567890 daemon --http 127.0.0.1:8080
//! ```
//!
//! or via the Docker image `bbernhard/signal-cli-rest-api`.

use serde::{Deserialize, Serialize};

/// Configuration for the Signal channel adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Base URL of the signal-cli REST API daemon (e.g. "http://127.0.0.1:8080").
    pub api_url: String,
    /// The bot's own Signal phone number in E.164 format (e.g. "+14155552671").
    pub phone_number: String,
    /// WebSocket URL of the brainwires-gateway.
    pub gateway_url: String,
    /// Optional auth token for the gateway handshake.
    pub gateway_token: Option<String>,
    /// In group chats, only respond when @mentioned by name.
    /// Direct messages always respond.
    #[serde(default)]
    pub group_mention_required: bool,
    /// The bot's display name used for @mention detection in group messages.
    #[serde(default)]
    pub bot_name: Option<String>,
    /// Additional keyword patterns (case-insensitive) that trigger a response
    /// in group messages even without an @mention.
    #[serde(default)]
    pub mention_patterns: Vec<String>,
    /// Allowed sender phone numbers. Empty = accept all senders.
    #[serde(default)]
    pub sender_allowlist: Vec<String>,
    /// Allowed group IDs (base64). Empty = accept all groups.
    #[serde(default)]
    pub group_allowlist: Vec<String>,
    /// Polling interval in milliseconds when WebSocket is not available.
    /// Default: 2000 ms.
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
}

fn default_poll_interval_ms() -> u64 {
    2000
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            api_url: "http://127.0.0.1:8080".to_string(),
            phone_number: String::new(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            group_mention_required: false,
            bot_name: None,
            mention_patterns: Vec::new(),
            sender_allowlist: Vec::new(),
            group_allowlist: Vec::new(),
            poll_interval_ms: 2000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = SignalConfig::default();
        assert_eq!(cfg.api_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.gateway_url, "ws://127.0.0.1:18789/ws");
        assert!(cfg.phone_number.is_empty());
        assert!(cfg.gateway_token.is_none());
        assert!(!cfg.group_mention_required);
        assert!(cfg.bot_name.is_none());
        assert!(cfg.mention_patterns.is_empty());
        assert!(cfg.sender_allowlist.is_empty());
        assert!(cfg.group_allowlist.is_empty());
        assert_eq!(cfg.poll_interval_ms, 2000);
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = SignalConfig {
            api_url: "http://signal:8080".to_string(),
            phone_number: "+14155551234".to_string(),
            gateway_url: "ws://gw:18789/ws".to_string(),
            gateway_token: Some("secret".to_string()),
            group_mention_required: true,
            bot_name: Some("BrainBot".to_string()),
            mention_patterns: vec!["help".to_string()],
            sender_allowlist: vec!["+1234567890".to_string()],
            group_allowlist: vec!["abc123==".to_string()],
            poll_interval_ms: 5000,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: SignalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.api_url, cfg.api_url);
        assert_eq!(back.phone_number, cfg.phone_number);
        assert_eq!(back.gateway_url, cfg.gateway_url);
        assert_eq!(back.gateway_token, cfg.gateway_token);
        assert_eq!(back.group_mention_required, cfg.group_mention_required);
        assert_eq!(back.bot_name, cfg.bot_name);
        assert_eq!(back.mention_patterns, cfg.mention_patterns);
        assert_eq!(back.sender_allowlist, cfg.sender_allowlist);
        assert_eq!(back.group_allowlist, cfg.group_allowlist);
        assert_eq!(back.poll_interval_ms, cfg.poll_interval_ms);
    }

    #[test]
    fn config_defaults_applied_for_missing_serde_fields() {
        let json = r#"{
            "api_url": "http://127.0.0.1:8080",
            "phone_number": "+14155551234",
            "gateway_url": "ws://localhost/ws"
        }"#;
        let cfg: SignalConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.group_mention_required);
        assert!(cfg.mention_patterns.is_empty());
        assert!(cfg.sender_allowlist.is_empty());
        assert_eq!(cfg.poll_interval_ms, 2000);
    }
}
