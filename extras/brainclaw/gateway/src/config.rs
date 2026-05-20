//! Gateway configuration.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for the gateway daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Host address to bind to.
    pub host: String,
    /// Port to listen on.
    pub port: u16,
    /// Maximum number of concurrent channel connections.
    pub max_connections: usize,
    /// Session inactivity timeout before automatic cleanup.
    pub session_timeout: Duration,
    /// Allowed API keys for channel connections.
    pub auth_tokens: Vec<String>,
    /// Whether the webhook endpoint is enabled.
    pub webhook_enabled: bool,
    /// URL path for the webhook endpoint.
    pub webhook_path: String,
    /// Whether the admin API is enabled.
    pub admin_enabled: bool,
    /// URL path prefix for admin endpoints.
    pub admin_path: String,
    /// Allowed WebSocket origins. Empty list = allow all (dev mode).
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    /// Whether the built-in WebChat UI is enabled.
    #[serde(default = "default_true")]
    pub webchat_enabled: bool,
    /// Optional shared secret used to sign/verify HS256 JWTs for the
    /// JWT-gated `/webchat/ws` endpoint.  When `None`, the webchat channel
    /// is registered but every upgrade attempt is refused.
    #[serde(default)]
    pub webchat_jwt_secret: Option<String>,
    /// Maximum number of history entries retained per webchat session.
    #[serde(default = "default_webchat_history_limit")]
    pub webchat_session_history_limit: usize,
    /// Maximum attachment size in megabytes for the media pipeline.
    #[serde(default = "default_max_attachment_size")]
    pub max_attachment_size_mb: u64,
    /// Whether to detect and strip system-message spoofing in inbound messages.
    #[serde(default = "default_true")]
    pub strip_system_spoofing: bool,
    /// Whether to redact secret patterns (API keys, SSNs, etc.) in outbound messages.
    #[serde(default = "default_true")]
    pub redact_secrets_in_output: bool,
    /// Maximum messages per minute per user (rate limiting).
    #[serde(default = "default_max_messages")]
    pub max_messages_per_minute: u32,
    /// Maximum tool calls per minute per user (rate limiting).
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls_per_minute: u32,
    /// Optional bearer token required for admin API access.
    /// When `None`, admin endpoints are open (backward compatible).
    #[serde(default)]
    pub admin_token: Option<String>,
    /// Optional shared secret for webhook HMAC-SHA256 signature verification.
    /// When `None`, webhook payloads are accepted without signature checks.
    #[serde(default)]
    pub webhook_secret: Option<String>,
    /// Master switch — when `false`, all channel connections are refused.
    #[serde(default = "default_true")]
    pub channels_enabled: bool,
    /// Allowed channel adapter types (e.g. `["discord", "telegram"]`).
    /// Empty = allow all types.
    #[serde(default)]
    pub allowed_channel_types: Vec<String>,
    /// Allowed channel adapter IDs. Empty = allow all IDs.
    /// These are checked during the WebSocket handshake after token auth.
    #[serde(default)]
    pub allowed_channel_ids: Vec<String>,
}

fn default_true() -> bool {
    true
}

fn default_max_attachment_size() -> u64 {
    10
}

fn default_max_messages() -> u32 {
    20
}

fn default_max_tool_calls() -> u32 {
    30
}

fn default_webchat_history_limit() -> usize {
    50
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 18789,
            max_connections: 256,
            session_timeout: Duration::from_secs(3600),
            auth_tokens: Vec::new(),
            webhook_enabled: true,
            webhook_path: "/webhook".to_string(),
            admin_enabled: true,
            admin_path: "/admin".to_string(),
            allowed_origins: Vec::new(),
            webchat_enabled: true,
            webchat_jwt_secret: None,
            webchat_session_history_limit: 50,
            max_attachment_size_mb: 10,
            strip_system_spoofing: true,
            redact_secrets_in_output: true,
            max_messages_per_minute: 20,
            max_tool_calls_per_minute: 30,
            admin_token: None,
            webhook_secret: None,
            channels_enabled: true,
            allowed_channel_types: Vec::new(),
            allowed_channel_ids: Vec::new(),
        }
    }
}

impl GatewayConfig {
    /// Validate an auth token against the configured allowed tokens.
    ///
    /// If no auth tokens are configured, all tokens are accepted (open mode).
    pub fn validate_token(&self, token: &str) -> bool {
        if self.auth_tokens.is_empty() {
            return true;
        }
        self.auth_tokens.iter().any(|t| t == token)
    }

    /// Returns the full bind address as `host:port`.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = GatewayConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 18789);
        assert_eq!(config.max_connections, 256);
        assert_eq!(config.session_timeout, Duration::from_secs(3600));
        assert!(config.auth_tokens.is_empty());
        assert!(config.webhook_enabled);
        assert_eq!(config.webhook_path, "/webhook");
        assert!(config.admin_enabled);
        assert_eq!(config.admin_path, "/admin");
    }

    #[test]
    fn validate_token_open_mode() {
        let config = GatewayConfig::default();
        assert!(config.validate_token("anything"));
        assert!(config.validate_token(""));
    }

    #[test]
    fn validate_token_with_configured_tokens() {
        let config = GatewayConfig {
            auth_tokens: vec!["secret-1".to_string(), "secret-2".to_string()],
            ..Default::default()
        };
        assert!(config.validate_token("secret-1"));
        assert!(config.validate_token("secret-2"));
        assert!(!config.validate_token("wrong-token"));
        assert!(!config.validate_token(""));
    }

    #[test]
    fn bind_address_format() {
        let config = GatewayConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:18789");
    }
}
