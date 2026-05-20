//! Configuration for the LINE adapter.

use serde::{Deserialize, Serialize};

/// Runtime configuration for the LINE Messaging API adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineConfig {
    /// LINE channel secret — HMAC key for webhook signature verification.
    pub channel_secret: String,
    /// Long-lived channel access token for outbound calls.
    pub channel_access_token: String,
    /// Gateway WebSocket URL.
    pub gateway_url: String,
    /// Optional gateway auth token.
    pub gateway_token: Option<String>,
    /// Webhook listen address (default `0.0.0.0:9104`).
    pub listen_addr: String,
}

impl Default for LineConfig {
    fn default() -> Self {
        Self {
            channel_secret: String::new(),
            channel_access_token: String::new(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            listen_addr: "0.0.0.0:9104".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_listen_addr() {
        let c = LineConfig::default();
        assert_eq!(c.listen_addr, "0.0.0.0:9104");
    }

    #[test]
    fn serde_roundtrip() {
        let c = LineConfig {
            channel_secret: "secret-placeholder".into(),
            channel_access_token: "token-placeholder".into(),
            gateway_url: "ws://gw".into(),
            gateway_token: None,
            listen_addr: "127.0.0.1:9104".into(),
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: LineConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.listen_addr, "127.0.0.1:9104");
    }
}
