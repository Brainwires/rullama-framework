//! Configuration for the Feishu / Lark adapter.

use serde::{Deserialize, Serialize};

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeishuConfig {
    /// Feishu app id.
    pub app_id: String,
    /// Feishu app secret.
    pub app_secret: String,
    /// Verification token — used for HMAC signing of inbound events.
    pub verification_token: String,
    /// Optional AES encryption key. When set, the webhook decrypts the
    /// body before parsing; we skip decryption when absent (MVP).
    pub encrypt_key: Option<String>,
    /// Gateway WebSocket URL.
    pub gateway_url: String,
    /// Optional gateway auth token.
    pub gateway_token: Option<String>,
    /// Webhook listen address (default `0.0.0.0:9105`).
    pub listen_addr: String,
}

impl Default for FeishuConfig {
    fn default() -> Self {
        Self {
            app_id: String::new(),
            app_secret: String::new(),
            verification_token: String::new(),
            encrypt_key: None,
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            listen_addr: "0.0.0.0:9105".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_listen_addr() {
        assert_eq!(FeishuConfig::default().listen_addr, "0.0.0.0:9105");
    }

    #[test]
    fn serde_roundtrip() {
        let c = FeishuConfig {
            app_id: "cli_a".into(),
            app_secret: "placeholder".into(),
            verification_token: "vt".into(),
            encrypt_key: Some("ek".into()),
            gateway_url: "ws://gw".into(),
            gateway_token: None,
            listen_addr: "127.0.0.1:9105".into(),
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: FeishuConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.app_id, "cli_a");
    }
}
