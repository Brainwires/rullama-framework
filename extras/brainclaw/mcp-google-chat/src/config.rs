//! Configuration for the Google Chat channel adapter.

use serde::{Deserialize, Serialize};

/// Runtime configuration.
///
/// Populated from CLI flags or environment variables in `main.rs`; never
/// persisted. Secret material (service account JSON) is referenced by
/// filesystem path — the daemon operator owns the lifecycle of those
/// files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleChatConfig {
    /// GCP project id that owns the Chat bot.
    pub project_id: String,
    /// Expected `aud` claim on Google-signed ingress JWTs.
    pub audience: String,
    /// Path to the bot service-account JSON key file. Contents are read
    /// lazily by the OAuth minter and never logged.
    pub service_account_key_path: String,
    /// WebSocket URL for the brainwires-gateway.
    pub gateway_url: String,
    /// Optional auth token sent in the gateway handshake.
    pub gateway_token: Option<String>,
    /// HTTP listen address for the webhook server.
    pub listen_addr: String,
}

impl Default for GoogleChatConfig {
    fn default() -> Self {
        Self {
            project_id: String::new(),
            audience: String::new(),
            service_account_key_path: String::new(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            listen_addr: "0.0.0.0:9101".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_expected_listen_addr() {
        let cfg = GoogleChatConfig::default();
        assert_eq!(cfg.listen_addr, "0.0.0.0:9101");
        assert_eq!(cfg.gateway_url, "ws://127.0.0.1:18789/ws");
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = GoogleChatConfig {
            project_id: "my-proj".into(),
            audience: "my-aud".into(),
            service_account_key_path: "/tmp/key.json".into(),
            gateway_url: "ws://gw:1234/ws".into(),
            gateway_token: Some("tok".into()),
            listen_addr: "127.0.0.1:9999".into(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: GoogleChatConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.project_id, "my-proj");
        assert_eq!(parsed.audience, "my-aud");
        assert_eq!(parsed.gateway_token.as_deref(), Some("tok"));
    }
}
