//! Configuration for the iMessage / BlueBubbles adapter.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImessageConfig {
    /// Base URL of the BlueBubbles server, e.g. `https://mac.tailnet.ts.net:1234`.
    pub server_url: String,
    /// BlueBubbles server password — appended as `?password=` on every request.
    pub password: String,
    /// Polling interval (seconds) between `GET /api/v1/message` cycles.
    pub poll_interval_secs: u64,
    /// Chat GUIDs to watch. Empty = watch all chats.
    pub chat_guids: Vec<String>,
    /// Gateway WebSocket URL.
    pub gateway_url: String,
    /// Optional gateway auth token.
    pub gateway_token: Option<String>,
    /// Directory in which to persist the polling cursor file.
    pub state_dir: PathBuf,
}

impl Default for ImessageConfig {
    fn default() -> Self {
        Self {
            server_url: String::new(),
            password: String::new(),
            poll_interval_secs: 2,
            chat_guids: Vec::new(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            state_dir: default_state_dir(),
        }
    }
}

/// Default location for the cursor state JSON file directory.
pub fn default_state_dir() -> PathBuf {
    if let Some(h) = dirs::home_dir() {
        h.join(".brainclaw").join("state")
    } else {
        PathBuf::from(".brainclaw/state")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_sane() {
        let cfg = ImessageConfig::default();
        assert_eq!(cfg.poll_interval_secs, 2);
        assert!(cfg.chat_guids.is_empty());
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = ImessageConfig {
            server_url: "https://mac.example.com".into(),
            password: "secret-placeholder".into(),
            poll_interval_secs: 5,
            chat_guids: vec!["iMessage;-;+15551234567".into()],
            gateway_url: "ws://gw/ws".into(),
            gateway_token: None,
            state_dir: PathBuf::from("/tmp/x"),
        };
        let s = serde_json::to_string(&cfg).unwrap();
        let back: ImessageConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.poll_interval_secs, 5);
        assert_eq!(back.chat_guids.len(), 1);
    }
}
