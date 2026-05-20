//! Configuration for the Nextcloud Talk adapter.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Runtime configuration for the Nextcloud Talk adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudConfig {
    /// Nextcloud base URL, e.g. `https://cloud.example.com`.
    pub server_url: String,
    /// Login user id (typically a short username, not an email).
    pub username: String,
    /// Nextcloud app password — never the account password.
    pub app_password: String,
    /// Room tokens to watch. Must not be empty.
    pub room_tokens: Vec<String>,
    /// Poll interval seconds (default 2).
    pub poll_interval_secs: u64,
    /// Gateway WebSocket URL.
    pub gateway_url: String,
    /// Optional gateway auth token.
    pub gateway_token: Option<String>,
    /// Directory for the cursor state file.
    pub state_dir: PathBuf,
}

impl Default for NextcloudConfig {
    fn default() -> Self {
        Self {
            server_url: String::new(),
            username: String::new(),
            app_password: String::new(),
            room_tokens: Vec::new(),
            poll_interval_secs: 2,
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            state_dir: default_state_dir(),
        }
    }
}

/// Default `state_dir` — under the user's home if known.
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
        let c = NextcloudConfig::default();
        assert_eq!(c.poll_interval_secs, 2);
    }

    #[test]
    fn serde_roundtrip() {
        let c = NextcloudConfig {
            server_url: "https://cloud.example.com".into(),
            username: "alice".into(),
            app_password: "placeholder".into(),
            room_tokens: vec!["abc123".into()],
            poll_interval_secs: 5,
            gateway_url: "ws://gw".into(),
            gateway_token: None,
            state_dir: PathBuf::from("/tmp/x"),
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: NextcloudConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.room_tokens.len(), 1);
    }
}
