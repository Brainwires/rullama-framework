//! Runtime configuration for the IRC adapter.

use serde::{Deserialize, Serialize};

/// Configuration for one IRC network connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcConfig {
    /// IRC server hostname (e.g. `irc.libera.chat`).
    pub server: String,
    /// IRC port (default 6697 for TLS, 6667 for plaintext).
    pub port: u16,
    /// Whether to wrap the socket in TLS.
    pub use_tls: bool,
    /// The nick the bot advertises.
    pub nick: String,
    /// USER/realname fields.
    pub username: String,
    /// GECOS / realname.
    pub realname: String,
    /// SASL PLAIN password, if authentication is required.
    pub sasl_password: Option<String>,
    /// Comma-separated list of channels to auto-join.
    pub channels: Vec<String>,
    /// Prefix that triggers forwarding for public channel messages.
    /// PMs to the bot always forward regardless of this setting.
    pub message_prefix: String,
    /// Gateway WS URL.
    pub gateway_url: String,
    /// Optional gateway handshake token.
    pub gateway_token: Option<String>,
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            server: "irc.libera.chat".into(),
            port: 6697,
            use_tls: true,
            nick: "brainclaw".into(),
            username: "brainclaw".into(),
            realname: "BrainClaw Bot".into(),
            sasl_password: None,
            channels: Vec::new(),
            message_prefix: "brainclaw: ".into(),
            gateway_url: "ws://127.0.0.1:18789/ws".into(),
            gateway_token: None,
        }
    }
}

/// Split a comma-separated `IRC_CHANNELS` env value into individual
/// channel names, trimming whitespace and dropping empties.
pub fn parse_channel_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_tls_6697() {
        let cfg = IrcConfig::default();
        assert_eq!(cfg.port, 6697);
        assert!(cfg.use_tls);
        assert_eq!(cfg.message_prefix, "brainclaw: ");
    }

    #[test]
    fn channel_list_parses() {
        let list = parse_channel_list("#one, #two ,#three");
        assert_eq!(list, vec!["#one", "#two", "#three"]);
    }

    #[test]
    fn channel_list_ignores_empty_commas() {
        let list = parse_channel_list(",,#x,,");
        assert_eq!(list, vec!["#x"]);
    }

    #[test]
    fn channel_list_empty_string_returns_empty() {
        assert!(parse_channel_list("").is_empty());
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = IrcConfig {
            server: "s".into(),
            port: 1234,
            use_tls: false,
            nick: "n".into(),
            username: "u".into(),
            realname: "r".into(),
            sasl_password: Some("p".into()),
            channels: vec!["#a".into()],
            message_prefix: "!".into(),
            gateway_url: "ws://x/".into(),
            gateway_token: None,
        };
        let s = serde_json::to_string(&cfg).unwrap();
        let parsed: IrcConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.port, 1234);
        assert_eq!(parsed.sasl_password.as_deref(), Some("p"));
    }
}
