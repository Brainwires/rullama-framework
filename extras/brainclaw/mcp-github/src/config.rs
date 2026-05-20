//! Configuration for the GitHub channel adapter.

use serde::{Deserialize, Serialize};

/// Configuration for the GitHub channel adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubConfig {
    /// Personal Access Token or GitHub App installation token.
    /// Required for outbound API calls. Can also be set via `GITHUB_TOKEN`.
    pub github_token: String,

    /// Webhook secret used to verify HMAC-SHA256 signatures on inbound payloads.
    /// Must match the secret configured in the GitHub webhook settings.
    /// Can also be set via `GITHUB_WEBHOOK_SECRET`.
    #[serde(default)]
    pub webhook_secret: Option<String>,

    /// Local address and port for the webhook HTTP server.
    /// Default: `127.0.0.1:9000`
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Skip webhook secret requirement for local development.
    /// When `false` (default), `webhook_secret` must be set or the server
    /// will refuse to start.
    #[serde(default)]
    pub insecure_dev_webhook: bool,

    /// Repositories to accept events from.
    /// Format: `["owner/repo", ...]`. Empty means accept all.
    #[serde(default)]
    pub repos: Vec<String>,

    /// GitHub event types to forward to the gateway.
    /// Defaults: issue_comment, issues, pull_request, pull_request_review_comment.
    #[serde(default = "default_events")]
    pub on_events: Vec<String>,

    /// WebSocket URL of the brainwires-gateway.
    #[serde(default = "default_gateway_url")]
    pub gateway_url: String,

    /// Optional authentication token for the gateway handshake.
    #[serde(default)]
    pub gateway_token: Option<String>,

    /// GitHub API base URL (override for GitHub Enterprise).
    #[serde(default = "default_api_url")]
    pub api_url: String,
}

fn default_listen_addr() -> String {
    "127.0.0.1:9000".to_string()
}

fn default_gateway_url() -> String {
    "ws://127.0.0.1:18789/ws".to_string()
}

fn default_api_url() -> String {
    "https://api.github.com".to_string()
}

fn default_events() -> Vec<String> {
    vec![
        "issue_comment".to_string(),
        "issues".to_string(),
        "pull_request".to_string(),
        "pull_request_review_comment".to_string(),
    ]
}

impl Default for GitHubConfig {
    fn default() -> Self {
        Self {
            github_token: String::new(),
            webhook_secret: None,
            insecure_dev_webhook: false,
            listen_addr: default_listen_addr(),
            repos: Vec::new(),
            on_events: default_events(),
            gateway_url: default_gateway_url(),
            gateway_token: None,
            api_url: default_api_url(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = GitHubConfig::default();
        assert_eq!(cfg.listen_addr, "127.0.0.1:9000");
        assert_eq!(cfg.gateway_url, "ws://127.0.0.1:18789/ws");
        assert_eq!(cfg.api_url, "https://api.github.com");
        assert!(cfg.repos.is_empty());
        assert!(!cfg.on_events.is_empty());
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = GitHubConfig {
            github_token: "ghp_test".to_string(),
            webhook_secret: Some("my-secret".to_string()),
            listen_addr: "127.0.0.1:8080".to_string(),
            repos: vec!["owner/repo".to_string()],
            on_events: vec!["issues".to_string()],
            gateway_url: "ws://gw:18789/ws".to_string(),
            gateway_token: Some("gw-token".to_string()),
            api_url: "https://github.example.com/api/v3".to_string(),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: GitHubConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.github_token, "ghp_test");
        assert_eq!(parsed.webhook_secret.as_deref(), Some("my-secret"));
        assert_eq!(parsed.repos, vec!["owner/repo"]);
    }

    #[test]
    fn repo_filter_logic() {
        let cfg = GitHubConfig {
            repos: vec!["octocat/hello-world".to_string()],
            ..Default::default()
        };
        assert!(cfg.repos.contains(&"octocat/hello-world".to_string()));
        assert!(!cfg.repos.contains(&"octocat/other".to_string()));
    }
}
