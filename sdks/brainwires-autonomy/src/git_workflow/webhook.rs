//! Axum webhook server with HMAC signature verification.

use std::collections::HashSet;
use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use tokio::sync::{RwLock, mpsc};

use super::forge::RepoRef;
use super::trigger::WorkflowEvent;

/// Webhook server state.
struct WebhookState {
    secret: Option<String>,
    tx: mpsc::Sender<WorkflowEvent>,
    /// Track issues currently being investigated to prevent duplicates.
    active_investigations: RwLock<HashSet<String>>,
}

/// Axum-based webhook server for receiving Git forge events.
///
/// Verifies HMAC signatures (SHA-256 or SHA-1), parses GitHub event payloads,
/// deduplicates active investigations, and forwards events to the pipeline.
pub struct WebhookServer {
    listen_addr: String,
    port: u16,
    secret: Option<String>,
}

impl WebhookServer {
    /// Create a new webhook server with the given listen address, port, and optional secret.
    pub fn new(listen_addr: String, port: u16, secret: Option<String>) -> Self {
        Self {
            listen_addr,
            port,
            secret,
        }
    }

    /// Start the webhook server and emit events to the given channel.
    pub async fn run(self, tx: mpsc::Sender<WorkflowEvent>) -> anyhow::Result<()> {
        let state = Arc::new(WebhookState {
            secret: self.secret,
            tx,
            active_investigations: RwLock::new(HashSet::new()),
        });

        let app = Router::new()
            .route("/health", get(health))
            .route("/webhook", post(handle_webhook))
            .with_state(state);

        let addr = format!("{}:{}", self.listen_addr, self.port);
        tracing::info!("Webhook server listening on {addr}");

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn handle_webhook(
    State(state): State<Arc<WebhookState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Verify HMAC signature if secret is configured
    if let Some(ref secret) = state.secret {
        let signature = headers
            .get("x-hub-signature-256")
            .or_else(|| headers.get("x-hub-signature"))
            .and_then(|v| v.to_str().ok());

        match signature {
            Some(sig) => {
                if !verify_signature(secret, &body, sig) {
                    tracing::warn!("Webhook signature verification failed");
                    return StatusCode::UNAUTHORIZED;
                }
            }
            None => {
                tracing::warn!("Webhook missing signature header");
                return StatusCode::UNAUTHORIZED;
            }
        }
    }

    // Parse event type
    let event_type = headers
        .get("x-github-event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");

    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Failed to parse webhook payload: {e}");
            return StatusCode::BAD_REQUEST;
        }
    };

    let event = match parse_github_event(event_type, &payload) {
        Some(e) => e,
        None => {
            tracing::debug!("Ignoring unhandled event type: {event_type}");
            return StatusCode::OK;
        }
    };

    // Check for duplicate investigations
    let key = event_key(&event);
    if let Some(key) = &key {
        let active = state.active_investigations.read().await;
        if active.contains(key) {
            tracing::info!("Skipping duplicate investigation for {key}");
            return StatusCode::OK;
        }
    }

    if let Some(key) = &key {
        state
            .active_investigations
            .write()
            .await
            .insert(key.clone());
    }

    if let Err(e) = state.tx.send(event).await {
        tracing::error!("Failed to send webhook event: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    StatusCode::OK
}

fn verify_signature(secret: &str, body: &[u8], signature: &str) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    use subtle::ConstantTimeEq;

    // Try SHA-256 first (x-hub-signature-256)
    if let Some(hex_sig) = signature.strip_prefix("sha256=") {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(body);
        let expected = mac.finalize().into_bytes();
        let expected_hex = hex::encode(expected);
        return expected_hex.as_bytes().ct_eq(hex_sig.as_bytes()).into();
    }

    // Fallback to SHA-1 (x-hub-signature)
    if let Some(hex_sig) = signature.strip_prefix("sha1=") {
        use sha1::Sha1;
        let mut mac =
            Hmac::<Sha1>::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
        mac.update(body);
        let expected = mac.finalize().into_bytes();
        let expected_hex = hex::encode(expected);
        return expected_hex.as_bytes().ct_eq(hex_sig.as_bytes()).into();
    }

    false
}

fn parse_github_event(event_type: &str, payload: &serde_json::Value) -> Option<WorkflowEvent> {
    let repo = RepoRef {
        owner: payload["repository"]["owner"]["login"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        name: payload["repository"]["name"]
            .as_str()
            .unwrap_or("")
            .to_string(),
    };

    match event_type {
        "issues" if payload["action"].as_str() == Some("opened") => {
            let issue = parse_issue(payload)?;
            Some(WorkflowEvent::IssueOpened { issue, repo })
        }
        "issue_comment" if payload["action"].as_str() == Some("created") => {
            let issue = parse_issue(&payload["issue"])?;
            let comment = super::forge::Comment {
                id: payload["comment"]["id"].to_string(),
                author: payload["comment"]["user"]["login"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                body: payload["comment"]["body"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
            };
            Some(WorkflowEvent::IssueCommented {
                issue,
                comment,
                repo,
            })
        }
        "push" => {
            let branch = payload["ref"]
                .as_str()
                .unwrap_or("")
                .trim_start_matches("refs/heads/")
                .to_string();
            let commits = payload["commits"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|c| super::forge::CommitRef {
                            sha: c["id"].as_str().unwrap_or("").to_string(),
                            message: c["message"].as_str().unwrap_or("").to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Some(WorkflowEvent::PushReceived {
                branch,
                commits,
                repo,
            })
        }
        _ => None,
    }
}

fn parse_issue(payload: &serde_json::Value) -> Option<super::forge::Issue> {
    Some(super::forge::Issue {
        id: payload["id"].to_string(),
        number: payload["number"].as_u64()?,
        title: payload["title"].as_str().unwrap_or("").to_string(),
        body: payload["body"].as_str().unwrap_or("").to_string(),
        labels: payload["labels"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|l| l["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        author: payload["user"]["login"].as_str().unwrap_or("").to_string(),
        url: payload["html_url"].as_str().unwrap_or("").to_string(),
    })
}

fn event_key(event: &WorkflowEvent) -> Option<String> {
    match event {
        WorkflowEvent::IssueOpened { issue, repo } => {
            Some(format!("{}#{}", repo.full_name(), issue.number))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── verify_signature ────────────────────────────────────────────────────

    #[cfg(feature = "webhook")]
    fn make_signature(secret: &str, body: &[u8]) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        let expected = mac.finalize().into_bytes();
        format!("sha256={}", hex::encode(expected))
    }

    #[cfg(feature = "webhook")]
    #[test]
    fn valid_sha256_signature_passes() {
        let body = b"hello world";
        let secret = "mysecret";
        let sig = make_signature(secret, body);
        assert!(verify_signature(secret, body, &sig));
    }

    #[cfg(feature = "webhook")]
    #[test]
    fn wrong_signature_fails() {
        let body = b"hello world";
        assert!(!verify_signature("secret", body, "sha256=badhex"));
    }

    #[cfg(feature = "webhook")]
    #[test]
    fn wrong_body_fails() {
        let body = b"hello world";
        let secret = "mysecret";
        let sig = make_signature(secret, b"other body");
        assert!(!verify_signature(secret, body, &sig));
    }

    #[cfg(feature = "webhook")]
    #[test]
    fn wrong_secret_fails() {
        let body = b"hello world";
        let sig = make_signature("correct-secret", body);
        assert!(!verify_signature("wrong-secret", body, &sig));
    }

    #[cfg(feature = "webhook")]
    #[test]
    fn unknown_prefix_fails() {
        assert!(!verify_signature("secret", b"body", "md5=abc123"));
    }

    // ── parse_github_event ─────────────────────────────────────────────────

    // Note: parse_github_event for "issues" calls parse_issue(payload) directly,
    // so issue fields must be at the top level of the payload (not nested under "issue").
    fn issues_opened_payload(
        number: u64,
        title: &str,
        owner: &str,
        repo: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "action": "opened",
            "id": number,
            "number": number,
            "title": title,
            "body": "body text",
            "labels": [],
            "user": {"login": "alice"},
            "html_url": "https://github.com/org/repo/issues/1",
            "repository": {
                "name": repo,
                "owner": {"login": owner},
            }
        })
    }

    #[test]
    fn parse_issues_opened_event() {
        let payload = issues_opened_payload(42, "Fix login bug", "myorg", "myrepo");
        let event = parse_github_event("issues", &payload).unwrap();
        match event {
            WorkflowEvent::IssueOpened { issue, repo } => {
                assert_eq!(issue.number, 42);
                assert_eq!(issue.title, "Fix login bug");
                assert_eq!(repo.owner, "myorg");
                assert_eq!(repo.name, "myrepo");
            }
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn parse_issues_non_opened_action_returns_none() {
        let mut payload = issues_opened_payload(1, "title", "owner", "repo");
        payload["action"] = serde_json::json!("closed");
        assert!(parse_github_event("issues", &payload).is_none());
    }

    #[test]
    fn parse_push_event() {
        let payload = serde_json::json!({
            "ref": "refs/heads/main",
            "commits": [
                {"id": "abc123", "message": "first commit"},
                {"id": "def456", "message": "second commit"},
            ],
            "repository": {
                "name": "myrepo",
                "owner": {"login": "myorg"},
            }
        });
        let event = parse_github_event("push", &payload).unwrap();
        match event {
            WorkflowEvent::PushReceived {
                branch,
                commits,
                repo,
            } => {
                assert_eq!(branch, "main");
                assert_eq!(commits.len(), 2);
                assert_eq!(commits[0].sha, "abc123");
                assert_eq!(repo.name, "myrepo");
            }
            _ => panic!("wrong event type"),
        }
    }

    #[test]
    fn parse_unknown_event_returns_none() {
        let payload = serde_json::json!({
            "repository": {"name": "r", "owner": {"login": "o"}}
        });
        assert!(parse_github_event("pull_request", &payload).is_none());
    }

    #[test]
    fn parse_issue_comment_event() {
        let payload = serde_json::json!({
            "action": "created",
            "issue": {
                "id": 1,
                "number": 5,
                "title": "Bug",
                "body": "desc",
                "labels": [],
                "user": {"login": "bob"},
                "html_url": "https://github.com/org/repo/issues/5",
            },
            "comment": {
                "id": 100,
                "user": {"login": "carol"},
                "body": "This is a comment",
            },
            "repository": {
                "name": "repo",
                "owner": {"login": "org"},
            }
        });
        let event = parse_github_event("issue_comment", &payload).unwrap();
        assert!(matches!(event, WorkflowEvent::IssueCommented { .. }));
    }

    // ── event_key ─────────────────────────────────────────────────────────

    #[test]
    fn event_key_for_issue_opened() {
        let event = WorkflowEvent::IssueOpened {
            issue: super::super::forge::Issue {
                id: "1".to_string(),
                number: 7,
                title: "title".to_string(),
                body: String::new(),
                labels: vec![],
                author: "alice".to_string(),
                url: String::new(),
            },
            repo: super::super::forge::RepoRef {
                owner: "org".to_string(),
                name: "repo".to_string(),
            },
        };
        assert_eq!(event_key(&event), Some("org/repo#7".to_string()));
    }

    #[test]
    fn event_key_for_non_issue_returns_none() {
        let event = WorkflowEvent::Manual {
            description: "test".to_string(),
            repo: super::super::forge::RepoRef {
                owner: "o".to_string(),
                name: "r".to_string(),
            },
        };
        assert_eq!(event_key(&event), None);
    }
}
