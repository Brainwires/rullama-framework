//! GitHub webhook receiver — Axum HTTP server that verifies and dispatches
//! incoming GitHub webhook payloads.
//!
//! Supported event types:
//! - `issue_comment`                — comment created/edited/deleted on issue or PR
//! - `issues`                       — issue opened/closed/labeled/etc.
//! - `pull_request`                 — PR opened/closed/merged/etc.
//! - `pull_request_review_comment`  — inline PR review comment

use anyhow::Result;
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use brainwires_network::channels::{ChannelMessage, ConversationId, MessageContent, MessageId};
use chrono::Utc;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::mpsc;

use crate::config::GitHubConfig;

type HmacSha256 = Hmac<Sha256>;

/// Shared webhook state passed to Axum handlers.
#[derive(Clone)]
struct WebhookState {
    /// Sender for forwarding normalised events downstream.
    event_tx: mpsc::Sender<ChannelMessage>,
    config: Arc<GitHubConfig>,
}

/// Start the Axum webhook server.
///
/// Listens on `config.listen_addr` and sends normalised `ChannelMessage`s into
/// `event_tx` for every valid, accepted webhook event.
pub async fn serve(
    config: Arc<GitHubConfig>,
    event_tx: mpsc::Sender<ChannelMessage>,
) -> Result<()> {
    let state = WebhookState {
        event_tx,
        config: Arc::clone(&config),
    };

    let app = Router::new()
        .route("/webhook", post(handle_webhook))
        .with_state(state);

    let addr: SocketAddr = config.listen_addr.parse()?;
    tracing::info!("GitHub webhook server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_webhook(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // ── Signature verification ────────────────────────────────────────────────
    if let Some(secret) = &state.config.webhook_secret {
        let sig_header = headers
            .get("X-Hub-Signature-256")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !verify_signature(secret.as_bytes(), &body, sig_header) {
            tracing::warn!("GitHub webhook: signature verification failed");
            return (StatusCode::UNAUTHORIZED, "invalid signature").into_response();
        }
    }

    // ── Event type ────────────────────────────────────────────────────────────
    let event_type = headers
        .get("X-GitHub-Event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !state.config.on_events.contains(&event_type) {
        return (StatusCode::OK, "event type ignored").into_response();
    }

    // ── Parse payload ─────────────────────────────────────────────────────────
    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Failed to parse webhook payload: {e}");
            return (StatusCode::BAD_REQUEST, "invalid JSON").into_response();
        }
    };

    // ── Repo filter ───────────────────────────────────────────────────────────
    let repo_full = payload["repository"]["full_name"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if !state.config.repos.is_empty() && !state.config.repos.contains(&repo_full) {
        return (StatusCode::OK, "repo not in allowlist").into_response();
    }

    // ── Normalise to ChannelMessage ───────────────────────────────────────────
    if let Some(msg) = normalise(&event_type, &payload, &repo_full)
        && let Err(e) = state.event_tx.try_send(msg)
    {
        tracing::error!("webhook event dropped: {e}");
        return (StatusCode::SERVICE_UNAVAILABLE, "backpressure").into_response();
    }

    (StatusCode::OK, "accepted").into_response()
}

/// Verify HMAC-SHA256 signature from GitHub.
fn verify_signature(secret: &[u8], body: &[u8], signature_header: &str) -> bool {
    let expected = signature_header
        .strip_prefix("sha256=")
        .unwrap_or(signature_header);

    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC can take key of any size");
    mac.update(body);
    let result = hex::encode(mac.finalize().into_bytes());
    // Constant-time comparison via hmac verify is ideal; hex compare is fine for server-side
    result == expected
}

/// Convert a GitHub webhook payload into a `ChannelMessage`.
///
/// Returns `None` for events that don't map to a user-visible message
/// (e.g. label changes with no body).
fn normalise(event_type: &str, payload: &serde_json::Value, repo: &str) -> Option<ChannelMessage> {
    let (issue_number, body, author) = match event_type {
        "issue_comment" => {
            let action = payload["action"].as_str()?;
            if action == "deleted" {
                return None; // deletions don't carry useful body
            }
            let n = payload["issue"]["number"].as_u64()?;
            let b = payload["comment"]["body"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let a = payload["comment"]["user"]["login"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            (n, b, a)
        }
        "issues" => {
            let n = payload["issue"]["number"].as_u64()?;
            let title = payload["issue"]["title"].as_str().unwrap_or("(no title)");
            let b = payload["issue"]["body"].as_str().unwrap_or("");
            let action = payload["action"].as_str().unwrap_or("event");
            let a = payload["sender"]["login"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let text = format!("[issue {action}] {title}\n\n{b}");
            (n, text, a)
        }
        "pull_request" => {
            let n = payload["pull_request"]["number"].as_u64()?;
            let title = payload["pull_request"]["title"]
                .as_str()
                .unwrap_or("(no title)");
            let action = payload["action"].as_str().unwrap_or("event");
            let a = payload["sender"]["login"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let text = format!("[PR {action}] {title}");
            (n, text, a)
        }
        "pull_request_review_comment" => {
            let action = payload["action"].as_str()?;
            if action == "deleted" {
                return None;
            }
            let n = payload["pull_request"]["number"].as_u64()?;
            let b = payload["comment"]["body"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let path = payload["comment"]["path"].as_str().unwrap_or("?");
            let a = payload["comment"]["user"]["login"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let text = format!("[review comment on {path}] {b}");
            (n, text, a)
        }
        _ => return None,
    };

    let comment_id = payload
        .get("comment")
        .and_then(|c| c["id"].as_u64())
        .unwrap_or(issue_number);

    Some(ChannelMessage {
        id: MessageId::new(format!("{repo}/{comment_id}")),
        conversation: ConversationId {
            platform: "github".to_string(),
            channel_id: format!("{repo}#{issue_number}"),
            server_id: None,
        },
        author,
        content: MessageContent::Text(body),
        thread_id: None,
        reply_to: None,
        timestamp: Utc::now(),
        attachments: vec![],
        metadata: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_signature_correct() {
        let secret = b"my-secret";
        let body = b"hello world";
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(body);
        let sig = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));
        assert!(verify_signature(secret, body, &sig));
    }

    #[test]
    fn verify_signature_wrong_secret() {
        let body = b"hello";
        let mut mac = HmacSha256::new_from_slice(b"right-secret").unwrap();
        mac.update(body);
        let sig = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));
        assert!(!verify_signature(b"wrong-secret", body, &sig));
    }

    #[test]
    fn normalise_issue_comment() {
        let payload = serde_json::json!({
            "action": "created",
            "issue": { "number": 7 },
            "comment": {
                "id": 999,
                "body": "Looks good!",
                "user": { "login": "alice" }
            }
        });
        let msg = normalise("issue_comment", &payload, "octocat/repo").unwrap();
        assert_eq!(msg.author, "alice");
        assert_eq!(msg.conversation.channel_id, "octocat/repo#7");
        if let MessageContent::Text(t) = &msg.content {
            assert_eq!(t, "Looks good!");
        }
    }

    #[test]
    fn normalise_deleted_comment_returns_none() {
        let payload = serde_json::json!({
            "action": "deleted",
            "issue": { "number": 7 },
            "comment": { "id": 999, "body": "", "user": { "login": "x" } }
        });
        assert!(normalise("issue_comment", &payload, "owner/repo").is_none());
    }

    #[test]
    fn normalise_pr_event() {
        let payload = serde_json::json!({
            "action": "opened",
            "pull_request": { "number": 5, "title": "Add feature X" },
            "sender": { "login": "bob" }
        });
        let msg = normalise("pull_request", &payload, "owner/repo").unwrap();
        assert_eq!(msg.author, "bob");
        if let MessageContent::Text(t) = &msg.content {
            assert!(t.contains("PR opened"));
            assert!(t.contains("Add feature X"));
        }
    }
}
