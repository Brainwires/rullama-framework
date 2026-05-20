//! HTTPS webhook receiver for LINE Messaging API.
//!
//! LINE signs each delivery with `X-Line-Signature: <base64(HMAC-SHA256(body, channel_secret))>`.
//! We reject any request that fails verification with 401. The
//! signature itself is never logged.

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::sync::mpsc;

use brainwires_network::channels::ChannelEvent;

use crate::line::{IngressEvent, ReplyTokenStore, parse_events};

/// Axum shared state.
#[derive(Clone)]
pub struct WebhookState {
    /// Channel secret used for HMAC verification.
    pub channel_secret: Arc<String>,
    /// Outbound event sink.
    pub event_tx: mpsc::Sender<ChannelEvent>,
    /// Store for reply tokens harvested from inbound events.
    pub reply_tokens: Arc<ReplyTokenStore>,
}

impl WebhookState {
    /// Construct a new webhook state.
    pub fn new(
        channel_secret: impl Into<String>,
        event_tx: mpsc::Sender<ChannelEvent>,
        reply_tokens: Arc<ReplyTokenStore>,
    ) -> Self {
        Self {
            channel_secret: Arc::new(channel_secret.into()),
            event_tx,
            reply_tokens,
        }
    }
}

/// Start the Axum webhook server.
pub async fn serve(state: WebhookState, listen_addr: &str) -> Result<()> {
    let app = Router::new()
        .route("/webhook", post(handle_webhook))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;
    tracing::info!(%listen_addr, "LINE webhook listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_webhook(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let sig = match headers
        .get("x-line-signature")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => {
            tracing::warn!("line webhook: missing X-Line-Signature");
            return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
        }
    };

    if !verify_signature(&state.channel_secret, &body, &sig) {
        tracing::warn!("line webhook: signature mismatch");
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "line webhook: invalid json");
            return (StatusCode::BAD_REQUEST, "invalid json").into_response();
        }
    };

    if let Err(e) = dispatch(&state, &payload).await {
        tracing::error!(error = %e, "line dispatch failed");
    }
    (StatusCode::OK, "ok").into_response()
}

/// Verify a LINE webhook signature against a channel secret.
pub fn verify_signature(channel_secret: &str, body: &[u8], signature_b64: &str) -> bool {
    type H = Hmac<Sha256>;
    let Ok(mut mac) = H::new_from_slice(channel_secret.as_bytes()) else {
        return false;
    };
    mac.update(body);
    let expected = mac.finalize().into_bytes();
    let Ok(got) = B64.decode(signature_b64.trim()) else {
        return false;
    };
    if got.len() != expected.len() {
        return false;
    }
    // Constant-time compare.
    let mut diff = 0u8;
    for (a, b) in got.iter().zip(expected.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Parse the payload and forward interesting events + remember reply tokens.
pub async fn dispatch(state: &WebhookState, payload: &serde_json::Value) -> Result<()> {
    for ev in parse_events(payload) {
        match ev {
            IngressEvent::Message(channel, reply_token) => {
                let channel = *channel;
                if let Some(token) = reply_token {
                    state
                        .reply_tokens
                        .remember(&channel.conversation.channel_id, token);
                }
                audit_log(&channel);
                state
                    .event_tx
                    .send(ChannelEvent::MessageReceived(channel))
                    .await
                    .context("forward line event")?;
            }
            IngressEvent::Dropped { event_type } => {
                tracing::debug!(%event_type, "line event ignored");
            }
        }
    }
    Ok(())
}

fn audit_log(msg: &brainwires_network::channels::ChannelMessage) {
    let user_digest = hashed_user(&msg.author);
    let len = match &msg.content {
        brainwires_network::channels::MessageContent::Text(t) => t.len(),
        brainwires_network::channels::MessageContent::RichText { markdown, .. } => markdown.len(),
        _ => 0,
    };
    tracing::info!(channel = "line", user = %user_digest, message_len = len, "forwarded");
}

fn hashed_user(author: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(author.as_bytes());
    hex::encode(&h.finalize()[..6])
}

/// Compute a signature — shared helper so tests can produce valid ones.
pub fn sign_body(channel_secret: &str, body: &[u8]) -> String {
    type H = Hmac<Sha256>;
    let mut mac = H::new_from_slice(channel_secret.as_bytes()).expect("hmac");
    mac.update(body);
    B64.encode(mac.finalize().into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn signature_roundtrip_is_accepted() {
        let secret = "test-secret-value";
        let body = br#"{"events":[]}"#;
        let sig = sign_body(secret, body);
        assert!(verify_signature(secret, body, &sig));
    }

    #[test]
    fn tampered_body_is_rejected() {
        let secret = "s";
        let sig = sign_body(secret, b"hello");
        assert!(!verify_signature(secret, b"hello world", &sig));
    }

    #[test]
    fn wrong_secret_is_rejected() {
        let sig = sign_body("s1", b"hi");
        assert!(!verify_signature("s2", b"hi", &sig));
    }

    #[test]
    fn garbage_signature_returns_false() {
        assert!(!verify_signature("s", b"x", "not-base64@@"));
        assert!(!verify_signature("s", b"x", ""));
    }

    #[tokio::test]
    async fn dispatch_forwards_valid_message() {
        let (tx, mut rx) = mpsc::channel(4);
        let tokens = Arc::new(ReplyTokenStore::default());
        let state = WebhookState::new("s", tx, Arc::clone(&tokens));
        let payload = json!({
            "events": [{
                "type": "message",
                "replyToken": "rt",
                "source": {"userId": "U1"},
                "message": {"type":"text","id":"m1","text":"hi"}
            }]
        });
        dispatch(&state, &payload).await.unwrap();
        let got = rx.recv().await.unwrap();
        match got {
            ChannelEvent::MessageReceived(m) => assert_eq!(m.conversation.channel_id, "U1"),
            _ => panic!(),
        }
        // Reply token was cached.
        assert!(tokens.take_fresh("U1").is_some());
    }
}
