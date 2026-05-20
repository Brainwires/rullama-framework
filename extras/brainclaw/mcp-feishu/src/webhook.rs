//! HTTPS webhook receiver for Feishu / Lark events.
//!
//! Feishu signs events with a custom HMAC-SHA256 scheme:
//!
//! ```text
//! signature = base64( HMAC-SHA256( timestamp || nonce || body, verification_token ) )
//! ```
//!
//! sent as `X-Lark-Signature` along with `X-Lark-Request-Timestamp`
//! and `X-Lark-Request-Nonce` headers. A valid signature is required
//! for all non-`url_verification` events; verification failures
//! return 401.
//!
//! The Feishu `url_verification` handshake (an initial challenge) is
//! answered without a cached session, as the spec requires.

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

use crate::feishu::{IngressEvent, parse_event};

/// Axum shared state.
#[derive(Clone)]
pub struct WebhookState {
    /// Verification token (HMAC key).
    pub verification_token: Arc<String>,
    /// Outbound event sink.
    pub event_tx: mpsc::Sender<ChannelEvent>,
}

impl WebhookState {
    /// Construct new state.
    pub fn new(
        verification_token: impl Into<String>,
        event_tx: mpsc::Sender<ChannelEvent>,
    ) -> Self {
        Self {
            verification_token: Arc::new(verification_token.into()),
            event_tx,
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
    tracing::info!(%listen_addr, "Feishu webhook listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_webhook(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Parse body first so we can detect the `url_verification` handshake
    // even if a spec-conformant signature is not yet available (Feishu
    // sometimes sends the initial challenge unsigned; the docs are
    // ambiguous on this, so we accept either path).
    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "feishu webhook: invalid JSON");
            return (StatusCode::BAD_REQUEST, "invalid json").into_response();
        }
    };

    // url_verification handshake.
    if let Some(challenge) = extract_challenge(&payload) {
        tracing::info!("feishu url_verification — responding with challenge");
        return axum::Json(serde_json::json!({ "challenge": challenge })).into_response();
    }

    // Every other event must be signed.
    let ts = headers
        .get("x-lark-request-timestamp")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let nonce = headers
        .get("x-lark-request-nonce")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let sig = headers
        .get("x-lark-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ts.is_empty() || nonce.is_empty() || sig.is_empty() {
        tracing::warn!("feishu webhook: missing signature headers");
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }
    if !verify_signature(&state.verification_token, ts, nonce, &body, sig) {
        tracing::warn!("feishu webhook: signature mismatch");
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    if let Err(e) = dispatch(&state, &payload).await {
        tracing::error!(error = %e, "feishu dispatch failed");
    }
    (StatusCode::OK, "ok").into_response()
}

/// Pull the `challenge` field out of a `url_verification` envelope. The
/// payload may arrive in either the v1 shape (`{"type":"url_verification","challenge":...}`)
/// or wrapped under a v2 schema — we handle both.
pub fn extract_challenge(payload: &serde_json::Value) -> Option<String> {
    if payload.get("type").and_then(|v| v.as_str()) == Some("url_verification") {
        return payload
            .get("challenge")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }
    // v2: header.event_type == "url_verification" with challenge under event.
    let et = payload
        .get("header")
        .and_then(|h| h.get("event_type"))
        .and_then(|v| v.as_str());
    if et == Some("url_verification") {
        return payload
            .get("event")
            .and_then(|e| e.get("challenge"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }
    None
}

/// Verify a Feishu webhook signature.
pub fn verify_signature(
    verification_token: &str,
    timestamp: &str,
    nonce: &str,
    body: &[u8],
    signature_b64: &str,
) -> bool {
    let expected = sign(verification_token, timestamp, nonce, body);
    let Ok(got) = B64.decode(signature_b64.trim()) else {
        return false;
    };
    let exp_bytes = match B64.decode(&expected) {
        Ok(b) => b,
        Err(_) => return false,
    };
    if got.len() != exp_bytes.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in got.iter().zip(exp_bytes.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// Compute the signature over (timestamp || nonce || body) using
/// `verification_token` as the HMAC key. Exposed so tests can generate
/// valid signatures.
pub fn sign(verification_token: &str, timestamp: &str, nonce: &str, body: &[u8]) -> String {
    type H = Hmac<Sha256>;
    let mut mac = H::new_from_slice(verification_token.as_bytes()).expect("hmac key");
    mac.update(timestamp.as_bytes());
    mac.update(nonce.as_bytes());
    mac.update(body);
    B64.encode(mac.finalize().into_bytes())
}

/// Dispatch a verified, parsed payload — forward messages, log drops.
pub async fn dispatch(state: &WebhookState, payload: &serde_json::Value) -> Result<()> {
    let ev = parse_event(payload)?;
    match ev {
        IngressEvent::Message(m) => {
            let m = *m;
            audit_log(&m);
            state
                .event_tx
                .send(ChannelEvent::MessageReceived(m))
                .await
                .context("forward feishu event")
        }
        IngressEvent::Dropped { event_type } => {
            tracing::debug!(%event_type, "feishu event ignored");
            Ok(())
        }
    }
}

fn audit_log(msg: &brainwires_network::channels::ChannelMessage) {
    let user_digest = hashed_user(&msg.author);
    let len = match &msg.content {
        brainwires_network::channels::MessageContent::Text(t) => t.len(),
        brainwires_network::channels::MessageContent::RichText { markdown, .. } => markdown.len(),
        _ => 0,
    };
    tracing::info!(channel = "feishu", user = %user_digest, message_len = len, "forwarded");
}

fn hashed_user(author: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(author.as_bytes());
    hex::encode(&h.finalize()[..6])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sign_and_verify_roundtrip() {
        let token = "verification-token-placeholder";
        let body = br#"{"schema":"2.0"}"#;
        let sig = sign(token, "1700000000", "nonce-42", body);
        assert!(verify_signature(
            token,
            "1700000000",
            "nonce-42",
            body,
            &sig
        ));
    }

    #[test]
    fn tampered_timestamp_rejected() {
        let token = "t";
        let sig = sign(token, "1", "n", b"x");
        assert!(!verify_signature(token, "2", "n", b"x", &sig));
    }

    #[test]
    fn tampered_body_rejected() {
        let token = "t";
        let sig = sign(token, "1", "n", b"x");
        assert!(!verify_signature(token, "1", "n", b"y", &sig));
    }

    #[test]
    fn url_verification_v1_is_extracted() {
        let v = json!({"type":"url_verification","challenge":"abc123"});
        assert_eq!(extract_challenge(&v).as_deref(), Some("abc123"));
    }

    #[test]
    fn url_verification_v2_is_extracted() {
        let v = json!({
            "schema":"2.0",
            "header":{"event_type":"url_verification"},
            "event":{"challenge":"zzz"}
        });
        assert_eq!(extract_challenge(&v).as_deref(), Some("zzz"));
    }

    #[test]
    fn not_a_challenge_returns_none() {
        let v = json!({"header":{"event_type":"im.message.receive_v1"}});
        assert!(extract_challenge(&v).is_none());
    }
}
