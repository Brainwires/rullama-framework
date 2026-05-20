//! HTTPS webhook server for Bot Framework activities.

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
use tokio::sync::mpsc;

use brainwires_network::channels::ChannelEvent;

use crate::jwt::BotFrameworkVerifier;
use crate::teams::{ActivityEvent, ServiceUrlStore, parse_activity};

/// Axum shared state.
#[derive(Clone)]
pub struct WebhookState {
    /// Verifier for Bot Framework JWTs.
    pub verifier: Arc<BotFrameworkVerifier>,
    /// Service URL cache shared with the egress channel.
    pub service_urls: Arc<ServiceUrlStore>,
    /// Event sink to the gateway client loop.
    pub event_tx: mpsc::Sender<ChannelEvent>,
}

/// Spawn the Axum server on `listen_addr`.
pub async fn serve(state: WebhookState, listen_addr: &str) -> Result<()> {
    let app = Router::new()
        .route("/api/messages", post(handle_messages))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;
    tracing::info!(%listen_addr, "Teams webhook listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_messages(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) else {
        tracing::warn!("teams webhook: missing Authorization");
        return (StatusCode::UNAUTHORIZED, "missing Authorization").into_response();
    };

    if let Err(e) = state.verifier.verify(auth).await {
        tracing::warn!(error = %e, "teams webhook: JWT verification failed");
        return (StatusCode::UNAUTHORIZED, "jwt verification failed").into_response();
    }

    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "teams webhook: invalid JSON");
            return (StatusCode::BAD_REQUEST, "invalid json").into_response();
        }
    };

    if let Err(e) = dispatch(&state, &payload).await {
        tracing::error!(error = %e, "teams dispatch");
    }
    (StatusCode::OK, "ok").into_response()
}

/// Route a parsed activity to the gateway (or record + drop).
pub async fn dispatch(state: &WebhookState, payload: &serde_json::Value) -> Result<()> {
    let event = parse_activity(payload)?;
    match event {
        ActivityEvent::Message(msg) => {
            if let Some(url) = msg.metadata.get("teams.service_url").cloned() {
                state
                    .service_urls
                    .record(&msg.conversation.channel_id, &url);
            }
            audit_log(&msg);
            state
                .event_tx
                .send(ChannelEvent::MessageReceived(msg))
                .await
                .context("forward message to gateway")
        }
        ActivityEvent::ConversationUpdate {
            conversation_id,
            service_url,
        } => {
            if !conversation_id.is_empty() && !service_url.is_empty() {
                state.service_urls.record(&conversation_id, &service_url);
            }
            tracing::info!(conversation = %conversation_id, "teams conversation update (not forwarded)");
            Ok(())
        }
        ActivityEvent::Ignore { activity_type } => {
            tracing::debug!(%activity_type, "teams activity ignored");
            Ok(())
        }
    }
}

fn audit_log(msg: &brainwires_network::channels::ChannelMessage) {
    use sha2::{Digest, Sha256};
    let user = msg
        .metadata
        .get("teams.user_id")
        .map(String::as_str)
        .unwrap_or(&msg.author);
    let mut h = Sha256::new();
    h.update(user.as_bytes());
    let out = h.finalize();
    let digest = hex::encode(&out[..6]);
    let len = match &msg.content {
        brainwires_network::channels::MessageContent::Text(t) => t.len(),
        brainwires_network::channels::MessageContent::RichText { markdown, .. } => markdown.len(),
        _ => 0,
    };
    tracing::info!(
        channel = "teams",
        user = %digest,
        message_len = len,
        "forwarded"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mk_state(tx: mpsc::Sender<ChannelEvent>) -> WebhookState {
        WebhookState {
            verifier: Arc::new(BotFrameworkVerifier::new("aud")),
            service_urls: Arc::new(ServiceUrlStore::new()),
            event_tx: tx,
        }
    }

    #[tokio::test]
    async fn dispatch_records_service_url_and_forwards() {
        let (tx, mut rx) = mpsc::channel(4);
        let state = mk_state(tx);
        let payload = json!({
            "type": "message",
            "id": "a1",
            "serviceUrl": "https://svc.example/",
            "conversation": { "id": "c1" },
            "from": { "id": "29:x", "name": "Alice" },
            "text": "hi",
            "timestamp": "2025-02-01T00:00:00Z",
        });
        dispatch(&state, &payload).await.unwrap();
        assert_eq!(
            state.service_urls.get("c1").as_deref(),
            Some("https://svc.example/")
        );
        let evt = rx.recv().await.unwrap();
        match evt {
            ChannelEvent::MessageReceived(m) => assert_eq!(m.author, "Alice"),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn dispatch_conversation_update_records_only() {
        let (tx, mut rx) = mpsc::channel(4);
        let state = mk_state(tx);
        let payload = json!({
            "type": "conversationUpdate",
            "conversation": { "id": "c2" },
            "serviceUrl": "https://svc2.example/",
        });
        dispatch(&state, &payload).await.unwrap();
        assert_eq!(
            state.service_urls.get("c2").as_deref(),
            Some("https://svc2.example/")
        );
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn dispatch_typing_is_dropped() {
        let (tx, mut rx) = mpsc::channel(4);
        let state = mk_state(tx);
        let payload = json!({ "type": "typing", "conversation": { "id": "c3" } });
        dispatch(&state, &payload).await.unwrap();
        assert!(rx.try_recv().is_err());
    }
}
