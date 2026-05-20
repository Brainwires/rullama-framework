//! WebChat channel — browser-based chat UI with JWT auth.
//!
//! This module provides two endpoints:
//!
//! 1. `GET /chat` — serves the legacy built-in static HTML chat UI.
//! 2. `GET /webchat/ws` — JWT-gated WebSocket endpoint used by the external
//!    `brainwires-webchat` Next.js application.
//!
//! # Protocol — `/webchat/ws`
//!
//! The client authenticates during the upgrade by supplying an HS256 JWT in
//! either the `token` query parameter or the `Sec-WebSocket-Protocol` header
//! (first subprotocol). The JWT's `sub` claim is used as the per-user
//! `user_id`; a channel session id of the form `webchat:<user_id>` is then
//! assigned.
//!
//! ## Client → Server (JSON text frames)
//!
//! ```json
//! { "type": "message", "content": "hello" }
//! { "type": "resume",  "session_id": "webchat:<user>" }
//! { "type": "typing",  "typing": true }
//! ```
//!
//! ## Server → Client (JSON text frames)
//!
//! ```json
//! { "type": "session", "id": "webchat:<user>" }
//! { "type": "chunk",   "content": "partial text" }
//! { "type": "message", "role": "assistant", "content": "final", "id": "uuid" }
//! { "type": "tool_use","name": "tool", "status": "start", "preview": "" }
//! { "type": "error",   "message": "reason" }
//! ```
//!
//! # History endpoint
//!
//! `POST /webchat/history/:session_id` (admin-auth-gated when an admin
//! token is configured) returns the last N assistant+user messages for the
//! given webchat session, used by the client to backfill on reconnect.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use chrono::Utc;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tokio::sync::mpsc;
use uuid::Uuid;

use brainwires_network::channels::ChannelCapabilities;
use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::identity::ConversationId;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};

use crate::channel_registry::ConnectedChannel;
use crate::state::AppState;

type HmacSha256 = Hmac<Sha256>;

/// The channel-type string used by WebChat connections in the `ChannelRegistry`.
pub const WEBCHAT_CHANNEL_TYPE: &str = "webchat";

/// Maximum size of an inbound WebSocket text frame we accept (bytes).
///
/// Larger frames are rejected with an `error` control frame. The default
/// matches the gateway's media attachment cap multiplied by a small factor
/// so control messages plus a base64-encoded small attachment fit.
const MAX_INBOUND_FRAME_BYTES: usize = 256 * 1024;

/// Capacity of the per-session outbound history ring buffer used by the
/// `POST /webchat/history/:session_id` endpoint.
const HISTORY_RING_CAPACITY: usize = 200;

/// Minimal HS256 JWT claims accepted by the webchat channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebChatClaims {
    /// Subject — the stable user identity used for session keying.
    pub sub: String,
    /// Expiry (Unix seconds). Required; webchat rejects tokens without `exp`.
    pub exp: u64,
    /// Optional issued-at (Unix seconds).
    #[serde(default)]
    pub iat: Option<u64>,
}

/// Verification outcome for a webchat bearer token.
#[derive(Debug, Clone)]
pub enum AuthVerdict {
    /// Token is valid; the authenticated principal id is returned.
    Authorized { user_id: String },
    /// Token is invalid, expired, malformed, or otherwise rejected.
    Rejected { reason: &'static str },
}

/// Trait for verifying webchat bearer tokens.
///
/// The default implementation ([`Hs256Verifier`]) validates an HS256 JWT
/// signed by the configured shared secret. Production deployments can
/// substitute an implementation that delegates to the
/// `brainwires-mcp-server` OAuth pipeline.
pub trait AuthVerifier: Send + Sync {
    /// Verify a bearer token. Returns [`AuthVerdict::Authorized`] for
    /// accepted tokens and [`AuthVerdict::Rejected`] otherwise.
    fn verify(&self, token: &str) -> AuthVerdict;
}

/// Default HS256 JWT verifier.
pub struct Hs256Verifier {
    secret: Vec<u8>,
}

impl Hs256Verifier {
    /// Construct a verifier from a raw HMAC secret.
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        Self {
            secret: secret.into(),
        }
    }
}

impl AuthVerifier for Hs256Verifier {
    fn verify(&self, token: &str) -> AuthVerdict {
        match decode_hs256_jwt(token, &self.secret) {
            Ok(claims) => AuthVerdict::Authorized {
                user_id: claims.sub,
            },
            Err(reason) => AuthVerdict::Rejected { reason },
        }
    }
}

/// Per-session history ring. Each send to the channel and each inbound
/// user message is appended so `/webchat/history/:session_id` can serve
/// a backfill.
#[derive(Debug, Default)]
pub struct WebChatHistory {
    sessions: DashMap<String, VecDeque<HistoryEntry>>,
}

/// A single message in the webchat history ring.
#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntry {
    /// `"user"` or `"assistant"`.
    pub role: &'static str,
    /// Message text content (never base64 / attachment bytes).
    pub content: String,
    /// Unix-seconds timestamp.
    pub timestamp: i64,
}

impl WebChatHistory {
    /// Construct an empty history store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry for the given session id.
    pub fn push(&self, session_id: &str, entry: HistoryEntry) {
        let mut buf = self.sessions.entry(session_id.to_string()).or_default();
        if buf.len() >= HISTORY_RING_CAPACITY {
            buf.pop_front();
        }
        buf.push_back(entry);
    }

    /// Copy the most recent `limit` entries for a session.
    pub fn recent(&self, session_id: &str, limit: usize) -> Vec<HistoryEntry> {
        self.sessions
            .get(session_id)
            .map(|buf| {
                let len = buf.len();
                let start = len.saturating_sub(limit);
                buf.iter().skip(start).cloned().collect()
            })
            .unwrap_or_default()
    }
}

/// Query parameters accepted on the `/webchat/ws` upgrade request.
#[derive(Debug, Deserialize)]
pub struct WebChatQuery {
    /// Bearer JWT — alternative to the `Sec-WebSocket-Protocol` header.
    #[serde(default)]
    pub token: Option<String>,
}

/// Request body for `POST /webchat/history/:session_id`.
#[derive(Debug, Deserialize, Default)]
pub struct HistoryRequest {
    /// Optional explicit limit; clamped to `session_history_limit` from
    /// [`crate::config::GatewayConfig::webchat_session_history_limit`].
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response body for `POST /webchat/history/:session_id`.
#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    session_id: String,
    messages: Vec<HistoryEntry>,
}

/// Serve the legacy static WebChat HTML page at `GET /chat`.
pub async fn serve_webchat_page() -> impl IntoResponse {
    Html(include_str!("../static/webchat.html"))
}

/// Serve the admin UI HTML page at `GET /admin/ui` (or configured admin path + `/ui`).
pub async fn serve_admin_ui_page() -> impl IntoResponse {
    Html(include_str!("../static/admin_ui.html"))
}

/// Handle the legacy unauthenticated `/chat/ws` WebSocket, used by the
/// built-in `/chat` static page. Emits raw [`ChannelEvent`] JSON just
/// like an external channel adapter, with no JWT gating.
pub async fn legacy_chat_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_legacy_chat_connection(socket, state))
        .into_response()
}

async fn handle_legacy_chat_connection(ws: WebSocket, state: AppState) {
    let channel_id = Uuid::new_v4();
    let user_id = Uuid::new_v4().to_string();

    tracing::info!(%channel_id, %user_id, "legacy /chat WebChat client connected");

    let (outbound_tx, mut outbound_rx) = mpsc::channel::<String>(256);

    let connected = ConnectedChannel {
        id: channel_id,
        channel_type: WEBCHAT_CHANNEL_TYPE.to_string(),
        capabilities: ChannelCapabilities::RICH_TEXT | ChannelCapabilities::TYPING_INDICATOR,
        connected_at: Utc::now(),
        last_heartbeat: Utc::now(),
        message_tx: outbound_tx,
    };
    state.channels.register(connected);

    let (mut ws_sender, mut ws_receiver) = ws.split();

    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            if ws_sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(result) = ws_receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                if let Ok(event) = serde_json::from_str::<ChannelEvent>(&text) {
                    let router = state.router.clone();
                    tokio::spawn(async move {
                        let _ = router.handle_inbound(channel_id, &event).await;
                    });
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(_)) => {
                state.channels.touch_heartbeat(&channel_id);
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    writer_handle.abort();
    state.channels.unregister(&channel_id);
}

/// Handle a WebSocket upgrade for the JWT-gated WebChat channel
/// at `GET /webchat/ws`.
///
/// Auth sequence:
/// 1. Extract token from `?token=` query or from `Sec-WebSocket-Protocol`.
/// 2. Verify via [`AppState::webchat_verifier`].
/// 3. On failure, return `401 Unauthorized` without upgrading.
/// 4. On success, upgrade and run the per-connection loop keyed on `sub`.
pub async fn webchat_ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    Query(params): Query<WebChatQuery>,
    State(state): State<AppState>,
) -> Response {
    // Extract bearer token from either source.
    let protocol_token = headers
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string());

    let token = match params.token.or(protocol_token) {
        Some(t) if !t.is_empty() => t,
        _ => {
            tracing::warn!("webchat: rejected upgrade — missing token");
            return (StatusCode::UNAUTHORIZED, "missing token").into_response();
        }
    };

    // Verify.
    let verifier = match &state.webchat_verifier {
        Some(v) => v.clone(),
        None => {
            tracing::error!("webchat: no verifier configured — rejecting connection");
            return (StatusCode::UNAUTHORIZED, "webchat disabled").into_response();
        }
    };

    let user_id = match verifier.verify(&token) {
        AuthVerdict::Authorized { user_id } => user_id,
        AuthVerdict::Rejected { reason } => {
            tracing::warn!(reason, "webchat: rejected upgrade — bad token");
            return (StatusCode::UNAUTHORIZED, reason).into_response();
        }
    };

    ws.on_upgrade(move |socket| handle_webchat_connection(socket, state, user_id))
        .into_response()
}

/// Run a single authenticated WebChat connection.
async fn handle_webchat_connection(ws: WebSocket, state: AppState, user_id: String) {
    let channel_id = Uuid::new_v4();
    let session_id = format!("webchat:{}", user_id);

    tracing::info!(%channel_id, %session_id, "WebChat client connected");

    // Outbound queue: everything destined for this browser funnels through here.
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<String>(256);

    // Bridge queue: we want to intercept outbound ChannelEvent frames that
    // `AgentInboundHandler` pushes onto our channel sender, translate them to
    // webchat wire frames, and also record them in the history ring.
    let (bridge_tx, mut bridge_rx) = mpsc::channel::<String>(256);

    let connected = ConnectedChannel {
        id: channel_id,
        channel_type: WEBCHAT_CHANNEL_TYPE.to_string(),
        capabilities: ChannelCapabilities::RICH_TEXT | ChannelCapabilities::TYPING_INDICATOR,
        connected_at: Utc::now(),
        last_heartbeat: Utc::now(),
        message_tx: bridge_tx.clone(),
    };
    state.channels.register(connected);

    let (mut ws_sender, mut ws_receiver) = ws.split();

    // Immediately send the session echo frame so the client can display
    // the canonical session id.
    let session_frame = serde_json::json!({ "type": "session", "id": session_id });
    let _ = outbound_tx.send(session_frame.to_string()).await;

    // Writer task: browser-destined JSON frames.
    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            if ws_sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Bridge task: translate ChannelEvent JSON → webchat wire frames
    // and push them to the writer.
    let history = state.webchat_history.clone();
    let session_id_bridge = session_id.clone();
    let writer_outbound = outbound_tx.clone();
    let bridge_handle = tokio::spawn(async move {
        while let Some(raw) = bridge_rx.recv().await {
            let frame = match translate_channel_event(&raw) {
                Some(f) => f,
                None => continue,
            };

            // Record assistant text into history.
            if let (Some(hist), Some(text)) = (history.as_ref(), frame.assistant_text()) {
                hist.push(
                    &session_id_bridge,
                    HistoryEntry {
                        role: "assistant",
                        content: text.to_string(),
                        timestamp: unix_now(),
                    },
                );
            }

            let Ok(json) = serde_json::to_string(&frame) else {
                continue;
            };
            if writer_outbound.send(json).await.is_err() {
                break;
            }
        }
    });

    // Read loop.
    while let Some(result) = ws_receiver.next().await {
        match result {
            Ok(Message::Text(text)) => {
                if text.len() > MAX_INBOUND_FRAME_BYTES {
                    send_error(&outbound_tx, "payload too large").await;
                    continue;
                }
                handle_inbound_frame(
                    &state,
                    &outbound_tx,
                    channel_id,
                    &user_id,
                    &session_id,
                    &text,
                )
                .await;
            }
            Ok(Message::Binary(_)) => {
                send_error(&outbound_tx, "binary frames are not supported").await;
            }
            Ok(Message::Ping(_)) => {
                state.channels.touch_heartbeat(&channel_id);
            }
            Ok(Message::Close(_)) => {
                tracing::info!(%channel_id, "WebChat client sent close frame");
                break;
            }
            Ok(_) => { /* Pong — ignore */ }
            Err(e) => {
                tracing::warn!(%channel_id, error = %e, "WebChat: read error");
                break;
            }
        }
    }

    writer_handle.abort();
    bridge_handle.abort();
    state.channels.unregister(&channel_id);

    tracing::info!(%channel_id, "WebChat client disconnected");
}

/// Handle a single inbound JSON text frame from the browser.
async fn handle_inbound_frame(
    state: &AppState,
    outbound_tx: &mpsc::Sender<String>,
    channel_id: Uuid,
    user_id: &str,
    session_id: &str,
    text: &str,
) {
    let frame: ClientFrame = match serde_json::from_str(text) {
        Ok(f) => f,
        Err(_) => {
            send_error(outbound_tx, "invalid JSON frame").await;
            return;
        }
    };

    match frame {
        ClientFrame::Message { content } => {
            if content.trim().is_empty() {
                send_error(outbound_tx, "empty message").await;
                return;
            }
            if let Some(ref hist) = state.webchat_history {
                hist.push(
                    session_id,
                    HistoryEntry {
                        role: "user",
                        content: content.clone(),
                        timestamp: unix_now(),
                    },
                );
            }

            // Translate to the gateway's generic channel event so that the
            // existing `AgentInboundHandler` path runs (rate limiter,
            // sanitizer, media, slash commands, agent dispatch).
            let event = ChannelEvent::MessageReceived(ChannelMessage {
                id: MessageId::new(Uuid::new_v4().to_string()),
                conversation: ConversationId {
                    platform: WEBCHAT_CHANNEL_TYPE.to_string(),
                    channel_id: session_id.to_string(),
                    server_id: None,
                },
                author: user_id.to_string(),
                content: MessageContent::Text(content),
                thread_id: None,
                reply_to: None,
                timestamp: Utc::now(),
                attachments: vec![],
                metadata: std::collections::HashMap::new(),
            });

            let router = state.router.clone();
            tokio::spawn(async move {
                if let Err(e) = router.handle_inbound(channel_id, &event).await {
                    tracing::error!(
                        channel_id = %channel_id,
                        error = %e,
                        "webchat: router failed"
                    );
                }
            });
        }
        ClientFrame::Resume {
            session_id: claimed,
        } => {
            // Only allow the user to resume their own session.
            if claimed != session_id {
                send_error(outbound_tx, "session mismatch").await;
                return;
            }
            let limit = state.config.webchat_session_history_limit;
            let entries = state
                .webchat_history
                .as_ref()
                .map(|h| h.recent(session_id, limit))
                .unwrap_or_default();
            let frame = serde_json::json!({
                "type": "history",
                "session_id": session_id,
                "messages": entries,
            });
            let _ = outbound_tx.send(frame.to_string()).await;
        }
        ClientFrame::Typing { .. } => {
            // Currently a no-op — typing indicators from the user are not
            // forwarded into the agent stack.
        }
    }
}

/// Admin-guarded history endpoint. The session-id path segment is
/// interpreted verbatim; admin token check is applied by the outer router
/// via [`crate::admin::require_admin`] when an admin token is configured.
pub async fn serve_history(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    body: Option<axum::Json<HistoryRequest>>,
) -> Response {
    let req = body.map(|axum::Json(b)| b).unwrap_or_default();
    let cap = state.config.webchat_session_history_limit;
    let limit = req.limit.unwrap_or(cap).min(cap);

    let messages = state
        .webchat_history
        .as_ref()
        .map(|h| h.recent(&session_id, limit))
        .unwrap_or_default();

    axum::Json(HistoryResponse {
        session_id,
        messages,
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// Protocol helpers
// ---------------------------------------------------------------------------

/// Client → server frame.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientFrame {
    Message {
        content: String,
    },
    Resume {
        session_id: String,
    },
    Typing {
        #[serde(default)]
        #[allow(dead_code)]
        typing: bool,
    },
}

/// Server → client frame. Serialised with `type` as the tag.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerFrame {
    /// Final assistant message.
    Message {
        role: &'static str,
        content: String,
        id: String,
    },
    /// Streaming partial token.
    #[allow(dead_code)]
    Chunk { content: String },
    /// Tool-call start/end event.
    ToolUse {
        name: String,
        status: &'static str,
        preview: String,
    },
    /// Error event.
    Error { message: String },
}

impl ServerFrame {
    /// If this frame carries assistant text (for history logging), return it.
    fn assistant_text(&self) -> Option<&str> {
        match self {
            ServerFrame::Message { content, .. } => Some(content),
            _ => None,
        }
    }
}

/// Translate a `ChannelEvent` JSON payload (produced upstream by
/// `AgentInboundHandler::send_response`) into a webchat [`ServerFrame`].
///
/// Returns `None` for events we don't surface to the browser
/// (heartbeats, typing, etc.).
fn translate_channel_event(raw: &str) -> Option<ServerFrame> {
    let event: ChannelEvent = serde_json::from_str(raw).ok()?;
    match event {
        ChannelEvent::MessageReceived(msg) => {
            // `AgentInboundHandler` emits assistant replies as
            // `MessageReceived` with `author == "assistant"`.
            let text = match msg.content {
                MessageContent::Text(t) => t,
                MessageContent::RichText { markdown, .. } => markdown,
                MessageContent::Mixed(parts) => parts
                    .into_iter()
                    .filter_map(|p| match p {
                        MessageContent::Text(t) => Some(t),
                        MessageContent::RichText { markdown, .. } => Some(markdown),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => return None,
            };
            Some(ServerFrame::Message {
                role: if msg.author == "assistant" {
                    "assistant"
                } else {
                    "user"
                },
                content: text,
                id: msg.id.to_string(),
            })
        }
        ChannelEvent::TypingStarted { .. } => Some(ServerFrame::ToolUse {
            name: "typing".to_string(),
            status: "start",
            preview: String::new(),
        }),
        _ => None,
    }
}

async fn send_error(tx: &mpsc::Sender<String>, msg: &str) {
    let frame = ServerFrame::Error {
        message: msg.to_string(),
    };
    if let Ok(json) = serde_json::to_string(&frame) {
        let _ = tx.send(json).await;
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Minimal HS256 JWT implementation
// ---------------------------------------------------------------------------

/// Decode and verify an HS256 JWT with the supplied secret.
///
/// Returns the deserialised claims on success, or a short static reason
/// string on failure. We roll our own (rather than pull in `jsonwebtoken`)
/// because the dependency surface here is intentionally tiny.
pub fn decode_hs256_jwt(token: &str, secret: &[u8]) -> Result<WebChatClaims, &'static str> {
    let mut parts = token.split('.');
    let header_b64 = parts.next().ok_or("malformed token")?;
    let payload_b64 = parts.next().ok_or("malformed token")?;
    let sig_b64 = parts.next().ok_or("malformed token")?;
    if parts.next().is_some() {
        return Err("malformed token");
    }

    // Verify algorithm.
    let header_bytes = url_b64_decode(header_b64).map_err(|_| "bad header encoding")?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|_| "bad header json")?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("HS256") {
        return Err("unsupported alg");
    }

    // Verify signature over `header.payload`.
    let signing_input = format!("{}.{}", header_b64, payload_b64);
    let expected_sig = url_b64_decode(sig_b64).map_err(|_| "bad signature encoding")?;

    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| "bad secret")?;
    mac.update(signing_input.as_bytes());
    mac.verify_slice(&expected_sig)
        .map_err(|_| "signature mismatch")?;

    // Decode payload.
    let payload_bytes = url_b64_decode(payload_b64).map_err(|_| "bad payload encoding")?;
    let claims: WebChatClaims =
        serde_json::from_slice(&payload_bytes).map_err(|_| "bad payload json")?;

    // Reject expired tokens.
    let now = unix_now();
    if now > 0 && (claims.exp as i64) < now {
        return Err("token expired");
    }
    if claims.sub.is_empty() {
        return Err("empty sub claim");
    }

    Ok(claims)
}

/// Issue an HS256 JWT for webchat. Used by the daemon to mint short-lived
/// tokens after successful admin-token exchange, and by tests.
pub fn issue_hs256_jwt(sub: &str, secret: &[u8], ttl_secs: u64) -> Result<String, &'static str> {
    let iat = unix_now().max(0) as u64;
    let exp = iat.saturating_add(ttl_secs);
    let claims = WebChatClaims {
        sub: sub.to_string(),
        exp,
        iat: Some(iat),
    };
    let header = serde_json::json!({ "alg": "HS256", "typ": "JWT" });
    let header_b64 = url_b64_encode(&serde_json::to_vec(&header).map_err(|_| "encode header")?);
    let payload_b64 = url_b64_encode(&serde_json::to_vec(&claims).map_err(|_| "encode payload")?);
    let signing_input = format!("{}.{}", header_b64, payload_b64);

    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| "bad secret")?;
    mac.update(signing_input.as_bytes());
    let sig = url_b64_encode(&mac.finalize().into_bytes());

    Ok(format!("{signing_input}.{sig}"))
}

/// URL-safe base64 decode without padding.
fn url_b64_decode(input: &str) -> Result<Vec<u8>, ()> {
    let mut s = input.replace('-', "+").replace('_', "/");
    while !s.len().is_multiple_of(4) {
        s.push('=');
    }
    // Minimal base64 decoder — we bring it in-module to avoid a new dep.
    base64_decode_standard(&s).map_err(|_| ())
}

fn url_b64_encode(input: &[u8]) -> String {
    base64_encode_standard(input)
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_string()
}

/// Minimal standard-base64 encoder.
fn base64_encode_standard(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() >= 2 {
            ALPHABET[((n >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() >= 3 {
            ALPHABET[(n & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Minimal standard-base64 decoder.
fn base64_decode_standard(input: &str) -> Result<Vec<u8>, ()> {
    fn v(c: u8) -> Result<u8, ()> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(()),
        }
    }
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let c0 = v(chunk[0])?;
        let c1 = v(chunk[1])?;
        let c2 = chunk[2];
        let c3 = chunk[3];

        let n = ((c0 as u32) << 18) | ((c1 as u32) << 12);
        if c2 == b'=' {
            out.push(((n >> 16) & 0xFF) as u8);
            break;
        }
        let c2v = v(c2)?;
        let n = n | ((c2v as u32) << 6);
        if c3 == b'=' {
            out.push(((n >> 16) & 0xFF) as u8);
            out.push(((n >> 8) & 0xFF) as u8);
            break;
        }
        let c3v = v(c3)?;
        let n = n | c3v as u32;
        out.push(((n >> 16) & 0xFF) as u8);
        out.push(((n >> 8) & 0xFF) as u8);
        out.push((n & 0xFF) as u8);
    }
    Ok(out)
}

/// Shared type alias used by [`AppState`] to hold a dyn verifier.
pub type SharedVerifier = Arc<dyn AuthVerifier>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webchat_html_is_embedded() {
        let html = include_str!("../static/webchat.html");
        assert!(!html.is_empty());
        assert!(html.contains("BrainClaw Chat"));
    }

    #[test]
    fn admin_ui_html_is_embedded() {
        let html = include_str!("../static/admin_ui.html");
        assert!(!html.is_empty());
        assert!(html.contains("BrainClaw Admin"));
    }

    #[test]
    fn webchat_enabled_default() {
        let config = crate::config::GatewayConfig::default();
        assert!(config.webchat_enabled);
    }

    #[test]
    fn base64_roundtrip_simple() {
        let cases: &[&[u8]] = &[b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"];
        for c in cases {
            let encoded = base64_encode_standard(c);
            let decoded = base64_decode_standard(&encoded).expect("decode");
            assert_eq!(&decoded[..], *c, "mismatch for input {:?}", c);
        }
    }

    #[test]
    fn hs256_jwt_issue_and_verify() {
        let secret = b"super-secret";
        let token = issue_hs256_jwt("alice", secret, 60).expect("issue");
        let verifier = Hs256Verifier::new(secret.to_vec());
        match verifier.verify(&token) {
            AuthVerdict::Authorized { user_id } => assert_eq!(user_id, "alice"),
            AuthVerdict::Rejected { reason } => panic!("unexpected reject: {reason}"),
        }
    }

    #[test]
    fn hs256_jwt_rejects_wrong_secret() {
        let token = issue_hs256_jwt("bob", b"one", 60).expect("issue");
        let verifier = Hs256Verifier::new(b"two".to_vec());
        assert!(matches!(
            verifier.verify(&token),
            AuthVerdict::Rejected { .. }
        ));
    }

    #[test]
    fn hs256_jwt_rejects_expired_token() {
        let secret = b"s3cret";
        // ttl=0 => exp==iat which is <= now.
        let token = issue_hs256_jwt("carol", secret, 0).expect("issue");
        // Guarantee the clock has moved past `exp`.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let verifier = Hs256Verifier::new(secret.to_vec());
        match verifier.verify(&token) {
            AuthVerdict::Rejected { reason } => assert_eq!(reason, "token expired"),
            AuthVerdict::Authorized { .. } => panic!("expired token accepted"),
        }
    }

    #[test]
    fn history_ring_respects_limit() {
        let hist = WebChatHistory::new();
        for i in 0..10 {
            hist.push(
                "webchat:u1",
                HistoryEntry {
                    role: "user",
                    content: format!("m{i}"),
                    timestamp: i as i64,
                },
            );
        }
        let recent = hist.recent("webchat:u1", 3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[2].content, "m9");
    }

    #[test]
    fn translate_assistant_message_round_trips() {
        let event = ChannelEvent::MessageReceived(ChannelMessage {
            id: MessageId::new("m-1".to_string()),
            conversation: ConversationId {
                platform: "webchat".into(),
                channel_id: "webchat:user".into(),
                server_id: None,
            },
            author: "assistant".into(),
            content: MessageContent::Text("hi there".into()),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: std::collections::HashMap::new(),
        });
        let raw = serde_json::to_string(&event).unwrap();
        let frame = translate_channel_event(&raw).expect("frame");
        match frame {
            ServerFrame::Message { role, content, .. } => {
                assert_eq!(role, "assistant");
                assert_eq!(content, "hi there");
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }
}
