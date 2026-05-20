//! Gmail push webhook handler + daemon wiring (OpenClaw parity P3.1).
//!
//! Workflow:
//!
//! 1. The daemon loads a [`GmailPushHandler`] per watched account and
//!    registers them in [`GmailPushRegistry`]. The gateway holds a
//!    `Option<Arc<GmailPushRegistry>>` in its `AppState`.
//! 2. Google Pub/Sub POSTs to `/webhooks/gmail-push`. The handler:
//!    - Parses the Pub/Sub envelope to find the mailbox address.
//!    - Verifies the Google-signed JWT.
//!    - Walks Gmail's history API starting from the persisted cursor.
//!    - Dispatches each fetched message into the gateway's
//!      [`InboundHandler`] as a synthetic `ChannelEvent::MessageReceived`
//!      on the `email` channel.
//!    - Persists the new cursor before returning `204 No Content`.
//! 3. Pub/Sub message ids are tracked in a small in-memory LRU for
//!    best-effort de-dup; the gateway accepts a second delivery silently
//!    without calling Gmail a second time.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::identity::ConversationId;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};
use brainwires_tools::gmail_push::{EmailMessage, GmailPushHandler, WatchResponse};

use crate::state::AppState;

/// Registry of [`GmailPushHandler`] instances keyed by watched mailbox
/// address (lowercase).
pub struct GmailPushRegistry {
    handlers: RwLock<HashMap<String, Arc<GmailPushHandler>>>,
    cursors: Arc<GmailCursorStore>,
    dedup: Mutex<VecDeque<String>>,
    dedup_cap: usize,
}

impl GmailPushRegistry {
    /// Create a new empty registry backed by `cursors`.
    pub fn new(cursors: Arc<GmailCursorStore>) -> Self {
        Self {
            handlers: RwLock::new(HashMap::new()),
            cursors,
            dedup: Mutex::new(VecDeque::with_capacity(1024)),
            dedup_cap: 1000,
        }
    }

    /// Register a handler for a specific mailbox.
    pub async fn register(&self, email_address: impl Into<String>, handler: Arc<GmailPushHandler>) {
        let key = email_address.into().to_ascii_lowercase();
        self.handlers.write().await.insert(key, handler);
    }

    /// Look up a handler for a mailbox (case-insensitive).
    pub async fn get(&self, email_address: &str) -> Option<Arc<GmailPushHandler>> {
        self.handlers
            .read()
            .await
            .get(&email_address.to_ascii_lowercase())
            .cloned()
    }

    /// Expose the shared cursor store (for the CLI / daemon tasks).
    pub fn cursors(&self) -> Arc<GmailCursorStore> {
        Arc::clone(&self.cursors)
    }

    /// Best-effort de-dup check. Returns `true` if the message id has not
    /// been seen before; records it otherwise.
    fn mark_seen(&self, message_id: &str) -> bool {
        let mut dq = self.dedup.lock().expect("gmail push dedup mutex poisoned");
        if dq.iter().any(|x| x == message_id) {
            return false;
        }
        if dq.len() >= self.dedup_cap {
            dq.pop_front();
        }
        dq.push_back(message_id.to_string());
        true
    }
}

/// On-disk store of `{ email_address: history_id }` pairs.
///
/// We persist the cursor so that a daemon restart doesn't replay the
/// whole mailbox (or worse, miss messages that arrived while offline —
/// Gmail's history API is bounded to roughly 30 days so there is always
/// some cursor to resume from).
pub struct GmailCursorStore {
    path: PathBuf,
    inner: RwLock<HashMap<String, u64>>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct CursorFile {
    #[serde(default)]
    cursors: HashMap<String, u64>,
}

impl GmailCursorStore {
    /// Load the cursor file from disk, or start fresh if it doesn't exist.
    pub async fn load(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = path.into();
        let inner = if path.exists() {
            let raw = tokio::fs::read_to_string(&path).await?;
            if raw.trim().is_empty() {
                HashMap::new()
            } else {
                let parsed: CursorFile = serde_json::from_str(&raw)?;
                parsed.cursors
            }
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            inner: RwLock::new(inner),
        })
    }

    /// Get the persisted cursor for an address, or `None` if the watch
    /// hasn't fired yet for this mailbox.
    pub async fn get(&self, email_address: &str) -> Option<u64> {
        self.inner
            .read()
            .await
            .get(&email_address.to_ascii_lowercase())
            .copied()
    }

    /// Persist the cursor for an address to disk.
    pub async fn put(&self, email_address: &str, history_id: u64) -> anyhow::Result<()> {
        {
            let mut guard = self.inner.write().await;
            guard.insert(email_address.to_ascii_lowercase(), history_id);
        }
        self.flush().await
    }

    async fn flush(&self) -> anyhow::Result<()> {
        let snapshot = {
            let guard = self.inner.read().await;
            CursorFile {
                cursors: guard.clone(),
            }
        };
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let tmp = tmp_path(&self.path);
        let json = serde_json::to_vec_pretty(&snapshot)?;
        tokio::fs::write(&tmp, &json).await?;
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }

    /// Path of the underlying JSON file (used by the CLI to surface it).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let fname = match tmp.file_name() {
        Some(n) => format!("{}.tmp", n.to_string_lossy()),
        None => return tmp,
    };
    tmp.set_file_name(fname);
    tmp
}

/// Handle a Pub/Sub push request at `/webhooks/gmail-push`.
///
/// - 401 on missing or invalid Authorization header.
/// - 400 on malformed Pub/Sub envelope.
/// - 404 when the envelope targets an unregistered mailbox.
/// - 204 on success (including when the payload was a Pub/Sub redelivery
///   suppressed by the in-memory de-dup LRU).
pub async fn handle_gmail_push(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let registry = match &state.gmail_push_registry {
        Some(r) => r.clone(),
        None => {
            return (StatusCode::NOT_FOUND, "Gmail push not enabled").into_response();
        }
    };

    // 1. Parse envelope first — cheap, doesn't leak OAuth state.
    let envelope = match GmailPushHandler::parse_envelope(&body) {
        Ok(env) => env,
        Err(e) => {
            tracing::warn!(error = %e, "Gmail push: malformed envelope");
            return (StatusCode::BAD_REQUEST, "invalid envelope").into_response();
        }
    };

    // 2. Look up the handler for the addressed mailbox.
    let handler = match registry.get(&envelope.email_address).await {
        Some(h) => h,
        None => {
            tracing::warn!(
                email = %envelope.email_address,
                "Gmail push: no handler registered for mailbox"
            );
            return (StatusCode::NOT_FOUND, "unknown mailbox").into_response();
        }
    };

    // 3. Verify Google's signed JWT.
    let bearer = match headers.get(axum::http::header::AUTHORIZATION) {
        Some(h) => match h.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                return (StatusCode::UNAUTHORIZED, "invalid auth header").into_response();
            }
        },
        None => {
            return (StatusCode::UNAUTHORIZED, "missing auth header").into_response();
        }
    };

    if let Err(e) = handler.verify_push_jwt(&bearer).await {
        // Do NOT include `bearer` in the log — it's a signed JWT that
        // contains claims the attacker may have crafted. The error
        // message from jsonwebtoken is safe to log.
        tracing::warn!(
            email = %envelope.email_address,
            error = %e,
            "Gmail push: JWT verification failed"
        );
        return (StatusCode::UNAUTHORIZED, "jwt rejected").into_response();
    }

    // 4. De-dup via Pub/Sub message id.
    if let Some(ref mid) = envelope.message_id
        && !registry.mark_seen(mid)
    {
        tracing::debug!(
            email = %envelope.email_address,
            message_id = %mid,
            "Gmail push: duplicate Pub/Sub delivery ignored"
        );
        return (StatusCode::NO_CONTENT, ()).into_response();
    }

    // 5. Load the last-known history cursor.
    let cursors = registry.cursors();
    let since = cursors
        .get(&envelope.email_address)
        .await
        .unwrap_or(envelope.history_id);

    // 6. Pull any new messages.
    let (messages, new_cursor) = match handler.fetch_new_messages(&envelope, since).await {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!(
                email = %envelope.email_address,
                error = %e,
                "Gmail push: fetch_new_messages failed (backing off)"
            );
            // Pub/Sub retries a 5xx for us, so surface as 503 — this
            // also nudges GCP's exponential backoff.
            return (StatusCode::SERVICE_UNAVAILABLE, "fetch failed").into_response();
        }
    };

    if messages.is_empty() {
        tracing::debug!(
            email = %envelope.email_address,
            since,
            new_cursor,
            "Gmail push: history delta was empty"
        );
    } else {
        tracing::info!(
            email = %envelope.email_address,
            count = messages.len(),
            new_cursor,
            "Gmail push: dispatching messages to agent"
        );
    }

    // 7. Dispatch each message into the inbound handler.
    let webhook_channel_id = Uuid::nil();
    for m in messages {
        let event = build_inbound_event(&envelope.email_address, &m);
        if let Err(e) = state
            .router
            .handle_inbound(webhook_channel_id, &event)
            .await
        {
            tracing::warn!(
                email = %envelope.email_address,
                message_id = %m.id,
                error = %e,
                "Gmail push: inbound dispatch failed"
            );
        }
    }

    // 8. Persist the cursor AFTER dispatch so a crash between fetch and
    //    dispatch is retried via Pub/Sub redelivery.
    if new_cursor > since
        && let Err(e) = cursors.put(&envelope.email_address, new_cursor).await
    {
        tracing::warn!(
            email = %envelope.email_address,
            error = %e,
            "Gmail push: failed to persist cursor"
        );
    }

    (StatusCode::NO_CONTENT, ()).into_response()
}

/// Build a synthetic `ChannelEvent` for a Gmail message — channel name
/// `"email"`, conversation keyed on the mailbox, and the body formatted
/// for an LLM to read.
fn build_inbound_event(mailbox: &str, msg: &EmailMessage) -> ChannelEvent {
    let subject_line = if msg.subject.is_empty() {
        "(no subject)"
    } else {
        msg.subject.as_str()
    };
    let body_preview = if msg.body_text.is_empty() {
        msg.body_html.clone().unwrap_or_default()
    } else {
        msg.body_text.clone()
    };
    let text = format!(
        "From: {}\nTo: {}\nSubject: {}\n\n{}",
        msg.from, mailbox, subject_line, body_preview
    );

    ChannelEvent::MessageReceived(ChannelMessage {
        id: MessageId::new(msg.id.clone()),
        conversation: ConversationId {
            platform: "email".to_string(),
            channel_id: mailbox.to_string(),
            server_id: None,
        },
        author: msg.from.clone(),
        content: MessageContent::Text(text),
        thread_id: if msg.thread_id.is_empty() {
            None
        } else {
            Some(brainwires_network::channels::message::ThreadId::new(
                msg.thread_id.clone(),
            ))
        },
        reply_to: None,
        timestamp: msg.received_at,
        attachments: vec![],
        metadata: std::collections::HashMap::new(),
    })
}

/// Convenience for CLI + daemon: re-register a watch on a mailbox and
/// seed the cursor store from the response.
pub async fn register_watch_and_seed(
    handler: &GmailPushHandler,
    email_address: &str,
    cursors: &GmailCursorStore,
) -> anyhow::Result<WatchResponse> {
    let resp = handler.register_watch().await?;
    // Only seed when we've never persisted a cursor for this mailbox —
    // otherwise we'd regress on a functional subscription.
    if cursors.get(email_address).await.is_none() {
        cursors.put(email_address, resp.history_id).await?;
    }
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn cursor_store_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("gmail_cursor.json");
        let store = GmailCursorStore::load(&path).await.unwrap();
        assert!(store.get("alice@example.com").await.is_none());

        store.put("Alice@Example.com", 42).await.unwrap();
        assert_eq!(store.get("alice@example.com").await, Some(42));

        let reopened = GmailCursorStore::load(&path).await.unwrap();
        assert_eq!(reopened.get("alice@example.com").await, Some(42));
    }

    #[test]
    fn build_inbound_event_uses_email_channel() {
        let msg = EmailMessage {
            id: "m-1".into(),
            thread_id: "t-1".into(),
            from: "alice@example.com".into(),
            to: vec!["bob@example.com".into()],
            cc: vec![],
            subject: "hi".into(),
            body_text: "hello world".into(),
            body_html: None,
            received_at: chrono::Utc::now(),
            labels: vec!["INBOX".into()],
        };
        let event = build_inbound_event("bob@example.com", &msg);
        match event {
            ChannelEvent::MessageReceived(m) => {
                assert_eq!(m.conversation.platform, "email");
                assert_eq!(m.conversation.channel_id, "bob@example.com");
                assert_eq!(m.author, "alice@example.com");
                match m.content {
                    MessageContent::Text(t) => {
                        assert!(t.contains("From: alice@example.com"));
                        assert!(t.contains("Subject: hi"));
                        assert!(t.contains("hello world"));
                    }
                    _ => panic!("expected text"),
                }
            }
            _ => panic!("expected MessageReceived"),
        }
    }
}
