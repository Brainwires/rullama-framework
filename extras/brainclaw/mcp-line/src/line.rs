//! LINE REST client + [`Channel`] implementation.
//!
//! Outbound flow:
//!
//! - `POST https://api.line.me/v2/bot/message/reply` — for timely replies
//!   to inbound events. Requires a `replyToken` minted at ingress and
//!   only valid for ~1 minute.
//! - `POST https://api.line.me/v2/bot/message/push` — for unprompted
//!   sends, or fallback when a reply token has expired.
//!
//! The adapter prefers `reply` when a fresh reply token is available
//! and falls back to `push` otherwise; this keeps free-tier quota
//! untouched where possible.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
};

/// Base URL for the LINE Messaging API. Tests override this.
pub const LINE_API_BASE: &str = "https://api.line.me/v2/bot";

/// Maximum age of a LINE reply token we'll attempt to use.
pub const REPLY_TOKEN_MAX_AGE: Duration = Duration::from_secs(55);

/// A cached reply token with creation timestamp.
#[derive(Debug, Clone)]
pub struct ReplyToken {
    /// The token string LINE gave us.
    pub token: String,
    /// When we cached it.
    pub issued_at: Instant,
}

impl ReplyToken {
    /// True if the token is within the safe (<55s) window.
    pub fn is_fresh(&self) -> bool {
        self.issued_at.elapsed() < REPLY_TOKEN_MAX_AGE
    }
}

/// Lock around the active reply-token store. Keyed by LINE user id.
#[derive(Debug, Default)]
pub struct ReplyTokenStore {
    inner: RwLock<HashMap<String, ReplyToken>>,
}

impl ReplyTokenStore {
    /// Remember a token for a user.
    pub fn remember(&self, user_id: &str, token: String) {
        self.inner.write().insert(
            user_id.to_string(),
            ReplyToken {
                token,
                issued_at: Instant::now(),
            },
        );
    }
    /// Take the token for `user_id` iff it is still fresh. Consumes
    /// the entry so it can't be re-used (LINE rejects double-replies).
    pub fn take_fresh(&self, user_id: &str) -> Option<String> {
        let mut g = self.inner.write();
        let entry = g.remove(user_id)?;
        if entry.is_fresh() {
            Some(entry.token)
        } else {
            None
        }
    }
}

/// Request type for the MCP `send_message` tool.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendMessageRequest {
    /// LINE user id (starts with `U…`).
    pub to: String,
    /// Text body to send.
    pub text: String,
}

/// [`Channel`] implementation over the LINE Messaging API.
pub struct LineChannel {
    http: reqwest::Client,
    access_token: String,
    api_base: String,
    reply_tokens: Arc<ReplyTokenStore>,
}

impl LineChannel {
    /// Construct a new channel wired to the given access token.
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            access_token: access_token.into(),
            api_base: LINE_API_BASE.to_string(),
            reply_tokens: Arc::new(ReplyTokenStore::default()),
        }
    }

    /// Shared reply-token store — exposed so the webhook can remember
    /// tokens as they arrive.
    pub fn reply_tokens(&self) -> Arc<ReplyTokenStore> {
        Arc::clone(&self.reply_tokens)
    }

    /// Override the API base — tests only.
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    async fn call(&self, path: &str, body: serde_json::Value) -> Result<()> {
        let url = format!("{}{path}", self.api_base);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            anyhow::bail!("LINE API returned {status}: {} bytes", txt.len());
        }
        Ok(())
    }
}

#[async_trait]
impl Channel for LineChannel {
    fn channel_type(&self) -> &str {
        "line"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT | ChannelCapabilities::MENTIONS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = extract_text(message);
        if text.is_empty() {
            anyhow::bail!("LINE send_message: empty text");
        }
        // Prefer the reply API when we still hold a fresh reply token.
        if let Some(token) = self.reply_tokens.take_fresh(&target.channel_id) {
            let body = json!({
                "replyToken": token,
                "messages": [{"type": "text", "text": text}],
            });
            self.call("/message/reply", body).await?;
            return Ok(MessageId::new(format!("reply:{}", target.channel_id)));
        }
        // Fallback: push API (counts against free-tier quota).
        let body = json!({
            "to": target.channel_id,
            "messages": [{"type": "text", "text": text}],
        });
        self.call("/message/push", body).await?;
        Ok(MessageId::new(format!("push:{}", target.channel_id)))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("edit_message is not supported by LINE")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        anyhow::bail!("delete_message is not supported by LINE")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("add_reaction is not supported by LINE (bot API)")
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        Ok(Vec::new())
    }
}

/// A single parsed inbound event.
#[derive(Debug, Clone)]
pub enum IngressEvent {
    /// User text / media / postback message we forward. Boxed so the
    /// enum size stays reasonable — `ChannelMessage` is large and most
    /// variants are tiny.
    Message(Box<ChannelMessage>, Option<String>),
    /// Event that we intentionally drop (follow/unfollow/read, etc.).
    Dropped {
        /// LINE event type string (e.g. "follow", "unfollow", "join").
        event_type: String,
    },
}

/// Parse the LINE webhook body. The body shape is `{"events": [..]}`.
///
/// Returns one parsed variant per raw event.
pub fn parse_events(body: &serde_json::Value) -> Vec<IngressEvent> {
    let Some(events) = body.get("events").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    events.iter().map(parse_event).collect()
}

/// Parse a single LINE event object.
pub fn parse_event(ev: &serde_json::Value) -> IngressEvent {
    let ev_type = ev.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
    let source = ev.get("source");
    let user_id = source
        .and_then(|s| s.get("userId"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let reply_token = ev
        .get("replyToken")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    match ev_type {
        "message" => {
            let m = ev.get("message");
            let msg_type = m
                .and_then(|m| m.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let message_id = m
                .and_then(|m| m.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let text = match msg_type {
                "text" => m
                    .and_then(|m| m.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                "image" | "video" | "audio" | "file" => {
                    // Forward an event-level marker; content requires a
                    // subsequent authenticated call that we don't make at
                    // MVP. The agent can request content via the MCP
                    // layer if needed.
                    format!("[{msg_type} attachment id={message_id}]")
                }
                "sticker" => {
                    let pkg = m
                        .and_then(|m| m.get("packageId"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let sid = m
                        .and_then(|m| m.get("stickerId"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    format!("[sticker {pkg}/{sid}]")
                }
                other => format!("[unsupported {other}]"),
            };
            let timestamp = ev
                .get("timestamp")
                .and_then(|v| v.as_i64())
                .and_then(chrono::DateTime::from_timestamp_millis)
                .unwrap_or_else(chrono::Utc::now);
            let mut metadata = HashMap::new();
            metadata.insert("line.user_id".into(), user_id.clone());
            metadata.insert("line.session_id".into(), format!("line:{user_id}"));
            if !message_id.is_empty() {
                metadata.insert("line.message_id".into(), message_id.clone());
            }
            let channel = ChannelMessage {
                id: MessageId::new(if message_id.is_empty() {
                    uuid::Uuid::new_v4().to_string()
                } else {
                    message_id
                }),
                conversation: ConversationId {
                    platform: "line".into(),
                    channel_id: user_id.clone(),
                    server_id: None,
                },
                author: user_id,
                content: MessageContent::Text(text),
                thread_id: None,
                reply_to: None,
                timestamp,
                attachments: Vec::new(),
                metadata,
            };
            IngressEvent::Message(Box::new(channel), reply_token)
        }
        "postback" => {
            let data = ev
                .get("postback")
                .and_then(|p| p.get("data"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let timestamp = ev
                .get("timestamp")
                .and_then(|v| v.as_i64())
                .and_then(chrono::DateTime::from_timestamp_millis)
                .unwrap_or_else(chrono::Utc::now);
            let mut metadata = HashMap::new();
            metadata.insert("line.user_id".into(), user_id.clone());
            metadata.insert("line.session_id".into(), format!("line:{user_id}"));
            metadata.insert("line.event_type".into(), "postback".into());
            let channel = ChannelMessage {
                id: MessageId::new(uuid::Uuid::new_v4().to_string()),
                conversation: ConversationId {
                    platform: "line".into(),
                    channel_id: user_id.clone(),
                    server_id: None,
                },
                author: user_id,
                content: MessageContent::Text(data),
                thread_id: None,
                reply_to: None,
                timestamp,
                attachments: Vec::new(),
                metadata,
            };
            IngressEvent::Message(Box::new(channel), reply_token)
        }
        other => IngressEvent::Dropped {
            event_type: other.to_string(),
        },
    }
}

fn extract_text(message: &ChannelMessage) -> String {
    match &message.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::RichText {
            markdown,
            fallback_plain,
            ..
        } => {
            if fallback_plain.is_empty() {
                markdown.clone()
            } else {
                fallback_plain.clone()
            }
        }
        MessageContent::Media(m) => match &m.caption {
            Some(c) => format!("{c}\n{}", m.url),
            None => m.url.clone(),
        },
        MessageContent::Embed(e) => {
            let mut parts = Vec::new();
            if let Some(t) = &e.title {
                parts.push(t.clone());
            }
            if let Some(d) = &e.description {
                parts.push(d.clone());
            }
            parts.join("\n")
        }
        MessageContent::Mixed(items) => items
            .iter()
            .map(|c| {
                let stub = ChannelMessage {
                    id: message.id.clone(),
                    conversation: message.conversation.clone(),
                    author: message.author.clone(),
                    content: c.clone(),
                    thread_id: None,
                    reply_to: None,
                    timestamp: message.timestamp,
                    attachments: Vec::new(),
                    metadata: HashMap::new(),
                };
                extract_text(&stub)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Build an outbound LINE body for either `reply` or `push`.
///
/// `reply_token` = `Some` → reply endpoint; `None` → push endpoint.
pub fn build_send_body(to_or_token: &str, text: &str, reply: bool) -> serde_json::Value {
    if reply {
        json!({
            "replyToken": to_or_token,
            "messages": [{"type":"text","text":text}],
        })
    } else {
        json!({
            "to": to_or_token,
            "messages": [{"type":"text","text":text}],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_text_event() {
        let ev = json!({
            "type": "message",
            "replyToken": "rt1",
            "timestamp": 1_700_000_000_000i64,
            "source": {"userId": "U123"},
            "message": {"type":"text","id":"m1","text":"hi"}
        });
        let parsed = parse_event(&ev);
        let (m, rt) = match parsed {
            IngressEvent::Message(m, rt) => (m, rt),
            _ => panic!(),
        };
        assert_eq!(m.conversation.channel_id, "U123");
        match m.content {
            MessageContent::Text(t) => assert_eq!(t, "hi"),
            _ => panic!(),
        }
        assert_eq!(rt.as_deref(), Some("rt1"));
    }

    #[test]
    fn parse_postback_as_message() {
        let ev = json!({
            "type": "postback",
            "replyToken": "rt2",
            "source": {"userId": "U"},
            "postback": {"data": "action=foo"}
        });
        match parse_event(&ev) {
            IngressEvent::Message(m, _) => match m.content {
                MessageContent::Text(t) => assert_eq!(t, "action=foo"),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn parse_follow_drops() {
        let ev = json!({"type":"follow","source":{"userId":"U"}});
        match parse_event(&ev) {
            IngressEvent::Dropped { event_type } => assert_eq!(event_type, "follow"),
            _ => panic!(),
        }
    }

    #[test]
    fn parse_image_stubs_out() {
        let ev = json!({
            "type":"message",
            "source":{"userId":"U"},
            "message":{"type":"image","id":"42"}
        });
        match parse_event(&ev) {
            IngressEvent::Message(m, _) => match m.content {
                MessageContent::Text(t) => assert!(t.contains("image") && t.contains("42")),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn build_send_body_shape() {
        let reply = build_send_body("rt", "hi", true);
        assert_eq!(reply["replyToken"], "rt");
        assert_eq!(reply["messages"][0]["text"], "hi");
        let push = build_send_body("U", "hi", false);
        assert_eq!(push["to"], "U");
    }

    #[test]
    fn reply_token_store_expires() {
        let s = ReplyTokenStore::default();
        s.remember("U", "rt".into());
        // Manually age the token by replacing issued_at with something old.
        {
            let mut g = s.inner.write();
            let t = g.get_mut("U").unwrap();
            t.issued_at = Instant::now() - Duration::from_secs(120);
        }
        assert!(s.take_fresh("U").is_none());
    }

    #[test]
    fn caps_plain() {
        let c = LineChannel::new("fake-access-token");
        assert_eq!(c.channel_type(), "line");
        let caps = c.capabilities();
        assert!(caps.contains(ChannelCapabilities::RICH_TEXT));
        assert!(!caps.contains(ChannelCapabilities::THREADS));
    }
}
