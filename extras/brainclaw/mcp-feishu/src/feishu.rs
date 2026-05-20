//! Feishu / Lark REST client + [`Channel`] implementation.
//!
//! Outbound flow:
//!
//! - Mint a tenant access token via [`crate::oauth::TenantTokenMinter`].
//! - POST `/open-apis/im/v1/messages?receive_id_type=open_id` (or
//!   `chat_id`, depending on the conversation shape).
//! - Body: `{receive_id, msg_type: "text", content: "{\"text\": ...}"}`.
//!
//! Event types we parse:
//!
//! - `url_verification` — the Feishu onboarding handshake. The webhook
//!   layer handles this before routing to [`parse_event`].
//! - `im.message.receive_v1` — user chat message (text or post).
//! - `im.message.message_read_v1` — ack, dropped.
//! - Interactive card clicks — `card.action.trigger`. Forwarded as text
//!   carrying the action tag.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
};

use crate::oauth::{FEISHU_BASE, TenantTokenMinter};

/// Request type for the MCP `send_message` tool.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendMessageRequest {
    /// Receive id — `open_id` for DMs, `chat_id` for groups.
    pub receive_id: String,
    /// Type of id: `open_id` | `user_id` | `union_id` | `chat_id` | `email`.
    pub receive_id_type: String,
    /// Message text.
    pub text: String,
}

/// Channel-event variant produced by [`parse_event`].
#[derive(Debug, Clone)]
pub enum IngressEvent {
    /// User message to forward. Boxed to keep the enum size bounded —
    /// `ChannelMessage` is the largest variant by far.
    Message(Box<ChannelMessage>),
    /// Event type we deliberately drop (read receipt, etc.).
    Dropped {
        /// Feishu event type id (e.g. `im.message.message_read_v1`).
        event_type: String,
    },
}

/// [`Channel`] implementation for Feishu / Lark.
pub struct FeishuChannel {
    minter: Arc<TenantTokenMinter>,
    http: reqwest::Client,
    base_url: String,
}

impl FeishuChannel {
    /// Construct a new channel.
    pub fn new(minter: Arc<TenantTokenMinter>) -> Self {
        Self {
            minter,
            http: reqwest::Client::new(),
            base_url: FEISHU_BASE.to_string(),
        }
    }

    /// Override the base URL — tests only.
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Low-level send — exposed for the MCP layer, where the agent
    /// picks the receive_id_type explicitly.
    pub async fn post_text(
        &self,
        receive_id: &str,
        receive_id_type: &str,
        text: &str,
    ) -> Result<MessageId> {
        let bearer = self
            .minter
            .bearer()
            .await
            .context("mint tenant access token")?;
        let url = format!(
            "{}/open-apis/im/v1/messages?receive_id_type={}",
            self.base_url.trim_end_matches('/'),
            urlencoding::encode(receive_id_type)
        );
        let content_str = serde_json::to_string(&json!({ "text": text }))?;
        let body = json!({
            "receive_id": receive_id,
            "msg_type": "text",
            "content": content_str,
        });
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&bearer)
            .json(&body)
            .send()
            .await
            .context("POST Feishu messages")?;
        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            anyhow::bail!("Feishu API returned {status}: {} bytes", txt.len());
        }
        #[derive(Deserialize)]
        struct R {
            code: i64,
            #[serde(default)]
            msg: Option<String>,
            #[serde(default)]
            data: Option<MsgData>,
        }
        #[derive(Deserialize)]
        struct MsgData {
            #[serde(default)]
            message_id: Option<String>,
        }
        let parsed: R = resp.json().await.context("parse Feishu response")?;
        if parsed.code != 0 {
            anyhow::bail!(
                "Feishu send returned code={} msg={:?}",
                parsed.code,
                parsed.msg
            );
        }
        Ok(MessageId::new(
            parsed
                .data
                .and_then(|d| d.message_id)
                .unwrap_or_else(|| format!("om_{}", uuid::Uuid::new_v4().simple())),
        ))
    }
}

#[async_trait]
impl Channel for FeishuChannel {
    fn channel_type(&self) -> &str {
        "feishu"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::THREADS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = extract_text(message);
        if text.is_empty() {
            anyhow::bail!("Feishu send_message: empty text payload");
        }
        // receive_id_type heuristic:
        // - `oc_*`       → open chat id.
        // - `ou_* / on_*` → open user id.
        // Fall back to `chat_id` otherwise (groups usually use `chat_id`).
        let id = &target.channel_id;
        let id_type = if id.starts_with("oc_") {
            "chat_id"
        } else if id.starts_with("ou_") {
            "open_id"
        } else if id.starts_with("on_") {
            "union_id"
        } else {
            // Respect caller-supplied metadata override.
            message
                .metadata
                .get("feishu.receive_id_type")
                .map(|s| s.as_str())
                .unwrap_or("chat_id")
        };
        self.post_text(id, id_type, &text).await
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("edit_message is not yet implemented for feishu")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        anyhow::bail!("delete_message is not yet implemented for feishu")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("add_reaction is not yet implemented for feishu")
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        Ok(Vec::new())
    }
}

/// Parse a Feishu v2 event envelope (`{"schema":"2.0","header":{..},"event":{..}}`).
///
/// Returns a [`IngressEvent`]. Call sites should handle
/// `url_verification` before invoking this.
pub fn parse_event(body: &serde_json::Value) -> Result<IngressEvent> {
    let event_type = body
        .get("header")
        .and_then(|h| h.get("event_type"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("event missing header.event_type"))?;
    let ev = body
        .get("event")
        .ok_or_else(|| anyhow::anyhow!("event body missing `event`"))?;
    match event_type {
        "im.message.receive_v1" => {
            let msg = ev
                .get("message")
                .ok_or_else(|| anyhow::anyhow!("missing event.message"))?;
            let sender = ev.get("sender");
            let open_id = sender
                .and_then(|s| s.get("sender_id"))
                .and_then(|sid| sid.get("open_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let chat_id = msg
                .get("chat_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let msg_id = msg
                .get("message_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let msg_type = msg
                .get("message_type")
                .and_then(|v| v.as_str())
                .unwrap_or("text");
            let create_ms = msg
                .get("create_time")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            let content_raw = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let text = extract_content_text(msg_type, content_raw);
            let timestamp: DateTime<Utc> =
                chrono::DateTime::from_timestamp_millis(create_ms).unwrap_or_else(Utc::now);
            let session_id = format!("feishu:{chat_id}:{open_id}");
            let mut metadata = HashMap::new();
            metadata.insert("feishu.chat_id".into(), chat_id.clone());
            metadata.insert("feishu.open_id".into(), open_id.clone());
            metadata.insert("feishu.session_id".into(), session_id);
            metadata.insert("feishu.receive_id_type".into(), "chat_id".into());
            metadata.insert("feishu.message_type".into(), msg_type.to_string());
            let channel_id = if chat_id.is_empty() {
                open_id.clone()
            } else {
                chat_id
            };
            Ok(IngressEvent::Message(Box::new(ChannelMessage {
                id: MessageId::new(if msg_id.is_empty() {
                    uuid::Uuid::new_v4().to_string()
                } else {
                    msg_id
                }),
                conversation: ConversationId {
                    platform: "feishu".into(),
                    channel_id,
                    server_id: None,
                },
                author: open_id,
                content: MessageContent::Text(text),
                thread_id: None,
                reply_to: None,
                timestamp,
                attachments: Vec::new(),
                metadata,
            })))
        }
        "card.action.trigger" => {
            let action_tag = ev
                .get("action")
                .and_then(|a| a.get("tag"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let open_id = ev
                .get("operator")
                .and_then(|o| o.get("open_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut metadata = HashMap::new();
            metadata.insert("feishu.open_id".into(), open_id.clone());
            metadata.insert("feishu.event_type".into(), "card.action.trigger".into());
            Ok(IngressEvent::Message(Box::new(ChannelMessage {
                id: MessageId::new(uuid::Uuid::new_v4().to_string()),
                conversation: ConversationId {
                    platform: "feishu".into(),
                    channel_id: open_id.clone(),
                    server_id: None,
                },
                author: open_id,
                content: MessageContent::Text(action_tag),
                thread_id: None,
                reply_to: None,
                timestamp: Utc::now(),
                attachments: Vec::new(),
                metadata,
            })))
        }
        other => Ok(IngressEvent::Dropped {
            event_type: other.to_string(),
        }),
    }
}

/// Extract plain text from Feishu's `content` field (a JSON-encoded string).
///
/// `text` messages have `{"text": "..."}`; `post` (rich) messages have
/// a nested array structure. We flatten rich text best-effort — full
/// markdown rendering is out of scope here.
pub fn extract_content_text(msg_type: &str, content_raw: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(content_raw).unwrap_or(serde_json::Value::Null);
    match msg_type {
        "text" => v
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        "post" => {
            // content.title and content.content (2d array).
            let mut parts: Vec<String> = Vec::new();
            if let Some(title) = v.get("title").and_then(|t| t.as_str())
                && !title.is_empty()
            {
                parts.push(title.to_string());
            }
            if let Some(paragraphs) = v.get("content").and_then(|c| c.as_array()) {
                for para in paragraphs {
                    if let Some(line_items) = para.as_array() {
                        let line: String = line_items
                            .iter()
                            .filter_map(|el| {
                                el.get("text")
                                    .and_then(|t| t.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        if !line.is_empty() {
                            parts.push(line);
                        }
                    }
                }
            }
            parts.join("\n")
        }
        "image" | "audio" | "media" | "file" | "sticker" => format!("[{msg_type}]"),
        other => format!("[{other}]"),
    }
}

/// Build the outbound body posted to `/open-apis/im/v1/messages`.
pub fn build_send_body(receive_id: &str, text: &str) -> serde_json::Value {
    json!({
        "receive_id": receive_id,
        "msg_type": "text",
        "content": serde_json::to_string(&json!({"text": text})).unwrap(),
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_text_message() {
        let content = serde_json::to_string(&json!({"text": "hi lark"})).unwrap();
        let body = json!({
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_abc"}},
                "message": {
                    "chat_id": "oc_xyz",
                    "message_id": "om_1",
                    "message_type": "text",
                    "create_time": "1700000000000",
                    "content": content,
                }
            }
        });
        match parse_event(&body).unwrap() {
            IngressEvent::Message(m) => {
                assert_eq!(m.id.0, "om_1");
                assert_eq!(m.conversation.channel_id, "oc_xyz");
                assert_eq!(m.author, "ou_abc");
                match m.content {
                    MessageContent::Text(t) => assert_eq!(t, "hi lark"),
                    _ => panic!(),
                }
                assert_eq!(
                    m.metadata.get("feishu.receive_id_type").map(String::as_str),
                    Some("chat_id")
                );
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_post_flattens_text() {
        let content = serde_json::to_string(&json!({
            "title": "Title",
            "content": [
                [{"tag":"text","text":"line1-a"},{"tag":"text","text":" line1-b"}],
                [{"tag":"a","href":"x","text":"link"}]
            ]
        }))
        .unwrap();
        let body = json!({
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_a"}},
                "message": {
                    "chat_id": "oc_a",
                    "message_id": "m",
                    "message_type": "post",
                    "create_time": "0",
                    "content": content,
                }
            }
        });
        match parse_event(&body).unwrap() {
            IngressEvent::Message(m) => match m.content {
                MessageContent::Text(t) => {
                    assert!(t.contains("Title"));
                    assert!(t.contains("line1-a line1-b"));
                    assert!(t.contains("link"));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn parse_message_read_is_dropped() {
        let body = json!({
            "schema": "2.0",
            "header": {"event_type": "im.message.message_read_v1"},
            "event": {}
        });
        match parse_event(&body).unwrap() {
            IngressEvent::Dropped { event_type } => {
                assert_eq!(event_type, "im.message.message_read_v1")
            }
            _ => panic!(),
        }
    }

    #[test]
    fn send_body_has_stringified_content() {
        let v = build_send_body("oc_1", "hi");
        assert_eq!(v["receive_id"], "oc_1");
        assert_eq!(v["msg_type"], "text");
        let c = v["content"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(c).unwrap();
        assert_eq!(inner["text"], "hi");
    }

    #[test]
    fn extract_content_text_image() {
        assert_eq!(extract_content_text("image", "{}"), "[image]");
    }
}
