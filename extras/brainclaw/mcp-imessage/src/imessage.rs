//! BlueBubbles REST client and [`Channel`] implementation.
//!
//! The BlueBubbles server exposes iMessage as a REST API. We interact
//! with the bare subset we need:
//!
//! - `GET  /api/v1/message?password=<pw>[&limit=<n>&after=<guid>]`
//!   returns recent messages, optionally constrained after a guid cursor.
//! - `POST /api/v1/message/text?password=<pw>` sends plain text to a chat.
//! - `POST /api/v1/message/react?password=<pw>` adds a tapback / reaction.
//!
//! URL details:
//! - The server password is passed as a `password` query parameter. It
//!   is never logged or stored in the ChannelMessage.
//! - All JSON field names come from BlueBubbles' documented schema.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
};

/// Minimal parsed view of a BlueBubbles message row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BbMessage {
    /// GUID, used as the message id and polling cursor.
    pub guid: String,
    /// Optional text body.
    #[serde(default)]
    pub text: Option<String>,
    /// Unix epoch (milliseconds) the message was created.
    #[serde(default, rename = "dateCreated")]
    pub date_created_ms: Option<i64>,
    /// The chat guid(s) the message belongs to. BlueBubbles ships chats
    /// as an array of objects; we only need the first one's guid.
    #[serde(default)]
    pub chats: Vec<BbChat>,
    /// The handle record — the sender's phone/email id.
    #[serde(default)]
    pub handle: Option<BbHandle>,
    /// Whether the message was sent by the local Mac user. These should
    /// NOT be forwarded inbound.
    #[serde(default, rename = "isFromMe")]
    pub is_from_me: bool,
}

/// BlueBubbles chat record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BbChat {
    /// Chat GUID (e.g. `iMessage;-;+15551234567`).
    pub guid: String,
}

/// BlueBubbles handle record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BbHandle {
    /// Phone number or email address of the remote party.
    #[serde(default)]
    pub address: Option<String>,
}

/// The paginated BlueBubbles message list response.
#[derive(Debug, Clone, Deserialize)]
pub struct BbMessageList {
    /// The data payload — a list of messages.
    #[serde(default)]
    pub data: Vec<BbMessage>,
}

/// Request body for `send_message`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendTextRequest {
    /// Target chat GUID.
    pub chat_guid: String,
    /// Message text to send.
    pub text: String,
}

/// [`Channel`] implementation over a BlueBubbles server.
pub struct ImessageChannel {
    http: reqwest::Client,
    server_url: String,
    password: String,
}

impl ImessageChannel {
    /// Build a new channel wired to the given BlueBubbles server.
    pub fn new(server_url: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            server_url: server_url.into(),
            password: password.into(),
        }
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Encoded `password=<pw>` query fragment.
    fn auth_query(&self) -> String {
        format!("password={}", urlencoding::encode(&self.password))
    }

    /// Poll the BlueBubbles server for messages created after `after_guid`.
    ///
    /// Returns messages in BlueBubbles' native order; callers are
    /// expected to filter out `is_from_me` and advance the cursor.
    pub async fn poll_messages(
        &self,
        limit: u32,
        after_guid: Option<&str>,
    ) -> Result<Vec<BbMessage>> {
        let base = self.server_url.trim_end_matches('/');
        let mut url = format!("{base}/api/v1/message?{}", self.auth_query());
        url.push_str(&format!("&limit={limit}"));
        if let Some(g) = after_guid {
            url.push_str(&format!("&after={}", urlencoding::encode(g)));
        }
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("GET BlueBubbles /api/v1/message")?;
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            anyhow::bail!("BlueBubbles rate-limited (429)");
        }
        if !resp.status().is_success() {
            anyhow::bail!("BlueBubbles returned {}", resp.status());
        }
        let list: BbMessageList = resp
            .json()
            .await
            .context("parse BlueBubbles message list")?;
        Ok(list.data)
    }

    /// Add a tapback reaction to a message via the BlueBubbles
    /// `/api/v1/message/react` endpoint.
    pub async fn react(&self, chat_guid: &str, message_guid: &str, emoji: &str) -> Result<()> {
        let base = self.server_url.trim_end_matches('/');
        let url = format!("{base}/api/v1/message/react?{}", self.auth_query());
        let body = json!({
            "chatGuid": chat_guid,
            "selectedMessageGuid": message_guid,
            "reaction": emoji,
        });
        let resp = self.http.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("BlueBubbles react returned {}", resp.status());
        }
        Ok(())
    }
}

#[async_trait]
impl Channel for ImessageChannel {
    fn channel_type(&self) -> &str {
        "imessage"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        // Plain text + reactions (tapbacks) + URL-reference attachments.
        ChannelCapabilities::REACTIONS | ChannelCapabilities::DELETE_MESSAGES
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = extract_text(message);
        if text.is_empty() {
            anyhow::bail!("iMessage send_message: empty text payload");
        }
        let base = self.server_url.trim_end_matches('/');
        let url = format!("{base}/api/v1/message/text?{}", self.auth_query());
        let body = json!({
            "chatGuid": target.channel_id,
            "message": text,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("POST BlueBubbles /api/v1/message/text")?;
        let status = resp.status();
        if !status.is_success() {
            let bytes = resp.bytes().await.unwrap_or_default();
            anyhow::bail!("BlueBubbles send returned {status}: {} bytes", bytes.len());
        }
        let parsed: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
        let guid = parsed
            .get("data")
            .and_then(|d| d.get("guid"))
            .and_then(|g| g.as_str())
            .unwrap_or("")
            .to_string();
        Ok(MessageId::new(if guid.is_empty() {
            format!("sent:{}", uuid::Uuid::new_v4())
        } else {
            guid
        }))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("edit_message is not supported by BlueBubbles")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        // BlueBubbles doesn't expose a stable message delete API we can
        // rely on here; surface the limitation rather than silently no-op.
        anyhow::bail!("delete_message is not supported by BlueBubbles")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        // We need the chat guid, which isn't carried in MessageId. The
        // MCP layer has a dedicated `react` tool that takes both ids.
        anyhow::bail!("add_reaction needs the chat guid — use the MCP react tool")
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        Ok(Vec::new())
    }
}

/// Convert a BlueBubbles message to a [`ChannelMessage`], filtering out
/// sender=local and empty payloads.
pub fn bb_to_channel(msg: &BbMessage) -> Option<ChannelMessage> {
    if msg.is_from_me {
        return None;
    }
    let chat_guid = msg.chats.first()?.guid.clone();
    let text = msg.text.clone().unwrap_or_default();
    if text.is_empty() {
        return None;
    }
    let handle = msg
        .handle
        .as_ref()
        .and_then(|h| h.address.clone())
        .unwrap_or_default();
    let ts = msg
        .date_created_ms
        .and_then(|ms| Utc.timestamp_millis_opt(ms).single())
        .unwrap_or_else(Utc::now);
    let session_id = format!("imessage:{chat_guid}:{handle}");
    let mut metadata = HashMap::new();
    metadata.insert("imessage.chat_guid".into(), chat_guid.clone());
    if !handle.is_empty() {
        metadata.insert("imessage.handle".into(), handle.clone());
    }
    metadata.insert("imessage.session_id".into(), session_id);
    Some(ChannelMessage {
        id: MessageId::new(msg.guid.clone()),
        conversation: ConversationId {
            platform: "imessage".into(),
            channel_id: chat_guid,
            server_id: None,
        },
        author: if handle.is_empty() {
            "unknown".into()
        } else {
            handle
        },
        content: MessageContent::Text(text),
        thread_id: None,
        reply_to: None,
        timestamp: ts,
        attachments: Vec::new(),
        metadata,
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

/// Shape-check for [`build_send_body`] — small pure helper used by tests.
pub fn build_send_body(chat_guid: &str, text: &str) -> serde_json::Value {
    json!({ "chatGuid": chat_guid, "message": text })
}

/// Convenience time parser — exposed for tests.
pub fn parse_ms(ms: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_millis_opt(ms).single()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bb_text(
        guid: &str,
        chat: &str,
        handle: &str,
        text: &str,
        from_me: bool,
    ) -> BbMessage {
        BbMessage {
            guid: guid.into(),
            text: Some(text.into()),
            date_created_ms: Some(1_700_000_000_000),
            chats: vec![BbChat { guid: chat.into() }],
            handle: Some(BbHandle {
                address: Some(handle.into()),
            }),
            is_from_me: from_me,
        }
    }

    #[test]
    fn parse_inbound_builds_channel_message() {
        let bb = sample_bb_text("m1", "iMessage;-;+155501", "+155502", "hello", false);
        let m = bb_to_channel(&bb).expect("parsed");
        assert_eq!(m.id.0, "m1");
        assert_eq!(m.conversation.channel_id, "iMessage;-;+155501");
        assert_eq!(m.author, "+155502");
        match m.content {
            MessageContent::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected text"),
        }
        assert_eq!(
            m.metadata.get("imessage.chat_guid").map(String::as_str),
            Some("iMessage;-;+155501")
        );
    }

    #[test]
    fn skips_outbound_self_messages() {
        let bb = sample_bb_text("m2", "chat", "me", "hi", true);
        assert!(bb_to_channel(&bb).is_none());
    }

    #[test]
    fn skips_empty_bodies() {
        let mut bb = sample_bb_text("m3", "chat", "who", "", false);
        bb.text = None;
        assert!(bb_to_channel(&bb).is_none());
    }

    #[test]
    fn send_body_shape() {
        let v = build_send_body("CHAT_GUID", "hello");
        assert_eq!(v["chatGuid"], "CHAT_GUID");
        assert_eq!(v["message"], "hello");
    }

    #[test]
    fn extract_text_from_rich() {
        let msg = ChannelMessage {
            id: MessageId::new("x"),
            conversation: ConversationId {
                platform: "imessage".into(),
                channel_id: "c".into(),
                server_id: None,
            },
            author: "a".into(),
            content: MessageContent::RichText {
                markdown: "*b*".into(),
                fallback_plain: "b".into(),
            },
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: Vec::new(),
            metadata: HashMap::new(),
        };
        assert_eq!(extract_text(&msg), "b");
    }

    #[test]
    fn caps_has_reactions() {
        let ch = ImessageChannel::new("http://x", "pw");
        assert!(ch.capabilities().contains(ChannelCapabilities::REACTIONS));
        assert!(
            !ch.capabilities()
                .contains(ChannelCapabilities::TYPING_INDICATOR)
        );
        assert_eq!(ch.channel_type(), "imessage");
    }
}
