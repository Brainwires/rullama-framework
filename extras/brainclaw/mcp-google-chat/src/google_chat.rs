//! Google Chat REST client + `Channel` trait implementation.
//!
//! Ingress event types handled in `webhook.rs`:
//!
//! - `MESSAGE` — user DM or @mention → forwarded as user message.
//! - `ADDED_TO_SPACE` / `REMOVED_FROM_SPACE` — lifecycle events → logged
//!   only, not forwarded.
//! - `CARD_CLICKED` — adaptive-card button press → forwarded as user
//!   message whose text is the action id.
//!
//! Egress uses `POST https://chat.googleapis.com/v1/spaces/{space}/messages`
//! with an OAuth bearer from [`crate::oauth::TokenMinter`].

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
    ThreadId,
};

use crate::oauth::TokenMinter;

/// Base URL for the Google Chat REST API.
pub const CHAT_API_BASE: &str = "https://chat.googleapis.com/v1";

/// `Channel` implementation for Google Chat.
pub struct GoogleChatChannel {
    minter: Arc<TokenMinter>,
    http: reqwest::Client,
    api_base: String,
}

impl GoogleChatChannel {
    /// Create a new channel wired to the given token minter.
    pub fn new(minter: Arc<TokenMinter>) -> Self {
        Self {
            minter,
            http: reqwest::Client::new(),
            api_base: CHAT_API_BASE.to_string(),
        }
    }

    /// Override the API base URL — tests only.
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }
}

#[async_trait]
impl Channel for GoogleChatChannel {
    fn channel_type(&self) -> &str {
        "google_chat"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        // Markdown subset, mentions, threads (one space ≈ one thread). No
        // reactions (Chat has them, but the bot API doesn't expose add).
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::THREADS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let bearer = self
            .minter
            .bearer()
            .await
            .context("mint OAuth bearer for Chat send_message")?;
        let space = &target.channel_id;
        let url = format!(
            "{}/spaces/{}/messages",
            self.api_base,
            urlencoding::encode(space)
        );

        let body = build_reply_body(message);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&bearer)
            .json(&body)
            .send()
            .await
            .context("POST Chat message")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Chat API returned {status}: {} bytes", body.len());
        }

        #[derive(Deserialize)]
        struct Resp {
            name: String,
        }
        let parsed: Resp = resp.json().await.context("parse Chat API response")?;
        Ok(MessageId::new(parsed.name))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        // Google Chat supports message updates via PUT, but the bot API
        // requires the original message name + `updateMask`. We don't
        // track original names past send here — flag as unsupported
        // rather than fabricating a no-op. Callers that need updates can
        // hit the REST API directly via the MCP tool layer.
        anyhow::bail!("edit_message is not yet implemented for google_chat");
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        let bearer = self.minter.bearer().await?;
        // Message IDs on ingress are the Chat REST "name", e.g.
        // `spaces/AAAA/messages/BBBB`. DELETE against that path.
        let url = format!("{}/{}", self.api_base, id.0);
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&bearer)
            .send()
            .await
            .context("DELETE Chat message")?;
        if !resp.status().is_success() {
            anyhow::bail!("Chat DELETE returned {}", resp.status());
        }
        Ok(())
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        // Google Chat bots can't broadcast typing state. No-op.
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("add_reaction is not supported by the Google Chat bot API");
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        // `spaces.messages.list` requires `chat.messages.readonly` scope
        // which is not provided by the bot scope. Return empty instead of
        // panicking — callers can fall back to the MCP layer if they've
        // granted the wider scope.
        Ok(Vec::new())
    }
}

// ── Event parsing ───────────────────────────────────────────────────────

/// A single ingress event type, as it appears in the Google-signed POST body.
#[derive(Debug, Clone)]
pub enum IngressEvent {
    /// A user sent a message or @mentioned the bot.
    Message(ChannelMessage),
    /// Bot was added or removed from a space (log-only).
    Lifecycle { space_id: String, added: bool },
    /// A button on a card was clicked; text is the action id.
    CardClicked(ChannelMessage),
    /// Any other event type we don't handle — logged and dropped.
    Other { event_type: String },
}

/// Parse an incoming Google Chat event payload.
///
/// This accepts both the "classic" event shape
/// (`{"type": "MESSAGE", "message": {..}, ..}`) and the Apps-Script-style
/// nested shape (`{"chat": {"messagePayload": {..}}}`). We prefer the
/// classic shape — the newer one is rejected loudly rather than silently
/// mis-parsed.
pub fn parse_event(body: &serde_json::Value) -> anyhow::Result<IngressEvent> {
    let event_type = body
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("event body missing `type`"))?;

    match event_type {
        "MESSAGE" => {
            let msg = body
                .get("message")
                .ok_or_else(|| anyhow::anyhow!("MESSAGE event missing `message`"))?;
            Ok(IngressEvent::Message(chat_message_to_channel_message(msg)?))
        }
        "CARD_CLICKED" => {
            let space_name = body
                .get("space")
                .and_then(|s| s.get("name"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("CARD_CLICKED event missing space.name"))?;
            let action_id = body
                .get("action")
                .and_then(|a| a.get("actionMethodName"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let user = body
                .get("user")
                .and_then(|u| u.get("displayName"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let space_id = extract_space_id(space_name);
            Ok(IngressEvent::CardClicked(ChannelMessage {
                id: MessageId::new(format!("card:{}:{}", space_id, action_id)),
                conversation: ConversationId {
                    platform: "google_chat".into(),
                    channel_id: space_id,
                    server_id: None,
                },
                author: user,
                content: MessageContent::Text(action_id.to_string()),
                thread_id: None,
                reply_to: None,
                timestamp: Utc::now(),
                attachments: vec![],
                metadata: HashMap::new(),
            }))
        }
        "ADDED_TO_SPACE" | "REMOVED_FROM_SPACE" => {
            let space_name = body
                .get("space")
                .and_then(|s| s.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(IngressEvent::Lifecycle {
                space_id: extract_space_id(space_name),
                added: event_type == "ADDED_TO_SPACE",
            })
        }
        other => Ok(IngressEvent::Other {
            event_type: other.to_string(),
        }),
    }
}

/// Convert an inbound Chat `Message` resource into a `ChannelMessage`.
pub fn chat_message_to_channel_message(msg: &serde_json::Value) -> anyhow::Result<ChannelMessage> {
    let name = msg
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("message missing `name`"))?
        .to_string();

    let space = msg
        .get("space")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let space_id = extract_space_id(space);

    let sender_name = msg
        .get("sender")
        .and_then(|s| s.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sender_id = extract_user_id(&sender_name);

    let display_name = msg
        .get("sender")
        .and_then(|s| s.get("displayName"))
        .and_then(|v| v.as_str())
        .unwrap_or(&sender_id)
        .to_string();

    // `argumentText` is the message text with bot mentions removed. Fall
    // back to raw `text` if unavailable.
    let text = msg
        .get("argumentText")
        .and_then(|v| v.as_str())
        .or_else(|| msg.get("text").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();

    let thread_id = msg
        .get("thread")
        .and_then(|t| t.get("name"))
        .and_then(|v| v.as_str())
        .map(|s| ThreadId::new(s.to_string()));

    let create_time = msg
        .get("createTime")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    Ok(ChannelMessage {
        id: MessageId::new(name),
        conversation: ConversationId {
            platform: "google_chat".into(),
            channel_id: space_id,
            server_id: None,
        },
        author: display_name,
        content: MessageContent::Text(text),
        thread_id,
        reply_to: None,
        timestamp: create_time,
        attachments: vec![],
        metadata: {
            let mut m = HashMap::new();
            if !sender_id.is_empty() {
                m.insert("google_chat.user_id".into(), sender_id);
            }
            m
        },
    })
}

fn extract_space_id(name: &str) -> String {
    // `spaces/AAAAAAAAAAA` → `AAAAAAAAAAA`.
    name.strip_prefix("spaces/").unwrap_or(name).to_string()
}

fn extract_user_id(name: &str) -> String {
    // `users/123456` → `123456`.
    name.strip_prefix("users/").unwrap_or(name).to_string()
}

/// Build the JSON body for a Chat `spaces.messages.create` call.
pub fn build_reply_body(message: &ChannelMessage) -> serde_json::Value {
    let text = match &message.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::RichText { markdown, .. } => markdown.clone(),
        MessageContent::Media(m) => {
            if let Some(caption) = &m.caption {
                format!("{}\n{}", caption, m.url)
            } else {
                m.url.clone()
            }
        }
        MessageContent::Embed(e) => {
            let mut parts = Vec::new();
            if let Some(t) = &e.title {
                parts.push(format!("*{t}*"));
            }
            if let Some(d) = &e.description {
                parts.push(d.clone());
            }
            for f in &e.fields {
                parts.push(format!("*{}*: {}", f.name, f.value));
            }
            parts.join("\n")
        }
        MessageContent::Mixed(items) => items
            .iter()
            .map(|c| {
                let stub = ChannelMessage {
                    content: c.clone(),
                    id: message.id.clone(),
                    conversation: message.conversation.clone(),
                    author: message.author.clone(),
                    thread_id: None,
                    reply_to: None,
                    timestamp: message.timestamp,
                    attachments: vec![],
                    metadata: HashMap::new(),
                };
                build_reply_body(&stub)
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };

    if let Some(thread) = &message.thread_id {
        json!({
            "text": text,
            "thread": { "name": thread.0 },
        })
    } else {
        json!({ "text": text })
    }
}

/// Request type for the MCP `send_message` tool.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendMessageRequest {
    /// The Chat space id (without the `spaces/` prefix).
    pub space_id: String,
    /// The message text to send.
    pub text: String,
    /// Optional thread name to reply into (full `spaces/.../threads/...` form).
    pub thread_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_message_event() {
        let body = json!({
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAA/messages/BBB",
                "sender": {
                    "name": "users/12345",
                    "displayName": "Alice",
                },
                "space": { "name": "spaces/AAA" },
                "text": "@bot hello world",
                "argumentText": "hello world",
                "createTime": "2025-06-01T12:00:00Z",
                "thread": { "name": "spaces/AAA/threads/T1" }
            }
        });
        let ev = parse_event(&body).unwrap();
        let msg = match ev {
            IngressEvent::Message(m) => m,
            _ => panic!("expected message"),
        };
        assert_eq!(msg.id.0, "spaces/AAA/messages/BBB");
        assert_eq!(msg.conversation.channel_id, "AAA");
        assert_eq!(msg.author, "Alice");
        match msg.content {
            MessageContent::Text(t) => assert_eq!(t, "hello world"),
            _ => panic!("expected text"),
        }
        assert_eq!(msg.thread_id.as_ref().unwrap().0, "spaces/AAA/threads/T1");
        assert_eq!(
            msg.metadata.get("google_chat.user_id").map(String::as_str),
            Some("12345")
        );
    }

    #[test]
    fn parse_added_to_space_is_lifecycle() {
        let body = json!({
            "type": "ADDED_TO_SPACE",
            "space": { "name": "spaces/XYZ" }
        });
        let ev = parse_event(&body).unwrap();
        match ev {
            IngressEvent::Lifecycle { space_id, added } => {
                assert_eq!(space_id, "XYZ");
                assert!(added);
            }
            _ => panic!("expected lifecycle"),
        }
    }

    #[test]
    fn parse_card_clicked_returns_action_id() {
        let body = json!({
            "type": "CARD_CLICKED",
            "space": { "name": "spaces/S" },
            "user": { "displayName": "Bob" },
            "action": { "actionMethodName": "approve_request" }
        });
        let ev = parse_event(&body).unwrap();
        let msg = match ev {
            IngressEvent::CardClicked(m) => m,
            _ => panic!("expected card clicked"),
        };
        assert_eq!(msg.conversation.channel_id, "S");
        match msg.content {
            MessageContent::Text(t) => assert_eq!(t, "approve_request"),
            _ => panic!(),
        }
        assert_eq!(msg.author, "Bob");
    }

    #[test]
    fn parse_missing_type_errors() {
        let body = json!({ "message": {} });
        assert!(parse_event(&body).is_err());
    }

    #[test]
    fn parse_unknown_event_is_other() {
        let body = json!({ "type": "SOMETHING_ELSE" });
        let ev = parse_event(&body).unwrap();
        match ev {
            IngressEvent::Other { event_type } => assert_eq!(event_type, "SOMETHING_ELSE"),
            _ => panic!(),
        }
    }

    #[test]
    fn build_reply_plain_text() {
        let msg = sample_message(MessageContent::Text("hi there".into()));
        let body = build_reply_body(&msg);
        assert_eq!(body["text"], "hi there");
        assert!(body.get("thread").is_none());
    }

    #[test]
    fn build_reply_with_thread() {
        let mut msg = sample_message(MessageContent::Text("reply".into()));
        msg.thread_id = Some(ThreadId::new("spaces/S/threads/T"));
        let body = build_reply_body(&msg);
        assert_eq!(body["text"], "reply");
        assert_eq!(body["thread"]["name"], "spaces/S/threads/T");
    }

    #[test]
    fn build_reply_rich_text_uses_markdown() {
        let msg = sample_message(MessageContent::RichText {
            markdown: "*bold*".into(),
            fallback_plain: "bold".into(),
        });
        let body = build_reply_body(&msg);
        assert_eq!(body["text"], "*bold*");
    }

    #[test]
    fn capabilities_contain_expected_flags() {
        let key = crate::oauth::ServiceAccountKey {
            client_email: "x@y.iam.gserviceaccount.com".into(),
            private_key: "-----BEGIN PRIVATE KEY-----\nQUJD\n-----END PRIVATE KEY-----\n".into(),
            token_uri: None,
            account_type: Some("service_account".into()),
        };
        let minter = Arc::new(crate::oauth::TokenMinter::from_key(key, "scope").unwrap());
        let chan = GoogleChatChannel::new(minter);
        let caps = chan.capabilities();
        assert!(caps.contains(ChannelCapabilities::RICH_TEXT));
        assert!(caps.contains(ChannelCapabilities::MENTIONS));
        assert!(caps.contains(ChannelCapabilities::THREADS));
        assert!(!caps.contains(ChannelCapabilities::REACTIONS));
        assert!(!caps.contains(ChannelCapabilities::VOICE));
        assert_eq!(chan.channel_type(), "google_chat");
    }

    fn sample_message(content: MessageContent) -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new("m"),
            conversation: ConversationId {
                platform: "google_chat".into(),
                channel_id: "S".into(),
                server_id: None,
            },
            author: "test".into(),
            content,
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        }
    }
}
