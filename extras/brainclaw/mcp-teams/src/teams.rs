//! Microsoft Teams `Channel` trait impl and Activity parser.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
    ThreadId,
};

use crate::oauth::BotTokenMinter;

/// In-memory store of `(conversation_id -> serviceUrl)` mappings.
///
/// The Bot Framework issues a different `serviceUrl` per region; replies
/// must POST back to the `serviceUrl` embedded in the inbound activity.
/// Entries expire 24h after the last recorded inbound activity.
#[derive(Debug, Default)]
pub struct ServiceUrlStore {
    entries: DashMap<String, ServiceUrlEntry>,
}

#[derive(Debug, Clone)]
struct ServiceUrlEntry {
    service_url: String,
    last_seen: std::time::Instant,
}

impl ServiceUrlStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Record (or refresh) a `serviceUrl` for a conversation.
    pub fn record(&self, conversation_id: &str, service_url: &str) {
        self.entries.insert(
            conversation_id.to_string(),
            ServiceUrlEntry {
                service_url: service_url.to_string(),
                last_seen: std::time::Instant::now(),
            },
        );
    }

    /// Look up a `serviceUrl`, pruning if older than 24h.
    pub fn get(&self, conversation_id: &str) -> Option<String> {
        let entry = self.entries.get(conversation_id)?;
        if entry.last_seen.elapsed() > Duration::from_secs(24 * 3600) {
            drop(entry);
            self.entries.remove(conversation_id);
            return None;
        }
        Some(entry.service_url.clone())
    }
}

/// Teams channel implementation.
pub struct TeamsChannel {
    minter: Arc<BotTokenMinter>,
    service_urls: Arc<ServiceUrlStore>,
    bot_app_id: String,
    http: reqwest::Client,
}

impl TeamsChannel {
    /// Construct a new Teams channel.
    pub fn new(
        bot_app_id: impl Into<String>,
        minter: Arc<BotTokenMinter>,
        service_urls: Arc<ServiceUrlStore>,
    ) -> Self {
        Self {
            minter,
            service_urls,
            bot_app_id: bot_app_id.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    fn service_url_for(&self, conversation_id: &str) -> Option<String> {
        self.service_urls.get(conversation_id)
    }
}

#[async_trait]
impl Channel for TeamsChannel {
    fn channel_type(&self) -> &str {
        "teams"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        // Markdown only (adaptive-card rendering deferred), mentions, threads.
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::THREADS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let service_url = self
            .service_url_for(&target.channel_id)
            .ok_or_else(|| anyhow::anyhow!(
                "no serviceUrl recorded for conversation {} — only reachable after an inbound activity",
                target.channel_id
            ))?;

        let bearer = self.minter.bearer().await.context("mint Teams bearer")?;
        let activity = build_reply_activity(&self.bot_app_id, target, message);
        let url = format!(
            "{}/v3/conversations/{}/activities",
            service_url.trim_end_matches('/'),
            urlencoding::encode(&target.channel_id)
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&bearer)
            .json(&activity)
            .send()
            .await
            .context("POST Teams activity")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Teams API returned {status}: {} bytes", body.len());
        }

        #[derive(Deserialize)]
        struct Resp {
            #[serde(default)]
            id: Option<String>,
        }
        let parsed: Resp = resp.json().await.unwrap_or(Resp { id: None });
        Ok(MessageId::new(parsed.id.unwrap_or_default()))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        anyhow::bail!("edit_message not implemented for Teams");
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        anyhow::bail!("delete_message not implemented for Teams");
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        Ok(())
    }

    async fn add_reaction(&self, _id: &MessageId, _emoji: &str) -> Result<()> {
        anyhow::bail!("reactions not implemented for Teams");
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        Ok(Vec::new())
    }
}

// ── Activity parsing ────────────────────────────────────────────────────

/// The result of parsing an inbound Bot Framework activity.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ActivityEvent {
    /// Forward as a user message.
    Message(ChannelMessage),
    /// Lifecycle change — record or drop.
    ConversationUpdate {
        /// Conversation id from the activity.
        conversation_id: String,
        /// Service URL to record.
        service_url: String,
    },
    /// Drop silently.
    Ignore { activity_type: String },
}

/// Parse a raw Bot Framework Activity JSON payload.
pub fn parse_activity(body: &serde_json::Value) -> anyhow::Result<ActivityEvent> {
    let activity_type = body
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("activity missing `type`"))?;

    let service_url = body
        .get("serviceUrl")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let conversation_id = body
        .get("conversation")
        .and_then(|c| c.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    match activity_type {
        "message" => {
            let from_id = body
                .get("from")
                .and_then(|f| f.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let from_name = body
                .get("from")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or(&from_id)
                .to_string();
            let text = body
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let id = body
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let timestamp = body
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            // Teams "threads" surface as conversation.id with a `;messageid=` suffix
            // on the first message of a thread. The thread id itself lives in
            // `conversation.id` — we keep it there and leave thread_id empty for
            // ad-hoc channel messages.
            let thread_id = body
                .get("channelData")
                .and_then(|c| c.get("teamsThreadId"))
                .and_then(|v| v.as_str())
                .map(|s| ThreadId::new(s.to_string()));

            Ok(ActivityEvent::Message(ChannelMessage {
                id: MessageId::new(id),
                conversation: ConversationId {
                    platform: "teams".into(),
                    channel_id: conversation_id,
                    server_id: None,
                },
                author: from_name,
                content: MessageContent::Text(text),
                thread_id,
                reply_to: None,
                timestamp,
                attachments: vec![],
                metadata: {
                    let mut m = HashMap::new();
                    if !from_id.is_empty() {
                        m.insert("teams.user_id".into(), from_id);
                    }
                    if !service_url.is_empty() {
                        m.insert("teams.service_url".into(), service_url);
                    }
                    m
                },
            }))
        }
        "invoke" => {
            // Adaptive-card actions: the `value` object holds the action
            // payload. We serialise it as a JSON string so agents see it.
            let value = body
                .get("value")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let text = value.to_string();
            let from_name = body
                .get("from")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let id = body
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(ActivityEvent::Message(ChannelMessage {
                id: MessageId::new(id),
                conversation: ConversationId {
                    platform: "teams".into(),
                    channel_id: conversation_id,
                    server_id: None,
                },
                author: from_name,
                content: MessageContent::Text(text),
                thread_id: None,
                reply_to: None,
                timestamp: Utc::now(),
                attachments: vec![],
                metadata: HashMap::from([("teams.invoke".into(), "true".into())]),
            }))
        }
        "conversationUpdate" => Ok(ActivityEvent::ConversationUpdate {
            conversation_id,
            service_url,
        }),
        other => Ok(ActivityEvent::Ignore {
            activity_type: other.to_string(),
        }),
    }
}

/// Build the JSON body for a reply Activity (POST to `serviceUrl`).
pub fn build_reply_activity(
    bot_app_id: &str,
    target: &ConversationId,
    message: &ChannelMessage,
) -> serde_json::Value {
    let text = match &message.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::RichText { markdown, .. } => markdown.clone(),
        MessageContent::Media(m) => m
            .caption
            .clone()
            .map(|c| format!("{c}\n{}", m.url))
            .unwrap_or_else(|| m.url.clone()),
        MessageContent::Embed(e) => {
            let mut parts = Vec::new();
            if let Some(t) = &e.title {
                parts.push(format!("**{t}**"));
            }
            if let Some(d) = &e.description {
                parts.push(d.clone());
            }
            for f in &e.fields {
                parts.push(format!("**{}**: {}", f.name, f.value));
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
                build_reply_activity(bot_app_id, target, &stub)
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    json!({
        "type": "message",
        "from": { "id": bot_app_id, "name": "BrainClaw" },
        "conversation": { "id": target.channel_id },
        "text": text,
        "textFormat": "markdown",
    })
}

/// MCP tool request.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SendMessageRequest {
    /// The Teams conversation id.
    pub conversation_id: String,
    /// The markdown text to send.
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_url_store_round_trip() {
        let store = ServiceUrlStore::new();
        store.record("conv-1", "https://smba.region.example/");
        assert_eq!(
            store.get("conv-1").as_deref(),
            Some("https://smba.region.example/")
        );
    }

    #[test]
    fn service_url_missing_returns_none() {
        let store = ServiceUrlStore::new();
        assert!(store.get("unknown").is_none());
    }

    #[test]
    fn parse_message_activity_forwards() {
        let body = json!({
            "type": "message",
            "id": "act-1",
            "serviceUrl": "https://smba.trafficmanager.net/",
            "conversation": { "id": "a:12345" },
            "from": { "id": "29:abcd", "name": "Alice" },
            "text": "hi",
            "timestamp": "2025-03-01T00:00:00Z"
        });
        let ev = parse_activity(&body).unwrap();
        let msg = match ev {
            ActivityEvent::Message(m) => m,
            _ => panic!("expected Message"),
        };
        assert_eq!(msg.author, "Alice");
        assert_eq!(msg.conversation.channel_id, "a:12345");
        match msg.content {
            MessageContent::Text(t) => assert_eq!(t, "hi"),
            _ => panic!(),
        }
        assert_eq!(
            msg.metadata.get("teams.service_url").map(String::as_str),
            Some("https://smba.trafficmanager.net/")
        );
    }

    #[test]
    fn parse_conversation_update_produces_lifecycle() {
        let body = json!({
            "type": "conversationUpdate",
            "conversation": { "id": "a:1" },
            "serviceUrl": "https://svc.example/"
        });
        match parse_activity(&body).unwrap() {
            ActivityEvent::ConversationUpdate {
                conversation_id,
                service_url,
            } => {
                assert_eq!(conversation_id, "a:1");
                assert_eq!(service_url, "https://svc.example/");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_invoke_forwards_value_as_json_text() {
        let body = json!({
            "type": "invoke",
            "id": "i-1",
            "conversation": { "id": "a:1" },
            "from": { "name": "Bob" },
            "value": { "action": "approve", "id": 42 }
        });
        let ev = parse_activity(&body).unwrap();
        let msg = match ev {
            ActivityEvent::Message(m) => m,
            _ => panic!(),
        };
        match msg.content {
            MessageContent::Text(t) => {
                assert!(t.contains("approve"));
                assert!(t.contains("42"));
            }
            _ => panic!(),
        }
        assert_eq!(
            msg.metadata.get("teams.invoke").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn parse_typing_is_ignored() {
        let body = json!({ "type": "typing", "conversation": { "id": "a:1" } });
        match parse_activity(&body).unwrap() {
            ActivityEvent::Ignore { activity_type } => assert_eq!(activity_type, "typing"),
            _ => panic!(),
        }
    }

    #[test]
    fn build_reply_activity_includes_text_and_markdown_format() {
        let conv = ConversationId {
            platform: "teams".into(),
            channel_id: "conv-1".into(),
            server_id: None,
        };
        let msg = ChannelMessage {
            id: MessageId::new("m"),
            conversation: conv.clone(),
            author: "bot".into(),
            content: MessageContent::RichText {
                markdown: "**hi**".into(),
                fallback_plain: "hi".into(),
            },
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        };
        let act = build_reply_activity("app-id", &conv, &msg);
        assert_eq!(act["type"], "message");
        assert_eq!(act["text"], "**hi**");
        assert_eq!(act["textFormat"], "markdown");
        assert_eq!(act["from"]["id"], "app-id");
        assert_eq!(act["conversation"]["id"], "conv-1");
    }
}
