//! Slack bot implementation of the `Channel` trait.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde_json::Value;
use tokio::sync::RwLock;

use brainwires_network::channels::{
    Attachment, ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
    ThreadId,
};

/// Base URL for Slack Web API.
const SLACK_API_BASE: &str = "https://slack.com/api";

/// Slack channel adapter implementing the `Channel` trait.
///
/// Wraps a `reqwest::Client` and bot token to interact with the Slack Web API.
pub struct SlackChannel {
    /// HTTP client for Slack Web API calls.
    http: Client,
    /// Slack bot token (xoxb-...).
    bot_token: String,
    /// Cache of workspace/team IDs we've seen, keyed by channel ID.
    team_cache: Arc<RwLock<HashMap<String, String>>>,
}

impl SlackChannel {
    /// Create a new `SlackChannel` with the given bot token.
    pub fn new(bot_token: String) -> Self {
        Self {
            http: Client::new(),
            bot_token,
            team_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a team/channel mapping (called by the event handler).
    pub async fn register_team(&self, channel_id: &str, team_id: &str) {
        let mut cache = self.team_cache.write().await;
        cache.insert(channel_id.to_string(), team_id.to_string());
    }

    /// Make an authenticated POST request to a Slack Web API method.
    fn api_post<'a>(
        &'a self,
        method: &'a str,
        body: &'a Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/{}", SLACK_API_BASE, method);
            let response = self
                .http
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.bot_token))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(body)
                .send()
                .await
                .with_context(|| format!("Failed to call Slack API: {}", method))?;

            // Handle rate limiting
            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(1);
                tracing::warn!(
                    method = %method,
                    retry_after = retry_after,
                    "Slack API rate limited"
                );
                tokio::time::sleep(std::time::Duration::from_secs(retry_after)).await;
                // Retry once
                return self.api_post(method, body).await;
            }

            let json: Value = response
                .json()
                .await
                .with_context(|| format!("Failed to parse Slack API response for {}", method))?;

            if json.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                let error = json
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown_error");
                anyhow::bail!("Slack API error in {}: {}", method, error);
            }

            Ok(json)
        })
    }
}

#[async_trait]
impl brainwires_network::channels::Channel for SlackChannel {
    fn channel_type(&self) -> &str {
        "slack"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::THREADS
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::TYPING_INDICATOR
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::MENTIONS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = channel_message_to_slack_text(message);

        let mut body = serde_json::json!({
            "channel": target.channel_id,
            "text": text,
        });

        // If there's a thread, send as a threaded reply
        if let Some(ref thread_id) = message.thread_id {
            body["thread_ts"] = serde_json::Value::String(thread_id.0.clone());
        }

        let response = self.api_post("chat.postMessage", &body).await?;

        let ts = response
            .get("ts")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        Ok(MessageId::new(ts))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let text = channel_message_to_slack_text(message);

        let body = serde_json::json!({
            "channel": message.conversation.channel_id,
            "ts": id.0,
            "text": text,
        });

        self.api_post("chat.update", &body).await?;
        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        // Composite ID format: "channel_id:ts"
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (channel_id, ts) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            anyhow::bail!(
                "delete_message requires composite ID format 'channel_id:ts', got: {}",
                id.0
            );
        };

        let body = serde_json::json!({
            "channel": channel_id,
            "ts": ts,
        });

        self.api_post("chat.delete", &body).await?;
        Ok(())
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        // Slack doesn't have a direct typing indicator API for bots.
        // Typing indicators are automatic during message processing in Socket Mode.
        // This is a no-op but we log it for observability.
        tracing::debug!(
            "Slack typing indicator requested (no-op: Slack handles this automatically)"
        );
        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        // Composite ID format: "channel_id:ts"
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (channel_id, ts) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            anyhow::bail!(
                "add_reaction requires composite ID format 'channel_id:ts', got: {}",
                id.0
            );
        };

        // Slack expects emoji names without colons (e.g., "thumbsup" not ":thumbsup:")
        let emoji_name = emoji.trim_matches(':');

        let body = serde_json::json!({
            "channel": channel_id,
            "timestamp": ts,
            "name": emoji_name,
        });

        self.api_post("reactions.add", &body).await?;
        Ok(())
    }

    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        let body = serde_json::json!({
            "channel": target.channel_id,
            "limit": limit.min(200),
        });

        let response = self.api_post("conversations.history", &body).await?;

        let messages = response
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let team_id = target.server_id.as_deref().unwrap_or("unknown");

        Ok(messages
            .into_iter()
            .filter_map(|m| slack_message_to_channel_message(&m, &target.channel_id, team_id).ok())
            .collect())
    }
}

// -- Conversion helpers -----------------------------------------------------------

/// Convert a Slack message JSON payload to a `ChannelMessage`.
pub fn slack_message_to_channel_message(
    msg: &Value,
    channel_id: &str,
    team_id: &str,
) -> Result<ChannelMessage> {
    let ts = msg.get("ts").and_then(|v| v.as_str()).unwrap_or("0.000000");

    let user = msg
        .get("user")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let text = msg.get("text").and_then(|v| v.as_str()).unwrap_or("");

    let thread_ts = msg
        .get("thread_ts")
        .and_then(|v| v.as_str())
        .map(ThreadId::new);

    // Parse attachments/files
    let attachments = msg
        .get("files")
        .and_then(|v| v.as_array())
        .map(|files| {
            files
                .iter()
                .filter_map(|f| {
                    let filename = f.get("name").and_then(|v| v.as_str()).unwrap_or("file");
                    let mimetype = f
                        .get("mimetype")
                        .and_then(|v| v.as_str())
                        .unwrap_or("application/octet-stream");
                    let url = f.get("url_private").and_then(|v| v.as_str()).unwrap_or("");
                    let size = f.get("size").and_then(|v| v.as_u64());

                    if url.is_empty() {
                        return None;
                    }

                    Some(Attachment {
                        filename: filename.to_string(),
                        content_type: mimetype.to_string(),
                        url: url.to_string(),
                        size_bytes: size,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Parse timestamp from Slack ts format (e.g., "1234567890.123456")
    let timestamp = parse_slack_ts(ts);

    Ok(ChannelMessage {
        id: MessageId::new(ts),
        conversation: ConversationId {
            platform: "slack".to_string(),
            channel_id: channel_id.to_string(),
            server_id: Some(team_id.to_string()),
        },
        author: user.to_string(),
        content: MessageContent::Text(text.to_string()),
        thread_id: thread_ts,
        reply_to: None,
        timestamp,
        attachments,
        metadata: HashMap::new(),
    })
}

/// Convert a `ChannelMessage` to Slack-compatible text content.
pub fn channel_message_to_slack_text(message: &ChannelMessage) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::RichText {
            markdown,
            fallback_plain: _,
        } => {
            // Slack uses mrkdwn (its own markdown variant) natively
            markdown.clone()
        }
        MessageContent::Media(media) => {
            if let Some(caption) = &media.caption {
                format!("{}\n{}", caption, media.url)
            } else {
                media.url.clone()
            }
        }
        MessageContent::Embed(embed) => {
            let mut parts = Vec::new();
            if let Some(title) = &embed.title {
                parts.push(format!("*{}*", title));
            }
            if let Some(desc) = &embed.description {
                parts.push(desc.clone());
            }
            for field in &embed.fields {
                parts.push(format!("*{}*: {}", field.name, field.value));
            }
            parts.join("\n")
        }
        MessageContent::Mixed(items) => {
            let sub: Vec<String> = items
                .iter()
                .map(|item| {
                    let temp = ChannelMessage {
                        content: item.clone(),
                        id: message.id.clone(),
                        conversation: message.conversation.clone(),
                        author: message.author.clone(),
                        thread_id: None,
                        reply_to: None,
                        timestamp: message.timestamp,
                        attachments: vec![],
                        metadata: HashMap::new(),
                    };
                    channel_message_to_slack_text(&temp)
                })
                .collect();
            sub.join("\n")
        }
    }
}

/// Parse a Slack timestamp string (e.g., "1234567890.123456") into a `DateTime<Utc>`.
pub fn parse_slack_ts(ts: &str) -> DateTime<Utc> {
    let secs = ts
        .split('.')
        .next()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);

    let micros = ts
        .split('.')
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    DateTime::from_timestamp(secs, micros * 1000).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_network::channels::{
        Channel, ChannelCapabilities, EmbedField, EmbedPayload, MediaPayload, MediaType,
    };
    use chrono::Utc;

    fn sample_conversation() -> ConversationId {
        ConversationId {
            platform: "slack".to_string(),
            channel_id: "C0123456789".to_string(),
            server_id: Some("T0123456789".to_string()),
        }
    }

    fn sample_channel_message(content: MessageContent) -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new("1234567890.123456"),
            conversation: sample_conversation(),
            author: "U0123456789".to_string(),
            content,
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn capabilities_returns_expected_flags() {
        let channel = SlackChannel::new("xoxb-test".to_string());
        let caps = channel.capabilities();

        assert!(caps.contains(ChannelCapabilities::RICH_TEXT));
        assert!(caps.contains(ChannelCapabilities::MEDIA_UPLOAD));
        assert!(caps.contains(ChannelCapabilities::THREADS));
        assert!(caps.contains(ChannelCapabilities::REACTIONS));
        assert!(caps.contains(ChannelCapabilities::TYPING_INDICATOR));
        assert!(caps.contains(ChannelCapabilities::EDIT_MESSAGES));
        assert!(caps.contains(ChannelCapabilities::DELETE_MESSAGES));
        assert!(caps.contains(ChannelCapabilities::MENTIONS));

        // Not supported
        assert!(!caps.contains(ChannelCapabilities::VOICE));
        assert!(!caps.contains(ChannelCapabilities::VIDEO));
        assert!(!caps.contains(ChannelCapabilities::READ_RECEIPTS));
        assert!(!caps.contains(ChannelCapabilities::EMBEDS));
    }

    #[test]
    fn channel_type_is_slack() {
        let channel = SlackChannel::new("xoxb-test".to_string());
        assert_eq!(channel.channel_type(), "slack");
    }

    #[test]
    fn text_content_to_slack() {
        let msg = sample_channel_message(MessageContent::Text("Hello world".to_string()));
        let result = channel_message_to_slack_text(&msg);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn rich_text_content_to_slack() {
        let msg = sample_channel_message(MessageContent::RichText {
            markdown: "*bold* and _italic_".to_string(),
            fallback_plain: "bold and italic".to_string(),
        });
        let result = channel_message_to_slack_text(&msg);
        assert_eq!(result, "*bold* and _italic_");
    }

    #[test]
    fn media_content_to_slack() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: Some("Check this out".to_string()),
            thumbnail_url: None,
        }));
        let result = channel_message_to_slack_text(&msg);
        assert_eq!(result, "Check this out\nhttps://example.com/image.png");
    }

    #[test]
    fn media_content_no_caption_to_slack() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: None,
            thumbnail_url: None,
        }));
        let result = channel_message_to_slack_text(&msg);
        assert_eq!(result, "https://example.com/image.png");
    }

    #[test]
    fn embed_content_to_slack() {
        let msg = sample_channel_message(MessageContent::Embed(EmbedPayload {
            title: Some("My Embed".to_string()),
            description: Some("A description".to_string()),
            url: None,
            color: None,
            fields: vec![EmbedField {
                name: "Field1".to_string(),
                value: "Value1".to_string(),
                inline: false,
            }],
            thumbnail: None,
            footer: None,
        }));
        let result = channel_message_to_slack_text(&msg);
        assert!(result.contains("*My Embed*"));
        assert!(result.contains("A description"));
        assert!(result.contains("*Field1*: Value1"));
    }

    #[test]
    fn mixed_content_to_slack() {
        let msg = sample_channel_message(MessageContent::Mixed(vec![
            MessageContent::Text("First part".to_string()),
            MessageContent::Text("Second part".to_string()),
        ]));
        let result = channel_message_to_slack_text(&msg);
        assert!(result.contains("First part"));
        assert!(result.contains("Second part"));
    }

    #[test]
    fn parse_slack_ts_valid() {
        let ts = "1234567890.123456";
        let dt = parse_slack_ts(ts);
        assert_eq!(dt.timestamp(), 1234567890);
    }

    #[test]
    fn parse_slack_ts_no_micros() {
        let ts = "1234567890";
        let dt = parse_slack_ts(ts);
        assert_eq!(dt.timestamp(), 1234567890);
    }

    #[test]
    fn parse_slack_ts_invalid() {
        let ts = "not-a-timestamp";
        let dt = parse_slack_ts(ts);
        // Should return epoch default
        assert_eq!(dt.timestamp(), 0);
    }

    #[test]
    fn slack_message_to_channel_message_basic() {
        let msg = serde_json::json!({
            "ts": "1234567890.123456",
            "user": "U0123456789",
            "text": "Hello from Slack!",
        });
        let result = slack_message_to_channel_message(&msg, "C0123456789", "T0123456789").unwrap();
        assert_eq!(result.id.0, "1234567890.123456");
        assert_eq!(result.author, "U0123456789");
        assert_eq!(result.conversation.platform, "slack");
        assert_eq!(result.conversation.channel_id, "C0123456789");
        assert_eq!(
            result.conversation.server_id.as_deref(),
            Some("T0123456789")
        );
        match &result.content {
            MessageContent::Text(t) => assert_eq!(t, "Hello from Slack!"),
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn slack_message_with_thread() {
        let msg = serde_json::json!({
            "ts": "1234567890.123456",
            "user": "U0123456789",
            "text": "Thread reply",
            "thread_ts": "1234567880.000000",
        });
        let result = slack_message_to_channel_message(&msg, "C01", "T01").unwrap();
        assert_eq!(result.thread_id.as_ref().unwrap().0, "1234567880.000000");
    }

    #[test]
    fn slack_message_with_files() {
        let msg = serde_json::json!({
            "ts": "1234567890.123456",
            "user": "U0123456789",
            "text": "Check this file",
            "files": [
                {
                    "name": "report.pdf",
                    "mimetype": "application/pdf",
                    "url_private": "https://files.slack.com/report.pdf",
                    "size": 1024
                }
            ]
        });
        let result = slack_message_to_channel_message(&msg, "C01", "T01").unwrap();
        assert_eq!(result.attachments.len(), 1);
        assert_eq!(result.attachments[0].filename, "report.pdf");
        assert_eq!(result.attachments[0].content_type, "application/pdf");
        assert_eq!(result.attachments[0].size_bytes, Some(1024));
    }

    #[test]
    fn channel_message_to_slack_payload_roundtrip() {
        let original = sample_channel_message(MessageContent::Text("Roundtrip test".to_string()));
        let slack_text = channel_message_to_slack_text(&original);
        assert_eq!(slack_text, "Roundtrip test");
    }
}
