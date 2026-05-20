//! Signal channel adapter implementing the `Channel` trait.
//!
//! Communicates with a running `signal-cli-rest-api` daemon.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

use brainwires_network::channels::{
    ChannelCapabilities, ChannelMessage, ConversationId, MessageContent, MessageId,
};

/// Signal channel adapter.
pub struct SignalChannel {
    http: Client,
    /// Base URL of the signal-cli REST API daemon.
    api_url: String,
    /// The bot's own phone number in E.164 format.
    pub phone_number: String,
}

impl SignalChannel {
    /// Create a new `SignalChannel`.
    pub fn new(api_url: String, phone_number: String) -> Self {
        let api_url = api_url.trim_end_matches('/').to_string();
        Self {
            http: Client::new(),
            api_url,
            phone_number,
        }
    }

    /// Send a plain-text message via the signal-cli REST API.
    ///
    /// `recipient` is either a phone number ("+14155552671") or a group ID
    /// prefixed with "group." (e.g. "group.abc123==").
    async fn post_message(&self, recipient: &str, text: &str) -> Result<String> {
        let url = format!("{}/v1/send", self.api_url);
        let body = if recipient.starts_with("group.") {
            // Group message: recipient list contains the group ID
            json!({
                "message": text,
                "number": self.phone_number,
                "recipients": [recipient],
            })
        } else {
            json!({
                "message": text,
                "number": self.phone_number,
                "recipients": [recipient],
            })
        };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send Signal message")?;

        let status = resp.status();
        let json: Value = resp.json().await.unwrap_or(Value::Null);

        if !status.is_success() {
            let msg = json
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("signal-cli API error {status}: {msg}");
        }

        // The REST API returns an array of timestamp objects
        let ts = json
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|o| o.get("timestamp"))
            .and_then(|v| v.as_i64())
            .map(|ts| ts.to_string())
            .unwrap_or_else(|| "sent".to_string());

        Ok(ts)
    }

    /// Add a reaction (emoji) to a message.
    async fn post_reaction(
        &self,
        recipient: &str,
        target_author: &str,
        target_ts: i64,
        emoji: &str,
    ) -> Result<()> {
        let url = format!("{}/v1/reactions/{}", self.api_url, self.phone_number);
        let body = json!({
            "recipient": recipient,
            "reaction": emoji,
            "target_author": target_author,
            "timestamp": target_ts,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to add Signal reaction")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Signal reaction error: {body}");
        }
        Ok(())
    }

    /// Fetch pending received messages via polling (`GET /v1/receive/{number}`).
    pub async fn receive_pending(&self) -> Result<Vec<Value>> {
        let url = format!("{}/v1/receive/{}", self.api_url, self.phone_number);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to poll Signal messages")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Signal receive error: {body}");
        }

        let json: Value = resp
            .json()
            .await
            .context("Failed to parse Signal receive response")?;
        Ok(json.as_array().cloned().unwrap_or_default())
    }
}

#[async_trait]
impl brainwires_network::channels::Channel for SignalChannel {
    fn channel_type(&self) -> &str {
        "signal"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::MENTIONS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let text = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            MessageContent::Mixed(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    MessageContent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => return Ok(MessageId::new("unsupported")),
        };

        let ts = self.post_message(&target.channel_id, &text).await?;
        Ok(MessageId::new(ts))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        // Signal does not support editing messages via the REST API.
        anyhow::bail!("Signal does not support message editing")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        // Signal does not support deleting messages via the REST API.
        anyhow::bail!("Signal does not support message deletion")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        // Signal REST API does not expose a typing indicator endpoint.
        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        // The message ID is "recipient:author:timestamp" (set by our event_handler).
        // Parse it back to call the reaction endpoint.
        let parts: Vec<&str> = id.0.splitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Signal reaction requires message ID in 'recipient:author:timestamp' format; got '{}'",
                id.0
            );
        }
        let recipient = parts[0];
        let author = parts[1];
        let ts: i64 = parts[2]
            .parse()
            .context("Failed to parse Signal message timestamp")?;

        self.post_reaction(recipient, author, ts, emoji).await
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        // Signal REST API does not expose message history.
        Ok(Vec::new())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the text content from a signal-cli envelope's dataMessage, if any.
pub fn envelope_text(envelope: &Value) -> Option<String> {
    let text = envelope
        .get("dataMessage")
        .and_then(|d| d.get("message"))
        .and_then(|v| v.as_str())?;
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Determine the reply target ("recipient") for an incoming envelope.
///
/// - Group messages → `"group.<base64 group ID>"`
/// - Direct messages → sender phone number
pub fn envelope_recipient(envelope: &Value, own_number: &str) -> String {
    // Check for group info
    if let Some(group_id) = envelope
        .get("dataMessage")
        .and_then(|d| d.get("groupInfo"))
        .and_then(|g| g.get("groupId"))
        .and_then(|v| v.as_str())
    {
        return format!("group.{}", group_id);
    }
    // Direct message: reply to sender
    envelope
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or(own_number)
        .to_string()
}

/// Build a `ChannelMessage` from a signal-cli event envelope.
///
/// Returns `None` if the envelope is not a user data message (e.g. delivery receipts).
pub fn parse_envelope(
    envelope: &Value,
    own_number: &str,
) -> Option<brainwires_network::channels::ChannelMessage> {
    let data_msg = envelope.get("dataMessage")?;
    let text = data_msg.get("message").and_then(|v| v.as_str())?;
    if text.is_empty() {
        return None;
    }

    let sender = envelope
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let ts_ms = data_msg
        .get("timestamp")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let recipient = envelope_recipient(envelope, own_number);

    // Compose a unique message ID that embeds enough info for reactions
    let msg_id = format!("{}:{}:{}", recipient, sender, ts_ms);

    let conversation = ConversationId {
        platform: "signal".to_string(),
        channel_id: recipient,
        server_id: None,
    };

    let timestamp =
        chrono::DateTime::from_timestamp(ts_ms / 1000, 0).unwrap_or_else(chrono::Utc::now);

    Some(brainwires_network::channels::ChannelMessage {
        id: MessageId::new(msg_id),
        conversation,
        author: sender,
        content: MessageContent::Text(text.to_string()),
        thread_id: None,
        reply_to: None,
        timestamp,
        attachments: Vec::new(),
        metadata: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dm_envelope(source: &str, text: &str, ts: i64) -> Value {
        json!({
            "source": source,
            "dataMessage": {
                "message": text,
                "timestamp": ts
            }
        })
    }

    fn group_envelope(source: &str, text: &str, ts: i64, group_id: &str) -> Value {
        json!({
            "source": source,
            "dataMessage": {
                "message": text,
                "timestamp": ts,
                "groupInfo": {
                    "groupId": group_id
                }
            }
        })
    }

    // --- envelope_text ---

    #[test]
    fn envelope_text_returns_message_text() {
        let env = dm_envelope("+1234567890", "hello world", 1000);
        assert_eq!(envelope_text(&env), Some("hello world".to_string()));
    }

    #[test]
    fn envelope_text_returns_none_for_empty_string() {
        let env = dm_envelope("+1234567890", "", 1000);
        assert_eq!(envelope_text(&env), None);
    }

    #[test]
    fn envelope_text_returns_none_when_no_data_message() {
        let env = json!({ "source": "+1", "receiptMessage": {} });
        assert_eq!(envelope_text(&env), None);
    }

    // --- envelope_recipient ---

    #[test]
    fn envelope_recipient_for_dm_returns_sender() {
        let env = dm_envelope("+14155551234", "hi", 1000);
        assert_eq!(envelope_recipient(&env, "+19995559999"), "+14155551234");
    }

    #[test]
    fn envelope_recipient_for_group_returns_group_prefix() {
        let env = group_envelope("+14155551234", "hi", 1000, "abc123==");
        let recipient = envelope_recipient(&env, "+19995559999");
        assert_eq!(recipient, "group.abc123==");
    }

    #[test]
    fn envelope_recipient_falls_back_to_own_number_when_no_source() {
        let env = json!({ "dataMessage": { "message": "x", "timestamp": 0 } });
        assert_eq!(envelope_recipient(&env, "+10000000000"), "+10000000000");
    }

    // --- parse_envelope ---

    #[test]
    fn parse_envelope_returns_channel_message_for_dm() {
        let env = dm_envelope("+14155551234", "Test message", 1_700_000_000_000);
        let own = "+19995559999";
        let msg = parse_envelope(&env, own).expect("should parse DM");

        assert_eq!(msg.author, "+14155551234");
        assert_eq!(msg.conversation.platform, "signal");
        assert_eq!(msg.conversation.channel_id, "+14155551234");
        match &msg.content {
            MessageContent::Text(t) => assert_eq!(t, "Test message"),
            _ => panic!("expected Text content"),
        }
        // Message ID embeds recipient:sender:timestamp
        assert!(msg.id.0.contains("+14155551234"));
        assert!(msg.id.0.contains("1700000000000"));
    }

    #[test]
    fn parse_envelope_returns_none_when_no_data_message() {
        let env = json!({ "source": "+1", "receiptMessage": {} });
        assert!(parse_envelope(&env, "+1").is_none());
    }

    #[test]
    fn parse_envelope_returns_none_for_empty_text() {
        let env = dm_envelope("+1234567890", "", 1000);
        assert!(parse_envelope(&env, "+9999999999").is_none());
    }

    #[test]
    fn parse_envelope_group_message_has_group_channel_id() {
        let env = group_envelope("+14155551234", "group msg", 1_000_000_000, "grpXYZ==");
        let msg = parse_envelope(&env, "+19999999999").expect("should parse group message");
        assert_eq!(msg.conversation.channel_id, "group.grpXYZ==");
    }
}
