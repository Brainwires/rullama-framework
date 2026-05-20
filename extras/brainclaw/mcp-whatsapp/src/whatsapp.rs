//! WhatsApp channel implementation — wraps the Meta Graph API.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MediaType, MessageContent,
    MessageId,
};

const GRAPH_API_BASE: &str = "https://graph.facebook.com/v18.0";

/// WhatsApp channel backed by the Meta Graph API.
///
/// The `channel_id` in `ConversationId` is the recipient's phone number in
/// E.164 format without the `+` (e.g. `"15551234567"`).
pub struct WhatsAppChannel {
    /// Meta Graph API bearer token.
    token: String,
    /// WhatsApp phone number ID (sender's phone number ID from Meta dashboard).
    phone_number_id: String,
    http: Client,
}

impl WhatsAppChannel {
    pub fn new(token: String, phone_number_id: String) -> Self {
        Self {
            token,
            phone_number_id,
            http: Client::new(),
        }
    }

    /// Extract plain text from a `ChannelMessage`.
    fn message_text(msg: &ChannelMessage) -> String {
        match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            MessageContent::Mixed(items) => items
                .iter()
                .filter_map(|c| match c {
                    MessageContent::Text(t) => Some(t.as_str()),
                    MessageContent::RichText { markdown, .. } => Some(markdown.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

#[async_trait]
impl Channel for WhatsAppChannel {
    fn channel_type(&self) -> &str {
        "whatsapp"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::READ_RECEIPTS
            | ChannelCapabilities::MENTIONS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let url = format!("{}/{}/messages", GRAPH_API_BASE, self.phone_number_id);

        // Route to the correct WhatsApp message type based on content
        let body = match &message.content {
            MessageContent::Media(media) => {
                // Map MediaType to WhatsApp API message type
                let (wa_type, media_key) = match media.media_type {
                    MediaType::Image => ("image", "image"),
                    MediaType::Video | MediaType::GIF => ("video", "video"),
                    MediaType::Audio => ("audio", "audio"),
                    MediaType::Document => ("document", "document"),
                    MediaType::Sticker => ("sticker", "sticker"),
                };

                let mut media_obj = json!({ "link": media.url });
                if let Some(caption) = &media.caption {
                    media_obj["caption"] = json!(caption);
                }

                json!({
                    "messaging_product": "whatsapp",
                    "to": target.channel_id,
                    "type": wa_type,
                    media_key: media_obj
                })
            }
            _ => {
                // Text / RichText / Embed / Mixed — convert to text
                let text = Self::message_text(message);
                if text.is_empty() {
                    bail!("WhatsApp: cannot send empty text message");
                }
                json!({
                    "messaging_product": "whatsapp",
                    "to": target.channel_id,
                    "type": "text",
                    "text": {
                        "preview_url": false,
                        "body": text
                    }
                })
            }
        };

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .context("Failed to send WhatsApp message")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("WhatsApp API error {}: {}", status, body);
        }

        let json: serde_json::Value = resp.json().await.context("Failed to parse send response")?;
        let msg_id = json["messages"][0]["id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        Ok(MessageId::new(msg_id))
    }

    async fn edit_message(&self, _id: &MessageId, _message: &ChannelMessage) -> Result<()> {
        bail!("WhatsApp does not support editing messages")
    }

    async fn delete_message(&self, _id: &MessageId) -> Result<()> {
        bail!("WhatsApp does not support deleting messages via the API")
    }

    async fn send_typing(&self, _target: &ConversationId) -> Result<()> {
        // WhatsApp Cloud API does not expose a typing indicator endpoint.
        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        // Reactions API: POST /{phone-number-id}/messages with type=reaction
        // id format: "recipient_phone:message_id"
        let parts: Vec<&str> = id.0.as_str().splitn(2, ':').collect();
        if parts.len() != 2 {
            bail!("WhatsApp add_reaction: id must be 'recipient_phone:message_id'");
        }
        let (recipient, message_id) = (parts[0], parts[1]);

        let url = format!("{}/{}/messages", GRAPH_API_BASE, self.phone_number_id);
        let body = json!({
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "reaction",
            "reaction": {
                "message_id": message_id,
                "emoji": emoji
            }
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .context("Failed to send WhatsApp reaction")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("WhatsApp reaction API error {}: {}", status, body);
        }

        Ok(())
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        // WhatsApp Cloud API does not provide message history retrieval.
        Ok(vec![])
    }
}

/// Parse inbound webhook message objects into `ChannelMessage` instances.
pub fn parse_webhook_messages(
    value: &serde_json::Value,
    phone_number_id: &str,
) -> Vec<ChannelMessage> {
    let mut out = Vec::new();

    let entries = match value.get("entry").and_then(|e| e.as_array()) {
        Some(e) => e,
        None => return out,
    };

    for entry in entries {
        let changes = match entry.get("changes").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => continue,
        };
        for change in changes {
            let v = match change.get("value") {
                Some(v) => v,
                None => continue,
            };
            let messages = match v.get("messages").and_then(|m| m.as_array()) {
                Some(m) => m,
                None => continue,
            };
            for msg in messages {
                if let Some(channel_msg) = parse_single_message(msg, phone_number_id) {
                    out.push(channel_msg);
                }
            }
        }
    }

    out
}

fn parse_single_message(msg: &serde_json::Value, phone_number_id: &str) -> Option<ChannelMessage> {
    let from = msg.get("from")?.as_str()?.to_string();
    let id = msg.get("id")?.as_str()?.to_string();
    let msg_type = msg.get("type")?.as_str()?;

    let content = match msg_type {
        "text" => {
            let body = msg.get("text")?.get("body")?.as_str()?;
            MessageContent::Text(body.to_string())
        }
        "image" | "audio" | "video" | "document" => {
            // Media messages — return type + caption as text
            let caption = msg
                .get(msg_type)
                .and_then(|m| m.get("caption"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let text = if caption.is_empty() {
                format!("[{} attachment]", msg_type)
            } else {
                format!("[{}: {}]", msg_type, caption)
            };
            MessageContent::Text(text)
        }
        _ => return None,
    };

    Some(ChannelMessage {
        id: MessageId::new(id),
        conversation: ConversationId {
            platform: "whatsapp".to_string(),
            channel_id: from.clone(),
            server_id: Some(phone_number_id.to_string()),
        },
        author: from,
        content,
        thread_id: None,
        reply_to: None,
        timestamp: chrono::Utc::now(),
        attachments: vec![],
        metadata: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_text_payload(from: &str, id: &str, body: &str) -> serde_json::Value {
        serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": from,
                            "id": id,
                            "type": "text",
                            "text": { "body": body }
                        }]
                    }
                }]
            }]
        })
    }

    fn make_image_payload(from: &str, id: &str, caption: Option<&str>) -> serde_json::Value {
        let mut image_obj = serde_json::json!({ "mime_type": "image/jpeg" });
        if let Some(cap) = caption {
            image_obj["caption"] = serde_json::json!(cap);
        }
        serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": from,
                            "id": id,
                            "type": "image",
                            "image": image_obj
                        }]
                    }
                }]
            }]
        })
    }

    // --- parse_webhook_messages ---

    #[test]
    fn text_message_parsed_correctly() {
        let payload = make_text_payload("15551234567", "msg-001", "Hello, world!");
        let msgs = parse_webhook_messages(&payload, "phone-id-123");

        assert_eq!(msgs.len(), 1);
        let msg = &msgs[0];
        assert_eq!(msg.author, "15551234567");
        assert_eq!(msg.id.0, "msg-001");
        assert_eq!(msg.conversation.platform, "whatsapp");
        assert_eq!(msg.conversation.channel_id, "15551234567");
        assert_eq!(msg.conversation.server_id.as_deref(), Some("phone-id-123"));
        assert!(matches!(&msg.content, MessageContent::Text(t) if t == "Hello, world!"));
    }

    #[test]
    fn image_message_without_caption() {
        let payload = make_image_payload("15551111111", "img-001", None);
        let msgs = parse_webhook_messages(&payload, "pid");

        assert_eq!(msgs.len(), 1);
        assert!(matches!(&msgs[0].content, MessageContent::Text(t) if t == "[image attachment]"));
    }

    #[test]
    fn image_message_with_caption() {
        let payload = make_image_payload("15551111111", "img-002", Some("My photo"));
        let msgs = parse_webhook_messages(&payload, "pid");

        assert_eq!(msgs.len(), 1);
        assert!(matches!(&msgs[0].content, MessageContent::Text(t) if t == "[image: My photo]"));
    }

    #[test]
    fn video_message_parsed() {
        let payload = serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "15559999999",
                            "id": "vid-001",
                            "type": "video",
                            "video": { "mime_type": "video/mp4" }
                        }]
                    }
                }]
            }]
        });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert_eq!(msgs.len(), 1);
        assert!(matches!(&msgs[0].content, MessageContent::Text(t) if t == "[video attachment]"));
    }

    #[test]
    fn multiple_messages_in_one_payload() {
        let payload = serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [
                            { "from": "111", "id": "a", "type": "text", "text": { "body": "msg1" } },
                            { "from": "222", "id": "b", "type": "text", "text": { "body": "msg2" } }
                        ]
                    }
                }]
            }]
        });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn unknown_message_type_skipped() {
        let payload = serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "111",
                            "id": "x",
                            "type": "location",
                            "location": { "latitude": 1.0, "longitude": 2.0 }
                        }]
                    }
                }]
            }]
        });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert!(msgs.is_empty(), "unknown types should be skipped");
    }

    #[test]
    fn empty_payload_returns_no_messages() {
        let payload = serde_json::json!({});
        let msgs = parse_webhook_messages(&payload, "pid");
        assert!(msgs.is_empty());
    }

    #[test]
    fn missing_entry_returns_no_messages() {
        let payload = serde_json::json!({ "object": "whatsapp_business_account" });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert!(msgs.is_empty());
    }

    #[test]
    fn multiple_entries_and_changes_handled() {
        let payload = serde_json::json!({
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    { "from": "111", "id": "a1", "type": "text", "text": { "body": "hi" } }
                                ]
                            }
                        }
                    ]
                },
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    { "from": "222", "id": "b1", "type": "text", "text": { "body": "hello" } }
                                ]
                            }
                        }
                    ]
                }
            ]
        });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn text_message_missing_body_skipped() {
        // text type but no "text" field
        let payload = serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "111",
                            "id": "x",
                            "type": "text"
                        }]
                    }
                }]
            }]
        });
        let msgs = parse_webhook_messages(&payload, "pid");
        assert!(
            msgs.is_empty(),
            "message with missing text body should be skipped"
        );
    }

    // --- GRAPH_API_BASE ---

    #[test]
    fn graph_api_base_is_meta_url() {
        assert!(GRAPH_API_BASE.starts_with("https://graph.facebook.com"));
    }
}
