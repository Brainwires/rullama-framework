//! Core message types for channel communication.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::identity::ConversationId;

/// A unique message identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct MessageId(pub String);

impl MessageId {
    /// Create a new `MessageId` from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A unique thread identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ThreadId(pub String);

impl ThreadId {
    /// Create a new `ThreadId` from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A message sent or received on a messaging channel.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChannelMessage {
    /// Unique identifier for this message.
    pub id: MessageId,
    /// The conversation this message belongs to.
    pub conversation: ConversationId,
    /// The author of the message (display name or user identifier).
    pub author: String,
    /// The message content.
    pub content: MessageContent,
    /// Optional thread this message belongs to.
    pub thread_id: Option<ThreadId>,
    /// Optional message this is a reply to.
    pub reply_to: Option<MessageId>,
    /// When the message was created.
    pub timestamp: DateTime<Utc>,
    /// File or media attachments.
    pub attachments: Vec<Attachment>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// The content of a channel message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Rich text with markdown and a plain-text fallback.
    RichText {
        /// Markdown-formatted content.
        markdown: String,
        /// Plain-text fallback for channels that don't support markdown.
        fallback_plain: String,
    },
    /// A media payload (image, video, audio, etc.).
    Media(MediaPayload),
    /// An embedded rich card.
    Embed(EmbedPayload),
    /// A combination of multiple content items.
    Mixed(Vec<MessageContent>),
}

/// A file or media attachment on a message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Attachment {
    /// The filename of the attachment.
    pub filename: String,
    /// MIME content type (e.g., "image/png").
    pub content_type: String,
    /// URL where the attachment can be downloaded.
    pub url: String,
    /// Size of the attachment in bytes, if known.
    pub size_bytes: Option<u64>,
}

/// A media payload embedded in a message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MediaPayload {
    /// The type of media.
    pub media_type: MediaType,
    /// URL of the media resource.
    pub url: String,
    /// Optional caption for the media.
    pub caption: Option<String>,
    /// Optional thumbnail URL.
    pub thumbnail_url: Option<String>,
}

/// The type of media in a `MediaPayload`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum MediaType {
    /// A still image.
    Image,
    /// A video clip.
    Video,
    /// An audio recording.
    Audio,
    /// A document file (PDF, Word, etc.).
    Document,
    /// A sticker.
    Sticker,
    /// An animated GIF.
    GIF,
}

/// An embedded rich card in a message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmbedPayload {
    /// Title of the embed.
    pub title: Option<String>,
    /// Description text.
    pub description: Option<String>,
    /// URL the embed links to.
    pub url: Option<String>,
    /// Color of the embed sidebar (as a hex integer, e.g., 0xFF5733).
    pub color: Option<u32>,
    /// Structured fields within the embed.
    pub fields: Vec<EmbedField>,
    /// Thumbnail URL for the embed.
    pub thumbnail: Option<String>,
    /// Footer text for the embed.
    pub footer: Option<String>,
}

/// A single field within an `EmbedPayload`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmbedField {
    /// The field name/title.
    pub name: String,
    /// The field value.
    pub value: String,
    /// Whether this field should be displayed inline.
    pub inline: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_message() -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new("msg-001"),
            conversation: ConversationId {
                platform: "discord".to_string(),
                channel_id: "general".to_string(),
                server_id: Some("srv-1".to_string()),
            },
            author: "alice".to_string(),
            content: MessageContent::Text("Hello, world!".to_string()),
            thread_id: None,
            reply_to: None,
            timestamp: Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn channel_message_serde_roundtrip() {
        let msg = sample_message();
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ChannelMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, msg.id);
        assert_eq!(deserialized.author, msg.author);
    }

    #[test]
    fn rich_text_serde_roundtrip() {
        let content = MessageContent::RichText {
            markdown: "**bold**".to_string(),
            fallback_plain: "bold".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        match deserialized {
            MessageContent::RichText {
                markdown,
                fallback_plain,
            } => {
                assert_eq!(markdown, "**bold**");
                assert_eq!(fallback_plain, "bold");
            }
            _ => panic!("expected RichText variant"),
        }
    }

    #[test]
    fn mixed_content_serde_roundtrip() {
        let content = MessageContent::Mixed(vec![
            MessageContent::Text("check this out".to_string()),
            MessageContent::Media(MediaPayload {
                media_type: MediaType::Image,
                url: "https://example.com/image.png".to_string(),
                caption: Some("A cool image".to_string()),
                thumbnail_url: None,
            }),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        match deserialized {
            MessageContent::Mixed(items) => assert_eq!(items.len(), 2),
            _ => panic!("expected Mixed variant"),
        }
    }

    #[test]
    fn embed_serde_roundtrip() {
        let embed = EmbedPayload {
            title: Some("Title".to_string()),
            description: Some("Description".to_string()),
            url: Some("https://example.com".to_string()),
            color: Some(0xFF5733),
            fields: vec![EmbedField {
                name: "Field".to_string(),
                value: "Value".to_string(),
                inline: true,
            }],
            thumbnail: None,
            footer: Some("Footer".to_string()),
        };
        let json = serde_json::to_string(&embed).unwrap();
        let deserialized: EmbedPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.title, embed.title);
        assert_eq!(deserialized.fields.len(), 1);
    }

    #[test]
    fn attachment_serde_roundtrip() {
        let att = Attachment {
            filename: "report.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            url: "https://example.com/report.pdf".to_string(),
            size_bytes: Some(1024),
        };
        let json = serde_json::to_string(&att).unwrap();
        let deserialized: Attachment = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.filename, "report.pdf");
    }
}
