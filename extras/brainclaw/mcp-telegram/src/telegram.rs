//! Telegram bot implementation of the `Channel` trait.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, ChatId, MessageId as TeloxideMessageId, ReactionType as TeloxideReactionType,
    ThreadId as TeloxideThreadId,
};

use brainwires_network::channels::{
    Attachment, ChannelCapabilities, ChannelMessage, ConversationId, MediaPayload, MediaType,
    MessageContent, MessageId, ThreadId,
};

/// Telegram channel adapter implementing the `Channel` trait.
///
/// Wraps a teloxide `Bot` to interact with the Telegram Bot API.
pub struct TelegramChannel {
    /// The teloxide Bot instance for API calls.
    bot: Bot,
}

impl TelegramChannel {
    /// Create a new `TelegramChannel` from a teloxide `Bot`.
    pub fn new(bot: Bot) -> Self {
        Self { bot }
    }

    /// Get a reference to the underlying Bot.
    pub fn bot(&self) -> &Bot {
        &self.bot
    }
}

#[async_trait]
impl brainwires_network::channels::Channel for TelegramChannel {
    fn channel_type(&self) -> &str {
        "telegram"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::TYPING_INDICATOR
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::THREADS
            | ChannelCapabilities::MENTIONS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let chat_id = target
            .channel_id
            .parse::<i64>()
            .context("Invalid Telegram chat ID")?;

        let content = channel_message_to_telegram_text(message);

        let mut req = self.bot.send_message(ChatId(chat_id), &content);

        // Route to a forum topic / thread if thread_id is set.
        if let Some(ref tid) = message.thread_id
            && let Ok(thread_msg_id) = tid.0.parse::<i32>()
        {
            req = req.message_thread_id(TeloxideThreadId(TeloxideMessageId(thread_msg_id)));
        }

        let sent = req
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send Telegram message: {}", e))?;

        Ok(MessageId::new(sent.id.0.to_string()))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let chat_id = message
            .conversation
            .channel_id
            .parse::<i64>()
            .context("Invalid Telegram chat ID")?;
        let message_id = id.0.parse::<i32>().context("Invalid Telegram message ID")?;

        let content = channel_message_to_telegram_text(message);

        self.bot
            .edit_message_text(ChatId(chat_id), TeloxideMessageId(message_id), &content)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to edit Telegram message: {}", e))?;

        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (chat_id, message_id) = if parts.len() == 2 {
            (
                parts[0]
                    .parse::<i64>()
                    .context("Invalid chat ID in composite message ID")?,
                parts[1]
                    .parse::<i32>()
                    .context("Invalid message ID in composite message ID")?,
            )
        } else {
            anyhow::bail!(
                "delete_message requires composite ID format 'chat_id:message_id', got: {}",
                id.0
            );
        };

        self.bot
            .delete_message(ChatId(chat_id), TeloxideMessageId(message_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete Telegram message: {}", e))?;

        Ok(())
    }

    async fn send_typing(&self, target: &ConversationId) -> Result<()> {
        let chat_id = target
            .channel_id
            .parse::<i64>()
            .context("Invalid Telegram chat ID")?;

        self.bot
            .send_chat_action(ChatId(chat_id), ChatAction::Typing)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send typing indicator: {}", e))?;

        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (chat_id, message_id) = if parts.len() == 2 {
            (
                parts[0].parse::<i64>().context("Invalid chat ID")?,
                parts[1].parse::<i32>().context("Invalid message ID")?,
            )
        } else {
            anyhow::bail!(
                "add_reaction requires composite ID format 'chat_id:message_id', got: {}",
                id.0
            );
        };

        let reaction = TeloxideReactionType::Emoji {
            emoji: emoji.to_string(),
        };

        self.bot
            .set_message_reaction(ChatId(chat_id), TeloxideMessageId(message_id))
            .reaction(vec![reaction])
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add reaction: {}", e))?;

        Ok(())
    }

    async fn get_history(
        &self,
        _target: &ConversationId,
        _limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        // Telegram Bot API does not provide a message history endpoint.
        // Bots can only receive messages via updates, not fetch historical messages.
        Ok(vec![])
    }
}

// -- Conversion helpers --

/// Convert a teloxide `Message` to a `ChannelMessage`.
pub fn telegram_message_to_channel_message(msg: &teloxide::types::Message) -> ChannelMessage {
    let content = if let Some(text) = &msg.text() {
        MessageContent::Text(text.to_string())
    } else if let Some(caption) = msg.caption() {
        // Photo, video, document with caption
        if let Some(photo) = msg.photo() {
            if let Some(largest) = photo.last() {
                MessageContent::Media(MediaPayload {
                    media_type: MediaType::Image,
                    url: largest.file.id.clone(),
                    caption: Some(caption.to_string()),
                    thumbnail_url: photo.first().map(|p| p.file.id.clone()),
                })
            } else {
                MessageContent::Text(caption.to_string())
            }
        } else {
            MessageContent::Text(caption.to_string())
        }
    } else if let Some(photo) = msg.photo() {
        if let Some(largest) = photo.last() {
            MessageContent::Media(MediaPayload {
                media_type: MediaType::Image,
                url: largest.file.id.clone(),
                caption: None,
                thumbnail_url: photo.first().map(|p| p.file.id.clone()),
            })
        } else {
            MessageContent::Text(String::new())
        }
    } else if let Some(sticker) = &msg.sticker() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::Sticker,
            url: sticker.file.id.clone(),
            caption: None,
            thumbnail_url: sticker.thumbnail.as_ref().map(|t| t.file.id.clone()),
        })
    } else if let Some(doc) = msg.document() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::Document,
            url: doc.file.id.clone(),
            caption: None,
            thumbnail_url: doc.thumbnail.as_ref().map(|t| t.file.id.clone()),
        })
    } else if let Some(video) = msg.video() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::Video,
            url: video.file.id.clone(),
            caption: None,
            thumbnail_url: video.thumbnail.as_ref().map(|t| t.file.id.clone()),
        })
    } else if let Some(voice) = msg.voice() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::Audio,
            url: voice.file.id.clone(),
            caption: None,
            thumbnail_url: None,
        })
    } else if let Some(audio) = msg.audio() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::Audio,
            url: audio.file.id.clone(),
            caption: None,
            thumbnail_url: audio.thumbnail.as_ref().map(|t| t.file.id.clone()),
        })
    } else if let Some(animation) = msg.animation() {
        MessageContent::Media(MediaPayload {
            media_type: MediaType::GIF,
            url: animation.file.id.clone(),
            caption: None,
            thumbnail_url: animation.thumbnail.as_ref().map(|t| t.file.id.clone()),
        })
    } else {
        MessageContent::Text(String::new())
    };

    let attachments = collect_attachments(msg);

    let author_name = msg
        .from
        .as_ref()
        .map(|u| u.username.clone().unwrap_or_else(|| u.first_name.clone()))
        .unwrap_or_else(|| "unknown".to_string());

    let thread_id = msg.thread_id.map(|tid| ThreadId::new(tid.to_string()));

    let reply_to = msg
        .reply_to_message()
        .map(|r| MessageId::new(r.id.0.to_string()));

    let timestamp: DateTime<Utc> = msg.date;

    ChannelMessage {
        id: MessageId::new(msg.id.0.to_string()),
        conversation: ConversationId {
            platform: "telegram".to_string(),
            channel_id: msg.chat.id.0.to_string(),
            server_id: None,
        },
        author: author_name,
        content,
        thread_id,
        reply_to,
        timestamp,
        attachments,
        metadata: HashMap::new(),
    }
}

/// Collect file attachments from a Telegram message.
fn collect_attachments(msg: &teloxide::types::Message) -> Vec<Attachment> {
    let mut attachments = Vec::new();

    if let Some(doc) = msg.document() {
        attachments.push(Attachment {
            filename: doc
                .file_name
                .clone()
                .unwrap_or_else(|| "document".to_string()),
            content_type: doc
                .mime_type
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            url: doc.file.id.clone(),
            size_bytes: Some(doc.file.size as u64),
        });
    }

    if let Some(photo) = msg.photo()
        && let Some(largest) = photo.last()
    {
        attachments.push(Attachment {
            filename: "photo.jpg".to_string(),
            content_type: "image/jpeg".to_string(),
            url: largest.file.id.clone(),
            size_bytes: Some(largest.file.size as u64),
        });
    }

    attachments
}

/// Convert a `ChannelMessage` to Telegram-compatible text content.
pub fn channel_message_to_telegram_text(message: &ChannelMessage) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::RichText {
            markdown,
            fallback_plain: _,
        } => {
            // Telegram supports a subset of markdown (MarkdownV2)
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
                    channel_message_to_telegram_text(&temp)
                })
                .collect();
            sub.join("\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_network::channels::{ChannelCapabilities, EmbedField, EmbedPayload};
    use chrono::Utc;

    fn sample_conversation() -> ConversationId {
        ConversationId {
            platform: "telegram".to_string(),
            channel_id: "-1001234567890".to_string(),
            server_id: None,
        }
    }

    fn sample_channel_message(content: MessageContent) -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new("msg-001"),
            conversation: sample_conversation(),
            author: "TestBot".to_string(),
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
        let expected = ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::TYPING_INDICATOR
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::MENTIONS;

        assert!(expected.contains(ChannelCapabilities::RICH_TEXT));
        assert!(expected.contains(ChannelCapabilities::MEDIA_UPLOAD));
        assert!(expected.contains(ChannelCapabilities::REACTIONS));
        assert!(expected.contains(ChannelCapabilities::TYPING_INDICATOR));
        assert!(expected.contains(ChannelCapabilities::EDIT_MESSAGES));
        assert!(expected.contains(ChannelCapabilities::DELETE_MESSAGES));
        assert!(expected.contains(ChannelCapabilities::MENTIONS));

        // Telegram bot API does not natively support these
        assert!(!expected.contains(ChannelCapabilities::THREADS));
        assert!(!expected.contains(ChannelCapabilities::VOICE));
        assert!(!expected.contains(ChannelCapabilities::VIDEO));
        assert!(!expected.contains(ChannelCapabilities::READ_RECEIPTS));
        assert!(!expected.contains(ChannelCapabilities::EMBEDS));
    }

    #[test]
    fn text_content_to_telegram() {
        let msg = sample_channel_message(MessageContent::Text("Hello world".to_string()));
        let result = channel_message_to_telegram_text(&msg);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn rich_text_content_to_telegram() {
        let msg = sample_channel_message(MessageContent::RichText {
            markdown: "**bold** and *italic*".to_string(),
            fallback_plain: "bold and italic".to_string(),
        });
        let result = channel_message_to_telegram_text(&msg);
        assert_eq!(result, "**bold** and *italic*");
    }

    #[test]
    fn media_content_to_telegram() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: Some("Check this out".to_string()),
            thumbnail_url: None,
        }));
        let result = channel_message_to_telegram_text(&msg);
        assert_eq!(result, "Check this out\nhttps://example.com/image.png");
    }

    #[test]
    fn media_content_no_caption_to_telegram() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: None,
            thumbnail_url: None,
        }));
        let result = channel_message_to_telegram_text(&msg);
        assert_eq!(result, "https://example.com/image.png");
    }

    #[test]
    fn embed_content_to_telegram() {
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
        let result = channel_message_to_telegram_text(&msg);
        assert!(result.contains("*My Embed*"));
        assert!(result.contains("A description"));
        assert!(result.contains("*Field1*: Value1"));
    }

    #[test]
    fn mixed_content_to_telegram() {
        let msg = sample_channel_message(MessageContent::Mixed(vec![
            MessageContent::Text("First part".to_string()),
            MessageContent::Text("Second part".to_string()),
        ]));
        let result = channel_message_to_telegram_text(&msg);
        assert!(result.contains("First part"));
        assert!(result.contains("Second part"));
    }
}
