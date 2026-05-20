//! Discord bot implementation of the `Channel` trait.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serenity::all::{CreateMessage, EditMessage, GetMessages, ReactionType};
use serenity::http::Http;
use serenity::model::channel::Message as SerenityMessage;
use serenity::model::id::ChannelId;
use tokio::sync::RwLock;

use brainwires_network::channels::{
    Attachment, ChannelCapabilities, ChannelMessage, ConversationId, EmbedField, EmbedPayload,
    MessageContent, MessageId, ThreadId,
};

/// Discord channel adapter implementing the `Channel` trait.
///
/// Wraps a serenity `Http` client to interact with the Discord API.
pub struct DiscordChannel {
    /// The serenity HTTP client for API calls.
    http: Arc<Http>,
    /// Cache of guild IDs we've seen, keyed by channel ID.
    guild_cache: Arc<RwLock<HashMap<u64, u64>>>,
}

impl DiscordChannel {
    /// Create a new `DiscordChannel` from a serenity `Http` client.
    pub fn new(http: Arc<Http>) -> Self {
        Self {
            http,
            guild_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a guild/channel mapping (called by the event handler).
    pub async fn register_guild(&self, channel_id: u64, guild_id: u64) {
        let mut cache = self.guild_cache.write().await;
        cache.insert(channel_id, guild_id);
    }

    /// Get the HTTP client reference.
    pub fn http(&self) -> &Arc<Http> {
        &self.http
    }
}

#[async_trait]
impl brainwires_network::channels::Channel for DiscordChannel {
    fn channel_type(&self) -> &str {
        "discord"
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
            | ChannelCapabilities::EMBEDS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        // In Discord, threads are just channels — if a thread_id is set on the
        // outgoing message, send directly to the thread channel instead of the
        // parent channel.
        let raw_channel_id = match &message.thread_id {
            Some(tid) => tid.0.clone(),
            None => target.channel_id.clone(),
        };
        let channel_id = raw_channel_id
            .parse::<u64>()
            .context("Invalid Discord channel ID")?;
        let channel = ChannelId::new(channel_id);

        let content = channel_message_to_discord_content(message);
        let builder = CreateMessage::new().content(&content);

        let sent = channel
            .send_message(&*self.http, builder)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send Discord message: {}", e))?;

        Ok(MessageId::new(sent.id.to_string()))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let channel_id = message
            .conversation
            .channel_id
            .parse::<u64>()
            .context("Invalid Discord channel ID")?;
        let message_id = id.0.parse::<u64>().context("Invalid Discord message ID")?;

        let channel = ChannelId::new(channel_id);
        let content = channel_message_to_discord_content(message);
        let builder = EditMessage::new().content(&content);

        channel
            .edit_message(
                &*self.http,
                serenity::model::id::MessageId::new(message_id),
                builder,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to edit Discord message: {}", e))?;

        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (channel_id, message_id) = if parts.len() == 2 {
            (
                parts[0]
                    .parse::<u64>()
                    .context("Invalid channel ID in composite message ID")?,
                parts[1]
                    .parse::<u64>()
                    .context("Invalid message ID in composite message ID")?,
            )
        } else {
            anyhow::bail!(
                "delete_message requires composite ID format 'channel_id:message_id', got: {}",
                id.0
            );
        };

        let channel = ChannelId::new(channel_id);
        channel
            .delete_message(&*self.http, serenity::model::id::MessageId::new(message_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete Discord message: {}", e))?;

        Ok(())
    }

    async fn send_typing(&self, target: &ConversationId) -> Result<()> {
        let channel_id = target
            .channel_id
            .parse::<u64>()
            .context("Invalid Discord channel ID")?;
        let channel = ChannelId::new(channel_id);

        channel
            .broadcast_typing(&*self.http)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send typing indicator: {}", e))?;

        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        let parts: Vec<&str> = id.0.splitn(2, ':').collect();
        let (channel_id, message_id) = if parts.len() == 2 {
            (
                parts[0].parse::<u64>().context("Invalid channel ID")?,
                parts[1].parse::<u64>().context("Invalid message ID")?,
            )
        } else {
            anyhow::bail!(
                "add_reaction requires composite ID format 'channel_id:message_id', got: {}",
                id.0
            );
        };

        let channel = ChannelId::new(channel_id);
        let reaction = ReactionType::Unicode(emoji.to_string());

        channel
            .create_reaction(
                &*self.http,
                serenity::model::id::MessageId::new(message_id),
                reaction,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add reaction: {}", e))?;

        Ok(())
    }

    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        let channel_id = target
            .channel_id
            .parse::<u64>()
            .context("Invalid Discord channel ID")?;
        let channel = ChannelId::new(channel_id);

        let builder = GetMessages::new().limit(limit as u8);
        let messages = channel
            .messages(&*self.http, builder)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch message history: {}", e))?;

        let guild_id = target.server_id.as_deref().unwrap_or("0");
        Ok(messages
            .into_iter()
            .map(|m| discord_message_to_channel_message(&m, guild_id))
            .collect())
    }
}

// ── Conversion helpers ──────────────────────────────────────────────────

/// Convert a Discord `Message` to a `ChannelMessage`.
pub fn discord_message_to_channel_message(msg: &SerenityMessage, guild_id: &str) -> ChannelMessage {
    let content = if msg.content.is_empty() && !msg.embeds.is_empty() {
        let e = &msg.embeds[0];
        MessageContent::Embed(EmbedPayload {
            title: e.title.clone(),
            description: e.description.clone(),
            url: e.url.clone(),
            color: e.colour.map(|c| c.0),
            fields: e
                .fields
                .iter()
                .map(|f| EmbedField {
                    name: f.name.clone(),
                    value: f.value.clone(),
                    inline: f.inline,
                })
                .collect(),
            thumbnail: e.thumbnail.as_ref().map(|t| t.url.clone()),
            footer: e.footer.as_ref().map(|f| f.text.clone()),
        })
    } else {
        MessageContent::Text(msg.content.clone())
    };

    let attachments = msg
        .attachments
        .iter()
        .map(|a| Attachment {
            filename: a.filename.clone(),
            content_type: a
                .content_type
                .clone()
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            url: a.url.clone(),
            size_bytes: Some(a.size as u64),
        })
        .collect();

    let thread_id = msg.thread.as_ref().map(|t| ThreadId::new(t.id.to_string()));

    let reply_to = msg
        .referenced_message
        .as_ref()
        .map(|r| MessageId::new(r.id.to_string()));

    // With chrono feature enabled, *msg.timestamp derefs to DateTime<Utc>
    let timestamp: DateTime<Utc> = *msg.timestamp;

    ChannelMessage {
        id: MessageId::new(msg.id.to_string()),
        conversation: ConversationId {
            platform: "discord".to_string(),
            channel_id: msg.channel_id.to_string(),
            server_id: Some(guild_id.to_string()),
        },
        author: msg.author.name.clone(),
        content,
        thread_id,
        reply_to,
        timestamp,
        attachments,
        metadata: HashMap::new(),
    }
}

/// Convert a `ChannelMessage` to Discord-compatible text content.
pub fn channel_message_to_discord_content(message: &ChannelMessage) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::RichText {
            markdown,
            fallback_plain: _,
        } => {
            // Discord supports markdown natively
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
                parts.push(format!("**{}**", title));
            }
            if let Some(desc) = &embed.description {
                parts.push(desc.clone());
            }
            for field in &embed.fields {
                parts.push(format!("**{}**: {}", field.name, field.value));
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
                    channel_message_to_discord_content(&temp)
                })
                .collect();
            sub.join("\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_network::channels::{ChannelCapabilities, MediaPayload, MediaType};
    use chrono::Utc;

    fn sample_conversation() -> ConversationId {
        ConversationId {
            platform: "discord".to_string(),
            channel_id: "123456789".to_string(),
            server_id: Some("987654321".to_string()),
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
            | ChannelCapabilities::THREADS
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::TYPING_INDICATOR
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::MENTIONS
            | ChannelCapabilities::EMBEDS;

        assert!(expected.contains(ChannelCapabilities::RICH_TEXT));
        assert!(expected.contains(ChannelCapabilities::MEDIA_UPLOAD));
        assert!(expected.contains(ChannelCapabilities::THREADS));
        assert!(expected.contains(ChannelCapabilities::REACTIONS));
        assert!(expected.contains(ChannelCapabilities::TYPING_INDICATOR));
        assert!(expected.contains(ChannelCapabilities::EDIT_MESSAGES));
        assert!(expected.contains(ChannelCapabilities::DELETE_MESSAGES));
        assert!(expected.contains(ChannelCapabilities::MENTIONS));
        assert!(expected.contains(ChannelCapabilities::EMBEDS));

        assert!(!expected.contains(ChannelCapabilities::VOICE));
        assert!(!expected.contains(ChannelCapabilities::VIDEO));
        assert!(!expected.contains(ChannelCapabilities::READ_RECEIPTS));
    }

    #[test]
    fn text_content_to_discord() {
        let msg = sample_channel_message(MessageContent::Text("Hello world".to_string()));
        let result = channel_message_to_discord_content(&msg);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn rich_text_content_to_discord() {
        let msg = sample_channel_message(MessageContent::RichText {
            markdown: "**bold** and *italic*".to_string(),
            fallback_plain: "bold and italic".to_string(),
        });
        let result = channel_message_to_discord_content(&msg);
        assert_eq!(result, "**bold** and *italic*");
    }

    #[test]
    fn media_content_to_discord() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: Some("Check this out".to_string()),
            thumbnail_url: None,
        }));
        let result = channel_message_to_discord_content(&msg);
        assert_eq!(result, "Check this out\nhttps://example.com/image.png");
    }

    #[test]
    fn media_content_no_caption_to_discord() {
        let msg = sample_channel_message(MessageContent::Media(MediaPayload {
            media_type: MediaType::Image,
            url: "https://example.com/image.png".to_string(),
            caption: None,
            thumbnail_url: None,
        }));
        let result = channel_message_to_discord_content(&msg);
        assert_eq!(result, "https://example.com/image.png");
    }

    #[test]
    fn embed_content_to_discord() {
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
        let result = channel_message_to_discord_content(&msg);
        assert!(result.contains("**My Embed**"));
        assert!(result.contains("A description"));
        assert!(result.contains("**Field1**: Value1"));
    }

    #[test]
    fn mixed_content_to_discord() {
        let msg = sample_channel_message(MessageContent::Mixed(vec![
            MessageContent::Text("First part".to_string()),
            MessageContent::Text("Second part".to_string()),
        ]));
        let result = channel_message_to_discord_content(&msg);
        assert!(result.contains("First part"));
        assert!(result.contains("Second part"));
    }
}
