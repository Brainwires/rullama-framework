//! Matrix channel implementation using the `matrix-sdk` crate.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use matrix_sdk::Client;
use matrix_sdk::room::MessagesOptions;
use matrix_sdk::ruma::{
    EventId, OwnedMxcUri, OwnedRoomId, RoomId, UInt,
    events::{
        AnySyncMessageLikeEvent, AnySyncTimelineEvent,
        reaction::ReactionEventContent,
        relation::{Annotation, Thread},
        room::message::{
            AudioMessageEventContent, FileMessageEventContent, ImageMessageEventContent,
            MessageType, Relation, ReplacementMetadata, RoomMessageEventContent,
            VideoMessageEventContent,
        },
    },
};
use reqwest::Client as HttpClient;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelMessage, ConversationId, MediaType, MessageContent,
    MessageId,
};

/// Matrix channel backed by the `matrix-sdk` `Client`.
///
/// The `channel_id` in `ConversationId` is the Matrix room ID
/// (e.g. `"!roomId:server.org"`).
pub struct MatrixChannel {
    client: Arc<Client>,
    http: HttpClient,
}

impl MatrixChannel {
    pub fn new(client: Arc<Client>) -> Self {
        Self {
            client,
            http: HttpClient::new(),
        }
    }

    /// Look up a `matrix_sdk::Room` by room ID string.
    fn get_room(&self, room_id: &str) -> Result<matrix_sdk::Room> {
        let rid: OwnedRoomId = RoomId::parse(room_id)
            .context("Invalid Matrix room ID")?
            .to_owned();
        self.client
            .get_room(&rid)
            .ok_or_else(|| anyhow::anyhow!("Room not found or not joined: {}", room_id))
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
impl Channel for MatrixChannel {
    fn channel_type(&self) -> &str {
        "matrix"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS
            | ChannelCapabilities::TYPING_INDICATOR
            | ChannelCapabilities::EDIT_MESSAGES
            | ChannelCapabilities::DELETE_MESSAGES
            | ChannelCapabilities::THREADS
    }

    async fn send_message(
        &self,
        target: &ConversationId,
        message: &ChannelMessage,
    ) -> Result<MessageId> {
        let room = self.get_room(&target.channel_id)?;

        // Build the message content, routing media types to proper Matrix events.
        let mut content = match &message.content {
            MessageContent::Media(media) => {
                // Download the remote media and upload it to the Matrix homeserver.
                let bytes = self
                    .http
                    .get(&media.url)
                    .send()
                    .await
                    .context("Failed to download media for Matrix upload")?
                    .bytes()
                    .await
                    .context("Failed to read media bytes")?;

                // Guess content-type from URL extension or fall back to octet-stream
                let mime_type: mime::Mime = media
                    .url
                    .rsplit('.')
                    .next()
                    .and_then(|ext| match ext.to_lowercase().as_str() {
                        "jpg" | "jpeg" => Some(mime::IMAGE_JPEG),
                        "png" => Some(mime::IMAGE_PNG),
                        "gif" => Some(mime::IMAGE_GIF),
                        "webp" => "image/webp".parse::<mime::Mime>().ok(),
                        "mp4" | "mov" => "video/mp4".parse::<mime::Mime>().ok(),
                        "webm" => "video/webm".parse::<mime::Mime>().ok(),
                        "mp3" => "audio/mpeg".parse::<mime::Mime>().ok(),
                        "ogg" => "audio/ogg".parse::<mime::Mime>().ok(),
                        "pdf" => "application/pdf".parse::<mime::Mime>().ok(),
                        _ => None,
                    })
                    .unwrap_or(mime::APPLICATION_OCTET_STREAM);

                let upload_resp = self
                    .client
                    .media()
                    .upload(&mime_type, bytes.to_vec(), None)
                    .await
                    .context("Failed to upload media to Matrix homeserver")?;
                let mxc_uri: OwnedMxcUri = upload_resp.content_uri;

                let filename = media.url.rsplit('/').next().unwrap_or("media").to_string();
                let caption = media.caption.clone().unwrap_or_else(|| filename.clone());

                let msgtype = match media.media_type {
                    MediaType::Image | MediaType::GIF => {
                        MessageType::Image(ImageMessageEventContent::plain(caption, mxc_uri))
                    }
                    MediaType::Video => {
                        MessageType::Video(VideoMessageEventContent::plain(caption, mxc_uri))
                    }
                    MediaType::Audio => {
                        MessageType::Audio(AudioMessageEventContent::plain(caption, mxc_uri))
                    }
                    MediaType::Document | MediaType::Sticker => {
                        MessageType::File(FileMessageEventContent::plain(caption, mxc_uri))
                    }
                };
                RoomMessageEventContent::new(msgtype)
            }
            _ => {
                let text = Self::message_text(message);
                if text.is_empty() {
                    bail!("Matrix: cannot send an empty text message");
                }
                RoomMessageEventContent::text_markdown(text)
            }
        };

        // If a thread_id is set, attach a Thread relation so the message is part of that thread.
        if let Some(ref tid) = message.thread_id {
            let thread_event_id = EventId::parse(tid.0.as_str())
                .context("Invalid Matrix thread event ID")?
                .to_owned();
            content.relates_to = Some(Relation::Thread(Thread::without_fallback(thread_event_id)));
        }

        let response = room
            .send(content)
            .await
            .context("Failed to send Matrix message")?;

        Ok(MessageId::new(response.event_id.to_string()))
    }

    async fn edit_message(&self, id: &MessageId, message: &ChannelMessage) -> Result<()> {
        let text = Self::message_text(message);
        let room = self.get_room(&message.conversation.channel_id)?;

        let event_id = EventId::parse(id.0.as_str()).context("Invalid Matrix event ID")?;
        let new_content = RoomMessageEventContent::text_markdown(text);
        let metadata = ReplacementMetadata::new(event_id.to_owned(), None);
        let replacement = new_content.make_replacement(metadata, None);

        room.send(replacement)
            .await
            .context("Failed to edit Matrix message")?;

        Ok(())
    }

    async fn delete_message(&self, id: &MessageId) -> Result<()> {
        // Composite format: "room_id:event_id"
        let parts: Vec<&str> = id.0.as_str().splitn(2, ':').collect();
        if parts.len() != 2 {
            bail!(
                "Matrix delete_message: id must be 'room_id:event_id', got: {}",
                id.0.as_str()
            );
        }
        let (room_id, event_id_str) = (parts[0], parts[1]);

        let room = self.get_room(room_id)?;
        let event_id = EventId::parse(event_id_str).context("Invalid Matrix event ID")?;

        room.redact(&event_id, None, None)
            .await
            .context("Failed to redact Matrix message")?;

        Ok(())
    }

    async fn send_typing(&self, target: &ConversationId) -> Result<()> {
        let room = self.get_room(&target.channel_id)?;
        room.typing_notice(true)
            .await
            .context("Failed to send Matrix typing notice")?;
        Ok(())
    }

    async fn add_reaction(&self, id: &MessageId, emoji: &str) -> Result<()> {
        // Composite format: "room_id:event_id"
        let parts: Vec<&str> = id.0.as_str().splitn(2, ':').collect();
        if parts.len() != 2 {
            bail!(
                "Matrix add_reaction: id must be 'room_id:event_id', got: {}",
                id.0.as_str()
            );
        }
        let (room_id, event_id_str) = (parts[0], parts[1]);

        let room = self.get_room(room_id)?;
        let event_id = EventId::parse(event_id_str).context("Invalid Matrix event ID")?;
        let annotation = Annotation::new(event_id.to_owned(), emoji.to_string());
        let content = ReactionEventContent::new(annotation);

        room.send(content)
            .await
            .context("Failed to send Matrix reaction")?;

        Ok(())
    }

    async fn get_history(
        &self,
        target: &ConversationId,
        limit: usize,
    ) -> Result<Vec<ChannelMessage>> {
        let room = self.get_room(&target.channel_id)?;

        let limit_u = u32::try_from(limit).unwrap_or(25);
        let mut options = MessagesOptions::backward();
        options.limit = UInt::from(limit_u);

        let response = room
            .messages(options)
            .await
            .context("Failed to fetch Matrix message history")?;

        let mut out = Vec::new();
        for timeline_event in response.chunk {
            let raw = timeline_event.raw();
            if let Ok(AnySyncTimelineEvent::MessageLike(AnySyncMessageLikeEvent::RoomMessage(
                msg_event,
            ))) = raw.deserialize()
                && let Some(original) = msg_event.as_original()
            {
                use matrix_sdk::ruma::events::room::message::MessageType;
                let text = match &original.content.msgtype {
                    MessageType::Text(t) => t.body.clone(),
                    other => format!("[{}]", other.msgtype()),
                };
                let channel_msg = ChannelMessage {
                    id: MessageId::new(original.event_id.to_string()),
                    conversation: target.clone(),
                    author: original.sender.to_string(),
                    content: MessageContent::Text(text),
                    thread_id: None,
                    reply_to: None,
                    timestamp: original
                        .origin_server_ts
                        .to_system_time()
                        .map(chrono::DateTime::from)
                        .unwrap_or_else(chrono::Utc::now),
                    attachments: vec![],
                    metadata: HashMap::new(),
                };
                out.push(channel_msg);
            }
        }

        Ok(out)
    }
}
