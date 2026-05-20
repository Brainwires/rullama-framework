//! Mattermost WebSocket event handler.
//!
//! Connects to the Mattermost WebSocket API (`/api/v4/websocket`), authenticates
//! with the personal access token, and dispatches incoming `posted` events to
//! the gateway via the provided `mpsc::Sender<ChannelEvent>`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use brainwires_network::channels::{
    ChannelEvent, ChannelMessage, ConversationId, MessageContent, MessageId, ThreadId,
};

use crate::mattermost::MattermostChannel;

/// Handles the Mattermost WebSocket event stream.
pub struct MattermostEventHandler {
    /// Mattermost server WebSocket URL (e.g. "wss://mattermost.example.com/api/v4/websocket").
    ws_url: String,
    /// Personal access token.
    token: String,
    /// Sender for channel events to the gateway.
    event_tx: mpsc::Sender<ChannelEvent>,
    /// The Mattermost channel adapter (for sending typing indicators or replies).
    #[allow(dead_code)]
    channel: Arc<MattermostChannel>,
    /// Bot user ID — used to filter out self-messages.
    bot_user_id: String,
    /// Whether to require a mention in group channels.
    group_mention_required: bool,
    /// Bot username for mention detection (e.g. "@mybot").
    bot_username: Option<String>,
    /// Additional keyword patterns that trigger a response.
    mention_patterns: Vec<String>,
    /// Channel allowlist — empty means all channels.
    channel_allowlist: Vec<String>,
}

impl MattermostEventHandler {
    /// Create a new `MattermostEventHandler`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ws_url: String,
        token: String,
        event_tx: mpsc::Sender<ChannelEvent>,
        channel: Arc<MattermostChannel>,
        bot_user_id: String,
        group_mention_required: bool,
        bot_username: Option<String>,
        mention_patterns: Vec<String>,
        channel_allowlist: Vec<String>,
    ) -> Self {
        Self {
            ws_url,
            token,
            event_tx,
            channel,
            bot_user_id,
            group_mention_required,
            bot_username,
            mention_patterns,
            channel_allowlist,
        }
    }

    /// Connect to Mattermost WebSocket and start processing events (blocking).
    pub async fn run(&self) -> Result<()> {
        tracing::info!(url = %self.ws_url, "Connecting to Mattermost WebSocket");

        let (ws_stream, _) = connect_async(&self.ws_url)
            .await
            .context("Failed to connect to Mattermost WebSocket")?;

        let (mut sender, mut receiver) = ws_stream.split();

        // Authenticate
        let auth_msg = json!({
            "seq": 1,
            "action": "authentication_challenge",
            "data": { "token": self.token }
        });
        sender
            .send(Message::Text(auth_msg.to_string().into()))
            .await
            .context("Failed to send authentication challenge")?;

        tracing::info!("Mattermost WebSocket authenticated, listening for events");

        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_event(&text).await {
                        tracing::warn!(error = %e, "Error handling Mattermost event");
                    }
                }
                Ok(Message::Ping(data)) => {
                    let _ = sender.send(Message::Pong(data)).await;
                }
                Ok(Message::Close(_)) => {
                    tracing::info!("Mattermost WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    tracing::error!(error = %e, "Mattermost WebSocket error");
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Handle a single WebSocket text frame from Mattermost.
    async fn handle_event(&self, text: &str) -> Result<()> {
        let value: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return Ok(()), // Not JSON — ignore
        };

        let event_type = value.get("event").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "posted" => self.handle_posted(&value).await?,
            "hello" => tracing::debug!("Mattermost WebSocket hello received"),
            "status_change" | "user_updated" | "typing" | "channel_viewed" => {} // Ignore
            other => tracing::trace!(event_type = other, "Unhandled Mattermost event"),
        }

        Ok(())
    }

    /// Handle a `posted` event: parse the post and forward to the gateway.
    async fn handle_posted(&self, value: &Value) -> Result<()> {
        let data = match value.get("data") {
            Some(d) => d,
            None => return Ok(()),
        };

        // The post is JSON-encoded inside the data field
        let post_str = data.get("post").and_then(|v| v.as_str()).unwrap_or("{}");
        let post: Value = serde_json::from_str(post_str).unwrap_or(Value::Null);

        let user_id = post
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let channel_id = post
            .get("channel_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let post_id = post
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let message = post
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let ts_ms = post.get("create_at").and_then(|v| v.as_i64()).unwrap_or(0);
        let root_id = post
            .get("root_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| ThreadId(s.to_string()));

        // Skip self-messages
        if user_id == self.bot_user_id {
            return Ok(());
        }

        // Skip empty messages
        if message.trim().is_empty() {
            return Ok(());
        }

        // Channel allowlist filter
        if !self.channel_allowlist.is_empty() && !self.channel_allowlist.contains(&channel_id) {
            return Ok(());
        }

        // Group mention filter
        let channel_type = data
            .get("channel_type")
            .and_then(|v| v.as_str())
            .unwrap_or("O");
        let is_dm = channel_type == "D" || channel_type == "G";

        if self.group_mention_required && !is_dm && !self.is_mentioned(&message) {
            return Ok(());
        }

        let team_id = data
            .get("team_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let conversation = ConversationId {
            platform: "mattermost".to_string(),
            channel_id: channel_id.clone(),
            server_id: team_id,
        };

        let timestamp =
            chrono::DateTime::from_timestamp(ts_ms / 1000, 0).unwrap_or_else(chrono::Utc::now);

        let msg = ChannelMessage {
            id: MessageId::new(post_id),
            conversation,
            author: user_id,
            content: MessageContent::Text(message),
            thread_id: root_id,
            reply_to: None,
            timestamp,
            attachments: Vec::new(),
            metadata: HashMap::new(),
        };

        let event = ChannelEvent::MessageReceived(msg);
        if self.event_tx.send(event).await.is_err() {
            tracing::warn!("Event channel closed; dropping Mattermost event");
        }

        Ok(())
    }

    /// Check whether `text` contains a mention of the bot.
    ///
    /// Checks:
    /// 1. `@<bot_username>` if `bot_username` is set
    /// 2. Any pattern in `mention_patterns` (case-insensitive substring)
    fn is_mentioned(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();

        if let Some(ref name) = self.bot_username {
            let mention = format!("@{}", name.trim_start_matches('@').to_lowercase());
            if text_lower.contains(&mention) {
                return true;
            }
        }

        for pattern in &self.mention_patterns {
            if text_lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        false
    }
}
