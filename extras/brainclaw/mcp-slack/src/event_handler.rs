//! Socket Mode event handler that converts Slack events to `ChannelEvent`.
//!
//! Connects to Slack's Socket Mode via WebSocket, receives events, acknowledges them,
//! and converts them to `ChannelEvent` values for the gateway client.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use reqwest::Client;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use brainwires_network::channels::{ChannelEvent, ChannelUser, ConversationId, MessageId};

use crate::slack::{SlackChannel, slack_message_to_channel_message};

/// Slack Socket Mode event handler that forwards events as `ChannelEvent` values
/// over an mpsc channel.
pub struct SlackEventHandler {
    /// Slack app-level token (xapp-...) for Socket Mode connections.
    app_token: String,
    /// Sender for forwarding events to the gateway client loop.
    event_tx: mpsc::Sender<ChannelEvent>,
    /// Reference to the Slack channel for team caching.
    slack_channel: Arc<SlackChannel>,
    /// HTTP client for Socket Mode URL requests and user lookups.
    http: Client,
    /// Cache of user display names, keyed by user ID.
    user_cache: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// Whether to require @mention in non-DM channels.
    group_mention_required: bool,
    /// Bot user ID for @mention detection (e.g. "U0123456789").
    bot_user_id: Option<String>,
    /// Additional keyword patterns for mention detection.
    mention_patterns: Vec<String>,
}

impl SlackEventHandler {
    /// Create a new Socket Mode event handler.
    pub fn new(
        app_token: String,
        event_tx: mpsc::Sender<ChannelEvent>,
        slack_channel: Arc<SlackChannel>,
        group_mention_required: bool,
        bot_user_id: Option<String>,
        mention_patterns: Vec<String>,
    ) -> Self {
        Self {
            app_token,
            event_tx,
            slack_channel,
            http: Client::new(),
            user_cache: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            group_mention_required,
            bot_user_id,
            mention_patterns,
        }
    }

    /// Returns true if this message should be forwarded.
    ///
    /// DMs (channel IDs starting with "D") always forward.
    /// In channels, forward when `group_mention_required` is false or the message
    /// contains `<@BOT_USER_ID>` or matches a keyword pattern.
    fn should_forward(&self, channel_id: &str, text: &str) -> bool {
        // DMs always forward (Slack DM channel IDs start with 'D')
        if channel_id.starts_with('D') {
            return true;
        }
        if !self.group_mention_required {
            return true;
        }
        // Check @mention syntax
        if let Some(ref bot_id) = self.bot_user_id {
            let mention = format!("<@{}>", bot_id);
            if text.contains(mention.as_str()) {
                return true;
            }
        }
        // Check keyword patterns
        let lower = text.to_lowercase();
        for pattern in &self.mention_patterns {
            if lower.contains(pattern.to_lowercase().as_str()) {
                return true;
            }
        }
        false
    }

    /// Open a Socket Mode WebSocket connection by calling `apps.connections.open`.
    pub async fn get_socket_mode_url(&self) -> Result<String> {
        let response = self
            .http
            .post("https://slack.com/api/apps.connections.open")
            .header("Authorization", format!("Bearer {}", self.app_token))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .context("Failed to call apps.connections.open")?;

        let json: Value = response
            .json()
            .await
            .context("Failed to parse apps.connections.open response")?;

        if json.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let error = json
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_error");
            anyhow::bail!("apps.connections.open failed: {}", error);
        }

        let url = json
            .get("url")
            .and_then(|v| v.as_str())
            .context("No URL in apps.connections.open response")?;

        Ok(url.to_string())
    }

    /// Resolve a Slack user ID to a display name via `users.info`.
    async fn resolve_user_name(&self, user_id: &str, bot_token: &str) -> String {
        // Check cache first
        {
            let cache = self.user_cache.read().await;
            if let Some(name) = cache.get(user_id) {
                return name.clone();
            }
        }

        // Fetch from API
        let result = self
            .http
            .get("https://slack.com/api/users.info")
            .header("Authorization", format!("Bearer {}", bot_token))
            .query(&[("user", user_id)])
            .send()
            .await;

        let display_name = match result {
            Ok(resp) => {
                let json: Value = resp.json().await.unwrap_or_default();
                json.get("user")
                    .and_then(|u| u.get("profile"))
                    .and_then(|p| {
                        p.get("display_name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .or_else(|| p.get("real_name").and_then(|v| v.as_str()))
                    })
                    .unwrap_or(user_id)
                    .to_string()
            }
            Err(_) => user_id.to_string(),
        };

        // Cache the result
        {
            let mut cache = self.user_cache.write().await;
            cache.insert(user_id.to_string(), display_name.clone());
        }

        display_name
    }

    /// Main run loop: connect to Socket Mode WebSocket and process events.
    pub async fn run(self, bot_token: String) -> Result<()> {
        loop {
            match self.run_connection(&bot_token).await {
                Ok(()) => {
                    tracing::info!("Socket Mode connection closed, reconnecting...");
                }
                Err(e) => {
                    tracing::error!("Socket Mode error: {:#}, reconnecting in 5s...", e);
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    /// Run a single Socket Mode WebSocket connection.
    async fn run_connection(&self, bot_token: &str) -> Result<()> {
        let wss_url = self.get_socket_mode_url().await?;
        tracing::info!("Connecting to Slack Socket Mode WebSocket");

        let (ws_stream, _) = connect_async(&wss_url)
            .await
            .context("Failed to connect to Socket Mode WebSocket")?;

        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        tracing::info!("Slack Socket Mode connected");

        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(Message::Text(text)) => {
                    let text_str: &str = &text;
                    match serde_json::from_str::<Value>(text_str) {
                        Ok(envelope) => {
                            // Acknowledge the envelope immediately
                            if let Some(envelope_id) =
                                envelope.get("envelope_id").and_then(|v| v.as_str())
                            {
                                let ack =
                                    serde_json::json!({"envelope_id": envelope_id}).to_string();
                                if let Err(e) = ws_sender.send(Message::Text(ack.into())).await {
                                    tracing::error!("Failed to send acknowledgment: {}", e);
                                    break;
                                }
                            }

                            // Process the event
                            if let Err(e) = self.handle_envelope(&envelope, bot_token).await {
                                tracing::warn!("Failed to handle Slack event: {:#}", e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse Socket Mode message: {}", e);
                        }
                    }
                }
                Ok(Message::Ping(_)) => {
                    // tungstenite handles pong automatically
                }
                Ok(Message::Close(_)) => {
                    tracing::info!("Socket Mode WebSocket closed by server");
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::error!("Socket Mode WebSocket error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a Socket Mode envelope.
    async fn handle_envelope(&self, envelope: &Value, bot_token: &str) -> Result<()> {
        let envelope_type = envelope.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match envelope_type {
            "events_api" => {
                let payload = envelope
                    .get("payload")
                    .context("Missing payload in events_api envelope")?;
                self.handle_events_api(payload, bot_token).await?;
            }
            "disconnect" => {
                tracing::info!("Received disconnect event from Slack, will reconnect");
            }
            "hello" => {
                tracing::info!("Received hello from Slack Socket Mode");
            }
            other => {
                tracing::debug!(envelope_type = %other, "Ignoring unknown envelope type");
            }
        }

        Ok(())
    }

    /// Handle an Events API payload.
    async fn handle_events_api(&self, payload: &Value, bot_token: &str) -> Result<()> {
        let event = payload
            .get("event")
            .context("Missing event in Events API payload")?;

        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        let team_id = payload
            .get("team_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match event_type {
            "message" => {
                self.handle_message_event(event, team_id, bot_token).await?;
            }
            "reaction_added" => {
                self.handle_reaction_added(event, bot_token).await?;
            }
            "reaction_removed" => {
                self.handle_reaction_removed(event, bot_token).await?;
            }
            other => {
                tracing::debug!(event_type = %other, "Ignoring unhandled Slack event type");
            }
        }

        Ok(())
    }

    /// Handle a message event (new, edited, or deleted).
    async fn handle_message_event(
        &self,
        event: &Value,
        team_id: &str,
        _bot_token: &str,
    ) -> Result<()> {
        let subtype = event.get("subtype").and_then(|v| v.as_str());

        // Skip bot messages to avoid loops
        if event.get("bot_id").is_some() {
            return Ok(());
        }

        let channel_id = event.get("channel").and_then(|v| v.as_str()).unwrap_or("");

        // Cache team mapping
        self.slack_channel.register_team(channel_id, team_id).await;

        // Group mention filter — check before processing any subtype
        let text = event.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if !self.should_forward(channel_id, text) {
            tracing::debug!(
                channel_id = %channel_id,
                "Skipping Slack channel message — bot not mentioned"
            );
            return Ok(());
        }

        match subtype {
            None => {
                // New message
                let channel_message = slack_message_to_channel_message(event, channel_id, team_id)?;
                let evt = ChannelEvent::MessageReceived(channel_message);
                self.event_tx.send(evt).await.ok();
            }
            Some("message_changed") => {
                if let Some(msg) = event.get("message") {
                    let channel_message =
                        slack_message_to_channel_message(msg, channel_id, team_id)?;
                    let evt = ChannelEvent::MessageEdited(channel_message);
                    self.event_tx.send(evt).await.ok();
                }
            }
            Some("message_deleted") => {
                let deleted_ts = event
                    .get("deleted_ts")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let evt = ChannelEvent::MessageDeleted {
                    message_id: MessageId::new(deleted_ts),
                    conversation: ConversationId {
                        platform: "slack".to_string(),
                        channel_id: channel_id.to_string(),
                        server_id: Some(team_id.to_string()),
                    },
                };
                self.event_tx.send(evt).await.ok();
            }
            Some(other) => {
                tracing::debug!(subtype = %other, "Ignoring message subtype");
            }
        }

        Ok(())
    }

    /// Handle a reaction_added event.
    async fn handle_reaction_added(&self, event: &Value, bot_token: &str) -> Result<()> {
        let user_id = event
            .get("user")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let emoji = event.get("reaction").and_then(|v| v.as_str()).unwrap_or("");
        let item_ts = event
            .get("item")
            .and_then(|v| v.get("ts"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let display_name = self.resolve_user_name(user_id, bot_token).await;

        let user = ChannelUser {
            platform: "slack".to_string(),
            platform_user_id: user_id.to_string(),
            display_name,
            username: None,
            avatar_url: None,
        };

        let evt = ChannelEvent::ReactionAdded {
            message_id: MessageId::new(item_ts),
            user,
            emoji: emoji.to_string(),
        };

        self.event_tx.send(evt).await.ok();
        Ok(())
    }

    /// Handle a reaction_removed event.
    async fn handle_reaction_removed(&self, event: &Value, bot_token: &str) -> Result<()> {
        let user_id = event
            .get("user")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let emoji = event.get("reaction").and_then(|v| v.as_str()).unwrap_or("");
        let item_ts = event
            .get("item")
            .and_then(|v| v.get("ts"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let display_name = self.resolve_user_name(user_id, bot_token).await;

        let user = ChannelUser {
            platform: "slack".to_string(),
            platform_user_id: user_id.to_string(),
            display_name,
            username: None,
            avatar_url: None,
        };

        let evt = ChannelEvent::ReactionRemoved {
            message_id: MessageId::new(item_ts),
            user,
            emoji: emoji.to_string(),
        };

        self.event_tx.send(evt).await.ok();
        Ok(())
    }
}

/// Parse a Socket Mode WSS URL to extract the host for logging.
pub fn parse_socket_mode_url(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_network::channels::MessageContent;

    #[test]
    fn parse_socket_mode_url_valid() {
        let url = "wss://wss-primary.slack.com/link/?ticket=abc123&app_id=A01";
        let host = parse_socket_mode_url(url);
        assert_eq!(host.as_deref(), Some("wss-primary.slack.com"));
    }

    #[test]
    fn parse_socket_mode_url_invalid() {
        let host = parse_socket_mode_url("not-a-url");
        assert!(host.is_none());
    }

    #[test]
    fn slack_event_to_message_received() {
        let event = serde_json::json!({
            "type": "message",
            "ts": "1234567890.123456",
            "user": "U0123456789",
            "text": "Hello from Slack!",
            "channel": "C0123456789",
        });
        let result =
            slack_message_to_channel_message(&event, "C0123456789", "T0123456789").unwrap();
        assert_eq!(result.id.0, "1234567890.123456");
        assert_eq!(result.author, "U0123456789");
        match &result.content {
            MessageContent::Text(t) => assert_eq!(t, "Hello from Slack!"),
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn slack_message_changed_event() {
        // The message_changed subtype wraps the new message in a "message" field
        let event = serde_json::json!({
            "type": "message",
            "subtype": "message_changed",
            "channel": "C01",
            "message": {
                "ts": "1234567890.123456",
                "user": "U01",
                "text": "Edited text",
            }
        });
        // The inner message can be parsed
        let inner = event.get("message").unwrap();
        let result = slack_message_to_channel_message(inner, "C01", "T01").unwrap();
        assert_eq!(result.author, "U01");
        match &result.content {
            MessageContent::Text(t) => assert_eq!(t, "Edited text"),
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn slack_message_deleted_event() {
        let event = serde_json::json!({
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C01",
            "deleted_ts": "1234567890.123456",
        });
        let deleted_ts = event.get("deleted_ts").and_then(|v| v.as_str()).unwrap();
        assert_eq!(deleted_ts, "1234567890.123456");
    }

    #[test]
    fn slack_reaction_event_structure() {
        let event = serde_json::json!({
            "type": "reaction_added",
            "user": "U0123456789",
            "reaction": "thumbsup",
            "item": {
                "type": "message",
                "channel": "C0123456789",
                "ts": "1234567890.123456"
            }
        });
        let user_id = event.get("user").and_then(|v| v.as_str()).unwrap();
        let emoji = event.get("reaction").and_then(|v| v.as_str()).unwrap();
        let ts = event
            .get("item")
            .and_then(|v| v.get("ts"))
            .and_then(|v| v.as_str())
            .unwrap();

        assert_eq!(user_id, "U0123456789");
        assert_eq!(emoji, "thumbsup");
        assert_eq!(ts, "1234567890.123456");
    }
}
