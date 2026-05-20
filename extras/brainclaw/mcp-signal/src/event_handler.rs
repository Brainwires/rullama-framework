//! Signal event handler — receives messages from signal-cli REST API.
//!
//! Supports two receive modes:
//! 1. **WebSocket** (`ws://host/v1/events`): preferred; real-time push from signal-cli-rest-api.
//! 2. **Polling** (`GET /v1/receive/{number}`): fallback; polls on a configurable interval.
//!
//! The handler filters messages (self-messages, mention requirements, allowlists),
//! converts them to `ChannelEvent::MessageReceived`, and sends them to the gateway.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use brainwires_network::channels::ChannelEvent;

use crate::signal::{SignalChannel, envelope_recipient, parse_envelope};

/// Handles incoming Signal messages, converting them to `ChannelEvent`s.
pub struct SignalEventHandler {
    /// Base WS URL derived from the REST API URL.
    ws_url: String,
    /// The REST API channel (for polling fallback).
    channel: Arc<SignalChannel>,
    /// Sender for channel events to the gateway.
    event_tx: mpsc::Sender<ChannelEvent>,
    /// Bot's own phone number — used to filter self-messages.
    phone_number: String,
    /// Whether to require a mention in group messages.
    group_mention_required: bool,
    /// Bot display name for @mention detection.
    bot_name: Option<String>,
    /// Additional trigger patterns.
    mention_patterns: Vec<String>,
    /// Allowed sender phone numbers (empty = all).
    sender_allowlist: Vec<String>,
    /// Allowed group IDs (empty = all).
    group_allowlist: Vec<String>,
    /// Polling interval (for fallback mode).
    poll_interval: Duration,
}

impl SignalEventHandler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_url: &str,
        channel: Arc<SignalChannel>,
        event_tx: mpsc::Sender<ChannelEvent>,
        phone_number: String,
        group_mention_required: bool,
        bot_name: Option<String>,
        mention_patterns: Vec<String>,
        sender_allowlist: Vec<String>,
        group_allowlist: Vec<String>,
        poll_interval_ms: u64,
    ) -> Self {
        // Build WebSocket URL from the REST API base URL
        let ws_url = api_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            .trim_end_matches('/')
            .to_string()
            + "/v1/events";

        Self {
            ws_url,
            channel,
            event_tx,
            phone_number,
            group_mention_required,
            bot_name,
            mention_patterns,
            sender_allowlist,
            group_allowlist,
            poll_interval: Duration::from_millis(poll_interval_ms),
        }
    }

    /// Run the event handler.
    ///
    /// Attempts WebSocket first; falls back to polling if the connection fails.
    pub async fn run(&self) -> Result<()> {
        tracing::info!(ws_url = %self.ws_url, "Connecting to signal-cli WebSocket");

        match connect_async(&self.ws_url).await {
            Ok((ws_stream, _)) => {
                tracing::info!("signal-cli WebSocket connected — using push mode");
                self.run_websocket(ws_stream).await
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "signal-cli WebSocket unavailable; falling back to polling"
                );
                self.run_polling().await
            }
        }
    }

    /// Process events from an open WebSocket stream.
    async fn run_websocket(
        &self,
        ws_stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> Result<()> {
        let (mut _sender, mut receiver) = ws_stream.split();

        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_raw(&text).await {
                        tracing::warn!(error = %e, "Error handling Signal event");
                    }
                }
                Ok(Message::Ping(data)) => {
                    let _ = _sender.send(Message::Pong(data)).await;
                }
                Ok(Message::Close(_)) => {
                    tracing::info!("signal-cli WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    tracing::error!(error = %e, "signal-cli WebSocket error");
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Poll the REST API for pending messages.
    async fn run_polling(&self) -> Result<()> {
        tracing::info!(
            interval_ms = self.poll_interval.as_millis(),
            "Signal polling started"
        );

        loop {
            match self.channel.receive_pending().await {
                Ok(messages) => {
                    for envelope_wrapper in messages {
                        // The polling API wraps each message in `{"envelope": {...}}`
                        let envelope = envelope_wrapper
                            .get("envelope")
                            .cloned()
                            .unwrap_or(envelope_wrapper);
                        if let Err(e) = self.handle_envelope(&envelope).await {
                            tracing::warn!(error = %e, "Error handling polled Signal message");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Signal polling error; retrying in 5s");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
            }

            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Handle a raw JSON string from either WebSocket or polling.
    async fn handle_raw(&self, text: &str) -> Result<()> {
        let value: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };

        // WebSocket events are wrapped: `{"envelope": {...}}`
        let envelope = value.get("envelope").cloned().unwrap_or(value);
        self.handle_envelope(&envelope).await
    }

    /// Process a single signal-cli envelope.
    async fn handle_envelope(&self, envelope: &Value) -> Result<()> {
        // Only process dataMessages (not receipts, typing indicators, etc.)
        let data_msg = match envelope.get("dataMessage") {
            Some(d) => d,
            None => return Ok(()),
        };

        // Skip messages without text (attachments only, reactions, etc.)
        let text = match data_msg.get("message").and_then(|v| v.as_str()) {
            Some(t) if !t.is_empty() => t,
            _ => return Ok(()),
        };

        let sender = envelope
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Skip self-messages
        if sender == self.phone_number {
            return Ok(());
        }

        // Sender allowlist
        if !self.sender_allowlist.is_empty() && !self.sender_allowlist.contains(&sender) {
            return Ok(());
        }

        // Determine recipient / channel
        let recipient = envelope_recipient(envelope, &self.phone_number);
        let is_group = recipient.starts_with("group.");
        let group_id = if is_group {
            Some(recipient.trim_start_matches("group.").to_string())
        } else {
            None
        };

        // Group allowlist
        if let Some(ref gid) = group_id
            && !self.group_allowlist.is_empty()
            && !self.group_allowlist.contains(gid)
        {
            return Ok(());
        }

        // Mention filter: group messages only respond when mentioned (if required)
        if self.group_mention_required && is_group && !self.is_mentioned(text) {
            return Ok(());
        }

        // Build ChannelMessage
        let msg = match parse_envelope(envelope, &self.phone_number) {
            Some(m) => m,
            None => return Ok(()),
        };

        let event = ChannelEvent::MessageReceived(msg);
        if self.event_tx.send(event).await.is_err() {
            tracing::warn!("Event channel closed; dropping Signal event");
        }

        Ok(())
    }

    /// Check whether `text` mentions the bot.
    fn is_mentioned(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        if let Some(ref name) = self.bot_name {
            let mention = format!("@{}", name.trim_start_matches('@').to_lowercase());
            if lower.contains(&mention) {
                return true;
            }
            // Also match bare name
            if lower.contains(&name.to_lowercase()) {
                return true;
            }
        }

        for pattern in &self.mention_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        false
    }
}
