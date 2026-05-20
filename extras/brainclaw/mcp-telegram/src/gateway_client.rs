//! WebSocket client for connecting to the brainwires-gateway.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing;

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelEvent, ChannelHandshake, ChannelHandshakeResponse,
    ChannelMessage,
};

use crate::telegram::TelegramChannel;

/// Client that maintains a WebSocket connection to the brainwires-gateway.
pub struct GatewayClient {
    /// Sender half of the WebSocket connection.
    ws_sender: futures::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Message,
    >,
    /// Receiver half of the WebSocket connection.
    ws_receiver: futures::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
}

impl GatewayClient {
    /// Connect to the gateway, perform the handshake, and return the client.
    pub async fn connect(
        url: &str,
        auth_token: &str,
        capabilities: ChannelCapabilities,
    ) -> Result<Self> {
        tracing::info!(url = %url, "Connecting to brainwires-gateway");

        let (ws_stream, _response) = connect_async(url)
            .await
            .context("Failed to connect to gateway WebSocket")?;

        let (mut sender, receiver) = ws_stream.split();

        // Send handshake
        let handshake = build_handshake(auth_token, capabilities);
        let handshake_json =
            serde_json::to_string(&handshake).context("Failed to serialize handshake")?;

        sender
            .send(Message::Text(handshake_json.into()))
            .await
            .context("Failed to send handshake")?;

        // Receive handshake response
        let mut client = Self {
            ws_sender: sender,
            ws_receiver: receiver,
        };

        let response_text = client
            .receive_raw()
            .await?
            .context("Gateway closed connection before handshake response")?;

        let response: ChannelHandshakeResponse =
            serde_json::from_str(&response_text).context("Failed to parse handshake response")?;

        if !response.accepted {
            let err_msg = response.error.unwrap_or_else(|| "unknown reason".into());
            anyhow::bail!("Gateway rejected handshake: {}", err_msg);
        }

        tracing::info!(
            channel_id = ?response.channel_id,
            "Gateway handshake accepted"
        );

        Ok(client)
    }

    /// Send a channel event to the gateway.
    pub async fn send_event(&mut self, event: &ChannelEvent) -> Result<()> {
        let json = serde_json::to_string(event).context("Failed to serialize event")?;
        self.ws_sender
            .send(Message::Text(json.into()))
            .await
            .context("Failed to send event to gateway")?;
        Ok(())
    }

    /// Receive a raw text message from the gateway, or `None` if the connection closed.
    pub async fn receive_raw(&mut self) -> Result<Option<String>> {
        while let Some(result) = self.ws_receiver.next().await {
            match result {
                Ok(Message::Text(text)) => return Ok(Some(text.to_string())),
                Ok(Message::Close(_)) => return Ok(None),
                Ok(Message::Ping(_)) => {
                    // tungstenite handles pong automatically
                    continue;
                }
                Ok(_) => continue,
                Err(e) => {
                    return Err(anyhow::anyhow!("WebSocket receive error: {}", e));
                }
            }
        }
        Ok(None)
    }

    /// Main run loop: forwards Telegram events to the gateway and relays
    /// outbound messages from the gateway back through Telegram.
    pub async fn run(
        mut self,
        mut event_rx: mpsc::Receiver<ChannelEvent>,
        telegram_channel: Arc<TelegramChannel>,
    ) -> Result<()> {
        tracing::info!("Gateway client run loop started");

        loop {
            tokio::select! {
                // Forward inbound Telegram events to gateway
                Some(event) = event_rx.recv() => {
                    if let Err(e) = self.send_event(&event).await {
                        tracing::error!("Failed to send event to gateway: {}", e);
                        break;
                    }
                }

                // Receive outbound messages from gateway
                result = self.receive_raw() => {
                    match result {
                        Ok(Some(text)) => {
                            match serde_json::from_str::<ChannelMessage>(&text) {
                                Ok(msg) => {
                                    if let Err(e) = telegram_channel.send_message(&msg.conversation, &msg).await {
                                        tracing::error!("Failed to send message to Telegram: {}", e);
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to parse gateway message: {}", e);
                                }
                            }
                        }
                        Ok(None) => {
                            tracing::info!("Gateway connection closed");
                            break;
                        }
                        Err(e) => {
                            tracing::error!("Gateway receive error: {}", e);
                            break;
                        }
                    }
                }
            }
        }

        tracing::info!("Gateway client run loop ended");
        Ok(())
    }
}

/// Build a `ChannelHandshake` message for the gateway.
pub fn build_handshake(auth_token: &str, capabilities: ChannelCapabilities) -> ChannelHandshake {
    ChannelHandshake {
        channel_type: "telegram".to_string(),
        channel_version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities,
        auth_token: auth_token.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_handshake_message() {
        let caps = ChannelCapabilities::RICH_TEXT | ChannelCapabilities::REACTIONS;
        let hs = build_handshake("my-secret-token", caps);

        assert_eq!(hs.channel_type, "telegram");
        assert_eq!(hs.auth_token, "my-secret-token");
        assert!(hs.capabilities.contains(ChannelCapabilities::RICH_TEXT));
        assert!(hs.capabilities.contains(ChannelCapabilities::REACTIONS));
        assert!(!hs.capabilities.contains(ChannelCapabilities::VOICE));
        assert!(!hs.channel_version.is_empty());
    }

    #[test]
    fn handshake_serialization() {
        let caps = ChannelCapabilities::RICH_TEXT
            | ChannelCapabilities::MEDIA_UPLOAD
            | ChannelCapabilities::REACTIONS;
        let hs = build_handshake("token-123", caps);
        let json = serde_json::to_string(&hs).unwrap();
        let parsed: ChannelHandshake = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.channel_type, "telegram");
        assert_eq!(parsed.auth_token, "token-123");
        assert!(parsed.capabilities.contains(ChannelCapabilities::RICH_TEXT));
    }
}
