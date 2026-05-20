//! WebSocket client for connecting to the brainwires-gateway.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelEvent, ChannelHandshake, ChannelHandshakeResponse,
    ChannelMessage,
};

use crate::mattermost::MattermostChannel;

/// Client that maintains a WebSocket connection to the brainwires-gateway.
pub struct GatewayClient {
    ws_sender: futures::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Message,
    >,
    ws_receiver: futures::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
}

impl GatewayClient {
    pub async fn connect(
        url: &str,
        auth_token: &str,
        capabilities: ChannelCapabilities,
    ) -> Result<Self> {
        tracing::info!(url = %url, "Connecting to brainwires-gateway");

        let (ws_stream, _) = connect_async(url)
            .await
            .context("Failed to connect to gateway WebSocket")?;

        let (mut sender, receiver) = ws_stream.split();

        let handshake = build_handshake(auth_token, capabilities);
        let handshake_json =
            serde_json::to_string(&handshake).context("Failed to serialize handshake")?;
        sender
            .send(Message::Text(handshake_json.into()))
            .await
            .context("Failed to send handshake")?;

        let mut client = Self {
            ws_sender: sender,
            ws_receiver: receiver,
        };

        let response_text = client
            .receive_raw()
            .await?
            .context("Gateway closed before handshake response")?;

        let response: ChannelHandshakeResponse =
            serde_json::from_str(&response_text).context("Failed to parse handshake response")?;

        if !response.accepted {
            let err = response.error.unwrap_or_else(|| "unknown reason".into());
            anyhow::bail!("Gateway rejected handshake: {err}");
        }

        tracing::info!(channel_id = ?response.channel_id, "Gateway handshake accepted");
        Ok(client)
    }

    pub async fn send_event(&mut self, event: &ChannelEvent) -> Result<()> {
        let json = serde_json::to_string(event).context("Failed to serialize event")?;
        self.ws_sender
            .send(Message::Text(json.into()))
            .await
            .context("Failed to send event to gateway")?;
        Ok(())
    }

    pub async fn receive_raw(&mut self) -> Result<Option<String>> {
        while let Some(result) = self.ws_receiver.next().await {
            match result {
                Ok(Message::Text(text)) => return Ok(Some(text.to_string())),
                Ok(Message::Close(_)) => return Ok(None),
                Ok(Message::Ping(_)) => continue,
                Ok(_) => continue,
                Err(e) => return Err(anyhow::anyhow!("WebSocket receive error: {e}")),
            }
        }
        Ok(None)
    }

    /// Main run loop: forwards Mattermost events to the gateway and relays
    /// outbound messages from the gateway back to Mattermost.
    pub async fn run(
        mut self,
        mut event_rx: mpsc::Receiver<ChannelEvent>,
        channel: Arc<MattermostChannel>,
    ) -> Result<()> {
        tracing::info!("Gateway client run loop started");

        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    if let Err(e) = self.send_event(&event).await {
                        tracing::error!("Failed to forward event to gateway: {e}");
                        break;
                    }
                }

                result = self.receive_raw() => {
                    match result {
                        Ok(Some(text)) => {
                            if let Ok(msg) = serde_json::from_str::<ChannelMessage>(&text) {
                                if let Err(e) = channel.send_message(&msg.conversation.clone(), &msg).await {
                                    tracing::error!("Failed to send message to Mattermost: {e}");
                                }
                            } else {
                                tracing::warn!("Failed to parse outbound gateway message");
                            }
                        }
                        Ok(None) => {
                            tracing::info!("Gateway connection closed");
                            break;
                        }
                        Err(e) => {
                            tracing::error!("Gateway receive error: {e}");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn build_handshake(auth_token: &str, capabilities: ChannelCapabilities) -> ChannelHandshake {
    ChannelHandshake {
        channel_type: "mattermost".to_string(),
        channel_version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities,
        auth_token: auth_token.to_string(),
    }
}
