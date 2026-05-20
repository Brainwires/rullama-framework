//! WebSocket client for the brainwires-gateway.
//!
//! Intentional copy of the pattern from `mcp-discord` / `mcp-slack`. Not
//! worth extracting into a shared crate yet — see each adapter's README.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use brainwires_network::channels::{
    Channel, ChannelCapabilities, ChannelEvent, ChannelHandshake, ChannelHandshakeResponse,
    ChannelMessage,
};

use crate::teams::TeamsChannel;

/// Gateway WS connection.
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
    /// Connect + handshake.
    pub async fn connect(
        url: &str,
        auth_token: &str,
        capabilities: ChannelCapabilities,
    ) -> Result<Self> {
        let (ws, _) = connect_async(url).await.context("gateway connect")?;
        let (mut sender, receiver) = ws.split();
        let hs = build_handshake(auth_token, capabilities);
        let hs_json = serde_json::to_string(&hs)?;
        sender.send(Message::Text(hs_json.into())).await?;
        let mut client = Self {
            ws_sender: sender,
            ws_receiver: receiver,
        };
        let resp_text = client
            .receive_raw()
            .await?
            .context("gateway closed before handshake response")?;
        let resp: ChannelHandshakeResponse =
            serde_json::from_str(&resp_text).context("parse handshake response")?;
        if !resp.accepted {
            anyhow::bail!(
                "gateway rejected handshake: {}",
                resp.error.unwrap_or_else(|| "unknown".into())
            );
        }
        tracing::info!(channel_id = ?resp.channel_id, "Gateway handshake accepted");
        Ok(client)
    }

    /// Send one event upstream.
    pub async fn send_event(&mut self, event: &ChannelEvent) -> Result<()> {
        let json = serde_json::to_string(event)?;
        self.ws_sender.send(Message::Text(json.into())).await?;
        Ok(())
    }

    async fn receive_raw(&mut self) -> Result<Option<String>> {
        while let Some(r) = self.ws_receiver.next().await {
            match r {
                Ok(Message::Text(t)) => return Ok(Some(t.to_string())),
                Ok(Message::Close(_)) => return Ok(None),
                Ok(_) => continue,
                Err(e) => return Err(anyhow::anyhow!("ws: {e}")),
            }
        }
        Ok(None)
    }

    /// Pump events.
    pub async fn run(
        mut self,
        mut event_rx: mpsc::Receiver<ChannelEvent>,
        chan: Arc<TeamsChannel>,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(ev) = event_rx.recv() => {
                    if let Err(e) = self.send_event(&ev).await {
                        tracing::error!("send_event: {e}");
                        break;
                    }
                }
                result = self.receive_raw() => match result {
                    Ok(Some(t)) => match serde_json::from_str::<ChannelMessage>(&t) {
                        Ok(msg) => {
                            if let Err(e) = chan.send_message(&msg.conversation, &msg).await {
                                tracing::error!("teams send: {e}");
                            }
                        }
                        Err(e) => tracing::warn!("parse gateway msg: {e}"),
                    },
                    Ok(None) => { tracing::info!("gateway closed"); break; }
                    Err(e) => { tracing::error!("gateway recv: {e}"); break; }
                }
            }
        }
        Ok(())
    }
}

/// Build the handshake envelope.
pub fn build_handshake(auth_token: &str, capabilities: ChannelCapabilities) -> ChannelHandshake {
    ChannelHandshake {
        channel_type: "teams".to_string(),
        channel_version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities,
        auth_token: auth_token.to_string(),
    }
}

/// Reconnect backoff.
pub fn backoff_next(current: Duration) -> Duration {
    let next = (current.as_millis() as u64).saturating_mul(2).max(2_000);
    Duration::from_millis(next.min(60_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_channel_type() {
        let hs = build_handshake("t", ChannelCapabilities::RICH_TEXT);
        assert_eq!(hs.channel_type, "teams");
    }

    #[test]
    fn backoff_caps() {
        let mut d = Duration::from_millis(500);
        for _ in 0..20 {
            d = backoff_next(d);
        }
        assert_eq!(d, Duration::from_millis(60_000));
    }
}
