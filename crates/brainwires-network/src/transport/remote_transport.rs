use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio::sync::{Mutex, broadcast};

use super::traits::{Transport, TransportAddress};
use crate::{MessageEnvelope, TransportType};

/// Remote transport for cloud-mediated agent communication.
///
/// This transport bridges agents through a remote backend (e.g. Supabase
/// Realtime or HTTP polling), enabling communication across networks
/// without direct peer-to-peer connectivity.
///
/// The transport works by posting message envelopes to a remote relay
/// and receiving envelopes pushed back from the relay.
///
/// # Usage
///
/// ```rust,ignore
/// let transport = RemoteTransport::new(
///     "https://api.example.com",
///     "api-key-here",
/// );
/// ```
pub struct RemoteTransport {
    /// Backend URL.
    backend_url: String,
    /// API key for authentication.
    api_key: String,
    /// Whether connected.
    connected: bool,
    /// Incoming message buffer.
    rx: Mutex<Option<broadcast::Receiver<MessageEnvelope>>>,
    /// Sender for incoming messages (used by the receive loop).
    tx: broadcast::Sender<MessageEnvelope>,
    /// HTTP client for posting messages.
    client: reqwest::Client,
}

impl RemoteTransport {
    /// Create a new remote transport targeting the given backend.
    pub fn new(backend_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let (tx, rx) = broadcast::channel(256);
        Self {
            backend_url: backend_url.into(),
            api_key: api_key.into(),
            connected: false,
            rx: Mutex::new(Some(rx)),
            tx,
            client: reqwest::Client::new(),
        }
    }

    /// Get a sender handle for pushing received messages into the transport.
    ///
    /// This is used by the underlying bridge implementation to feed
    /// messages into the transport's receive queue.
    pub fn message_sender(&self) -> broadcast::Sender<MessageEnvelope> {
        self.tx.clone()
    }
}

#[async_trait]
impl Transport for RemoteTransport {
    async fn connect(&mut self, target: &TransportAddress) -> Result<()> {
        match target {
            TransportAddress::Url(_url) => {
                // The remote transport uses its pre-configured backend_url.
                // The target URL can be used to override or specify a channel.
                self.connected = true;
                Ok(())
            }
            _ => bail!("RemoteTransport only supports URL addresses"),
        }
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    async fn send(&self, envelope: &MessageEnvelope) -> Result<()> {
        if !self.connected {
            bail!("RemoteTransport not connected");
        }

        let url = format!("{}/api/v1/agent/message", self.backend_url);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(envelope)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Remote send failed ({status}): {body}");
        }

        Ok(())
    }

    async fn receive(&self) -> Result<Option<MessageEnvelope>> {
        if !self.connected {
            bail!("RemoteTransport not connected");
        }

        let mut rx_guard = self.rx.lock().await;
        if let Some(rx) = rx_guard.as_mut() {
            match rx.recv().await {
                Ok(envelope) => Ok(Some(envelope)),
                Err(broadcast::error::RecvError::Closed) => Ok(None),
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("RemoteTransport receiver lagged by {n} messages");
                    // Try again after lag
                    match rx.recv().await {
                        Ok(envelope) => Ok(Some(envelope)),
                        _ => Ok(None),
                    }
                }
            }
        } else {
            Ok(None)
        }
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Remote
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use uuid::Uuid;

    #[test]
    fn remote_transport_type() {
        let t = RemoteTransport::new("https://example.com", "key");
        assert_eq!(t.transport_type(), TransportType::Remote);
        assert!(!t.is_connected());
    }

    #[tokio::test]
    async fn remote_transport_message_sender() {
        let transport = RemoteTransport::new("https://example.com", "key");
        let sender = transport.message_sender();

        // Push a message via sender
        let env = MessageEnvelope::broadcast(Uuid::new_v4(), Payload::Text("test".into()));
        sender.send(env.clone()).unwrap();

        // Receive via transport (need to connect first for real usage,
        // but the channel works regardless)
        let mut rx_guard = transport.rx.lock().await;
        if let Some(rx) = rx_guard.as_mut() {
            let received = rx.recv().await.unwrap();
            assert_eq!(received.id, env.id);
        }
    }
}
