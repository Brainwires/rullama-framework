use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio::sync::{Mutex, broadcast};

use super::traits::{Transport, TransportAddress};
use crate::{MessageEnvelope, MessageTarget, Payload, TransportType};

/// A2A protocol transport for inter-agent communication.
///
/// Bridges the networking stack's [`MessageEnvelope`] format with the
/// A2A protocol's `Message`/`Task` model. Messages are sent via the
/// A2A client and received via a broadcast channel that an external
/// A2A server handler pushes incoming messages into.
///
/// # Usage
///
/// ```rust,ignore
/// use brainwires_a2a::A2aClient;
///
/// let a2a_client = A2aClient::new_jsonrpc("https://remote-agent.example.com");
/// let transport = A2aTransport::new(a2a_client);
/// ```
pub struct A2aTransport {
    /// The A2A client used to send messages.
    client: Arc<brainwires_a2a::A2aClient>,
    /// Whether connected.
    connected: bool,
    /// Remote endpoint URL.
    endpoint: Option<String>,
    /// Incoming message channel (populated by an external A2A server handler).
    incoming_tx: broadcast::Sender<MessageEnvelope>,
    /// Receiver for incoming messages.
    incoming_rx: Mutex<Option<broadcast::Receiver<MessageEnvelope>>>,
}

impl A2aTransport {
    /// Create a new A2A transport wrapping an existing A2A client.
    pub fn new(client: brainwires_a2a::A2aClient) -> Self {
        let (incoming_tx, incoming_rx) = broadcast::channel(256);
        Self {
            client: Arc::new(client),
            connected: false,
            endpoint: None,
            incoming_tx,
            incoming_rx: Mutex::new(Some(incoming_rx)),
        }
    }

    /// Create an A2A transport that will connect to the given URL.
    ///
    /// Returns an error if the URL is not valid.
    pub fn from_url(url: impl Into<String>) -> Result<Self> {
        let url_str = url.into();
        let parsed: url::Url = url_str
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid A2A URL: {e}"))?;
        let client = brainwires_a2a::A2aClient::new_jsonrpc(parsed);
        let (incoming_tx, incoming_rx) = broadcast::channel(256);
        Ok(Self {
            client: Arc::new(client),
            connected: false,
            endpoint: Some(url_str),
            incoming_tx,
            incoming_rx: Mutex::new(Some(incoming_rx)),
        })
    }

    /// Get a sender handle for pushing received A2A messages into the
    /// transport's receive queue.
    ///
    /// When running an A2A server alongside this transport, the server
    /// handler should convert incoming A2A messages to
    /// [`MessageEnvelope`]s and send them through this channel.
    pub fn incoming_sender(&self) -> broadcast::Sender<MessageEnvelope> {
        self.incoming_tx.clone()
    }

    /// Get a reference to the underlying A2A client.
    pub fn a2a_client(&self) -> &brainwires_a2a::A2aClient {
        &self.client
    }
}

impl std::fmt::Debug for A2aTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aTransport")
            .field("connected", &self.connected)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

#[async_trait]
impl Transport for A2aTransport {
    async fn connect(&mut self, target: &TransportAddress) -> Result<()> {
        match target {
            TransportAddress::Url(url) => {
                self.endpoint = Some(url.clone());
                self.connected = true;
                Ok(())
            }
            _ => bail!("A2aTransport only supports URL addresses"),
        }
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    async fn send(&self, envelope: &MessageEnvelope) -> Result<()> {
        if !self.connected {
            bail!("A2aTransport not connected");
        }

        // Convert MessageEnvelope payload to A2A Message
        let text = match &envelope.payload {
            Payload::Text(s) => s.clone(),
            Payload::Json(v) => serde_json::to_string(v)?,
            Payload::Binary(b) => {
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b)
            }
        };

        let mut message = brainwires_a2a::Message::user_text(&text);

        // Use the envelope ID as the message context
        message.context_id = Some(envelope.id.to_string());

        // Set correlation as task ID if present
        if let Some(corr) = &envelope.correlation_id {
            message.task_id = Some(corr.to_string());
        }

        let req = brainwires_a2a::SendMessageRequest {
            message,
            tenant: None,
            configuration: None,
            metadata: None,
        };

        self.client
            .send_message(req)
            .await
            .map_err(|e| anyhow::anyhow!("A2A send failed: {e}"))?;

        Ok(())
    }

    async fn receive(&self) -> Result<Option<MessageEnvelope>> {
        if !self.connected {
            bail!("A2aTransport not connected");
        }

        let mut rx_guard = self.incoming_rx.lock().await;
        if let Some(rx) = rx_guard.as_mut() {
            match rx.recv().await {
                Ok(envelope) => Ok(Some(envelope)),
                Err(broadcast::error::RecvError::Closed) => Ok(None),
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("A2aTransport receiver lagged by {n} messages");
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
        TransportType::A2a
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

/// Convert an incoming A2A [`Message`](brainwires_a2a::Message) to a
/// [`MessageEnvelope`].
///
/// This helper is useful when running an A2A server: convert the
/// incoming A2A message and push it into the transport via
/// [`A2aTransport::incoming_sender`].
pub fn a2a_message_to_envelope(
    msg: &brainwires_a2a::Message,
    sender_id: uuid::Uuid,
    recipient: MessageTarget,
) -> MessageEnvelope {
    // Extract text from parts
    let text = msg
        .parts
        .iter()
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");

    let payload = if text.is_empty() {
        // Serialize the entire message as JSON
        Payload::Json(serde_json::to_value(msg).unwrap_or_default())
    } else {
        Payload::Text(text)
    };

    let mut envelope = MessageEnvelope::direct(sender_id, uuid::Uuid::nil(), payload);
    envelope.recipient = recipient;

    // Map context_id to correlation
    if let Some(ctx) = &msg.context_id
        && let Ok(uuid) = ctx.parse()
    {
        envelope.correlation_id = Some(uuid);
    }

    envelope
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a2a_transport_type() {
        let url: url::Url = "https://example.com".parse().unwrap();
        let client = brainwires_a2a::A2aClient::new_jsonrpc(url);
        let t = A2aTransport::new(client);
        assert_eq!(t.transport_type(), TransportType::A2a);
        assert!(!t.is_connected());
    }

    #[test]
    fn a2a_transport_debug() {
        let t = A2aTransport::from_url("https://example.com/a2a").unwrap();
        let debug = format!("{t:?}");
        assert!(debug.contains("A2aTransport"));
        assert!(debug.contains("https://example.com/a2a"));
    }

    #[test]
    fn a2a_message_conversion() {
        let msg = brainwires_a2a::Message::user_text("Hello from A2A");
        let env = a2a_message_to_envelope(&msg, uuid::Uuid::new_v4(), MessageTarget::Broadcast);
        match env.payload {
            Payload::Text(s) => assert_eq!(s, "Hello from A2A"),
            _ => panic!("expected Text payload"),
        }
    }
}
