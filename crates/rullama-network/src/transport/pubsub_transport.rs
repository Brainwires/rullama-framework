use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio::sync::{Mutex, broadcast};

use super::traits::{Transport, TransportAddress};
use crate::{MessageEnvelope, TransportType};

/// In-process pub/sub transport for topic-based agent messaging.
///
/// Messages sent to this transport are delivered to all connected
/// subscribers via `tokio::sync::broadcast`. This is useful for
/// same-process agent communication without network overhead.
///
/// For cross-process or cross-network pub/sub, use an external broker
/// transport (NATS, Redis) in the future.
pub struct PubSubTransport {
    /// Map of topic → broadcast sender.
    topics: Arc<Mutex<HashMap<String, broadcast::Sender<MessageEnvelope>>>>,
    /// Incoming message buffer (from subscriptions).
    incoming_tx: broadcast::Sender<MessageEnvelope>,
    /// Receiver for incoming messages.
    incoming_rx: Mutex<Option<broadcast::Receiver<MessageEnvelope>>>,
    /// Whether the transport is "connected" (active).
    connected: bool,
    /// Buffer size for each topic channel.
    buffer_size: usize,
}

impl PubSubTransport {
    /// Create a new in-process pub/sub transport.
    pub fn new() -> Self {
        Self::with_buffer_size(256)
    }

    /// Create a pub/sub transport with a custom buffer size per topic.
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        let (incoming_tx, incoming_rx) = broadcast::channel(buffer_size);
        Self {
            topics: Arc::new(Mutex::new(HashMap::new())),
            incoming_tx,
            incoming_rx: Mutex::new(Some(incoming_rx)),
            connected: false,
            buffer_size,
        }
    }

    /// Subscribe to a topic, returning a receiver for messages on that topic.
    pub async fn subscribe_topic(&self, topic: &str) -> broadcast::Receiver<MessageEnvelope> {
        let mut topics = self.topics.lock().await;
        let sender = topics
            .entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(self.buffer_size).0);
        sender.subscribe()
    }

    /// Get or create a sender for a topic.
    async fn get_topic_sender(&self, topic: &str) -> broadcast::Sender<MessageEnvelope> {
        let mut topics = self.topics.lock().await;
        topics
            .entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(self.buffer_size).0)
            .clone()
    }

    /// Get a sender handle for pushing received messages into the transport.
    ///
    /// External pub/sub adapters can use this to feed messages from an
    /// external broker into the transport's receive queue.
    pub fn incoming_sender(&self) -> broadcast::Sender<MessageEnvelope> {
        self.incoming_tx.clone()
    }
}

impl Default for PubSubTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for PubSubTransport {
    async fn connect(&mut self, target: &TransportAddress) -> Result<()> {
        match target {
            TransportAddress::Channel(_) => {
                self.connected = true;
                Ok(())
            }
            _ => bail!("PubSubTransport only supports Channel addresses"),
        }
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    async fn send(&self, envelope: &MessageEnvelope) -> Result<()> {
        if !self.connected {
            bail!("PubSubTransport not connected");
        }

        match &envelope.recipient {
            crate::MessageTarget::Topic(topic) => {
                let sender = self.get_topic_sender(topic).await;
                // It's OK if no one is listening (send returns Err if no receivers)
                let _ = sender.send(envelope.clone());
                Ok(())
            }
            crate::MessageTarget::Broadcast => {
                // Broadcast to all topics
                let topics = self.topics.lock().await;
                for sender in topics.values() {
                    let _ = sender.send(envelope.clone());
                }
                Ok(())
            }
            crate::MessageTarget::Direct(_) => {
                bail!("PubSubTransport does not support direct messages; use topic addressing");
            }
        }
    }

    async fn receive(&self) -> Result<Option<MessageEnvelope>> {
        if !self.connected {
            bail!("PubSubTransport not connected");
        }

        let mut rx_guard = self.incoming_rx.lock().await;
        if let Some(rx) = rx_guard.as_mut() {
            match rx.recv().await {
                Ok(envelope) => Ok(Some(envelope)),
                Err(broadcast::error::RecvError::Closed) => Ok(None),
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("PubSubTransport receiver lagged by {n} messages");
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
        TransportType::PubSub
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

    #[tokio::test]
    async fn pubsub_topic_delivery() {
        let transport = PubSubTransport::new();

        // Subscribe before sending
        let mut rx = transport.subscribe_topic("events").await;

        // Send a topic message
        let env = MessageEnvelope::topic(Uuid::new_v4(), "events", Payload::Text("update".into()));
        let sender = transport.get_topic_sender("events").await;
        sender.send(env.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, env.id);
    }

    #[tokio::test]
    async fn pubsub_no_subscribers_ok() {
        let mut transport = PubSubTransport::new();
        transport
            .connect(&TransportAddress::Channel("test".into()))
            .await
            .unwrap();

        let env = MessageEnvelope::topic(
            Uuid::new_v4(),
            "nobody-listening",
            Payload::Text("hello".into()),
        );

        // Should not error even with no subscribers
        transport.send(&env).await.unwrap();
    }

    #[tokio::test]
    async fn pubsub_rejects_direct() {
        let mut transport = PubSubTransport::new();
        transport
            .connect(&TransportAddress::Channel("test".into()))
            .await
            .unwrap();

        let env = MessageEnvelope::direct(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Payload::Text("hello".into()),
        );

        assert!(transport.send(&env).await.is_err());
    }

    #[test]
    fn pubsub_transport_type() {
        let t = PubSubTransport::new();
        assert_eq!(t.transport_type(), TransportType::PubSub);
        assert!(!t.is_connected());
    }
}
