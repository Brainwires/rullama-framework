use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

use crate::discovery::Discovery;
use crate::event::TransportType;
use crate::identity::AgentIdentity;
use crate::routing::{BroadcastRouter, ContentRouter, DirectRouter, PeerTable, Router};
use crate::transport::{Transport, TransportAddress};
use crate::{MessageEnvelope, MessageTarget, NetworkEvent, Payload};

/// The user-facing API for the networking stack.
///
/// `NetworkManager` ties together all five layers (identity, transport,
/// routing, discovery, application) and provides a simple interface for
/// sending messages, discovering peers, and subscribing to network events.
///
/// # Example
///
/// ```rust,ignore
/// let manager = NetworkManagerBuilder::new(my_identity)
///     .add_transport(IpcTransport::new(None))
///     .with_discovery(ManualDiscovery::new())
///     .build();
///
/// manager.send(peer_id, "hello").await?;
///
/// let mut events = manager.subscribe();
/// while let Ok(event) = events.recv().await {
///     // handle events
/// }
/// ```
pub struct NetworkManager {
    /// This agent's identity.
    identity: AgentIdentity,
    /// Connected transports keyed by type.
    transports: HashMap<TransportType, Box<dyn Transport>>,
    /// Registered routers.
    direct_router: DirectRouter,
    broadcast_router: BroadcastRouter,
    content_router: ContentRouter,
    /// Custom router (optional override).
    custom_router: Option<Box<dyn Router>>,
    /// Discovery services.
    discoveries: Vec<Box<dyn Discovery>>,
    /// Peer table (shared with routers).
    peer_table: Arc<RwLock<PeerTable>>,
    /// Event broadcast channel.
    event_tx: broadcast::Sender<NetworkEvent>,
}

impl NetworkManager {
    /// Get this agent's identity.
    pub fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    /// Subscribe to network events.
    pub fn subscribe(&self) -> broadcast::Receiver<NetworkEvent> {
        self.event_tx.subscribe()
    }

    /// Get a read lock on the peer table.
    pub async fn peer_table(&self) -> tokio::sync::RwLockReadGuard<'_, PeerTable> {
        self.peer_table.read().await
    }

    /// Get a write lock on the peer table.
    pub async fn peer_table_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, PeerTable> {
        self.peer_table.write().await
    }

    /// List all known peers.
    pub async fn peers(&self) -> Vec<AgentIdentity> {
        self.peer_table.read().await.all_peers().cloned().collect()
    }

    /// Send a message to a specific peer.
    pub async fn send(&self, target: Uuid, payload: impl Into<Payload>) -> Result<()> {
        let envelope = MessageEnvelope::direct(self.identity.id, target, payload);
        self.send_envelope(envelope).await
    }

    /// Broadcast a message to all known peers.
    pub async fn broadcast(&self, payload: impl Into<Payload>) -> Result<()> {
        let envelope = MessageEnvelope::broadcast(self.identity.id, payload);
        self.send_envelope(envelope).await
    }

    /// Publish a message to a topic.
    pub async fn publish(
        &self,
        topic: impl Into<String>,
        payload: impl Into<Payload>,
    ) -> Result<()> {
        let envelope = MessageEnvelope::topic(self.identity.id, topic, payload);
        self.send_envelope(envelope).await
    }

    /// Send a pre-built envelope.
    pub async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        let peer_table = self.peer_table.read().await;

        // Route the message
        let addresses = match &envelope.recipient {
            MessageTarget::Direct(_) => {
                if let Some(router) = &self.custom_router {
                    router.route(&envelope, &peer_table).await?
                } else {
                    self.direct_router.route(&envelope, &peer_table).await?
                }
            }
            MessageTarget::Broadcast => self.broadcast_router.route(&envelope, &peer_table).await?,
            MessageTarget::Topic(_) => self.content_router.route(&envelope, &peer_table).await?,
        };

        drop(peer_table);

        if addresses.is_empty() {
            bail!("No delivery addresses resolved for message");
        }

        // Deliver to each address via the appropriate transport
        for addr in &addresses {
            let transport_type = transport_type_for_address(addr);
            if let Some(transport) = self.transports.get(&transport_type) {
                transport.send(&envelope).await?;
            } else {
                tracing::warn!(
                    "No transport available for address {addr} (type {transport_type:?})"
                );
            }
        }

        Ok(())
    }

    /// Add a transport to the manager.
    pub fn add_transport(&mut self, transport: Box<dyn Transport>) {
        let t = transport.transport_type();
        self.transports.insert(t, transport);
    }

    /// Set a custom router (overrides the default direct router for
    /// point-to-point messages).
    pub fn set_custom_router(&mut self, router: Box<dyn Router>) {
        self.custom_router = Some(router);
    }

    /// Add a discovery service.
    pub fn add_discovery(&mut self, discovery: Box<dyn Discovery>) {
        self.discoveries.push(discovery);
    }

    /// Register this agent with all discovery services.
    pub async fn register_self(&self) -> Result<()> {
        for d in &self.discoveries {
            d.register(&self.identity).await?;
        }
        Ok(())
    }

    /// Deregister this agent from all discovery services.
    pub async fn deregister_self(&self) -> Result<()> {
        for d in &self.discoveries {
            d.deregister(&self.identity.id).await?;
        }
        Ok(())
    }

    /// Run discovery across all services and update the peer table.
    pub async fn discover_peers(&self) -> Result<Vec<AgentIdentity>> {
        let mut all_peers = Vec::new();

        for d in &self.discoveries {
            match d.discover().await {
                Ok(peers) => all_peers.extend(peers),
                Err(e) => {
                    tracing::warn!("Discovery via {:?} failed: {e}", d.protocol());
                }
            }
        }

        // Deduplicate by UUID
        let mut seen = std::collections::HashSet::new();
        all_peers.retain(|p| seen.insert(p.id));

        // Update peer table and emit events
        let mut table = self.peer_table.write().await;
        for peer in &all_peers {
            if peer.id == self.identity.id {
                continue; // Don't add self
            }
            if table.get(&peer.id).is_none() {
                // New peer
                let addrs = endpoint_to_addresses(peer);
                table.upsert(peer.clone(), addrs);
                let _ = self.event_tx.send(NetworkEvent::PeerJoined(peer.clone()));
            }
        }

        Ok(all_peers)
    }

    /// Emit a network event.
    pub fn emit(&self, event: NetworkEvent) {
        let _ = self.event_tx.send(event);
    }
}

/// Builder for [`NetworkManager`].
pub struct NetworkManagerBuilder {
    identity: AgentIdentity,
    transports: HashMap<TransportType, Box<dyn Transport>>,
    custom_router: Option<Box<dyn Router>>,
    discoveries: Vec<Box<dyn Discovery>>,
    event_buffer: usize,
}

impl NetworkManagerBuilder {
    /// Start building a NetworkManager for the given agent identity.
    pub fn new(identity: AgentIdentity) -> Self {
        Self {
            identity,
            transports: HashMap::new(),
            custom_router: None,
            discoveries: Vec::new(),
            event_buffer: 256,
        }
    }

    /// Add a transport.
    pub fn add_transport(mut self, transport: Box<dyn Transport>) -> Self {
        let t = transport.transport_type();
        self.transports.insert(t, transport);
        self
    }

    /// Set a custom router for direct messages.
    pub fn with_router(mut self, router: Box<dyn Router>) -> Self {
        self.custom_router = Some(router);
        self
    }

    /// Add a discovery service.
    pub fn add_discovery(mut self, discovery: Box<dyn Discovery>) -> Self {
        self.discoveries.push(discovery);
        self
    }

    /// Set the event broadcast buffer size (default: 256).
    pub fn event_buffer(mut self, size: usize) -> Self {
        self.event_buffer = size;
        self
    }

    /// Build the [`NetworkManager`].
    pub fn build(self) -> NetworkManager {
        let (event_tx, _) = broadcast::channel(self.event_buffer);

        NetworkManager {
            identity: self.identity,
            transports: self.transports,
            direct_router: DirectRouter::new(),
            broadcast_router: BroadcastRouter::new(),
            content_router: ContentRouter::new(),
            custom_router: self.custom_router,
            discoveries: self.discoveries,
            peer_table: Arc::new(RwLock::new(PeerTable::new())),
            event_tx,
        }
    }
}

/// Infer the transport type from a transport address.
fn transport_type_for_address(addr: &TransportAddress) -> TransportType {
    match addr {
        TransportAddress::Unix(_) => TransportType::Ipc,
        TransportAddress::Tcp(_) => TransportType::Tcp,
        TransportAddress::Url(_) => TransportType::Remote,
        TransportAddress::Channel(_) => TransportType::PubSub,
    }
}

/// Extract transport addresses from an agent's advertised endpoint.
fn endpoint_to_addresses(identity: &AgentIdentity) -> Vec<TransportAddress> {
    let Some(endpoint) = &identity.agent_card.endpoint else {
        return Vec::new();
    };

    if let Some(path) = endpoint.strip_prefix("unix://") {
        vec![TransportAddress::Unix(path.into())]
    } else if let Some(addr) = endpoint.strip_prefix("tcp://") {
        if let Ok(sock) = addr.parse() {
            vec![TransportAddress::Tcp(sock)]
        } else {
            vec![TransportAddress::Url(endpoint.clone())]
        }
    } else {
        vec![TransportAddress::Url(endpoint.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::ManualDiscovery;

    #[tokio::test]
    async fn builder_creates_manager() {
        let identity = AgentIdentity::new("test-agent");
        let manager = NetworkManagerBuilder::new(identity.clone())
            .add_discovery(Box::new(ManualDiscovery::new()))
            .build();

        assert_eq!(manager.identity().name, "test-agent");
        assert!(manager.peers().await.is_empty());
    }

    #[tokio::test]
    async fn discover_peers_populates_table() {
        let agent_a = AgentIdentity::new("agent-a");
        let agent_b = AgentIdentity::new("agent-b");

        let discovery = ManualDiscovery::with_peers(vec![agent_b.clone()]);

        let manager = NetworkManagerBuilder::new(agent_a)
            .add_discovery(Box::new(discovery))
            .build();

        let mut events = manager.subscribe();

        let found = manager.discover_peers().await.unwrap();
        assert_eq!(found.len(), 1);

        let peers = manager.peers().await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].name, "agent-b");

        // Should have emitted PeerJoined
        let event = events.try_recv().unwrap();
        match event {
            NetworkEvent::PeerJoined(p) => assert_eq!(p.id, agent_b.id),
            _ => panic!("expected PeerJoined"),
        }
    }

    #[tokio::test]
    async fn register_and_deregister_self() {
        let identity = AgentIdentity::new("self");
        let discovery = ManualDiscovery::new();

        let manager = NetworkManagerBuilder::new(identity.clone())
            .add_discovery(Box::new(discovery.clone()))
            .build();

        manager.register_self().await.unwrap();

        let peers = discovery.discover().await.unwrap();
        assert_eq!(peers.len(), 1);

        manager.deregister_self().await.unwrap();

        let peers = discovery.discover().await.unwrap();
        assert!(peers.is_empty());
    }

    #[test]
    fn transport_type_inference() {
        assert_eq!(
            transport_type_for_address(&TransportAddress::Unix("/tmp/test.sock".into())),
            TransportType::Ipc
        );
        assert_eq!(
            transport_type_for_address(&TransportAddress::Tcp("127.0.0.1:9090".parse().unwrap())),
            TransportType::Tcp
        );
        assert_eq!(
            transport_type_for_address(&TransportAddress::Url("https://example.com".into())),
            TransportType::Remote
        );
        assert_eq!(
            transport_type_for_address(&TransportAddress::Channel("events".into())),
            TransportType::PubSub
        );
    }

    #[test]
    fn endpoint_parsing() {
        let mut identity = AgentIdentity::new("test");

        identity.agent_card.endpoint = Some("unix:///tmp/agent.sock".into());
        let addrs = endpoint_to_addresses(&identity);
        assert_eq!(
            addrs,
            vec![TransportAddress::Unix("/tmp/agent.sock".into())]
        );

        identity.agent_card.endpoint = Some("tcp://127.0.0.1:9090".into());
        let addrs = endpoint_to_addresses(&identity);
        assert_eq!(
            addrs,
            vec![TransportAddress::Tcp("127.0.0.1:9090".parse().unwrap())]
        );

        identity.agent_card.endpoint = Some("https://api.example.com".into());
        let addrs = endpoint_to_addresses(&identity);
        assert_eq!(
            addrs,
            vec![TransportAddress::Url("https://api.example.com".into())]
        );

        identity.agent_card.endpoint = None;
        let addrs = endpoint_to_addresses(&identity);
        assert!(addrs.is_empty());
    }
}
