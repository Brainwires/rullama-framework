use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::traits::{Discovery, DiscoveryProtocol};
use crate::identity::AgentIdentity;

/// Manual peer discovery backed by an in-memory peer list.
///
/// Peers are added and removed explicitly via [`add_peer`](ManualDiscovery::add_peer)
/// and [`remove_peer`](ManualDiscovery::remove_peer). No network calls
/// are made — this is useful for testing, static configurations, and
/// bootstrapping other discovery mechanisms.
#[derive(Debug, Clone)]
pub struct ManualDiscovery {
    peers: Arc<RwLock<HashMap<Uuid, AgentIdentity>>>,
}

impl ManualDiscovery {
    /// Create a new empty manual discovery instance.
    pub fn new() -> Self {
        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a manual discovery pre-populated with peers.
    pub fn with_peers(peers: Vec<AgentIdentity>) -> Self {
        let map: HashMap<Uuid, AgentIdentity> = peers.into_iter().map(|p| (p.id, p)).collect();
        Self {
            peers: Arc::new(RwLock::new(map)),
        }
    }

    /// Explicitly add a peer.
    pub async fn add_peer(&self, identity: AgentIdentity) {
        self.peers.write().await.insert(identity.id, identity);
    }

    /// Explicitly remove a peer.
    pub async fn remove_peer(&self, id: &Uuid) -> Option<AgentIdentity> {
        self.peers.write().await.remove(id)
    }
}

impl Default for ManualDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Discovery for ManualDiscovery {
    async fn register(&self, identity: &AgentIdentity) -> Result<()> {
        self.peers
            .write()
            .await
            .insert(identity.id, identity.clone());
        Ok(())
    }

    async fn deregister(&self, id: &Uuid) -> Result<()> {
        self.peers.write().await.remove(id);
        Ok(())
    }

    async fn discover(&self) -> Result<Vec<AgentIdentity>> {
        Ok(self.peers.read().await.values().cloned().collect())
    }

    async fn lookup(&self, id: &Uuid) -> Result<Option<AgentIdentity>> {
        Ok(self.peers.read().await.get(id).cloned())
    }

    fn protocol(&self) -> DiscoveryProtocol {
        DiscoveryProtocol::Manual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn manual_register_and_discover() {
        let discovery = ManualDiscovery::new();

        let agent_a = AgentIdentity::new("agent-a");
        let agent_b = AgentIdentity::new("agent-b");

        discovery.register(&agent_a).await.unwrap();
        discovery.register(&agent_b).await.unwrap();

        let peers = discovery.discover().await.unwrap();
        assert_eq!(peers.len(), 2);
    }

    #[tokio::test]
    async fn manual_lookup() {
        let discovery = ManualDiscovery::new();

        let agent = AgentIdentity::new("test");
        let id = agent.id;
        discovery.register(&agent).await.unwrap();

        let found = discovery.lookup(&id).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "test");

        let not_found = discovery.lookup(&Uuid::new_v4()).await.unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn manual_deregister() {
        let discovery = ManualDiscovery::new();

        let agent = AgentIdentity::new("test");
        let id = agent.id;
        discovery.register(&agent).await.unwrap();
        discovery.deregister(&id).await.unwrap();

        let peers = discovery.discover().await.unwrap();
        assert!(peers.is_empty());
    }

    #[tokio::test]
    async fn manual_with_peers() {
        let peers = vec![
            AgentIdentity::new("a"),
            AgentIdentity::new("b"),
            AgentIdentity::new("c"),
        ];
        let discovery = ManualDiscovery::with_peers(peers);

        let discovered = discovery.discover().await.unwrap();
        assert_eq!(discovered.len(), 3);
    }

    #[test]
    fn manual_protocol() {
        let d = ManualDiscovery::new();
        assert_eq!(d.protocol(), DiscoveryProtocol::Manual);
    }
}
