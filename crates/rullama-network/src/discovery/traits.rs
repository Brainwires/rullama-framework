use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::identity::AgentIdentity;

/// Identifies which discovery protocol is in use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    /// Explicit manual peer list.
    Manual,
    /// HTTP-backed central registry.
    Registry,
    /// mDNS zero-config LAN discovery.
    Mdns,
    /// Gossip-based decentralized peer exchange.
    Gossip,
    /// Custom protocol with a user-defined identifier.
    Custom(String),
}

/// The core discovery trait.
///
/// Implementations provide a way for agents to register their presence,
/// discover peers, and look up specific agents by UUID.
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Register this agent's identity with the discovery service.
    async fn register(&self, identity: &AgentIdentity) -> Result<()>;

    /// Deregister this agent from the discovery service.
    async fn deregister(&self, id: &Uuid) -> Result<()>;

    /// Discover all currently known/reachable peers.
    async fn discover(&self) -> Result<Vec<AgentIdentity>>;

    /// Look up a specific agent by UUID.
    async fn lookup(&self, id: &Uuid) -> Result<Option<AgentIdentity>>;

    /// The discovery protocol this implementation uses.
    fn protocol(&self) -> DiscoveryProtocol;
}
