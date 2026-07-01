use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::error::MeshError;
use super::node::MeshNode;

/// Protocol used for discovering peers in the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    /// Multicast DNS for local-network discovery.
    Mdns,
    /// Gossip-based protocol for decentralized peer exchange.
    Gossip,
    /// Centralized registry service.
    Registry,
    /// Manually configured peer list.
    Manual,
}

/// Trait for peer discovery within the mesh.
#[async_trait]
pub trait PeerDiscovery: Send + Sync {
    /// Discover available peers using the configured protocol.
    async fn discover_peers(&self) -> Result<Vec<MeshNode>, MeshError>;

    /// Register this node so it can be discovered by others.
    fn register_self(&mut self, node: MeshNode) -> Result<(), MeshError>;

    /// Remove this node from the discovery mechanism.
    fn deregister(&mut self) -> Result<(), MeshError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_protocol_equality() {
        assert_eq!(DiscoveryProtocol::Mdns, DiscoveryProtocol::Mdns);
        assert_eq!(DiscoveryProtocol::Gossip, DiscoveryProtocol::Gossip);
        assert_eq!(DiscoveryProtocol::Registry, DiscoveryProtocol::Registry);
        assert_eq!(DiscoveryProtocol::Manual, DiscoveryProtocol::Manual);
        assert_ne!(DiscoveryProtocol::Mdns, DiscoveryProtocol::Gossip);
    }

    #[test]
    fn discovery_protocol_serde_roundtrip() {
        let variants = vec![
            DiscoveryProtocol::Mdns,
            DiscoveryProtocol::Gossip,
            DiscoveryProtocol::Registry,
            DiscoveryProtocol::Manual,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: DiscoveryProtocol = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn discovery_protocol_debug() {
        assert!(format!("{:?}", DiscoveryProtocol::Mdns).contains("Mdns"));
        assert!(format!("{:?}", DiscoveryProtocol::Gossip).contains("Gossip"));
    }

    #[test]
    fn discovery_protocol_clone() {
        let original = DiscoveryProtocol::Registry;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn discovery_protocol_json_values() {
        // Verify the exact JSON representations
        assert_eq!(
            serde_json::to_string(&DiscoveryProtocol::Mdns).unwrap(),
            "\"Mdns\""
        );
        assert_eq!(
            serde_json::to_string(&DiscoveryProtocol::Manual).unwrap(),
            "\"Manual\""
        );
    }
}
