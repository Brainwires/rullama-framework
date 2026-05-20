use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::NetworkError;
use crate::identity::AgentIdentity;

/// Supported mesh topology shapes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Central coordinator with spoke nodes.
    Star,
    /// Circular ring where each node connects to the next.
    Ring,
    /// Every node connects to every other node.
    FullMesh,
    /// Tree-like structure with parent/child relationships.
    Hierarchical,
    /// User-defined topology with explicit adjacency.
    Custom(String),
}

/// Trait for managing the shape of the agent mesh.
#[async_trait]
pub trait MeshTopology: Send + Sync {
    /// Add a node to the topology.
    async fn add_node(&mut self, node: AgentIdentity) -> Result<(), NetworkError>;

    /// Remove a node from the topology by its identifier.
    async fn remove_node(&mut self, node_id: &Uuid) -> Result<(), NetworkError>;

    /// Return the identifiers of nodes adjacent to the given node.
    async fn get_neighbors(&self, node_id: &Uuid) -> Result<Vec<Uuid>, NetworkError>;

    /// Return the type of this topology.
    fn topology_type(&self) -> TopologyType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_type_equality() {
        assert_eq!(TopologyType::Star, TopologyType::Star);
        assert_eq!(TopologyType::Ring, TopologyType::Ring);
        assert_eq!(TopologyType::FullMesh, TopologyType::FullMesh);
        assert_eq!(TopologyType::Hierarchical, TopologyType::Hierarchical);
        assert_ne!(TopologyType::Star, TopologyType::Ring);
    }

    #[test]
    fn topology_type_custom_equality() {
        let a = TopologyType::Custom("my-topo".into());
        let b = TopologyType::Custom("my-topo".into());
        let c = TopologyType::Custom("other".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn topology_type_serde_roundtrip() {
        let variants = vec![
            TopologyType::Star,
            TopologyType::Ring,
            TopologyType::FullMesh,
            TopologyType::Hierarchical,
            TopologyType::Custom("lattice".into()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: TopologyType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn topology_type_custom_serde_preserves_name() {
        let topo = TopologyType::Custom("hypercube".into());
        let json = serde_json::to_string(&topo).unwrap();
        assert!(json.contains("hypercube"));
        let deserialized: TopologyType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, TopologyType::Custom("hypercube".into()));
    }

    #[test]
    fn topology_type_debug() {
        let topo = TopologyType::FullMesh;
        let debug = format!("{:?}", topo);
        assert!(debug.contains("FullMesh"));
    }

    #[test]
    fn topology_type_clone() {
        let original = TopologyType::Custom("tree".into());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }
}
