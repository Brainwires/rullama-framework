use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::MeshError;

/// Strategy used to route messages between mesh nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Send directly to a specific node.
    DirectRoute,
    /// Use the shortest path through the mesh.
    ShortestPath,
    /// Distribute messages across nodes to balance load.
    LoadBalanced,
    /// Send to all nodes in the mesh.
    Broadcast,
    /// Send to a specific subset of nodes.
    Multicast(Vec<Uuid>),
}

/// A single entry in the routing table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEntry {
    /// Final destination node.
    pub destination: Uuid,
    /// Next hop on the path to the destination.
    pub next_hop: Uuid,
    /// Routing cost / metric for this path.
    pub cost: f64,
    /// Time-to-live (max hops remaining).
    pub ttl: u32,
}

/// Trait for routing messages through the mesh.
#[async_trait]
pub trait MessageRouter: Send + Sync {
    /// Route a serialized message to the given destination using the specified strategy.
    async fn route_message(
        &self,
        destination: &Uuid,
        payload: &[u8],
        strategy: &RoutingStrategy,
    ) -> Result<(), MeshError>;

    /// Return the current routing table.
    fn get_route_table(&self) -> Vec<RouteEntry>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_strategy_equality() {
        assert_eq!(RoutingStrategy::DirectRoute, RoutingStrategy::DirectRoute);
        assert_eq!(RoutingStrategy::ShortestPath, RoutingStrategy::ShortestPath);
        assert_eq!(RoutingStrategy::LoadBalanced, RoutingStrategy::LoadBalanced);
        assert_eq!(RoutingStrategy::Broadcast, RoutingStrategy::Broadcast);
        assert_ne!(RoutingStrategy::DirectRoute, RoutingStrategy::Broadcast);
    }

    #[test]
    fn routing_strategy_multicast_equality() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let a = RoutingStrategy::Multicast(vec![id1, id2]);
        let b = RoutingStrategy::Multicast(vec![id1, id2]);
        let c = RoutingStrategy::Multicast(vec![id1]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn routing_strategy_serde_roundtrip() {
        let id = Uuid::new_v4();
        let variants = vec![
            RoutingStrategy::DirectRoute,
            RoutingStrategy::ShortestPath,
            RoutingStrategy::LoadBalanced,
            RoutingStrategy::Broadcast,
            RoutingStrategy::Multicast(vec![id]),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: RoutingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn route_entry_construction() {
        let dest = Uuid::new_v4();
        let hop = Uuid::new_v4();
        let entry = RouteEntry {
            destination: dest,
            next_hop: hop,
            cost: 1.5,
            ttl: 10,
        };
        assert_eq!(entry.destination, dest);
        assert_eq!(entry.next_hop, hop);
        assert!((entry.cost - 1.5).abs() < f64::EPSILON);
        assert_eq!(entry.ttl, 10);
    }

    #[test]
    fn route_entry_serde_roundtrip() {
        let entry = RouteEntry {
            destination: Uuid::new_v4(),
            next_hop: Uuid::new_v4(),
            cost: 3.14,
            ttl: 64,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: RouteEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.destination, entry.destination);
        assert_eq!(deserialized.next_hop, entry.next_hop);
        assert!((deserialized.cost - entry.cost).abs() < f64::EPSILON);
        assert_eq!(deserialized.ttl, entry.ttl);
    }

    #[test]
    fn route_entry_zero_ttl() {
        let entry = RouteEntry {
            destination: Uuid::nil(),
            next_hop: Uuid::nil(),
            cost: 0.0,
            ttl: 0,
        };
        assert_eq!(entry.ttl, 0);
        assert!((entry.cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn routing_strategy_multicast_empty() {
        let strategy = RoutingStrategy::Multicast(vec![]);
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: RoutingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, RoutingStrategy::Multicast(vec![]));
    }
}
