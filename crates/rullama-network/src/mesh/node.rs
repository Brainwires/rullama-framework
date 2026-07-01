use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The current state of a mesh node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is starting up and not yet ready.
    Initializing,
    /// Node is active and accepting work.
    Active,
    /// Node is draining in-flight tasks before shutdown.
    Draining,
    /// Node has lost connectivity.
    Disconnected,
    /// Node has encountered an unrecoverable failure.
    Failed,
}

/// Capabilities advertised by a mesh node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Maximum number of tasks the node can run concurrently.
    pub max_concurrent_tasks: usize,
    /// Protocol identifiers the node supports (e.g. "a2a", "mcp").
    pub supported_protocols: Vec<String>,
    /// Tool names available on this node.
    pub available_tools: Vec<String>,
    /// Abstract compute capacity score (higher is more powerful).
    pub compute_capacity: f64,
}

/// A single node within the agent mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshNode {
    /// Unique identifier for this node.
    pub id: Uuid,
    /// Network address (e.g. "host:port" or URI).
    pub address: String,
    /// Current lifecycle state.
    pub state: NodeState,
    /// Advertised capabilities.
    pub capabilities: NodeCapabilities,
    /// Last time this node was seen (ISO-8601 timestamp).
    pub last_seen: String,
    /// Arbitrary metadata attached to the node.
    pub metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            max_concurrent_tasks: 8,
            supported_protocols: vec!["a2a".into(), "mcp".into()],
            available_tools: vec!["read_file".into(), "write_file".into()],
            compute_capacity: 42.5,
        }
    }

    fn sample_node() -> MeshNode {
        MeshNode {
            id: Uuid::nil(),
            address: "127.0.0.1:9090".into(),
            state: NodeState::Active,
            capabilities: sample_capabilities(),
            last_seen: "2025-01-01T00:00:00Z".into(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn node_state_equality() {
        assert_eq!(NodeState::Active, NodeState::Active);
        assert_ne!(NodeState::Active, NodeState::Draining);
        assert_ne!(NodeState::Initializing, NodeState::Failed);
    }

    #[test]
    fn node_state_serde_roundtrip() {
        let states = vec![
            NodeState::Initializing,
            NodeState::Active,
            NodeState::Draining,
            NodeState::Disconnected,
            NodeState::Failed,
        ];
        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let deserialized: NodeState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, deserialized);
        }
    }

    #[test]
    fn node_capabilities_serde_roundtrip() {
        let caps = sample_capabilities();
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: NodeCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_concurrent_tasks, 8);
        assert_eq!(deserialized.supported_protocols.len(), 2);
        assert_eq!(deserialized.available_tools.len(), 2);
        assert!((deserialized.compute_capacity - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mesh_node_construction_and_fields() {
        let node = sample_node();
        assert_eq!(node.id, Uuid::nil());
        assert_eq!(node.address, "127.0.0.1:9090");
        assert_eq!(node.state, NodeState::Active);
        assert!(node.metadata.is_empty());
    }

    #[test]
    fn mesh_node_serde_roundtrip() {
        let mut node = sample_node();
        node.metadata.insert("region".into(), "us-east".into());

        let json = serde_json::to_string(&node).unwrap();
        let deserialized: MeshNode = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, node.id);
        assert_eq!(deserialized.address, node.address);
        assert_eq!(deserialized.state, node.state);
        assert_eq!(deserialized.metadata.get("region").unwrap(), "us-east");
    }

    #[test]
    fn mesh_node_clone() {
        let node = sample_node();
        let cloned = node.clone();
        assert_eq!(cloned.id, node.id);
        assert_eq!(cloned.address, node.address);
    }

    #[test]
    fn node_capabilities_empty_lists() {
        let caps = NodeCapabilities {
            max_concurrent_tasks: 0,
            supported_protocols: vec![],
            available_tools: vec![],
            compute_capacity: 0.0,
        };
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: NodeCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_concurrent_tasks, 0);
        assert!(deserialized.supported_protocols.is_empty());
    }

    #[test]
    fn mesh_node_with_metadata() {
        let mut node = sample_node();
        node.metadata.insert("version".into(), "1.0".into());
        node.metadata.insert("role".into(), "worker".into());
        assert_eq!(node.metadata.len(), 2);
        assert_eq!(node.metadata["version"], "1.0");
    }
}
