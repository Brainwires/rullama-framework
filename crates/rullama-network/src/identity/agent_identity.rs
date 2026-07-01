use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A protocol identifier (e.g. `"mcp"`, `"a2a"`, `"ipc"`).
pub type ProtocolId = String;

/// An agent's identity on the network.
///
/// Every agent that participates in the networking layer has an identity
/// consisting of a unique ID, a human-readable name, and an [`AgentCard`]
/// that advertises its capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Globally unique identifier for this agent.
    pub id: Uuid,
    /// Human-readable name (e.g. `"code-review-agent"`).
    pub name: String,
    /// Capability advertisement.
    pub agent_card: AgentCard,
}

impl AgentIdentity {
    /// Create a new identity with the given name and an empty agent card.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            agent_card: AgentCard::default(),
        }
    }

    /// Create a new identity with a specific UUID.
    pub fn with_id(id: Uuid, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            agent_card: AgentCard::default(),
        }
    }
}

/// Capability advertisement for an agent.
///
/// Inspired by A2A's AgentCard concept — describes what an agent can do,
/// which protocols it speaks, and how to reach it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentCard {
    /// High-level capabilities this agent offers (e.g. `"code-review"`,
    /// `"file-editing"`, tool names).
    pub capabilities: Vec<String>,
    /// Protocol identifiers this agent supports (e.g. `"mcp"`, `"a2a"`,
    /// `"ipc"`).
    pub supported_protocols: Vec<ProtocolId>,
    /// Arbitrary key-value metadata (e.g. model name, version, region).
    pub metadata: HashMap<String, serde_json::Value>,
    /// Network endpoint if the agent is directly reachable (e.g.
    /// `"tcp://192.168.1.5:9090"` or `"unix:///tmp/agent.sock"`).
    pub endpoint: Option<String>,
    /// Maximum number of tasks this agent can handle concurrently.
    pub max_concurrent_tasks: Option<usize>,
    /// Abstract compute capacity score (higher is more powerful).
    pub compute_capacity: Option<f64>,
}

impl AgentCard {
    /// Check whether this agent supports a given protocol.
    pub fn supports_protocol(&self, protocol: &str) -> bool {
        self.supported_protocols
            .iter()
            .any(|p| p.eq_ignore_ascii_case(protocol))
    }

    /// Check whether this agent has a specific capability.
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case(capability))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_identity_has_unique_id() {
        let a = AgentIdentity::new("agent-a");
        let b = AgentIdentity::new("agent-b");
        assert_ne!(a.id, b.id);
        assert_eq!(a.name, "agent-a");
    }

    #[test]
    fn with_id_preserves_uuid() {
        let id = Uuid::nil();
        let identity = AgentIdentity::with_id(id, "test");
        assert_eq!(identity.id, Uuid::nil());
    }

    #[test]
    fn agent_card_protocol_check() {
        let card = AgentCard {
            supported_protocols: vec!["mcp".into(), "a2a".into()],
            ..Default::default()
        };
        assert!(card.supports_protocol("MCP"));
        assert!(card.supports_protocol("a2a"));
        assert!(!card.supports_protocol("ipc"));
    }

    #[test]
    fn agent_card_capability_check() {
        let card = AgentCard {
            capabilities: vec!["code-review".into(), "file-editing".into()],
            ..Default::default()
        };
        assert!(card.has_capability("Code-Review"));
        assert!(!card.has_capability("deploy"));
    }

    #[test]
    fn identity_serde_roundtrip() {
        let mut identity = AgentIdentity::new("test-agent");
        identity.agent_card.capabilities = vec!["search".into()];
        identity.agent_card.endpoint = Some("tcp://localhost:9090".into());

        let json = serde_json::to_string(&identity).unwrap();
        let deserialized: AgentIdentity = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, identity.id);
        assert_eq!(deserialized.name, "test-agent");
        assert_eq!(deserialized.agent_card.capabilities, vec!["search"]);
        assert_eq!(
            deserialized.agent_card.endpoint.as_deref(),
            Some("tcp://localhost:9090")
        );
    }

    #[test]
    fn default_agent_card_is_empty() {
        let card = AgentCard::default();
        assert!(card.capabilities.is_empty());
        assert!(card.supported_protocols.is_empty());
        assert!(card.metadata.is_empty());
        assert!(card.endpoint.is_none());
        assert!(card.max_concurrent_tasks.is_none());
        assert!(card.compute_capacity.is_none());
    }
}
