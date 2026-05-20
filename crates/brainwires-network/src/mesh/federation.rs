use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::NetworkError;
use crate::identity::AgentIdentity;

/// Policy governing which peers may join a federated mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederationPolicy {
    /// Any peer may join.
    Open,
    /// Only explicitly listed peers may join.
    AllowList(Vec<Uuid>),
    /// All peers except those listed may join.
    DenyList(Vec<Uuid>),
    /// Peers are admitted based on required capabilities.
    CapabilityBased(Vec<String>),
}

/// Trait for managing federation between mesh clusters.
#[async_trait]
pub trait FederationGateway: Send + Sync {
    /// Evaluate and optionally accept a peer into the federation.
    async fn accept_peer(&mut self, peer: &AgentIdentity) -> Result<bool, NetworkError>;

    /// Return the current federation policy.
    fn policy(&self) -> &FederationPolicy;

    /// List the identifiers of all currently federated peers.
    fn list_federated_peers(&self) -> Vec<Uuid>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federation_policy_equality() {
        assert_eq!(FederationPolicy::Open, FederationPolicy::Open);
        assert_ne!(FederationPolicy::Open, FederationPolicy::AllowList(vec![]));
    }

    #[test]
    fn federation_policy_allow_list() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let policy = FederationPolicy::AllowList(vec![id1, id2]);
        if let FederationPolicy::AllowList(ids) = &policy {
            assert_eq!(ids.len(), 2);
            assert!(ids.contains(&id1));
            assert!(ids.contains(&id2));
        } else {
            panic!("expected AllowList");
        }
    }

    #[test]
    fn federation_policy_deny_list() {
        let id = Uuid::new_v4();
        let policy = FederationPolicy::DenyList(vec![id]);
        if let FederationPolicy::DenyList(ids) = &policy {
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], id);
        } else {
            panic!("expected DenyList");
        }
    }

    #[test]
    fn federation_policy_capability_based() {
        let policy =
            FederationPolicy::CapabilityBased(vec!["inference".into(), "embedding".into()]);
        if let FederationPolicy::CapabilityBased(caps) = &policy {
            assert_eq!(caps.len(), 2);
            assert!(caps.contains(&"inference".to_string()));
        } else {
            panic!("expected CapabilityBased");
        }
    }

    #[test]
    fn federation_policy_serde_roundtrip() {
        let id = Uuid::new_v4();
        let variants = vec![
            FederationPolicy::Open,
            FederationPolicy::AllowList(vec![id]),
            FederationPolicy::DenyList(vec![id]),
            FederationPolicy::CapabilityBased(vec!["tool_use".into()]),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: FederationPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn federation_policy_empty_collections() {
        let empty_allow = FederationPolicy::AllowList(vec![]);
        let empty_deny = FederationPolicy::DenyList(vec![]);
        let empty_caps = FederationPolicy::CapabilityBased(vec![]);

        // All are distinct from each other
        assert_ne!(empty_allow, empty_deny);
        assert_ne!(empty_deny, empty_caps);
        assert_ne!(empty_allow, empty_caps);

        // Serde roundtrip with empty collections
        for policy in [empty_allow, empty_deny, empty_caps] {
            let json = serde_json::to_string(&policy).unwrap();
            let deserialized: FederationPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, deserialized);
        }
    }

    #[test]
    fn federation_policy_clone() {
        let id = Uuid::new_v4();
        let original = FederationPolicy::AllowList(vec![id]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }
}
