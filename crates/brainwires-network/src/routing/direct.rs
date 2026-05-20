use anyhow::{Result, bail};
use async_trait::async_trait;

use super::peer_table::PeerTable;
use super::traits::{Router, RoutingStrategy};
use crate::transport::TransportAddress;
use crate::{MessageEnvelope, MessageTarget};

/// Point-to-point router.
///
/// Looks up the recipient UUID in the peer table and returns its
/// transport addresses. Fails if the recipient is unknown.
#[derive(Debug, Default)]
pub struct DirectRouter;

impl DirectRouter {
    /// Create a new direct router.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Router for DirectRouter {
    async fn route(
        &self,
        envelope: &MessageEnvelope,
        peers: &PeerTable,
    ) -> Result<Vec<TransportAddress>> {
        match &envelope.recipient {
            MessageTarget::Direct(id) => peers
                .addresses(id)
                .map(|addrs| addrs.to_vec())
                .ok_or_else(|| anyhow::anyhow!("no route to peer {id}")),
            MessageTarget::Broadcast => {
                bail!("DirectRouter does not handle broadcast messages");
            }
            MessageTarget::Topic(_) => {
                bail!("DirectRouter does not handle topic messages");
            }
        }
    }

    fn strategy(&self) -> RoutingStrategy {
        RoutingStrategy::Direct
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use crate::identity::AgentIdentity;
    use uuid::Uuid;

    #[tokio::test]
    async fn direct_routes_to_known_peer() {
        let router = DirectRouter::new();
        let mut peers = PeerTable::new();

        let identity = AgentIdentity::new("target");
        let target_id = identity.id;
        let addr = TransportAddress::Tcp("127.0.0.1:9090".parse().unwrap());
        peers.upsert(identity, vec![addr.clone()]);

        let env = MessageEnvelope::direct(Uuid::new_v4(), target_id, Payload::Text("hello".into()));

        let addrs = router.route(&env, &peers).await.unwrap();
        assert_eq!(addrs, vec![addr]);
    }

    #[tokio::test]
    async fn direct_fails_for_unknown_peer() {
        let router = DirectRouter::new();
        let peers = PeerTable::new();

        let env = MessageEnvelope::direct(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Payload::Text("hello".into()),
        );

        let result = router.route(&env, &peers).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn direct_rejects_broadcast() {
        let router = DirectRouter::new();
        let peers = PeerTable::new();

        let env = MessageEnvelope::broadcast(Uuid::new_v4(), Payload::Text("hello".into()));
        let result = router.route(&env, &peers).await;
        assert!(result.is_err());
    }
}
