use anyhow::Result;
use async_trait::async_trait;

use super::peer_table::PeerTable;
use super::traits::{Router, RoutingStrategy};
use crate::MessageEnvelope;
use crate::transport::TransportAddress;

/// Broadcast router.
///
/// Returns the transport addresses of all known peers (excluding the
/// sender). Used for broadcast and peer-discovery announcements.
#[derive(Debug, Default)]
pub struct BroadcastRouter;

impl BroadcastRouter {
    /// Create a new broadcast router.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Router for BroadcastRouter {
    async fn route(
        &self,
        envelope: &MessageEnvelope,
        peers: &PeerTable,
    ) -> Result<Vec<TransportAddress>> {
        let mut addrs = Vec::new();

        for peer_id in peers.all_peer_ids() {
            // Don't send back to the sender
            if *peer_id == envelope.sender {
                continue;
            }
            if let Some(peer_addrs) = peers.addresses(peer_id) {
                addrs.extend_from_slice(peer_addrs);
            }
        }

        Ok(addrs)
    }

    fn strategy(&self) -> RoutingStrategy {
        RoutingStrategy::Broadcast
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use crate::identity::AgentIdentity;
    use uuid::Uuid;

    #[tokio::test]
    async fn broadcast_reaches_all_except_sender() {
        let router = BroadcastRouter::new();
        let mut peers = PeerTable::new();

        let sender = AgentIdentity::new("sender");
        let sender_id = sender.id;
        let peer_a = AgentIdentity::new("a");
        let peer_b = AgentIdentity::new("b");

        peers.upsert(
            sender,
            vec![TransportAddress::Tcp("127.0.0.1:1000".parse().unwrap())],
        );
        peers.upsert(
            peer_a,
            vec![TransportAddress::Tcp("127.0.0.1:2000".parse().unwrap())],
        );
        peers.upsert(
            peer_b,
            vec![TransportAddress::Tcp("127.0.0.1:3000".parse().unwrap())],
        );

        let env = MessageEnvelope::broadcast(sender_id, Payload::Text("ping".into()));
        let addrs = router.route(&env, &peers).await.unwrap();

        // Should get addresses for peer_a and peer_b, not sender
        assert_eq!(addrs.len(), 2);
        assert!(!addrs.contains(&TransportAddress::Tcp("127.0.0.1:1000".parse().unwrap())));
    }

    #[tokio::test]
    async fn broadcast_empty_peers() {
        let router = BroadcastRouter::new();
        let peers = PeerTable::new();

        let env = MessageEnvelope::broadcast(Uuid::new_v4(), Payload::Text("ping".into()));
        let addrs = router.route(&env, &peers).await.unwrap();
        assert!(addrs.is_empty());
    }
}
