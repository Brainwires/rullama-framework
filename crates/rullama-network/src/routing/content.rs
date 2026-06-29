use anyhow::{Result, bail};
use async_trait::async_trait;

use super::peer_table::PeerTable;
use super::traits::{Router, RoutingStrategy};
use crate::transport::TransportAddress;
use crate::{MessageEnvelope, MessageTarget};

/// Content-based (topic) router.
///
/// Routes messages addressed to a [`MessageTarget::Topic`] to all peers
/// that are subscribed to that topic in the [`PeerTable`].
#[derive(Debug, Default)]
pub struct ContentRouter;

impl ContentRouter {
    /// Create a new content router.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Router for ContentRouter {
    async fn route(
        &self,
        envelope: &MessageEnvelope,
        peers: &PeerTable,
    ) -> Result<Vec<TransportAddress>> {
        match &envelope.recipient {
            MessageTarget::Topic(topic) => {
                let subscribers = peers.subscribers(topic);
                let mut addrs = Vec::new();

                for sub_id in &subscribers {
                    // Don't send back to sender
                    if *sub_id == envelope.sender {
                        continue;
                    }
                    if let Some(peer_addrs) = peers.addresses(sub_id) {
                        addrs.extend_from_slice(peer_addrs);
                    }
                }

                Ok(addrs)
            }
            MessageTarget::Direct(_) => {
                bail!("ContentRouter does not handle direct messages");
            }
            MessageTarget::Broadcast => {
                bail!("ContentRouter does not handle broadcast messages");
            }
        }
    }

    fn strategy(&self) -> RoutingStrategy {
        RoutingStrategy::ContentBased
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Payload;
    use crate::identity::AgentIdentity;
    use uuid::Uuid;

    #[tokio::test]
    async fn content_routes_to_subscribers() {
        let router = ContentRouter::new();
        let mut peers = PeerTable::new();

        let sender = AgentIdentity::new("sender");
        let sender_id = sender.id;
        let sub_a = AgentIdentity::new("sub-a");
        let sub_a_id = sub_a.id;
        let sub_b = AgentIdentity::new("sub-b");
        let sub_b_id = sub_b.id;
        let non_sub = AgentIdentity::new("non-sub");

        let addr_a = TransportAddress::Tcp("127.0.0.1:1000".parse().unwrap());
        let addr_b = TransportAddress::Tcp("127.0.0.1:2000".parse().unwrap());
        let addr_ns = TransportAddress::Tcp("127.0.0.1:3000".parse().unwrap());

        peers.upsert(sender, vec![]);
        peers.upsert(sub_a, vec![addr_a.clone()]);
        peers.upsert(sub_b, vec![addr_b.clone()]);
        peers.upsert(non_sub, vec![addr_ns.clone()]);

        peers.subscribe(sub_a_id, "events");
        peers.subscribe(sub_b_id, "events");

        let env = MessageEnvelope::topic(sender_id, "events", Payload::Text("update".into()));
        let addrs = router.route(&env, &peers).await.unwrap();

        assert_eq!(addrs.len(), 2);
        assert!(addrs.contains(&addr_a));
        assert!(addrs.contains(&addr_b));
        assert!(!addrs.contains(&addr_ns));
    }

    #[tokio::test]
    async fn content_empty_topic() {
        let router = ContentRouter::new();
        let peers = PeerTable::new();

        let env = MessageEnvelope::topic(
            Uuid::new_v4(),
            "no-subscribers",
            Payload::Text("hello".into()),
        );
        let addrs = router.route(&env, &peers).await.unwrap();
        assert!(addrs.is_empty());
    }

    #[tokio::test]
    async fn content_rejects_direct() {
        let router = ContentRouter::new();
        let peers = PeerTable::new();

        let env = MessageEnvelope::direct(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Payload::Text("hello".into()),
        );
        assert!(router.route(&env, &peers).await.is_err());
    }
}
