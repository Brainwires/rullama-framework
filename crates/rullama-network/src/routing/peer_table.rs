use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::identity::AgentIdentity;
use crate::transport::TransportAddress;

/// A table of known peers and their reachable transport addresses.
///
/// The peer table is the central data structure for routing decisions.
/// It maps agent UUIDs to their identity and one or more transport
/// addresses through which they can be reached.
#[derive(Debug, Default)]
pub struct PeerTable {
    /// Map from agent UUID to identity.
    peers: HashMap<Uuid, AgentIdentity>,
    /// Map from agent UUID to known transport addresses.
    addresses: HashMap<Uuid, Vec<TransportAddress>>,
    /// Map from topic name to subscribed agent UUIDs.
    subscriptions: HashMap<String, HashSet<Uuid>>,
}

impl PeerTable {
    /// Create an empty peer table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a peer in the table.
    pub fn upsert(&mut self, identity: AgentIdentity, addresses: Vec<TransportAddress>) {
        let id = identity.id;
        self.peers.insert(id, identity);
        self.addresses.insert(id, addresses);
    }

    /// Remove a peer from the table.
    pub fn remove(&mut self, id: &Uuid) -> Option<AgentIdentity> {
        self.addresses.remove(id);
        // Remove from all topic subscriptions
        for subs in self.subscriptions.values_mut() {
            subs.remove(id);
        }
        self.peers.remove(id)
    }

    /// Look up a peer's identity.
    pub fn get(&self, id: &Uuid) -> Option<&AgentIdentity> {
        self.peers.get(id)
    }

    /// Get all transport addresses for a peer.
    pub fn addresses(&self, id: &Uuid) -> Option<&[TransportAddress]> {
        self.addresses.get(id).map(|v| v.as_slice())
    }

    /// Get all known peers.
    pub fn all_peers(&self) -> impl Iterator<Item = &AgentIdentity> {
        self.peers.values()
    }

    /// Get all known peer IDs.
    pub fn all_peer_ids(&self) -> impl Iterator<Item = &Uuid> {
        self.peers.keys()
    }

    /// Number of known peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// Subscribe a peer to a topic.
    pub fn subscribe(&mut self, peer_id: Uuid, topic: impl Into<String>) {
        self.subscriptions
            .entry(topic.into())
            .or_default()
            .insert(peer_id);
    }

    /// Unsubscribe a peer from a topic.
    pub fn unsubscribe(&mut self, peer_id: &Uuid, topic: &str) {
        if let Some(subs) = self.subscriptions.get_mut(topic) {
            subs.remove(peer_id);
            if subs.is_empty() {
                self.subscriptions.remove(topic);
            }
        }
    }

    /// Get all peers subscribed to a topic.
    pub fn subscribers(&self, topic: &str) -> Vec<Uuid> {
        self.subscriptions
            .get(topic)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(name: &str) -> (AgentIdentity, Vec<TransportAddress>) {
        let identity = AgentIdentity::new(name);
        let addrs = vec![TransportAddress::Tcp("127.0.0.1:9090".parse().unwrap())];
        (identity, addrs)
    }

    #[test]
    fn upsert_and_get() {
        let mut table = PeerTable::new();
        let (identity, addrs) = make_peer("agent-a");
        let id = identity.id;

        table.upsert(identity, addrs);

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
        assert!(table.get(&id).is_some());
        assert_eq!(table.get(&id).unwrap().name, "agent-a");
        assert_eq!(table.addresses(&id).unwrap().len(), 1);
    }

    #[test]
    fn remove_peer() {
        let mut table = PeerTable::new();
        let (identity, addrs) = make_peer("agent-b");
        let id = identity.id;

        table.upsert(identity, addrs);
        let removed = table.remove(&id);
        assert!(removed.is_some());
        assert_eq!(table.len(), 0);
        assert!(table.get(&id).is_none());
    }

    #[test]
    fn topic_subscriptions() {
        let mut table = PeerTable::new();
        let (id_a, addrs_a) = make_peer("a");
        let (id_b, addrs_b) = make_peer("b");
        let uuid_a = id_a.id;
        let uuid_b = id_b.id;

        table.upsert(id_a, addrs_a);
        table.upsert(id_b, addrs_b);

        table.subscribe(uuid_a, "status");
        table.subscribe(uuid_b, "status");
        table.subscribe(uuid_a, "errors");

        assert_eq!(table.subscribers("status").len(), 2);
        assert_eq!(table.subscribers("errors").len(), 1);
        assert!(table.subscribers("unknown").is_empty());

        table.unsubscribe(&uuid_a, "status");
        assert_eq!(table.subscribers("status").len(), 1);
    }

    #[test]
    fn remove_peer_cleans_subscriptions() {
        let mut table = PeerTable::new();
        let (identity, addrs) = make_peer("agent-c");
        let id = identity.id;

        table.upsert(identity, addrs);
        table.subscribe(id, "events");
        assert_eq!(table.subscribers("events").len(), 1);

        table.remove(&id);
        assert!(table.subscribers("events").is_empty());
    }
}
