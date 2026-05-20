//! Integration tests for routing — verifying that routers, peer tables,
//! and message envelopes work together correctly across modules.

use brainwires_network::identity::{AgentCard, AgentIdentity};
use brainwires_network::routing::{
    BroadcastRouter, ContentRouter, DirectRouter, PeerTable, Router, RoutingStrategy,
};
use brainwires_network::transport::TransportAddress;
use brainwires_network::{MessageEnvelope, Payload};
use uuid::Uuid;

/// Helper: create an agent identity with a TCP endpoint.
fn agent_with_tcp(name: &str, port: u16) -> (AgentIdentity, Vec<TransportAddress>) {
    let identity = AgentIdentity::new(name);
    let addr = TransportAddress::Tcp(format!("127.0.0.1:{port}").parse().unwrap());
    (identity, vec![addr])
}

/// Test that all three routers report the correct strategy.
#[test]
fn router_strategies() {
    assert_eq!(DirectRouter::new().strategy(), RoutingStrategy::Direct);
    assert_eq!(
        BroadcastRouter::new().strategy(),
        RoutingStrategy::Broadcast
    );
    assert_eq!(
        ContentRouter::new().strategy(),
        RoutingStrategy::ContentBased
    );
}

/// Test direct routing to a peer that was discovered and added to the peer table.
#[tokio::test]
async fn direct_route_through_populated_peer_table() {
    let router = DirectRouter::new();
    let mut peers = PeerTable::new();

    let (target, target_addrs) = agent_with_tcp("code-reviewer", 9001);
    let target_id = target.id;
    let expected_addr = target_addrs[0].clone();
    peers.upsert(target, target_addrs);

    // Also add other peers to ensure direct routing is selective
    let (other, other_addrs) = agent_with_tcp("file-editor", 9002);
    peers.upsert(other, other_addrs);

    let sender_id = Uuid::new_v4();
    let envelope = MessageEnvelope::direct(sender_id, target_id, "review src/lib.rs");
    let result = router.route(&envelope, &peers).await.unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_addr);
}

/// Test broadcast routing delivers to all peers except the sender.
#[tokio::test]
async fn broadcast_excludes_sender_includes_all_others() {
    let router = BroadcastRouter::new();
    let mut peers = PeerTable::new();

    let (sender, sender_addrs) = agent_with_tcp("orchestrator", 8000);
    let sender_id = sender.id;
    peers.upsert(sender, sender_addrs);

    let (worker_a, worker_a_addrs) = agent_with_tcp("worker-a", 8001);
    let addr_a = worker_a_addrs[0].clone();
    peers.upsert(worker_a, worker_a_addrs);

    let (worker_b, worker_b_addrs) = agent_with_tcp("worker-b", 8002);
    let addr_b = worker_b_addrs[0].clone();
    peers.upsert(worker_b, worker_b_addrs);

    let envelope = MessageEnvelope::broadcast(sender_id, "shutdown-requested");
    let addrs = router.route(&envelope, &peers).await.unwrap();

    assert_eq!(addrs.len(), 2);
    assert!(addrs.contains(&addr_a));
    assert!(addrs.contains(&addr_b));
    // Sender's address should NOT be in the result
    assert!(!addrs.contains(&TransportAddress::Tcp("127.0.0.1:8000".parse().unwrap())));
}

/// Test content routing only delivers to topic subscribers.
#[tokio::test]
async fn content_router_delivers_only_to_subscribers() {
    let router = ContentRouter::new();
    let mut peers = PeerTable::new();

    let (sender, _) = agent_with_tcp("publisher", 7000);
    let sender_id = sender.id;
    peers.upsert(sender, vec![]);

    let (sub_1, sub_1_addrs) = agent_with_tcp("subscriber-1", 7001);
    let sub_1_id = sub_1.id;
    let addr_1 = sub_1_addrs[0].clone();
    peers.upsert(sub_1, sub_1_addrs);

    let (sub_2, sub_2_addrs) = agent_with_tcp("subscriber-2", 7002);
    let sub_2_id = sub_2.id;
    let addr_2 = sub_2_addrs[0].clone();
    peers.upsert(sub_2, sub_2_addrs);

    let (non_sub, non_sub_addrs) = agent_with_tcp("bystander", 7003);
    let addr_ns = non_sub_addrs[0].clone();
    peers.upsert(non_sub, non_sub_addrs);

    // Subscribe agents to topic
    peers.subscribe(sub_1_id, "build-results");
    peers.subscribe(sub_2_id, "build-results");

    let envelope = MessageEnvelope::topic(sender_id, "build-results", "build passed");
    let addrs = router.route(&envelope, &peers).await.unwrap();

    assert_eq!(addrs.len(), 2);
    assert!(addrs.contains(&addr_1));
    assert!(addrs.contains(&addr_2));
    assert!(!addrs.contains(&addr_ns));
}

/// Test that removing a peer from the peer table causes routing to fail for direct messages.
#[tokio::test]
async fn direct_route_fails_after_peer_removal() {
    let router = DirectRouter::new();
    let mut peers = PeerTable::new();

    let (target, addrs) = agent_with_tcp("ephemeral-agent", 6000);
    let target_id = target.id;
    peers.upsert(target, addrs);

    // Route should succeed
    let envelope = MessageEnvelope::direct(Uuid::new_v4(), target_id, "ping");
    assert!(router.route(&envelope, &peers).await.is_ok());

    // Remove peer
    peers.remove(&target_id);

    // Route should now fail
    let result = router.route(&envelope, &peers).await;
    assert!(result.is_err());
}

/// Test that unsubscribing from a topic removes the agent from content routing.
#[tokio::test]
async fn unsubscribe_removes_from_content_routing() {
    let router = ContentRouter::new();
    let mut peers = PeerTable::new();

    let sender_id = Uuid::new_v4();

    let (sub, sub_addrs) = agent_with_tcp("subscriber", 5000);
    let sub_id = sub.id;
    peers.upsert(sub, sub_addrs);

    peers.subscribe(sub_id, "metrics");

    let envelope = MessageEnvelope::topic(sender_id, "metrics", "cpu: 42%");

    // Should route to subscriber
    let addrs = router.route(&envelope, &peers).await.unwrap();
    assert_eq!(addrs.len(), 1);

    // Unsubscribe
    peers.unsubscribe(&sub_id, "metrics");

    // Should route to nobody
    let addrs = router.route(&envelope, &peers).await.unwrap();
    assert!(addrs.is_empty());
}

/// Test that each router correctly rejects message types it does not handle.
#[tokio::test]
async fn routers_reject_unsupported_message_types() {
    let peers = PeerTable::new();

    // DirectRouter rejects Broadcast
    let broadcast_env = MessageEnvelope::broadcast(Uuid::new_v4(), "test");
    assert!(
        DirectRouter::new()
            .route(&broadcast_env, &peers)
            .await
            .is_err()
    );

    // DirectRouter rejects Topic
    let topic_env = MessageEnvelope::topic(Uuid::new_v4(), "topic", "test");
    assert!(DirectRouter::new().route(&topic_env, &peers).await.is_err());

    // ContentRouter rejects Direct
    let direct_env =
        MessageEnvelope::direct(Uuid::new_v4(), Uuid::new_v4(), Payload::Text("test".into()));
    assert!(
        ContentRouter::new()
            .route(&direct_env, &peers)
            .await
            .is_err()
    );

    // ContentRouter rejects Broadcast
    assert!(
        ContentRouter::new()
            .route(&broadcast_env, &peers)
            .await
            .is_err()
    );
}

/// Test peer table with multiple transport addresses per peer.
#[tokio::test]
async fn peer_with_multiple_addresses() {
    let router = DirectRouter::new();
    let mut peers = PeerTable::new();

    let identity = AgentIdentity::new("multi-transport-agent");
    let id = identity.id;
    let addrs = vec![
        TransportAddress::Tcp("127.0.0.1:9000".parse().unwrap()),
        TransportAddress::Unix("/tmp/agent.sock".into()),
        TransportAddress::Url("https://agent.example.com/mcp".into()),
    ];
    peers.upsert(identity, addrs.clone());

    let envelope = MessageEnvelope::direct(Uuid::new_v4(), id, "hello");
    let result = router.route(&envelope, &peers).await.unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result, addrs);
}

/// Test that peer table upsert updates addresses for an existing peer.
#[test]
fn peer_table_upsert_updates_existing() {
    let mut peers = PeerTable::new();

    let mut identity = AgentIdentity::new("updatable-agent");
    let id = identity.id;

    // Initial insert
    let addr_v1 = vec![TransportAddress::Tcp("127.0.0.1:3000".parse().unwrap())];
    peers.upsert(identity.clone(), addr_v1);
    assert_eq!(peers.addresses(&id).unwrap().len(), 1);

    // Upsert with new address
    identity.agent_card = AgentCard {
        capabilities: vec!["updated".into()],
        ..Default::default()
    };
    let addr_v2 = vec![
        TransportAddress::Tcp("127.0.0.1:4000".parse().unwrap()),
        TransportAddress::Unix("/tmp/updated.sock".into()),
    ];
    peers.upsert(identity, addr_v2);

    assert_eq!(peers.addresses(&id).unwrap().len(), 2);
    assert_eq!(
        peers.get(&id).unwrap().agent_card.capabilities,
        vec!["updated"]
    );
    assert_eq!(peers.len(), 1, "upsert should not duplicate entries");
}
