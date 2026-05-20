//! Integration tests for the NetworkManager — verifying the builder pattern,
//! event subscription, peer table management, and discovery integration.

use brainwires_network::discovery::{Discovery, ManualDiscovery};
use brainwires_network::identity::AgentIdentity;
use brainwires_network::transport::TransportAddress;
use brainwires_network::{
    ConnectionState, NetworkError, NetworkEvent, NetworkManagerBuilder, TransportType,
};

/// Test that NetworkManagerBuilder creates a properly configured manager.
#[tokio::test]
async fn builder_configures_manager_correctly() {
    let identity = AgentIdentity::new("test-manager");
    let discovery = ManualDiscovery::new();

    let manager = NetworkManagerBuilder::new(identity.clone())
        .add_discovery(Box::new(discovery))
        .event_buffer(128)
        .build();

    assert_eq!(manager.identity().id, identity.id);
    assert_eq!(manager.identity().name, "test-manager");
    assert!(manager.peers().await.is_empty());
}

/// Test register_self and deregister_self with discovery.
#[tokio::test]
async fn register_and_deregister_through_manager() {
    let identity = AgentIdentity::new("registering-agent");
    let discovery = ManualDiscovery::new();

    let manager = NetworkManagerBuilder::new(identity.clone())
        .add_discovery(Box::new(discovery.clone()))
        .build();

    // Register
    manager.register_self().await.unwrap();
    let found = discovery.discover().await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].id, identity.id);

    // Deregister
    manager.deregister_self().await.unwrap();
    let found = discovery.discover().await.unwrap();
    assert!(found.is_empty());
}

/// Test that discover_peers deduplicates across multiple discovery sources.
#[tokio::test]
async fn discover_peers_deduplicates() {
    let my_identity = AgentIdentity::new("dedup-tester");

    let mut shared_agent = AgentIdentity::new("shared-peer");
    shared_agent.agent_card.endpoint = Some("tcp://10.0.0.1:9000".into());

    // Both discovery services know about the same agent
    let discovery_1 = ManualDiscovery::with_peers(vec![shared_agent.clone()]);
    let discovery_2 = ManualDiscovery::with_peers(vec![shared_agent.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery_1))
        .add_discovery(Box::new(discovery_2))
        .build();

    manager.discover_peers().await.unwrap();

    // Peer table should have exactly one entry, not two
    let peers = manager.peers().await;
    assert_eq!(peers.len(), 1);
}

/// Test that repeated discovery does not re-emit PeerJoined for known peers.
#[tokio::test]
async fn repeated_discovery_does_not_duplicate_events() {
    let my_identity = AgentIdentity::new("event-tester");
    let mut peer = AgentIdentity::new("stable-peer");
    peer.agent_card.endpoint = Some("tcp://10.0.0.5:9000".into());

    let discovery = ManualDiscovery::with_peers(vec![peer.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery))
        .build();

    let mut events = manager.subscribe();

    // First discovery
    manager.discover_peers().await.unwrap();
    assert!(events.try_recv().is_ok()); // PeerJoined

    // Second discovery — peer already known
    manager.discover_peers().await.unwrap();
    assert!(
        events.try_recv().is_err(),
        "should not emit PeerJoined for already-known peer"
    );
}

/// Test emitting custom network events.
#[tokio::test]
async fn emit_custom_events() {
    let identity = AgentIdentity::new("event-emitter");
    let manager = NetworkManagerBuilder::new(identity).build();

    let mut rx = manager.subscribe();

    // Emit connection state change
    manager.emit(NetworkEvent::ConnectionStateChanged {
        transport: TransportType::Tcp,
        state: ConnectionState::Connected,
    });

    let event = rx.try_recv().unwrap();
    match event {
        NetworkEvent::ConnectionStateChanged { transport, state } => {
            assert_eq!(transport, TransportType::Tcp);
            assert_eq!(state, ConnectionState::Connected);
        }
        other => panic!("expected ConnectionStateChanged, got {other:?}"),
    }

    // Emit error event
    manager.emit(NetworkEvent::Error(NetworkError::Timeout(
        "discovery timed out after 30s".into(),
    )));

    let event = rx.try_recv().unwrap();
    match event {
        NetworkEvent::Error(err) => {
            assert_eq!(err.to_string(), "timeout: discovery timed out after 30s");
        }
        other => panic!("expected Error, got {other:?}"),
    }
}

/// Test peer table read/write access through the manager.
#[tokio::test]
async fn peer_table_access_through_manager() {
    let my_identity = AgentIdentity::new("table-accessor");
    let manager = NetworkManagerBuilder::new(my_identity).build();

    // Write to peer table
    {
        let mut table = manager.peer_table_mut().await;
        let peer = AgentIdentity::new("manual-peer");
        let addr = TransportAddress::Tcp("127.0.0.1:5000".parse().unwrap());
        table.upsert(peer, vec![addr]);
    }

    // Read from peer table
    {
        let table = manager.peer_table().await;
        assert_eq!(table.len(), 1);
    }

    // Also accessible via peers()
    let peers = manager.peers().await;
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].name, "manual-peer");
}

/// Test that peers discovered with endpoints get correct transport addresses.
#[tokio::test]
async fn discovered_endpoints_resolve_to_transport_addresses() {
    let my_identity = AgentIdentity::new("resolver");

    let mut tcp_peer = AgentIdentity::new("tcp-peer");
    tcp_peer.agent_card.endpoint = Some("tcp://10.0.0.1:8080".into());

    let mut unix_peer = AgentIdentity::new("unix-peer");
    unix_peer.agent_card.endpoint = Some("unix:///var/run/agent.sock".into());

    let mut http_peer = AgentIdentity::new("http-peer");
    http_peer.agent_card.endpoint = Some("https://agent.cloud.example.com/mcp".into());

    let discovery =
        ManualDiscovery::with_peers(vec![tcp_peer.clone(), unix_peer.clone(), http_peer.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery))
        .build();

    manager.discover_peers().await.unwrap();

    let table = manager.peer_table().await;

    // TCP peer should have a Tcp address
    let tcp_addrs = table.addresses(&tcp_peer.id).unwrap();
    assert!(matches!(&tcp_addrs[0], TransportAddress::Tcp(addr) if addr.port() == 8080));

    // Unix peer should have a Unix address
    let unix_addrs = table.addresses(&unix_peer.id).unwrap();
    assert!(
        matches!(&unix_addrs[0], TransportAddress::Unix(path) if path.to_str().unwrap().contains("agent.sock"))
    );

    // HTTP peer should have a Url address
    let http_addrs = table.addresses(&http_peer.id).unwrap();
    assert!(
        matches!(&http_addrs[0], TransportAddress::Url(url) if url.contains("agent.cloud.example.com"))
    );
}

/// Test that peers without endpoints get empty address lists.
#[tokio::test]
async fn peers_without_endpoints_have_no_addresses() {
    let my_identity = AgentIdentity::new("no-endpoint-tester");

    let peer = AgentIdentity::new("no-endpoint-agent");
    let peer_id = peer.id;

    let discovery = ManualDiscovery::with_peers(vec![peer]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery))
        .build();

    manager.discover_peers().await.unwrap();

    let table = manager.peer_table().await;
    let addrs = table.addresses(&peer_id).unwrap();
    assert!(addrs.is_empty());
}
