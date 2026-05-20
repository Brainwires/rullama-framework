//! Integration tests for agent identity and discovery — verifying that
//! identities flow correctly through the discovery layer and interact
//! with routing and network events.

use brainwires_network::discovery::{Discovery, DiscoveryProtocol, ManualDiscovery};
use brainwires_network::identity::{AgentCard, AgentIdentity, SigningKey, VerifyingKey};
use brainwires_network::routing::PeerTable;
use brainwires_network::transport::TransportAddress;
use brainwires_network::{NetworkEvent, NetworkManagerBuilder};

/// Test that agents discovered via ManualDiscovery can be looked up
/// and have their capabilities inspected.
#[tokio::test]
async fn discover_agents_and_inspect_capabilities() {
    let mut agent_a = AgentIdentity::new("code-reviewer");
    agent_a.agent_card = AgentCard {
        capabilities: vec!["code-review".into(), "linting".into()],
        supported_protocols: vec!["mcp".into()],
        endpoint: Some("tcp://192.168.1.10:9000".into()),
        max_concurrent_tasks: Some(5),
        compute_capacity: Some(0.8),
        ..Default::default()
    };

    let mut agent_b = AgentIdentity::new("file-editor");
    agent_b.agent_card = AgentCard {
        capabilities: vec!["file-editing".into(), "code-generation".into()],
        supported_protocols: vec!["mcp".into(), "a2a".into()],
        endpoint: Some("tcp://192.168.1.11:9000".into()),
        max_concurrent_tasks: Some(3),
        ..Default::default()
    };

    let discovery = ManualDiscovery::with_peers(vec![agent_a.clone(), agent_b.clone()]);

    // Discover all peers
    let peers = discovery.discover().await.unwrap();
    assert_eq!(peers.len(), 2);

    // Look up specific agent by UUID
    let found = discovery.lookup(&agent_a.id).await.unwrap().unwrap();
    assert_eq!(found.name, "code-reviewer");
    assert!(found.agent_card.has_capability("code-review"));
    assert!(found.agent_card.supports_protocol("MCP"));
    assert!(!found.agent_card.supports_protocol("a2a"));

    // Check agent_b capabilities
    let found_b = discovery.lookup(&agent_b.id).await.unwrap().unwrap();
    assert!(found_b.agent_card.has_capability("file-editing"));
    assert!(found_b.agent_card.supports_protocol("a2a"));
}

/// Test that deregistering an agent makes it undiscoverable.
#[tokio::test]
async fn deregister_makes_agent_undiscoverable() {
    let discovery = ManualDiscovery::new();
    let agent = AgentIdentity::new("temporary-worker");
    let id = agent.id;

    discovery.register(&agent).await.unwrap();
    assert!(discovery.lookup(&id).await.unwrap().is_some());

    discovery.deregister(&id).await.unwrap();
    assert!(discovery.lookup(&id).await.unwrap().is_none());
    assert!(discovery.discover().await.unwrap().is_empty());
}

/// Test that discovered peers flow into the peer table via NetworkManager.
#[tokio::test]
async fn discovered_peers_populate_peer_table_and_emit_events() {
    let my_identity = AgentIdentity::new("orchestrator");

    let mut worker = AgentIdentity::new("worker-1");
    worker.agent_card.endpoint = Some("tcp://127.0.0.1:9090".into());

    let discovery = ManualDiscovery::with_peers(vec![worker.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery))
        .build();

    let mut events = manager.subscribe();

    // Trigger discovery
    let found = manager.discover_peers().await.unwrap();
    assert_eq!(found.len(), 1);

    // Check peer table was populated
    let peers = manager.peers().await;
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].name, "worker-1");

    // Verify PeerJoined event was emitted
    let event = events.try_recv().unwrap();
    match event {
        NetworkEvent::PeerJoined(peer) => {
            assert_eq!(peer.id, worker.id);
            assert_eq!(peer.name, "worker-1");
        }
        other => panic!("expected PeerJoined, got {other:?}"),
    }
}

/// Test that NetworkManager does not add itself to the peer table.
#[tokio::test]
async fn network_manager_excludes_self_from_peer_table() {
    let my_identity = AgentIdentity::new("self-agent");

    // Discovery service that returns self
    let discovery = ManualDiscovery::with_peers(vec![my_identity.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery))
        .build();

    manager.discover_peers().await.unwrap();

    let peers = manager.peers().await;
    assert!(peers.is_empty(), "should not add self to peer table");
}

/// Test sign-and-verify round-trip for agent credentials.
#[test]
fn credential_sign_verify_roundtrip() {
    let shared_secret = "agent-network-session-key-2024";
    let signer = SigningKey::from_secret(shared_secret);
    let verifier = VerifyingKey::from_secret(shared_secret);

    // Sign an agent identity as JSON
    let identity = AgentIdentity::new("authenticated-agent");
    let identity_json = serde_json::to_vec(&identity).unwrap();

    let signed = signer.sign(&identity_json).unwrap();
    let recovered = verifier.verify(&signed).unwrap();

    let deserialized: AgentIdentity = serde_json::from_slice(&recovered).unwrap();
    assert_eq!(deserialized.id, identity.id);
    assert_eq!(deserialized.name, "authenticated-agent");
}

/// Test that credentials from different secrets fail verification.
#[test]
fn credentials_from_different_secrets_reject() {
    let signer = SigningKey::from_secret("cluster-alpha");
    let verifier = VerifyingKey::from_secret("cluster-beta");

    let message = b"cross-cluster message";
    let signed = signer.sign(message).unwrap();

    assert!(
        verifier.verify(&signed).is_err(),
        "different secrets should fail verification"
    );
}

/// Test identity serialization preserves all agent card fields.
#[test]
fn identity_serde_preserves_all_card_fields() {
    let mut identity = AgentIdentity::new("full-card-agent");
    identity.agent_card = AgentCard {
        capabilities: vec!["search".into(), "code-gen".into()],
        supported_protocols: vec!["mcp".into(), "ipc".into()],
        metadata: [
            ("model".to_string(), serde_json::json!("claude-3.5-sonnet")),
            ("version".to_string(), serde_json::json!("1.0.0")),
        ]
        .into_iter()
        .collect(),
        endpoint: Some("tcp://10.0.0.5:9090".into()),
        max_concurrent_tasks: Some(10),
        compute_capacity: Some(0.95),
    };

    let json = serde_json::to_string(&identity).unwrap();
    let restored: AgentIdentity = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.id, identity.id);
    assert_eq!(restored.name, "full-card-agent");
    assert_eq!(restored.agent_card.capabilities.len(), 2);
    assert_eq!(restored.agent_card.supported_protocols.len(), 2);
    assert_eq!(restored.agent_card.metadata.len(), 2);
    assert_eq!(
        restored.agent_card.metadata["model"],
        serde_json::json!("claude-3.5-sonnet")
    );
    assert_eq!(
        restored.agent_card.endpoint.as_deref(),
        Some("tcp://10.0.0.5:9090")
    );
    assert_eq!(restored.agent_card.max_concurrent_tasks, Some(10));
    assert_eq!(restored.agent_card.compute_capacity, Some(0.95));
}

/// Test that peer table integrates with discovery for building a complete network view.
#[tokio::test]
async fn peer_table_from_discovery_results() {
    let discovery = ManualDiscovery::new();

    // Register agents with different transport endpoints
    let mut tcp_agent = AgentIdentity::new("tcp-agent");
    tcp_agent.agent_card.endpoint = Some("tcp://192.168.1.100:9000".into());
    tcp_agent.agent_card.capabilities.push("compute".into());

    let mut ipc_agent = AgentIdentity::new("ipc-agent");
    ipc_agent.agent_card.endpoint = Some("unix:///tmp/ipc-agent.sock".into());
    ipc_agent.agent_card.capabilities.push("file-ops".into());

    discovery.register(&tcp_agent).await.unwrap();
    discovery.register(&ipc_agent).await.unwrap();

    // Build peer table from discovery results
    let peers_discovered = discovery.discover().await.unwrap();
    let mut table = PeerTable::new();

    for peer in &peers_discovered {
        let addrs = match &peer.agent_card.endpoint {
            Some(ep) if ep.starts_with("tcp://") => {
                let addr_str = ep.strip_prefix("tcp://").unwrap();
                vec![TransportAddress::Tcp(addr_str.parse().unwrap())]
            }
            Some(ep) if ep.starts_with("unix://") => {
                let path = ep.strip_prefix("unix://").unwrap();
                vec![TransportAddress::Unix(path.into())]
            }
            _ => vec![],
        };
        table.upsert(peer.clone(), addrs);
    }

    assert_eq!(table.len(), 2);

    // Verify TCP agent's address
    let tcp_addrs = table.addresses(&tcp_agent.id).unwrap();
    assert_eq!(tcp_addrs.len(), 1);
    assert!(matches!(&tcp_addrs[0], TransportAddress::Tcp(_)));

    // Verify IPC agent's address
    let ipc_addrs = table.addresses(&ipc_agent.id).unwrap();
    assert_eq!(ipc_addrs.len(), 1);
    assert!(matches!(&ipc_addrs[0], TransportAddress::Unix(_)));
}

/// Test multiple discovery services in NetworkManager.
#[tokio::test]
async fn multiple_discovery_services() {
    let my_identity = AgentIdentity::new("coordinator");

    let mut agent_a = AgentIdentity::new("service-a");
    agent_a.agent_card.endpoint = Some("tcp://10.0.0.1:9000".into());

    let mut agent_b = AgentIdentity::new("service-b");
    agent_b.agent_card.endpoint = Some("tcp://10.0.0.2:9000".into());

    let discovery_1 = ManualDiscovery::with_peers(vec![agent_a.clone()]);
    let discovery_2 = ManualDiscovery::with_peers(vec![agent_b.clone()]);

    let manager = NetworkManagerBuilder::new(my_identity)
        .add_discovery(Box::new(discovery_1))
        .add_discovery(Box::new(discovery_2))
        .build();

    let found = manager.discover_peers().await.unwrap();
    assert_eq!(found.len(), 2);

    let peers = manager.peers().await;
    assert_eq!(peers.len(), 2);
}

/// Test ManualDiscovery protocol identification.
#[test]
fn manual_discovery_reports_correct_protocol() {
    let discovery = ManualDiscovery::new();
    assert_eq!(discovery.protocol(), DiscoveryProtocol::Manual);
}
