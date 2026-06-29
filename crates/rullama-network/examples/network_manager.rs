//! Network Manager — agent registration, messaging, and event handling.
//!
//! Demonstrates:
//! - Building a `NetworkManager` with `NetworkManagerBuilder`
//! - Registering agents via `AgentIdentity` and `AgentCard`
//! - Peer discovery with `ManualDiscovery`
//! - Sending `MessageEnvelope` (direct, broadcast, topic)
//! - Subscribing to `NetworkEvent` and inspecting `ConnectionState`
//!
//! ```bash
//! cargo run -p rullama-network --example network_manager \
//!     --features "server,client,ipc-transport"
//! ```

use rullama_network::discovery::ManualDiscovery;
use rullama_network::{
    AgentCard, AgentIdentity, ConnectionState, MessageEnvelope, NetworkEvent,
    NetworkManagerBuilder, Payload, TransportType,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Network Manager Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create agent identities with capability cards
    // -----------------------------------------------------------------------
    println!("--- Agent Identities ---");

    let mut orchestrator = AgentIdentity::new("orchestrator");
    orchestrator.agent_card = AgentCard {
        capabilities: vec!["task-routing".into(), "load-balancing".into()],
        supported_protocols: vec!["mcp".into(), "ipc".into()],
        endpoint: Some("unix:///tmp/orchestrator.sock".into()),
        max_concurrent_tasks: Some(50),
        compute_capacity: Some(1.0),
        ..Default::default()
    };
    println!(
        "  {} (id={})\n    capabilities: {:?}\n    protocols:    {:?}\n    endpoint:     {:?}",
        orchestrator.name,
        orchestrator.id,
        orchestrator.agent_card.capabilities,
        orchestrator.agent_card.supported_protocols,
        orchestrator.agent_card.endpoint,
    );

    let mut worker_a = AgentIdentity::new("worker-alpha");
    worker_a.agent_card = AgentCard {
        capabilities: vec!["code-generation".into()],
        supported_protocols: vec!["mcp".into(), "ipc".into()],
        endpoint: Some("unix:///tmp/worker-a.sock".into()),
        max_concurrent_tasks: Some(10),
        compute_capacity: Some(0.6),
        ..Default::default()
    };
    println!(
        "  {} (id={})\n    capabilities: {:?}",
        worker_a.name, worker_a.id, worker_a.agent_card.capabilities,
    );

    let mut worker_b = AgentIdentity::new("worker-beta");
    worker_b.agent_card = AgentCard {
        capabilities: vec!["code-review".into(), "testing".into()],
        supported_protocols: vec!["mcp".into(), "a2a".into()],
        endpoint: Some("tcp://127.0.0.1:9091".into()),
        max_concurrent_tasks: Some(5),
        compute_capacity: Some(0.8),
        ..Default::default()
    };
    println!(
        "  {} (id={})\n    capabilities: {:?}",
        worker_b.name, worker_b.id, worker_b.agent_card.capabilities,
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. Build the NetworkManager with manual discovery
    // -----------------------------------------------------------------------
    println!("--- Build NetworkManager ---");

    let discovery = ManualDiscovery::with_peers(vec![worker_a.clone(), worker_b.clone()]);

    let manager = NetworkManagerBuilder::new(orchestrator.clone())
        .add_discovery(Box::new(discovery))
        .event_buffer(128)
        .build();

    println!(
        "  Manager identity: {} (id={})",
        manager.identity().name,
        manager.identity().id
    );
    println!();

    // -----------------------------------------------------------------------
    // 3. Subscribe to network events
    // -----------------------------------------------------------------------
    println!("--- Subscribe to Events ---");
    let mut events = manager.subscribe();
    println!("  Event subscriber created");
    println!();

    // -----------------------------------------------------------------------
    // 4. Discover peers
    // -----------------------------------------------------------------------
    println!("--- Peer Discovery ---");

    let found = manager.discover_peers().await?;
    println!("  Discovered {} peer(s):", found.len());

    let peers = manager.peers().await;
    for peer in &peers {
        println!(
            "    {} — protocols: {:?}, endpoint: {:?}",
            peer.name, peer.agent_card.supported_protocols, peer.agent_card.endpoint
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. Drain PeerJoined events
    // -----------------------------------------------------------------------
    println!("--- Network Events ---");

    while let Ok(event) = events.try_recv() {
        match &event {
            NetworkEvent::PeerJoined(peer) => {
                println!("  Event: PeerJoined — {}", peer.name);
            }
            NetworkEvent::PeerLeft(id) => {
                println!("  Event: PeerLeft — {id}");
            }
            NetworkEvent::MessageReceived(env) => {
                println!("  Event: MessageReceived — from {}", env.sender);
            }
            NetworkEvent::ConnectionStateChanged { transport, state } => {
                println!("  Event: ConnectionStateChanged — {transport:?} -> {state:?}");
            }
            NetworkEvent::Error(e) => {
                println!("  Event: Error — {e:?}");
            }
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Demonstrate message envelope construction
    // -----------------------------------------------------------------------
    println!("--- Message Envelopes ---");

    // Direct message to worker-alpha
    let direct = MessageEnvelope::direct(
        orchestrator.id,
        worker_a.id,
        Payload::Text("Please generate a Rust HTTP handler".into()),
    );
    println!(
        "  Direct:    sender={}, recipient={:?}, payload=Text(\"...\")",
        direct.sender, direct.recipient,
    );

    // Broadcast to all peers
    let broadcast = MessageEnvelope::broadcast(
        orchestrator.id,
        Payload::Text("System: reloading config".into()),
    );
    println!(
        "  Broadcast: sender={}, recipient={:?}",
        broadcast.sender, broadcast.recipient,
    );

    // Topic-addressed message
    let topic = MessageEnvelope::topic(
        orchestrator.id,
        "build-events",
        serde_json::json!({ "status": "success", "duration_ms": 1234 }),
    );
    println!(
        "  Topic:     sender={}, recipient={:?}",
        topic.sender, topic.recipient,
    );

    // Reply construction
    let reply = direct.reply(worker_a.id, Payload::Text("Handler generated!".into()));
    println!(
        "  Reply:     sender={}, correlation_id={:?}",
        reply.sender, reply.correlation_id,
    );

    // TTL-limited message
    let ttl_msg = MessageEnvelope::broadcast(orchestrator.id, "heartbeat").with_ttl(3);
    println!("  TTL msg:   ttl={:?}", ttl_msg.ttl);
    println!();

    // -----------------------------------------------------------------------
    // 7. Inspect connection state and transport type enums
    // -----------------------------------------------------------------------
    println!("--- Connection States & Transport Types ---");

    let states = [
        ConnectionState::Disconnected,
        ConnectionState::Connecting,
        ConnectionState::Connected,
        ConnectionState::Reconnecting,
    ];
    for state in &states {
        println!("  ConnectionState: {state:?}");
    }

    let transports = [
        TransportType::Ipc,
        TransportType::Remote,
        TransportType::Tcp,
        TransportType::A2a,
        TransportType::PubSub,
        TransportType::Custom("nats".into()),
    ];
    for t in &transports {
        println!("  TransportType:   {t:?}");
    }
    println!();

    // -----------------------------------------------------------------------
    // 8. Emit a synthetic event
    // -----------------------------------------------------------------------
    println!("--- Emit Synthetic Event ---");

    manager.emit(NetworkEvent::ConnectionStateChanged {
        transport: TransportType::Ipc,
        state: ConnectionState::Connected,
    });

    if let Ok(event) = events.try_recv() {
        println!("  Received: {event:?}");
    }

    println!("\nDone.");
    Ok(())
}
