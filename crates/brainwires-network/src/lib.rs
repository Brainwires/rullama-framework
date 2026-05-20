#![deny(missing_docs)]
//! # Brainwires Agent Network
//!
//! Agent-to-agent networking layer for the Brainwires Agent Framework.
//!
//! Provides IPC, remote bridge, mesh networking, routing, discovery,
//! and pluggable transports for agent communication.
//!
//! ## Protocol-Layer Stack
//!
//! The networking layer is organized as a 5-layer protocol stack:
//!
//! 1. **Identity** — agent identity, capability advertisement, credentials
//! 2. **Transport** — how bytes move (IPC, Remote, TCP, A2A, Pub/Sub)
//! 3. **Routing** — where messages go (direct, topology, broadcast, content)
//! 4. **Discovery** — how agents find each other (mDNS, registry, manual)
//! 5. **Application** — user-facing API (NetworkManager, events)

/// Networking transport layer — pluggable transports for agent communication.
pub mod transport;

// ============================================================================
// Agent Communication Backbone (IPC, Auth, Remote)
// ============================================================================
/// Authentication for agent network connections.
pub mod auth;
/// IPC (inter-process communication) socket protocol.
pub mod ipc;
/// Remote bridge and realtime protocol.
pub mod remote;
/// Common agent network traits.
pub mod traits;

// ============================================================================
// Client
// ============================================================================
/// Client for connecting to a remote agent network server.
#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "client")]
pub use client::{AgentConfig, AgentNetworkClient, AgentNetworkClientError};

// ============================================================================
// Mesh Networking (topology, routing, discovery, federation)
// ============================================================================
/// Distributed agent mesh networking — topology, routing, discovery, federation.
#[cfg(feature = "mesh")]
pub mod mesh;

// ============================================================================
// Protocol-Layer Stack (Identity, Network Core)
// ============================================================================
/// Peer discovery — how agents find each other on the network.
pub mod discovery;
/// Agent identity, capability advertisement, and credentials.
pub mod identity;
/// Message routing — direct, broadcast, and content-based routing.
pub mod routing;

/// Message envelopes and payload types exchanged across the network.
pub mod envelope;
/// Network lifecycle events and connection state.
pub mod event;
/// Application-layer entry point: `NetworkManager` + builder.
pub mod manager;
/// Errors emitted by the protocol-stack layers (identity / transport / routing / discovery / application).
pub mod network_error;

/// Universal messaging channels (absorbed from brainwires-channels).
pub mod channels;

/// LAN inspection tooling — NIC enumeration, IP config, ARP discovery, port scanning.
///
/// These are **operator** tools (akin to `ip`, `ifconfig`, `arp`, `nmap`), distinct
/// from the agent-discovery primitives in [`mod@discovery`]. Originally lived in
/// `brainwires-hardware::network`.
#[cfg(feature = "lan")]
pub mod lan;

pub use envelope::{MessageEnvelope, MessageTarget, Payload};
pub use event::{ConnectionState, NetworkEvent, TransportType};
pub use identity::{AgentCard, AgentIdentity, ProtocolId};
pub use manager::{NetworkManager, NetworkManagerBuilder};
pub use network_error::NetworkError;
pub use transport::{Transport, TransportAddress};

#[cfg(feature = "ipc-transport")]
pub use transport::IpcTransport;
#[cfg(feature = "pubsub-transport")]
pub use transport::PubSubTransport;
#[cfg(feature = "remote-transport")]
pub use transport::RemoteTransport;
#[cfg(feature = "tcp-transport")]
pub use transport::TcpTransport;
#[cfg(feature = "a2a-transport")]
pub use transport::{A2aTransport, a2a_message_to_envelope};
