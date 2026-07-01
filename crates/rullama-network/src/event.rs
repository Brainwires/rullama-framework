use serde::{Deserialize, Serialize};

use crate::envelope::MessageEnvelope;
use crate::identity::AgentIdentity;
use crate::network_error::NetworkError;

/// The connection state of a transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Transport is disconnected.
    Disconnected,
    /// Transport is attempting to connect.
    Connecting,
    /// Transport is connected and ready.
    Connected,
    /// Transport is reconnecting after a failure.
    Reconnecting,
}

/// A transport type identifier for events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportType {
    /// Local Unix socket IPC.
    Ipc,
    /// Remote bridge (Supabase Realtime / HTTP polling).
    Remote,
    /// Direct TCP peer-to-peer.
    Tcp,
    /// A2A protocol transport.
    A2a,
    /// Pub/Sub event-driven transport.
    PubSub,
    /// Custom transport with a user-defined identifier.
    Custom(String),
}

/// Events emitted by the networking stack.
///
/// Subscribe to these via `NetworkManager::subscribe()` to react to
/// peer changes, incoming messages, and connection state transitions.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// A new peer was discovered or joined the network.
    PeerJoined(AgentIdentity),
    /// A peer left the network or became unreachable.
    PeerLeft(uuid::Uuid),
    /// A message was received from the network.
    MessageReceived(MessageEnvelope),
    /// A transport's connection state changed.
    ConnectionStateChanged {
        /// Which transport changed.
        transport: TransportType,
        /// The new state.
        state: ConnectionState,
    },
    /// A network-level error occurred.
    Error(NetworkError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn connection_state_equality() {
        assert_eq!(ConnectionState::Connected, ConnectionState::Connected);
        assert_ne!(ConnectionState::Connected, ConnectionState::Disconnected);
    }

    #[test]
    fn transport_type_equality() {
        assert_eq!(TransportType::Ipc, TransportType::Ipc);
        assert_ne!(TransportType::Ipc, TransportType::Tcp);
        assert_eq!(
            TransportType::Custom("nats".into()),
            TransportType::Custom("nats".into())
        );
    }

    #[test]
    fn transport_type_serde_roundtrip() {
        let types = vec![
            TransportType::Ipc,
            TransportType::Remote,
            TransportType::Tcp,
            TransportType::A2a,
            TransportType::PubSub,
            TransportType::Custom("redis".into()),
        ];
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let deserialized: TransportType = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, t);
        }
    }

    #[test]
    fn network_event_variants_constructible() {
        let identity = AgentIdentity::new("test");
        let _ = NetworkEvent::PeerJoined(identity);
        let _ = NetworkEvent::PeerLeft(Uuid::new_v4());
        let _ = NetworkEvent::ConnectionStateChanged {
            transport: TransportType::Tcp,
            state: ConnectionState::Connected,
        };
        let _ = NetworkEvent::Error(NetworkError::Timeout("5s".into()));
    }
}
