//! # Transport Layer
//!
//! The transport layer defines how bytes move between agents. Each
//! networking paradigm (IPC, Remote Bridge, TCP, A2A, Pub/Sub) is
//! implemented as a [`Transport`] that can send and receive
//! [`MessageEnvelope`](crate::MessageEnvelope)s.
//!
//! ## Provided transports
//!
//! | Transport | Feature flag | Description |
//! |-----------|-------------|-------------|
//! | `IpcTransport` | `ipc-transport` | Local Unix-socket IPC with ChaCha20 encryption |
//! | `RemoteTransport` | `remote-transport` | Supabase Realtime / HTTP polling bridge |
//! | `TcpTransport` | `tcp-transport` | Direct TCP peer-to-peer connections |

mod traits;

#[cfg(feature = "a2a-transport")]
mod a2a_transport;
#[cfg(feature = "ipc-transport")]
mod ipc_transport;
#[cfg(feature = "pubsub-transport")]
mod pubsub_transport;
#[cfg(feature = "remote-transport")]
mod remote_transport;
#[cfg(feature = "tcp-transport")]
mod tcp_transport;

pub use traits::{Transport, TransportAddress};

#[cfg(feature = "a2a-transport")]
pub use a2a_transport::{A2aTransport, a2a_message_to_envelope};
#[cfg(feature = "ipc-transport")]
pub use ipc_transport::IpcTransport;
#[cfg(feature = "pubsub-transport")]
pub use pubsub_transport::PubSubTransport;
#[cfg(feature = "remote-transport")]
pub use remote_transport::RemoteTransport;
#[cfg(feature = "tcp-transport")]
pub use tcp_transport::TcpTransport;
