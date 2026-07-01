pub mod attachments;
// LEGACY (disabled by default): the Studio remote-bridge let the user reach CLI
// agents over the web via a cloud relay. The hosted relay is discontinued, but
// the code is kept behind the off-by-default `remote-bridge` feature because a
// similar capability is likely to return. Enable with `--features remote-bridge`.
#[cfg(feature = "remote-bridge")]
pub mod bridge;
pub mod command_queue;
pub mod heartbeat;
#[cfg(feature = "remote-bridge")]
pub mod manager;
pub mod permission_relay;
pub mod protocol;
pub mod realtime;
pub mod telemetry;

pub use command_queue::{CommandQueue, QueueEntry, QueueError, QueueStats};
pub use permission_relay::{PermissionDecision, PermissionRelay};
pub use protocol::{
    AgentEventType, BackendCommand, CommandPriority, CompressionAlgorithm, DeviceStatus,
    NegotiatedProtocol, OrgPolicies, PrioritizedCommand, ProtocolAccept, ProtocolCapability,
    ProtocolHello, RemoteAgentInfo, RemoteMessage, RetryPolicy, StreamChunkType,
    compute_device_fingerprint,
};
pub use telemetry::{ConnectionQuality, MetricsSnapshot, ProtocolMetrics};

pub use attachments::AttachmentReceiver;
#[cfg(feature = "remote-bridge")]
pub use bridge::{BridgeConfig, BridgeState, ConnectionMode, RealtimeCredentials, RemoteBridge};
pub use heartbeat::{AgentEvent, HeartbeatCollector, HeartbeatData};
#[cfg(feature = "remote-bridge")]
pub use manager::{RemoteBridgeManager, RemoteBridgeStatus};
pub use realtime::{RealtimeClient, RealtimeConfig, RealtimeState};
