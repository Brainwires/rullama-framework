/// Error types shared across all home automation sub-modules.
pub mod error;
/// Shared domain types: [`HomeDevice`], [`HomeAutoEvent`], [`AttributeValue`], [`Capability`].
pub mod types;

/// Zigbee coordinator — EZSP (Silicon Labs) and ZNP (TI Z-Stack) backends.
#[cfg(feature = "zigbee")]
pub mod zigbee;

/// Z-Wave controller — direct Z-Wave Serial API (ZAPI) over USB stick.
#[cfg(feature = "zwave")]
pub mod zwave;

/// Thread — OpenThread Border Router (OTBR) REST API client.
#[cfg(feature = "thread")]
pub mod thread;

/// Matter — controller (commission + cluster client) and device server.
#[cfg(feature = "matter")]
pub mod matter;

// ── Flat re-exports ──────────────────────────────────────────────────────────

pub use error::{HomeAutoError, HomeAutoResult};
pub use types::{AttributeValue, Capability, HomeAutoEvent, HomeDevice, Protocol};

#[cfg(feature = "zigbee")]
pub use zigbee::ezsp::EzspCoordinator;
#[cfg(feature = "zigbee")]
pub use zigbee::znp::ZnpCoordinator;
#[cfg(feature = "zigbee")]
pub use zigbee::{
    ZigbeeAddr, ZigbeeAttrId, ZigbeeClusterId, ZigbeeCoordinator, ZigbeeDevice, ZigbeeDeviceKind,
};

#[cfg(feature = "zwave")]
pub use zwave::serial_api::ZWaveSerialController;
#[cfg(feature = "zwave")]
pub use zwave::{CommandClass, NodeId, ZWaveController, ZWaveNode, ZWaveNodeKind};

#[cfg(feature = "thread")]
pub use thread::border_router::ThreadBorderRouter;
#[cfg(feature = "thread")]
pub use thread::types::{ThreadNeighbor, ThreadNetworkDataset, ThreadNodeInfo};

#[cfg(feature = "matter")]
pub use matter::controller::MatterController;
#[cfg(feature = "matter")]
pub use matter::server::MatterDeviceServer;
#[cfg(feature = "matter")]
pub use matter::types::{MatterDevice, MatterDeviceConfig, MatterEndpoint};

/// `BoxStream` alias used by all event-stream methods.
pub type BoxStream<'a, T> = std::pin::Pin<Box<dyn futures::Stream<Item = T> + Send + 'a>>;
