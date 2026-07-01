//! Zigbee coordinator — two interchangeable USB-stick backends.
//!
//! Implements Zigbee 3.0 / ZCL over either Silicon Labs EZSP (EFR32 chipsets,
//! ASH framing over UART) or TI Z-Stack 3.x ZNP (CC2652/CC1352 chipsets, MT
//! API). The `ZigbeeCoordinator` trait unifies both.

/// High-level helpers + constants for common ZCL clusters (On/Off, Level,
/// Color, Temp, Humidity, IAS Zone, Door Lock).
pub mod clusters;
/// Typed records: `ZigbeeAddr`, `ZigbeeDevice`, `ZigbeeDeviceKind`, cluster/attr IDs.
pub mod types;

/// Silicon Labs EZSP v8 coordinator backend.
pub mod ezsp;
/// TI Z-Stack 3.x ZNP coordinator backend.
pub mod znp;

use async_trait::async_trait;

use super::BoxStream;
use super::error::HomeAutoResult;
use super::types::{AttributeValue, HomeAutoEvent};
pub use ezsp::EzspCoordinator;
pub use types::{
    IeeeAddr, NwkAddr, ZigbeeAddr, ZigbeeAttrId, ZigbeeClusterId, ZigbeeDevice, ZigbeeDeviceKind,
    cluster_id,
};
pub use znp::ZnpCoordinator;

/// Abstraction over a Zigbee network coordinator.
///
/// Implemented by [`EzspCoordinator`] (Silicon Labs) and [`ZnpCoordinator`] (TI Z-Stack).
#[async_trait]
pub trait ZigbeeCoordinator: Send + Sync {
    /// Open the serial port and initialise the coordinator.
    async fn start(&self) -> HomeAutoResult<()>;

    /// Close the serial port and shut down.
    async fn stop(&self) -> HomeAutoResult<()>;

    /// Open the join window for `duration_secs` seconds (0 = close, 0xFF = forever).
    async fn permit_join(&self, duration_secs: u8) -> HomeAutoResult<()>;

    /// Return a snapshot of all known devices on the network.
    async fn devices(&self) -> HomeAutoResult<Vec<ZigbeeDevice>>;

    /// Read a ZCL attribute. The request is sent synchronously; the attribute value
    /// arrives asynchronously through [`events`] as an [`HomeAutoEvent::AttributeChanged`].
    /// Returns `AttributeValue::Null` immediately after confirming the send succeeded.
    async fn read_attribute(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        attr: ZigbeeAttrId,
    ) -> HomeAutoResult<AttributeValue>;

    /// Write a ZCL attribute value.
    async fn write_attribute(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        attr: ZigbeeAttrId,
        value: AttributeValue,
    ) -> HomeAutoResult<()>;

    /// Send a ZCL cluster-specific command.
    async fn invoke_command(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        cmd: u8,
        payload: &[u8],
    ) -> HomeAutoResult<()>;

    /// Subscribe to a stream of events from this coordinator.
    /// Events include device join/leave and incoming ZCL attribute reports.
    fn events(&self) -> BoxStream<'static, HomeAutoEvent>;
}
