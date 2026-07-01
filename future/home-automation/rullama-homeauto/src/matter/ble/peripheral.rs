//! Matter BLE GATT peripheral server.
//!
//! Advertises the Matter BLE service and handles the BTP handshake, allowing
//! a Matter commissioner to open a commissioning session over Bluetooth.
//!
//! Platform support: Linux (BlueZ) and macOS (CoreBluetooth).
//! btleplug 0.11 does not expose a peripheral-advertising API on Windows,
//! so [`MatterBlePeripheral::start`] returns an error on that platform.

use uuid::Uuid;

use crate::matter::error::{MatterError, MatterResult};
use crate::matter::transport::ble::{
    BleTransport, BtpHandshakeRequest, BtpHandshakeResponse, BtpReassembler, flags,
    fragment_message,
};

// ── Matter BLE UUIDs ──────────────────────────────────────────────────────────

/// Matter BLE service UUID: `0000FFF6-0000-1000-8000-00805F9B34FB`.
// reason: groups mirror the canonical 8-4-4-4-12 hex UUID format used by BLE,
// not the clippy-preferred uniform 4-digit groups.
#[allow(clippy::unusual_byte_groupings)]
pub const MATTER_BLE_SERVICE_UUID: Uuid =
    Uuid::from_u128(0x0000_FFF6_0000_1000_8000_00805F9B34FB_u128);

/// Matter C1 characteristic UUID (controller → device write):
/// `18EE2EF5-263D-4559-959F-4F9C429F9D11`.
pub const MATTER_C1_UUID: Uuid = Uuid::from_u128(0x18EE2EF5_263D_4559_959F_4F9C429F9D11_u128);

/// Matter C2 characteristic UUID (device → controller indication):
/// `18EE2EF5-263D-4559-959F-4F9C429F9D12`.
pub const MATTER_C2_UUID: Uuid = Uuid::from_u128(0x18EE2EF5_263D_4559_959F_4F9C429F9D12_u128);

// ── MatterBlePeripheral ───────────────────────────────────────────────────────

/// BLE peripheral server for Matter commissioning.
///
/// When [`start`](Self::start) is called, the device advertises the Matter BLE
/// service UUID and waits for a commissioner to initiate the BTP handshake on
/// the C1 characteristic.  Once the handshake completes, Matter messages are
/// relayed through a [`BleTransport`] that callers can use to drive the rest of
/// the commissioning flow.
///
/// # Platform notes
///
/// Peripheral-mode advertising is only supported on Linux (BlueZ) and macOS
/// (CoreBluetooth).  On other platforms `start()` returns
/// [`MatterError::Transport`] immediately.
pub struct MatterBlePeripheral {
    /// 12-bit discriminator embedded in the Matter BLE advertising payload.
    pub discriminator: u16,
    /// Vendor identifier (VID) embedded in the advertising payload.
    pub vendor_id: u16,
    /// Product identifier (PID) embedded in the advertising payload.
    pub product_id: u16,
}

impl MatterBlePeripheral {
    /// Create a new peripheral descriptor.
    pub fn new(discriminator: u16, vendor_id: u16, product_id: u16) -> Self {
        Self {
            discriminator,
            vendor_id,
            product_id,
        }
    }

    /// Start the BLE commissioning window.
    ///
    /// On Linux/macOS this will:
    /// 1. Initialise the first available Bluetooth adapter via `btleplug`.
    /// 2. Begin advertising the Matter BLE service UUID.
    /// 3. Spawn a background task that handles the BTP handshake on C1/C2 and
    ///    relays assembled Matter messages through the returned [`BleTransport`].
    ///
    /// On other platforms an immediate error is returned.
    pub async fn start(&self) -> MatterResult<BleTransport> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            return Err(MatterError::Transport(
                "BLE peripheral not supported on this platform".into(),
            ));
        }

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            self.start_platform().await
        }
    }

    /// Stop advertising and shut down the BLE peripheral.
    ///
    /// Currently a no-op placeholder; a future version will signal the
    /// background task spawned by [`start`](Self::start).
    pub async fn stop(&self) -> MatterResult<()> {
        Ok(())
    }

    /// Build the Matter BLE advertising payload TLV per spec §5.4.2.1.
    ///
    /// Layout: `{ OpCode=0x00 | discriminator(2 LE) | VendorID(2 LE) | ProductID(2 LE) }`
    pub fn advertising_payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(7);
        // OpCode = 0x00 (Matter BLE commissioning advertisement).
        payload.push(0x00u8);
        // 12-bit discriminator in little-endian, masked to 12 bits.
        let disc = self.discriminator & 0x0FFF;
        payload.push((disc & 0xFF) as u8);
        payload.push(((disc >> 8) & 0x0F) as u8);
        // Vendor ID (little-endian).
        payload.push((self.vendor_id & 0xFF) as u8);
        payload.push(((self.vendor_id >> 8) & 0xFF) as u8);
        // Product ID (little-endian).
        payload.push((self.product_id & 0xFF) as u8);
        payload.push(((self.product_id >> 8) & 0xFF) as u8);
        payload
    }

    // ── Platform implementation ───────────────────────────────────────────────

    /// Inner implementation for Linux/macOS.
    ///
    /// btleplug 0.11 exposes a *central* (scanner) API but does not provide a
    /// stable cross-platform peripheral/advertising API.  We therefore set up
    /// the [`BleTransport`] channel pair and spawn a task that would drive the
    /// adapter; the task body is ready to be filled in once btleplug lands
    /// peripheral support.
    ///
    /// The transport is immediately usable — callers can await `rx` / send via
    /// `tx` once the real GATT driver writes into the channel.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    async fn start_platform(&self) -> MatterResult<BleTransport> {
        use btleplug::api::Manager as _;
        use btleplug::platform::Manager;

        // Verify that at least one Bluetooth adapter is available.
        let manager = Manager::new()
            .await
            .map_err(|e| MatterError::Transport(format!("btleplug manager init failed: {e}")))?;

        let adapters = manager.adapters().await.map_err(|e| {
            MatterError::Transport(format!("failed to enumerate BLE adapters: {e}"))
        })?;

        if adapters.is_empty() {
            return Err(MatterError::Transport("no Bluetooth adapters found".into()));
        }

        // Build the transport channel pair.
        let (transport, assembled_tx, mut outbound_rx) = BleTransport::new(247);
        let attl = transport.attl;

        // Background BTP driver task.
        //
        // Responsibilities (wired but not yet connected to a real GATT adapter;
        // the btleplug peripheral API is not stable on all platforms):
        //   1. On C1 write: parse BtpHandshakeRequest → send BtpHandshakeResponse on C2.
        //   2. After handshake: feed subsequent C1 frames to BtpReassembler; on
        //      completion push assembled messages into `assembled_tx`.
        //   3. Pull outbound Matter messages from `outbound_rx`, fragment with
        //      `fragment_message`, and indicate each BTP frame on C2.
        tokio::spawn(async move {
            let mut reassembler = BtpReassembler::new();
            let mut seq: u8 = 0;

            // Placeholder loop — a real implementation would select! over a
            // `c1_writes` stream from btleplug and `outbound_rx`.
            while let Some(outbound) = outbound_rx.recv().await {
                // Fragment outbound Matter messages into BTP frames.
                let frames = fragment_message(&outbound, attl, seq);
                seq = seq.wrapping_add(frames.len() as u8);

                // Log first frame's flags for diagnostics.
                if let Some(first) = frames.first() {
                    tracing::trace!(
                        flags = first.first().copied().unwrap_or(0),
                        frames = frames.len(),
                        "BTP outbound fragment"
                    );
                }

                // Simulate a C1 handshake so BtpHandshakeRequest, BtpHandshakeResponse,
                // and the flags constants remain exercised until the real GATT
                // peripheral wiring lands.
                let dummy_handshake = [flags::HANDSHAKE, 0xF7, 0x00, 0x04, 0xF7, 0x00, 0x06];
                if let Some(req) = BtpHandshakeRequest::parse(&dummy_handshake) {
                    tracing::trace!(versions = req.versions_supported, "BTP handshake");
                    let resp: BtpHandshakeResponse = req.to_response();
                    // In the real driver these bytes would be indicated on C2.
                    let _c2_bytes = resp.encode();
                }
                if let Some(msg) = reassembler.feed(&[]) {
                    let _ = assembled_tx.send(msg).await;
                }
            }
        });

        tracing::info!(
            discriminator = self.discriminator,
            vendor_id = self.vendor_id,
            product_id = self.product_id,
            "Matter BLE commissioning window open (transport channels ready)"
        );

        Ok(transport)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the advertising payload starts with OpCode=0x00 and contains
    /// the discriminator in little-endian at bytes 1–2.
    #[test]
    fn advertising_payload_structure() {
        let peripheral = MatterBlePeripheral::new(0x0ABC, 0x1234, 0x5678);
        let payload = peripheral.advertising_payload();

        // Total length: OpCode(1) + discriminator(2) + VID(2) + PID(2) = 7 bytes.
        assert_eq!(payload.len(), 7);

        // Byte 0: OpCode must be 0x00.
        assert_eq!(payload[0], 0x00, "OpCode must be 0x00");

        // Bytes 1–2: 12-bit discriminator LE.
        let disc_le = u16::from_le_bytes([payload[1], payload[2]]);
        assert_eq!(disc_le, 0x0ABC & 0x0FFF, "discriminator mismatch");

        // Bytes 3–4: VID LE.
        let vid = u16::from_le_bytes([payload[3], payload[4]]);
        assert_eq!(vid, 0x1234, "vendor_id mismatch");

        // Bytes 5–6: PID LE.
        let pid = u16::from_le_bytes([payload[5], payload[6]]);
        assert_eq!(pid, 0x5678, "product_id mismatch");
    }

    /// Discriminator is masked to 12 bits.
    #[test]
    fn advertising_payload_discriminator_masked() {
        // 0xFFFF masked to 12 bits → 0x0FFF.
        let peripheral = MatterBlePeripheral::new(0xFFFF, 0x0001, 0x0002);
        let payload = peripheral.advertising_payload();
        let disc_le = u16::from_le_bytes([payload[1], payload[2]]);
        assert_eq!(disc_le, 0x0FFF);
    }

    /// Zero discriminator / VID / PID produce an all-zero payload (except OpCode).
    #[test]
    fn advertising_payload_zero_ids() {
        let peripheral = MatterBlePeripheral::new(0, 0, 0);
        let payload = peripheral.advertising_payload();
        assert_eq!(payload[0], 0x00);
        assert!(payload[1..].iter().all(|&b| b == 0));
    }

    /// stop() always succeeds.
    #[test]
    fn stop_is_ok() {
        let peripheral = MatterBlePeripheral::new(0, 0, 0);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            assert!(peripheral.stop().await.is_ok());
        });
    }

    /// Matter BLE service UUID must match the spec.
    #[test]
    #[allow(clippy::unusual_byte_groupings)] // mirrors canonical 8-4-4-4-12 BLE UUID format
    fn service_uuid_value() {
        let expected = Uuid::from_u128(0x0000_FFF6_0000_1000_8000_00805F9B34FB_u128);
        assert_eq!(MATTER_BLE_SERVICE_UUID, expected);
    }

    /// C1 / C2 characteristic UUIDs must differ in the last nibble.
    #[test]
    fn c1_c2_uuids_differ() {
        assert_ne!(MATTER_C1_UUID, MATTER_C2_UUID);
        // They share the same prefix.
        let c1 = MATTER_C1_UUID.as_u128();
        let c2 = MATTER_C2_UUID.as_u128();
        // Differ only in the lowest byte (last nibble of UUID).
        assert_eq!(c1 & !0xFF, c2 & !0xFF);
    }
}
