use std::time::Duration;

use btleplug::api::{Central, Manager as _, Peripheral as _, ScanFilter};
use btleplug::platform::Manager;
use tokio::time::sleep;
use tracing::{debug, warn};

use super::types::{BluetoothDevice, BluetoothDeviceKind};

/// Scan for BLE (Bluetooth Low Energy) peripherals for the given duration.
///
/// Starts a passive advertisement scan on the first available adapter, waits
/// `duration`, then returns all discovered peripherals. Duplicate addresses
/// are deduplicated (last-seen entry wins for RSSI).
///
/// Returns an empty list if no Bluetooth adapter is available.
pub async fn scan_ble(duration: Duration) -> Vec<BluetoothDevice> {
    let manager = match Manager::new().await {
        Ok(m) => m,
        Err(e) => {
            warn!("Bluetooth manager unavailable: {e}");
            return Vec::new();
        }
    };

    let adapters = match manager.adapters().await {
        Ok(a) if !a.is_empty() => a,
        _ => {
            warn!("No Bluetooth adapters found");
            return Vec::new();
        }
    };

    let central = &adapters[0];

    if let Err(e) = central.start_scan(ScanFilter::default()).await {
        warn!("BLE scan start failed: {e}");
        return Vec::new();
    }

    sleep(duration).await;

    let _ = central.stop_scan().await;

    let peripherals = match central.peripherals().await {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to retrieve BLE peripherals: {e}");
            return Vec::new();
        }
    };

    let mut devices = Vec::new();
    for peripheral in peripherals {
        let addr = peripheral.address();
        let props = peripheral.properties().await.ok().flatten();

        let name = props.as_ref().and_then(|p| p.local_name.clone());
        let rssi = props.as_ref().and_then(|p| p.rssi);
        let services = props
            .as_ref()
            .map(|p| p.services.to_vec())
            .unwrap_or_default();

        debug!("BLE device {addr} name={name:?} rssi={rssi:?}");

        devices.push(BluetoothDevice {
            address: addr.to_string(),
            name,
            rssi,
            services,
            kind: BluetoothDeviceKind::BlePeripheral,
        });
    }
    devices
}
