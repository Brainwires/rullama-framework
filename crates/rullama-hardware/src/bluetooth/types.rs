use btleplug::api::BDAddr;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A local Bluetooth adapter (radio).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BluetoothAdapter {
    /// Platform-specific adapter identifier.
    pub id: String,
    /// Human-readable adapter name (e.g. "hci0").
    pub name: String,
}

/// Information about a discovered Bluetooth device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BluetoothDevice {
    /// Bluetooth device address (48-bit MAC).
    pub address: String,
    /// Advertised local name, if available.
    pub name: Option<String>,
    /// Received signal strength in dBm.
    pub rssi: Option<i16>,
    /// Advertised GATT service UUIDs.
    pub services: Vec<Uuid>,
    /// Device classification.
    pub kind: BluetoothDeviceKind,
}

impl BluetoothDevice {
    /// Create from a btleplug peripheral address + optional metadata.
    pub fn from_addr(addr: BDAddr) -> Self {
        Self {
            address: addr.to_string(),
            name: None,
            rssi: None,
            services: Vec::new(),
            kind: BluetoothDeviceKind::BlePeripheral,
        }
    }
}

/// Classification of a discovered Bluetooth device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BluetoothDeviceKind {
    /// BLE (Bluetooth Low Energy) peripheral.
    BlePeripheral,
    /// Classic Bluetooth device (BR/EDR).
    Classic,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BluetoothAdapter ---

    #[test]
    fn adapter_serde_roundtrip() {
        let adapter = BluetoothAdapter {
            id: "hci0".to_string(),
            name: "Intel Bluetooth".to_string(),
        };
        let json = serde_json::to_string(&adapter).unwrap();
        let back: BluetoothAdapter = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "hci0");
        assert_eq!(back.name, "Intel Bluetooth");
    }

    // --- BluetoothDeviceKind ---

    #[test]
    fn device_kind_serde_roundtrip() {
        for kind in [
            BluetoothDeviceKind::BlePeripheral,
            BluetoothDeviceKind::Classic,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: BluetoothDeviceKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    // --- BluetoothDevice ---

    #[test]
    fn device_serde_roundtrip() {
        let device = BluetoothDevice {
            address: "AA:BB:CC:DD:EE:FF".to_string(),
            name: Some("My Headphones".to_string()),
            rssi: Some(-70),
            services: vec![],
            kind: BluetoothDeviceKind::BlePeripheral,
        };
        let json = serde_json::to_string(&device).unwrap();
        let back: BluetoothDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.address, "AA:BB:CC:DD:EE:FF");
        assert_eq!(back.name.as_deref(), Some("My Headphones"));
        assert_eq!(back.rssi, Some(-70));
        assert_eq!(back.kind, BluetoothDeviceKind::BlePeripheral);
    }

    #[test]
    fn device_optional_fields_when_none() {
        let device = BluetoothDevice {
            address: "11:22:33:44:55:66".to_string(),
            name: None,
            rssi: None,
            services: vec![],
            kind: BluetoothDeviceKind::Classic,
        };
        let json = serde_json::to_string(&device).unwrap();
        let back: BluetoothDevice = serde_json::from_str(&json).unwrap();
        assert!(back.name.is_none());
        assert!(back.rssi.is_none());
    }
}
