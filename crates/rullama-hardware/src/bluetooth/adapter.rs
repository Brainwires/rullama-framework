use btleplug::api::{Central, Manager as _};
use btleplug::platform::Manager;

use super::types::BluetoothAdapter;

/// Enumerate all available Bluetooth adapters on this system.
///
/// Returns an empty list if the Bluetooth stack is unavailable or no adapters
/// are installed.
pub async fn list_adapters() -> Vec<BluetoothAdapter> {
    let manager = match Manager::new().await {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    let adapters = match manager.adapters().await {
        Ok(a) => a,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.adapter_info().await;
        let name = info.unwrap_or_else(|_| format!("hci{i}"));
        result.push(BluetoothAdapter {
            id: format!("adapter-{i}"),
            name,
        });
    }
    result
}
