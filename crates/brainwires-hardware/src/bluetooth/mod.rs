//! Bluetooth hardware discovery and scanning.
//!
//! Provides BLE advertisement scanning and adapter enumeration using
//! [`btleplug`](https://crates.io/crates/btleplug) for cross-platform support
//! (Linux/BlueZ, macOS CoreBluetooth, Windows WinRT).
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use brainwires_hardware::bluetooth;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let adapters = bluetooth::list_adapters().await;
//!     println!("Adapters: {adapters:?}");
//!
//!     let devices = bluetooth::scan_ble(Duration::from_secs(5)).await;
//!     for d in &devices {
//!         println!("{} — {:?} ({:?} dBm)", d.address, d.name, d.rssi);
//!     }
//! }
//! ```

/// Enumerate host BLE adapters.
pub mod adapter;
/// BLE advertisement scanning helpers.
pub mod scanner;
/// Typed adapter / device / kind enums.
pub mod types;

pub use adapter::list_adapters;
pub use scanner::scan_ble;
pub use types::{BluetoothAdapter, BluetoothDevice, BluetoothDeviceKind};
