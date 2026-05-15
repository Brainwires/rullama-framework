//! BLE commissioning support for Matter (`matter-ble` feature).
//!
//! Implements the Matter BLE Transport Protocol (BTP) peripheral role,
//! allowing Matter controllers to commission this device over Bluetooth.
//!
//! # Platform support
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux (BlueZ)         | Supported |
//! | macOS (CoreBluetooth) | Supported |
//! | Windows               | Not supported (btleplug 0.11 has no peripheral API) |
//!
//! # Usage
//!
//! ```rust,no_run
//! # #[cfg(feature = "matter-ble")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use brainwires_homeauto::matter::ble::MatterBlePeripheral;
//!
//! let peripheral = MatterBlePeripheral::new(
//!     0x0ABC, // 12-bit discriminator
//!     0x1234, // vendor ID
//!     0x5678, // product ID
//! );
//!
//! let mut transport = peripheral.start().await?;
//!
//! // Receive assembled Matter messages from the commissioner.
//! while let Some(msg) = transport.rx.recv().await {
//!     // Process commissioning message …
//!     let _ = msg;
//! }
//! # Ok(())
//! # }
//! ```

pub mod peripheral;

pub use peripheral::{
    MATTER_BLE_SERVICE_UUID, MATTER_C1_UUID, MATTER_C2_UUID, MatterBlePeripheral,
};
