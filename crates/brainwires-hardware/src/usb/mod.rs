//! Raw USB device access.
//!
//! Provides device enumeration and raw bulk, control, and interrupt transfers
//! using [`nusb`](https://crates.io/crates/nusb) — a pure-Rust async USB
//! library with no system `libusb` dependency.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use brainwires_hardware::usb;
//!
//! // List all attached USB devices
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let devices = usb::list_usb_devices();
//!     for d in &devices {
//!         println!(
//!             "{} — {} {} ({})",
//!             d.vid_pid(),
//!             d.manufacturer.as_deref().unwrap_or("?"),
//!             d.product.as_deref().unwrap_or("?"),
//!             d.speed,
//!         );
//!     }
//!
//!     // Open a specific device for raw transfers
//!     // (replace VID/PID and interface number for your device)
//!     let handle = usb::UsbHandle::open(0x046d, 0xc52b, 0).await?;
//!
//!     // Bulk read 64 bytes from auto-detected IN endpoint
//!     if let Some(ep) = handle.bulk_in_endpoint() {
//!         use std::time::Duration;
//!         let data = handle.bulk_read(Some(ep), 64, Duration::from_millis(500)).await?;
//!         println!("Read {} bytes: {:?}", data.len(), &data[..data.len().min(16)]);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Permissions
//!
//! On Linux, USB device access typically requires either `root` or a `udev`
//! rule granting access to the device node (e.g. `SUBSYSTEM=="usb",
//! ATTR{idVendor}=="046d", MODE="0666"`).
//!
//! On macOS, no special permissions are needed for non-HID class devices.
//!
//! On Windows, a WinUSB-compatible driver must be installed for the target
//! device (e.g. via [Zadig](https://zadig.akeo.ie/)).

/// USB device enumeration and lookup helpers.
pub mod device;
/// Open-device handle and bulk/interrupt/control transfer primitives.
pub mod transfer;
/// Typed descriptors: device, class codes, speeds, errors.
pub mod types;

pub use device::{find_device, list_usb_devices};
pub use transfer::UsbHandle;
pub use types::{UsbClass, UsbDevice, UsbError, UsbSpeed};
