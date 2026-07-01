use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Information about a USB device attached to the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsbDevice {
    /// USB bus number.
    pub bus: u8,
    /// Device address on the bus.
    pub device_address: u8,
    /// USB Vendor ID (VID).
    pub vendor_id: u16,
    /// USB Product ID (PID).
    pub product_id: u16,
    /// Device class code.
    pub class: UsbClass,
    /// Negotiated bus speed.
    pub speed: UsbSpeed,
    /// Manufacturer string, if readable.
    pub manufacturer: Option<String>,
    /// Product string, if readable.
    pub product: Option<String>,
    /// Serial number string, if readable.
    pub serial: Option<String>,
}

impl UsbDevice {
    /// Format as `VID:PID` hex string (e.g. `"046d:c52b"`).
    pub fn vid_pid(&self) -> String {
        format!("{:04x}:{:04x}", self.vendor_id, self.product_id)
    }
}

/// USB device class codes as per the USB-IF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsbClass {
    /// 0x01 — Audio
    Audio,
    /// 0x02 — Communications and CDC Control
    Cdc,
    /// 0x03 — Human Interface Device
    Hid,
    /// 0x05 — Physical
    Physical,
    /// 0x06 — Still Imaging / Image
    Image,
    /// 0x07 — Printer
    Printer,
    /// 0x08 — Mass Storage
    MassStorage,
    /// 0x09 — Hub
    Hub,
    /// 0x0A — CDC-Data
    CdcData,
    /// 0x0B — Smart Card
    SmartCard,
    /// 0x0D — Content Security
    ContentSecurity,
    /// 0x0E — Video
    Video,
    /// 0x0F — Personal Healthcare
    PersonalHealthcare,
    /// 0x10 — Audio/Video Devices
    AudioVideo,
    /// 0x11 — Billboard Device Class
    Billboard,
    /// 0x12 — USB Type-C Bridge Class
    TypeCBridge,
    /// 0xDC — Diagnostic Device
    Diagnostic,
    /// 0xE0 — Wireless Controller (Bluetooth, etc.)
    WirelessController,
    /// 0xEF — Miscellaneous
    Miscellaneous,
    /// 0xFE — Application Specific
    ApplicationSpecific,
    /// 0xFF — Vendor Specific
    VendorSpecific,
    /// Class code not in the above list.
    Unknown(u8),
}

impl UsbClass {
    /// Decode a USB class byte from a device descriptor into the named variant.
    pub fn from_code(code: u8) -> Self {
        match code {
            0x01 => Self::Audio,
            0x02 => Self::Cdc,
            0x03 => Self::Hid,
            0x05 => Self::Physical,
            0x06 => Self::Image,
            0x07 => Self::Printer,
            0x08 => Self::MassStorage,
            0x09 => Self::Hub,
            0x0A => Self::CdcData,
            0x0B => Self::SmartCard,
            0x0D => Self::ContentSecurity,
            0x0E => Self::Video,
            0x0F => Self::PersonalHealthcare,
            0x10 => Self::AudioVideo,
            0x11 => Self::Billboard,
            0x12 => Self::TypeCBridge,
            0xDC => Self::Diagnostic,
            0xE0 => Self::WirelessController,
            0xEF => Self::Miscellaneous,
            0xFE => Self::ApplicationSpecific,
            0xFF => Self::VendorSpecific,
            other => Self::Unknown(other),
        }
    }
}

impl std::fmt::Display for UsbClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Audio => "Audio",
            Self::Cdc => "CDC",
            Self::Hid => "HID",
            Self::Physical => "Physical",
            Self::Image => "Image",
            Self::Printer => "Printer",
            Self::MassStorage => "Mass Storage",
            Self::Hub => "Hub",
            Self::CdcData => "CDC-Data",
            Self::SmartCard => "Smart Card",
            Self::ContentSecurity => "Content Security",
            Self::Video => "Video",
            Self::PersonalHealthcare => "Personal Healthcare",
            Self::AudioVideo => "Audio/Video",
            Self::Billboard => "Billboard",
            Self::TypeCBridge => "Type-C Bridge",
            Self::Diagnostic => "Diagnostic",
            Self::WirelessController => "Wireless Controller",
            Self::Miscellaneous => "Miscellaneous",
            Self::ApplicationSpecific => "Application Specific",
            Self::VendorSpecific => "Vendor Specific",
            Self::Unknown(n) => return write!(f, "Unknown(0x{n:02x})"),
        };
        f.write_str(name)
    }
}

/// USB bus speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsbSpeed {
    /// 1.5 Mbit/s
    Low,
    /// 12 Mbit/s
    Full,
    /// 480 Mbit/s
    High,
    /// 5 Gbit/s
    Super,
    /// 10 Gbit/s
    SuperPlus,
    /// Speed not reported by the OS.
    Unknown,
}

impl std::fmt::Display for UsbSpeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Low => "Low-Speed (1.5 Mbit/s)",
            Self::Full => "Full-Speed (12 Mbit/s)",
            Self::High => "High-Speed (480 Mbit/s)",
            Self::Super => "SuperSpeed (5 Gbit/s)",
            Self::SuperPlus => "SuperSpeed+ (10 Gbit/s)",
            Self::Unknown => "Unknown",
        };
        f.write_str(s)
    }
}

/// Errors from USB operations.
#[derive(Debug, Error)]
pub enum UsbError {
    /// No device with the requested VID:PID is attached.
    #[error("no device found with VID:PID {vendor_id:04x}:{product_id:04x}")]
    DeviceNotFound {
        /// Requested USB Vendor ID.
        vendor_id: u16,
        /// Requested USB Product ID.
        product_id: u16,
    },
    /// Device was discoverable but could not be opened.
    #[error("failed to open device: {0}")]
    OpenFailed(String),
    /// Interface claim (`claim_interface`) failed.
    #[error("failed to claim interface {0}: {1}")]
    ClaimFailed(u8, String),
    /// Bulk/interrupt/control transfer failed.
    #[error("transfer failed on endpoint 0x{endpoint:02x}: {reason}")]
    TransferFailed {
        /// Endpoint address (0x80 bit = IN, else OUT).
        endpoint: u8,
        /// OS-level failure reason.
        reason: String,
    },
    /// Transfer did not complete within its timeout window.
    #[error("transfer timed out on endpoint 0x{0:02x}")]
    Timeout(u8),
    /// Catch-all for platform-specific failures.
    #[error("USB error: {0}")]
    Other(String),
}
