use nusb::DeviceInfo;
use tracing::{debug, warn};

use super::types::{UsbClass, UsbDevice, UsbError, UsbSpeed};

/// Enumerate all USB devices currently attached to the system.
///
/// String descriptors (manufacturer, product, serial) are read where
/// accessible. Devices that cannot be opened for string descriptor reads
/// (e.g. due to permissions) will have `None` for those fields.
pub fn list_usb_devices() -> Vec<UsbDevice> {
    let devices = match nusb::list_devices() {
        Ok(d) => d,
        Err(e) => {
            warn!("Failed to list USB devices: {e}");
            return Vec::new();
        }
    };

    devices.filter_map(device_from_info).collect()
}

/// Find the first device matching `vendor_id:product_id`.
pub fn find_device(vendor_id: u16, product_id: u16) -> Result<DeviceInfo, UsbError> {
    let mut iter = nusb::list_devices().map_err(|e| UsbError::OpenFailed(e.to_string()))?;
    iter.find(|d| d.vendor_id() == vendor_id && d.product_id() == product_id)
        .ok_or(UsbError::DeviceNotFound {
            vendor_id,
            product_id,
        })
}

fn device_from_info(info: DeviceInfo) -> Option<UsbDevice> {
    let vendor_id = info.vendor_id();
    let product_id = info.product_id();
    let class = UsbClass::from_code(info.class());
    let speed = info.speed().map_or(UsbSpeed::Unknown, speed_from_nusb);

    debug!(
        "{:04x}:{:04x} class={class} speed={speed}",
        vendor_id, product_id
    );

    // Attempt to read string descriptors — requires device open permission.
    // Failures are non-fatal; we just leave the fields as None.
    let (manufacturer, product, serial) = read_strings(&info);

    Some(UsbDevice {
        bus: info.bus_number(),
        device_address: info.device_address(),
        vendor_id,
        product_id,
        class,
        speed,
        manufacturer,
        product,
        serial,
    })
}

fn read_strings(info: &DeviceInfo) -> (Option<String>, Option<String>, Option<String>) {
    // nusb exposes string descriptor indices; open the device to read them.
    // This is best-effort — silently fail on permission errors.
    let device = match info.open() {
        Ok(d) => d,
        Err(_) => return (None, None, None),
    };

    let manufacturer = info.manufacturer_string().map(|s| s.to_string());

    let product = info.product_string().map(|s| s.to_string());

    let serial = info.serial_number().map(|s| s.to_string());

    drop(device);
    (manufacturer, product, serial)
}

fn speed_from_nusb(speed: nusb::Speed) -> UsbSpeed {
    match speed {
        nusb::Speed::Low => UsbSpeed::Low,
        nusb::Speed::Full => UsbSpeed::Full,
        nusb::Speed::High => UsbSpeed::High,
        nusb::Speed::Super => UsbSpeed::Super,
        nusb::Speed::SuperPlus => UsbSpeed::SuperPlus,
        _ => UsbSpeed::Unknown,
    }
}
