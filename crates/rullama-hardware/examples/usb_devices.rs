//! List all USB devices attached to the system.
//!
//! Run with:
//! ```bash
//! cargo run -p rullama-hardware --example usb_devices --features usb
//! ```
//!
//! On Linux you may need to run as root or add a udev rule for full access
//! to string descriptors.

use rullama_hardware::usb;

fn main() {
    let devices = usb::list_usb_devices();

    if devices.is_empty() {
        println!("No USB devices found (may need elevated permissions).");
        return;
    }

    println!("{} USB device(s):\n", devices.len());
    println!(
        "{:<12} {:>4} {:>4}  {:<22}  {:<16}  {:<}",
        "BUS:ADDR", "VID", "PID", "Class", "Speed", "Product"
    );
    println!("{}", "-".repeat(90));

    for d in &devices {
        let product = match (&d.manufacturer, &d.product) {
            (Some(mfr), Some(prd)) => format!("{mfr} {prd}"),
            (None, Some(prd)) => prd.clone(),
            (Some(mfr), None) => mfr.clone(),
            (None, None) => String::new(),
        };
        let serial = d.serial.as_deref().unwrap_or("");

        println!(
            "{:03}:{:03}   {:04x}:{:04x}  {:<22}  {:<16}  {}{}",
            d.bus,
            d.device_address,
            d.vendor_id,
            d.product_id,
            d.class.to_string(),
            d.speed.to_string(),
            product,
            if serial.is_empty() {
                String::new()
            } else {
                format!(" [{serial}]")
            },
        );
    }
}
