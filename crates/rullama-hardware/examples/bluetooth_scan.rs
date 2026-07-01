//! Scan for nearby Bluetooth devices and print results.
//!
//! Run with:
//! ```bash
//! cargo run -p rullama-hardware --example bluetooth_scan --features bluetooth
//! ```

use std::time::Duration;

#[tokio::main]
async fn main() {
    let adapters = rullama_hardware::bluetooth::list_adapters().await;
    println!("Bluetooth adapters ({}):", adapters.len());
    for a in &adapters {
        println!("  {} — {}", a.id, a.name);
    }

    if adapters.is_empty() {
        eprintln!("No Bluetooth adapters found.");
        return;
    }

    println!("\nScanning for BLE devices (5 seconds)...");
    let devices = rullama_hardware::bluetooth::scan_ble(Duration::from_secs(5)).await;
    println!("Found {} device(s):", devices.len());
    for d in &devices {
        println!(
            "  {} — name={:?}  rssi={:?} dBm  services={}",
            d.address,
            d.name,
            d.rssi,
            d.services.len()
        );
    }
}
