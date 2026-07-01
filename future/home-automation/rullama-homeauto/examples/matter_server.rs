//! Matter 1.3 device server — exposes a rullama agent as a Matter device.
//!
//! Run:
//! ```bash
//! cargo run --example matter_server --features matter
//! ```
//!
//! Then use the displayed QR code URL or manual pairing code to add the device
//! to Apple Home, Google Home, Home Assistant, or another Matter controller.
//!
//! ## What this does
//!
//! 1. Creates a `MatterDeviceServer` with a test Vendor ID (0xFFF1) and
//!    Product ID (0x8001), discriminator 3840, and the standard development
//!    passcode `20202021`.
//! 2. Registers on_off, level, color-temperature, and thermostat handlers that
//!    print to stdout so you can see incoming commands.
//! 3. Prints the QR code URL (paste into a browser or scan with your phone)
//!    and the manual 11-digit pairing code.
//! 4. Starts the server on UDP port 5540 (the standard Matter port) and
//!    advertises via mDNS so any Matter controller on the same network can
//!    discover and commission it.
//! 5. Runs until Ctrl+C.
//!
//! ## Real-hardware note
//!
//! The server performs a full PASE commissioning handshake and, once
//! commissioned, accepts CASE operational sessions.  Any Matter 1.x
//! controller (chip-tool, Apple Home, Google Home, Home Assistant Matter
//! integration) that can reach this machine on UDP 5540 will work.

use anyhow::Result;
use rullama_homeauto::matter::{MatterDeviceConfig, MatterDeviceServer};
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // 1. Build device config (test VID/PID, standard dev passcode).
    let config = MatterDeviceConfig::builder()
        .device_name("Brainwires Agent Light")
        .vendor_id(0xFFF1) // 0xFFF1 = test/development vendor
        .product_id(0x8001)
        .discriminator(3840)
        .passcode(20202021) // standard Matter development passcode
        .storage_path("/tmp/rullama-matter-server")
        .port(5540) // standard Matter UDP port
        .build();

    // 2. Create the server.
    let server = MatterDeviceServer::new(config).await?;

    // 3. Register cluster handlers — print every incoming command to stdout.
    server.set_on_off_handler(|on| {
        let state = if on { "ON" } else { "OFF" };
        println!("[Matter] On/Off → {state}");
    });

    server.set_level_handler(|level| {
        let pct = level as f32 / 254.0 * 100.0;
        println!("[Matter] Level → {level}/254  ({pct:.0}%)");
    });

    server.set_color_temp_handler(|mireds| {
        let kelvin = 1_000_000u32.checked_div(mireds as u32).unwrap_or(0);
        println!("[Matter] Color temperature → {mireds} mireds  (~{kelvin} K)");
    });

    server.set_thermostat_handler(|celsius| {
        println!("[Matter] Thermostat setpoint → {celsius:.1}°C");
    });

    // 4. Print commissioning information.
    println!();
    println!("==========================================================");
    println!("  Matter 1.3 device server");
    println!("==========================================================");
    println!("  QR code:       {}", server.qr_code());
    println!("  Pairing code:  {}", server.pairing_code());
    println!(
        "  QR URL:        https://project-chip.github.io/connectedhomeip/qrcode.html?data={}",
        urlencoded(server.qr_code())
    );
    println!();
    println!("  Scan the QR code (or enter the pairing code) in:");
    println!("  - Apple Home (iOS 16.2+ / macOS 13+)");
    println!("  - Google Home");
    println!("  - Home Assistant  (Settings → Devices → Add integration → Matter)");
    println!(
        "  - chip-tool:  chip-tool pairing qrcode 1 \"{}\"",
        server.qr_code()
    );
    println!("==========================================================");
    println!();

    // 5. Print mDNS service info.
    info!("Advertising as '_matterc._udp' (discriminator=3840) on UDP port 5540");
    info!("Waiting for commissioner…  Press Ctrl+C to stop.");

    // 6. Run server until Ctrl+C.
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    // Wrap in Option so the oneshot sender is only consumed on the first signal.
    let mut tx_opt = Some(tx);
    ctrlc::set_handler(move || {
        if let Some(tx) = tx_opt.take() {
            let _ = tx.send(());
        }
    })?;

    let server_task = tokio::spawn(async move {
        if let Err(e) = server_clone.start().await {
            eprintln!("Matter server error: {e}");
        }
    });

    // Block until Ctrl+C.
    let _ = rx.await;
    info!("Shutting down…");
    server.stop().await?;
    server_task.abort();

    Ok(())
}

/// Percent-encode a string for use in a URL query parameter.
fn urlencoded(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c]
            }
            c => format!("%{:02X}", c as u32).chars().collect::<Vec<_>>(),
        })
        .collect()
}
