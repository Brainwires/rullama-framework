//! Commission a Matter device and send On/Off commands.
//!
//! This example demonstrates `MatterController` commissioning a real Matter
//! device over UDP and then toggling it on and off.
//!
//! ## Usage
//!
//! ### Commission a device from its QR code
//! ```bash
//! cargo run --example matter_on_off --features matter -- commission "MT:Y.K9042C00KA0648G00"
//! ```
//!
//! Set `MATTER_DEVICE_QR` to avoid passing the code on the command line:
//! ```bash
//! export MATTER_DEVICE_QR="MT:Y.K9042C00KA0648G00"
//! cargo run --example matter_on_off --features matter -- commission
//! ```
//!
//! ### Commission using an 11-digit manual pairing code
//! ```bash
//! cargo run --example matter_on_off --features matter -- code "34970112332"
//! ```
//!
//! ## Real hardware required
//!
//! `commission_qr` and `commission_code` perform mDNS discovery over the
//! local network.  The device must:
//! - Be powered on and in commissioning mode (factory-reset or pairing window open).
//! - Be discoverable via `_matterc._udp` on the same network segment.
//! - Accept the matching passcode.
//!
//! Suitable hardware: any Matter 1.x certified on/off bulb, plug, or switch.
//! The development passcode `20202021` is standard on many test/dev builds.

use anyhow::Result;
use rullama_homeauto::matter::MatterController;
use std::env;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let subcommand = env::args().nth(1).unwrap_or_else(|| "commission".into());
    match subcommand.as_str() {
        "commission" => {
            // Prefer env var, fall back to CLI arg, then a hard-coded test QR.
            let qr_code = env::var("MATTER_DEVICE_QR")
                .ok()
                .or_else(|| env::args().nth(2))
                .unwrap_or_else(|| "MT:Y.K9042C00KA0648G00".into());
            commission_and_toggle(&qr_code, /*use_qr=*/ true).await
        }
        "code" => {
            let pairing_code = env::var("MATTER_DEVICE_CODE")
                .ok()
                .or_else(|| env::args().nth(2))
                .unwrap_or_else(|| "34970112332".into());
            commission_and_toggle(&pairing_code, /*use_qr=*/ false).await
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  matter_on_off commission [QR_CODE | $MATTER_DEVICE_QR]");
            eprintln!("  matter_on_off code       [11-DIGIT-CODE | $MATTER_DEVICE_CODE]");
            Ok(())
        }
    }
}

/// Commission a device then send On → wait 2 s → Off.
async fn commission_and_toggle(payload: &str, use_qr: bool) -> Result<()> {
    // 1. Create controller — persists fabric state to a temp directory.
    let storage = std::path::Path::new("/tmp/rullama-matter-controller");
    let controller = MatterController::new("Brainwires Fabric", storage).await?;

    // 2. Commission via QR code or manual pairing code.
    let device = if use_qr {
        info!("Commissioning via QR code: {payload}");
        controller.commission_qr(payload, 1).await?
    } else {
        info!("Commissioning via manual pairing code: {payload}");
        controller.commission_code(payload, 1).await?
    };

    info!(
        "Device commissioned: node_id={} VID={:#06x} PID={:#06x}",
        device.node_id, device.vendor_id, device.product_id
    );

    // 3. Turn on.
    info!("Sending On command to endpoint 1…");
    controller.on_off(&device, 1, true).await?;
    info!("On command sent.");

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // 4. Turn off.
    info!("Sending Off command to endpoint 1…");
    controller.on_off(&device, 1, false).await?;
    info!("Off command sent.");

    Ok(())
}
