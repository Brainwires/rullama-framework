use crate::output::Output;
use anyhow::Result;
use brainwires_homeauto::{MatterDeviceConfig, MatterDeviceServer};
use std::path::PathBuf;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    device_name: String,
    vendor_id: u16,
    product_id: u16,
    discriminator: u16,
    passcode: u32,
    port: u16,
    storage: Option<PathBuf>,
    out: &Output,
) -> Result<()> {
    let storage_path = storage.unwrap_or_else(|| {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("matter-tool")
            .join("server")
    });

    let config = MatterDeviceConfig {
        device_name: device_name.clone(),
        vendor_id,
        product_id,
        discriminator,
        passcode,
        port,
        storage_path,
    };

    let server = MatterDeviceServer::new(config).await?;

    // Register simple logging handlers
    server.set_on_off_handler(|on| {
        println!("[onoff] → {}", if on { "ON" } else { "OFF" });
    });
    server.set_level_handler(|level| {
        println!("[level] → {level}");
    });
    server.set_color_temp_handler(|mireds| {
        println!("[color_temp] → {mireds} mireds");
    });
    server.set_thermostat_handler(|celsius| {
        println!("[thermostat] setpoint → {celsius:.1}°C");
    });

    out.raw(&format!("QR code:      {}", server.qr_code()));
    out.raw(&format!("Pairing code: {}", server.pairing_code()));
    out.raw(&format!(
        "Listening on UDP:{port} as '{device_name}' (VID={vendor_id:#06x} PID={product_id:#06x})"
    ));
    out.raw("Press Ctrl-C to stop.");

    // Block until Ctrl-C
    let srv_handle = {
        let server = server;
        tokio::spawn(async move { server.start().await })
    };

    tokio::signal::ctrl_c().await?;
    println!("\nShutting down…");
    srv_handle.abort();

    Ok(())
}
