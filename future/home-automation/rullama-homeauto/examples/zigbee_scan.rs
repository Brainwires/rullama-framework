/// Example: Scan for Zigbee devices using an EZSP coordinator.
///
/// Usage:
/// ```bash
/// cargo run --example zigbee_scan --features zigbee -- /dev/ttyUSB0
/// ```
///
/// Connects to a Silicon Labs EZSP coordinator (e.g. Sonoff Zigbee 3.0 USB Dongle Plus),
/// opens a 60-second join window, and prints any devices that join.
///
/// For a TI Z-Stack coordinator (CC2652, etc.), change `EzspCoordinator` to `ZnpCoordinator`.
use std::env;
use std::time::Duration;

use anyhow::Result;
use rullama_homeauto::zigbee::{EzspCoordinator, ZigbeeCoordinator};
use futures::StreamExt;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let port = env::args().nth(1).unwrap_or_else(|| "/dev/ttyUSB0".into());
    let baud = 115_200u32;

    info!("Opening EZSP coordinator on {port} @ {baud}");
    let coordinator = EzspCoordinator::new(&port, baud);
    coordinator.start().await?;

    info!("Permitting joins for 60 seconds…");
    coordinator.permit_join(60).await?;

    // Subscribe to events in background
    let mut events = coordinator.events();
    let event_task = tokio::spawn(async move {
        while let Some(event) = events.next().await {
            info!("Event: {event:?}");
        }
    });

    tokio::time::sleep(Duration::from_secs(60)).await;
    coordinator.permit_join(0).await?;

    let devices = coordinator.devices().await?;
    info!("Known devices ({}):", devices.len());
    for dev in &devices {
        info!(
            "  {:016x} ({:#06x}) — {:?}",
            dev.addr.ieee, dev.addr.nwk, dev.kind
        );
    }

    coordinator.stop().await?;
    event_task.abort();
    Ok(())
}
