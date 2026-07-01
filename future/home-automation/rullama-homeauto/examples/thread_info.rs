use anyhow::Result;
use rullama_homeauto::thread::ThreadBorderRouter;
/// Example: Print Thread network topology via OTBR REST API.
///
/// Usage:
/// ```bash
/// cargo run --example thread_info --features thread -- http://192.168.1.100:8081
/// ```
///
/// Connects to an OpenThread Border Router (OTBR) REST API and prints:
/// - Node info (role, RLOC16, network name)
/// - Neighbor table
use std::env;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let otbr_url = env::args()
        .nth(1)
        .unwrap_or_else(|| "http://localhost:8081".into());

    info!("Connecting to OTBR at {otbr_url}");
    let client = ThreadBorderRouter::new(&otbr_url)?;

    let info = client.node_info().await?;
    info!("Thread Node Info:");
    info!("  Role:         {:?}", info.role);
    info!(
        "  RLOC16:       {}",
        info.rloc16.as_deref().unwrap_or("N/A")
    );
    info!(
        "  Network Name: {}",
        info.network_name.as_deref().unwrap_or("N/A")
    );
    info!(
        "  Ext Address:  {}",
        info.ext_address.as_deref().unwrap_or("N/A")
    );

    let neighbors = client.neighbors().await?;
    info!("\nNeighbors ({}):", neighbors.len());
    for n in &neighbors {
        info!(
            "  {} RLOC16={} RSSI={}",
            n.ext_address.as_deref().unwrap_or("?"),
            n.rloc16.as_deref().unwrap_or("?"),
            n.rssi.map(|r| r.to_string()).as_deref().unwrap_or("?"),
        );
    }

    let dataset = client.active_dataset().await?;
    info!(
        "\nActive Dataset (TLV hex): {}",
        &dataset.active_dataset[..dataset.active_dataset.len().min(64)]
    );

    Ok(())
}
