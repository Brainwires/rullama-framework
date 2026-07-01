use anyhow::Result;
use rullama_homeauto::zwave::{
    CommandClass, ZWaveController, ZWaveSerialController, command_class::switch_binary_set,
};
/// Example: List Z-Wave nodes and toggle a binary switch.
///
/// Usage:
/// ```bash
/// cargo run --example zwave_nodes --features zwave -- /dev/ttyUSB0 [node_id]
/// ```
///
/// Connects to a Z-Wave USB stick (Aeotec Z-Stick Gen5+, etc.), lists all known nodes,
/// and optionally toggles the binary switch on `node_id`.
use std::env;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let port = env::args().nth(1).unwrap_or_else(|| "/dev/ttyUSB0".into());
    let toggle_node: Option<u8> = env::args().nth(2).and_then(|s| s.parse().ok());

    info!("Opening Z-Wave controller on {port}");
    let controller = ZWaveSerialController::new(&port, 115_200);
    controller.start().await?;

    let nodes = controller.nodes().await?;
    info!("Known nodes ({}):", nodes.len());
    for node in &nodes {
        info!(
            "  Node {:03} — {:?} (listening={})",
            node.node_id, node.kind, node.is_listening
        );
    }

    if let Some(node_id) = toggle_node {
        info!("Toggling binary switch on node {node_id}…");
        // Turn on
        controller
            .send_cc(
                node_id,
                CommandClass::SwitchBinary,
                &switch_binary_set(true)[1..],
            )
            .await?;
        info!("Sent ON to node {node_id}");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        // Turn off
        controller
            .send_cc(
                node_id,
                CommandClass::SwitchBinary,
                &switch_binary_set(false)[1..],
            )
            .await?;
        info!("Sent OFF to node {node_id}");
    }

    controller.stop().await?;
    Ok(())
}
