use crate::cli::PairAction;
use crate::output::Output;
use anyhow::Result;
use brainwires_homeauto::MatterController;
use std::path::Path;

pub async fn run(action: PairAction, fabric_dir: &Path, out: &Output) -> Result<()> {
    match action {
        PairAction::Qr { node_id, qr_code } => {
            let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
            let device = ctrl.commission_qr(&qr_code, node_id).await?;
            out.ok(&format!(
                "Commissioned node_id={} VID={:#06x} PID={:#06x}",
                device.node_id, device.vendor_id, device.product_id
            ));
        }
        PairAction::Code {
            node_id,
            manual_code,
        } => {
            let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
            let device = ctrl.commission_code(&manual_code, node_id).await?;
            out.ok(&format!(
                "Commissioned node_id={} VID={:#06x} PID={:#06x}",
                device.node_id, device.vendor_id, device.product_id
            ));
        }
        PairAction::Ble {
            node_id: _,
            passcode: _,
            discriminator: _,
        } => {
            #[cfg(feature = "ble")]
            {
                // BLE commissioning path — feature-gated
                anyhow::bail!(
                    "BLE commissioning is not yet implemented. The matter-ble \
                     transport stack exists but is not wired into MatterController. \
                     Use mDNS/UDP commissioning (QR or manual code) instead."
                );
            }
            #[cfg(not(feature = "ble"))]
            {
                anyhow::bail!(
                    "BLE commissioning requires the 'ble' feature.\n\
                     Rebuild with: cargo build -p matter-tool --features ble"
                );
            }
        }
        PairAction::Unpair { node_id } => {
            // Remove the device from the local fabric by rewriting devices.json
            // without the target node.  No network interaction is needed.
            let devices_file = fabric_dir.join("devices.json");
            let mut devices: Vec<brainwires_homeauto::MatterDevice> =
                if devices_file.exists() {
                    let raw = tokio::fs::read_to_string(&devices_file).await?;
                    serde_json::from_str(&raw)?
                } else {
                    vec![]
                };
            let before = devices.len();
            devices.retain(|d| d.node_id != node_id);
            if devices.len() == before {
                anyhow::bail!("node_id={node_id} not found in fabric");
            }
            let json = serde_json::to_string_pretty(&devices)?;
            tokio::fs::write(&devices_file, json).await?;
            out.ok(&format!("Unpaired node_id={node_id}"));
        }
    }
    Ok(())
}
