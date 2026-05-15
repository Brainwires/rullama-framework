use crate::cli::FabricAction;
use crate::fabric;
use crate::output::Output;
use anyhow::Result;
use std::path::PathBuf;

pub async fn run(action: FabricAction, fabric_dir: &PathBuf, out: &Output) -> Result<()> {
    match action {
        FabricAction::Info => {
            let devices = fabric::load_devices(fabric_dir).await?;
            if out.json {
                println!(
                    "{{\"fabric_dir\":{},\"commissioned_nodes\":{}}}",
                    serde_json::to_string(&fabric_dir.display().to_string()).unwrap(),
                    devices.len()
                );
            } else {
                println!("Fabric directory: {}", fabric_dir.display());
                println!("Commissioned nodes: {}", devices.len());
            }
        }
        FabricAction::Reset => {
            if !fabric::confirm_destructive("This will wipe ALL fabric storage.") {
                out.err("Aborted — fabric NOT wiped.");
                return Ok(());
            }
            let devices_file = fabric_dir.join("devices.json");
            if devices_file.exists() {
                tokio::fs::remove_file(&devices_file).await?;
            }
            // Remove the entire fabric directory if it exists
            if fabric_dir.exists() {
                tokio::fs::remove_dir_all(fabric_dir).await?;
            }
            out.ok("Fabric storage wiped.");
        }
    }
    Ok(())
}
