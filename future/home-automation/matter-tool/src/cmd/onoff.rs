use crate::cli::OnoffAction;
use crate::output::Output;
use anyhow::Result;
use brainwires_homeauto::matter::clusters::on_off;
use brainwires_homeauto::{AttributeValue, MatterController};
use std::path::Path;

pub async fn run(action: OnoffAction, fabric_dir: &Path, out: &Output) -> Result<()> {
    match action {
        OnoffAction::On { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            ctrl.on_off(&device, endpoint, true).await?;
            out.ok(&format!("node_id={node_id} ep={endpoint} → ON"));
        }
        OnoffAction::Off { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            ctrl.on_off(&device, endpoint, false).await?;
            out.ok(&format!("node_id={node_id} ep={endpoint} → OFF"));
        }
        OnoffAction::Toggle { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            ctrl.invoke(
                &device,
                endpoint,
                on_off::CLUSTER_ID,
                on_off::CMD_TOGGLE,
                &[],
            )
            .await?;
            out.ok(&format!("node_id={node_id} ep={endpoint} → TOGGLE"));
        }
        OnoffAction::Read { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            let val = ctrl
                .read_attr(&device, endpoint, on_off::CLUSTER_ID, on_off::ATTR_ON_OFF)
                .await?;
            let state = match &val {
                AttributeValue::Bool(true) => "on",
                AttributeValue::Bool(false) => "off",
                _ => "unknown",
            };
            if out.json {
                out.kv(
                    "on_off",
                    &format!("{}", matches!(val, AttributeValue::Bool(true))),
                );
            } else {
                println!("node_id={node_id} ep={endpoint} on/off={state}");
            }
        }
    }
    Ok(())
}

async fn get_ctrl_and_device(
    fabric_dir: &Path,
    node_id: u64,
) -> Result<(
    MatterController,
    brainwires_homeauto::MatterDevice,
)> {
    let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
    let devices = ctrl.devices().await?;
    let device = devices
        .into_iter()
        .find(|d| d.node_id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node_id={node_id} not found in fabric"))?;
    Ok((ctrl, device))
}
