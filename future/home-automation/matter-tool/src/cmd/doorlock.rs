use crate::cli::DoorlockAction;
use crate::output::Output;
use anyhow::Result;
use rullama_homeauto::matter::clusters::door_lock;
use rullama_homeauto::{AttributeValue, MatterController, MatterDevice};
use std::path::Path;

pub async fn run(action: DoorlockAction, fabric_dir: &Path, out: &Output) -> Result<()> {
    match action {
        DoorlockAction::Lock { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            ctrl.door_lock(&device, endpoint, true, None).await?;
            out.ok(&format!("node_id={node_id} ep={endpoint} → LOCKED"));
        }
        DoorlockAction::Unlock { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            ctrl.door_lock(&device, endpoint, false, None).await?;
            out.ok(&format!("node_id={node_id} ep={endpoint} → UNLOCKED"));
        }
        DoorlockAction::Read { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            let val = ctrl
                .read_attr(
                    &device,
                    endpoint,
                    door_lock::CLUSTER_ID,
                    door_lock::ATTR_LOCK_STATE,
                )
                .await?;
            // LockState: 0=Not Fully Locked, 1=Locked, 2=Unlocked
            let state_str = match &val {
                AttributeValue::U8(1) => "locked",
                AttributeValue::U8(2) => "unlocked",
                AttributeValue::U8(0) => "not_fully_locked",
                _ => "unknown",
            };
            if out.json {
                out.kv("lock_state", state_str);
            } else {
                println!("node_id={node_id} ep={endpoint} lock_state={state_str}");
            }
        }
    }
    Ok(())
}

async fn get_ctrl_and_device(
    fabric_dir: &Path,
    node_id: u64,
) -> Result<(MatterController, MatterDevice)> {
    let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
    let devices = ctrl.devices().await?;
    let device = devices
        .into_iter()
        .find(|d| d.node_id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node_id={node_id} not found in fabric"))?;
    Ok((ctrl, device))
}
