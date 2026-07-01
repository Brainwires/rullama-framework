use crate::cli::LevelAction;
use crate::output::Output;
use anyhow::Result;
use rullama_homeauto::matter::clusters::level_control;
use rullama_homeauto::{AttributeValue, MatterController, MatterDevice};
use std::path::Path;

pub async fn run(action: LevelAction, fabric_dir: &Path, out: &Output) -> Result<()> {
    match action {
        LevelAction::Set {
            node_id,
            endpoint,
            level,
            transition,
        } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            let tlv = level_control::move_to_level_tlv(level, Some(transition));
            ctrl.invoke(
                &device,
                endpoint,
                level_control::CLUSTER_ID,
                level_control::CMD_MOVE_TO_LEVEL_WITH_ON_OFF,
                &tlv,
            )
            .await?;
            out.ok(&format!(
                "node_id={node_id} ep={endpoint} level={level} transition={transition}"
            ));
        }
        LevelAction::Read { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            let val = ctrl
                .read_attr(
                    &device,
                    endpoint,
                    level_control::CLUSTER_ID,
                    level_control::ATTR_CURRENT_LEVEL,
                )
                .await?;
            if out.json {
                let n = match &val {
                    AttributeValue::U8(n) => *n as u32,
                    _ => 0,
                };
                out.kv("level", &n.to_string());
            } else {
                println!("node_id={node_id} ep={endpoint} level={val}");
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
