use crate::output::Output;
use anyhow::Result;
use rullama_homeauto::MatterController;
use std::path::Path;

pub async fn run_invoke(
    node_id: u64,
    endpoint: u16,
    cluster_id: u32,
    command_id: u32,
    payload_hex: Option<String>,
    fabric_dir: &Path,
    out: &Output,
) -> Result<()> {
    let tlv = if let Some(hex) = payload_hex {
        matter_tool::parse_tlv_hex(&hex)?
    } else {
        vec![]
    };

    let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
    let devices = ctrl.devices().await?;
    let device = devices
        .into_iter()
        .find(|d| d.node_id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node_id={node_id} not found in fabric"))?;

    ctrl.invoke(&device, endpoint, cluster_id, command_id, &tlv)
        .await?;
    out.ok(&format!(
        "invoke node_id={node_id} ep={endpoint} cluster={cluster_id:#010x} cmd={command_id:#010x} OK"
    ));
    Ok(())
}

pub async fn run_read(
    node_id: u64,
    endpoint: u16,
    cluster_id: u32,
    attribute_id: u32,
    fabric_dir: &Path,
    out: &Output,
) -> Result<()> {
    let ctrl = MatterController::new("matter-tool", fabric_dir).await?;
    let devices = ctrl.devices().await?;
    let device = devices
        .into_iter()
        .find(|d| d.node_id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node_id={node_id} not found in fabric"))?;

    let val = ctrl
        .read_attr(&device, endpoint, cluster_id, attribute_id)
        .await?;
    out.attribute(node_id, endpoint, cluster_id, attribute_id, &val);
    Ok(())
}
