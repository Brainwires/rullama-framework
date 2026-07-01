use crate::cli::ThermostatAction;
use crate::output::Output;
use anyhow::Result;
use rullama_homeauto::matter::clusters::thermostat;
use rullama_homeauto::{AttributeValue, MatterController, MatterDevice};
use std::path::Path;

pub async fn run(action: ThermostatAction, fabric_dir: &Path, out: &Output) -> Result<()> {
    match action {
        ThermostatAction::Setpoint {
            node_id,
            endpoint,
            celsius,
        } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;

            // Read current heating setpoint (in 0.01°C units) to compute the delta.
            let current = ctrl
                .read_attr(
                    &device,
                    endpoint,
                    thermostat::CLUSTER_ID,
                    thermostat::ATTR_OCCUPIED_HEATING_SETPOINT,
                )
                .await?;
            let current_raw = match current {
                AttributeValue::I16(n) => n,
                _ => 0,
            };

            let desired_raw = (celsius * 100.0) as i16;
            // SetpointRaiseLower amount is in 0.1°C steps, so divide by 10.
            let delta_tenths = (desired_raw - current_raw) / 10;
            if delta_tenths > i8::MAX as i16 || delta_tenths < i8::MIN as i16 {
                anyhow::bail!(
                    "setpoint delta {:.1}°C exceeds single-command range (±12.7°C); \
                     current={:.1}°C, desired={celsius:.1}°C",
                    delta_tenths as f32 / 10.0,
                    current_raw as f32 / 100.0
                );
            }
            let tlv = thermostat::setpoint_raise_lower_tlv(0 /* Heat */, delta_tenths as i8);
            ctrl.invoke(
                &device,
                endpoint,
                thermostat::CLUSTER_ID,
                thermostat::CMD_SET_SETPOINT_RAISE_LOWER,
                &tlv,
            )
            .await?;
            out.ok(&format!(
                "node_id={node_id} ep={endpoint} heating setpoint={celsius:.1}°C"
            ));
        }
        ThermostatAction::Read { node_id, endpoint } => {
            let (ctrl, device) = get_ctrl_and_device(fabric_dir, node_id).await?;
            let local_temp = ctrl
                .read_attr(
                    &device,
                    endpoint,
                    thermostat::CLUSTER_ID,
                    thermostat::ATTR_LOCAL_TEMP,
                )
                .await?;
            let heating_sp = ctrl
                .read_attr(
                    &device,
                    endpoint,
                    thermostat::CLUSTER_ID,
                    thermostat::ATTR_OCCUPIED_HEATING_SETPOINT,
                )
                .await?;

            let raw_to_c = |v: AttributeValue| -> f32 {
                match v {
                    AttributeValue::I16(n) => n as f32 / 100.0,
                    _ => 0.0,
                }
            };

            let local_c = raw_to_c(local_temp);
            let sp_c = raw_to_c(heating_sp);

            if out.json {
                println!(
                    "{{\"node_id\":{node_id},\"endpoint\":{endpoint},\
                     \"local_temp_c\":{local_c:.2},\"heating_setpoint_c\":{sp_c:.2}}}"
                );
            } else {
                println!(
                    "node_id={node_id} ep={endpoint}  local={local_c:.1}°C  \
                     heating_setpoint={sp_c:.1}°C"
                );
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
