use crate::fabric;
use crate::output::Output;
use anyhow::Result;
use std::path::Path;

pub async fn run(fabric_dir: &Path, out: &Output) -> Result<()> {
    let devices = fabric::load_devices(fabric_dir).await?;
    out.devices(&devices);
    Ok(())
}
