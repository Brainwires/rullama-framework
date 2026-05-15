use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

mod cli;
mod cmd;
mod fabric;
mod output;

use cli::{Cli, Command};
use output::Output;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };
    fmt().with_env_filter(filter).with_target(false).init();

    let fabric_dir = fabric::resolve_fabric_dir(cli.fabric_dir.as_ref());
    let out = Output::new(cli.json);

    match cli.command {
        Command::Pair { action } => cmd::pair::run(action, &fabric_dir, &out).await?,
        Command::Onoff { action } => cmd::onoff::run(action, &fabric_dir, &out).await?,
        Command::Level { action } => cmd::level::run(action, &fabric_dir, &out).await?,
        Command::Thermostat { action } => cmd::thermostat::run(action, &fabric_dir, &out).await?,
        Command::Doorlock { action } => cmd::doorlock::run(action, &fabric_dir, &out).await?,
        Command::Invoke {
            node_id,
            endpoint,
            cluster_id,
            command_id,
            payload_hex,
        } => {
            cmd::invoke::run_invoke(
                node_id,
                endpoint,
                cluster_id,
                command_id,
                payload_hex,
                &fabric_dir,
                &out,
            )
            .await?
        }
        Command::Read {
            node_id,
            endpoint,
            cluster_id,
            attribute_id,
        } => {
            cmd::invoke::run_read(
                node_id,
                endpoint,
                cluster_id,
                attribute_id,
                &fabric_dir,
                &out,
            )
            .await?
        }
        Command::Discover { timeout } => cmd::discover::run(timeout, &out).await?,
        Command::Serve {
            device_name,
            vendor_id,
            product_id,
            discriminator,
            passcode,
            port,
            storage,
        } => {
            cmd::serve::run(
                device_name,
                vendor_id,
                product_id,
                discriminator,
                passcode,
                port,
                storage,
                &out,
            )
            .await?
        }
        Command::Devices => cmd::devices::run(&fabric_dir, &out).await?,
        Command::Fabric { action } => cmd::fabric::run(action, &fabric_dir, &out).await?,
    }

    Ok(())
}
