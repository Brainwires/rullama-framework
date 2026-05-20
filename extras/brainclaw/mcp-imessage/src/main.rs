//! iMessage / BlueBubbles channel adapter — binary entrypoint.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::{mpsc, watch};

use brainwires_network::channels::Channel;

use brainwires_imessage_channel::config::{ImessageConfig, default_state_dir};
use brainwires_imessage_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_imessage_channel::imessage::ImessageChannel;
use brainwires_imessage_channel::ingress::Ingress;
use brainwires_imessage_channel::mcp_server::ImessageMcpServer;

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-imessage",
    version = env!("CARGO_PKG_VERSION"),
    about = "iMessage (BlueBubbles) channel adapter for the Brainwires gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run polling ingress + gateway connection (default).
    Serve {
        #[arg(long, env = "BB_SERVER_URL")]
        server_url: String,
        #[arg(long, env = "BB_PASSWORD")]
        password: String,
        #[arg(long, env = "BB_POLL_INTERVAL_SECS", default_value_t = 2)]
        poll_interval_secs: u64,
        #[arg(long, env = "BB_CHATS", default_value = "")]
        chats: String,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, env = "BB_STATE_DIR")]
        state_dir: Option<PathBuf>,
        /// Expose the MCP stdio server alongside the ingress loop.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP tool server (no polling, no gateway).
    Mcp {
        #[arg(long, env = "BB_SERVER_URL")]
        server_url: String,
        #[arg(long, env = "BB_PASSWORD")]
        password: String,
    },
    /// Print version.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Version) => {
            println!("brainclaw-mcp-imessage v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp {
            server_url,
            password,
        }) => {
            let chan = Arc::new(ImessageChannel::new(server_url, password));
            ImessageMcpServer::serve_stdio(chan).await
        }
        Some(Command::Serve {
            server_url,
            password,
            poll_interval_secs,
            chats,
            gateway_url,
            gateway_token,
            state_dir,
            mcp,
        }) => {
            let chat_guids: Vec<String> = chats
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let cfg = ImessageConfig {
                server_url,
                password,
                poll_interval_secs,
                chat_guids,
                gateway_url,
                gateway_token,
                state_dir: state_dir.unwrap_or_else(default_state_dir),
            };
            run_adapter(cfg, mcp).await
        }
        None => {
            eprintln!("No subcommand. Run with --help for usage.");
            std::process::exit(1);
        }
    }
}

async fn run_adapter(config: ImessageConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!(url = %config.server_url, "starting imessage adapter");

    let channel = Arc::new(ImessageChannel::new(
        config.server_url.clone(),
        config.password.clone(),
    ));
    let caps = channel.capabilities();
    // Ingress → gateway channel.
    let (event_tx, mut event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = ImessageMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {e:#}");
            }
        });
    }

    // Gateway reconnect loop.
    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let (gw_tx, mut gw_rx) = mpsc::channel(512);

    // Bridge: fan every event into the gateway task so a gateway outage
    // doesn't block the polling loop.
    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            if gw_tx.send(ev).await.is_err() {
                tracing::warn!("gateway bridge receiver closed");
                break;
            }
        }
    });

    tokio::spawn(async move {
        let mut backoff = Duration::from_millis(1_000);
        loop {
            match GatewayClient::connect(&gw_url, &gw_token, caps).await {
                Ok(gw) => {
                    backoff = Duration::from_millis(1_000);
                    let (local_tx, local_rx) = mpsc::channel(512);
                    let forwarder = tokio::spawn(async move {
                        while let Some(ev) = gw_rx.recv().await {
                            if local_tx.send(ev).await.is_err() {
                                break;
                            }
                        }
                        gw_rx
                    });
                    let chan = Arc::clone(&gw_channel);
                    if let Err(e) = gw.run(local_rx, chan).await {
                        tracing::error!("gateway run: {e:#}");
                    }
                    match forwarder.await {
                        Ok(rx_back) => gw_rx = rx_back,
                        Err(e) => {
                            tracing::error!("forwarder join: {e}");
                            return;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("gateway connect failed: {e:#} (retry in {:?})", backoff);
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = backoff_next(backoff);
        }
    });

    // Ingress polling loop + shutdown signal.
    let cursor_path = config.state_dir.join("imessage.json");
    let ingress = Ingress::new(
        Arc::clone(&channel),
        config.chat_guids.clone(),
        Duration::from_secs(config.poll_interval_secs.max(1)),
        cursor_path,
    )?;
    let (sd_tx, sd_rx) = watch::channel(false);
    let shutdown = shutdown_signal();
    tokio::select! {
        r = ingress.run(event_tx, sd_rx) => r,
        _ = shutdown => {
            tracing::info!("shutdown signal received");
            let _ = sd_tx.send(true);
            Ok(())
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install ctrl-c handler");
    };

    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
