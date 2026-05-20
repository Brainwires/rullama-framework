//! Nextcloud Talk adapter — binary entrypoint.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::{mpsc, watch};

use brainwires_network::channels::Channel;

use brainwires_nextcloud_talk_channel::config::{NextcloudConfig, default_state_dir};
use brainwires_nextcloud_talk_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_nextcloud_talk_channel::ingress::Ingress;
use brainwires_nextcloud_talk_channel::mcp_server::NextcloudTalkMcpServer;
use brainwires_nextcloud_talk_channel::nextcloud_talk::NextcloudTalkChannel;

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-nextcloud-talk",
    version = env!("CARGO_PKG_VERSION"),
    about = "Nextcloud Talk channel adapter for the Brainwires gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run polling ingress + gateway connection (default).
    Serve {
        #[arg(long, env = "NEXTCLOUD_URL")]
        server_url: String,
        #[arg(long, env = "NEXTCLOUD_USERNAME")]
        username: String,
        #[arg(long, env = "NEXTCLOUD_APP_PASSWORD")]
        app_password: String,
        #[arg(long, env = "NEXTCLOUD_ROOMS", default_value = "")]
        rooms: String,
        #[arg(long, env = "NEXTCLOUD_POLL_INTERVAL_SECS", default_value_t = 2)]
        poll_interval_secs: u64,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, env = "NEXTCLOUD_STATE_DIR")]
        state_dir: Option<PathBuf>,
        /// Expose the MCP stdio server alongside the ingress loop.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP tool server.
    Mcp {
        #[arg(long, env = "NEXTCLOUD_URL")]
        server_url: String,
        #[arg(long, env = "NEXTCLOUD_USERNAME")]
        username: String,
        #[arg(long, env = "NEXTCLOUD_APP_PASSWORD")]
        app_password: String,
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
            println!(
                "brainclaw-mcp-nextcloud-talk v{}",
                env!("CARGO_PKG_VERSION")
            );
            Ok(())
        }
        Some(Command::Mcp {
            server_url,
            username,
            app_password,
        }) => {
            let chan = Arc::new(NextcloudTalkChannel::new(
                server_url,
                username,
                app_password,
            ));
            NextcloudTalkMcpServer::serve_stdio(chan).await
        }
        Some(Command::Serve {
            server_url,
            username,
            app_password,
            rooms,
            poll_interval_secs,
            gateway_url,
            gateway_token,
            state_dir,
            mcp,
        }) => {
            let room_tokens: Vec<String> = rooms
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if room_tokens.is_empty() {
                anyhow::bail!("NEXTCLOUD_ROOMS is required — specify at least one room token");
            }
            let cfg = NextcloudConfig {
                server_url,
                username,
                app_password,
                room_tokens,
                poll_interval_secs,
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

async fn run_adapter(config: NextcloudConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!(url = %config.server_url, "starting nextcloud-talk adapter");

    let channel = Arc::new(NextcloudTalkChannel::new(
        config.server_url.clone(),
        config.username.clone(),
        config.app_password.clone(),
    ));
    let caps = channel.capabilities();
    let (event_tx, mut event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = NextcloudTalkMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {e:#}");
            }
        });
    }

    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let (gw_tx, mut gw_rx) = mpsc::channel(512);

    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            if gw_tx.send(ev).await.is_err() {
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

    let cursor_path = config.state_dir.join("nextcloud_talk.json");
    let ingress = Ingress::new(
        Arc::clone(&channel),
        config.room_tokens.clone(),
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
