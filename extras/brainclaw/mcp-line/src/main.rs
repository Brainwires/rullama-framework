//! LINE channel adapter — binary entrypoint.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;

use brainwires_line_channel::config::LineConfig;
use brainwires_line_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_line_channel::line::LineChannel;
use brainwires_line_channel::mcp_server::LineMcpServer;
use brainwires_line_channel::webhook::{WebhookState, serve};

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-line",
    version = env!("CARGO_PKG_VERSION"),
    about = "LINE Messaging API channel adapter"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run webhook server + gateway connection.
    Serve {
        #[arg(long, env = "LINE_CHANNEL_SECRET")]
        channel_secret: String,
        #[arg(long, env = "LINE_CHANNEL_ACCESS_TOKEN")]
        channel_access_token: String,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, default_value = "0.0.0.0:9104", env = "LISTEN_ADDR")]
        listen_addr: String,
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP tool server.
    Mcp {
        #[arg(long, env = "LINE_CHANNEL_ACCESS_TOKEN")]
        channel_access_token: String,
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
            println!("brainclaw-mcp-line v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp {
            channel_access_token,
        }) => {
            let chan = Arc::new(LineChannel::new(channel_access_token));
            LineMcpServer::serve_stdio(chan).await
        }
        Some(Command::Serve {
            channel_secret,
            channel_access_token,
            gateway_url,
            gateway_token,
            listen_addr,
            mcp,
        }) => {
            let cfg = LineConfig {
                channel_secret,
                channel_access_token,
                gateway_url,
                gateway_token,
                listen_addr,
            };
            run_adapter(cfg, mcp).await
        }
        None => {
            eprintln!("No subcommand. Run with --help for usage.");
            std::process::exit(1);
        }
    }
}

async fn run_adapter(config: LineConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("starting line adapter");

    let channel = Arc::new(LineChannel::new(config.channel_access_token.clone()));
    let caps = channel.capabilities();
    let tokens = channel.reply_tokens();
    let (event_tx, event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = LineMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {e:#}");
            }
        });
    }

    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let (gw_tx, mut gw_rx) = mpsc::channel(512);

    tokio::spawn(async move {
        let mut rx = event_rx;
        while let Some(ev) = rx.recv().await {
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

    let state = WebhookState::new(config.channel_secret, event_tx, tokens);
    let shutdown = shutdown_signal();
    tokio::select! {
        r = serve(state, &config.listen_addr) => r,
        _ = shutdown => {
            tracing::info!("shutdown signal received");
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
