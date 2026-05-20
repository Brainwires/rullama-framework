//! Feishu / Lark adapter — binary entrypoint.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;

use brainwires_feishu_channel::config::FeishuConfig;
use brainwires_feishu_channel::feishu::FeishuChannel;
use brainwires_feishu_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_feishu_channel::mcp_server::FeishuMcpServer;
use brainwires_feishu_channel::oauth::TenantTokenMinter;
use brainwires_feishu_channel::webhook::{WebhookState, serve};

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-feishu",
    version = env!("CARGO_PKG_VERSION"),
    about = "Feishu / Lark channel adapter"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run webhook server + gateway connection.
    Serve {
        #[arg(long, env = "FEISHU_APP_ID")]
        app_id: String,
        #[arg(long, env = "FEISHU_APP_SECRET")]
        app_secret: String,
        #[arg(long, env = "FEISHU_VERIFICATION_TOKEN")]
        verification_token: String,
        #[arg(long, env = "FEISHU_ENCRYPT_KEY")]
        encrypt_key: Option<String>,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, default_value = "0.0.0.0:9105", env = "LISTEN_ADDR")]
        listen_addr: String,
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP tool server (egress only).
    Mcp {
        #[arg(long, env = "FEISHU_APP_ID")]
        app_id: String,
        #[arg(long, env = "FEISHU_APP_SECRET")]
        app_secret: String,
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
            println!("brainclaw-mcp-feishu v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp { app_id, app_secret }) => {
            let minter = Arc::new(TenantTokenMinter::new(app_id, app_secret));
            let chan = Arc::new(FeishuChannel::new(minter));
            FeishuMcpServer::serve_stdio(chan).await
        }
        Some(Command::Serve {
            app_id,
            app_secret,
            verification_token,
            encrypt_key,
            gateway_url,
            gateway_token,
            listen_addr,
            mcp,
        }) => {
            if encrypt_key.is_some() {
                tracing::warn!(
                    "FEISHU_ENCRYPT_KEY is set but AES decryption is not wired at MVP — \
                     disable event encryption in the Feishu console, or expect 401."
                );
            }
            let cfg = FeishuConfig {
                app_id,
                app_secret,
                verification_token,
                encrypt_key,
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

async fn run_adapter(config: FeishuConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("starting feishu adapter");

    let minter = Arc::new(TenantTokenMinter::new(
        config.app_id.clone(),
        config.app_secret.clone(),
    ));
    let channel = Arc::new(FeishuChannel::new(Arc::clone(&minter)));
    let caps = channel.capabilities();
    let (event_tx, event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = FeishuMcpServer::serve_stdio(mcp_channel).await {
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

    let state = WebhookState::new(config.verification_token, event_tx);
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
