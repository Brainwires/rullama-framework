//! Google Chat channel adapter — binary entrypoint.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;

use brainwires_google_chat_channel::config::GoogleChatConfig;
use brainwires_google_chat_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_google_chat_channel::google_chat::GoogleChatChannel;
use brainwires_google_chat_channel::mcp_server::GoogleChatMcpServer;
use brainwires_google_chat_channel::oauth::{CHAT_BOT_SCOPE, TokenMinter};
use brainwires_google_chat_channel::webhook::{WebhookState, serve};

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-google-chat",
    version = env!("CARGO_PKG_VERSION"),
    about = "Google Chat channel adapter for the Brainwires gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run webhook server + gateway connection (default).
    Serve {
        #[arg(long, env = "GOOGLE_CHAT_PROJECT_ID")]
        project_id: String,
        #[arg(long, env = "GOOGLE_CHAT_AUDIENCE")]
        audience: String,
        #[arg(long, env = "GOOGLE_CHAT_SERVICE_ACCOUNT_KEY")]
        service_account_key: String,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, default_value = "0.0.0.0:9101", env = "LISTEN_ADDR")]
        listen_addr: String,
        /// Expose the MCP stdio server alongside the webhook/gateway.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP tool server (no webhook, no gateway).
    Mcp {
        #[arg(long, env = "GOOGLE_CHAT_SERVICE_ACCOUNT_KEY")]
        service_account_key: String,
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
            println!("brainclaw-mcp-google-chat v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp {
            service_account_key,
        }) => {
            let minter = Arc::new(TokenMinter::from_key_path(
                &service_account_key,
                CHAT_BOT_SCOPE,
            )?);
            let chan = Arc::new(GoogleChatChannel::new(minter));
            GoogleChatMcpServer::serve_stdio(chan).await
        }
        Some(Command::Serve {
            project_id,
            audience,
            service_account_key,
            gateway_url,
            gateway_token,
            listen_addr,
            mcp,
        }) => {
            let config = GoogleChatConfig {
                project_id,
                audience: audience.clone(),
                service_account_key_path: service_account_key.clone(),
                gateway_url: gateway_url.clone(),
                gateway_token,
                listen_addr: listen_addr.clone(),
            };
            run_adapter(config, mcp).await
        }
        None => {
            eprintln!("No subcommand. Run with --help for usage.");
            std::process::exit(1);
        }
    }
}

async fn run_adapter(config: GoogleChatConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!(project_id = %config.project_id, "starting google-chat adapter");

    let minter = Arc::new(TokenMinter::from_key_path(
        &config.service_account_key_path,
        CHAT_BOT_SCOPE,
    )?);
    let channel = Arc::new(GoogleChatChannel::new(Arc::clone(&minter)));
    let caps = channel.capabilities();
    let (event_tx, event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = GoogleChatMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {e:#}");
            }
        });
    }

    // Gateway reconnect loop.
    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let (gw_tx, mut gw_rx) = mpsc::channel(512);

    // Bridge: fan every event into both the gateway loop and a local
    // drain so that a gateway outage doesn't block webhook handlers.
    tokio::spawn(async move {
        let mut rx = event_rx;
        while let Some(ev) = rx.recv().await {
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
                    // Drain the shared channel into a per-attempt channel.
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
                    // Pull rx back to keep looping.
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

    // Webhook server on the main task.
    let state = WebhookState::new(config.audience, event_tx);
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
