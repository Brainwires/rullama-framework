//! Microsoft Teams channel adapter — entrypoint.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;

use brainwires_teams_channel::config::TeamsConfig;
use brainwires_teams_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_teams_channel::jwt::BotFrameworkVerifier;
use brainwires_teams_channel::mcp_server::TeamsMcpServer;
use brainwires_teams_channel::oauth::BotTokenMinter;
use brainwires_teams_channel::teams::{ServiceUrlStore, TeamsChannel};
use brainwires_teams_channel::webhook::{WebhookState, serve};

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-teams",
    version = env!("CARGO_PKG_VERSION"),
    about = "Microsoft Teams channel adapter for the Brainwires gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run webhook server + gateway connection (default).
    Serve {
        #[arg(long, env = "TEAMS_APP_ID")]
        app_id: String,
        #[arg(long, env = "TEAMS_APP_PASSWORD")]
        app_password: String,
        #[arg(long, default_value = "common", env = "TEAMS_TENANT_ID")]
        tenant_id: String,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        #[arg(long, default_value = "0.0.0.0:9102", env = "LISTEN_ADDR")]
        listen_addr: String,
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP server only.
    Mcp {
        #[arg(long, env = "TEAMS_APP_ID")]
        app_id: String,
        #[arg(long, env = "TEAMS_APP_PASSWORD")]
        app_password: String,
        #[arg(long, default_value = "common", env = "TEAMS_TENANT_ID")]
        tenant_id: String,
    },
    /// Version info.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Version) => {
            println!("brainclaw-mcp-teams v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp {
            app_id,
            app_password,
            tenant_id,
        }) => {
            let minter = Arc::new(BotTokenMinter::new(&app_id, &app_password, &tenant_id));
            let urls = Arc::new(ServiceUrlStore::new());
            let channel = Arc::new(TeamsChannel::new(app_id, minter, urls));
            TeamsMcpServer::serve_stdio(channel).await
        }
        Some(Command::Serve {
            app_id,
            app_password,
            tenant_id,
            gateway_url,
            gateway_token,
            listen_addr,
            mcp,
        }) => {
            let config = TeamsConfig {
                app_id,
                app_password,
                tenant_id,
                gateway_url,
                gateway_token,
                listen_addr,
            };
            run_adapter(config, mcp).await
        }
        None => {
            eprintln!("No subcommand. Run with --help for usage.");
            std::process::exit(1);
        }
    }
}

async fn run_adapter(config: TeamsConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!(app_id = %config.app_id, tenant = %config.tenant_id, "starting teams adapter");

    let verifier = Arc::new(BotFrameworkVerifier::new(config.app_id.clone()));
    let minter = Arc::new(BotTokenMinter::new(
        config.app_id.clone(),
        config.app_password.clone(),
        config.tenant_id.clone(),
    ));
    let service_urls = Arc::new(ServiceUrlStore::new());
    let channel = Arc::new(TeamsChannel::new(
        config.app_id.clone(),
        Arc::clone(&minter),
        Arc::clone(&service_urls),
    ));
    let caps = channel.capabilities();

    let (event_tx, event_rx) = mpsc::channel(512);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = TeamsMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("mcp: {e:#}");
            }
        });
    }

    // Bridge the webhook -> gateway loop so gateway outages don't block.
    let (gw_tx, mut gw_rx) = mpsc::channel(512);
    tokio::spawn(async move {
        let mut rx = event_rx;
        while let Some(ev) = rx.recv().await {
            if gw_tx.send(ev).await.is_err() {
                break;
            }
        }
    });

    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
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
                        tracing::error!("gateway: {e:#}");
                    }
                    match forwarder.await {
                        Ok(back) => gw_rx = back,
                        Err(e) => {
                            tracing::error!("forwarder join: {e}");
                            return;
                        }
                    }
                }
                Err(e) => tracing::error!("gateway connect failed: {e:#} (retry in {:?})", backoff),
            }
            tokio::time::sleep(backoff).await;
            backoff = backoff_next(backoff);
        }
    });

    let state = WebhookState {
        verifier,
        service_urls,
        event_tx,
    };
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
    let ctrl_c = async { tokio::signal::ctrl_c().await.unwrap() };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .unwrap()
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
