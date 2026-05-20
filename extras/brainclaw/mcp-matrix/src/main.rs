use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use matrix_sdk::{Client, config::SyncSettings};
use tokio::sync::mpsc;

use brainwires_matrix_channel::config::MatrixConfig;
use brainwires_matrix_channel::event_handler::register_handlers;
use brainwires_matrix_channel::gateway_client::GatewayClient;
use brainwires_matrix_channel::matrix::MatrixChannel;
use brainwires_matrix_channel::mcp_server::MatrixMcpServer;
use brainwires_network::channels::Channel;

/// Brainwires Matrix Channel Adapter
#[derive(Parser)]
#[command(name = "brainclaw-mcp-matrix")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Matrix channel adapter for the Brainwires gateway")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Matrix adapter (default mode).
    Serve {
        /// Matrix homeserver URL (e.g. "https://matrix.org").
        #[arg(long, env = "MATRIX_HOMESERVER_URL")]
        homeserver_url: String,

        /// Matrix username (localpart, e.g. "mybot").
        #[arg(long, env = "MATRIX_USERNAME")]
        username: String,

        /// Matrix account password.
        #[arg(long, env = "MATRIX_PASSWORD")]
        password: String,

        /// Device display name shown in session list.
        #[arg(long, default_value = "BrainClaw", env = "MATRIX_DEVICE_NAME")]
        device_name: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// Also start the MCP server on stdio.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Show version information.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => {
            println!("brainclaw-mcp-matrix v{}", env!("CARGO_PKG_VERSION"));
        }
        Some(Commands::Serve {
            homeserver_url,
            username,
            password,
            device_name,
            gateway_url,
            gateway_token,
            mcp,
        }) => {
            let config = MatrixConfig {
                homeserver_url,
                username,
                password,
                device_name,
                gateway_url,
                gateway_token,
            };
            run_adapter(config, mcp).await?;
        }
        None => {
            eprintln!("No subcommand given. Use `serve` to start or `version` for info.");
            eprintln!("Run with --help for usage details.");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_adapter(config: MatrixConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Matrix adapter");

    // Build Matrix client (in-memory store — no rusqlite conflict with workspace)
    let client = Client::builder()
        .homeserver_url(&config.homeserver_url)
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build Matrix client: {}", e))?;

    // Login
    client
        .matrix_auth()
        .login_username(&config.username, &config.password)
        .device_id(config.device_name.as_str())
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Matrix login failed: {}", e))?;

    tracing::info!(username = %config.username, "Matrix login successful");

    // Perform an initial sync to load room state before registering event handlers
    client
        .sync_once(SyncSettings::default())
        .await
        .map_err(|e| anyhow::anyhow!("Initial Matrix sync failed: {}", e))?;

    let client = Arc::new(client);
    let channel = Arc::new(MatrixChannel::new(Arc::clone(&client)));
    let capabilities = channel.capabilities();

    // Event channel — Matrix events → gateway
    let (event_tx, event_rx) = mpsc::channel(512);

    // Register SDK event handlers
    register_handlers(&client, event_tx);

    // Optionally start MCP server on stdio
    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = MatrixMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to gateway
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let gw_url = config.gateway_url.clone();

    tokio::spawn(async move {
        match GatewayClient::connect(&gw_url, &gw_token, capabilities).await {
            Ok(gw_client) => {
                if let Err(e) = gw_client.run(event_rx, gw_channel).await {
                    tracing::error!("Gateway client error: {:#}", e);
                }
            }
            Err(e) => {
                tracing::error!("Failed to connect to gateway: {:#}", e);
                tracing::info!("Running in sync-only mode (no gateway)");
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Matrix sync loop (this blocks until shutdown)
    tracing::info!("Starting Matrix sync loop");
    let sync_client = Arc::clone(&client);
    tokio::select! {
        result = sync_client.sync(SyncSettings::default()) => {
            if let Err(e) = result {
                tracing::error!("Matrix sync error: {:#}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Shutting down Matrix adapter");
        }
    }

    Ok(())
}
