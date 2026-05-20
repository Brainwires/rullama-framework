use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_mattermost_channel::config::MattermostConfig;
use brainwires_mattermost_channel::event_handler::MattermostEventHandler;
use brainwires_mattermost_channel::gateway_client::GatewayClient;
use brainwires_mattermost_channel::mattermost::MattermostChannel;
use brainwires_mattermost_channel::mcp_server::MattermostMcpServer;
use brainwires_network::channels::Channel;

/// Brainwires Mattermost Channel Adapter
#[derive(Parser)]
#[command(name = "brainclaw-mcp-mattermost")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Mattermost channel adapter for the Brainwires gateway — also serves as an MCP tool server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
// reason: clap subcommand enums naturally have one large `Serve` variant
// alongside zero-sized informational variants like `Version`; boxing each
// CLI argument set adds noise without runtime benefit.
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Start the Mattermost adapter.
    /// Connects to Mattermost via WebSocket and the brainwires-gateway, forwarding events bidirectionally.
    Serve {
        /// Mattermost server base URL (e.g. "https://mattermost.example.com").
        #[arg(long, env = "MATTERMOST_SERVER_URL")]
        server_url: String,

        /// Mattermost personal access token.
        #[arg(long, env = "MATTERMOST_ACCESS_TOKEN")]
        access_token: String,

        /// The bot's Mattermost user ID (used to filter self-messages and add reactions).
        #[arg(long, env = "MATTERMOST_BOT_USER_ID")]
        bot_user_id: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// Optional team ID to scope channel operations.
        #[arg(long, env = "MATTERMOST_TEAM_ID")]
        team_id: Option<String>,

        /// In public/private channels, only respond when @mentioned.
        /// DMs always respond regardless.
        #[arg(long, default_value_t = false, env = "GROUP_MENTION_REQUIRED")]
        group_mention_required: bool,

        /// The bot's username for @mention detection (e.g. "mybot").
        #[arg(long, env = "BOT_USERNAME")]
        bot_username: Option<String>,

        /// Keyword patterns (comma-separated) that trigger a response in channels.
        /// Only active when `--group-mention-required` is set.
        #[arg(long, env = "MENTION_PATTERNS", value_delimiter = ',')]
        mention_patterns: Vec<String>,

        /// Channel IDs to include (comma-separated). Empty = all subscribed channels.
        #[arg(long, env = "CHANNEL_ALLOWLIST", value_delimiter = ',')]
        channel_allowlist: Vec<String>,

        /// Also start the MCP server on stdio (for direct tool access).
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Show version and system information.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => {
            show_version_info();
        }
        Some(Commands::Serve {
            server_url,
            access_token,
            bot_user_id,
            gateway_url,
            gateway_token,
            team_id,
            group_mention_required,
            bot_username,
            mention_patterns,
            channel_allowlist,
            mcp,
        }) => {
            let config = MattermostConfig {
                server_url,
                access_token,
                bot_user_id,
                gateway_url,
                gateway_token,
                team_id,
                group_mention_required,
                bot_username,
                mention_patterns,
                channel_allowlist,
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

async fn run_adapter(config: MattermostConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Mattermost adapter");

    // Build WebSocket URL for Mattermost event stream.
    let ws_url = config
        .server_url
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + "/api/v4/websocket";

    // Create the MattermostChannel adapter.
    let mm_channel = Arc::new(MattermostChannel::new(
        config.server_url.clone(),
        config.access_token.clone(),
        config.bot_user_id.clone(),
    ));

    // Determine capabilities.
    let capabilities = mm_channel.capabilities();

    // Create event channel.
    let (event_tx, event_rx) = mpsc::channel(512);

    // Optionally start MCP server on stdio.
    if enable_mcp {
        let mcp_channel = Arc::clone(&mm_channel);
        tokio::spawn(async move {
            if let Err(e) = MattermostMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to the brainwires-gateway.
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&mm_channel);
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
                tracing::info!("Running in Mattermost-only mode (no gateway)");
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Mattermost WebSocket event handler (blocking).
    tracing::info!("Mattermost WebSocket event handler starting...");
    let event_handler = MattermostEventHandler::new(
        ws_url,
        config.access_token,
        event_tx,
        Arc::clone(&mm_channel),
        config.bot_user_id,
        config.group_mention_required,
        config.bot_username,
        config.mention_patterns,
        config.channel_allowlist,
    );
    event_handler.run().await?;

    Ok(())
}

fn show_version_info() {
    println!("brainclaw-mcp-mattermost v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Channel Type:      mattermost");
    println!("Connection Mode:   WebSocket (/api/v4/websocket)");
    println!("Gateway Default:   ws://127.0.0.1:18789/ws");
    println!();
    println!("MCP Tools:");
    println!("  send_message     — Send a post to a Mattermost channel");
    println!("  edit_message     — Edit a previously sent post");
    println!("  delete_message   — Delete a post");
    println!("  get_history      — Fetch message history");
    println!("  add_reaction     — Add emoji reaction");
}
