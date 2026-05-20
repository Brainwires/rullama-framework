use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;
use brainwires_slack_channel::config::SlackConfig;
use brainwires_slack_channel::event_handler::SlackEventHandler;
use brainwires_slack_channel::gateway_client::GatewayClient;
use brainwires_slack_channel::mcp_server::SlackMcpServer;
use brainwires_slack_channel::slack::SlackChannel;

/// Brainwires Slack Channel Adapter
#[derive(Parser)]
#[command(name = "brainwires-slack")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Slack channel adapter for the Brainwires gateway — also serves as an MCP tool server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Slack adapter (default mode).
    /// Connects to Slack via Socket Mode and the brainwires-gateway, forwarding events bidirectionally.
    Serve {
        /// Slack bot token (xoxb-...). Can also be set via SLACK_BOT_TOKEN env var.
        #[arg(long, env = "SLACK_BOT_TOKEN")]
        slack_bot_token: String,

        /// Slack app-level token (xapp-...) for Socket Mode. Can also be set via SLACK_APP_TOKEN env var.
        #[arg(long, env = "SLACK_APP_TOKEN")]
        slack_app_token: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// In public/private channels, only respond when @mentioned.
        /// DMs always respond regardless.
        #[arg(long, default_value_t = false, env = "GROUP_MENTION_REQUIRED")]
        group_mention_required: bool,

        /// The bot's Slack user ID (e.g. "U0123456789") for @mention detection.
        #[arg(long, env = "BOT_USER_ID")]
        bot_user_id: Option<String>,

        /// Keyword patterns (comma-separated) that trigger a response in channels.
        /// Only active when `--group-mention-required` is set.
        #[arg(long, env = "MENTION_PATTERNS", value_delimiter = ',')]
        mention_patterns: Vec<String>,

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
            slack_bot_token,
            slack_app_token,
            gateway_url,
            gateway_token,
            group_mention_required,
            bot_user_id,
            mention_patterns,
            mcp,
        }) => {
            let config = SlackConfig {
                slack_bot_token,
                slack_app_token,
                gateway_url,
                gateway_token,
                group_mention_required,
                bot_user_id,
                mention_patterns,
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

async fn run_adapter(config: SlackConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Slack adapter");

    // Extract mention filter settings before config fields are consumed.
    let group_mention_required = config.group_mention_required;
    let bot_user_id = config.bot_user_id.clone();
    let mention_patterns = config.mention_patterns.clone();

    // Create event channel
    let (event_tx, event_rx) = mpsc::channel(512);

    // Create the SlackChannel
    let slack_channel = Arc::new(SlackChannel::new(config.slack_bot_token.clone()));

    // Determine capabilities
    let capabilities = slack_channel.capabilities();

    // Optionally start MCP server
    if enable_mcp {
        let mcp_channel = Arc::clone(&slack_channel);
        tokio::spawn(async move {
            if let Err(e) = SlackMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to gateway
    let gw_token = config.gateway_token.unwrap_or_default();
    let gw_channel = Arc::clone(&slack_channel);
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
                tracing::info!("Running in Slack-only mode (no gateway)");
                // Drain events so the sender doesn't block
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Socket Mode event handler (blocking)
    tracing::info!("Slack Socket Mode starting...");
    let event_handler = SlackEventHandler::new(
        config.slack_app_token,
        event_tx,
        Arc::clone(&slack_channel),
        group_mention_required,
        bot_user_id,
        mention_patterns,
    );
    event_handler.run(config.slack_bot_token).await?;

    Ok(())
}

fn show_version_info() {
    println!("brainwires-slack v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("System Information:");
    println!("  Build Date:      {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:      {}", env!("GIT_COMMIT_HASH"));
    println!();
    println!("Channel Type:      slack");
    println!("Connection Mode:   Socket Mode (WebSocket)");
    println!("Gateway Default:   ws://127.0.0.1:18789/ws");
    println!();
    println!("MCP Tools:");
    println!("  send_message     — Send a message to a Slack channel");
    println!("  edit_message     — Edit a previously sent message");
    println!("  delete_message   — Delete a message");
    println!("  get_history      — Fetch message history");
    println!("  add_reaction     — Add emoji reaction");
}
