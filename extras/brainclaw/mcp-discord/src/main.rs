use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serenity::Client;
use serenity::all::GatewayIntents;
use tokio::sync::mpsc;

use brainwires_discord_channel::config::DiscordConfig;
use brainwires_discord_channel::discord::DiscordChannel;
use brainwires_discord_channel::event_handler::DiscordEventHandler;
use brainwires_discord_channel::gateway_client::GatewayClient;
use brainwires_discord_channel::mcp_server::DiscordMcpServer;
use brainwires_network::channels::Channel;

/// Brainwires Discord Channel Adapter
#[derive(Parser)]
#[command(name = "brainwires-discord")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Discord channel adapter for the Brainwires gateway — also serves as an MCP tool server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Discord adapter (default mode).
    /// Connects to Discord and the brainwires-gateway, forwarding events bidirectionally.
    Serve {
        /// Discord bot token. Can also be set via DISCORD_TOKEN env var.
        #[arg(long, env = "DISCORD_TOKEN")]
        discord_token: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// Optional command prefix for the bot (e.g., "!").
        #[arg(long, env = "BOT_PREFIX")]
        bot_prefix: Option<String>,

        /// In guild channels, only respond when @mentioned (or prefix matches).
        /// DMs always get a response regardless of this setting.
        #[arg(long, default_value_t = false, env = "GROUP_MENTION_REQUIRED")]
        group_mention_required: bool,

        /// Additional keyword patterns that trigger a response in group channels.
        /// Comma-separated (e.g. "brainclaw,hey bot"). Only used when
        /// `--group-mention-required` is set.
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
            discord_token,
            gateway_url,
            gateway_token,
            bot_prefix,
            group_mention_required,
            mention_patterns,
            mcp,
        }) => {
            let config = DiscordConfig {
                discord_token,
                gateway_url,
                gateway_token,
                bot_prefix,
                group_mention_required,
                mention_patterns,
            };
            run_adapter(config, mcp).await?;
        }
        None => {
            // Default: print help
            eprintln!("No subcommand given. Use `serve` to start or `version` for info.");
            eprintln!("Run with --help for usage details.");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_adapter(config: DiscordConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Discord adapter");

    // Create event channel
    let (event_tx, event_rx) = mpsc::channel(512);

    // Build serenity client
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT
        | GatewayIntents::GUILD_MESSAGE_REACTIONS
        | GatewayIntents::GUILD_MESSAGE_TYPING;

    let event_handler = DiscordEventHandler::new(event_tx, config.clone());

    let mut client = Client::builder(&config.discord_token, intents)
        .event_handler(event_handler)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to build Discord client: {}", e))?;

    // Create the DiscordChannel from the HTTP client
    let http = client.http.clone();
    let discord_channel = Arc::new(DiscordChannel::new(http));

    // Determine capabilities
    let capabilities = discord_channel.capabilities();

    // Optionally start MCP server
    if enable_mcp {
        let mcp_channel = Arc::clone(&discord_channel);
        tokio::spawn(async move {
            if let Err(e) = DiscordMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to gateway
    let gw_token = config.gateway_token.unwrap_or_default();
    let gw_channel = Arc::clone(&discord_channel);
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
                tracing::info!("Running in Discord-only mode (no gateway)");
                // Drain events so the sender doesn't block
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Discord client (blocking)
    tracing::info!("Discord bot starting...");
    client
        .start()
        .await
        .map_err(|e| anyhow::anyhow!("Discord client error: {}", e))?;

    Ok(())
}

fn show_version_info() {
    println!("brainwires-discord v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("System Information:");
    println!("  Build Date:      {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:      {}", env!("GIT_COMMIT_HASH"));
    println!();
    println!("Channel Type:      discord");
    println!("Discord Library:   serenity 0.12");
    println!("Gateway Default:   ws://127.0.0.1:18789/ws");
    println!();
    println!("MCP Tools:");
    println!("  send_message     — Send a message to a Discord channel");
    println!("  edit_message     — Edit a previously sent message");
    println!("  delete_message   — Delete a message");
    println!("  get_history      — Fetch message history");
    println!("  list_channels    — List accessible channels");
    println!("  send_typing      — Show typing indicator");
    println!("  add_reaction     — Add emoji reaction");
}
