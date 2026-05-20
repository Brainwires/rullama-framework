use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;
use brainwires_telegram_channel::config::TelegramConfig;
use brainwires_telegram_channel::event_handler;
use brainwires_telegram_channel::gateway_client::GatewayClient;
use brainwires_telegram_channel::mcp_server::TelegramMcpServer;
use brainwires_telegram_channel::telegram::TelegramChannel;

/// Brainwires Telegram Channel Adapter
#[derive(Parser)]
#[command(name = "brainwires-telegram")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Telegram channel adapter for the Brainwires gateway — also serves as an MCP tool server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Telegram adapter (default mode).
    /// Connects to Telegram and the brainwires-gateway, forwarding events bidirectionally.
    Serve {
        /// Telegram bot token. Can also be set via TELEGRAM_BOT_TOKEN env var.
        /// Obtain from @BotFather on Telegram.
        #[arg(long, env = "TELEGRAM_BOT_TOKEN")]
        telegram_token: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// In group/supergroup chats, only respond when @mentioned.
        /// Private chats always respond.
        #[arg(long, default_value_t = false, env = "GROUP_MENTION_REQUIRED")]
        group_mention_required: bool,

        /// The bot's @username (without @) for mention detection in groups.
        #[arg(long, env = "BOT_USERNAME")]
        bot_username: Option<String>,

        /// Keyword patterns (comma-separated) that trigger a response in groups
        /// even without an explicit @mention. Only active when
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
            telegram_token,
            gateway_url,
            gateway_token,
            group_mention_required,
            bot_username,
            mention_patterns,
            mcp,
        }) => {
            let config = TelegramConfig {
                telegram_token,
                gateway_url,
                gateway_token,
                group_mention_required,
                bot_username,
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

async fn run_adapter(config: TelegramConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Telegram adapter");

    // Create event channel
    let (event_tx, event_rx) = mpsc::channel(512);

    // Build teloxide bot
    let bot = teloxide::Bot::new(&config.telegram_token);

    // Create the TelegramChannel
    let telegram_channel = Arc::new(TelegramChannel::new(bot.clone()));

    // Determine capabilities
    let capabilities = telegram_channel.capabilities();

    // Optionally start MCP server
    if enable_mcp {
        let mcp_channel = Arc::clone(&telegram_channel);
        tokio::spawn(async move {
            if let Err(e) = TelegramMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to gateway
    let gw_token = config.gateway_token.unwrap_or_default();
    let gw_channel = Arc::clone(&telegram_channel);
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
                tracing::info!("Running in Telegram-only mode (no gateway)");
                // Drain events so the sender doesn't block
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Telegram bot dispatcher (blocking)
    tracing::info!("Telegram bot starting...");
    // config.gateway_token has already been partially moved; reconstruct with remaining fields.
    let filter_config = TelegramConfig {
        telegram_token: config.telegram_token,
        gateway_url: config.gateway_url,
        gateway_token: None, // already moved above; not used by dispatcher
        group_mention_required: config.group_mention_required,
        bot_username: config.bot_username,
        mention_patterns: config.mention_patterns,
    };
    event_handler::run_dispatcher(bot, event_tx, filter_config).await;

    Ok(())
}

fn show_version_info() {
    println!("brainwires-telegram v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("System Information:");
    println!("  Build Date:      {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:      {}", env!("GIT_COMMIT_HASH"));
    println!();
    println!("Channel Type:      telegram");
    println!("Telegram Library:  teloxide 0.13");
    println!("Gateway Default:   ws://127.0.0.1:18789/ws");
    println!();
    println!("MCP Tools:");
    println!("  send_message     — Send a message to a Telegram chat");
    println!("  edit_message     — Edit a previously sent message");
    println!("  delete_message   — Delete a message");
    println!("  send_typing      — Show typing indicator");
    println!("  add_reaction     — Add emoji reaction");
}
