use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::Channel;
use brainwires_signal_channel::config::SignalConfig;
use brainwires_signal_channel::event_handler::SignalEventHandler;
use brainwires_signal_channel::gateway_client::GatewayClient;
use brainwires_signal_channel::mcp_server::SignalMcpServer;
use brainwires_signal_channel::signal::SignalChannel;

/// Brainwires Signal Channel Adapter
#[derive(Parser)]
#[command(name = "brainclaw-mcp-signal")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Signal channel adapter for BrainClaw via signal-cli REST API")]
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
    /// Start the Signal adapter.
    ///
    /// Requires a running signal-cli daemon: `signal-cli -a <number> daemon --http <host>:<port>`
    Serve {
        /// Base URL of the signal-cli REST API daemon.
        #[arg(long, default_value = "http://127.0.0.1:8080", env = "SIGNAL_API_URL")]
        api_url: String,

        /// The bot's own Signal phone number in E.164 format (e.g. "+14155552671").
        #[arg(long, env = "SIGNAL_PHONE_NUMBER")]
        phone_number: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// In group chats, only respond when @mentioned or pattern matched.
        /// Direct messages always respond.
        #[arg(long, default_value_t = false, env = "GROUP_MENTION_REQUIRED")]
        group_mention_required: bool,

        /// The bot's display name for @mention detection (e.g. "mybot").
        #[arg(long, env = "BOT_NAME")]
        bot_name: Option<String>,

        /// Keyword patterns (comma-separated) that trigger a response in groups.
        #[arg(long, env = "MENTION_PATTERNS", value_delimiter = ',')]
        mention_patterns: Vec<String>,

        /// Allowed sender phone numbers (comma-separated). Empty = all.
        #[arg(long, env = "SENDER_ALLOWLIST", value_delimiter = ',')]
        sender_allowlist: Vec<String>,

        /// Allowed group IDs (comma-separated, base64). Empty = all.
        #[arg(long, env = "GROUP_ALLOWLIST", value_delimiter = ',')]
        group_allowlist: Vec<String>,

        /// Polling interval in milliseconds (used if WebSocket is unavailable).
        #[arg(long, default_value_t = 2000, env = "POLL_INTERVAL_MS")]
        poll_interval_ms: u64,

        /// Also start the MCP server on stdio.
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
            api_url,
            phone_number,
            gateway_url,
            gateway_token,
            group_mention_required,
            bot_name,
            mention_patterns,
            sender_allowlist,
            group_allowlist,
            poll_interval_ms,
            mcp,
        }) => {
            let config = SignalConfig {
                api_url,
                phone_number,
                gateway_url,
                gateway_token,
                group_mention_required,
                bot_name,
                mention_patterns,
                sender_allowlist,
                group_allowlist,
                poll_interval_ms,
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

async fn run_adapter(config: SignalConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!("Starting Brainwires Signal adapter");

    let signal_channel = Arc::new(SignalChannel::new(
        config.api_url.clone(),
        config.phone_number.clone(),
    ));

    let capabilities = signal_channel.capabilities();
    let (event_tx, event_rx) = mpsc::channel(512);

    // Optionally start MCP server on stdio
    if enable_mcp {
        let mcp_channel = Arc::clone(&signal_channel);
        tokio::spawn(async move {
            if let Err(e) = SignalMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("MCP server error: {:#}", e);
            }
        });
    }

    // Connect to the brainwires-gateway
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&signal_channel);
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
                tracing::info!("Running in Signal-only mode (no gateway)");
                let mut rx = event_rx;
                while rx.recv().await.is_some() {}
            }
        }
    });

    // Start Signal event handler (tries WebSocket first, falls back to polling)
    let event_handler = SignalEventHandler::new(
        &config.api_url,
        Arc::clone(&signal_channel),
        event_tx,
        config.phone_number,
        config.group_mention_required,
        config.bot_name,
        config.mention_patterns,
        config.sender_allowlist,
        config.group_allowlist,
        config.poll_interval_ms,
    );
    event_handler.run().await?;

    Ok(())
}

fn show_version_info() {
    println!("brainclaw-mcp-signal v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Channel Type:      signal");
    println!("Backend:           signal-cli REST API (bbernhard/signal-cli-rest-api)");
    println!("Receive Mode:      WebSocket push (with polling fallback)");
    println!("Gateway Default:   ws://127.0.0.1:18789/ws");
    println!();
    println!("Prerequisites:");
    println!("  signal-cli -a +<number> daemon --http 127.0.0.1:8080");
    println!("  (or: docker run -p 8080:8080 bbernhard/signal-cli-rest-api)");
    println!();
    println!("MCP Tools:");
    println!("  send_message     — Send a message to a phone number or group");
    println!("  add_reaction     — Add emoji reaction to a message");
}
