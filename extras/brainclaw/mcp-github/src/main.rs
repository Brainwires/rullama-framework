mod config;
mod gateway_client;
mod github;
mod mcp_server;
mod webhook;

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use brainwires_network::channels::{Channel, ChannelEvent};

use config::GitHubConfig;
use gateway_client::GatewayClient;
use github::GitHubChannel;
use mcp_server::GitHubMcpServer;

/// Brainwires GitHub Channel Adapter
#[derive(Parser)]
#[command(name = "brainclaw-mcp-github")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "GitHub channel adapter for the Brainwires gateway — receives webhooks, forwards events, and serves as an MCP tool server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the GitHub adapter (default mode).
    ///
    /// Opens a webhook HTTP server for inbound events and connects to the
    /// brainwires-gateway over WebSocket. Optionally also starts an MCP server
    /// on stdio for direct tool access.
    Serve {
        /// GitHub Personal Access Token or App installation token.
        /// Can also be set via GITHUB_TOKEN.
        #[arg(long, env = "GITHUB_TOKEN")]
        github_token: String,

        /// Webhook secret for HMAC-SHA256 signature verification.
        /// Must match the secret configured in GitHub webhook settings.
        /// Can also be set via GITHUB_WEBHOOK_SECRET.
        #[arg(long, env = "GITHUB_WEBHOOK_SECRET")]
        webhook_secret: Option<String>,

        /// Local address for the webhook HTTP server.
        #[arg(long, default_value = "127.0.0.1:9000", env = "WEBHOOK_ADDR")]
        listen_addr: String,

        /// WebSocket URL of the brainwires-gateway.
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,

        /// Optional auth token for the gateway handshake.
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,

        /// Skip webhook secret requirement (INSECURE — development only).
        #[arg(long, default_value_t = false, env = "INSECURE_DEV_WEBHOOK")]
        insecure_dev_webhook: bool,

        /// Comma-separated list of repos to accept events from (e.g. `owner/repo`).
        /// Empty means all repos.
        #[arg(long, env = "GITHUB_REPOS", value_delimiter = ',')]
        repos: Vec<String>,

        /// GitHub API base URL (override for GitHub Enterprise Server).
        #[arg(long, default_value = "https://api.github.com", env = "GITHUB_API_URL")]
        api_url: String,

        /// Also start the MCP server on stdio for direct tool access.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },

    /// Show version and capability information.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => show_version_info(),

        Some(Commands::Serve {
            github_token,
            webhook_secret,
            listen_addr,
            gateway_url,
            gateway_token,
            repos,
            api_url,
            mcp,
            insecure_dev_webhook,
        }) => {
            // Require webhook secret unless explicitly opted out
            if webhook_secret.is_none() && !insecure_dev_webhook {
                anyhow::bail!(
                    "webhook_secret is required for production use. \
                     Set --webhook-secret / GITHUB_WEBHOOK_SECRET, \
                     or pass --insecure-dev-webhook to skip (development only)."
                );
            }

            let config = Arc::new(GitHubConfig {
                github_token: github_token.clone(),
                webhook_secret,
                insecure_dev_webhook,
                listen_addr,
                repos,
                gateway_url,
                gateway_token,
                api_url: api_url.clone(),
                ..Default::default()
            });

            run_adapter(config, github_token, api_url, mcp).await?;
        }

        None => {
            eprintln!("No subcommand given. Use `serve` or `version`.");
            eprintln!("Run with --help for usage.");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_adapter(
    config: Arc<GitHubConfig>,
    token: String,
    api_url: String,
    enable_mcp: bool,
) -> Result<()> {
    tracing::info!("Starting Brainwires GitHub adapter");

    // Create GitHub channel
    let github_channel = Arc::new(
        GitHubChannel::new(&token, &api_url)
            .map_err(|e| anyhow::anyhow!("Failed to build GitHub client: {e}"))?,
    );

    let capabilities = github_channel.capabilities();

    // Channel for webhook → gateway pipeline
    let (event_tx, event_rx) = mpsc::channel::<ChannelEvent>(512);
    // Channel for normalized webhook messages (before wrapping in ChannelEvent)
    let (msg_tx, mut msg_rx) = mpsc::channel(512);

    // Start webhook receiver
    let wh_config = Arc::clone(&config);
    tokio::spawn(async move {
        if let Err(e) = webhook::serve(wh_config, msg_tx).await {
            tracing::error!("Webhook server error: {e:#}");
        }
    });

    // Forward ChannelMessage → ChannelEvent for the gateway
    tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            let event = ChannelEvent::MessageReceived(msg);
            if event_tx.send(event).await.is_err() {
                break;
            }
        }
    });

    // Optionally start MCP server on stdio
    if enable_mcp {
        let mcp_channel = Arc::clone(&github_channel);
        let mcp_api_url = api_url.clone();
        let mcp_token = token.clone();
        tokio::spawn(async move {
            if let Err(e) = GitHubMcpServer::serve_stdio(mcp_channel, mcp_api_url, mcp_token).await
            {
                tracing::error!("MCP server error: {e:#}");
            }
        });
    }

    // Connect to gateway
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_url = config.gateway_url.clone();
    let gw_channel = Arc::clone(&github_channel);

    match GatewayClient::connect(&gw_url, &gw_token, capabilities).await {
        Ok(gw_client) => {
            if let Err(e) = gw_client.run(event_rx, gw_channel).await {
                tracing::error!("Gateway client error: {e:#}");
            }
        }
        Err(e) => {
            tracing::error!("Failed to connect to gateway: {e:#}");
            tracing::info!("Running in webhook-only mode (no gateway)");
            // Drain events so the sender doesn't block
            let mut rx = event_rx;
            while rx.recv().await.is_some() {}
        }
    }

    Ok(())
}

fn show_version_info() {
    println!("brainclaw-mcp-github v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Channel Type:  github");
    println!("Webhook:       Axum HTTP server (HMAC-SHA256 verified)");
    println!("Gateway:       WebSocket (brainwires-gateway)");
    println!("Gateway URL:   ws://127.0.0.1:18789/ws  (default)");
    println!("Webhook Port:  9000  (default)");
    println!();
    println!("MCP Tools:");
    println!("  post_comment          — Post a comment on an issue or PR");
    println!("  edit_comment          — Edit an existing comment");
    println!("  delete_comment        — Delete a comment");
    println!("  get_comments          — Fetch comment history");
    println!("  create_issue          — Open a new issue");
    println!("  close_issue           — Close an issue");
    println!("  add_labels            — Add labels to issue or PR");
    println!("  create_pull_request   — Open a new pull request");
    println!("  merge_pull_request    — Merge a pull request");
    println!("  add_reaction          — React to a comment");
    println!();
    println!("Supported events:");
    println!("  issue_comment  issues  pull_request  pull_request_review_comment");
}
