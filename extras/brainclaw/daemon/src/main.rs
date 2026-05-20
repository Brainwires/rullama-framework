use anyhow::Result;
use clap::{Parser, Subcommand};

use brainclaw::doctor::{self, DoctorArgs};
use brainclaw::onboard::{self, OnboardArgs};
use brainclaw::{BrainClaw, BrainClawConfig};

/// BrainClaw — personal AI assistant daemon
#[derive(Parser)]
#[command(name = "brainclaw")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Personal AI assistant daemon built on the Brainwires Framework")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to configuration file
    #[arg(long, global = true)]
    config: Option<String>,

    /// Host address to bind to
    #[arg(long, global = true)]
    host: Option<String>,

    /// Port to listen on
    #[arg(long, global = true)]
    port: Option<u16>,

    /// AI provider (anthropic, openai, google, groq, ollama, etc.)
    #[arg(long, global = true)]
    provider: Option<String>,

    /// Model name
    #[arg(long, global = true)]
    model: Option<String>,

    /// API key for the provider
    #[arg(long, global = true)]
    api_key: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the BrainClaw daemon (default)
    Serve,
    /// Show version information
    Version,
    /// Validate the configuration file
    ConfigCheck,
    /// DM pairing administration (approve/reject peer DMs).
    #[command(subcommand)]
    Pairing(PairingCmd),
    /// Run diagnostic checks across all subsystems.
    Doctor(DoctorArgs),
    /// Interactive setup wizard — writes a ready-to-use `brainclaw.toml`.
    Onboard(OnboardArgs),
    /// Manage Gmail Pub/Sub watches (register, status, reset cursor).
    #[cfg(feature = "email-push")]
    #[command(subcommand)]
    GmailWatch(GmailWatchCmd),
}

/// `brainclaw gmail-watch ...` subcommands.
#[cfg(feature = "email-push")]
#[derive(Subcommand)]
enum GmailWatchCmd {
    /// Register (or re-register) the Gmail watch for a configured account.
    Register {
        /// Email address of the watched mailbox. Must appear in
        /// `gmail_push.accounts`.
        #[arg(long)]
        account: String,
    },
    /// Print the current status of each configured Gmail watch.
    Status,
    /// Reset the persisted history cursor for an account so the next
    /// push replays from the beginning of the available history window.
    ResetCursor {
        /// Email address to reset.
        #[arg(long)]
        account: String,
    },
}

#[derive(Subcommand)]
enum PairingCmd {
    /// List pending pairing codes.
    Pending,
    /// List approved peers.
    List,
    /// Approve a pending pairing code.
    Approve {
        /// The 6-digit code.
        code: String,
    },
    /// Reject (discard) a pending pairing code.
    Reject {
        /// The 6-digit code.
        code: String,
    },
    /// Revoke a previously-approved peer.
    Revoke {
        /// Channel name (e.g. `discord`).
        channel: String,
        /// Platform user id.
        user: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => {
            show_version();
        }
        Some(Commands::ConfigCheck) => {
            config_check(&cli)?;
        }
        Some(Commands::Pairing(ref cmd)) => {
            pairing_cmd(&cli, cmd).await?;
        }
        Some(Commands::Doctor(ref args)) => {
            let (_results, exit) = doctor::run(cli.config.as_deref(), args).await?;
            if exit != 0 {
                std::process::exit(exit);
            }
        }
        Some(Commands::Onboard(ref args)) => {
            // Allow `--config` at the top-level to stand in for the per-subcommand flag.
            let mut args = args.clone();
            if args.config.is_none() {
                args.config = cli.config.clone();
            }
            onboard::run(&args).await?;
        }
        #[cfg(feature = "email-push")]
        Some(Commands::GmailWatch(ref cmd)) => {
            gmail_watch_cmd(&cli, cmd).await?;
        }
        Some(Commands::Serve) | None => {
            serve(cli).await?;
        }
    }

    Ok(())
}

#[cfg(feature = "email-push")]
async fn gmail_watch_cmd(cli: &Cli, cmd: &GmailWatchCmd) -> Result<()> {
    use brainwires_gateway::gmail_push::{GmailCursorStore, register_watch_and_seed};

    let config = load_config(cli)?;
    if !config.gmail_push.enabled {
        anyhow::bail!("gmail_push.enabled = false; edit brainclaw.toml first");
    }

    let cursor_path = match config.gmail_push.cursor_store.clone() {
        Some(p) => p,
        None => dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("HOME not set; can't resolve cursor store path"))?
            .join(".brainclaw")
            .join("gmail_cursor.json"),
    };
    let cursors = GmailCursorStore::load(&cursor_path).await?;

    match cmd {
        GmailWatchCmd::Register { account } => {
            let acct = config
                .gmail_push
                .accounts
                .iter()
                .find(|a| a.email_address.eq_ignore_ascii_case(account))
                .ok_or_else(|| anyhow::anyhow!("no gmail_push account matches {account}"))?;
            let handler = build_handler_from_acct(acct)?;
            let resp = register_watch_and_seed(&handler, &acct.email_address, &cursors).await?;
            println!(
                "registered: {} (history_id={}, expires={})",
                acct.email_address, resp.history_id, resp.expiration
            );
        }
        GmailWatchCmd::Status => {
            if config.gmail_push.accounts.is_empty() {
                println!("(no accounts configured)");
                return Ok(());
            }
            println!("cursor store: {}", cursor_path.display());
            for acct in &config.gmail_push.accounts {
                let cursor = cursors
                    .get(&acct.email_address)
                    .await
                    .map(|h| h.to_string())
                    .unwrap_or_else(|| "(none)".to_string());
                let token_src = match (&acct.oauth_token_env, &acct.oauth_token) {
                    (Some(env), _) => format!("env:{env}"),
                    (None, Some(_)) => "inline".to_string(),
                    _ => "missing".to_string(),
                };
                println!(
                    "  {}  topic={}  cursor={}  token={}",
                    acct.email_address, acct.topic_name, cursor, token_src
                );
            }
        }
        GmailWatchCmd::ResetCursor { account } => {
            // Set cursor to 0 so the next push reads the full available
            // history window. (Gmail caps this at ~30 days server-side,
            // so this is safe even for very old mailboxes.)
            cursors.put(account, 0).await?;
            println!("cursor reset for {account}");
        }
    }
    Ok(())
}

#[cfg(feature = "email-push")]
fn build_handler_from_acct(
    acct: &brainclaw::GmailAccountConfig,
) -> Result<brainwires_tools::gmail_push::GmailPushHandler> {
    use brainwires_tools::gmail_push::{GmailPushConfig, GmailPushHandler};
    let token = match (&acct.oauth_token_env, &acct.oauth_token) {
        (Some(env), _) => {
            std::env::var(env).map_err(|_| anyhow::anyhow!("oauth_token_env '{env}' is not set"))?
        }
        (None, Some(inline)) => inline.clone(),
        _ => anyhow::bail!("account {} has no oauth token source", acct.email_address),
    };
    Ok(GmailPushHandler::new(GmailPushConfig {
        project_id: acct.project_id.clone(),
        topic_name: acct.topic_name.clone(),
        push_audience: acct.push_audience.clone(),
        watched_label_ids: if acct.watched_label_ids.is_empty() {
            vec!["INBOX".to_string()]
        } else {
            acct.watched_label_ids.clone()
        },
        oauth_token: token,
        gmail_base_url: None,
    }))
}

/// Run a `brainclaw pairing ...` subcommand against the local gateway's
/// admin API.
async fn pairing_cmd(cli: &Cli, cmd: &PairingCmd) -> Result<()> {
    let config = load_config(cli)?;
    let host = cli
        .host
        .clone()
        .unwrap_or_else(|| config.gateway.host.clone());
    let port = cli.port.unwrap_or(config.gateway.port);
    // The daemon listens on `127.0.0.1` by default; when host is `0.0.0.0`
    // we still address it via loopback on this machine.
    let connect_host = if host == "0.0.0.0" {
        "127.0.0.1".to_string()
    } else {
        host
    };
    let base = format!("http://{connect_host}:{port}/admin/pairing");
    let token = config.security.admin_token.clone();
    let client = reqwest::Client::new();

    let auth = |req: reqwest::RequestBuilder| match &token {
        Some(t) => req.bearer_auth(t),
        None => req,
    };

    match cmd {
        PairingCmd::Pending => {
            let resp = auth(client.get(format!("{base}/pending")))
                .send()
                .await?
                .error_for_status()?;
            let codes: Vec<serde_json::Value> = resp.json().await?;
            if codes.is_empty() {
                println!("(no pending codes)");
            } else {
                for pc in codes {
                    println!(
                        "{}  {}:{}  {} (expires {})",
                        pc.get("code").and_then(|v| v.as_str()).unwrap_or("?"),
                        pc.get("channel").and_then(|v| v.as_str()).unwrap_or("?"),
                        pc.get("user_id").and_then(|v| v.as_str()).unwrap_or("?"),
                        pc.get("peer_display")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        pc.get("expires_at").and_then(|v| v.as_str()).unwrap_or(""),
                    );
                }
            }
        }
        PairingCmd::List => {
            let resp = auth(client.get(format!("{base}/approved")))
                .send()
                .await?
                .error_for_status()?;
            let peers: Vec<String> = resp.json().await?;
            if peers.is_empty() {
                println!("(no approved peers)");
            } else {
                for p in peers {
                    println!("{p}");
                }
            }
        }
        PairingCmd::Approve { code } => {
            let resp = auth(client.post(format!("{base}/approve")))
                .json(&serde_json::json!({ "code": code }))
                .send()
                .await?
                .error_for_status()?;
            let body: serde_json::Value = resp.json().await?;
            if body
                .get("approved")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                println!(
                    "approved {}:{}",
                    body.get("channel").and_then(|v| v.as_str()).unwrap_or("?"),
                    body.get("user_id").and_then(|v| v.as_str()).unwrap_or("?"),
                );
            } else {
                let reason = body
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                println!("not approved: {reason}");
            }
        }
        PairingCmd::Reject { code } => {
            let resp = auth(client.post(format!("{base}/reject")))
                .json(&serde_json::json!({ "code": code }))
                .send()
                .await?
                .error_for_status()?;
            let body: serde_json::Value = resp.json().await?;
            let rejected = body
                .get("rejected")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if rejected {
                println!("rejected code {code}");
            } else {
                println!("code {code} not found");
            }
        }
        PairingCmd::Revoke { channel, user } => {
            let resp = auth(client.post(format!("{base}/revoke")))
                .json(&serde_json::json!({ "channel": channel, "user_id": user }))
                .send()
                .await?
                .error_for_status()?;
            let body: serde_json::Value = resp.json().await?;
            if body
                .get("revoked")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                println!("revoked {channel}:{user}");
            } else {
                println!("revoke failed for {channel}:{user}");
            }
        }
    }

    Ok(())
}

fn show_version() {
    println!("brainclaw v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Components:");
    println!("  Gateway:     brainwires-gateway (WebSocket + webhook)");
    println!("  Agents:      brainwires-agent (ChatAgent with tool loops)");
    println!("  Providers:   brainwires-providers (Anthropic, OpenAI, Google, etc.)");
    println!("  Tools:       brainwires-tools (bash, files, git, search, web, validation)");
    println!("  Skills:      brainwires-skills (SKILL.md-based extensibility)");
    println!("  Channels:    brainwires-channels (Discord, Telegram, Slack, etc.)");
}

fn config_check(cli: &Cli) -> Result<()> {
    let config = load_config(cli)?;
    config.validate()?;
    println!("Configuration is valid.");
    println!();
    println!("  Provider:     {}", config.provider.default_provider);
    println!(
        "  Model:        {}",
        config
            .provider
            .default_model
            .as_deref()
            .unwrap_or("(provider default)")
    );
    println!(
        "  Listen:       {}:{}",
        config.gateway.host, config.gateway.port
    );
    println!("  Persona:      {}", config.persona.name);
    println!("  Tools:        {} enabled", config.tools.enabled.len());
    println!(
        "  Skills:       {}",
        if config.skills.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    Ok(())
}

async fn serve(cli: Cli) -> Result<()> {
    let mut config = load_config(&cli)?;

    // CLI overrides
    if let Some(ref host) = cli.host {
        config.gateway.host = host.clone();
    }
    if let Some(port) = cli.port {
        config.gateway.port = port;
    }
    if let Some(ref provider) = cli.provider {
        config.provider.default_provider = provider.clone();
    }
    if let Some(ref model) = cli.model {
        config.provider.default_model = Some(model.clone());
    }
    if let Some(ref api_key) = cli.api_key {
        config.provider.api_key = Some(api_key.clone());
    }

    config.validate()?;

    let app = BrainClaw::new(config);
    app.run().await
}

fn load_config(cli: &Cli) -> Result<BrainClawConfig> {
    if let Some(ref path) = cli.config {
        BrainClawConfig::load(path)
    } else {
        BrainClawConfig::load_or_default()
    }
}
