//! `brainwires-home` — dial-home daemon for the chat PWA.
//!
//! Subcommands:
//! * (default) / `serve`  — long-running signaling + pairing daemon.
//! * `pair`               — print a QR + 6-digit code, wait for the PWA
//!   to claim + confirm, persist a `device_token` to
//!   `~/.brainwires/home/devices.json`, exit.

use anyhow::{Context, Result};
use brainwires_home::{
    HomeServer, HomeServerBuilder,
    pairing::{self, PairingState},
};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(version, about = "Brainwires chat-PWA dial-home daemon")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Bind address for the signaling server.
    #[arg(
        long,
        env = "BRAINWIRES_HOME_BIND",
        default_value = "127.0.0.1:7878",
        global = true
    )]
    bind: SocketAddr,

    /// Add an exact-match origin to the CORS allow-list. Repeatable.
    #[arg(long = "cors-origin", value_name = "URL", global = true)]
    cors_origin: Vec<String>,

    /// Allow any origin. **Dev only.**
    #[arg(
        long = "cors-permissive",
        env = "BRAINWIRES_HOME_CORS_PERMISSIVE",
        default_value_t = false,
        global = true
    )]
    cors_permissive: bool,

    /// Cloudflare Calls TURN key id (M7).
    #[arg(
        long = "cf-turn-key-id",
        env = "CF_TURN_KEY_ID",
        value_name = "ID",
        global = true
    )]
    cf_turn_key_id: Option<String>,

    /// Cloudflare **Calls** API token (M7).
    #[arg(
        long = "cf-turn-token",
        env = "CF_TURN_TOKEN",
        value_name = "TOKEN",
        global = true
    )]
    cf_turn_token: Option<String>,

    /// Lifetime of minted TURN credentials in seconds.
    #[arg(
        long = "turn-ttl",
        env = "TURN_TTL",
        default_value_t = 600,
        global = true
    )]
    turn_ttl: u32,

    /// Pre-provisioned Cloudflare Access service-token client id (M8).
    #[arg(
        long = "cf-access-client-id",
        env = "CF_ACCESS_CLIENT_ID",
        value_name = "ID",
        global = true
    )]
    cf_access_client_id: Option<String>,

    /// Pre-provisioned Cloudflare Access service-token client secret (M8).
    #[arg(
        long = "cf-access-client-secret",
        env = "CF_ACCESS_CLIENT_SECRET",
        value_name = "SECRET",
        global = true
    )]
    cf_access_client_secret: Option<String>,

    /// Override the directory used for pairing state. Defaults to
    /// `$HOME/.brainwires/home`.
    #[arg(
        long = "state-dir",
        env = "BRAINWIRES_HOME_STATE_DIR",
        value_name = "DIR",
        global = true
    )]
    state_dir: Option<PathBuf>,

    /// Path to the sync database file for cross-device data sync.
    /// Defaults to `$STATE_DIR/sync.db`.
    #[arg(
        long = "sync-db",
        env = "BRAINWIRES_HOME_SYNC_DB",
        value_name = "PATH",
        global = true
    )]
    sync_db: Option<PathBuf>,

    /// `tracing-subscriber` env-filter directive.
    #[arg(
        long,
        env = "RUST_LOG",
        default_value = "brainwires_home=info,info",
        global = true
    )]
    log: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the long-lived signaling + pairing daemon. (Default.)
    Serve,
    /// Mint a single pairing offer, print the QR + 6-digit confirm code,
    /// wait up to 5 minutes for the PWA to claim + confirm, exit.
    Pair {
        /// Tunnel URL the PWA should reach the daemon at. The QR encodes
        /// this so the PWA knows where to send `/pair/claim`.
        #[arg(
            long = "tunnel-url",
            env = "BRAINWIRES_HOME_TUNNEL_URL",
            value_name = "URL"
        )]
        tunnel_url: String,

        /// Friendly name to print alongside the QR. Cosmetic — the PWA
        /// supplies its own `device_name` on claim.
        #[arg(long = "label", default_value = "this device")]
        label: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt().with_env_filter(&cli.log).init();

    match cli.command.as_ref().unwrap_or(&Command::Serve) {
        Command::Serve => run_serve(cli).await,
        Command::Pair { tunnel_url, label } => {
            let url = tunnel_url.clone();
            let lbl = label.clone();
            run_pair(cli, url, lbl).await
        }
    }
}

/// Build a [`PairingState`] from the CLI flags.
fn build_pairing(cli: &Cli) -> Result<PairingState> {
    let dir = cli
        .state_dir
        .clone()
        .or_else(pairing::default_state_dir)
        .context("could not resolve state directory: set --state-dir or HOME")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("create_dir_all {}", dir.display()))?;
    let identity = pairing::load_or_create_identity(&dir.join("identity.json"))?;
    let cf_access = match (
        cli.cf_access_client_id.as_ref(),
        cli.cf_access_client_secret.as_ref(),
    ) {
        (Some(id), Some(secret)) => Some(pairing::CfAccessConfig {
            client_id: id.clone(),
            client_secret: secret.clone(),
        }),
        (Some(_), None) | (None, Some(_)) => {
            tracing::warn!(
                "--cf-access-client-id and --cf-access-client-secret must be set together; \
                 ignoring CF Access for /pair/confirm"
            );
            None
        }
        (None, None) => None,
    };
    Ok(PairingState::new(
        dir.join("devices.json"),
        cf_access,
        identity.peer_pubkey,
    ))
}

/// Apply the CORS / TURN flags to a freshly-constructed builder. Returns
/// the builder by value so the caller can chain further setters.
fn apply_common(mut b: HomeServerBuilder, cli: &Cli) -> HomeServerBuilder {
    if cli.cors_permissive {
        if !cli.cors_origin.is_empty() {
            tracing::warn!(
                "--cors-permissive supersedes --cors-origin; ignoring the explicit allow-list"
            );
        }
        tracing::warn!(
            "CORS is permissive (any origin); intended for dev only — \
             use --cors-origin <URL> in production"
        );
        b = b.cors_permissive();
    } else {
        for origin in &cli.cors_origin {
            b = b.cors_allow_origin(origin.clone());
        }
        if cli.cors_origin.is_empty() {
            tracing::info!(
                "no --cors-origin flags; defaulting to chat-PWA dev origins (localhost:8080 etc.)"
            );
        } else {
            tracing::info!(origins = ?cli.cors_origin, "CORS allow-list configured");
        }
    }

    match (cli.cf_turn_key_id.as_deref(), cli.cf_turn_token.as_deref()) {
        (Some(key_id), Some(token)) => {
            b = b
                .with_cloudflare_turn(key_id, token)
                .with_turn_ttl(cli.turn_ttl);
            tracing::info!(
                ttl_secs = cli.turn_ttl,
                "Cloudflare Calls TURN minting enabled"
            );
        }
        (Some(_), None) | (None, Some(_)) => {
            tracing::warn!(
                "--cf-turn-key-id and --cf-turn-token must be set together; falling back to STUN-only"
            );
        }
        (None, None) => {
            tracing::info!("no TURN config; serving STUN-only ICE servers");
        }
    }

    b
}

async fn run_serve(cli: Cli) -> Result<()> {
    let bind = cli.bind;
    let pairing_state = build_pairing(&cli)?;
    tracing::info!(
        peer_pubkey = %short_fp(pairing_state.peer_pubkey()),
        "loaded home identity"
    );

    let mut builder = HomeServer::builder().bind(bind).with_pairing(pairing_state);
    builder = apply_common(builder, &cli);

    let sync_path = cli.sync_db.unwrap_or_else(|| {
        let dir = cli
            .state_dir
            .clone()
            .or_else(pairing::default_state_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        dir.join("sync.db")
    });
    builder = builder.with_sync(&sync_path)?;
    tracing::info!(path = %sync_path.display(), "sync store enabled");

    let server = builder.build()?;
    tracing::info!(addr = %bind, "brainwires-home: starting");
    server.serve().await
}

async fn run_pair(cli: Cli, tunnel_url: String, label: String) -> Result<()> {
    let pairing_state = build_pairing(&cli)?;
    let peer_pubkey = pairing_state.peer_pubkey().to_string();
    let fingerprint = short_fp(&peer_pubkey);

    let builder = HomeServer::builder()
        .bind(cli.bind)
        .with_pairing(pairing_state.clone());
    let builder = apply_common(builder, &cli);
    let server = builder.build()?;

    let offer = pairing_state.create_offer().await;
    let qr_url = offer.qr_url(&tunnel_url, &fingerprint);

    println!();
    println!("┌─────────────────────────────────────────────────────────");
    println!("│  Pairing {label}");
    println!("│  Tunnel : {tunnel_url}");
    println!("│  Home FP: {fingerprint}");
    println!("│");
    println!("│  Scan this URL with the PWA (or paste it into the");
    println!("│  pairing dialog under Settings → Home agent):");
    println!("│");
    println!("│    {qr_url}");
    println!("│");
    println!("│  Then enter this 6-digit confirm code in the PWA:");
    println!("│");
    println!("│             {}", offer.confirm_code);
    println!("│");
    println!("│  (offer expires in 5 minutes)");
    println!("└─────────────────────────────────────────────────────────");
    println!();

    // Run the server in the background so /pair/* answers HTTP calls.
    let serve_handle = tokio::spawn(async move {
        if let Err(e) = server.serve().await {
            tracing::error!(error = %e, "pair-mode server exited");
        }
    });

    // Poll devices.json: a successful confirm appends a record. We don't
    // need access to the in-memory pending map for this — once the
    // ledger grows we know the pair landed.
    let deadline = std::time::Instant::now() + pairing::OFFER_TTL;
    let baseline_count = pairing_state.read_devices().unwrap_or_default().len();
    loop {
        if std::time::Instant::now() >= deadline {
            tracing::warn!("pair: offer expired without a successful confirm");
            serve_handle.abort();
            anyhow::bail!("pairing timed out — run `brainwires-home pair` again");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        let devices = pairing_state.read_devices().unwrap_or_default();
        if devices.len() > baseline_count {
            let last = devices.last().expect("just appended");
            println!();
            println!("Pair confirmed:");
            println!("  device_name : {}", last.device_name);
            println!("  device_pubkey: {}", short_fp(&last.device_pubkey));
            println!("  granted_at  : {}", last.granted_at);
            println!();
            println!("Wrote {}", pairing_state.devices_path().display());
            serve_handle.abort();
            return Ok(());
        }
    }
}

fn short_fp(hex: &str) -> String {
    hex.chars().take(8).collect()
}
