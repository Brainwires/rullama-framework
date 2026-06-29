//! # brainwires-scheduler
//!
//! Local-machine MCP server for cron-based job scheduling with optional Docker sandboxing.
//!
//! ## Modes
//!
//! **stdio (default)** — for use as a Claude Code MCP server:
//! ```sh
//! brainwires-scheduler
//! ```
//!
//! **HTTP + stdio** — expose both transports simultaneously:
//! ```sh
//! brainwires-scheduler --http 127.0.0.1:3200
//! ```
//!
//! ## Register with Claude Code
//!
//! ```sh
//! claude mcp add --transport stdio brainwires-scheduler brainwires-scheduler
//! ```

mod config;
mod daemon;
mod executor;
mod job;
mod server;
mod store;

use clap::Parser;
use config::Config;
use daemon::SchedulerDaemon;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ServiceExt, transport::io::stdio};
use server::SchedulerServer;
use std::sync::Arc;
use store::JobStore;
use tokio::sync::RwLock;

#[derive(Parser)]
#[command(
    name = "brainwires-scheduler",
    version,
    about = "Local MCP server for cron job scheduling with optional Docker sandboxing"
)]
struct Cli {
    /// Directory for jobs.json and per-job logs (default: ~/.brainwires/scheduler/)
    #[arg(long)]
    jobs_dir: Option<String>,

    /// Maximum number of jobs that may run concurrently (default: 4)
    #[arg(long, default_value = "4")]
    max_concurrent: usize,

    /// Also expose an HTTP MCP endpoint at this address (e.g. 127.0.0.1:3200)
    #[arg(long)]
    http: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr) // keep stdout clean for MCP stdio
        .init();

    let cli = Cli::parse();

    let cfg = Config {
        jobs_dir: Config::resolve_jobs_dir(cli.jobs_dir),
        max_concurrent: cli.max_concurrent,
        http_addr: cli.http,
    };

    // Open / create the job store
    let store = JobStore::open(&cfg.jobs_dir).await?;
    let store = Arc::new(RwLock::new(store));

    // Create the scheduler daemon and the handle the MCP server uses
    let (daemon, handle, cancel_tx) = SchedulerDaemon::new(Arc::clone(&store), cfg.max_concurrent);

    // Spawn the daemon loop in the background
    tokio::spawn(daemon.run());

    // Signal the daemon to stop gracefully on Ctrl+C or SIGTERM
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = cancel_tx.send(true);
    });

    let http_addr = cfg.http_addr.clone();

    // If an HTTP address is specified, spin up the HTTP transport in parallel
    if let Some(addr) = http_addr {
        tracing::info!("HTTP MCP endpoint: http://{addr}/mcp");
        let handle_for_http = handle.clone();
        let addr_clone = addr.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_http(handle_for_http, &addr_clone).await {
                tracing::error!("HTTP MCP server error: {e:#}");
            }
        });
    }

    // Always serve on stdio (primary transport for Claude Code)
    tracing::info!("starting stdio MCP transport");
    let server = SchedulerServer::new(handle);
    let transport = stdio();
    server.serve(transport).await?.waiting().await?;

    Ok(())
}

async fn serve_http(handle: daemon::DaemonHandle, addr: &str) -> anyhow::Result<()> {
    let session_manager = Arc::new(LocalSessionManager::default());
    let http_config = StreamableHttpServerConfig::default();

    let service = StreamableHttpService::new(
        {
            let handle = handle.clone();
            move || Ok(SchedulerServer::new(handle.clone()))
        },
        session_manager,
        http_config,
    );

    let app = axum::Router::new().route("/mcp", axum::routing::any_service(service));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("HTTP MCP server listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Ctrl+C received — stopping scheduler daemon");
        }
        _ = sigterm.recv() => {
            tracing::info!("SIGTERM received — stopping scheduler daemon");
        }
    }
}
