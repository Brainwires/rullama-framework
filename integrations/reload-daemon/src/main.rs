//! # Reload Daemon
//!
//! A minimal MCP server that exposes a single `reload_app` tool. AI coding
//! clients (Claude Code, Cursor, etc.) call this tool to kill themselves and
//! restart with transformed arguments. Restart strategies are config-driven.
//!
//! ## Usage
//!
//! ```sh
//! cargo run -p reload-daemon -- \
//!   --config extras/reload-daemon/config.json
//! ```
//!
//! ## Register with Claude Code
//!
//! ```sh
//! claude mcp add --transport http reload-daemon http://127.0.0.1:3100/mcp
//! ```

use clap::Parser;
use reload_daemon::config::DaemonConfig;
use reload_daemon::server::ReloadServer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "reload-daemon", about = "MCP server for process reload")]
struct Cli {
    /// Path to the JSON config file.
    #[arg(long)]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config_text = std::fs::read_to_string(&cli.config)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {e}", cli.config))?;
    let daemon_config: DaemonConfig = serde_json::from_str(&config_text)
        .map_err(|e| anyhow::anyhow!("invalid config JSON: {e}"))?;

    let listen_addr = daemon_config.listen.clone();
    let config = Arc::new(daemon_config);

    let session_manager = Arc::new(LocalSessionManager::default());
    let streamable_http_config = StreamableHttpServerConfig::default();

    let service = StreamableHttpService::new(
        {
            let config = Arc::clone(&config);
            move || Ok(ReloadServer::new(Arc::clone(&config)))
        },
        session_manager,
        streamable_http_config,
    );

    let app = axum::Router::new().route("/mcp", axum::routing::any_service(service));

    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    tracing::info!("reload-daemon listening on {listen_addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    tracing::info!("shutting down");
}
