//! brainwires-memory-server — Mem0-compatible memory REST server backed by
//! `brainwires-knowledge`.
//!
//! Configuration is via environment variables:
//!   MEMORY_HOST  — bind address (default: 127.0.0.1)
//!   MEMORY_PORT  — listen port (default: 8765)
//!   MEMORY_DB    — knowledge storage directory
//!                  (default: ~/.local/share/brainwires/memory)

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use brainwires_memory_server::{AppState, build_app, build_client};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "brainwires_memory_server=info,tower_http=debug".to_string()),
        )
        .init();

    let host = std::env::var("MEMORY_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("MEMORY_PORT")
        .unwrap_or_else(|_| "8765".to_string())
        .parse()
        .context("MEMORY_PORT must be a valid port number")?;
    let storage_dir: PathBuf = std::env::var("MEMORY_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("brainwires")
                .join("memory")
        });

    tracing::info!(
        "brainwires-memory-server storage directory: {}",
        storage_dir.display()
    );
    let client = build_client(&storage_dir).await?;
    let app = build_app(AppState::new(client));

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("brainwires-memory-server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
