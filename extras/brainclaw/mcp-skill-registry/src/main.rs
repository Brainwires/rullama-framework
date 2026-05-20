use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use tokio::sync::Mutex;

use brainwires_skill_registry::api;
use brainwires_skill_registry::storage::SkillStore;

/// Skill Marketplace: registry server for distributable skill packages
#[derive(Parser)]
#[command(name = "brainwires-skill-registry")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "HTTP registry server for publishing and discovering skill packages")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the registry HTTP server (default mode)
    Serve {
        /// Listen address
        #[arg(long, default_value = "0.0.0.0:3000")]
        listen: String,
        /// SQLite database path
        #[arg(long, default_value = "skills.db")]
        db: String,
    },
    /// Show version information
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => {
            println!("brainwires-skill-registry v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Skill Marketplace registry server");
            println!("  Storage: SQLite with FTS5 full-text search");
            println!("  API:     RESTful JSON over HTTP");
        }
        Some(Commands::Serve { listen, db }) => {
            start_server(listen, db).await?;
        }
        None => {
            let listen = "0.0.0.0:3000".to_string();
            let db = "skills.db".to_string();
            start_server(listen, db).await?;
        }
    }

    Ok(())
}

async fn start_server(listen: String, db: String) -> Result<()> {
    tracing::info!("Opening database at {}", db);
    let store = SkillStore::open(&db)?;
    let state = Arc::new(Mutex::new(store));

    let app = api::router(state);

    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("Skill registry listening on {}", listen);

    axum::serve(listener, app).await?;

    Ok(())
}
