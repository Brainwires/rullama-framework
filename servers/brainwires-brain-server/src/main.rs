use anyhow::Result;
use brainwires_brain_server::mcp_server::BrainMcpServer;
use clap::{Parser, Subcommand};
use std::panic;

/// Open Brain: persistent knowledge MCP server for any AI tool
#[derive(Parser)]
#[command(name = "brainwires-brain")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "MCP server exposing persistent knowledge (thoughts, PKS, BKS) to any AI tool")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio (default mode)
    Serve,
    /// Show version and system information
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
        Some(Commands::Serve) | None => {
            setup_panic_handler();

            if let Err(e) = BrainMcpServer::serve_stdio().await {
                tracing::error!("Fatal error in Brain MCP server: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn show_version_info() {
    println!("brainwires-brain v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("System Information:");
    println!("  Build Date:      {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:      {}", env!("GIT_COMMIT_HASH"));
    println!("  Rust Version:    {}", env!("CARGO_PKG_RUST_VERSION"));
    println!();
    println!("Storage:");
    println!("  Thoughts:        LanceDB (embedded, ~/.brainwires/brain/)");
    println!("  PKS:             SQLite  (personal facts, ~/.brainwires/pks.db)");
    println!("  BKS:             SQLite  (behavioral truths, ~/.brainwires/bks.db)");
    println!();
    println!("Embedding Model:");
    println!("  Model:           all-MiniLM-L6-v2");
    println!("  Dimensions:      384");
    println!("  Provider:        FastEmbed (local, no API calls)");
    println!();
    println!("MCP Tools:");
    println!("  capture_thought  — Store a thought with auto-categorization");
    println!("  search_memory    — Semantic search across all memory");
    println!("  list_recent      — Browse recent thoughts");
    println!("  get_thought      — Retrieve thought by ID");
    println!("  search_knowledge — Query PKS/BKS knowledge");
    println!("  memory_stats     — Dashboard of knowledge patterns");
    println!("  delete_thought   — Remove a thought");
}

fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown location".to_string());

        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic message".to_string()
        };

        tracing::error!(
            "PANIC at {}: {}\nBacktrace:\n{:?}",
            location,
            message,
            backtrace
        );

        eprintln!("\n!!! PANIC !!!");
        eprintln!("Location: {}", location);
        eprintln!("Message: {}", message);
        eprintln!("Backtrace:\n{:?}", backtrace);
        eprintln!("!!! END PANIC !!!\n");
    }));

    tracing::info!("Global panic handler initialized");
}
