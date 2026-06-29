use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use claude_brain::mcp_server::ClaudeBrainMcpServer;
use std::panic;

/// Claude Brain: Brainwires context management for Claude Code
#[derive(Parser)]
#[command(name = "claude-brain")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Replaces Claude Code compaction with Brainwires tiered memory, dream consolidation, and semantic recall"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio (default mode)
    Serve,
    /// Handle a Claude Code hook event
    Hook {
        /// Which hook event to handle
        event: HookEvent,
    },
    /// Show version and system information
    Version,
}

#[derive(Debug, Clone, ValueEnum)]
enum HookEvent {
    /// SessionStart — load relevant context from all memory tiers
    SessionStart,
    /// Stop — capture assistant turn into hot-tier storage
    Stop,
    /// PreCompact — export full conversation before compaction
    PreCompact,
    /// PostCompact — inject Brainwires context after compaction
    PostCompact,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Only init tracing for serve mode — hooks write to stdout
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Version) => {
            show_version_info();
        }
        Some(Commands::Hook { event }) => {
            // Hooks: minimal logging, output goes to stdout for Claude Code
            if let Err(e) = handle_hook(event).await {
                // Write error to stderr (Claude Code captures stdout)
                eprintln!("claude-brain hook error: {:#}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Serve) | None => {
            tracing_subscriber::fmt::init();
            setup_panic_handler();

            if let Err(e) = ClaudeBrainMcpServer::serve_stdio().await {
                tracing::error!("Fatal error in Claude Brain MCP server: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

async fn handle_hook(event: HookEvent) -> Result<()> {
    match event {
        HookEvent::SessionStart => claude_brain::hooks::session_start::handle().await,
        HookEvent::Stop => claude_brain::hooks::stop::handle().await,
        HookEvent::PreCompact => claude_brain::hooks::pre_compact::handle().await,
        HookEvent::PostCompact => claude_brain::hooks::post_compact::handle().await,
    }
}

fn show_version_info() {
    println!("claude-brain v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Brainwires Context Management for Claude Code");
    println!();
    println!("Modes:");
    println!("  serve          Start MCP server (default)");
    println!("  hook <event>   Handle Claude Code lifecycle hook");
    println!();
    println!("Hook Events:");
    println!("  session-start  Load context from all memory tiers");
    println!("  stop           Capture turn into hot-tier storage");
    println!("  pre-compact    Export conversation before compaction");
    println!("  post-compact   Inject Brainwires context after compaction");
    println!();
    println!("MCP Tools:");
    println!("  recall_context   Search past conversation history");
    println!("  capture_thought  Persist decisions and insights");
    println!("  search_memory    Semantic search across all tiers");
    println!("  search_knowledge Query PKS/BKS knowledge base");
    println!("  memory_stats     Knowledge statistics dashboard");
    println!();
    println!("Storage:");
    println!("  Thoughts:  LanceDB  (~/.brainwires/claude-brain/)");
    println!("  PKS:       SQLite   (~/.brainwires/pks.db)");
    println!("  BKS:       SQLite   (~/.brainwires/bks.db)");
    println!();
    println!("Embedding: all-MiniLM-L6-v2 (384d, local via FastEmbed)");
}

fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();
        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());

        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };

        tracing::error!("PANIC at {location}: {message}\n{backtrace:?}");
        eprintln!("PANIC at {location}: {message}");
    }));
}
