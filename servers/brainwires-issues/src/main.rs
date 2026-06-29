use anyhow::Result;
use brainwires_issues::mcp_server::IssuesMcpServer;
use clap::{Parser, Subcommand};
use std::panic;

/// Brainwires Issues: lightweight project issue tracking MCP server
#[derive(Parser)]
#[command(name = "brainwires-issues")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "MCP server for issue and bug tracking", long_about = None)]
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

            if let Err(e) = IssuesMcpServer::serve_stdio().await {
                tracing::error!("Fatal error in MCP server: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn show_version_info() {
    println!("brainwires-issues v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("System Information:");
    println!("  Build Date:   {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:   {}", env!("GIT_COMMIT_HASH"));
    println!("  Rust Version: {}", env!("CARGO_PKG_RUST_VERSION"));
    println!();
    println!("Storage:");
    println!("  Backend:      LanceDB (embedded)");
    use brainwires_storage::LanceDatabase;
    let default_path = LanceDatabase::default_lancedb_path();
    println!("  Default Path: {}", default_path);
    println!();
    println!("MCP Tools:");
    println!("  Issues:   create_issue, get_issue, list_issues, update_issue,");
    println!("            close_issue, delete_issue, search_issues");
    println!("  Comments: add_comment, list_comments, delete_comment");
    println!();
    println!("MCP Prompts:");
    println!("  /create, /list, /search, /triage");
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
