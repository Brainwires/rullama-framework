use anyhow::Result;
use brainwires_gateway::config::GatewayConfig;
use brainwires_gateway::server::Gateway;
use clap::{Parser, Subcommand};
use std::panic;

/// Brainwires Gateway: routes messages between channel servers and agent sessions
#[derive(Parser)]
#[command(name = "brainwires-gateway")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "Always-on WebSocket gateway that routes messages between channel MCP servers and agent sessions"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gateway server (default mode)
    Serve {
        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value_t = 18789)]
        port: u16,

        /// Path to optional configuration file
        #[arg(long)]
        config: Option<String>,
    },
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
        Some(Commands::Serve {
            host,
            port,
            config: _config,
        }) => {
            setup_panic_handler();

            let gateway_config = GatewayConfig {
                host,
                port,
                ..Default::default()
            };

            let gateway = Gateway::new(gateway_config);
            if let Err(e) = gateway.run().await {
                tracing::error!("Fatal error in gateway: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
        None => {
            setup_panic_handler();

            let gateway = Gateway::new(GatewayConfig::default());
            if let Err(e) = gateway.run().await {
                tracing::error!("Fatal error in gateway: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn show_version_info() {
    println!("brainwires-gateway v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Gateway Daemon:");
    println!("  Description:  Routes messages between channel servers and agent sessions");
    println!("  Protocol:     WebSocket (channel adapters connect to gateway)");
    println!("  Default Port: 18789");
    println!();
    println!("Endpoints:");
    println!("  /ws             WebSocket upgrade for channel connections");
    println!("  /webhook        Webhook endpoint for HTTP-based channels");
    println!("  /admin/health   Health check");
    println!("  /admin/channels List connected channels");
    println!("  /admin/sessions List active sessions");
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
