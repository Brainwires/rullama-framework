//! IRC channel adapter — entrypoint.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tokio::sync::{broadcast, mpsc};

use brainwires_network::channels::Channel;

use brainwires_irc_channel::config::{IrcConfig, parse_channel_list};
use brainwires_irc_channel::gateway_client::{GatewayClient, backoff_next};
use brainwires_irc_channel::irc_client::{IrcChannel, run};
use brainwires_irc_channel::mcp_server::IrcMcpServer;

#[derive(Parser)]
#[command(
    name = "brainclaw-mcp-irc",
    version = env!("CARGO_PKG_VERSION"),
    about = "IRC channel adapter for the Brainwires gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Connect to an IRC network and bridge to the gateway (default).
    Serve {
        #[arg(long, default_value = "irc.libera.chat", env = "IRC_SERVER")]
        server: String,
        #[arg(long, default_value_t = 6697, env = "IRC_PORT")]
        port: u16,
        #[arg(long, default_value_t = true, env = "IRC_USE_TLS")]
        use_tls: bool,
        #[arg(long, env = "IRC_NICK")]
        nick: String,
        #[arg(long, default_value = "brainclaw", env = "IRC_USERNAME")]
        username: String,
        #[arg(long, default_value = "BrainClaw Bot", env = "IRC_REALNAME")]
        realname: String,
        #[arg(long, env = "IRC_SASL_PASSWORD")]
        sasl_password: Option<String>,
        #[arg(long, env = "IRC_CHANNELS")]
        channels: Option<String>,
        #[arg(long, default_value = "brainclaw: ", env = "IRC_MESSAGE_PREFIX")]
        message_prefix: String,
        #[arg(long, default_value = "ws://127.0.0.1:18789/ws", env = "GATEWAY_URL")]
        gateway_url: String,
        #[arg(long, env = "GATEWAY_TOKEN")]
        gateway_token: Option<String>,
        /// Also expose an MCP stdio server.
        #[arg(long, default_value_t = false)]
        mcp: bool,
    },
    /// Stdio MCP server only. Connects to IRC but does not bridge to a
    /// gateway — useful for ad-hoc scripting.
    Mcp {
        #[arg(long, env = "IRC_SERVER")]
        server: String,
        #[arg(long, default_value_t = 6697, env = "IRC_PORT")]
        port: u16,
        #[arg(long, default_value_t = true, env = "IRC_USE_TLS")]
        use_tls: bool,
        #[arg(long, env = "IRC_NICK")]
        nick: String,
    },
    /// Print version info.
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Version) => {
            println!("brainclaw-mcp-irc v{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Mcp {
            server,
            port,
            use_tls,
            nick,
        }) => {
            let cfg = IrcConfig {
                server: server.clone(),
                port,
                use_tls,
                nick: nick.clone(),
                username: nick.clone(),
                realname: nick.clone(),
                sasl_password: None,
                channels: Vec::new(),
                message_prefix: String::new(),
                gateway_url: String::new(),
                gateway_token: None,
            };
            let channel = Arc::new(IrcChannel::new(cfg.server.clone()));
            let (event_tx, _event_rx) = mpsc::channel(512);
            let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
            let runner_chan = Arc::clone(&channel);
            tokio::spawn(async move {
                let _ = run(cfg, runner_chan, event_tx, shutdown_rx).await;
            });
            let result = IrcMcpServer::serve_stdio(Arc::clone(&channel)).await;
            let _ = shutdown_tx.send(());
            result
        }
        Some(Command::Serve {
            server,
            port,
            use_tls,
            nick,
            username,
            realname,
            sasl_password,
            channels,
            message_prefix,
            gateway_url,
            gateway_token,
            mcp,
        }) => {
            let config = IrcConfig {
                server,
                port,
                use_tls,
                nick,
                username,
                realname,
                sasl_password,
                channels: channels
                    .as_deref()
                    .map(parse_channel_list)
                    .unwrap_or_default(),
                message_prefix,
                gateway_url,
                gateway_token,
            };
            run_adapter(config, mcp).await
        }
        None => {
            eprintln!("No subcommand. Run with --help for usage.");
            std::process::exit(1);
        }
    }
}

async fn run_adapter(config: IrcConfig, enable_mcp: bool) -> Result<()> {
    tracing::info!(server = %config.server, nick = %config.nick, "starting irc adapter");

    let channel = Arc::new(IrcChannel::new(config.server.clone()));
    let caps = channel.capabilities();
    let (event_tx, event_rx) = mpsc::channel(512);
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    if enable_mcp {
        let mcp_channel = Arc::clone(&channel);
        tokio::spawn(async move {
            if let Err(e) = IrcMcpServer::serve_stdio(mcp_channel).await {
                tracing::error!("mcp: {e:#}");
            }
        });
    }

    let irc_config = config.clone();
    let irc_channel = Arc::clone(&channel);
    let irc_tx = event_tx.clone();
    tokio::spawn(async move {
        let _ = run(irc_config, irc_channel, irc_tx, shutdown_rx).await;
    });

    // Gateway reconnect loop.
    let (gw_tx, mut gw_rx) = mpsc::channel(512);
    tokio::spawn(async move {
        let mut rx = event_rx;
        while let Some(ev) = rx.recv().await {
            if gw_tx.send(ev).await.is_err() {
                break;
            }
        }
    });
    let gw_url = config.gateway_url.clone();
    let gw_token = config.gateway_token.clone().unwrap_or_default();
    let gw_channel = Arc::clone(&channel);
    let gw_handle = tokio::spawn(async move {
        let mut backoff = Duration::from_millis(1_000);
        loop {
            match GatewayClient::connect(&gw_url, &gw_token, caps).await {
                Ok(gw) => {
                    backoff = Duration::from_millis(1_000);
                    let (local_tx, local_rx) = mpsc::channel(512);
                    let forwarder = tokio::spawn(async move {
                        while let Some(ev) = gw_rx.recv().await {
                            if local_tx.send(ev).await.is_err() {
                                break;
                            }
                        }
                        gw_rx
                    });
                    let chan = Arc::clone(&gw_channel);
                    if let Err(e) = gw.run(local_rx, chan).await {
                        tracing::error!("gateway: {e:#}");
                    }
                    match forwarder.await {
                        Ok(back) => gw_rx = back,
                        Err(e) => {
                            tracing::error!("forwarder join: {e}");
                            return;
                        }
                    }
                }
                Err(e) => tracing::error!("gateway connect: {e:#} (retry in {:?})", backoff),
            }
            tokio::time::sleep(backoff).await;
            backoff = backoff_next(backoff);
        }
    });

    shutdown_signal().await;
    let _ = shutdown_tx.send(());
    gw_handle.abort();
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.unwrap() };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .unwrap()
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
