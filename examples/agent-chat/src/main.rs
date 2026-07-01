use anyhow::Result;
use clap::Parser;

use agent_chat::chat_session::ChatSession;
use agent_chat::cli::{Cli, Command};
use agent_chat::commands;
use agent_chat::config::ChatConfig;
use agent_chat::provider_setup;
use agent_chat::tool_setup;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Config(args)) => commands::config_cmd::run(args)?,
        Some(Command::Models(args)) => commands::models_cmd::run(args).await?,
        Some(Command::Auth(args)) => commands::auth_cmd::run(args)?,
        None => {
            let config = ChatConfig::load()?;
            let provider = provider_setup::create_provider(&cli, &config)?;
            let tools = tool_setup::build_registry();
            let session = ChatSession::new(provider, tools, &cli, &config);

            if cli.tui {
                agent_chat::tui::run(session).await?;
            } else {
                agent_chat::plain::run(session).await?;
            }
        }
    }

    Ok(())
}
