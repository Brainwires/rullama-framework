use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "agent-chat",
    about = "AI chat client powered by the Brainwires Framework",
    version
)]
pub struct Cli {
    /// AI provider (anthropic, openai, google, groq, ollama, together, fireworks, brainwires)
    #[arg(long, short, env = "AGENT_CHAT_PROVIDER")]
    pub provider: Option<String>,

    /// Model name (e.g. claude-sonnet-4-20250514, gpt-4o)
    #[arg(long, short, env = "AGENT_CHAT_MODEL")]
    pub model: Option<String>,

    /// System prompt
    #[arg(long, short)]
    pub system: Option<String>,

    /// Use fullscreen TUI mode
    #[arg(long)]
    pub tui: bool,

    /// Maximum tokens to generate
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Temperature (0.0 - 1.0)
    #[arg(long)]
    pub temperature: Option<f32>,

    /// API key (overrides env/config)
    #[arg(long, env = "AGENT_CHAT_API_KEY")]
    pub api_key: Option<String>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Manage configuration
    Config(ConfigArgs),
    /// List available models
    Models(ModelsArgs),
    /// Manage API keys
    Auth(AuthArgs),
}

#[derive(Parser)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: ConfigAction,
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// List all configuration values
    List,
    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },
    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
}

#[derive(Parser)]
pub struct ModelsArgs {
    /// Filter by provider
    #[arg(long, short)]
    pub provider: Option<String>,
}

#[derive(Parser)]
pub struct AuthArgs {
    #[command(subcommand)]
    pub action: AuthAction,
}

#[derive(Subcommand)]
pub enum AuthAction {
    /// Set an API key for a provider
    Set {
        /// Provider name
        provider: String,
    },
    /// Show configured providers
    Show,
    /// Remove an API key
    Remove {
        /// Provider name
        provider: String,
    },
}
