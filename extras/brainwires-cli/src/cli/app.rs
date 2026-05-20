use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::build_info;

#[derive(Parser)]
#[command(name = "brainwires")]
#[command(about = "AI-powered agentic CLI for autonomous coding assistance")]
#[command(version = build_info::FULL_VERSION)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Authenticate with Brainwires Studio
    #[command(subcommand)]
    Auth(super::auth::AuthCommands),

    /// Start an interactive chat session
    Chat {
        #[arg(short, long)]
        model: Option<String>,

        #[arg(short, long)]
        provider: Option<String>,

        #[arg(long)]
        system: Option<String>,

        /// Use local development server (default: localhost:3000)
        #[arg(long)]
        dev: bool,

        /// Port for local development server (default: 3000, implies --dev)
        #[arg(long, default_value = "3000")]
        dev_port: u16,

        /// Use full-screen TUI mode instead of line-based interaction
        #[arg(long)]
        tui: bool,

        /// Start the TUI session in the background (detached PTY)
        #[arg(long, short = 'b')]
        background: bool,

        /// Resume/attach to an existing session by ID (for background reattachment)
        #[arg(long)]
        session: Option<String>,

        /// Internal flag: indicates TUI is running inside a PTY session (skips IPC agent connection)
        #[arg(long, hide = true)]
        pty_session: bool,

        /// Output full conversation as JSON on exit
        #[arg(long)]
        json: bool,

        /// Run as MCP server (stdio protocol) instead of interactive chat
        #[arg(long)]
        mcp_server: bool,

        /// Single-shot mode: send one prompt and exit (no conversation loop)
        #[arg(long)]
        prompt: Option<String>,

        /// Quiet mode: suppress decorative output for scripting
        #[arg(short, long)]
        quiet: bool,

        /// Batch mode: process multiple prompts from stdin, one per line
        #[arg(long)]
        batch: bool,

        /// Output format: plain (response only), full (default), json
        #[arg(long, default_value = "full")]
        format: String,

        // MDAP (Massively Decomposed Agentic Processes) options
        /// Enable MDAP mode for high-reliability task execution
        #[arg(long)]
        mdap: bool,

        /// MDAP vote margin threshold (k in first-to-ahead-by-k voting)
        #[arg(long, default_value = "3")]
        mdap_k: u32,

        /// MDAP target success probability (0.0-1.0)
        #[arg(long, default_value = "0.95")]
        mdap_target: f64,

        /// MDAP maximum parallel sampling threads (1-4)
        #[arg(long, default_value = "4", value_parser = clap::value_parser!(u32).range(1..=4))]
        mdap_parallel: u32,

        /// Show MDAP cost estimate before execution
        #[arg(long)]
        mdap_estimate: bool,

        /// MDAP max samples per subtask before failure
        #[arg(long, default_value = "50")]
        mdap_max_samples: u32,

        /// MDAP fail fast on first subtask failure
        #[arg(long)]
        mdap_fail_fast: bool,

        /// Send the full tool registry (100+ tools) instead of the curated core
        /// set. Most users don't need this — the core set already includes
        /// `search_tools` so agents can discover and call anything else on
        /// demand. Use only when you want every tool eagerly enumerated,
        /// e.g. for benchmarks or deterministic replay.
        #[arg(long)]
        all_tools: bool,

        /// Bash sandbox mode. `off` (default) runs commands with normal
        /// privileges. `network-deny` wraps every bash invocation in
        /// `unshare -U -r -n` (Linux only) so outbound network is blocked.
        /// On non-Linux platforms this is silently a no-op — build your
        /// own sandbox or keep `off`.
        #[arg(long, value_parser = ["off", "network-deny"])]
        sandbox: Option<String>,
    },

    /// Manage configuration
    Config {
        #[arg(short, long)]
        list: bool,

        #[arg(short, long)]
        get: Option<String>,

        /// Set a config value (format: key=value OR key value)
        #[arg(short, long, num_args = 1..=2)]
        set: Option<Vec<String>>,
    },

    /// Manage execution plans
    #[command(subcommand)]
    Plan(super::plan::PlanCommands),

    /// Execute a one-off task
    Task {
        prompt: String,

        #[arg(short, long)]
        model: Option<String>,

        #[arg(short, long)]
        provider: Option<String>,
    },

    /// View API usage and costs
    Cost {
        #[arg(short, long)]
        period: Option<String>,

        #[arg(long)]
        reset: bool,
    },

    /// Manage MCP servers
    #[command(subcommand)]
    Mcp(super::mcp::McpCommands),

    /// List available AI models and statistics
    Models {
        #[command(subcommand)]
        command: Option<super::models::ModelsCommands>,
    },

    /// Manage conversation history
    #[command(subcommand)]
    History(super::history::HistoryCommands),

    /// Attach to a backgrounded TUI session
    Attach {
        /// Session ID to attach to (default: most recent)
        session: Option<String>,
    },

    /// List backgrounded TUI sessions
    Sessions,

    /// Terminate a backgrounded TUI session (graceful - waits until idle)
    Exit {
        /// Session ID to terminate (default: most recent)
        session: Option<String>,
    },

    /// Forcefully kill a backgrounded TUI session (immediate termination)
    Kill {
        /// Session ID to kill (default: most recent)
        session: Option<String>,
    },

    /// Initialize project configuration
    Init,

    /// View analytics — cost, tool usage, agent summaries, raw events
    #[command(subcommand)]
    Analytics(super::analytics::AnalyticsCommands),

    /// Manage remote bridge connection to brainwires-studio
    #[command(subcommand)]
    Remote(super::remote::RemoteCommands),

    /// Manage local LLM models for CPU-based inference
    #[command(subcommand, alias = "local")]
    LocalModels(super::local_models::LocalModelCommands),

    /// Run eval-driven autonomous self-improvement feedback loop
    EvalImprove {
        /// Path to load/save eval baselines JSON
        #[arg(long, default_value = "eval-baselines.json")]
        baselines_path: String,

        /// Maximum outer feedback-loop rounds
        #[arg(long, default_value = "3")]
        max_rounds: u32,

        /// Trials per eval case per run
        #[arg(long, default_value = "10")]
        n_trials: usize,

        /// Minimum success-rate improvement to update baselines (0.0–1.0)
        #[arg(long, default_value = "0.05")]
        improvement_threshold: f64,

        /// Automatically update baselines when improvement threshold is met
        #[arg(long, default_value = "true")]
        auto_update_baselines: bool,

        /// Commit updated baselines to git
        #[arg(long)]
        commit_baselines: bool,

        /// Show detected faults without running self-improvement agents
        #[arg(long)]
        dry_run: bool,

        /// Maximum budget in dollars passed to the improvement controller
        #[arg(long, default_value = "10.0")]
        max_budget: f64,

        /// Disable MCP bridge execution path
        #[arg(long)]
        no_bridge: bool,

        /// Disable direct agent execution path
        #[arg(long)]
        no_direct: bool,
    },

    /// Run autonomous self-improvement loop
    SelfImprove {
        /// Maximum number of improvement cycles
        #[arg(long, default_value = "10")]
        max_cycles: u32,

        /// Maximum budget in dollars
        #[arg(long, default_value = "10.0")]
        max_budget: f64,

        /// List tasks without executing
        #[arg(long)]
        dry_run: bool,

        /// Comma-separated list of strategies (empty = all)
        #[arg(long)]
        strategies: Option<String>,

        /// Max iterations per agent
        #[arg(long, default_value = "25")]
        agent_iterations: u32,

        /// Max diff lines per task
        #[arg(long, default_value = "200")]
        max_diff: u32,

        /// Create pull requests for changes
        #[arg(long)]
        create_prs: bool,

        /// Branch name prefix
        #[arg(long, default_value = "self-improve/")]
        branch_prefix: String,

        /// Disable MCP bridge execution path
        #[arg(long)]
        no_bridge: bool,

        /// Disable direct agent execution path
        #[arg(long)]
        no_direct: bool,

        /// Model to use for agents
        #[arg(long)]
        model: Option<String>,

        /// Provider to use for agents
        #[arg(long)]
        provider: Option<String>,
    },

    /// Run as a background agent process (internal - used by TUI for session persistence)
    #[command(hide = true)]
    Agent {
        /// Session ID for this agent
        session_id: String,

        /// Model to use
        #[arg(long)]
        model: Option<String>,

        /// Enable MDAP mode
        #[arg(long)]
        mdap: bool,

        /// MDAP vote margin
        #[arg(long, default_value = "3")]
        mdap_k: u32,

        /// MDAP target success rate
        #[arg(long, default_value = "0.95")]
        mdap_target: f64,

        /// MDAP parallel samples
        #[arg(long, default_value = "4")]
        mdap_parallel: u32,

        /// MDAP max samples per subtask
        #[arg(long, default_value = "50")]
        mdap_max_samples: u32,

        /// MDAP fail fast
        #[arg(long)]
        mdap_fail_fast: bool,
    },
}

pub struct App {
    cli: Cli,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self { cli: Cli::parse() }
    }

    pub async fn run(self) -> Result<()> {
        // Initialize logging for CLI commands
        // Note: TUI mode handles its own logging initialization
        // Initialize analytics before logging so the AnalyticsLayer is available.
        crate::utils::logger::init_analytics();

        if let Some(Commands::Chat { tui: true, .. }) = &self.cli.command {
            // Skip logging init for TUI mode
        } else {
            // Honor `--quiet` / `--format json` on `chat` by raising the
            // default console filter to warn so stderr stays clean for
            // scripts. RUST_LOG still wins (see logger::init_with_options).
            let quiet_chat = match &self.cli.command {
                Some(Commands::Chat { quiet, format, .. }) => *quiet || format == "json",
                _ => false,
            };
            crate::utils::logger::init_with_options(true, quiet_chat);
        }

        // First-run provider picker: triggers on the first user-facing
        // interactive command when no config exists and no provider was
        // detected from the environment. Keeps non-TTY invocations usable
        // by emitting a clear error instead of hanging on a prompt.
        let needs_provider_setup = matches!(
            &self.cli.command,
            Some(Commands::Chat {
                mcp_server: false,
                ..
            }) | Some(Commands::Task { .. })
        );
        if needs_provider_setup {
            let mut config_manager = crate::config::ConfigManager::new()?;
            if config_manager.is_first_run() {
                match super::first_run::maybe_prompt(&mut config_manager).await {
                    Ok(_) => {}
                    Err(e) => {
                        // `maybe_prompt` already printed helpful guidance for
                        // non-TTY cases; surface the error and exit.
                        return Err(e);
                    }
                }
            }
        }

        // Check for local models setup on interactive commands
        // Only show for Chat and Task commands (main user-facing commands)
        if matches!(
            &self.cli.command,
            Some(Commands::Chat {
                mcp_server: false,
                prompt: None,
                batch: false,
                ..
            }) | Some(Commands::Task { .. })
        ) && super::local_models_setup::should_prompt_for_setup()
        {
            // Show setup dialog - don't fail if it errors
            if let Err(e) = super::local_models_setup::show_setup_dialog().await {
                tracing::warn!("Local models setup dialog error: {}", e);
            }
        }

        match self.cli.command {
            Some(Commands::Auth(cmd)) => super::auth::handle_auth(cmd).await,
            Some(Commands::Chat {
                model,
                provider,
                system,
                dev,
                dev_port,
                tui,
                background,
                session,
                pty_session,
                json,
                mcp_server,
                prompt,
                quiet,
                batch,
                format,
                mdap,
                mdap_k,
                mdap_target,
                mdap_parallel,
                mdap_estimate,
                mdap_max_samples,
                mdap_fail_fast,
                all_tools,
                sandbox,
            }) => {
                let mdap_config = if mdap {
                    Some(
                        crate::mdap::MdapConfig::builder()
                            .k(mdap_k)
                            .target_success_rate(mdap_target)
                            .parallel_samples(mdap_parallel)
                            .max_samples_per_subtask(mdap_max_samples)
                            .fail_fast(mdap_fail_fast)
                            .build()
                            .expect("Invalid MDAP configuration"),
                    )
                } else {
                    None
                };
                if all_tools {
                    crate::tools::set_all_tools_override(true);
                }
                if let Some(mode) = sandbox.as_deref() {
                    // SAFETY: mutating process env during startup before any
                    // threads are spawned that read BRAINWIRES_BASH_SANDBOX.
                    // The bash tool only reads this on command build, never
                    // writes, so a single writer at startup is sound.
                    unsafe {
                        std::env::set_var("BRAINWIRES_BASH_SANDBOX", mode);
                    }
                }
                super::chat::handle_chat(
                    model,
                    provider,
                    system,
                    dev,
                    dev_port,
                    tui,
                    background,
                    session,
                    pty_session,
                    json,
                    mcp_server,
                    prompt,
                    quiet,
                    batch,
                    format,
                    mdap_config,
                    mdap_estimate,
                )
                .await
            }
            Some(Commands::Config { list, get, set }) => {
                super::config::handle_config(list, get, set).await
            }
            Some(Commands::Plan(cmd)) => super::plan::handle_plan_command(cmd).await,
            Some(Commands::Task {
                prompt,
                model,
                provider,
            }) => super::task::handle_task(prompt, model, provider).await,
            Some(Commands::Cost { period, reset }) => super::cost::handle_cost(period, reset).await,
            Some(Commands::Mcp(cmd)) => super::mcp::handle_mcp(cmd).await,
            Some(Commands::Models { command }) => super::models::handle_models(command).await,
            Some(Commands::History(cmd)) => super::history::handle_history(cmd).await,
            Some(Commands::Attach { session }) => super::attach::attach(session).await,
            Some(Commands::Sessions) => super::attach::list_sessions().await,
            Some(Commands::Exit { session }) => super::attach::exit_session(session).await,
            Some(Commands::Kill { session }) => super::attach::kill_session(session).await,
            Some(Commands::Init) => {
                crate::utils::logger::Logger::warn("Init not yet implemented");
                // Non-zero exit so scripts can detect the no-op and not mistake it for success.
                Err(anyhow::anyhow!("Init not yet implemented"))
            }
            Some(Commands::Analytics(cmd)) => super::analytics::handle_analytics(cmd).await,
            Some(Commands::EvalImprove {
                baselines_path,
                max_rounds,
                n_trials,
                improvement_threshold,
                auto_update_baselines,
                commit_baselines,
                dry_run,
                max_budget,
                no_bridge,
                no_direct,
            }) => {
                super::self_improve_cmd::handle_eval_improve(
                    baselines_path,
                    max_rounds,
                    n_trials,
                    improvement_threshold,
                    auto_update_baselines,
                    commit_baselines,
                    dry_run,
                    max_budget,
                    no_bridge,
                    no_direct,
                )
                .await
            }
            Some(Commands::SelfImprove {
                max_cycles,
                max_budget,
                dry_run,
                strategies,
                agent_iterations,
                max_diff,
                create_prs,
                branch_prefix,
                no_bridge,
                no_direct,
                model,
                provider,
            }) => {
                super::self_improve_cmd::handle_self_improve(
                    max_cycles,
                    max_budget,
                    dry_run,
                    strategies,
                    agent_iterations,
                    max_diff,
                    create_prs,
                    branch_prefix,
                    no_bridge,
                    no_direct,
                    model,
                    provider,
                )
                .await
            }
            Some(Commands::Remote(cmd)) => super::remote::handle_remote(cmd).await,
            Some(Commands::LocalModels(cmd)) => {
                super::local_models::handle_local_models(Some(cmd)).await
            }
            Some(Commands::Agent {
                session_id,
                model,
                mdap,
                mdap_k,
                mdap_target,
                mdap_parallel,
                mdap_max_samples,
                mdap_fail_fast,
            }) => {
                let mdap_config = if mdap {
                    Some(
                        crate::mdap::MdapConfig::builder()
                            .k(mdap_k)
                            .target_success_rate(mdap_target)
                            .parallel_samples(mdap_parallel)
                            .max_samples_per_subtask(mdap_max_samples)
                            .fail_fast(mdap_fail_fast)
                            .build()
                            .expect("Invalid MDAP configuration"),
                    )
                } else {
                    None
                };
                // Run as Agent process - this blocks until agent exits
                let agent =
                    crate::agent::AgentProcess::new(Some(session_id), model, mdap_config).await?;
                agent.run().await
            }
            None => {
                // No command provided, show help
                Cli::parse_from(["brainwires", "--help"]);
                Ok(())
            }
        }
    }
}
