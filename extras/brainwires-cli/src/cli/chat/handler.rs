//! Chat Handler Entry Points
//!
//! Main entry points for different chat modes (TUI, CLI, MCP server).

use anyhow::Result;

use crate::mdap::MdapConfig;
use crate::utils::logger::Logger;

/// Main chat handler - routes to appropriate mode
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat(
    model: Option<String>,
    provider: Option<String>,
    system: Option<String>,
    dev: bool,
    dev_port: u16,
    tui: bool,
    background: bool,
    session: Option<String>,
    pty_session: bool,
    json: bool,
    mcp_server: bool,
    prompt: Option<String>,
    quiet: bool,
    batch: bool,
    format: String,
    mdap_config: Option<MdapConfig>,
    mdap_estimate: bool,
) -> Result<()> {
    // Construct backend URL override if dev mode is enabled
    let backend_url_override = if dev || dev_port != 3000 {
        Some(format!("http://localhost:{}", dev_port))
    } else {
        None
    };

    // Show MDAP estimate if requested
    if mdap_estimate && let Some(config) = mdap_config.as_ref() {
        show_mdap_estimate(config);
    }

    if mcp_server {
        // Launch MCP server mode (MDAP not supported in MCP mode)
        handle_mcp_server(model, system, backend_url_override).await
    } else if background || tui {
        // Background mode: spawn a PTY session in the background (detached)
        // TUI mode: launch interactive TUI
        if background {
            // Spawn a backgrounded PTY session
            handle_background_session(model, mdap_config).await
        } else {
            // Launch TUI mode with MDAP config if enabled
            // If session is provided, we're resuming a backgrounded session
            // If pty_session is true, skip IPC agent connection (running in PTY)
            crate::tui::run_tui(session, model, mdap_config, pty_session).await
        }
    } else if let Some(prompt_text) = prompt {
        // Single-shot mode with optional MDAP
        if mdap_config.is_some() {
            super::conversation::handle_prompt_mode_mdap(
                model,
                provider,
                system,
                prompt_text,
                quiet,
                &format,
                mdap_config,
                backend_url_override,
            )
            .await
        } else {
            super::conversation::handle_prompt_mode(
                model,
                provider,
                system,
                prompt_text,
                quiet,
                &format,
                backend_url_override,
            )
            .await
        }
    } else if batch {
        // Batch processing mode (MDAP not yet supported in batch)
        super::conversation::handle_batch_mode(
            model,
            provider,
            system,
            quiet,
            &format,
            backend_url_override,
        )
        .await
    } else {
        // Use traditional line-based chat with optional MDAP
        let json_output = json || format == "json";
        super::conversation::handle_chat_with_conversation(
            model,
            provider,
            system,
            None,
            json_output,
            quiet,
            &format,
            mdap_config,
            backend_url_override,
        )
        .await
    }
}

/// Show MDAP cost estimate before execution
fn show_mdap_estimate(config: &MdapConfig) {
    use crate::mdap::scaling::estimate_mdap;

    Logger::info("MDAP Configuration:");
    Logger::info(format!("  Vote margin (k): {}", config.k));
    Logger::info(format!(
        "  Target success rate: {:.1}%",
        config.target_success_rate * 100.0
    ));
    Logger::info(format!("  Parallel samples: {}", config.parallel_samples));
    Logger::info(format!(
        "  Max samples/subtask: {}",
        config.max_samples_per_subtask
    ));

    // Rough estimate for a typical 10-step task
    if let Ok(estimate) = estimate_mdap(
        10, // Assume 10 steps for estimate
        0.99,
        0.95,
        config.cost_per_sample_usd.unwrap_or(0.0001),
        config.target_success_rate,
    ) {
        Logger::info("Estimated cost (for 10-step task):");
        Logger::info(format!("  API calls: ~{}", estimate.expected_api_calls));
        Logger::info(format!("  Cost: ~${:.4}", estimate.expected_cost_usd));
        Logger::info(format!(
            "  Success probability: {:.2}%",
            estimate.success_probability * 100.0
        ));
        Logger::info(format!("  Recommended k: {}", estimate.recommended_k));
    }
}

/// Handle MCP server mode - expose CLI as an MCP server over stdin/stdout
async fn handle_mcp_server(
    model: Option<String>,
    system: Option<String>,
    backend_url_override: Option<String>,
) -> Result<()> {
    use crate::mcp_server::McpServerHandler;

    // MCP stdio protocol requires stdout to contain ONLY JSON-RPC frames.
    // Route status messages to stderr to avoid corrupting the protocol stream.
    if let Some(ref url) = backend_url_override {
        eprintln!(
            "{} Starting MCP server mode (stdio) - using dev backend: {}",
            console::style("ℹ").blue(),
            url
        );
    } else {
        eprintln!(
            "{} Starting MCP server mode (stdio)",
            console::style("ℹ").blue()
        );
    }

    let handler = McpServerHandler::new(model, system, backend_url_override).await?;
    handler.run().await
}

/// Handle background session mode - spawn a PTY session in the background
async fn handle_background_session(
    model: Option<String>,
    _mdap_config: Option<MdapConfig>,
) -> Result<()> {
    use crate::config::ConfigManager;
    use crate::session;

    // Generate a session ID
    let session_id = format!("session-{}", chrono::Local::now().format("%Y%m%d-%H%M%S"));

    // Get model from config if not provided
    let model = match model {
        Some(m) => m,
        None => {
            let config_manager = ConfigManager::new()?;
            config_manager.get().model.clone()
        }
    };

    Logger::info(format!("Starting background session: {}", session_id));
    Logger::info(format!("Model: {}", model));

    // Spawn the PTY session in background mode
    match session::spawn_background_session(&session_id, &model).await {
        Ok(_) => {
            Logger::info(format!("✓ Session started in background: {}", session_id));
            Logger::info("Use 'brainwires attach' to connect to the session");
            Logger::info("Use 'brainwires sessions' to list all sessions");
            Ok(())
        }
        Err(e) => {
            Logger::error(format!("Failed to start background session: {}", e));
            Err(e)
        }
    }
}
