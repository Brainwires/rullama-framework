use anyhow::Result;
use clap::Subcommand;
use console::style;
use std::collections::BTreeMap;

use crate::config::{ConfigManager, ModelRegistry, ModelService};
use crate::providers::{AvailableModel, ProviderType};

#[derive(Subcommand)]
pub enum ModelsCommands {
    /// List all available models (default)
    #[command(name = "list")]
    List {
        /// List models from a specific provider (anthropic, openai, google, groq, ollama)
        #[arg(short, long)]
        provider: Option<String>,

        /// Include non-chat models (embeddings, audio, image generation)
        #[arg(short, long)]
        all: bool,

        /// Bypass cache and fetch fresh data from provider
        #[arg(short, long)]
        refresh: bool,
    },

    /// Show model statistics
    #[command(name = "stats")]
    Stats {
        /// Show stats for a specific provider
        #[arg(short, long)]
        provider: Option<String>,

        /// Bypass cache
        #[arg(short, long)]
        refresh: bool,
    },
}

pub async fn handle_models(cmd: Option<ModelsCommands>) -> Result<()> {
    match cmd {
        None => handle_models_list(None, false, false).await,
        Some(ModelsCommands::List {
            provider,
            all,
            refresh,
        }) => handle_models_list(provider, all, refresh).await,
        Some(ModelsCommands::Stats { provider, refresh }) => {
            handle_models_stats(provider, refresh).await
        }
    }
}

/// Resolve which provider to query. Returns (provider_type, api_key, base_url).
fn resolve_provider(
    explicit: Option<&str>,
) -> Result<(ProviderType, Option<String>, Option<String>)> {
    if let Some(name) = explicit {
        let pt = ProviderType::from_str_opt(name).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown provider: '{}'. Supported: anthropic, openai, google, groq, ollama",
                name
            )
        })?;
        // Load credentials from config/keyring
        let config_manager = ConfigManager::new()?;
        let config = config_manager.get();

        let api_key = if pt == config.provider_type {
            config_manager
                .get_provider_api_key()?
                .map(|z| z.to_string())
        } else {
            config_manager
                .get_provider_api_key_for(pt)?
                .map(|z| z.to_string())
        };
        let base_url = if pt == config.provider_type {
            config.provider_base_url.clone()
        } else {
            None
        };
        Ok((pt, api_key, base_url))
    } else {
        // Use active provider
        let config_manager = ConfigManager::new()?;
        let config = config_manager.get();
        let api_key = config_manager
            .get_provider_api_key()?
            .map(|z| z.to_string());
        Ok((
            config.provider_type,
            api_key,
            config.provider_base_url.clone(),
        ))
    }
}

async fn handle_models_list(provider: Option<String>, show_all: bool, refresh: bool) -> Result<()> {
    let (provider_type, api_key, base_url) = resolve_provider(provider.as_deref())?;

    // Brainwires SaaS → existing ModelRegistry flow
    if provider_type == ProviderType::Brainwires {
        return handle_brainwires_models_list().await;
    }

    println!(
        "\n{} ({})\n",
        style("Available Models:").cyan().bold(),
        style(provider_type.as_str()).yellow()
    );

    let use_cache = !refresh;
    let models = if show_all {
        ModelService::list_models_for_provider(
            provider_type,
            api_key.as_deref(),
            base_url.as_deref(),
            use_cache,
        )
        .await?
    } else {
        ModelService::list_chat_models_for_provider(
            provider_type,
            api_key.as_deref(),
            base_url.as_deref(),
            use_cache,
        )
        .await?
    };

    if models.is_empty() {
        eprintln!("{}", style("No models available").red());
        return Ok(());
    }

    // Get active model for marking
    let config = ConfigManager::new()?;
    let active_model = config.get().model.clone();

    display_provider_models(&models, &active_model);

    if !show_all {
        println!(
            "{}",
            style("Tip: use --all to include embeddings, audio, and image models").dim()
        );
    }
    if !refresh {
        println!(
            "{}",
            style("Tip: use --refresh to bypass cache and fetch fresh data").dim()
        );
    }

    Ok(())
}

/// Display a list of provider models, grouped by capability.
fn display_provider_models(models: &[AvailableModel], active_model: &str) {
    let mut sorted: Vec<&AvailableModel> = models.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    for model in sorted {
        let active_marker = if model.id == active_model {
            " (active)"
        } else {
            ""
        };

        let caps: Vec<String> = model.capabilities.iter().map(|c| c.to_string()).collect();
        let cap_str = if caps.is_empty() {
            String::new()
        } else {
            format!(" [{}]", caps.join(", "))
        };

        let context_str = model
            .context_window
            .map(|c| format!(" [context: {}K]", c / 1000))
            .unwrap_or_default();

        let name_str = model
            .display_name
            .as_ref()
            .map(|n| format!(" - {}", n))
            .unwrap_or_default();

        println!(
            "  {}{}{}{}{}",
            style(&model.id).cyan(),
            name_str,
            context_str,
            style(&cap_str).dim(),
            style(active_marker).green()
        );
    }
    println!();
}

/// Original Brainwires SaaS model listing.
async fn handle_brainwires_models_list() -> Result<()> {
    println!("\n{}\n", style("Available Models:").cyan().bold());

    let models = ModelRegistry::get_all_models().await?;

    if models.is_empty() {
        eprintln!("{}", style("No models available").red());
        return Ok(());
    }

    let mut vendor_groups: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for model in models {
        vendor_groups
            .entry(model.ai_vendor.clone())
            .or_insert_with(Vec::new)
            .push(model);
    }

    for (vendor, mut models) in vendor_groups {
        models.sort_by(|a, b| a.id.cmp(&b.id));

        println!("{}:", style(&vendor).bold().yellow());
        for model in models {
            let default_marker = if model.is_default { " (default)" } else { "" };
            let context = format!("{}K", model.context_window / 1000);
            println!(
                "  {} - {} [context: {}]{}",
                style(&model.id).cyan(),
                model.name,
                context,
                style(default_marker).green()
            );
        }
        println!();
    }

    println!(
        "{}",
        style("Tip: use 'brainwires models stats' to see model statistics").dim()
    );

    Ok(())
}

async fn handle_models_stats(provider: Option<String>, refresh: bool) -> Result<()> {
    let (provider_type, api_key, base_url) = resolve_provider(provider.as_deref())?;

    // Brainwires SaaS → existing flow
    if provider_type == ProviderType::Brainwires {
        return handle_brainwires_models_stats().await;
    }

    println!(
        "\n{} ({})\n",
        style("Model Statistics:").cyan().bold(),
        style(provider_type.as_str()).yellow()
    );

    let use_cache = !refresh;
    let models = ModelService::list_models_for_provider(
        provider_type,
        api_key.as_deref(),
        base_url.as_deref(),
        use_cache,
    )
    .await?;

    if models.is_empty() {
        eprintln!("{}", style("No models available").red());
        return Ok(());
    }

    // Capability breakdown
    let chat_count = models.iter().filter(|m| m.is_chat_capable()).count();
    let embedding_count = models
        .iter()
        .filter(|m| {
            m.capabilities
                .contains(&crate::providers::ModelCapability::Embedding)
        })
        .count();
    let vision_count = models
        .iter()
        .filter(|m| {
            m.capabilities
                .contains(&crate::providers::ModelCapability::Vision)
        })
        .count();

    println!("{}:", style("Overall").bold());
    println!("  Total models: {}", style(models.len()).cyan());
    println!("  Chat models: {}", style(chat_count).cyan());
    if embedding_count > 0 {
        println!("  Embedding models: {}", style(embedding_count).cyan());
    }
    if vision_count > 0 {
        println!("  Vision-capable: {}", style(vision_count).cyan());
    }
    println!();

    // Context window stats (only for models that report it)
    let with_context: Vec<u32> = models.iter().filter_map(|m| m.context_window).collect();
    if !with_context.is_empty() {
        let total: u64 = with_context.iter().map(|&c| c as u64).sum();
        let avg = total / with_context.len() as u64;
        let max = with_context.iter().max().copied().unwrap_or(0);
        let min = with_context.iter().min().copied().unwrap_or(0);

        println!("{}:", style("Context Windows").bold());
        println!("  Average: {}K", style(avg / 1000).cyan());
        println!("  Maximum: {}K", style(max / 1000).cyan());
        println!("  Minimum: {}K", style(min / 1000).cyan());
    }

    Ok(())
}

/// Original Brainwires SaaS model stats.
async fn handle_brainwires_models_stats() -> Result<()> {
    println!("\n{}\n", style("Model Statistics:").cyan().bold());

    let models = ModelRegistry::get_all_models().await?;

    if models.is_empty() {
        eprintln!("{}", style("No models available").red());
        return Ok(());
    }

    let mut vendor_groups: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for model in &models {
        vendor_groups
            .entry(model.ai_vendor.clone())
            .or_insert_with(Vec::new)
            .push(model);
    }

    println!("{}:", style("Overall").bold());
    println!("  Total models: {}", style(models.len()).cyan());
    println!("  AI vendors: {}", style(vendor_groups.len()).cyan());
    println!();

    println!("{}:", style("By AI Vendor").bold());
    for (vendor, models) in &vendor_groups {
        println!("  {}: {} models", style(vendor).yellow(), models.len());
    }
    println!();

    let total_context: u64 = models.iter().map(|m| m.context_window as u64).sum();
    let avg_context = total_context / models.len() as u64;
    let max_context = models.iter().map(|m| m.context_window).max().unwrap_or(0);
    let min_context = models.iter().map(|m| m.context_window).min().unwrap_or(0);

    println!("{}:", style("Context Windows").bold());
    println!("  Average: {}K", style(avg_context / 1000).cyan());
    println!("  Maximum: {}K", style(max_context / 1000).cyan());
    println!("  Minimum: {}K", style(min_context / 1000).cyan());
    println!();

    if let Some(default_model) = models.iter().find(|m| m.is_default) {
        println!("{}:", style("Default Model").bold());
        println!(
            "  {} - {} ({})",
            style(&default_model.id).cyan(),
            default_model.name,
            style(&default_model.ai_vendor).yellow()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_commands_list() {
        let cmd = ModelsCommands::List {
            provider: None,
            all: false,
            refresh: false,
        };
        assert!(matches!(cmd, ModelsCommands::List { .. }));
    }

    #[test]
    fn test_models_commands_stats() {
        let cmd = ModelsCommands::Stats {
            provider: None,
            refresh: false,
        };
        assert!(matches!(cmd, ModelsCommands::Stats { .. }));
    }
}
