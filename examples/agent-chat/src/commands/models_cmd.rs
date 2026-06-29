use anyhow::Result;

use brainwires_provider::{ProviderType, create_model_lister};

use crate::auth;
use crate::cli::ModelsArgs;

/// Providers that support live model listing.
const LISTABLE_PROVIDERS: &[(&str, ProviderType)] = &[
    ("anthropic", ProviderType::Anthropic),
    ("openai", ProviderType::OpenAI),
    ("google", ProviderType::Google),
    ("groq", ProviderType::Groq),
    ("together", ProviderType::Together),
    ("fireworks", ProviderType::Fireworks),
    ("ollama", ProviderType::Ollama),
];

pub async fn run(args: ModelsArgs) -> Result<()> {
    let filter = args.provider.as_deref().map(|s| s.to_lowercase());

    let providers: Vec<_> = LISTABLE_PROVIDERS
        .iter()
        .filter(|(name, _)| filter.as_deref().is_none_or(|f| *name == f))
        .collect();

    if providers.is_empty() {
        if let Some(ref f) = filter {
            eprintln!("Unknown or unsupported provider: {f}");
        }
        return Ok(());
    }

    for (name, provider_type) in providers {
        let api_key = auth::resolve_api_key(name, None)?;

        let lister = match create_model_lister(*provider_type, api_key.as_deref(), None) {
            Ok(l) => l,
            Err(_) => {
                println!("{name}:");
                println!("  (no API key configured — set with: agent-chat auth set {name})");
                println!();
                continue;
            }
        };

        match lister.list_models().await {
            Ok(models) => {
                let mut chat_models: Vec<_> =
                    models.into_iter().filter(|m| m.is_chat_capable()).collect();
                chat_models.sort_by(|a, b| a.id.cmp(&b.id));

                println!("{name}: ({} chat models)", chat_models.len());
                for model in &chat_models {
                    let display = model
                        .display_name
                        .as_deref()
                        .map(|d| format!("  {:<45} {d}", model.id))
                        .unwrap_or_else(|| format!("  {}", model.id));
                    println!("{display}");
                }
                println!();
            }
            Err(e) => {
                println!("{name}:");
                println!("  (failed to fetch models: {e})");
                println!();
            }
        }
    }

    Ok(())
}
