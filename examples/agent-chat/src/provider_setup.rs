use anyhow::{Result, bail};
use std::sync::Arc;

use brainwires_core::Provider;
use brainwires_provider::{ChatProviderFactory, ProviderConfig, ProviderType};

use crate::auth;
use crate::cli::Cli;
use crate::config::ChatConfig;

pub fn create_provider(cli: &Cli, config: &ChatConfig) -> Result<Arc<dyn Provider>> {
    let provider_str = cli.provider.as_deref().unwrap_or(&config.default_provider);

    let provider_type: ProviderType = provider_str
        .parse()
        .map_err(|_| anyhow::anyhow!("Unknown provider: {provider_str}"))?;

    let model = cli
        .model
        .as_deref()
        .unwrap_or_else(|| provider_type.default_model())
        .to_string();

    let api_key = auth::resolve_api_key(provider_str, cli.api_key.as_deref())?;

    if provider_type.requires_api_key() && api_key.is_none() {
        bail!(
            "No API key found for provider '{provider_str}'.\n\
             Set it via:\n  \
             - Environment variable (e.g. ANTHROPIC_API_KEY)\n  \
             - agent-chat auth set {provider_str}\n  \
             - --api-key flag"
        );
    }

    let mut prov_config = ProviderConfig::new(provider_type, model);
    if let Some(key) = api_key {
        prov_config = prov_config.with_api_key(key);
    }

    ChatProviderFactory::create(&prov_config)
}
