//! Example: Using the ChatProviderFactory and model listing
//!
//! Demonstrates how to use `ProviderConfig` with the builder pattern,
//! create chat providers via `ChatProviderFactory`, and list available
//! models with the `ModelLister` trait.
//!
//! Run: cargo run -p brainwires-provider --example provider_factory --features native

use brainwires_provider::{
    ChatProviderFactory, ProviderConfig, ProviderType, create_model_lister, registry,
};

fn main() {
    println!("=== Provider Factory & Model Listing Example ===\n");

    // ── 1. Browse the provider registry ─────────────────────────────────
    println!("--- Known Chat Providers ---");
    for entry in registry::PROVIDER_REGISTRY {
        println!(
            "  {:12} | protocol: {:?} | default model: {}",
            entry.provider_type, entry.chat_protocol, entry.default_model,
        );
    }
    println!();

    // ── 2. Build configs with the ProviderConfig builder ────────────────
    println!("--- Building Provider Configs ---");

    // Ollama does not require an API key
    let ollama_config = ProviderConfig::new(ProviderType::Ollama, "llama3.3".into())
        .with_base_url("http://localhost:11434");
    println!("Ollama config: {:?}", ollama_config);

    // OpenAI needs an API key
    let openai_config = ProviderConfig::new(ProviderType::OpenAI, "gpt-5-mini".into())
        .with_api_key("sk-demo-key-not-real");
    println!("OpenAI config: {:?}", openai_config);

    // Groq uses the OpenAI-compatible protocol with a different base URL
    let groq_config = ProviderConfig::new(ProviderType::Groq, "llama-3.3-70b-versatile".into())
        .with_api_key("gsk-demo-key-not-real");
    println!("Groq   config: {:?}", groq_config);
    println!();

    // ── 3. Create providers via the factory ─────────────────────────────
    println!("--- Creating Providers via ChatProviderFactory ---");

    // Ollama succeeds without a key
    match ChatProviderFactory::create(&ollama_config) {
        Ok(provider) => println!("  Created '{}' provider successfully", provider.name()),
        Err(e) => println!("  Failed to create Ollama provider: {}", e),
    }

    // OpenAI succeeds with a key (no network call at creation time)
    match ChatProviderFactory::create(&openai_config) {
        Ok(provider) => println!("  Created '{}' provider successfully", provider.name()),
        Err(e) => println!("  Failed to create OpenAI provider: {}", e),
    }

    // Groq is dispatched via OpenAI-compatible protocol
    match ChatProviderFactory::create(&groq_config) {
        Ok(provider) => println!("  Created '{}' provider successfully", provider.name()),
        Err(e) => println!("  Failed to create Groq provider: {}", e),
    }

    // Audio-only providers are rejected by the chat factory
    let elevenlabs_config =
        ProviderConfig::new(ProviderType::ElevenLabs, "eleven_multilingual_v2".into())
            .with_api_key("el-demo");
    match ChatProviderFactory::create(&elevenlabs_config) {
        Ok(_) => println!("  Unexpected: ElevenLabs should not be a chat provider"),
        Err(e) => println!("  Expected rejection for ElevenLabs: {}", e),
    }
    println!();

    // ── 4. Default models per provider ──────────────────────────────────
    println!("--- Default Models ---");
    let provider_types = [
        ProviderType::Anthropic,
        ProviderType::OpenAI,
        ProviderType::Google,
        ProviderType::Groq,
        ProviderType::Ollama,
        ProviderType::Together,
    ];
    for pt in &provider_types {
        println!("  {:12} -> {}", pt, pt.default_model());
    }
    println!();

    // ── 5. Model lister creation (no actual API calls) ──────────────────
    println!("--- Model Lister Availability ---");
    for pt in &provider_types {
        let requires_key = pt.requires_api_key();
        let key = if requires_key { Some("demo-key") } else { None };
        match create_model_lister(*pt, key, None) {
            Ok(_) => println!("  {:12} -> ModelLister created (would query API)", pt),
            Err(e) => println!("  {:12} -> {}", pt, e),
        }
    }
    println!();

    // ── 6. Inspect model capabilities helper ────────────────────────────
    println!("--- Capability Inference (OpenAI-format IDs) ---");
    let test_ids = [
        "gpt-4o",
        "gpt-3.5-turbo",
        "text-embedding-3-small",
        "dall-e-3",
        "whisper-1",
    ];
    for id in &test_ids {
        let caps = brainwires_provider::model_listing::infer_openai_capabilities(id);
        let names: Vec<String> = caps.iter().map(|c| format!("{}", c)).collect();
        println!("  {:30} -> [{}]", id, names.join(", "));
    }

    println!("\nDone! In a real application you would call provider.chat() or");
    println!("lister.list_models().await to interact with the APIs.");
}
