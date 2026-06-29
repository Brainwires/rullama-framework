//! Example: Multi-provider chat using the facade crate
//!
//! Demonstrates how to configure and create multiple AI providers using
//! `ProviderConfig`, `ChatProviderFactory`, and `ProviderType`. Each provider
//! is created from a registry-driven factory and satisfies the `Provider` trait,
//! so you can swap backends without changing your chat logic.
//!
//! Run: cargo run -p rullama --example multi_provider_chat --features providers,chat

use anyhow::Result;
use rullama::prelude::*;

// The facade re-exports provider types under `rullama::providers::*`
// and chat factory types under `rullama::chat::*`.
use rullama::chat::ChatProviderFactory;
use rullama::providers::{ProviderConfig, ProviderType};

/// Send a single user message through a provider and print the response.
async fn chat_with_provider(provider: &dyn Provider, prompt: &str) -> Result<()> {
    let messages = vec![Message::user(prompt)];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await?;
    let reply = response.message.text().unwrap_or_default();

    println!("  Model reply : {}", reply);
    println!(
        "  Token usage : {} prompt + {} completion",
        response.usage.prompt_tokens, response.usage.completion_tokens,
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. Ollama (local, no API key required) ──────────────────────────
    //
    // Ollama runs locally so it never needs a key. This makes it ideal for
    // development and CI.
    println!("=== Ollama (local) ===");
    let ollama_config = ProviderConfig::new(ProviderType::Ollama, "llama3.1".to_string());
    let ollama = ChatProviderFactory::create(&ollama_config)?;
    println!("  Provider name: {}", ollama.name());

    // ── 2. OpenAI ───────────────────────────────────────────────────────
    //
    // The builder pattern lets you chain `.with_api_key()` for providers
    // that require authentication.
    println!("\n=== OpenAI ===");
    let openai_config = ProviderConfig::new(ProviderType::OpenAI, "gpt-4o-mini".to_string())
        .with_api_key("sk-demo-key");
    let openai = ChatProviderFactory::create(&openai_config)?;
    println!("  Provider name: {}", openai.name());

    // ── 3. Anthropic ────────────────────────────────────────────────────
    println!("\n=== Anthropic ===");
    let anthropic_config = ProviderConfig::new(
        ProviderType::Anthropic,
        "claude-sonnet-4-20250514".to_string(),
    )
    .with_api_key("sk-ant-demo-key");
    let anthropic = ChatProviderFactory::create(&anthropic_config)?;
    println!("  Provider name: {}", anthropic.name());

    // ── 4. Groq (OpenAI-compatible endpoint) ────────────────────────────
    //
    // Groq, Together, Fireworks, and Anyscale all use the OpenAI Chat
    // Completions protocol — the factory dispatches to the correct base URL
    // automatically based on `ProviderType`.
    println!("\n=== Groq ===");
    let groq_config =
        ProviderConfig::new(ProviderType::Groq, "llama-3.3-70b-versatile".to_string())
            .with_api_key("gsk_demo");
    let groq = ChatProviderFactory::create(&groq_config)?;
    println!("  Provider name: {}", groq.name());

    // ── 5. Send a chat request through Ollama ───────────────────────────
    //
    // All providers implement `dyn Provider`, so the same chat logic works
    // regardless of the backend. We only demo Ollama here because it does
    // not require a real API key.
    println!("\n=== Chat demo (Ollama) ===");
    if let Err(e) = chat_with_provider(ollama.as_ref(), "Say hello in one sentence.").await {
        // Expected to fail if Ollama is not running — that's OK for a demo.
        println!("  (Ollama not reachable: {e})");
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("\nAll providers created successfully. Swap ProviderType to switch backends!");

    Ok(())
}
