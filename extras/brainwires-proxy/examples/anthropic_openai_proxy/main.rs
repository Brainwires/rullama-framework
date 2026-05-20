//! # anthropic_openai_proxy example
//!
//! Proxies Anthropic Messages API → OpenAI Chat Completions API using the
//! brainwires-proxy framework. Based on the `claude-adapter` TypeScript project.
//!
//! ## Usage
//!
//! ```sh
//! cargo run -p brainwires-proxy --example anthropic_openai_proxy --features http -- \
//!   --config ./config.json
//! ```
//!
//! ## Config format
//!
//! ```json
//! {
//!   "providers": {
//!     "ollama": {
//!       "baseUrl": "http://localhost:11434",
//!       "apiKey": "ollama"
//!     }
//!   },
//!   "models": {
//!     "opus":   { "provider": "ollama", "model": "llama3.1" },
//!     "sonnet": { "provider": "ollama", "model": "llama3.1" },
//!     "haiku":  { "provider": "ollama", "model": "llama3.2" }
//!   }
//! }
//! ```
//!
//! ## Test
//!
//! ```sh
//! curl http://localhost:3080/health
//! curl -X POST http://localhost:3080/v1/messages \
//!   -H "Content-Type: application/json" \
//!   -H "x-api-key: dummy" \
//!   -d '{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[{"role":"user","content":"Say hello"}]}'
//! ```

mod adapter_layer;
mod config;
mod convert_request;
mod convert_response;
mod convert_sse;
mod convert_tools;
mod tool_name_mapper;
mod types_anthropic;
mod types_openai;

use adapter_layer::AdapterLayer;
use brainwires_proxy::builder::ProxyBuilder;
use clap::Parser;
use config::AdapterConfig;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(about = "Proxy Anthropic Messages API to OpenAI Chat Completions API")]
struct Args {
    /// Path to the adapter config JSON file
    #[arg(long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let config = AdapterConfig::load_from(args.config)?;
    let port = config.port.unwrap_or(3080);

    // Resolve the default model slot (sonnet) to discover the upstream URL.
    let resolved = config.resolve_model("claude-sonnet-4-20250514")?;
    let upstream_url = format!(
        "{}/v1/chat/completions",
        resolved.provider.base_url.trim_end_matches('/')
    );

    eprintln!("anthropic-openai-proxy (brainwires-proxy)");
    eprintln!("  listen:   http://127.0.0.1:{}", port);
    eprintln!("  upstream: {}", resolved.provider.base_url);
    eprintln!("  model:    sonnet → {}", resolved.target_model);

    let proxy = ProxyBuilder::new()
        .listen_on(&format!("127.0.0.1:{}", port))
        .upstream_url(&upstream_url)
        .layer(AdapterLayer::new(config))
        .with_logging()
        .timeout(Duration::from_secs(300))
        .build()?;

    proxy.run().await?;
    Ok(())
}
