//! End-to-end example: a CLI that takes a question and asks an Ollama-backed
//! chat agent to answer it, with the framework's `fetch_url` tool available
//! for reaching out to the open web. Demonstrates wiring `ChatAgent`,
//! `BuiltinToolExecutor`, `BudgetProvider`, and `AgentBuilder` against a real
//! provider in ~80 lines.
//!
//! Usage:
//!
//! ```bash
//! # default backend is http://localhost:11434, default model is "gemma4:e2b"
//! cargo run -p brainwires-web-search-agent -- "what is the capital of France?"
//!
//! # override via env vars
//! OLLAMA_BASE_URL=http://my-host:11434 \
//! OLLAMA_DEFAULT_MODEL=llama3:8b \
//!     cargo run -p brainwires-web-search-agent -- "summarize https://example.com"
//! ```

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use brainwires_call_policy::{BudgetConfig, BudgetGuard, BudgetProvider};
use brainwires_core::{ChatOptions, Provider, ToolContext};
use brainwires_inference::AgentBuilder;
use brainwires_provider::OllamaProvider;
use brainwires_tool_builtins::{BuiltinToolExecutor, WebTool};
use brainwires_tool_runtime::{ToolExecutor, ToolRegistry};

const SYSTEM_PROMPT: &str = "You are a concise research assistant. \
When a question requires up-to-date information or a specific URL's content, \
call the `fetch_url` tool. Otherwise answer directly. Keep answers under 4 sentences.";

#[tokio::main]
async fn main() -> Result<()> {
    let question = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: brainwires-web-search-agent <question>"))?;

    let base_url = std::env::var("OLLAMA_BASE_URL").ok();
    let model =
        std::env::var("OLLAMA_DEFAULT_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());

    let ollama = Arc::new(OllamaProvider::new(model.clone(), base_url));

    let guard = BudgetGuard::new(BudgetConfig {
        max_tokens: Some(20_000),
        max_usd_cents: Some(5),
        max_rounds: Some(8),
    });
    let provider: Arc<dyn Provider> =
        Arc::new(BudgetProvider::new(ollama, guard.clone()));

    let mut registry = ToolRegistry::new();
    registry.register_tools(WebTool::get_tools());
    let executor: Arc<dyn ToolExecutor> = Arc::new(BuiltinToolExecutor::new(
        registry,
        ToolContext::default(),
    ));

    let mut agent = AgentBuilder::new()
        .provider(provider)
        .tools(executor)
        .options(ChatOptions::default().model(model.clone()))
        .system(SYSTEM_PROMPT)
        .max_iterations(6)
        .budget(guard.clone())
        .build_chat_agent()
        .context("failed to build ChatAgent")?;

    let (answer, report) = agent
        .process_message_with_report(&question)
        .await
        .context("agent run failed")?;

    println!("\nQ: {question}\nA: {answer}\n");
    println!(
        "── usage ── prompt={} completion={} total={} duration={}ms",
        report.prompt_tokens, report.completion_tokens, report.total_tokens, report.duration_ms
    );
    println!(
        "── budget ── tokens_consumed={} usd_cents={} rounds={}",
        guard.tokens_consumed(),
        guard.usd_cents_consumed(),
        guard.rounds_consumed(),
    );

    Ok(())
}
