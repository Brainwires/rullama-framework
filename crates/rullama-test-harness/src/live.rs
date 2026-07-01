//! Environment-driven gating for Tier-D live-provider cases.
//!
//! Tier-D cases hit real provider APIs (Ollama / OpenAI / Anthropic). To keep
//! the default `cargo xtask test-harness run` offline and free, each live case
//! checks the relevant env var first. If absent, the case returns
//! [`TrialResult::skipped`] so the run records it as PASS-equivalent without
//! tanking the success rate.
//!
//! Convention: use `RULLAMA_LIVE_*` rather than reading raw `OPENAI_API_KEY`
//! / `ANTHROPIC_API_KEY` directly, so the harness can never accidentally pick
//! up keys from an unrelated CI environment.

/// Returns the Ollama base URL if `RULLAMA_LIVE_OLLAMA_BASE` is set.
pub fn live_ollama_base() -> Option<String> {
    std::env::var("RULLAMA_LIVE_OLLAMA_BASE").ok()
}

/// Returns the Ollama model name if `RULLAMA_LIVE_OLLAMA_MODEL` is set,
/// defaulting to `gemma4:e2b` when only the base URL is configured.
pub fn live_ollama_model() -> String {
    std::env::var("RULLAMA_LIVE_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string())
}

/// Returns the OpenAI API key if `RULLAMA_LIVE_OPENAI_KEY` is set.
pub fn live_openai_key() -> Option<String> {
    std::env::var("RULLAMA_LIVE_OPENAI_KEY").ok()
}

/// Returns the OpenAI model name, defaulting to `gpt-5-nano`.
pub fn live_openai_model() -> String {
    std::env::var("RULLAMA_LIVE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-5-nano".to_string())
}

/// Returns the Anthropic API key if `RULLAMA_LIVE_ANTHROPIC_KEY` is set.
pub fn live_anthropic_key() -> Option<String> {
    std::env::var("RULLAMA_LIVE_ANTHROPIC_KEY").ok()
}

/// Returns the Anthropic model name, defaulting to `claude-haiku-4-5`.
pub fn live_anthropic_model() -> String {
    std::env::var("RULLAMA_LIVE_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-haiku-4-5".to_string())
}
