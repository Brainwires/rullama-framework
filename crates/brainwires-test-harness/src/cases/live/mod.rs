//! Tier-D live-provider integration cases (D.1-D.7, D.11).
//!
//! All cases self-skip via [`crate::live`] env-var helpers when their
//! required key/URL is absent. The default `cargo xtask test-harness run`
//! (no `--tier=d`) doesn't include these at all.

pub mod anthropic_chat;
pub mod anthropic_streaming;
pub mod ollama_chat;
pub mod ollama_streaming;
pub mod ollama_tool_dispatch;
pub mod openai_chat;
pub mod openai_streaming;
pub mod usage_token_counts;
