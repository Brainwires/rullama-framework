use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::message::{ChatResponse, Message, StreamChunk};
use crate::tool::Tool;

/// Base provider trait for AI providers
#[async_trait]
pub trait Provider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// Get the model's maximum output tokens (for setting appropriate limits)
    /// Returns None if the model doesn't have a specific limit
    fn max_output_tokens(&self) -> Option<u32> {
        None // Default implementation - providers can override
    }

    /// Chat completion (non-streaming)
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// Chat completion (streaming)
    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>>;
}

/// Prompt-cache strategy for providers that support explicit caching
/// (Anthropic Messages API today; a no-op elsewhere).
///
/// Controls which parts of a request receive `cache_control` breakpoints.
/// Caching reuses cached prompt bytes across turns for a 50–90% input-token
/// discount on subsequent calls, at the cost of a one-time "creation" charge
/// on first population.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheStrategy {
    /// No cache breakpoints. Fresh compute on every call.
    Off,
    /// Cache only the system prompt.
    SystemOnly,
    /// Cache the system prompt and tool definitions (the default).
    #[default]
    SystemAndTools,
    /// Cache system + tools + the tail of the conversation once the message
    /// history reaches the given approximate token threshold.
    SystemAndTailTurn {
        /// Minimum conversation size (approximate tokens) before the tail
        /// breakpoint is emitted. Avoids wasting a cache slot on short chats.
        threshold_tokens: u32,
    },
}

/// Chat completion options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Temperature (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Per-request model override.
    ///
    /// When `Some`, providers MUST use this model name instead of their default.
    /// This enables per-session model switching without replacing the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Prompt-cache strategy. Ignored by providers without prompt caching.
    #[serde(default)]
    pub cache_strategy: CacheStrategy,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            max_tokens: Some(4096),
            top_p: None,
            stop: None,
            system: None,
            model: None,
            cache_strategy: CacheStrategy::default(),
        }
    }
}

impl ChatOptions {
    /// Create new chat options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set system prompt
    pub fn system<S: Into<String>>(mut self, system: S) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set top-p sampling
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Override the model for this request.
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the prompt-cache strategy.
    pub fn cache_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.cache_strategy = strategy;
        self
    }

    /// Deterministic classification/routing (temp=0, few tokens)
    pub fn deterministic(max_tokens: u32) -> Self {
        Self {
            temperature: Some(0.0),
            max_tokens: Some(max_tokens),
            ..Default::default()
        }
    }

    /// Low-temperature factual generation
    pub fn factual(max_tokens: u32) -> Self {
        Self {
            temperature: Some(0.1),
            max_tokens: Some(max_tokens),
            top_p: Some(0.9),
            ..Default::default()
        }
    }

    /// Creative generation with moderate temperature
    pub fn creative(max_tokens: u32) -> Self {
        Self {
            temperature: Some(0.3),
            max_tokens: Some(max_tokens),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_options_default() {
        let opts = ChatOptions::default();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.max_tokens, Some(4096));
    }

    #[test]
    fn test_chat_options_builder() {
        let opts = ChatOptions::new()
            .temperature(0.5)
            .max_tokens(2048)
            .system("Test");
        assert_eq!(opts.temperature, Some(0.5));
        assert_eq!(opts.max_tokens, Some(2048));
        assert_eq!(opts.system, Some("Test".to_string()));
    }

    #[test]
    fn test_chat_options_deterministic() {
        let opts = ChatOptions::deterministic(50);
        assert_eq!(opts.temperature, Some(0.0));
        assert_eq!(opts.max_tokens, Some(50));
    }

    #[test]
    fn test_chat_options_factual() {
        let opts = ChatOptions::factual(200);
        assert_eq!(opts.temperature, Some(0.1));
        assert_eq!(opts.max_tokens, Some(200));
        assert_eq!(opts.top_p, Some(0.9));
    }

    #[test]
    fn test_chat_options_creative() {
        let opts = ChatOptions::creative(400);
        assert_eq!(opts.temperature, Some(0.3));
        assert_eq!(opts.max_tokens, Some(400));
    }
}
