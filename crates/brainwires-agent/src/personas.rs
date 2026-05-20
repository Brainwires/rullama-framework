//! Pluggable persona construction for any chat agent (e.g. `ChatAgent` in
//! `brainwires-inference`).
//!
//! Previously every app rolled its own system-prompt assembly: `agent-chat`
//! read CLI args, `brainwires-cli` stitched a config value, `brainclaw` had a
//! `/persona` command, and `voice-assistant` hardcoded. The
//! [`PersonaProvider`] trait gives them a single contract: given some
//! lightweight context, return the content blocks that should populate the
//! system turn.
//!
//! A [`CompositePersonaProvider`] chains multiple providers so apps can
//! assemble a base persona, then RAG injections, then entity memory, then
//! locale/time — without each layer knowing about the others.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use brainwires_core::ContentBlock;

/// Context passed to a [`PersonaProvider`] before each turn.
///
/// Fields are all optional so providers can ignore whatever they don't care
/// about (a static-text provider ignores everything; a RAG-injection
/// provider reads `user_id` and `last_user_message`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersonaContext {
    /// Stable user identifier, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Stable session identifier, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// BCP-47 locale tag such as `"en-US"`, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// Names of tools the agent is about to run with.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// The user's latest message, when the provider needs it to pull in
    /// relevant retrievals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_user_message: Option<String>,
}

impl PersonaContext {
    /// Start a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the user id.
    pub fn with_user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Set the session id.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Set the locale.
    pub fn with_locale(mut self, locale: impl Into<String>) -> Self {
        self.locale = Some(locale.into());
        self
    }
}

/// Extension point for system-prompt assembly.
#[async_trait]
pub trait PersonaProvider: Send + Sync {
    /// Produce the content blocks that should make up the system turn for
    /// this conversation. May return an empty vec when the provider is a
    /// no-op for the given context.
    async fn build(&self, ctx: &PersonaContext) -> Result<Vec<ContentBlock>>;
}

/// A persona provider that always returns the same single text block.
pub struct StaticPersonaProvider {
    text: String,
}

impl StaticPersonaProvider {
    /// Build a static provider from any string-like value.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[async_trait]
impl PersonaProvider for StaticPersonaProvider {
    async fn build(&self, _ctx: &PersonaContext) -> Result<Vec<ContentBlock>> {
        if self.text.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![ContentBlock::Text {
            text: self.text.clone(),
        }])
    }
}

/// Chain multiple providers — their outputs concatenate in declaration order.
///
/// A typical stack:
///
/// 1. `StaticPersonaProvider` with the base identity.
/// 2. A RAG retriever that injects relevant entities for `ctx.user_id`.
/// 3. A locale / time provider.
pub struct CompositePersonaProvider {
    providers: Vec<Arc<dyn PersonaProvider>>,
}

impl CompositePersonaProvider {
    /// Build a composite from an ordered list of providers.
    pub fn new(providers: Vec<Arc<dyn PersonaProvider>>) -> Self {
        Self { providers }
    }

    /// Append a provider to the chain.
    pub fn push(mut self, p: Arc<dyn PersonaProvider>) -> Self {
        self.providers.push(p);
        self
    }
}

#[async_trait]
impl PersonaProvider for CompositePersonaProvider {
    async fn build(&self, ctx: &PersonaContext) -> Result<Vec<ContentBlock>> {
        let mut out = Vec::new();
        for p in &self.providers {
            out.extend(p.build(ctx).await?);
        }
        Ok(out)
    }
}

/// Collapse a list of content blocks into a single system-prompt string.
///
/// Non-text blocks (images, tool use/result) are rendered as one-line
/// placeholders — the system turn is almost always pure text, and this
/// matches what every provider's `ChatOptions::system` field accepts.
pub fn blocks_to_system_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            ContentBlock::Text { text } => {
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str(text);
            }
            ContentBlock::Image { .. } => {
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str("[persona: image attachment omitted]");
            }
            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {
                // Persona providers shouldn't emit tool blocks, but be
                // defensive — silently drop them.
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_provider_returns_single_block() {
        let p = StaticPersonaProvider::new("you are helpful");
        let blocks = p.build(&PersonaContext::new()).await.unwrap();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text, "you are helpful"),
            _ => panic!("expected text block"),
        }
    }

    #[tokio::test]
    async fn static_empty_returns_empty() {
        let p = StaticPersonaProvider::new("");
        let blocks = p.build(&PersonaContext::new()).await.unwrap();
        assert!(blocks.is_empty());
    }

    #[tokio::test]
    async fn composite_chains_in_order() {
        let a = Arc::new(StaticPersonaProvider::new("base")) as Arc<dyn PersonaProvider>;
        let b = Arc::new(StaticPersonaProvider::new("addendum")) as Arc<dyn PersonaProvider>;
        let composite = CompositePersonaProvider::new(vec![a, b]);
        let blocks = composite.build(&PersonaContext::new()).await.unwrap();
        assert_eq!(blocks.len(), 2);
        match (&blocks[0], &blocks[1]) {
            (ContentBlock::Text { text: t1 }, ContentBlock::Text { text: t2 }) => {
                assert_eq!(t1, "base");
                assert_eq!(t2, "addendum");
            }
            _ => panic!("expected two text blocks in order"),
        }
    }

    #[tokio::test]
    async fn composite_push_extends() {
        let a = Arc::new(StaticPersonaProvider::new("first")) as Arc<dyn PersonaProvider>;
        let b = Arc::new(StaticPersonaProvider::new("second")) as Arc<dyn PersonaProvider>;
        let composite = CompositePersonaProvider::new(vec![a]).push(b);
        let text = blocks_to_system_text(&composite.build(&PersonaContext::new()).await.unwrap());
        assert_eq!(text, "first\n\nsecond");
    }

    #[test]
    fn blocks_to_system_text_joins_and_escapes() {
        let blocks = vec![
            ContentBlock::Text { text: "one".into() },
            ContentBlock::Text { text: "two".into() },
        ];
        assert_eq!(blocks_to_system_text(&blocks), "one\n\ntwo");
    }

    #[test]
    fn context_builders() {
        let c = PersonaContext::new()
            .with_user_id("u1")
            .with_session_id("s1")
            .with_locale("en-US");
        assert_eq!(c.user_id.as_deref(), Some("u1"));
        assert_eq!(c.session_id.as_deref(), Some("s1"));
        assert_eq!(c.locale.as_deref(), Some("en-US"));
    }
}
