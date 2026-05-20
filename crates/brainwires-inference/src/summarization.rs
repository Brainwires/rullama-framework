//! LLM-powered conversation summarization.
//!
//! Long-running agent sessions accumulate message history that eventually
//! overflows the provider's context window. Simple trimming (the legacy
//! behavior of [`ChatAgent::compact_history`](crate::chat_agent::ChatAgent::compact_history))
//! drops information the agent may still need. A summarizer compresses the
//! dropped section into a single synthetic assistant turn that preserves the
//! important facts, tool-call outcomes, and decisions.
//!
//! The [`Summarizer`] trait is the extension point:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use brainwires_agent::summarization::{LlmSummarizer, Summarizer};
//!
//! let summarizer = LlmSummarizer::new(provider.clone());
//! let summary = summarizer.summarize(&old_messages).await?;
//! ```
//!
//! [`LlmSummarizer`] uses a small second-cheap prompt and is designed to work
//! with any `brainwires_core::Provider` — typically pointed at a cheaper
//! model (Haiku, gpt-4o-mini, Flash) than the main agent.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use brainwires_core::{ChatOptions, Message, Provider};

/// Compresses a run of messages into a single-turn summary.
///
/// Implementations must preserve:
/// - Tool-call outcomes (not raw tool JSON)
/// - Decisions the agent made
/// - Commitments/constraints the user expressed
///
/// They should discard:
/// - Verbatim tool-call arguments
/// - Verbose intermediate reasoning
/// - Redundant pleasantries
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Produce a compact textual summary of `messages`.
    ///
    /// The returned string is embedded verbatim into the agent's history as
    /// a synthetic assistant turn, so it should read naturally to a downstream
    /// LLM that picks up the conversation.
    async fn summarize(&self, messages: &[Message]) -> Result<String>;
}

/// Default summarizer: calls a provided `Provider` with a compact instruction.
///
/// The summarizer's provider does not need to match the host agent's
/// provider — point it at a cheaper model (Haiku, gpt-4o-mini, Flash) for a
/// ~90% cost reduction on summarization calls.
pub struct LlmSummarizer {
    provider: Arc<dyn Provider>,
    options: ChatOptions,
    system_prompt: String,
}

impl LlmSummarizer {
    /// Default system prompt used by [`LlmSummarizer`]. Tuned for terse,
    /// tool-aware summaries that preserve downstream coherence.
    pub const DEFAULT_SYSTEM_PROMPT: &'static str = concat!(
        "You compress conversation history for an AI agent. ",
        "Produce a single compact summary (5-15 sentences) that preserves: ",
        "tool-call outcomes and errors, decisions the assistant made, ",
        "commitments or constraints the user expressed, and any unresolved ",
        "questions. Discard: raw tool arguments, intermediate reasoning, ",
        "verbatim source code, and pleasantries. Write in past tense as a ",
        "neutral observer ('The assistant read X and concluded Y. The user ",
        "asked for Z.'). Output plain text only — no headings or markdown."
    );

    /// Build a summarizer over the given provider. Uses the default system
    /// prompt and deterministic options.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            options: ChatOptions::default().temperature(0.0).max_tokens(1024),
            system_prompt: Self::DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Override the system prompt (for red-teaming, style changes, or
    /// domain-specific compression rules).
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Override the chat options passed to the underlying provider.
    pub fn with_options(mut self, options: ChatOptions) -> Self {
        self.options = options;
        self
    }
}

#[async_trait]
impl Summarizer for LlmSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // Render the transcript as a single user-turn payload so the
        // summarizer model sees every role clearly labelled. This is robust
        // to providers whose `Messages` API would otherwise reject arbitrary
        // role sequences or tool-use blocks as input.
        let transcript = render_transcript(messages);

        let mut opts = self.options.clone();
        opts.system = Some(self.system_prompt.clone());

        let payload = format!(
            "Summarize the following conversation per the system-prompt rules. \
             Return the summary only — no preamble.\n\n---\n{transcript}\n---"
        );

        let resp = self
            .provider
            .chat(&[Message::user(payload)], None, &opts)
            .await?;
        Ok(resp.message.text().unwrap_or_default().trim().to_string())
    }
}

/// Render messages as a single plain-text transcript. Tool use/result blocks
/// are serialised to one-line descriptions so the summarizer can see what
/// happened without being swamped by JSON.
fn render_transcript(messages: &[Message]) -> String {
    use brainwires_core::{ContentBlock, MessageContent, Role};

    let mut out = String::new();
    for m in messages {
        let role = match m.role {
            Role::System => "SYSTEM",
            Role::User => "USER",
            Role::Assistant => "ASSISTANT",
            Role::Tool => "TOOL",
        };
        match &m.content {
            MessageContent::Text(t) => {
                out.push_str(&format!("{role}: {t}\n"));
            }
            MessageContent::Blocks(blocks) => {
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => {
                            out.push_str(&format!("{role}: {text}\n"));
                        }
                        ContentBlock::ToolUse { name, .. } => {
                            out.push_str(&format!("{role}: (called tool `{name}`)\n"));
                        }
                        ContentBlock::ToolResult {
                            content, is_error, ..
                        } => {
                            let status = if is_error.unwrap_or(false) {
                                "err"
                            } else {
                                "ok"
                            };
                            let snippet: String = content.chars().take(200).collect();
                            out.push_str(&format!("{role}: (tool {status}) {snippet}\n"));
                        }
                        ContentBlock::Image { .. } => {
                            out.push_str(&format!("{role}: (image attached)\n"));
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{
        ChatResponse, ContentBlock, Message, MessageContent, Provider, Role, StreamChunk, Tool,
        Usage,
    };
    use futures::stream::BoxStream;

    struct EchoingProvider;

    #[async_trait]
    impl Provider for EchoingProvider {
        fn name(&self) -> &str {
            "echo"
        }
        async fn chat(
            &self,
            messages: &[Message],
            _: Option<&[Tool]>,
            _: &ChatOptions,
        ) -> Result<ChatResponse> {
            // The summarizer sends exactly one user message containing the
            // rendered transcript. Echo a short "summary:" prefix plus the
            // first line of that message to prove we received the right input.
            let last = messages.last().and_then(|m| m.text()).unwrap_or_default();
            let first_line = last.lines().next().unwrap_or_default();
            Ok(ChatResponse {
                message: Message::assistant(format!("summary-of: {first_line}")),
                usage: Usage::new(10, 4),
                finish_reason: Some("stop".into()),
            })
        }
        fn stream_chat<'a>(
            &'a self,
            _: &'a [Message],
            _: Option<&'a [Tool]>,
            _: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    #[tokio::test]
    async fn summarizes_empty_history() {
        let s = LlmSummarizer::new(Arc::new(EchoingProvider));
        assert_eq!(s.summarize(&[]).await.unwrap(), "");
    }

    #[tokio::test]
    async fn renders_mixed_content_into_transcript() {
        let msgs = vec![
            Message::system("be helpful"),
            Message::user("read foo.rs"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "I'll read it.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "read_file".into(),
                        input: serde_json::json!({"path":"foo.rs"}),
                    },
                ]),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: "fn main() {}".into(),
                    is_error: Some(false),
                }]),
                name: None,
                metadata: None,
            },
        ];

        let rendered = render_transcript(&msgs);
        assert!(rendered.contains("SYSTEM: be helpful"));
        assert!(rendered.contains("USER: read foo.rs"));
        assert!(rendered.contains("ASSISTANT: I'll read it."));
        assert!(rendered.contains("ASSISTANT: (called tool `read_file`)"));
        assert!(rendered.contains("USER: (tool ok) fn main() {}"));
    }

    #[tokio::test]
    async fn llm_summarizer_invokes_provider_and_returns_text() {
        let s = LlmSummarizer::new(Arc::new(EchoingProvider));
        let msgs = vec![Message::user("hello world"), Message::assistant("hi!")];
        let summary = s.summarize(&msgs).await.unwrap();
        assert!(summary.starts_with("summary-of:"));
    }
}
