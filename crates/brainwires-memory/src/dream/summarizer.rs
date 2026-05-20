//! LLM-powered conversation summariser for the dream consolidation pipeline.

use anyhow::Result;

use brainwires_core::{ChatOptions, Message, Provider};

/// Stateless helper that calls an LLM to summarise a batch of messages.
pub struct DreamSummarizer;

impl DreamSummarizer {
    /// Summarise the given conversation messages into a concise text.
    ///
    /// The prompt instructs the LLM to:
    /// - Preserve key decisions, tool outcomes, and user preferences
    /// - Convert relative dates to absolute where possible
    /// - Keep the summary concise but complete
    pub async fn summarize_messages(
        messages: &[Message],
        provider: &dyn Provider,
    ) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // Build a text representation of the messages
        let mut conversation_text = String::new();
        for msg in messages {
            let role = match msg.role {
                brainwires_core::Role::User => "User",
                brainwires_core::Role::Assistant => "Assistant",
                brainwires_core::Role::System => "System",
                brainwires_core::Role::Tool => "Tool",
            };
            let text = msg.text_or_summary();
            conversation_text.push_str(&format!("{role}: {text}\n\n"));
        }

        let prompt = format!(
            "Synthesize the following conversation into a concise summary. \
             Preserve key decisions, tool outcomes, and user preferences. \
             Convert relative dates to absolute where possible. \
             Focus on information that would be useful for future interactions.\n\n\
             Conversation:\n{conversation_text}\n\n\
             Summary:"
        );

        let llm_messages = vec![Message::user(&prompt)];
        let options = ChatOptions {
            temperature: Some(0.3),
            max_tokens: Some(1024),
            ..Default::default()
        };

        let response = provider.chat(&llm_messages, None, &options).await?;
        Ok(response.message.text_or_summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use anyhow::Result;
    use async_trait::async_trait;
    use brainwires_core::{ChatResponse, StreamChunk, Tool, Usage};
    use futures::stream::BoxStream;

    /// Captures the prompt the summarizer dispatches and returns a fixed reply.
    struct CapturingProvider {
        captured_prompt: Mutex<Option<String>>,
        reply: String,
    }

    impl CapturingProvider {
        fn new(reply: &str) -> Self {
            Self {
                captured_prompt: Mutex::new(None),
                reply: reply.to_string(),
            }
        }

        fn captured(&self) -> Option<String> {
            self.captured_prompt.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for CapturingProvider {
        fn name(&self) -> &str {
            "capturing"
        }

        async fn chat(
            &self,
            messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            // summarize_messages sends a single user message containing the
            // built prompt — capture its text.
            let prompt = messages
                .last()
                .map(|m| m.text_or_summary())
                .unwrap_or_default();
            *self.captured_prompt.lock().unwrap() = Some(prompt);

            Ok(ChatResponse {
                message: Message::assistant(self.reply.clone()),
                usage: Usage::default(),
                finish_reason: Some("stop".into()),
            })
        }

        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    #[tokio::test]
    async fn empty_input_returns_empty_summary_without_provider_call() {
        let provider = CapturingProvider::new("UNREACHABLE");
        let out = DreamSummarizer::summarize_messages(&[], &provider)
            .await
            .unwrap();
        assert_eq!(out, "", "empty input must short-circuit to empty output");
        assert!(
            provider.captured().is_none(),
            "no provider call should have been made on empty input"
        );
    }

    #[tokio::test]
    async fn prompt_contains_role_prefixes_and_synthesize_instruction() {
        let provider = CapturingProvider::new("a tidy summary");
        let msgs = vec![
            Message::user("we agreed to use rusqlite over sqlx"),
            Message::assistant("noted, will update the migration plan"),
        ];

        let out = DreamSummarizer::summarize_messages(&msgs, &provider)
            .await
            .unwrap();
        assert_eq!(out, "a tidy summary");

        let prompt = provider.captured().expect("provider must have been called");
        assert!(
            prompt.contains("Synthesize"),
            "prompt must include the literal Synthesize instruction"
        );
        assert!(
            prompt.contains("User:"),
            "prompt must label user turns with 'User:'"
        );
        assert!(
            prompt.contains("Assistant:"),
            "prompt must label assistant turns with 'Assistant:'"
        );
        assert!(
            prompt.contains("we agreed to use rusqlite over sqlx"),
            "user message body must appear in the prompt"
        );
        assert!(
            prompt.contains("noted, will update the migration plan"),
            "assistant message body must appear in the prompt"
        );
    }
}
