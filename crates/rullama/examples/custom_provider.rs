//! Example: Implementing a custom AI provider
//!
//! Shows how to implement the `Provider` trait to plug in any LLM backend
//! (local model, custom API, research prototype, etc.).
//!
//! Run: cargo run -p rullama --example custom_provider --features providers

use anyhow::Result;
use async_trait::async_trait;
use rullama::prelude::*;
use futures::stream::BoxStream;

/// A minimal custom provider that echoes input back.
/// Replace the body of `chat` and `stream_chat` with your own LLM calls.
struct EchoProvider;

#[async_trait]
impl Provider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }

    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Extract the last user message
        let last = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.text())
            .unwrap_or_default();

        let token_count = last.len() as u32;

        Ok(ChatResponse {
            message: Message::assistant(format!("Echo: {}", last)),
            usage: Usage::new(token_count, token_count),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // For simplicity, wrap the non-streaming response
        Box::pin(async_stream::stream! {
            let response = self.chat(messages, tools, options).await?;
            yield Ok(StreamChunk::Text(response.message.text().unwrap_or_default().to_string()));
            yield Ok(StreamChunk::Done);
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider = EchoProvider;

    let messages = vec![Message::user("Hello, custom provider!")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await?;
    println!("Response: {}", response.message.text().unwrap_or_default());
    println!(
        "Tokens used: {} in, {} out",
        response.usage.prompt_tokens, response.usage.completion_tokens
    );

    Ok(())
}
