//! Consolidated mock provider implementations.
//!
//! Replaces ad-hoc `MockProvider` / `FakeProvider` / `RecordingProvider` /
//! `ScriptedProvider` impls that were previously inlined in test modules.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{
    ChatOptions, ChatResponse, ContentBlock, Message, MessageContent, Provider, Role, StreamChunk,
    Tool, Usage,
};
use futures::stream::{self, BoxStream};

/// A queued response in a [`ScriptedProvider`]. One entry is consumed per
/// `chat()` or `stream_chat()` call.
#[derive(Debug, Clone)]
pub enum ScriptedResponse {
    /// Plain assistant text. In `chat()` returns a single-message response;
    /// in `stream_chat()` emits `Text` then `Done`.
    Text(String),
    /// One or more tool-use blocks. In `chat()` returns a `MessageContent::Blocks`
    /// response; in `stream_chat()` emits `ToolUse` + `ToolInputDelta` per call
    /// then `Done`. Each entry is `(call_id, tool_name, arguments_json)`.
    ToolCalls(Vec<(String, String, serde_json::Value)>),
    /// A fully pre-built `ChatResponse` (for cases needing usage tweaks,
    /// custom finish_reason, image blocks, etc.). In `stream_chat()` the
    /// message text is replayed as a single `Text` chunk.
    Custom(ChatResponse),
    /// Raw stream chunks. Only consumed by `stream_chat()`; `chat()` flattens
    /// the contained `Text` chunks into a single assistant message.
    Stream(Vec<StreamChunk>),
    /// Returns `Err` with the contained message.
    Error(String),
}

/// What happens after the queue is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Exhaustion {
    /// Return `Err("no more scripted responses")`. Surfaces over-call bugs.
    #[default]
    Error,
    /// Repeat the last queued response forever.
    RepeatLast,
}

/// A provider that returns a scripted sequence of responses.
///
/// Replaces the previously-inlined `MockProvider` / `ScriptedProvider`
/// implementations across `rullama-inference`, `rullama-provider`,
/// and `rullama-memory`.
pub struct ScriptedProvider {
    name: String,
    queue: Mutex<VecDeque<ScriptedResponse>>,
    last: Mutex<Option<ScriptedResponse>>,
    exhaustion: Exhaustion,
}

impl ScriptedProvider {
    /// Create an empty scripted provider with the given name. Add responses
    /// with the `then_*` / `always_*` builder methods.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            queue: Mutex::new(VecDeque::new()),
            last: Mutex::new(None),
            exhaustion: Exhaustion::Error,
        }
    }

    /// Shortcut: a provider that returns the given text on every call forever.
    /// Replaces the very common "single static response" pattern.
    pub fn always_text(name: impl Into<String>, text: impl Into<String>) -> Self {
        let mut p = Self::new(name);
        p.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Text(text.into()));
        p.exhaustion = Exhaustion::RepeatLast;
        p
    }

    /// Shortcut: a provider that returns the given `ChatResponse` on every
    /// call forever.
    pub fn always_response(name: impl Into<String>, response: ChatResponse) -> Self {
        let mut p = Self::new(name);
        p.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Custom(response));
        p.exhaustion = Exhaustion::RepeatLast;
        p
    }

    /// Enqueue an assistant text response.
    pub fn then_text(self, text: impl Into<String>) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Text(text.into()));
        self
    }

    /// Enqueue a tool-call response with one tool use.
    pub fn then_tool_call(
        self,
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::ToolCalls(vec![(
                call_id.into(),
                tool_name.into(),
                arguments,
            )]));
        self
    }

    /// Enqueue a tool-call response with multiple parallel tool uses.
    pub fn then_tool_calls(
        self,
        calls: Vec<(String, String, serde_json::Value)>,
    ) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::ToolCalls(calls));
        self
    }

    /// Enqueue a pre-built `ChatResponse`.
    pub fn then_response(self, response: ChatResponse) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Custom(response));
        self
    }

    /// Enqueue a raw stream-chunk sequence (for stream_chat). Caller is
    /// responsible for emitting a terminating `StreamChunk::Done` at the
    /// end of the vec.
    pub fn then_stream(self, chunks: Vec<StreamChunk>) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Stream(chunks));
        self
    }

    /// Enqueue an error response.
    pub fn then_error(self, message: impl Into<String>) -> Self {
        self.queue
            .lock()
            .unwrap()
            .push_back(ScriptedResponse::Error(message.into()));
        self
    }

    /// Change the queue-exhaustion behaviour.
    pub fn with_exhaustion(mut self, exhaustion: Exhaustion) -> Self {
        self.exhaustion = exhaustion;
        self
    }

    fn pop_next(&self) -> Result<ScriptedResponse> {
        let mut queue = self.queue.lock().unwrap();
        if let Some(next) = queue.pop_front() {
            *self.last.lock().unwrap() = Some(next.clone());
            return Ok(next);
        }
        drop(queue);
        match self.exhaustion {
            Exhaustion::Error => {
                anyhow::bail!("ScriptedProvider '{}' has no more responses", self.name)
            }
            Exhaustion::RepeatLast => {
                let last = self.last.lock().unwrap();
                last.clone()
                    .ok_or_else(|| anyhow::anyhow!("ScriptedProvider '{}' is empty", self.name))
            }
        }
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        match self.pop_next()? {
            ScriptedResponse::Text(text) => Ok(ChatResponse {
                message: Message::assistant(&text),
                usage: Usage::default(),
                finish_reason: Some("stop".to_string()),
            }),
            ScriptedResponse::ToolCalls(calls) => {
                let blocks = calls
                    .into_iter()
                    .map(|(id, name, input)| ContentBlock::ToolUse { id, name, input })
                    .collect();
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(blocks),
                        name: None,
                        metadata: None,
                    },
                    usage: Usage::default(),
                    finish_reason: Some("tool_use".to_string()),
                })
            }
            ScriptedResponse::Custom(response) => Ok(response),
            ScriptedResponse::Stream(chunks) => {
                let mut text = String::new();
                for chunk in chunks {
                    if let StreamChunk::Text(t) = chunk {
                        text.push_str(&t);
                    }
                }
                Ok(ChatResponse {
                    message: Message::assistant(&text),
                    usage: Usage::default(),
                    finish_reason: Some("stop".to_string()),
                })
            }
            ScriptedResponse::Error(msg) => anyhow::bail!("{}", msg),
        }
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        match self.pop_next() {
            Ok(ScriptedResponse::Text(text)) => Box::pin(stream::iter(vec![
                Ok(StreamChunk::Text(text)),
                Ok(StreamChunk::Done),
            ])),
            Ok(ScriptedResponse::ToolCalls(calls)) => {
                let mut chunks: Vec<Result<StreamChunk>> = Vec::with_capacity(calls.len() * 2 + 1);
                for (id, name, input) in calls {
                    chunks.push(Ok(StreamChunk::ToolUse {
                        id: id.clone(),
                        name,
                    }));
                    chunks.push(Ok(StreamChunk::ToolInputDelta {
                        id,
                        partial_json: input.to_string(),
                    }));
                }
                chunks.push(Ok(StreamChunk::Done));
                Box::pin(stream::iter(chunks))
            }
            Ok(ScriptedResponse::Custom(response)) => {
                // Emit Text + Usage + Done so consumers tracking
                // cumulative_usage (e.g. ChatAgent) see the same totals
                // they'd see from the chat() path. Without the Usage
                // chunk, anyone scripting a non-zero Usage on the
                // canned response would observe 0 tokens accumulating
                // through stream_chat, which is a footgun.
                let text = response.message.text().unwrap_or("").to_string();
                let mut chunks: Vec<Result<StreamChunk>> = Vec::with_capacity(3);
                chunks.push(Ok(StreamChunk::Text(text)));
                if response.usage.total_tokens > 0
                    || response.usage.cache_creation_input_tokens > 0
                    || response.usage.cache_read_input_tokens > 0
                {
                    chunks.push(Ok(StreamChunk::Usage(response.usage)));
                }
                chunks.push(Ok(StreamChunk::Done));
                Box::pin(stream::iter(chunks))
            }
            Ok(ScriptedResponse::Stream(chunks)) => {
                Box::pin(stream::iter(chunks.into_iter().map(Ok)))
            }
            Ok(ScriptedResponse::Error(msg)) => {
                Box::pin(stream::iter(vec![Err(anyhow::anyhow!(msg))]))
            }
            Err(e) => Box::pin(stream::iter(vec![Err(e)])),
        }
    }
}

/// A provider that always returns the configured error from both `chat()`
/// and `stream_chat()`. Useful for asserting that callers handle provider
/// failures (e.g. that a budget guard rejects a call before the provider
/// is reached).
pub struct FailingProvider {
    name: String,
    error: String,
}

impl FailingProvider {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            name: "failing".to_string(),
            error: error.into(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Provider for FailingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        anyhow::bail!("{}", self.error)
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        let err = anyhow::anyhow!(self.error.clone());
        Box::pin(stream::iter(vec![Err(err)]))
    }
}

/// A single recorded call against a [`RecordingProvider`].
#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub method: &'static str,
    pub messages: Vec<Message>,
    pub options: ChatOptions,
    pub message_count: usize,
}

/// Wraps another `Provider`, logging every call before delegating.
pub struct RecordingProvider<P: Provider> {
    inner: P,
    log: Mutex<Vec<RecordedCall>>,
}

impl<P: Provider> RecordingProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    /// Return a clone of every recorded call so far.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.log.lock().unwrap().clone()
    }

    /// Number of times either `chat()` or `stream_chat()` has been invoked.
    pub fn call_count(&self) -> usize {
        self.log.lock().unwrap().len()
    }

    /// Borrow the wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

#[async_trait]
impl<P: Provider> Provider for RecordingProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.log.lock().unwrap().push(RecordedCall {
            method: "chat",
            messages: messages.to_vec(),
            options: options.clone(),
            message_count: messages.len(),
        });
        self.inner.chat(messages, tools, options).await
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        self.log.lock().unwrap().push(RecordedCall {
            method: "stream_chat",
            messages: messages.to_vec(),
            options: options.clone(),
            message_count: messages.len(),
        });
        self.inner.stream_chat(messages, tools, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn scripted_text_chat() {
        let p = ScriptedProvider::new("test").then_text("hello");
        let r = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text(), Some("hello"));
    }

    #[tokio::test]
    async fn scripted_queue_pops_in_order() {
        let p = ScriptedProvider::new("test")
            .then_text("one")
            .then_text("two");
        let r1 = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap();
        let r2 = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r1.message.text(), Some("one"));
        assert_eq!(r2.message.text(), Some("two"));
    }

    #[tokio::test]
    async fn scripted_exhausted_errors() {
        let p = ScriptedProvider::new("test").then_text("once");
        let _ok = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap();
        let err = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no more responses"));
    }

    #[tokio::test]
    async fn scripted_always_text_repeats() {
        let p = ScriptedProvider::always_text("test", "forever");
        for _ in 0..3 {
            let r = p
                .chat(&[], None, &ChatOptions::default())
                .await
                .unwrap();
            assert_eq!(r.message.text(), Some("forever"));
        }
    }

    #[tokio::test]
    async fn scripted_tool_call_in_stream() {
        let p = ScriptedProvider::new("test").then_tool_call(
            "call-0",
            "bash",
            serde_json::json!({"command": "ls"}),
        );
        let chunks: Vec<_> = p
            .stream_chat(&[], None, &ChatOptions::default())
            .collect()
            .await;
        assert_eq!(chunks.len(), 3);
        match &chunks[0] {
            Ok(StreamChunk::ToolUse { id, name }) => {
                assert_eq!(id, "call-0");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn failing_provider_errors_on_chat() {
        let p = FailingProvider::new("nope");
        let err = p
            .chat(&[], None, &ChatOptions::default())
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "nope");
    }

    #[tokio::test]
    async fn recording_provider_logs_calls() {
        let inner = ScriptedProvider::always_text("inner", "ok");
        let p = RecordingProvider::new(inner);
        let _ = p
            .chat(
                &[Message::user("hi")],
                None,
                &ChatOptions::default(),
            )
            .await
            .unwrap();
        let _ = p
            .chat(
                &[Message::user("hi"), Message::assistant("ok")],
                None,
                &ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(p.call_count(), 2);
        assert_eq!(p.calls()[0].message_count, 1);
        assert_eq!(p.calls()[1].message_count, 2);
    }
}
