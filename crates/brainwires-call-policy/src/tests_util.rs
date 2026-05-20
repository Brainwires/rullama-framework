//! Shared test utilities: minimal `Provider` mocks used across decorator tests.

#![allow(dead_code)] // Mocks are consumed by subsets of tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_core::message::{ChatResponse, Message, StreamChunk, Usage};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;

/// A trivial provider that always returns a fixed result — Ok, a persistent
/// error, or a countdown of N errors followed by success.
pub struct EchoProvider {
    name: &'static str,
    mode: Mode,
    remaining_errors: AtomicU32,
    calls: AtomicU32,
}

#[derive(Debug, Clone)]
enum Mode {
    AlwaysOk,
    AlwaysErr(&'static str),
    ErrThenOk(&'static str),
}

impl EchoProvider {
    /// A provider that always succeeds with an empty assistant message.
    pub fn ok(name: &'static str) -> Self {
        Self {
            name,
            mode: Mode::AlwaysOk,
            remaining_errors: AtomicU32::new(0),
            calls: AtomicU32::new(0),
        }
    }

    /// A provider that always returns an error with the given message.
    pub fn always_err(name: &'static str, msg: &'static str) -> Self {
        Self {
            name,
            mode: Mode::AlwaysErr(msg),
            remaining_errors: AtomicU32::new(0),
            calls: AtomicU32::new(0),
        }
    }

    /// A provider that returns `errors` error responses, then succeeds.
    pub fn err_then_ok(name: &'static str, errors: u32, msg: &'static str) -> Self {
        Self {
            name,
            mode: Mode::ErrThenOk(msg),
            remaining_errors: AtomicU32::new(errors),
            calls: AtomicU32::new(0),
        }
    }

    /// Total call count.
    pub fn calls(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Provider for EchoProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match &self.mode {
            Mode::AlwaysOk => Ok(ChatResponse {
                message: Message::assistant("ok"),
                usage: Usage::new(4, 2),
                finish_reason: Some("stop".into()),
            }),
            Mode::AlwaysErr(m) => Err(anyhow::anyhow!("{m}")),
            Mode::ErrThenOk(m) => {
                let left = self.remaining_errors.fetch_sub(1, Ordering::Relaxed);
                if left > 0 {
                    Err(anyhow::anyhow!("{m}"))
                } else {
                    self.remaining_errors.store(0, Ordering::Relaxed);
                    Ok(ChatResponse {
                        message: Message::assistant("ok"),
                        usage: Usage::new(4, 2),
                        finish_reason: Some("stop".into()),
                    })
                }
            }
        }
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        let u = Usage::new(4, 2);
        Box::pin(futures::stream::iter(vec![
            Ok(StreamChunk::Text("ok".into())),
            Ok(StreamChunk::Usage(u)),
        ]))
    }
}

/// A provider whose success/failure behavior can be flipped at runtime.
pub struct ToggleProvider {
    name: &'static str,
    fail: Arc<AtomicBool>,
}

impl ToggleProvider {
    /// Create a new toggle provider (initially succeeding).
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            fail: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Flip failure mode.
    pub fn set_fail(&self, fail: bool) {
        self.fail.store(fail, Ordering::Relaxed);
    }
}

#[async_trait]
impl Provider for ToggleProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        if self.fail.load(Ordering::Relaxed) {
            Err(anyhow::anyhow!("500 internal server error"))
        } else {
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                usage: Usage::new(4, 2),
                finish_reason: Some("stop".into()),
            })
        }
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
