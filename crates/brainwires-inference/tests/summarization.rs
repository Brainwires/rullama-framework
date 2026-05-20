//! End-to-end test for ChatAgent history compaction via LLM summarizer.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_core::ToolContext;
use brainwires_core::{
    ChatOptions, ChatResponse, Message, Provider, Role, StreamChunk, Tool, Usage,
};
use brainwires_inference::ChatAgent;
use brainwires_inference::summarization::LlmSummarizer;
use brainwires_tool_builtins::BuiltinToolExecutor;
use brainwires_tool_runtime::{ToolExecutor, ToolRegistry};

/// Minimal provider — never called by compact_history() itself.
struct NoopProvider;
#[async_trait]
impl Provider for NoopProvider {
    fn name(&self) -> &str {
        "noop"
    }
    async fn chat(
        &self,
        _: &[Message],
        _: Option<&[Tool]>,
        _: &ChatOptions,
    ) -> Result<ChatResponse> {
        unreachable!("process_message not exercised in this test")
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

/// Summarizer provider that records how many messages it was shown and
/// returns a deterministic summary.
struct RecordingProvider {
    observed_msg_lens: Arc<Mutex<Vec<usize>>>,
    calls: AtomicU32,
}

impl RecordingProvider {
    fn new() -> Self {
        Self {
            observed_msg_lens: Arc::new(Mutex::new(Vec::new())),
            calls: AtomicU32::new(0),
        }
    }
    fn calls(&self) -> u32 {
        self.calls.load(Ordering::Relaxed)
    }
    fn observed_lens(&self) -> Vec<usize> {
        self.observed_msg_lens.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
    }
    async fn chat(
        &self,
        messages: &[Message],
        _: Option<&[Tool]>,
        _: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let first = messages
            .last()
            .and_then(|m| m.text())
            .unwrap_or_default()
            .to_string();
        self.observed_msg_lens.lock().unwrap().push(first.len());
        Ok(ChatResponse {
            message: Message::assistant("earlier work covered topics A, B, and C"),
            usage: Usage::new(30, 10),
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

fn make_executor() -> Arc<dyn ToolExecutor> {
    Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ))
}

#[tokio::test]
async fn compact_history_replaces_middle_with_summary() {
    let recorder = Arc::new(RecordingProvider::new());
    let summarizer = Arc::new(LlmSummarizer::new(recorder.clone() as Arc<dyn Provider>));

    let mut agent = ChatAgent::new(
        Arc::new(NoopProvider) as Arc<dyn Provider>,
        make_executor(),
        ChatOptions::default(),
    )
    .with_system_prompt("you are helpful")
    .with_summarizer(summarizer)
    .with_summarization_keep_tail(3);

    // Seed: 1 system + 12 alternating user/assistant messages.
    for i in 0..12 {
        let m = if i % 2 == 0 {
            Message::user(format!("user msg {i}"))
        } else {
            Message::assistant(format!("assistant reply {i}"))
        };
        agent.restore_messages({
            let mut msgs = agent.messages().to_vec();
            msgs.push(m);
            msgs
        });
    }
    assert_eq!(agent.message_count(), 13);

    agent.compact_history().await.expect("compact");

    // Expected layout: system + synthetic summary + last 3 → 5 messages.
    assert_eq!(agent.message_count(), 5);
    assert_eq!(agent.messages()[0].role, Role::System);
    assert_eq!(agent.messages()[1].role, Role::Assistant);
    assert!(
        agent.messages()[1]
            .text()
            .unwrap_or_default()
            .starts_with("[Prior conversation summary]")
    );

    // Tail preserved verbatim: the last three original messages were
    // indexes 9, 10, 11 → "user msg 10" is the content at index 10.
    let tail_texts: Vec<&str> = agent
        .messages()
        .iter()
        .skip(2)
        .map(|m| m.text().unwrap_or(""))
        .collect();
    assert!(
        tail_texts
            .iter()
            .any(|t| t.contains("msg 9") || t.contains("reply 9"))
    );
    assert!(
        tail_texts
            .iter()
            .any(|t| t.contains("msg 10") || t.contains("reply 10"))
    );
    assert!(
        tail_texts
            .iter()
            .any(|t| t.contains("msg 11") || t.contains("reply 11"))
    );

    // Summarizer provider was called exactly once.
    assert_eq!(recorder.calls(), 1);
    // Transcript passed in was nonempty.
    assert!(recorder.observed_lens().iter().all(|&n| n > 0));
}

#[tokio::test]
async fn compact_history_is_noop_when_history_is_short() {
    let recorder = Arc::new(RecordingProvider::new());
    let summarizer = Arc::new(LlmSummarizer::new(recorder.clone() as Arc<dyn Provider>));

    let mut agent = ChatAgent::new(
        Arc::new(NoopProvider) as Arc<dyn Provider>,
        make_executor(),
        ChatOptions::default(),
    )
    .with_system_prompt("sys")
    .with_summarizer(summarizer)
    .with_summarization_keep_tail(6);

    // Only 3 non-system messages — below keep_tail + 1, so nothing to compact.
    agent.restore_messages(vec![
        Message::system("sys"),
        Message::user("a"),
        Message::assistant("b"),
        Message::user("c"),
    ]);

    agent.compact_history().await.unwrap();

    assert_eq!(agent.message_count(), 4);
    assert_eq!(recorder.calls(), 0, "summarizer should not be invoked");
}

#[tokio::test]
async fn compact_history_without_summarizer_falls_back_to_trim() {
    let mut agent = ChatAgent::new(
        Arc::new(NoopProvider) as Arc<dyn Provider>,
        make_executor(),
        ChatOptions::default(),
    )
    .with_system_prompt("sys");

    // 1 system + 30 user → trim should keep system + last 19.
    let mut msgs = vec![Message::system("sys")];
    for i in 0..30 {
        msgs.push(Message::user(format!("msg {i}")));
    }
    agent.restore_messages(msgs);

    agent.compact_history().await.unwrap();

    assert_eq!(agent.message_count(), 20); // system + 19 user
    assert_eq!(agent.messages()[0].role, Role::System);
    // Last message preserved.
    let last = agent.messages().last().and_then(|m| m.text()).unwrap();
    assert_eq!(last, "msg 29");
}
