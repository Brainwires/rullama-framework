//! End-to-end test for ChatAgent history compaction via LLM summarizer.

use std::sync::Arc;

use rullama_core::{ChatOptions, Message, Provider, Role, ToolContext};
use rullama_inference::ChatAgent;
use rullama_inference::summarization::LlmSummarizer;
use rullama_test_fixtures::{FailingProvider, RecordingProvider, ScriptedProvider};
use rullama_tool_builtins::BuiltinToolExecutor;
use rullama_tool_runtime::{ToolExecutor, ToolRegistry};

/// Per-call last-message length, derived from a `RecordingProvider` log.
fn observed_lens<P: Provider>(rec: &RecordingProvider<P>) -> Vec<usize> {
    rec.calls()
        .iter()
        .map(|c| {
            c.messages
                .last()
                .and_then(|m| m.text())
                .map(|s| s.len())
                .unwrap_or(0)
        })
        .collect()
}

fn make_summarizer_recorder() -> Arc<RecordingProvider<ScriptedProvider>> {
    Arc::new(RecordingProvider::new(ScriptedProvider::always_text(
        "recording",
        "earlier work covered topics A, B, and C",
    )))
}

/// Main agent provider — never expected to be called by compact_history().
/// Using `FailingProvider` makes any accidental invocation a clear test failure.
fn unreachable_agent_provider() -> Arc<dyn Provider> {
    Arc::new(FailingProvider::new(
        "compact_history must not invoke the main provider",
    ))
}

fn make_executor() -> Arc<dyn ToolExecutor> {
    Arc::new(BuiltinToolExecutor::new(
        ToolRegistry::new(),
        ToolContext::default(),
    ))
}

#[tokio::test]
async fn compact_history_replaces_middle_with_summary() {
    let recorder = make_summarizer_recorder();
    let summarizer = Arc::new(LlmSummarizer::new(recorder.clone() as Arc<dyn Provider>));

    let mut agent = ChatAgent::new(
        unreachable_agent_provider(),
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
    assert_eq!(recorder.call_count(), 1);
    // Transcript passed in was nonempty.
    assert!(observed_lens(&recorder).iter().all(|&n| n > 0));
}

#[tokio::test]
async fn compact_history_is_noop_when_history_is_short() {
    let recorder = make_summarizer_recorder();
    let summarizer = Arc::new(LlmSummarizer::new(recorder.clone() as Arc<dyn Provider>));

    let mut agent = ChatAgent::new(
        unreachable_agent_provider(),
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
    assert_eq!(recorder.call_count(), 0, "summarizer should not be invoked");
}

#[tokio::test]
async fn compact_history_without_summarizer_falls_back_to_trim() {
    let mut agent = ChatAgent::new(
        unreachable_agent_provider(),
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
