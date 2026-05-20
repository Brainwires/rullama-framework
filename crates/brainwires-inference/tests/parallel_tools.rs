//! Integration tests for parallel tool dispatch in `ChatAgent`.
//!
//! Verifies that:
//! - Tools with `serialize: false` execute concurrently up to `tool_concurrency`.
//! - Tools with `serialize: true` execute sequentially before the parallel batch.
//! - Result order in the outgoing user message matches the original tool-use order
//!   even when tools finish out of order.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, Message, MessageContent, Provider, StreamChunk, Tool,
    ToolContext, ToolInputSchema, ToolResult, ToolUse, Usage,
};
use brainwires_inference::ChatAgent;
use brainwires_tool_runtime::ToolExecutor;

/// Provider that on the first stream_chat emits N tool uses, then on the
/// second call emits a single final text chunk. Sufficient to drive one
/// round of parallel-tool dispatch in `ChatAgent::run_completion`.
struct ScriptedProvider {
    tool_names: Vec<String>,
    calls: AtomicU32,
}

impl ScriptedProvider {
    fn new(names: Vec<String>) -> Self {
        Self {
            tool_names: names,
            calls: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::assistant("done"),
            usage: Usage::new(1, 1),
            finish_reason: Some("stop".into()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        let chunks: Vec<Result<StreamChunk>> = if call == 0 {
            // Round 1: emit one tool_use per configured name.
            self.tool_names
                .iter()
                .enumerate()
                .flat_map(|(i, name)| {
                    vec![
                        Ok(StreamChunk::ToolUse {
                            id: format!("call-{i}"),
                            name: name.clone(),
                        }),
                        Ok(StreamChunk::ToolInputDelta {
                            id: format!("call-{i}"),
                            partial_json: "{}".into(),
                        }),
                    ]
                })
                .chain(std::iter::once(Ok(StreamChunk::Done)))
                .collect()
        } else {
            // Round 2: final assistant text, no more tool uses → loop exits.
            vec![
                Ok(StreamChunk::Text("all tools done".into())),
                Ok(StreamChunk::Done),
            ]
        };
        Box::pin(futures::stream::iter(chunks))
    }
}

/// Executor that sleeps `delay_ms` per tool call and tracks the maximum
/// number of in-flight executions.
struct LatencyExecutor {
    tools: Vec<Tool>,
    in_flight: Arc<AtomicU32>,
    max_in_flight: Arc<AtomicU32>,
    delay: Duration,
    call_order: Arc<Mutex<Vec<String>>>,
}

impl LatencyExecutor {
    fn new(tools: Vec<Tool>, delay: Duration) -> Self {
        Self {
            tools,
            in_flight: Arc::new(AtomicU32::new(0)),
            max_in_flight: Arc::new(AtomicU32::new(0)),
            delay,
            call_order: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn max_in_flight(&self) -> u32 {
        self.max_in_flight.load(Ordering::Relaxed)
    }

    fn call_order(&self) -> Vec<String> {
        self.call_order.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolExecutor for LatencyExecutor {
    async fn execute(&self, tool_use: &ToolUse, _context: &ToolContext) -> Result<ToolResult> {
        let now = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_in_flight.fetch_max(now, Ordering::Relaxed);
        self.call_order.lock().unwrap().push(tool_use.id.clone());
        tokio::time::sleep(self.delay).await;
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        Ok(ToolResult::success(
            tool_use.id.clone(),
            format!("ok:{}", tool_use.name),
        ))
    }

    fn available_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }
}

fn mk_tool(name: &str, serialize: bool) -> Tool {
    Tool {
        name: name.into(),
        description: String::new(),
        input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
        requires_approval: false,
        defer_loading: false,
        allowed_callers: vec![],
        input_examples: vec![],
        serialize,
    }
}

#[tokio::test]
async fn five_parallel_tools_run_concurrently() {
    // Five parallel-eligible tools, each taking 100ms. With concurrency=4 the
    // full round should finish in roughly ~200ms (two waves), and max
    // in-flight should exceed 1. Serial execution would take ~500ms.
    let names: Vec<String> = (0..5).map(|i| format!("read_{i}")).collect();
    let tools: Vec<Tool> = names.iter().map(|n| mk_tool(n, false)).collect();
    let executor = Arc::new(LatencyExecutor::new(tools, Duration::from_millis(100)));

    let provider = Arc::new(ScriptedProvider::new(names.clone()));
    let mut agent = ChatAgent::new(
        provider.clone(),
        executor.clone() as Arc<dyn ToolExecutor>,
        ChatOptions::default(),
    )
    .with_tool_concurrency(4);

    let start = std::time::Instant::now();
    let text = agent
        .process_message("run them")
        .await
        .expect("agent completes");
    let elapsed = start.elapsed();

    assert_eq!(text, "all tools done");
    assert!(
        executor.max_in_flight() >= 2,
        "expected >= 2 concurrent in-flight, got {}",
        executor.max_in_flight()
    );
    // Generous upper bound — serial would be ~500ms; concurrent should be well under 400ms.
    assert!(
        elapsed < Duration::from_millis(400),
        "parallel dispatch too slow: {elapsed:?}"
    );

    // Result order in the agent's message history must still match tool-use order.
    let last_user = agent
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == brainwires_core::Role::User)
        .expect("tool-result user message exists");
    let ids: Vec<String> = match &last_user.content {
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect(),
        MessageContent::Text(_) => Vec::new(),
    };
    assert_eq!(
        ids,
        (0..5).map(|i| format!("call-{i}")).collect::<Vec<_>>(),
        "result order must match tool_use order"
    );
}

#[tokio::test]
async fn serialize_flag_forces_sequential_execution() {
    // Three tools, all marked serialize=true. Concurrency=4 must not parallelize them.
    let names: Vec<String> = (0..3).map(|i| format!("write_{i}")).collect();
    let tools: Vec<Tool> = names.iter().map(|n| mk_tool(n, true)).collect();
    let executor = Arc::new(LatencyExecutor::new(tools, Duration::from_millis(50)));

    let provider = Arc::new(ScriptedProvider::new(names));
    let mut agent = ChatAgent::new(
        provider,
        executor.clone() as Arc<dyn ToolExecutor>,
        ChatOptions::default(),
    )
    .with_tool_concurrency(4);

    agent.process_message("write them").await.unwrap();

    assert_eq!(
        executor.max_in_flight(),
        1,
        "serialize=true tools must never overlap; saw max_in_flight={}",
        executor.max_in_flight()
    );
    assert_eq!(
        executor.call_order(),
        vec!["call-0", "call-1", "call-2"],
        "serial tools must invoke in emission order"
    );
}

#[tokio::test]
async fn concurrency_of_one_preserves_legacy_behavior() {
    // 3 parallel-eligible tools but concurrency=1 — should behave identically
    // to pre-0.11 sequential execution.
    let names: Vec<String> = (0..3).map(|i| format!("read_{i}")).collect();
    let tools: Vec<Tool> = names.iter().map(|n| mk_tool(n, false)).collect();
    let executor = Arc::new(LatencyExecutor::new(tools, Duration::from_millis(30)));

    let provider = Arc::new(ScriptedProvider::new(names));
    let mut agent = ChatAgent::new(
        provider,
        executor.clone() as Arc<dyn ToolExecutor>,
        ChatOptions::default(),
    )
    .with_tool_concurrency(1);

    agent.process_message("run").await.unwrap();
    assert_eq!(executor.max_in_flight(), 1);
}
