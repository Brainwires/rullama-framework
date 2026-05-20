//! Unit tests for `TaskAgent`.

use std::sync::Arc;

use async_trait::async_trait;
use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, Message, MessageContent, Role, StreamChunk, Task,
    Tool, ToolContext, ToolResult, ToolUse, Usage,
};
use brainwires_tool_runtime::ToolExecutor;
use futures::stream::BoxStream;

use anyhow::Result;

use crate::context::AgentContext;
use brainwires_agent::communication::CommunicationHub;
use brainwires_agent::file_locks::FileLockManager;

use super::agent::TaskAgent;
use super::spawn::spawn_task_agent;
use super::types::{TaskAgentConfig, TaskAgentStatus};

// ── Mock provider ──────────────────────────────────────────────────────────

struct MockProvider {
    responses: std::sync::Mutex<Vec<ChatResponse>>,
}

impl MockProvider {
    fn single(text: &str) -> Self {
        Self {
            responses: std::sync::Mutex::new(vec![ChatResponse {
                message: Message::assistant(text),
                finish_reason: Some("stop".to_string()),
                usage: Usage::default(),
            }]),
        }
    }
}

#[async_trait]
impl brainwires_core::Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let mut guard = self.responses.lock().unwrap();
        if guard.is_empty() {
            anyhow::bail!("no more mock responses")
        }
        Ok(guard.remove(0))
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

// ── Mock tool executor ─────────────────────────────────────────────────────

struct NoOpExecutor;

#[async_trait]
impl ToolExecutor for NoOpExecutor {
    async fn execute(&self, tool_use: &ToolUse, _ctx: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::success(tool_use.id.clone(), "ok".to_string()))
    }

    fn available_tools(&self) -> Vec<Tool> {
        vec![]
    }
}

fn make_context() -> Arc<AgentContext> {
    Arc::new(AgentContext::new(
        "/tmp",
        Arc::new(NoOpExecutor),
        Arc::new(CommunicationHub::new()),
        Arc::new(FileLockManager::new()),
    ))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_creation() {
    let task = Task::new("t-1", "Do something");
    let agent = TaskAgent::new(
        "agent-1".to_string(),
        task,
        Arc::new(MockProvider::single("done")),
        make_context(),
        TaskAgentConfig::default(),
    );
    assert_eq!(agent.id(), "agent-1");
    assert_eq!(agent.status().await, TaskAgentStatus::Idle);
}

#[tokio::test]
async fn test_execution_completes() {
    let task = Task::new("t-1", "Simple task");
    let agent = Arc::new(TaskAgent::new(
        "agent-1".to_string(),
        task,
        Arc::new(MockProvider::single("Task completed successfully")),
        make_context(),
        TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        },
    ));

    let result = agent.execute().await.unwrap();
    assert!(result.success);
    assert_eq!(result.agent_id, "agent-1");
    assert_eq!(result.task_id, "t-1");
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn test_spawn_task_agent() {
    let task = Task::new("t-1", "Background task");
    let agent = Arc::new(TaskAgent::new(
        "agent-1".to_string(),
        task,
        Arc::new(MockProvider::single("done")),
        make_context(),
        TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        },
    ));

    let handle = spawn_task_agent(agent);
    let result = handle.await.unwrap().unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn test_status_display() {
    assert_eq!(TaskAgentStatus::Idle.to_string(), "Idle");
    assert_eq!(
        TaskAgentStatus::Working("reading".to_string()).to_string(),
        "Working: reading"
    );
    assert_eq!(
        TaskAgentStatus::Failed("oops".to_string()).to_string(),
        "Failed: oops"
    );
}

#[tokio::test]
async fn test_result_has_execution_graph() {
    let task = Task::new("t-1", "Simple task");
    let agent = Arc::new(TaskAgent::new(
        "agent-1".to_string(),
        task,
        Arc::new(MockProvider::single("done")),
        make_context(),
        TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        },
    ));

    let result = agent.execute().await.unwrap();
    assert!(result.success);
    // One iteration = one step node
    assert_eq!(result.execution_graph.steps.len(), 1);
    assert_eq!(result.execution_graph.steps[0].iteration, 1);
    // prompt_hash must be non-empty
    assert!(!result.execution_graph.prompt_hash.is_empty());
    // telemetry must match
    assert_eq!(result.telemetry.total_iterations, 1);
    assert!(result.telemetry.success);
    assert_eq!(
        result.telemetry.prompt_hash,
        result.execution_graph.prompt_hash
    );
}

#[tokio::test]
async fn test_pre_execute_hook_reject() {
    use brainwires_tool_runtime::{PreHookDecision, ToolPreHook};

    struct RejectAll;
    #[async_trait]
    impl ToolPreHook for RejectAll {
        async fn before_execute(
            &self,
            tool_use: &ToolUse,
            _ctx: &ToolContext,
        ) -> anyhow::Result<PreHookDecision> {
            Ok(PreHookDecision::Reject(format!(
                "rejected: {}",
                tool_use.name
            )))
        }
    }

    // Provider that requests a tool call on iteration 1, then stops.
    struct ToolThenStop;
    #[async_trait]
    impl brainwires_core::Provider for ToolThenStop {
        fn name(&self) -> &str {
            "tool-then-stop"
        }
        async fn chat(
            &self,
            messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            // First call: return a tool use. Subsequent calls: return done.
            let has_tool_result = messages.iter().any(|m| {
                matches!(&m.content, MessageContent::Blocks(b) if b.iter().any(|cb| matches!(cb, ContentBlock::ToolResult { .. })))
            });
            if has_tool_result {
                return Ok(ChatResponse {
                    message: Message::assistant("done after hook rejection"),
                    finish_reason: Some("stop".to_string()),
                    usage: Usage::default(),
                });
            }
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "tu-1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo hi"}),
                    }]),
                    name: None,
                    metadata: None,
                },
                finish_reason: None,
                usage: Usage::default(),
            })
        }
        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> futures::stream::BoxStream<'a, Result<brainwires_core::StreamChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    let ctx = Arc::new(
        AgentContext::new(
            "/tmp",
            Arc::new(NoOpExecutor),
            Arc::new(CommunicationHub::new()),
            Arc::new(FileLockManager::new()),
        )
        .with_pre_execute_hook(Arc::new(RejectAll)),
    );

    let task = Task::new("t-hook", "test hook rejection");
    let agent = Arc::new(TaskAgent::new(
        "agent-hook".to_string(),
        task,
        Arc::new(ToolThenStop),
        ctx,
        TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        },
    ));

    let result = agent.execute().await.unwrap();
    assert!(result.success);
    // The rejected tool call should appear in the graph as is_error=true
    let rejected: Vec<_> = result
        .execution_graph
        .steps
        .iter()
        .flat_map(|s| s.tool_calls.iter())
        .filter(|tc| tc.is_error)
        .collect();
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0].tool_name, "bash");
    // And "bash" should still appear in the tool_sequence
    assert!(
        result
            .execution_graph
            .tool_sequence
            .contains(&"bash".to_string())
    );
}
