//! Unit tests for `TaskAgent`.

use std::sync::Arc;

use async_trait::async_trait;
use rullama_core::{Task, Tool, ToolContext, ToolResult, ToolUse};
use rullama_test_fixtures::ScriptedProvider;
use rullama_tool_runtime::ToolExecutor;

use anyhow::Result;

use crate::context::AgentContext;
use rullama_agent::communication::CommunicationHub;
use rullama_agent::file_locks::FileLockManager;

use super::agent::TaskAgent;
use super::spawn::spawn_task_agent;
use super::types::{TaskAgentConfig, TaskAgentStatus};

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
        Arc::new(ScriptedProvider::new("mock").then_text("done")),
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
        Arc::new(ScriptedProvider::new("mock").then_text("Task completed successfully")),
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
        Arc::new(ScriptedProvider::new("mock").then_text("done")),
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
        Arc::new(ScriptedProvider::new("mock").then_text("done")),
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
    use rullama_tool_runtime::{PreHookDecision, ToolPreHook};

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

    // Provider that requests a tool call on iteration 1, then stops on iteration 2.
    let provider = Arc::new(
        ScriptedProvider::new("tool-then-stop")
            .then_tool_call("tu-1", "bash", serde_json::json!({"command": "echo hi"}))
            .then_text("done after hook rejection"),
    );

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
        provider,
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
