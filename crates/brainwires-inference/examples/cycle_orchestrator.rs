//! Example: Full Plan->Work->Judge cycle orchestration
//!
//! Demonstrates `CycleOrchestrator::run()` with a queued mock provider that
//! serves role-appropriate responses in execution order: planner JSON, then
//! worker "Done" responses, then a judge Complete verdict.
//!
//! Run: cargo run -p brainwires-inference --example cycle_orchestrator

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tokio::sync::Mutex;

use brainwires_agent::brainwires_core::{
    ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool, ToolContext, ToolResult,
    ToolUse, Usage,
};
use brainwires_agent::brainwires_tool_runtime::ToolExecutor;
use brainwires_agent::{CommunicationHub, FileLockManager};
use brainwires_inference::{
    CycleOrchestrator, CycleOrchestratorConfig, JudgeAgentConfig, JudgeVerdict, PlannerAgentConfig,
    TaskAgentConfig,
};

// ── Queued Mock Provider ──────────────────────────────────────────────────

/// A mock provider that serves pre-queued responses in FIFO order.
///
/// This lets us script the exact sequence of LLM responses the orchestrator
/// will see: planner output -> worker completions -> judge verdict.
struct QueuedMockProvider {
    responses: Mutex<VecDeque<ChatResponse>>,
    fallback: ChatResponse,
}

impl QueuedMockProvider {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            fallback: ChatResponse {
                message: Message::assistant("Done"),
                finish_reason: Some("stop".to_string()),
                usage: Usage::default(),
            },
        }
    }

    fn make_response(text: &str) -> ChatResponse {
        ChatResponse {
            message: Message::assistant(text),
            finish_reason: Some("stop".to_string()),
            usage: Usage::default(),
        }
    }
}

#[async_trait]
impl Provider for QueuedMockProvider {
    fn name(&self) -> &str {
        "queued-mock"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let mut queue = self.responses.lock().await;
        Ok(queue.pop_front().unwrap_or_else(|| self.fallback.clone()))
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

// ── No-Op Tool Executor ───────────────────────────────────────────────────

struct NoOpExecutor;

#[async_trait]
impl ToolExecutor for NoOpExecutor {
    async fn execute(&self, tu: &ToolUse, _ctx: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::success(tu.id.clone(), "ok".to_string()))
    }

    fn available_tools(&self) -> Vec<Tool> {
        vec![]
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("=== Cycle Orchestrator Demo ===\n");
    println!("Running a Plan->Work->Judge cycle with mock responses.\n");

    // Pre-queue responses in the order the orchestrator will call the provider:
    //
    // 1. Planner: produces 2 tasks
    // 2. Worker 1: completes its task
    // 3. Worker 2: completes its task
    // 4. Judge: declares the goal complete
    let responses = vec![
        // Planner response
        QueuedMockProvider::make_response(
            r#"Here is my plan:

```json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Create the greeting module with hello() function",
      "files_involved": ["src/greeting.rs"],
      "depends_on": [],
      "priority": "high",
      "estimated_iterations": 5
    },
    {
      "id": "task-2",
      "description": "Add tests for the greeting module",
      "files_involved": ["tests/greeting_test.rs"],
      "depends_on": ["task-1"],
      "priority": "normal",
      "estimated_iterations": 3
    }
  ],
  "rationale": "Create the module first, then add tests that depend on it"
}
```"#,
        ),
        // Worker 1 response
        QueuedMockProvider::make_response("Created src/greeting.rs with hello() function. Done."),
        // Worker 2 response
        QueuedMockProvider::make_response("Added tests for greeting module. All tests pass. Done."),
        // Judge response
        QueuedMockProvider::make_response(
            r#"```json
{
  "verdict": "complete",
  "summary": "Both tasks completed successfully. The greeting module is implemented with tests."
}
```"#,
        ),
    ];

    let provider: Arc<dyn Provider> = Arc::new(QueuedMockProvider::new(responses));
    let tool_executor: Arc<dyn ToolExecutor> = Arc::new(NoOpExecutor);
    let hub = Arc::new(CommunicationHub::new());
    let lock_manager = Arc::new(FileLockManager::new());

    // Configure the orchestrator
    let config = CycleOrchestratorConfig {
        max_cycles: 2,
        max_workers: 3,
        planner_config: PlannerAgentConfig {
            max_iterations: 5,
            ..Default::default()
        },
        judge_config: JudgeAgentConfig {
            max_iterations: 5,
            ..Default::default()
        },
        worker_config: {
            let mut cfg = TaskAgentConfig::default();
            cfg.max_iterations = 5;
            cfg.validation_config = None;
            cfg
        },
        auto_merge: false, // No git repo in this demo
        #[cfg(feature = "native")]
        use_worktrees: false,
        ..Default::default()
    };

    let orchestrator = CycleOrchestrator::new(
        provider,
        tool_executor,
        hub,
        lock_manager,
        "/tmp/demo-project",
        config,
    );

    // Run the orchestration
    let result = orchestrator
        .run("Implement a greeting module with tests")
        .await?;

    // Print the results
    println!("\n=== Orchestration Result ===");
    println!("Success:          {}", result.success);
    println!("Cycles used:      {}", result.cycles_used);
    println!("Tasks completed:  {}", result.total_tasks_completed);
    println!("Tasks failed:     {}", result.total_tasks_failed);

    match &result.final_verdict {
        JudgeVerdict::Complete { summary } => {
            println!("Final verdict:    COMPLETE");
            println!("Summary:          {summary}");
        }
        JudgeVerdict::Continue { summary, .. } => {
            println!("Final verdict:    CONTINUE");
            println!("Summary:          {summary}");
        }
        JudgeVerdict::FreshRestart { reason, .. } => {
            println!("Final verdict:    FRESH_RESTART");
            println!("Reason:           {reason}");
        }
        JudgeVerdict::Abort { reason, .. } => {
            println!("Final verdict:    ABORT");
            println!("Reason:           {reason}");
        }
    }

    // Print cycle history
    println!("\n=== Cycle History ===");
    for record in &result.cycle_history {
        println!("\nCycle {}:", record.cycle_number);
        println!("  Planner rationale: {}", record.planner_output.rationale);
        println!("  Tasks planned:     {}", record.planner_output.tasks.len());
        for task in &record.planner_output.tasks {
            println!("    - [{}] {}", task.id, task.description);
        }
        println!("  Worker results:    {}", record.worker_results.len());
        for wr in &record.worker_results {
            println!(
                "    - {} success={} iterations={}",
                wr.task_description, wr.agent_result.success, wr.agent_result.iterations
            );
        }
        println!("  Judge verdict:     {}", record.verdict.verdict_type());
        println!("  Duration:          {:.2}s", record.duration_secs);
    }

    println!("\nCycle orchestrator demo complete.");
    Ok(())
}
