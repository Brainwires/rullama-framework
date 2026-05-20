//! Example: Concurrent agent management with AgentPool
//!
//! Shows how to set up shared infrastructure (CommunicationHub, FileLockManager),
//! spawn multiple agents into an `AgentPool`, monitor their status, and collect
//! results. Uses a simple `MockProvider` that returns "Done" immediately.
//!
//! Run: cargo run -p brainwires-inference --example agent_pool

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_agent::brainwires_core::{
    ChatOptions, ChatResponse, Message, Provider, StreamChunk, Task, Tool, ToolContext, ToolResult,
    ToolUse, Usage,
};
use brainwires_agent::brainwires_tool_runtime::ToolExecutor;
use brainwires_agent::{AgentMessage, CommunicationHub, FileLockManager, LockType};
use brainwires_inference::{AgentPool, TaskAgentConfig};

// ── Mock Provider ──────────────────────────────────────────────────────────

/// A minimal provider that always returns a fixed "Done" response.
struct MockProvider;

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::assistant("Done"),
            finish_reason: Some("stop".to_string()),
            usage: Usage::default(),
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

// ── No-Op Tool Executor ───────────────────────────────────────────────────

/// A tool executor that returns "ok" for any tool call.
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
    tracing_subscriber::fmt::init();

    // 1. Create shared infrastructure
    let hub = Arc::new(CommunicationHub::new());
    let lock_manager = Arc::new(FileLockManager::new());

    println!("=== Agent Pool Demo ===\n");

    // 2. Create the agent pool (max 5 concurrent agents)
    let pool = AgentPool::new(
        5,
        Arc::new(MockProvider),
        Arc::new(NoOpExecutor),
        Arc::clone(&hub),
        Arc::clone(&lock_manager),
        "/tmp/demo-project",
    );

    // 3. Spawn multiple agents with different tasks
    let make_config = || {
        let mut cfg = TaskAgentConfig::default();
        cfg.validation_config = None;
        Some(cfg)
    };

    let id1 = pool
        .spawn_agent(
            Task::new("task-1", "Implement authentication module"),
            make_config(),
        )
        .await?;
    println!("Spawned agent: {id1}");

    let id2 = pool
        .spawn_agent(
            Task::new("task-2", "Add unit tests for parser"),
            make_config(),
        )
        .await?;
    println!("Spawned agent: {id2}");

    let id3 = pool
        .spawn_agent(
            Task::new("task-3", "Refactor error handling"),
            make_config(),
        )
        .await?;
    println!("Spawned agent: {id3}");

    // 4. Print pool stats
    let stats = pool.stats().await;
    println!("\nPool stats:");
    println!("  Max agents:   {}", stats.max_agents);
    println!("  Total agents: {}", stats.total_agents);
    println!("  Running:      {}", stats.running);
    println!("  Completed:    {}", stats.completed);

    // 5. List active agents
    let active = pool.list_active().await;
    println!("\nActive agents:");
    for (id, status) in &active {
        println!("  {}: {}", &id[..20.min(id.len())], status);
    }

    // 6. Await all agents and print results
    println!("\nAwaiting all agent completions...");
    let results = pool.await_all().await;
    for (id, result) in &results {
        match result {
            Ok(r) => println!(
                "  {} => success={}, iterations={}, summary={}",
                &id[..20.min(id.len())],
                r.success,
                r.iterations,
                &r.summary[..50.min(r.summary.len())]
            ),
            Err(e) => println!("  {} => error: {}", &id[..20.min(id.len())], e),
        }
    }

    // 7. Demonstrate file lock coordination
    println!("\n=== File Lock Coordination ===");

    // Multiple concurrent reads are allowed
    {
        let read1 = lock_manager
            .acquire_lock("demo-agent-1", "src/lib.rs", LockType::Read)
            .await?;
        let read2 = lock_manager
            .acquire_lock("demo-agent-2", "src/lib.rs", LockType::Read)
            .await?;
        println!("Two agents reading src/lib.rs concurrently - OK");

        // Explicitly release before the guards drop (Drop spawns async tasks
        // that race with the next acquire_lock call). Forget the guards to
        // prevent the Drop handler from double-releasing.
        lock_manager
            .release_lock("demo-agent-1", "src/lib.rs", LockType::Read)
            .await?;
        lock_manager
            .release_lock("demo-agent-2", "src/lib.rs", LockType::Read)
            .await?;
        std::mem::forget(read1);
        std::mem::forget(read2);
    }

    // Exclusive write lock
    {
        let _write = lock_manager
            .acquire_lock("demo-agent-1", "src/lib.rs", LockType::Write)
            .await?;
        println!("One agent has exclusive write access to src/lib.rs - OK");
    }

    // 8. Demonstrate hub message broadcasting
    println!("\n=== Communication Hub ===");

    hub.register_agent("listener-1".to_string()).await?;
    hub.register_agent("listener-2".to_string()).await?;

    hub.broadcast(
        "orchestrator".to_string(),
        AgentMessage::StatusUpdate {
            agent_id: "orchestrator".to_string(),
            status: "planning".to_string(),
            details: Some("Starting cycle 1".to_string()),
        },
    )
    .await?;

    // Each registered agent receives the broadcast
    for listener in &["listener-1", "listener-2"] {
        match hub.receive_message(listener).await {
            Some(env) => println!("  {listener} received: {:?}", env.message),
            None => println!("  {listener}: no messages"),
        }
    }

    println!("\nAgent pool demo complete.");
    Ok(())
}
