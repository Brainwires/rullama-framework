//! Agent Runtime - Generic execution loop for autonomous agents
//!
//! Provides the `AgentRuntime` trait and `run_agent_loop()` function that
//! implement the core agentic execution pattern:
//!
//! ```text
//! Register → Loop {
//!     Check iteration limit
//!     Call provider
//!     Check completion (finish_reason)
//!     Extract tool uses
//!     Execute tools (with optional file locking)
//!     Add results to conversation
//! } → Complete & Unregister
//! ```
//!
//! Consumers implement `AgentRuntime` with their specific provider, tool
//! executor, and context types, then call `run_agent_loop()` to get the
//! standard orchestration with communication hub and file lock coordination.
//!
//! ## Example
//!
//! ```rust,ignore
//! use crate::runtime::{AgentRuntime, run_agent_loop, AgentExecutionResult};
//! use brainwires_agent::{CommunicationHub, FileLockManager};
//!
//! struct MyAgent { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl AgentRuntime for MyAgent {
//!     fn agent_id(&self) -> &str { "my-agent" }
//!     fn max_iterations(&self) -> usize { 20 }
//!     // ... implement other methods
//! }
//!
//! let hub = CommunicationHub::new();
//! let locks = std::sync::Arc::new(FileLockManager::new());
//! let agent = MyAgent { /* ... */ };
//! let result = run_agent_loop(&agent, &hub, &locks).await?;
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use tokio::sync::RwLock;

use brainwires_core::{ChatResponse, Message, ToolResult, ToolUse};

use crate::agent_hooks::{
    AgentLifecycleHooks, ConversationView, IterationContext, IterationDecision, ToolDecision,
};
use brainwires_agent::communication::CommunicationHub;
use brainwires_agent::file_locks::{FileLockManager, LockType};

/// Result of an agent execution loop.
#[derive(Debug, Clone)]
pub struct AgentExecutionResult {
    /// The agent's unique ID
    pub agent_id: String,
    /// Whether the agent completed successfully
    pub success: bool,
    /// Output message (completion summary or error description)
    pub output: String,
    /// Number of iterations consumed
    pub iterations: usize,
    /// Names of tools that were invoked
    pub tools_used: Vec<String>,
}

/// Tracks the last N tool-call names and detects when the same tool is called
/// consecutively (a sign the agent is stuck in a loop).
struct LoopDetector {
    window_size: usize,
    enabled: bool,
    recent: VecDeque<String>,
}

impl LoopDetector {
    fn new(window_size: usize, enabled: bool) -> Self {
        Self {
            window_size,
            enabled,
            recent: VecDeque::with_capacity(window_size),
        }
    }

    /// Record a tool call. Returns `Some(tool_name)` when a loop is detected.
    fn record(&mut self, tool_name: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if self.recent.len() == self.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(tool_name.to_string());
        if self.recent.len() == self.window_size && self.recent.iter().all(|n| n == tool_name) {
            Some(tool_name.to_string())
        } else {
            None
        }
    }
}

/// Trait that defines the core operations of an agentic execution loop.
///
/// Implementors provide the provider interaction, tool execution, and
/// completion logic. The generic [`run_agent_loop()`] function orchestrates
/// these operations with communication hub and file lock coordination.
///
/// The trait uses interior mutability (e.g. `RwLock<Vec<Message>>`) so all
/// methods take `&self` rather than `&mut self`, enabling the runtime to be
/// shared across async tasks.
#[async_trait]
pub trait AgentRuntime: Send + Sync {
    /// Get the agent's unique identifier.
    fn agent_id(&self) -> &str;

    /// Maximum number of iterations before the loop terminates.
    fn max_iterations(&self) -> usize;

    /// Call the AI provider with the current conversation state.
    ///
    /// The implementor manages its own conversation history, system prompt,
    /// tool definitions, and chat options internally.
    async fn call_provider(&self) -> Result<ChatResponse>;

    /// Extract tool use requests from a provider response.
    fn extract_tool_uses(&self, response: &ChatResponse) -> Vec<ToolUse>;

    /// Check if a response indicates the agent wants to complete.
    ///
    /// Typically checks `response.finish_reason` for "end_turn" or "stop".
    fn is_completion(&self, response: &ChatResponse) -> bool;

    /// Execute a single tool and return the result.
    async fn execute_tool(&self, tool_use: &ToolUse) -> Result<ToolResult>;

    /// Determine the file lock requirement for a tool invocation.
    ///
    /// Returns `Some((path, lock_type))` if a lock is needed before executing
    /// the tool. For example, `write_file` needs a `Write` lock on the path,
    /// while `read_file` needs a `Read` lock.
    fn get_lock_requirement(&self, tool_use: &ToolUse) -> Option<(String, LockType)>;

    /// Called when the provider returns a response that contains tool uses.
    ///
    /// The implementor should add the assistant's message (with tool use
    /// requests) to its conversation history.
    async fn on_provider_response(&self, response: &ChatResponse);

    /// Called when a tool produces a result.
    ///
    /// The implementor should add the tool result to its conversation history
    /// and update the working set if it's a file operation.
    async fn on_tool_result(&self, tool_use: &ToolUse, result: &ToolResult);

    /// Called when the agent attempts to complete.
    ///
    /// The implementor should run validation (if configured), update task
    /// status, and return `Ok(Some(output))` if completion is accepted or
    /// `Ok(None)` if validation failed and the loop should continue.
    ///
    /// When returning `None`, the implementor should inject validation
    /// feedback into the conversation history so the agent can self-correct.
    async fn on_completion(&self, response: &ChatResponse) -> Result<Option<String>>;

    /// Called when the iteration limit is reached without completion.
    ///
    /// The implementor should mark the task as failed and return a
    /// description of what happened.
    async fn on_iteration_limit(&self, iterations: usize) -> String;

    /// Optional lifecycle hooks for granular loop control.
    ///
    /// When returning `Some`, the generic [`run_agent_loop`] will call hooks
    /// at iteration boundaries, before/after provider calls, before/after
    /// tool execution, and at completion. Default: `None`.
    fn lifecycle_hooks(&self) -> Option<&dyn AgentLifecycleHooks> {
        None
    }

    /// Context budget in tokens for pressure callbacks. Default: `None`.
    fn context_budget_tokens(&self) -> Option<u64> {
        None
    }

    /// Access to the agent's conversation history for hook-based mutation.
    ///
    /// When this returns `Some`, hooks that accept a [`ConversationView`]
    /// will receive a mutable view. Default: `None` (conversation-access
    /// hooks are skipped).
    fn conversation(&self) -> Option<&RwLock<Vec<Message>>> {
        None
    }
}

/// Run the standard agent execution loop with communication hub and file
/// lock coordination.
///
/// This function implements the common agentic pattern shared across all
/// agent types: iterate calling the provider, executing requested tools
/// (with file locking when needed), and checking for completion.
///
/// The loop terminates when:
/// - The agent signals completion and validation passes (`on_completion` returns `Some`)
/// - The iteration limit is reached
/// - An unrecoverable error occurs
#[tracing::instrument(name = "agent.execute", skip_all, fields(agent_id = agent.agent_id()))]
pub async fn run_agent_loop(
    agent: &dyn AgentRuntime,
    hub: &CommunicationHub,
    lock_manager: &Arc<FileLockManager>,
) -> Result<AgentExecutionResult> {
    let agent_id = agent.agent_id().to_string();
    let mut iterations: usize = 0;
    let mut tools_used = Vec::new();
    let mut loop_detector = LoopDetector::new(5, true);
    let start_time = std::time::Instant::now();

    // Register with communication hub
    if !hub.is_registered(&agent_id).await {
        hub.register_agent(agent_id.clone()).await?;
    }

    let hooks = agent.lifecycle_hooks();

    loop {
        // Check iteration limit
        if iterations >= agent.max_iterations() {
            tracing::warn!(agent_id = %agent_id, iterations, "agent hit iteration limit");
            let output = agent.on_iteration_limit(iterations).await;
            let _ = hub.unregister_agent(&agent_id).await;
            lock_manager.release_all_locks(&agent_id).await;
            return Ok(AgentExecutionResult {
                agent_id,
                success: false,
                output,
                iterations,
                tools_used,
            });
        }

        iterations += 1;

        // ── Hook A: on_before_iteration ──────────────────────────────────
        if let Some(hooks) = hooks
            && let Some(conv_lock) = agent.conversation()
        {
            let conv_len = conv_lock.read().await.len();
            let iter_ctx = IterationContext {
                agent_id: &agent_id,
                iteration: iterations as u32,
                max_iterations: agent.max_iterations() as u32,
                total_tokens_used: 0,
                total_cost_usd: 0.0,
                elapsed: start_time.elapsed(),
                conversation_len: conv_len,
            };
            let mut history = conv_lock.write().await;
            let mut view = ConversationView::new(&mut history);
            match hooks.on_before_iteration(&iter_ctx, &mut view).await {
                IterationDecision::Continue => {}
                IterationDecision::Skip => continue,
                IterationDecision::Abort(reason) => {
                    let output = format!("Aborted by hook: {}", reason);
                    let _ = hub.unregister_agent(&agent_id).await;
                    lock_manager.release_all_locks(&agent_id).await;
                    return Ok(AgentExecutionResult {
                        agent_id,
                        success: false,
                        output,
                        iterations,
                        tools_used,
                    });
                }
            }
        }

        // ── Hook B: on_before_provider_call ──────────────────────────────
        if let Some(hooks) = hooks
            && let Some(conv_lock) = agent.conversation()
        {
            let conv_len = conv_lock.read().await.len();
            let iter_ctx = IterationContext {
                agent_id: &agent_id,
                iteration: iterations as u32,
                max_iterations: agent.max_iterations() as u32,
                total_tokens_used: 0,
                total_cost_usd: 0.0,
                elapsed: start_time.elapsed(),
                conversation_len: conv_len,
            };
            let mut history = conv_lock.write().await;
            let mut view = ConversationView::new(&mut history);
            hooks.on_before_provider_call(&iter_ctx, &mut view).await;
        }

        // Call provider
        let response = agent.call_provider().await?;

        // ── Hook C: on_after_provider_call ───────────────────────────────
        if let Some(hooks) = hooks {
            let conv_len = match agent.conversation() {
                Some(c) => c.read().await.len(),
                None => 0,
            };
            let iter_ctx = IterationContext {
                agent_id: &agent_id,
                iteration: iterations as u32,
                max_iterations: agent.max_iterations() as u32,
                total_tokens_used: 0,
                total_cost_usd: 0.0,
                elapsed: start_time.elapsed(),
                conversation_len: conv_len,
            };
            hooks.on_after_provider_call(&iter_ctx, &response).await;
        }

        // Check for completion
        if agent.is_completion(&response) {
            if let Some(output) = agent.on_completion(&response).await? {
                let _ = hub.unregister_agent(&agent_id).await;
                lock_manager.release_all_locks(&agent_id).await;
                return Ok(AgentExecutionResult {
                    agent_id,
                    success: true,
                    output,
                    iterations,
                    tools_used,
                });
            }
            // Validation failed — loop continues so agent can self-correct
            continue;
        }

        // Extract tool uses
        let tool_use_requests = agent.extract_tool_uses(&response);

        if tool_use_requests.is_empty() {
            // No tools and no explicit completion signal — try completion anyway
            if let Some(output) = agent.on_completion(&response).await? {
                let _ = hub.unregister_agent(&agent_id).await;
                lock_manager.release_all_locks(&agent_id).await;
                return Ok(AgentExecutionResult {
                    agent_id,
                    success: true,
                    output,
                    iterations,
                    tools_used,
                });
            }
            continue;
        }

        // Add the assistant's tool-use message to conversation history
        agent.on_provider_response(&response).await;

        // Execute each tool (with file locking when required)
        for tool_use in &tool_use_requests {
            // ── Hook D: on_before_tool_execution ─────────────────────────
            if let Some(hooks) = hooks {
                let conv_len = match agent.conversation() {
                    Some(c) => c.read().await.len(),
                    None => 0,
                };
                let iter_ctx = IterationContext {
                    agent_id: &agent_id,
                    iteration: iterations as u32,
                    max_iterations: agent.max_iterations() as u32,
                    total_tokens_used: 0,
                    total_cost_usd: 0.0,
                    elapsed: start_time.elapsed(),
                    conversation_len: conv_len,
                };
                match hooks.on_before_tool_execution(&iter_ctx, tool_use).await {
                    ToolDecision::Execute => {} // proceed normally
                    ToolDecision::Override(result) => {
                        agent.on_tool_result(tool_use, &result).await;
                        tools_used.push(tool_use.name.clone());
                        continue;
                    }
                    ToolDecision::Delegate(request) => {
                        match hooks.execute_delegation(&request).await {
                            Ok(delegation_result) => {
                                let result = ToolResult::success(
                                    tool_use.id.clone(),
                                    format!(
                                        "Delegated to sub-agent {}: {}",
                                        delegation_result.agent_id, delegation_result.output
                                    ),
                                );
                                agent.on_tool_result(tool_use, &result).await;
                            }
                            Err(e) => {
                                let result = ToolResult::error(
                                    tool_use.id.clone(),
                                    format!("Delegation failed: {}", e),
                                );
                                agent.on_tool_result(tool_use, &result).await;
                            }
                        }
                        tools_used.push(tool_use.name.clone());
                        continue;
                    }
                }
            }

            tools_used.push(tool_use.name.clone());

            let tool_result = if let Some((path, lock_type)) = agent.get_lock_requirement(tool_use)
            {
                // Tool needs a file lock
                match lock_manager.acquire_lock(&agent_id, &path, lock_type).await {
                    Ok(_guard) => match agent.execute_tool(tool_use).await {
                        Ok(result) => result,
                        Err(e) => ToolResult::error(
                            tool_use.id.clone(),
                            format!("Tool execution failed: {}", e),
                        ),
                    },
                    Err(e) => {
                        tracing::warn!(
                            agent_id = %agent_id,
                            path = %path,
                            "failed to acquire file lock: {}",
                            e
                        );
                        ToolResult::error(
                            tool_use.id.clone(),
                            format!("Failed to acquire lock on {}: {}", path, e),
                        )
                    }
                }
            } else {
                // No lock needed
                match agent.execute_tool(tool_use).await {
                    Ok(result) => result,
                    Err(e) => ToolResult::error(
                        tool_use.id.clone(),
                        format!("Tool execution failed: {}", e),
                    ),
                }
            };

            agent.on_tool_result(tool_use, &tool_result).await;

            // ── Hook E: on_after_tool_execution ──────────────────────────
            if let Some(hooks) = hooks
                && let Some(conv_lock) = agent.conversation()
            {
                let conv_len = conv_lock.read().await.len();
                let iter_ctx = IterationContext {
                    agent_id: &agent_id,
                    iteration: iterations as u32,
                    max_iterations: agent.max_iterations() as u32,
                    total_tokens_used: 0,
                    total_cost_usd: 0.0,
                    elapsed: start_time.elapsed(),
                    conversation_len: conv_len,
                };
                let mut history = conv_lock.write().await;
                let mut view = ConversationView::new(&mut history);
                hooks
                    .on_after_tool_execution(&iter_ctx, tool_use, &tool_result, &mut view)
                    .await;
            }
        }

        // ── Loop detection ───────────────────────────────────────────────────
        for tool_use in &tool_use_requests {
            if let Some(stuck) = loop_detector.record(&tool_use.name) {
                let output = format!(
                    "Loop detected: '{}' called {} times consecutively. Aborting.",
                    stuck, loop_detector.window_size
                );
                tracing::error!(agent_id = %agent_id, %output);
                let _ = hub.unregister_agent(&agent_id).await;
                lock_manager.release_all_locks(&agent_id).await;
                return Ok(AgentExecutionResult {
                    agent_id,
                    success: false,
                    output,
                    iterations,
                    tools_used,
                });
            }
        }

        // ── Hook G: on_after_iteration + context pressure ────────────────
        if let Some(hooks) = hooks
            && let Some(conv_lock) = agent.conversation()
        {
            let conv_len = conv_lock.read().await.len();
            let iter_ctx = IterationContext {
                agent_id: &agent_id,
                iteration: iterations as u32,
                max_iterations: agent.max_iterations() as u32,
                total_tokens_used: 0,
                total_cost_usd: 0.0,
                elapsed: start_time.elapsed(),
                conversation_len: conv_len,
            };

            // Context pressure check
            if let Some(budget) = agent.context_budget_tokens() {
                let mut history = conv_lock.write().await;
                let mut view = ConversationView::new(&mut history);
                let est_tokens = view.estimated_tokens();
                if est_tokens > budget {
                    hooks
                        .on_context_pressure(&iter_ctx, &mut view, est_tokens, budget)
                        .await;
                }
            }

            // After-iteration hook
            let mut history = conv_lock.write().await;
            let mut view = ConversationView::new(&mut history);
            hooks.on_after_iteration(&iter_ctx, &mut view).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{ContentBlock, Message, MessageContent, Role, Usage};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::RwLock;

    /// A minimal test agent that completes after a fixed number of iterations.
    struct TestAgent {
        id: String,
        max_iters: usize,
        call_count: AtomicUsize,
        complete_after: usize,
        tool_results: RwLock<Vec<String>>,
    }

    impl TestAgent {
        fn new(id: &str, max_iters: usize, complete_after: usize) -> Self {
            Self {
                id: id.to_string(),
                max_iters,
                call_count: AtomicUsize::new(0),
                complete_after,
                tool_results: RwLock::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl AgentRuntime for TestAgent {
        fn agent_id(&self) -> &str {
            &self.id
        }

        fn max_iterations(&self) -> usize {
            self.max_iters
        }

        async fn call_provider(&self) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            let finish = if count >= self.complete_after {
                Some("end_turn".to_string())
            } else {
                None
            };
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(format!("Response #{}", count)),
                    name: None,
                    metadata: None,
                },
                usage: Usage::new(10, 20),
                finish_reason: finish,
            })
        }

        fn extract_tool_uses(&self, _response: &ChatResponse) -> Vec<ToolUse> {
            vec![]
        }

        fn is_completion(&self, response: &ChatResponse) -> bool {
            response
                .finish_reason
                .as_deref()
                .is_some_and(|r| r == "end_turn" || r == "stop")
        }

        async fn execute_tool(&self, tool_use: &ToolUse) -> Result<ToolResult> {
            Ok(ToolResult::success(tool_use.id.clone(), "ok".to_string()))
        }

        fn get_lock_requirement(&self, _tool_use: &ToolUse) -> Option<(String, LockType)> {
            None
        }

        async fn on_provider_response(&self, _response: &ChatResponse) {}

        async fn on_tool_result(&self, _tool_use: &ToolUse, result: &ToolResult) {
            self.tool_results.write().await.push(result.content.clone());
        }

        async fn on_completion(&self, response: &ChatResponse) -> Result<Option<String>> {
            // Only accept completion if the provider signaled it
            if response
                .finish_reason
                .as_deref()
                .is_some_and(|r| r == "end_turn" || r == "stop")
            {
                if let MessageContent::Text(ref text) = response.message.content {
                    Ok(Some(text.clone()))
                } else {
                    Ok(Some("completed".to_string()))
                }
            } else {
                Ok(None)
            }
        }

        async fn on_iteration_limit(&self, iterations: usize) -> String {
            format!("Hit iteration limit at {}", iterations)
        }
    }

    /// A test agent that uses tools before completing.
    struct ToolUsingAgent {
        id: String,
        call_count: AtomicUsize,
    }

    impl ToolUsingAgent {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl AgentRuntime for ToolUsingAgent {
        fn agent_id(&self) -> &str {
            &self.id
        }

        fn max_iterations(&self) -> usize {
            10
        }

        async fn call_provider(&self) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                // First call: return tool use
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                            id: "tool-1".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "/tmp/test.txt"}),
                        }]),
                        name: None,
                        metadata: None,
                    },
                    usage: Usage::new(10, 20),
                    finish_reason: None,
                })
            } else {
                // Second call: complete
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Done!".to_string()),
                        name: None,
                        metadata: None,
                    },
                    usage: Usage::new(10, 20),
                    finish_reason: Some("end_turn".to_string()),
                })
            }
        }

        fn extract_tool_uses(&self, response: &ChatResponse) -> Vec<ToolUse> {
            if let MessageContent::Blocks(ref blocks) = response.message.content {
                blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolUse { id, name, input } = b {
                            Some(ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            }
        }

        fn is_completion(&self, response: &ChatResponse) -> bool {
            response
                .finish_reason
                .as_deref()
                .is_some_and(|r| r == "end_turn" || r == "stop")
        }

        async fn execute_tool(&self, tool_use: &ToolUse) -> Result<ToolResult> {
            Ok(ToolResult::success(
                tool_use.id.clone(),
                "file contents".to_string(),
            ))
        }

        fn get_lock_requirement(&self, tool_use: &ToolUse) -> Option<(String, LockType)> {
            if tool_use.name == "read_file" {
                tool_use
                    .input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|p| (p.to_string(), LockType::Read))
            } else {
                None
            }
        }

        async fn on_provider_response(&self, _response: &ChatResponse) {}

        async fn on_tool_result(&self, _tool_use: &ToolUse, _result: &ToolResult) {}

        async fn on_completion(&self, _response: &ChatResponse) -> Result<Option<String>> {
            Ok(Some("Done!".to_string()))
        }

        async fn on_iteration_limit(&self, iterations: usize) -> String {
            format!("Limit at {}", iterations)
        }
    }

    #[tokio::test]
    async fn test_agent_completes_successfully() {
        let agent = TestAgent::new("test-1", 10, 2);
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        let result = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        assert!(result.success);
        assert_eq!(result.agent_id, "test-1");
        assert_eq!(result.iterations, 3); // 2 non-completion + 1 completion
        assert!(result.tools_used.is_empty());
    }

    #[tokio::test]
    async fn test_agent_hits_iteration_limit() {
        let agent = TestAgent::new("test-2", 3, 100); // complete_after > max_iters
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        let result = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        assert!(!result.success);
        assert_eq!(result.iterations, 3);
        assert!(result.output.contains("iteration limit"));
    }

    #[tokio::test]
    async fn test_agent_with_tool_use() {
        let agent = ToolUsingAgent::new("test-3");
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        let result = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        assert!(result.success);
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tools_used, vec!["read_file"]);
    }

    #[tokio::test]
    async fn test_agent_unregisters_on_completion() {
        let agent = TestAgent::new("test-4", 10, 0);
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        let _ = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        // Agent should be unregistered after completion
        assert!(!hub.is_registered("test-4").await);
    }

    #[tokio::test]
    async fn test_agent_releases_locks_on_completion() {
        let agent = TestAgent::new("test-5", 10, 0);
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        // Pre-acquire a lock for this agent
        let _guard = locks
            .acquire_lock("test-5", "/tmp/some_file.txt", LockType::Write)
            .await
            .unwrap();
        std::mem::forget(_guard); // Prevent auto-release

        let _ = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        // Lock should be released
        let agent_locks = locks.locks_for_agent("test-5").await;
        assert!(agent_locks.is_empty());
    }

    /// Agent that always returns a tool use with the same name, triggering loop detection.
    struct LoopingAgent {
        id: String,
    }

    #[async_trait]
    impl AgentRuntime for LoopingAgent {
        fn agent_id(&self) -> &str {
            &self.id
        }
        fn max_iterations(&self) -> usize {
            100
        }

        async fn call_provider(&self) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "t".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    }]),
                    name: None,
                    metadata: None,
                },
                usage: Usage::new(10, 20),
                finish_reason: None,
            })
        }

        fn extract_tool_uses(&self, response: &ChatResponse) -> Vec<ToolUse> {
            if let MessageContent::Blocks(ref blocks) = response.message.content {
                blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolUse { id, name, input } = b {
                            Some(ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            }
        }

        fn is_completion(&self, _response: &ChatResponse) -> bool {
            false
        }

        async fn execute_tool(&self, tool_use: &ToolUse) -> Result<ToolResult> {
            Ok(ToolResult::success(tool_use.id.clone(), "ok".to_string()))
        }

        fn get_lock_requirement(&self, _tool_use: &ToolUse) -> Option<(String, LockType)> {
            None
        }
        async fn on_provider_response(&self, _response: &ChatResponse) {}
        async fn on_tool_result(&self, _tool_use: &ToolUse, _result: &ToolResult) {}
        async fn on_completion(&self, _response: &ChatResponse) -> Result<Option<String>> {
            Ok(None)
        }
        async fn on_iteration_limit(&self, iterations: usize) -> String {
            format!("Limit at {}", iterations)
        }
    }

    #[tokio::test]
    async fn test_loop_detection_aborts() {
        let agent = LoopingAgent {
            id: "loop-agent".to_string(),
        };
        let hub = CommunicationHub::new();
        let locks = Arc::new(FileLockManager::new());

        let result = run_agent_loop(&agent, &hub, &locks).await.unwrap();

        assert!(!result.success);
        assert!(
            result.output.contains("Loop detected"),
            "got: {}",
            result.output
        );
        // Loop fires after 5 consecutive same-tool calls (window_size=5)
        assert_eq!(result.tools_used.len(), 5);
    }
}
