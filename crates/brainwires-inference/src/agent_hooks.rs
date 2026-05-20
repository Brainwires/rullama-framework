//! Agent Loop Hooks - Granular control over the agent execution loop
//!
//! Provides [`AgentLifecycleHooks`] for intercepting every phase of the agent
//! loop: iteration boundaries, provider calls, tool execution, completion, and
//! context management.
//!
//! Unlike the observational [`LifecycleEvent`][brainwires_core::lifecycle::LifecycleEvent]
//! system in `brainwires-core`, these hooks can **control** the loop — skip
//! iterations, override tool results, delegate work to sub-agents, or compress
//! conversation history.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::agent_hooks::*;
//! use brainwires_agent::AgentContext;
//!
//! struct MyHooks;
//!
//! #[async_trait::async_trait]
//! impl AgentLifecycleHooks for MyHooks {
//!     async fn on_before_tool_execution(
//!         &self,
//!         _ctx: &IterationContext<'_>,
//!         tool_use: &ToolUse,
//!     ) -> ToolDecision {
//!         if tool_use.name == "write_file" {
//!             ToolDecision::Delegate(Box::new(DelegationRequest {
//!                 task_description: format!("Write file: {:?}", tool_use.input),
//!                 ..Default::default()
//!             }))
//!         } else {
//!             ToolDecision::Execute
//!         }
//!     }
//! }
//!
//! let context = AgentContext::new(/* ... */)
//!     .with_lifecycle_hooks(Arc::new(MyHooks));
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use brainwires_core::{Message, Role, ToolResult, ToolUse, estimate_tokens_from_size};

use crate::pool::AgentPool;
use crate::task_agent::TaskAgentConfig;

// ── Iteration context ────────────────────────────────────────────────────────

/// Read-only snapshot of the current iteration state, passed to every hook.
#[derive(Debug, Clone)]
pub struct IterationContext<'a> {
    /// The agent's unique identifier.
    pub agent_id: &'a str,
    /// Current iteration number (1-based).
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Cumulative tokens consumed so far.
    pub total_tokens_used: u64,
    /// Cumulative estimated cost in USD.
    pub total_cost_usd: f64,
    /// Wall-clock time since execution started.
    pub elapsed: Duration,
    /// Number of messages in the conversation history.
    pub conversation_len: usize,
}

// ── Decision enums ───────────────────────────────────────────────────────────

/// Decision returned by [`AgentLifecycleHooks::on_before_iteration`].
#[derive(Debug, Clone)]
pub enum IterationDecision {
    /// Proceed with the normal iteration.
    Continue,
    /// Skip the provider call this iteration. Use when the hook already
    /// injected messages or handled the iteration itself.
    Skip,
    /// Abort the agent loop with a failure message.
    Abort(String),
}

/// Decision returned by [`AgentLifecycleHooks::on_before_tool_execution`].
#[derive(Debug, Clone)]
pub enum ToolDecision {
    /// Execute the tool normally.
    Execute,
    /// Skip execution and inject this result instead.
    Override(ToolResult),
    /// Delegate the tool call to a sub-agent.
    Delegate(Box<DelegationRequest>),
}

// ── Delegation types ─────────────────────────────────────────────────────────

/// A request to delegate work to a sub-agent.
#[derive(Debug, Clone)]
pub struct DelegationRequest {
    /// Description of the sub-task for the spawned agent.
    pub task_description: String,
    /// Optional config override for the sub-agent.
    pub config: Option<TaskAgentConfig>,
    /// Messages to seed the sub-agent's conversation with.
    pub seed_messages: Vec<Message>,
    /// If `true`, block until the sub-agent completes and return its output.
    /// If `false`, the framework injects a placeholder and continues.
    pub blocking: bool,
}

impl Default for DelegationRequest {
    fn default() -> Self {
        Self {
            task_description: String::new(),
            config: None,
            seed_messages: Vec::new(),
            blocking: true,
        }
    }
}

/// Result of a completed delegation.
#[derive(Debug, Clone)]
pub struct DelegationResult {
    /// The sub-agent's unique ID.
    pub agent_id: String,
    /// Whether the sub-agent completed successfully.
    pub success: bool,
    /// Output summary from the sub-agent.
    pub output: String,
    /// Iterations the sub-agent consumed.
    pub iterations_used: u32,
    /// Tokens the sub-agent consumed.
    pub tokens_used: u64,
}

// ── Conversation view ────────────────────────────────────────────────────────

/// Controlled read/write handle to the conversation history.
///
/// Passed to hooks that need to inspect or mutate messages (e.g., for
/// summarization or context injection). The underlying `Vec<Message>` is
/// borrowed mutably for the duration of the hook call.
pub struct ConversationView<'a> {
    messages: &'a mut Vec<Message>,
}

impl<'a> ConversationView<'a> {
    /// Create a new view over a message vector.
    pub fn new(messages: &'a mut Vec<Message>) -> Self {
        Self { messages }
    }

    /// Number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Read-only access to all messages.
    pub fn messages(&self) -> &[Message] {
        self.messages
    }

    /// The last `n` messages (or all if fewer exist).
    pub fn last_n(&self, n: usize) -> &[Message] {
        let start = self.messages.len().saturating_sub(n);
        &self.messages[start..]
    }

    /// Append a message to the end of the conversation.
    pub fn push(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Insert a message at a specific position.
    pub fn insert(&mut self, index: usize, msg: Message) {
        self.messages.insert(index, msg);
    }

    /// Remove and return messages in the given range.
    pub fn drain(&mut self, range: std::ops::Range<usize>) -> Vec<Message> {
        self.messages.drain(range).collect()
    }

    /// Replace a range of messages with a single summary message.
    ///
    /// Useful for compressing old context to save tokens.
    pub fn summarize_range(&mut self, range: std::ops::Range<usize>, summary: Message) {
        let start = range.start;
        self.messages.drain(range);
        self.messages.insert(start, summary);
    }

    /// Estimate total tokens across all messages.
    ///
    /// Uses byte-length heuristic via [`estimate_tokens_from_size`].
    pub fn estimated_tokens(&self) -> u64 {
        self.messages
            .iter()
            .map(|m| {
                let bytes = match &m.content {
                    brainwires_core::MessageContent::Text(t) => t.len() as u64,
                    brainwires_core::MessageContent::Blocks(blocks) => {
                        blocks.iter().map(|b| format!("{:?}", b).len() as u64).sum()
                    }
                };
                estimate_tokens_from_size(bytes) as u64
            })
            .sum()
    }

    /// Get the text of the most recent assistant message, if any.
    pub fn last_assistant_text(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .and_then(|m| m.text())
    }
}

// ── Main trait ───────────────────────────────────────────────────────────────

/// Granular lifecycle hooks for controlling an agent's execution loop.
///
/// All methods have default no-op implementations — consumers only override
/// the hooks they need. Hooks receive an [`IterationContext`] for read-only
/// state inspection, and some receive a [`ConversationView`] for history
/// manipulation.
///
/// # Relationship to `brainwires_core::lifecycle`
///
/// The core lifecycle system is **observational** — hooks receive events and
/// can cancel operations, but cannot modify conversation state or delegate work.
///
/// This trait is **controlling** — hooks can skip iterations, override tool
/// results, delegate to sub-agents, and rewrite conversation history. It is
/// wired into the agent loop itself, not the event dispatch system.
#[async_trait]
pub trait AgentLifecycleHooks: Send + Sync {
    // ── Iteration-level hooks ────────────────────────────────────────

    /// Called at the top of each iteration, before budget checks and the
    /// provider call.
    ///
    /// Return [`IterationDecision::Skip`] to skip the provider call
    /// (e.g., if you injected messages yourself).
    /// Return [`IterationDecision::Abort`] to stop the loop with failure.
    async fn on_before_iteration(
        &self,
        _ctx: &IterationContext<'_>,
        _conversation: &mut ConversationView<'_>,
    ) -> IterationDecision {
        IterationDecision::Continue
    }

    /// Called after all tools have been executed (or after a completion
    /// check), before the next iteration starts.
    ///
    /// Good for context management, summarization, and metrics.
    async fn on_after_iteration(
        &self,
        _ctx: &IterationContext<'_>,
        _conversation: &mut ConversationView<'_>,
    ) {
    }

    // ── Provider call hooks ──────────────────────────────────────────

    /// Called immediately before the provider is called.
    ///
    /// Can modify conversation (e.g., inject context summaries).
    async fn on_before_provider_call(
        &self,
        _ctx: &IterationContext<'_>,
        _conversation: &mut ConversationView<'_>,
    ) {
    }

    /// Called immediately after the provider returns a response.
    async fn on_after_provider_call(
        &self,
        _ctx: &IterationContext<'_>,
        _response: &brainwires_core::ChatResponse,
    ) {
    }

    // ── Tool execution hooks ─────────────────────────────────────────

    /// Called before each tool is executed.
    ///
    /// Return [`ToolDecision::Delegate`] to spawn a sub-agent instead of
    /// executing the tool directly. Return [`ToolDecision::Override`] to
    /// skip execution and inject a custom result.
    async fn on_before_tool_execution(
        &self,
        _ctx: &IterationContext<'_>,
        _tool_use: &ToolUse,
    ) -> ToolDecision {
        ToolDecision::Execute
    }

    /// Called after each tool execution completes.
    ///
    /// Can inspect the result and modify conversation (e.g., spawn a
    /// sub-agent to analyze complex results).
    async fn on_after_tool_execution(
        &self,
        _ctx: &IterationContext<'_>,
        _tool_use: &ToolUse,
        _result: &ToolResult,
        _conversation: &mut ConversationView<'_>,
    ) {
    }

    // ── Completion hooks ─────────────────────────────────────────────

    /// Called when the agent signals completion, before validation runs.
    ///
    /// Return `false` to reject the completion attempt and force the
    /// agent to continue iterating.
    async fn on_before_completion(
        &self,
        _ctx: &IterationContext<'_>,
        _completion_text: &str,
    ) -> bool {
        true
    }

    /// Called after a successful completion (validation passed).
    async fn on_after_completion(
        &self,
        _ctx: &IterationContext<'_>,
        _result: &crate::task_agent::TaskAgentResult,
    ) {
    }

    // ── Context management hooks ─────────────────────────────────────

    /// Called when the estimated conversation token count exceeds the
    /// configured budget.
    ///
    /// The consumer can summarize, compress, or evict messages to stay
    /// within the budget. Only called when
    /// [`TaskAgentConfig::context_budget_tokens`] is set.
    async fn on_context_pressure(
        &self,
        _ctx: &IterationContext<'_>,
        _conversation: &mut ConversationView<'_>,
        _estimated_tokens: u64,
        _budget_tokens: u64,
    ) {
    }

    // ── Delegation hooks ─────────────────────────────────────────────

    /// Called when a [`ToolDecision::Delegate`] needs to be fulfilled.
    ///
    /// The default returns an error — consumers must override this if
    /// they use delegation. See [`DefaultDelegationHandler`] for a
    /// ready-made implementation wrapping [`AgentPool`].
    async fn execute_delegation(&self, _request: &DelegationRequest) -> Result<DelegationResult> {
        Err(anyhow::anyhow!(
            "Delegation unsupported by this hook provider. \
             Override execute_delegation() or use DefaultDelegationHandler."
        ))
    }
}

// ── Default delegation handler ───────────────────────────────────────────────

/// Convenience handler that delegates work via an [`AgentPool`].
///
/// Consumers can compose this into their own [`AgentLifecycleHooks`]
/// implementation to get delegation support without writing pool logic:
///
/// ```rust,ignore
/// struct MyHooks {
///     delegator: DefaultDelegationHandler,
/// }
///
/// #[async_trait::async_trait]
/// impl AgentLifecycleHooks for MyHooks {
///     async fn execute_delegation(&self, req: &DelegationRequest) -> Result<DelegationResult> {
///         self.delegator.delegate(req).await
///     }
/// }
/// ```
pub struct DefaultDelegationHandler {
    pool: Arc<AgentPool>,
}

impl DefaultDelegationHandler {
    /// Create a handler backed by the given agent pool.
    pub fn new(pool: Arc<AgentPool>) -> Self {
        Self { pool }
    }

    /// Execute a delegation request by spawning a sub-agent in the pool.
    pub async fn delegate(&self, request: &DelegationRequest) -> Result<DelegationResult> {
        let task = brainwires_core::Task::new(
            uuid::Uuid::new_v4().to_string(),
            request.task_description.clone(),
        );
        let agent_id = self.pool.spawn_agent(task, request.config.clone()).await?;

        if request.blocking {
            let result = self.pool.await_completion(&agent_id).await?;
            Ok(DelegationResult {
                agent_id: result.agent_id,
                success: result.success,
                output: result.summary,
                iterations_used: result.iterations,
                tokens_used: result.total_tokens_used,
            })
        } else {
            Ok(DelegationResult {
                agent_id,
                success: true,
                output: "Delegation started (non-blocking)".to_string(),
                iterations_used: 0,
                tokens_used: 0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{Message, MessageContent, Role};

    #[test]
    fn test_conversation_view_len_and_empty() {
        let mut msgs = vec![];
        let view = ConversationView::new(&mut msgs);
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
    }

    #[test]
    fn test_conversation_view_push_and_messages() {
        let mut msgs = vec![];
        let mut view = ConversationView::new(&mut msgs);
        view.push(Message::user("hello"));
        view.push(Message::user("world"));
        assert_eq!(view.len(), 2);
        assert_eq!(view.messages()[0].text(), Some("hello"));
    }

    #[test]
    fn test_conversation_view_last_n() {
        let mut msgs = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        let view = ConversationView::new(&mut msgs);
        assert_eq!(view.last_n(2).len(), 2);
        assert_eq!(view.last_n(2)[0].text(), Some("b"));
        assert_eq!(view.last_n(100).len(), 3); // clamps to total
    }

    #[test]
    fn test_conversation_view_drain() {
        let mut msgs = vec![
            Message::user("keep"),
            Message::user("remove1"),
            Message::user("remove2"),
            Message::user("keep2"),
        ];
        let mut view = ConversationView::new(&mut msgs);
        let removed = view.drain(1..3);
        assert_eq!(removed.len(), 2);
        assert_eq!(view.len(), 2);
        assert_eq!(view.messages()[0].text(), Some("keep"));
        assert_eq!(view.messages()[1].text(), Some("keep2"));
    }

    #[test]
    fn test_conversation_view_summarize_range() {
        let mut msgs = vec![
            Message::user("first"),
            Message::user("old1"),
            Message::user("old2"),
            Message::user("old3"),
            Message::user("last"),
        ];
        let mut view = ConversationView::new(&mut msgs);
        view.summarize_range(1..4, Message::user("[summary of 3 messages]"));
        assert_eq!(view.len(), 3);
        assert_eq!(view.messages()[0].text(), Some("first"));
        assert_eq!(view.messages()[1].text(), Some("[summary of 3 messages]"));
        assert_eq!(view.messages()[2].text(), Some("last"));
    }

    #[test]
    fn test_conversation_view_estimated_tokens() {
        let mut msgs = vec![Message::user("hello world, this is a test message")];
        let view = ConversationView::new(&mut msgs);
        let tokens = view.estimated_tokens();
        assert!(tokens > 0);
    }

    #[test]
    fn test_conversation_view_last_assistant_text() {
        let mut msgs = vec![
            Message::user("question"),
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("answer".to_string()),
                name: None,
                metadata: None,
            },
            Message::user("follow-up"),
        ];
        let view = ConversationView::new(&mut msgs);
        assert_eq!(view.last_assistant_text(), Some("answer"));
    }

    #[test]
    fn test_conversation_view_insert() {
        let mut msgs = vec![Message::user("first"), Message::user("last")];
        let mut view = ConversationView::new(&mut msgs);
        view.insert(1, Message::user("middle"));
        assert_eq!(view.len(), 3);
        assert_eq!(view.messages()[1].text(), Some("middle"));
    }

    #[test]
    fn test_iteration_decision_variants() {
        let _continue = IterationDecision::Continue;
        let _skip = IterationDecision::Skip;
        let _abort = IterationDecision::Abort("reason".to_string());
    }

    #[test]
    fn test_tool_decision_variants() {
        let _execute = ToolDecision::Execute;
        let _override =
            ToolDecision::Override(ToolResult::success("id".to_string(), "ok".to_string()));
        let _delegate = ToolDecision::Delegate(Box::default());
    }

    #[test]
    fn test_delegation_request_default() {
        let req = DelegationRequest::default();
        assert!(req.task_description.is_empty());
        assert!(req.config.is_none());
        assert!(req.seed_messages.is_empty());
        assert!(req.blocking);
    }
}
