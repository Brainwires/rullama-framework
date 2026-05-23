//! A simple chat agent that processes messages through an LLM provider with tool support.
//!
//! [`ChatAgent`] is the framework's ready-to-use agent for text message to response
//! flows, including automatic tool call dispatch via a `BuiltinToolExecutor`
//! (from `brainwires-tool-builtins`).

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;

use brainwires_call_policy::BudgetGuard;
use brainwires_core::{
    ChatOptions, ContentBlock, Message, MessageContent, Provider, Role, StreamChunk, Tool,
    ToolContext, ToolUse, Usage,
};
use brainwires_tool_runtime::{PreHookDecision, ToolExecutor, ToolPreHook};

use crate::summarization::Summarizer;

/// Rough character-level token estimator for the auto-compact threshold.
///
/// Used by [`ChatAgent::with_auto_compact_at`]. The estimator mirrors the
/// 4-chars-per-token heuristic in `brainwires_call_policy::budget` —
/// imprecise but cheap and dep-free. When `brainwires-call-policy`'s
/// `Tokenizer` trait lands (Tier 2.1) this helper should be replaced by
/// a per-provider tokenizer for ±5% accuracy.
fn estimate_history_tokens(messages: &[Message]) -> u64 {
    let mut chars: usize = 0;
    for m in messages {
        match &m.content {
            MessageContent::Text(t) => chars += t.len(),
            MessageContent::Blocks(blocks) => {
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => chars += text.len(),
                        ContentBlock::ToolUse { input, .. } => chars += input.to_string().len(),
                        ContentBlock::ToolResult { content, .. } => chars += content.len(),
                        ContentBlock::Image { .. } => chars += 512,
                    }
                }
            }
        }
    }
    (chars as u64) / 4
}

/// A simple chat agent that processes messages through an LLM provider with tool support.
///
/// This is the framework's ready-to-use agent for text message -> response flows.
/// It manages conversation history, streams responses from the provider, and
/// automatically dispatches tool calls through a `BuiltinToolExecutor`
/// (from `brainwires-tool-builtins`).
///
/// # Example
///
/// ```rust,ignore
/// use brainwires_agent::ChatAgent;
/// use brainwires_tool_builtins::BuiltinToolExecutor;
/// use brainwires_tool_runtime::ToolRegistry;
/// use brainwires_core::{ChatOptions, ToolContext};
/// use std::sync::Arc;
///
/// let provider = /* create a provider */;
/// let registry = brainwires_tool_builtins::registry_with_builtins();
/// let context = ToolContext::default();
/// let executor = Arc::new(BuiltinToolExecutor::new(registry, context));
/// let options = ChatOptions::default();
///
/// let mut agent = ChatAgent::new(provider, executor, options)
///     .with_system_prompt("You are a helpful assistant.")
///     .with_max_tool_rounds(5);
///
/// let response = agent.process_message("Hello!").await?;
/// println!("{}", response);
/// ```
pub struct ChatAgent {
    provider: Arc<dyn Provider>,
    executor: Arc<dyn ToolExecutor>,
    messages: Vec<Message>,
    options: ChatOptions,
    max_tool_rounds: usize,
    pre_execute_hook: Option<Arc<dyn ToolPreHook>>,
    /// Accumulated token usage across all completions in this session.
    cumulative_usage: Usage,
    /// Optional shared budget guard enforcing token/cost/round caps across
    /// this agent (and any others sharing the same guard).
    budget: Option<BudgetGuard>,
    /// Maximum concurrent tool executions within a single agent round.
    ///
    /// Tools with `Tool::serialize == true` always run sequentially, before
    /// the parallel batch. `1` preserves the legacy fully-sequential behavior.
    tool_concurrency: usize,
    /// Optional summarizer used by [`Self::compact_history`] to replace the
    /// middle of the conversation with an LLM-generated summary instead of
    /// plain trimming.
    summarizer: Option<Arc<dyn Summarizer>>,
    /// How many trailing messages to preserve verbatim when summarization
    /// runs. Default 6. The first message (if system) is always kept.
    summarization_keep_tail: usize,
    /// When `Some(n)`, the agent calls [`Self::compact_history`] before
    /// each provider call if the rough-token estimate of `messages`
    /// exceeds `n`. The intended value is `0.85 * model_context_window`
    /// — e.g. `0.85 * 200_000 ≈ 170_000` for a 200k-token context.
    /// `None` (default) preserves the legacy hands-off behaviour.
    auto_compact_at_tokens: Option<u64>,
}

impl ChatAgent {
    /// Create a new `ChatAgent`.
    ///
    /// Defaults `max_tool_rounds` to 10.
    pub fn new(
        provider: Arc<dyn Provider>,
        executor: Arc<dyn ToolExecutor>,
        options: ChatOptions,
    ) -> Self {
        Self {
            provider,
            executor,
            messages: Vec::new(),
            options,
            max_tool_rounds: 10,
            pre_execute_hook: None,
            cumulative_usage: Usage::default(),
            budget: None,
            tool_concurrency: 4,
            summarizer: None,
            summarization_keep_tail: 6,
            auto_compact_at_tokens: None,
        }
    }

    /// Enable automatic history compaction when the rough token estimate
    /// of the conversation exceeds `threshold_tokens`. Compaction uses the
    /// configured [`Summarizer`] if one is attached, otherwise a plain
    /// tail-keep trim. Pass `0.85 * model_context_window` for a typical
    /// safety margin against context-window overflow.
    ///
    /// The check runs once per provider call inside `run_completion`, so a
    /// runaway tool loop that grows history rapidly still gets compacted
    /// before the next round.
    pub fn with_auto_compact_at(mut self, threshold_tokens: u64) -> Self {
        self.auto_compact_at_tokens = Some(threshold_tokens);
        self
    }

    /// Attach a [`Summarizer`]. When set, [`Self::compact_history`] will
    /// invoke the summarizer on the middle of the conversation instead of
    /// dropping those messages.
    pub fn with_summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Override the number of trailing messages preserved verbatim during
    /// summarization. Default: 6.
    pub fn with_summarization_keep_tail(mut self, keep: usize) -> Self {
        self.summarization_keep_tail = keep.max(1);
        self
    }

    /// Set the maximum number of tool calls to dispatch concurrently within a
    /// single agent round. `1` forces fully sequential execution (the legacy
    /// behavior prior to 0.11). The default is 4.
    ///
    /// Tools whose [`Tool::serialize`] flag is set always run sequentially
    /// regardless of this value.
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.tool_concurrency = concurrency.max(1);
        self
    }

    /// Set the maximum number of tool-call rounds before the agent stops.
    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.max_tool_rounds = rounds;
        self
    }

    /// Attach a shared [`BudgetGuard`].
    ///
    /// When set, each tool round checks the guard before invoking the provider;
    /// a tripped cap ends the loop and surfaces a
    /// `brainwires_call_policy::ResilienceError::BudgetExceeded` error (wrapped
    /// in `anyhow::Error`).
    ///
    /// The same guard can be shared across multiple agents to enforce a
    /// session-wide or tenant-wide budget.
    pub fn with_budget(mut self, guard: BudgetGuard) -> Self {
        self.budget = Some(guard);
        self
    }

    /// Attach a pre-execution hook that can allow or reject tool calls before they run.
    pub fn with_pre_execute_hook(mut self, hook: Arc<dyn ToolPreHook>) -> Self {
        self.pre_execute_hook = Some(hook);
        self
    }

    /// Add a system prompt as the first message in the conversation.
    ///
    /// If messages already exist, the system message is inserted at position 0.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        // Remove any existing system message at position 0
        if let Some(first) = self.messages.first()
            && first.role == Role::System
        {
            self.messages.remove(0);
        }
        self.messages.insert(0, Message::system(prompt));
        self
    }

    /// Process a user message and return the final assistant text response.
    ///
    /// This is the core completion loop:
    /// 1. Adds the user message to history
    /// 2. Streams the provider response, collecting text and tool calls
    /// 3. If tool calls are present, executes them and loops
    /// 4. Returns the final accumulated text once no more tool calls remain
    ///    (or `max_tool_rounds` is reached)
    pub async fn process_message(&mut self, input: &str) -> Result<String> {
        self.messages.push(Message::user(input));
        self.run_completion(None::<fn(&str)>).await
    }

    /// Process a user message and return both the assistant text AND a
    /// [`TurnReport`](brainwires_core::TurnReport) of token / duration usage for this single turn.
    ///
    /// The report is computed by snapshotting `cumulative_usage` before
    /// and after the turn, plus measuring wall-clock duration. Useful
    /// for cost dashboards, per-turn billing, and debugging which
    /// turn ran the budget down.
    pub async fn process_message_with_report(
        &mut self,
        input: &str,
    ) -> Result<(String, brainwires_core::TurnReport)> {
        let before = self.cumulative_usage.clone();
        let started = std::time::Instant::now();
        self.messages.push(Message::user(input));
        let text = self.run_completion(None::<fn(&str)>).await?;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let report = brainwires_core::TurnReport::from_usage_delta(
            &before,
            &self.cumulative_usage,
            elapsed_ms,
        );
        Ok((text, report))
    }

    /// Process a user message with streaming — calls `on_chunk` for each text
    /// fragment as it arrives from the provider.
    ///
    /// Returns the full accumulated text once the completion loop finishes.
    pub async fn process_message_streaming<F>(&mut self, input: &str, on_chunk: F) -> Result<String>
    where
        F: Fn(&str) + Send + Sync,
    {
        self.messages.push(Message::user(input));
        self.run_completion(Some(on_chunk)).await
    }

    /// Access the conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Replace the entire message history with the provided messages.
    ///
    /// This is used by session persistence to restore a previously saved
    /// conversation when an agent session is recreated.
    pub fn restore_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Clear all messages (including any system prompt).
    pub fn clear_history(&mut self) {
        self.messages.clear();
    }

    /// Keep only the last `max_messages` messages, preserving the system prompt
    /// at position 0 if one exists.
    pub fn trim_history(&mut self, max_messages: usize) {
        if self.messages.len() <= max_messages {
            return;
        }

        let has_system = self
            .messages
            .first()
            .map(|m| m.role == Role::System)
            .unwrap_or(false);

        if has_system && max_messages > 0 {
            let system = self.messages.remove(0);
            let keep = max_messages.saturating_sub(1);
            let start = self.messages.len().saturating_sub(keep);
            self.messages = std::iter::once(system)
                .chain(self.messages.drain(start..))
                .collect();
        } else {
            let start = self.messages.len().saturating_sub(max_messages);
            self.messages = self.messages.drain(start..).collect();
        }
    }

    /// Return the number of messages in the conversation.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Return the accumulated token usage for this agent session.
    ///
    /// Counts prompt + completion tokens across all completions. Updated
    /// whenever the provider emits a `StreamChunk::Usage` event.
    pub fn cumulative_usage(&self) -> &Usage {
        &self.cumulative_usage
    }

    /// Reset the cumulative token usage counter.
    pub fn reset_usage(&mut self) {
        self.cumulative_usage = Usage::default();
    }

    /// Compact conversation history.
    ///
    /// If a [`Summarizer`] has been attached via
    /// [`Self::with_summarizer`], the middle of the conversation is replaced
    /// with an LLM-generated summary injected as a synthetic assistant turn;
    /// the system prompt and the last `summarization_keep_tail` messages
    /// (default 6) are preserved verbatim.
    ///
    /// Without a summarizer this falls back to a plain trim keeping the
    /// system prompt plus the last 20 messages.
    pub async fn compact_history(&mut self) -> Result<()> {
        let Some(summarizer) = self.summarizer.clone() else {
            self.trim_history(20);
            return Ok(());
        };

        let keep_tail = self.summarization_keep_tail;
        if self.messages.len() <= keep_tail + 1 {
            // Nothing worth compacting yet.
            return Ok(());
        }

        let has_system = self
            .messages
            .first()
            .map(|m| m.role == Role::System)
            .unwrap_or(false);
        let head_end = if has_system { 1 } else { 0 };
        let tail_start = self.messages.len().saturating_sub(keep_tail);
        if tail_start <= head_end {
            return Ok(());
        }

        let to_summarize: Vec<Message> = self.messages[head_end..tail_start].to_vec();
        let summary = summarizer.summarize(&to_summarize).await?;

        // Replace the middle span with a single synthetic assistant message
        // so the downstream provider sees a contiguous, well-formed history.
        let synthetic = Message::assistant(format!("[Prior conversation summary] {summary}"));
        let tail: Vec<Message> = self.messages[tail_start..].to_vec();

        let mut new_messages = Vec::with_capacity(head_end + 1 + tail.len());
        if has_system {
            new_messages.push(self.messages[0].clone());
        }
        new_messages.push(synthetic);
        new_messages.extend(tail);
        self.messages = new_messages;

        Ok(())
    }

    // ── Internal completion loop ─────────────────────────────────────────

    async fn run_completion<F>(&mut self, on_chunk: Option<F>) -> Result<String>
    where
        F: Fn(&str) + Send + Sync,
    {
        let mut final_text = String::new();

        for _ in 0..self.max_tool_rounds {
            // Auto-compact pre-flight: if a threshold is configured and the
            // rough token estimate of the current history is past it, compact
            // before issuing the next provider call. Catches the "agent grew
            // its own history past the context window via tool loops" failure
            // mode that would otherwise surface as a hard provider error.
            if let Some(threshold) = self.auto_compact_at_tokens
                && estimate_history_tokens(&self.messages) > threshold
            {
                self.compact_history().await?;
            }

            // Budget pre-flight: stop the loop if any cap has been reached.
            if let Some(ref guard) = self.budget {
                guard.check_and_tick().map_err(anyhow::Error::from)?;
            }

            let tool_defs: Vec<Tool> = self.executor.available_tools();
            let tools_opt = if tool_defs.is_empty() {
                None
            } else {
                Some(tool_defs.as_slice())
            };

            let (text_buf, tool_uses, response_id, compaction) =
                self.collect_stream(tools_opt, &on_chunk).await?;

            // Apply context compaction if the model summarised the history.
            // Must happen after collect_stream returns so the stream's borrow
            // on self.messages is released.
            if let Some((summary, tokens_freed)) = compaction {
                tracing::info!(
                    tokens_freed = ?tokens_freed,
                    "context compaction triggered; replacing history with model summary"
                );
                let system_msg = self
                    .messages
                    .iter()
                    .find(|m| m.role == Role::System)
                    .cloned();
                self.messages.clear();
                if let Some(sys) = system_msg {
                    self.messages.push(sys);
                }
                self.messages.push(Message::assistant(&summary));
            }

            if tool_uses.is_empty() {
                // No tool calls — this is the final response
                self.messages.push(Message::assistant(&text_buf));
                final_text = text_buf;
                break;
            }

            // Build assistant message with text + tool use blocks
            let mut blocks = Vec::new();
            if !text_buf.is_empty() {
                blocks.push(ContentBlock::Text {
                    text: text_buf.clone(),
                });
            }
            for tu in &tool_uses {
                blocks.push(ContentBlock::ToolUse {
                    id: tu.id.clone(),
                    name: tu.name.clone(),
                    input: tu.input.clone(),
                });
            }
            let metadata = response_id.map(|rid| serde_json::json!({"response_id": rid}));
            self.messages.push(Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(blocks),
                name: None,
                metadata,
            });

            // Execute tool calls, honoring each tool's `serialize` flag.
            //
            // Tools marked `serialize: true` (write-like, state-mutating) run
            // sequentially first so their side effects are ordered.
            // Remaining tools dispatch concurrently with
            // `buffer_unordered(self.tool_concurrency)`, but result order in
            // the outgoing user message still matches the input `tool_uses`
            // order via position-indexed slots.
            let serialize_map: std::collections::HashMap<&str, bool> = tool_defs
                .iter()
                .map(|t| (t.name.as_str(), t.serialize))
                .collect();

            let (serial_idx, parallel_idx): (Vec<usize>, Vec<usize>) = (0..tool_uses.len())
                .partition(|&i| {
                    serialize_map
                        .get(tool_uses[i].name.as_str())
                        .copied()
                        .unwrap_or(false)
                });

            let mut slots: Vec<Option<ContentBlock>> = (0..tool_uses.len()).map(|_| None).collect();

            // Serial tools — preserve legacy behavior for mutating tools.
            for i in serial_idx {
                let tu = &tool_uses[i];
                let block =
                    execute_one_tool(tu, self.executor.clone(), self.pre_execute_hook.clone())
                        .await;
                slots[i] = Some(block);
            }

            // Parallel-eligible tools — dispatch with bounded concurrency.
            if !parallel_idx.is_empty() {
                use futures::StreamExt as _;
                use futures::future::BoxFuture;

                let concurrency = self.tool_concurrency.max(1);
                let executor = self.executor.clone();
                let hook = self.pre_execute_hook.clone();

                let futures: Vec<BoxFuture<'static, (usize, ContentBlock)>> = parallel_idx
                    .into_iter()
                    .map(|i| {
                        let tu = tool_uses[i].clone();
                        let exec = executor.clone();
                        let hk = hook.clone();
                        Box::pin(async move { (i, execute_one_tool(&tu, exec, hk).await) })
                            as BoxFuture<'static, (usize, ContentBlock)>
                    })
                    .collect();

                let results: Vec<(usize, ContentBlock)> = futures::stream::iter(futures)
                    .buffer_unordered(concurrency)
                    .collect()
                    .await;

                for (i, block) in results {
                    slots[i] = Some(block);
                }
            }

            let mut result_blocks: Vec<ContentBlock> = slots
                .into_iter()
                .map(|b| b.expect("every tool use produced a result"))
                .collect();

            // Tool-error rephrase hint: when any tool returned `is_error=true`,
            // append a guidance text block so the next iteration's provider
            // call sees an explicit "try with different inputs" cue instead
            // of looping on the same args. The previous behaviour was to
            // surface the error blocks verbatim and let the model figure it
            // out, which models often didn't — they'd re-call with identical
            // arguments. The hint catches this without changing the agent
            // loop's control flow.
            let failed: Vec<String> = result_blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult {
                        is_error: Some(true),
                        content,
                        ..
                    } => Some(content.clone()),
                    _ => None,
                })
                .collect();
            if !failed.is_empty() {
                let joined = failed
                    .iter()
                    .map(|c| {
                        // Cap each error message at 200 chars so a single
                        // huge stderr dump doesn't crowd the next prompt.
                        if c.len() > 200 {
                            format!("{}…", &c[..200])
                        } else {
                            c.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                result_blocks.push(ContentBlock::Text {
                    text: format!(
                        "[Tool error] {joined}. Reconsider the inputs — try a different \
                         path, argument, or approach instead of repeating the same call."
                    ),
                });
            }

            self.messages.push(Message {
                role: Role::User,
                content: MessageContent::Blocks(result_blocks),
                name: None,
                metadata: None,
            });

            // Keep the last text in case we hit max rounds
            final_text = text_buf;
        }

        Ok(final_text)
    }

    /// Collect the stream into accumulated text + tool uses.
    ///
    /// Returns `(text, tool_uses, response_id, compaction)`.
    /// `compaction` is `Some((summary, tokens_freed))` when the model emitted a
    /// `context_window_management_event` during the stream.  The caller is
    /// responsible for applying compaction to `self.messages` after the borrow
    /// on `self.messages` (held by the stream) is released.
    async fn collect_stream<F>(
        &mut self,
        tools_opt: Option<&[Tool]>,
        on_chunk: &Option<F>,
    ) -> Result<(
        String,
        Vec<ToolUse>,
        Option<String>,
        Option<(String, Option<u32>)>,
    )>
    where
        F: Fn(&str) + Send + Sync,
    {
        let mut stream = self
            .provider
            .stream_chat(&self.messages, tools_opt, &self.options);

        let mut text_buf = String::new();
        let mut tool_uses: Vec<ToolUse> = Vec::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::new();
        let mut last_response_id: Option<String> = None;
        let mut compaction: Option<(String, Option<u32>)> = None;

        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Text(t) => {
                    if let Some(cb) = on_chunk {
                        cb(&t);
                    }
                    text_buf.push_str(&t);
                }
                StreamChunk::ToolUse { id, name } => {
                    // Flush previous tool if any
                    if !current_tool_id.is_empty() {
                        let input: serde_json::Value = serde_json::from_str(&current_tool_input)
                            .unwrap_or(serde_json::Value::Null);
                        tool_uses.push(ToolUse {
                            id: std::mem::take(&mut current_tool_id),
                            name: std::mem::take(&mut current_tool_name),
                            input,
                        });
                        current_tool_input.clear();
                    }
                    current_tool_id = id;
                    current_tool_name = name;
                }
                StreamChunk::ToolInputDelta { partial_json, .. } => {
                    current_tool_input.push_str(&partial_json);
                }
                StreamChunk::ToolCall {
                    call_id,
                    response_id,
                    tool_name,
                    parameters,
                    ..
                } => {
                    last_response_id = Some(response_id);
                    tool_uses.push(ToolUse {
                        id: call_id,
                        name: tool_name,
                        input: parameters,
                    });
                }
                StreamChunk::Usage(u) => {
                    self.cumulative_usage.prompt_tokens += u.prompt_tokens;
                    self.cumulative_usage.completion_tokens += u.completion_tokens;
                    self.cumulative_usage.total_tokens += u.total_tokens;
                    if let Some(ref guard) = self.budget {
                        guard.record_usage(&u);
                    }
                }
                StreamChunk::Done => {}
                StreamChunk::ContextCompacted {
                    summary,
                    tokens_freed,
                } => {
                    // Record compaction info; applied to self.messages after the stream
                    // borrow is released (see run_completion).
                    compaction = Some((summary, tokens_freed));
                }
            }
        }

        // Flush last tool if any
        if !current_tool_id.is_empty() {
            let input: serde_json::Value =
                serde_json::from_str(&current_tool_input).unwrap_or(serde_json::Value::Null);
            tool_uses.push(ToolUse {
                id: current_tool_id,
                name: current_tool_name,
                input,
            });
        }

        Ok((text_buf, tool_uses, last_response_id, compaction))
    }
}

/// Execute a single tool call and return the resulting `ContentBlock`.
///
/// Consolidated out of the main loop so it can be driven both sequentially
/// (for `serialize: true` tools) and concurrently (for everything else) by
/// `run_completion`.
async fn execute_one_tool(
    tu: &ToolUse,
    executor: Arc<dyn ToolExecutor>,
    pre_execute_hook: Option<Arc<dyn ToolPreHook>>,
) -> ContentBlock {
    if let Some(ref hook) = pre_execute_hook {
        let ctx = ToolContext::default();
        match hook.before_execute(tu, &ctx).await {
            Ok(PreHookDecision::Allow) => {}
            Ok(PreHookDecision::Reject(reason)) => {
                return ContentBlock::ToolResult {
                    tool_use_id: tu.id.clone(),
                    content: reason,
                    is_error: Some(true),
                };
            }
            Err(e) => {
                tracing::warn!(tool = %tu.name, error = %e, "Pre-execute hook error");
            }
        }
    }

    let exec_ctx = ToolContext::default();
    let result = match executor.execute(tu, &exec_ctx).await {
        Ok(r) => r,
        Err(e) => {
            brainwires_core::ToolResult::error(tu.id.clone(), format!("tool executor error: {e}"))
        }
    };
    ContentBlock::ToolResult {
        tool_use_id: tu.id.clone(),
        content: result.content,
        is_error: Some(result.is_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{ToolContext, ToolInputSchema};
    use brainwires_test_fixtures::ScriptedProvider;
    use brainwires_tool_builtins::BuiltinToolExecutor;
    use brainwires_tool_runtime::ToolRegistry;
    use std::collections::HashMap;

    fn make_executor() -> Arc<dyn ToolExecutor> {
        let mut registry = ToolRegistry::new();
        registry.register(Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            ..Default::default()
        });
        let context = ToolContext::default();
        Arc::new(BuiltinToolExecutor::new(registry, context))
    }

    fn make_agent() -> ChatAgent {
        let provider = Arc::new(ScriptedProvider::always_text("mock", "Hello from mock!"));
        let executor = make_executor();
        ChatAgent::new(provider, executor, ChatOptions::default())
    }

    #[test]
    fn test_new_creates_successfully() {
        let agent = make_agent();
        assert_eq!(agent.message_count(), 0);
        assert_eq!(agent.max_tool_rounds, 10);
    }

    #[test]
    fn test_with_system_prompt_adds_system_message() {
        let agent = make_agent().with_system_prompt("You are helpful.");
        assert_eq!(agent.message_count(), 1);
        assert_eq!(agent.messages()[0].role, Role::System);
        assert_eq!(agent.messages()[0].text(), Some("You are helpful."));
    }

    #[test]
    fn test_with_system_prompt_replaces_existing() {
        let agent = make_agent()
            .with_system_prompt("First prompt")
            .with_system_prompt("Second prompt");
        assert_eq!(agent.message_count(), 1);
        assert_eq!(agent.messages()[0].text(), Some("Second prompt"));
    }

    #[test]
    fn test_with_max_tool_rounds() {
        let agent = make_agent().with_max_tool_rounds(5);
        assert_eq!(agent.max_tool_rounds, 5);
    }

    #[test]
    fn test_messages_returns_history() {
        let mut agent = make_agent();
        assert!(agent.messages().is_empty());
        // Manually push to test accessor
        agent.messages.push(Message::user("test"));
        assert_eq!(agent.messages().len(), 1);
    }

    #[test]
    fn test_clear_history() {
        let mut agent = make_agent().with_system_prompt("sys");
        agent.messages.push(Message::user("hello"));
        assert_eq!(agent.message_count(), 2);
        agent.clear_history();
        assert_eq!(agent.message_count(), 0);
    }

    #[test]
    fn test_trim_history_no_system() {
        let mut agent = make_agent();
        for i in 0..10 {
            agent.messages.push(Message::user(format!("msg {}", i)));
        }
        assert_eq!(agent.message_count(), 10);
        agent.trim_history(3);
        assert_eq!(agent.message_count(), 3);
        // Should keep the last 3
        assert_eq!(agent.messages()[0].text(), Some("msg 7"));
        assert_eq!(agent.messages()[1].text(), Some("msg 8"));
        assert_eq!(agent.messages()[2].text(), Some("msg 9"));
    }

    #[test]
    fn test_trim_history_preserves_system() {
        let mut agent = make_agent().with_system_prompt("system prompt");
        for i in 0..10 {
            agent.messages.push(Message::user(format!("msg {}", i)));
        }
        assert_eq!(agent.message_count(), 11); // 1 system + 10 user
        agent.trim_history(4);
        assert_eq!(agent.message_count(), 4);
        assert_eq!(agent.messages()[0].role, Role::System);
        assert_eq!(agent.messages()[0].text(), Some("system prompt"));
        // Last 3 user messages
        assert_eq!(agent.messages()[1].text(), Some("msg 7"));
        assert_eq!(agent.messages()[2].text(), Some("msg 8"));
        assert_eq!(agent.messages()[3].text(), Some("msg 9"));
    }

    #[test]
    fn test_trim_history_under_limit_is_noop() {
        let mut agent = make_agent();
        agent.messages.push(Message::user("only one"));
        agent.trim_history(10);
        assert_eq!(agent.message_count(), 1);
    }

    #[test]
    fn test_message_count() {
        let mut agent = make_agent();
        assert_eq!(agent.message_count(), 0);
        agent.messages.push(Message::user("a"));
        assert_eq!(agent.message_count(), 1);
        agent.messages.push(Message::assistant("b"));
        assert_eq!(agent.message_count(), 2);
    }

    #[tokio::test]
    async fn test_process_message_returns_text() {
        let mut agent = make_agent();
        let result = agent.process_message("Hi").await.unwrap();
        assert_eq!(result, "Hello from mock!");
        // Should have user message + assistant response
        assert_eq!(agent.message_count(), 2);
        assert_eq!(agent.messages()[0].role, Role::User);
        assert_eq!(agent.messages()[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_process_message_streaming() {
        let mut agent = make_agent();
        let chunks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let chunks_clone = chunks.clone();

        let result = agent
            .process_message_streaming("Hi", move |chunk| {
                chunks_clone.lock().unwrap().push(chunk.to_string());
            })
            .await
            .unwrap();

        assert_eq!(result, "Hello from mock!");
        let received = chunks.lock().unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0], "Hello from mock!");
    }
}
