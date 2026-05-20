//! `TaskAgent` struct and its complete `impl` block.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, ContentSource, IntendedWrites, Message,
    MessageContent, Provider, Role, Task, ToolContext, ToolResult, ToolUse,
    estimate_tokens_from_size,
};
use brainwires_tool_runtime::{PreHookDecision, wrap_with_content_source};

use crate::agent_hooks::{ConversationView, IterationContext, IterationDecision, ToolDecision};
use crate::context::AgentContext;
use crate::validation_loop::{format_validation_feedback, run_validation};
use brainwires_agent::communication::AgentMessage;
use brainwires_agent::execution_graph::{ExecutionGraph, RunTelemetry, ToolCallRecord};
use brainwires_agent::file_locks::LockType;

use super::types::{
    EXTERNAL_CONTENT_TOOLS, FailureCategory, TaskAgentConfig, TaskAgentResult, TaskAgentStatus,
};

/// Autonomous task agent that runs a provider + tool loop until completion.
///
/// Create with [`TaskAgent::new`], then call [`TaskAgent::execute`] (or spawn
/// it on a background task with [`super::spawn_task_agent`]).
pub struct TaskAgent {
    /// Unique agent ID.
    pub id: String,
    /// Task being executed (mutated as iterations progress).
    pub(super) task: Arc<RwLock<Task>>,
    /// AI provider for chat completions.
    pub(super) provider: Arc<dyn Provider>,
    /// Shared environment context.
    pub(super) context: Arc<AgentContext>,
    /// Agent configuration.
    pub(super) config: TaskAgentConfig,
    /// Current status (observable from outside the agent).
    pub(super) status: Arc<RwLock<TaskAgentStatus>>,
    /// Conversation history (internal — grows each iteration).
    pub(super) conversation_history: Arc<RwLock<Vec<Message>>>,
    /// Internal replan cycle counter.
    pub(super) replan_count: Arc<RwLock<u32>>,
}

impl TaskAgent {
    /// Create a new task agent.
    ///
    /// The agent starts in [`TaskAgentStatus::Idle`] and does not begin
    /// execution until [`execute`][Self::execute] is called.
    pub fn new(
        id: String,
        task: Task,
        provider: Arc<dyn Provider>,
        context: Arc<AgentContext>,
        config: TaskAgentConfig,
    ) -> Self {
        Self {
            id,
            task: Arc::new(RwLock::new(task)),
            provider,
            context,
            config,
            status: Arc::new(RwLock::new(TaskAgentStatus::Idle)),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            replan_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the agent's unique ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the current status.
    pub async fn status(&self) -> TaskAgentStatus {
        self.status.read().await.clone()
    }

    /// Get a snapshot of the task.
    pub async fn task(&self) -> Task {
        self.task.read().await.clone()
    }

    /// Get a read-only snapshot of the conversation history.
    pub async fn conversation_snapshot(&self) -> Vec<Message> {
        self.conversation_history.read().await.clone()
    }

    /// Get the current message count.
    pub async fn conversation_len(&self) -> usize {
        self.conversation_history.read().await.len()
    }

    /// Inject a message into the conversation (e.g., from a parent agent).
    pub async fn inject_message(&self, msg: Message) {
        self.conversation_history.write().await.push(msg);
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    pub(super) async fn set_status(&self, status: TaskAgentStatus) {
        *self.status.write().await = status.clone();
        let _ = self
            .context
            .communication_hub
            .broadcast(
                self.id.clone(),
                AgentMessage::StatusUpdate {
                    agent_id: self.id.clone(),
                    status: status.to_string(),
                    details: None,
                },
            )
            .await;
    }

    /// Returns `true` for tool names that operate on a specific file.
    fn is_file_operation(tool_name: &str) -> bool {
        matches!(
            tool_name,
            "read_file" | "write_file" | "edit_file" | "append_to_file" | "delete_file"
        )
    }

    /// Extract the file path from a tool use's input, if present.
    fn extract_file_path(tool_use: &ToolUse) -> Option<PathBuf> {
        let path_str = tool_use
            .input
            .get("file_path")
            .or_else(|| tool_use.input.get("path"))
            .and_then(|v| v.as_str())?;
        Some(PathBuf::from(path_str))
    }

    /// Returns `true` if `path` is permitted by the file scope whitelist.
    fn is_file_path_allowed(path: &str, allowed: &[PathBuf]) -> bool {
        if allowed.is_empty() {
            return false;
        }
        let candidate = PathBuf::from(path);
        allowed.iter().any(|prefix| candidate.starts_with(prefix))
    }

    /// Determine whether a tool requires a file lock, and what kind.
    fn get_lock_requirement(tool_use: &ToolUse) -> Option<(String, LockType)> {
        let path = tool_use
            .input
            .get("path")
            .or_else(|| tool_use.input.get("file_path"))
            .and_then(|v| v.as_str())?;

        let lock_type = match tool_use.name.as_str() {
            "read_file" | "list_directory" | "search_code" => LockType::Read,
            "write_file" | "edit_file" | "patch_file" | "delete_file" | "create_directory" => {
                LockType::Write
            }
            _ => return None,
        };
        Some((path.to_string(), lock_type))
    }

    /// Extract all `ToolUse` blocks from a provider message.
    fn extract_tool_uses(message: &Message) -> Vec<ToolUse> {
        match &message.content {
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Build the `Message` that wraps a tool result in the conversation.
    fn tool_result_message(result: &ToolResult) -> Message {
        Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: result.tool_use_id.clone(),
                content: result.content.clone(),
                is_error: Some(result.is_error),
            }]),
            name: None,
            metadata: None,
        }
    }

    /// Call the AI provider with the current conversation state.
    async fn call_provider(&self) -> Result<ChatResponse> {
        let history = self.conversation_history.read().await.clone();
        let tools = self.context.tool_executor.available_tools();

        let system_prompt = self.config.system_prompt.clone().unwrap_or_else(|| {
            crate::system_prompts::reasoning_agent_prompt(&self.id, &self.context.working_directory)
        });

        let options = ChatOptions {
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
            top_p: None,
            stop: None,
            system: Some(system_prompt),
            model: None,
            cache_strategy: Default::default(),
        };

        self.provider.chat(&history, Some(&tools), &options).await
    }

    /// Run validation checks and, if they pass, finalise the task.
    ///
    /// Returns `Some(result)` when the agent should stop (validation passed),
    /// `None` when validation failed and the loop should continue so the agent
    /// can self-correct.
    // reason: too_many_arguments — refactoring this orchestration function would
    // require introducing a struct just to bundle args, which obscures the
    // call site. The arguments are all distinct semantically.
    #[allow(clippy::too_many_arguments)]
    async fn attempt_validated_completion(
        &self,
        message_text: &str,
        total_tokens_used: u64,
        total_cost_usd: f64,
        replan_count: u32,
        execution_graph: ExecutionGraph,
        pre_execution_plan: Option<brainwires_core::SerializablePlan>,
        intended_writes: &IntendedWrites,
    ) -> Result<Option<TaskAgentResult>> {
        let task_id = self.task.read().await.id.clone();

        if let Some(ref validation_config) = self.config.validation_config
            && validation_config.enabled
        {
            tracing::info!(
                agent_id = %self.id,
                "running validation before completion"
            );

            let working_set_files = {
                let ws = self.context.working_set.read().await;
                ws.file_paths()
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
            };

            let mut config_with_ws = validation_config.clone();
            config_with_ws.working_set_files = working_set_files;
            // Wire in the shared intended-writes registry so validation can
            // detect post-validation clobber (see validation_loop.rs).
            if config_with_ws.intended_writes.is_none() {
                config_with_ws.intended_writes = Some(intended_writes.clone());
            }

            // Mirror intended-writes into the agent's WorkingSet so external
            // observers (hooks, persona reporters) can inspect the per-file
            // hash history alongside existence/token state.
            {
                let mut ws = self.context.working_set.write().await;
                for (path, hash) in intended_writes.snapshot() {
                    ws.record_write(&path, hash);
                }
            }

            match run_validation(&config_with_ws).await {
                Ok(result) if !result.passed => {
                    tracing::warn!(
                        agent_id = %self.id,
                        issues = result.issues.len(),
                        "validation failed, continuing loop"
                    );
                    let feedback = format_validation_feedback(&result);
                    self.conversation_history
                        .write()
                        .await
                        .push(Message::user(feedback));
                    return Ok(None);
                }
                Ok(_) => {
                    tracing::info!(agent_id = %self.id, "validation passed");
                }
                Err(e) => {
                    // Validation infrastructure error — proceed anyway.
                    tracing::error!(agent_id = %self.id, "validation error: {}", e);
                }
            }
        }

        // Finalise the task.
        self.task.write().await.complete(message_text);
        self.set_status(TaskAgentStatus::Completed(message_text.to_string()))
            .await;

        let _ = self
            .context
            .communication_hub
            .broadcast(
                self.id.clone(),
                AgentMessage::TaskResult {
                    task_id: task_id.clone(),
                    success: true,
                    result: message_text.to_string(),
                },
            )
            .await;

        let _ = self
            .context
            .communication_hub
            .unregister_agent(&self.id)
            .await;
        self.context
            .file_lock_manager
            .release_all_locks(&self.id)
            .await;

        let iterations = self.task.read().await.iterations;
        let run_ended_at = Utc::now();
        let telemetry =
            RunTelemetry::from_graph(&execution_graph, run_ended_at, true, total_cost_usd);

        Ok(Some(TaskAgentResult {
            agent_id: self.id.clone(),
            task_id,
            success: true,
            summary: message_text.to_string(),
            iterations,
            replan_count,
            budget_exhausted: false,
            partial_output: None,
            total_tokens_used,
            total_cost_usd,
            timed_out: false,
            failure_category: None,
            execution_graph,
            telemetry,
            pre_execution_plan,
        }))
    }

    // ── Public execution entry point ─────────────────────────────────────────

    /// Execute the task to completion, returning the result.
    ///
    /// Blocks the calling async task until the agent finishes. Use
    /// [`super::spawn_task_agent`] to run the agent on a Tokio background task.
    pub async fn execute(&self) -> Result<TaskAgentResult> {
        let result = self.execute_impl().await?;
        self.maybe_emit_run_analytics(&result);
        Ok(result)
    }

    async fn execute_impl(&self) -> Result<TaskAgentResult> {
        let task_id = self.task.read().await.id.clone();
        let task_description = self.task.read().await.description.clone();

        tracing::info!(
            agent_id = %self.id,
            task_id = %task_id,
            "TaskAgent starting execution"
        );

        // Register with the communication hub.
        if !self.context.communication_hub.is_registered(&self.id).await {
            self.context
                .communication_hub
                .register_agent(self.id.clone())
                .await?;
        }

        self.task.write().await.start();
        self.set_status(TaskAgentStatus::Working(task_description.clone()))
            .await;

        // Seed conversation with the task description as the first user message.
        self.conversation_history
            .write()
            .await
            .push(Message::user(task_description.clone()));

        // ── Prompt hash + execution graph initialisation ─────────────────────
        let prompt_hash = {
            let system_prompt = self.config.system_prompt.clone().unwrap_or_else(|| {
                crate::system_prompts::reasoning_agent_prompt(
                    &self.id,
                    &self.context.working_directory,
                )
            });
            let mut tool_names: Vec<String> = self
                .context
                .tool_executor
                .available_tools()
                .iter()
                .map(|t| t.name.clone())
                .collect();
            tool_names.sort_unstable();
            let mut hasher = Sha256::new();
            hasher.update(system_prompt.as_bytes());
            for name in &tool_names {
                hasher.update(name.as_bytes());
            }
            hex::encode(hasher.finalize())
        };
        let run_started_at = Utc::now();
        let mut execution_graph = ExecutionGraph::new(prompt_hash, run_started_at);

        // ── Pre-execution planning phase ─────────────────────────────────────
        // When plan_budget is set, ask the model for a structured JSON plan and
        // validate it against the budget before any side effects occur.
        let mut pre_execution_plan: Option<brainwires_core::SerializablePlan> = None;
        if let Some(ref budget) = self.config.plan_budget {
            let planning_msg = Message::user(format!(
                "Before beginning work, produce a JSON execution plan for this task.\n\n\
                 Task: {task_description}\n\n\
                 Reply with ONLY a JSON object in this exact format:\n\
                 {{\"steps\":[{{\"description\":\"short description\",\"tool\":\"tool_name\",\"estimated_tokens\":500}},...]}}\n\n\
                 Estimate 200–2000 tokens per step based on expected complexity. \
                 Do not perform any work yet — only plan.",
            ));
            let planning_options = brainwires_core::ChatOptions {
                temperature: Some(0.1),
                max_tokens: Some(2048),
                top_p: None,
                stop: None,
                system: Some(
                    "You are a planning assistant. Respond only with a valid JSON execution plan."
                        .to_string(),
                ),
                model: None,
                cache_strategy: Default::default(),
            };
            match self
                .provider
                .chat(&[planning_msg], None, &planning_options)
                .await
            {
                Ok(response) => {
                    let plan_text = response.message.text().unwrap_or("").to_string();
                    if let Some(plan) = brainwires_core::SerializablePlan::parse_from_text(
                        task_description.clone(),
                        &plan_text,
                    ) {
                        match budget.check(&plan) {
                            Ok(()) => {
                                tracing::info!(
                                    agent_id = %self.id,
                                    steps = plan.step_count(),
                                    estimated_tokens = plan.total_estimated_tokens(),
                                    "pre-execution plan accepted"
                                );
                                pre_execution_plan = Some(plan);
                            }
                            Err(reason) => {
                                let error = format!(
                                    "Agent {} rejected by plan budget before execution: {}",
                                    self.id, reason
                                );
                                tracing::error!(agent_id = %self.id, %error);
                                self.task.write().await.fail(&error);
                                self.set_status(TaskAgentStatus::Failed(error.clone()))
                                    .await;
                                let _ = self
                                    .context
                                    .communication_hub
                                    .broadcast(
                                        self.id.clone(),
                                        AgentMessage::TaskResult {
                                            task_id: task_id.clone(),
                                            success: false,
                                            result: error.clone(),
                                        },
                                    )
                                    .await;
                                let _ = self
                                    .context
                                    .communication_hub
                                    .unregister_agent(&self.id)
                                    .await;
                                self.context
                                    .file_lock_manager
                                    .release_all_locks(&self.id)
                                    .await;
                                let run_ended_at = Utc::now();
                                let telemetry = RunTelemetry::from_graph(
                                    &execution_graph,
                                    run_ended_at,
                                    false,
                                    0.0,
                                );
                                return Ok(TaskAgentResult {
                                    agent_id: self.id.clone(),
                                    task_id,
                                    success: false,
                                    summary: error,
                                    iterations: 0,
                                    replan_count: 0,
                                    budget_exhausted: true,
                                    partial_output: None,
                                    total_tokens_used: 0,
                                    total_cost_usd: 0.0,
                                    timed_out: false,
                                    failure_category: Some(FailureCategory::PlanBudgetExceeded),
                                    execution_graph,
                                    telemetry,
                                    pre_execution_plan: None,
                                });
                            }
                        }
                    } else {
                        tracing::warn!(
                            agent_id = %self.id,
                            "could not parse pre-execution plan from model response; \
                             proceeding without budget guard"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        agent_id = %self.id,
                        error = %e,
                        "planning phase provider call failed; proceeding without plan"
                    );
                }
            }
        }

        let mut iterations = 0u32;
        let mut total_tokens_used: u64 = 0;
        let mut total_cost_usd: f64 = 0.0;
        const COST_PER_TOKEN: f64 = 0.000003; // $3/M tokens conservative estimate
        let start_time = std::time::Instant::now();
        let mut recent_tool_names: VecDeque<String> = VecDeque::with_capacity(
            self.config
                .loop_detection
                .as_ref()
                .map(|c| c.window_size)
                .unwrap_or(5),
        );
        // Shared intended-writes registry for this run.  The tool layer
        // (write_file) records the SHA-256 of each write; the validation
        // loop re-reads at finalisation and compares to detect
        // post-validation clobber by a concurrent writer.
        let intended_writes = IntendedWrites::new();
        let tool_context = ToolContext {
            working_directory: self.context.working_directory.clone(),
            // Each agent run gets its own idempotency registry so that
            // identical write operations within a single run are deduplicated.
            idempotency_registry: Some(brainwires_core::IdempotencyRegistry::new()),
            intended_writes: Some(intended_writes.clone()),
            ..Default::default()
        };

        loop {
            iterations += 1;
            self.task.write().await.increment_iteration();

            tracing::debug!(
                agent_id = %self.id,
                iteration = iterations,
                max = self.config.max_iterations,
                "iteration starting"
            );

            let step_started_at = Utc::now();
            let step_idx = execution_graph.push_step(iterations, step_started_at);

            // ── Hook A: on_before_iteration ──────────────────────────────────
            if let Some(ref hooks) = self.context.lifecycle_hooks {
                let conv_len = self.conversation_history.read().await.len();
                let iter_ctx = self.build_iteration_context(
                    iterations,
                    total_tokens_used,
                    total_cost_usd,
                    &start_time,
                    conv_len,
                );
                let mut history = self.conversation_history.write().await;
                let mut view = ConversationView::new(&mut history);
                match hooks.on_before_iteration(&iter_ctx, &mut view).await {
                    IterationDecision::Continue => {}
                    IterationDecision::Skip => {
                        drop(history);
                        continue;
                    }
                    IterationDecision::Abort(reason) => {
                        drop(history);
                        let error = format!("Agent {} aborted by hook: {}", self.id, reason);
                        tracing::error!(agent_id = %self.id, %error);
                        self.task.write().await.fail(&error);
                        self.set_status(TaskAgentStatus::Failed(error.clone()))
                            .await;
                        let _ = self
                            .context
                            .communication_hub
                            .broadcast(
                                self.id.clone(),
                                AgentMessage::TaskResult {
                                    task_id: task_id.clone(),
                                    success: false,
                                    result: error.clone(),
                                },
                            )
                            .await;
                        let _ = self
                            .context
                            .communication_hub
                            .unregister_agent(&self.id)
                            .await;
                        self.context
                            .file_lock_manager
                            .release_all_locks(&self.id)
                            .await;
                        let run_ended_at = Utc::now();
                        let telemetry = RunTelemetry::from_graph(
                            &execution_graph,
                            run_ended_at,
                            false,
                            total_cost_usd,
                        );
                        return Ok(TaskAgentResult {
                            agent_id: self.id.clone(),
                            task_id,
                            success: false,
                            summary: error,
                            iterations,
                            replan_count: *self.replan_count.read().await,
                            budget_exhausted: false,
                            partial_output: None,
                            total_tokens_used,
                            total_cost_usd,
                            timed_out: false,
                            failure_category: Some(FailureCategory::Unknown),
                            execution_graph: execution_graph.clone(),
                            telemetry,
                            pre_execution_plan: pre_execution_plan.clone(),
                        });
                    }
                }
            }

            // ── Iteration limit ──────────────────────────────────────────────
            if iterations > self.config.max_iterations {
                let error = format!(
                    "Agent {} exceeded maximum iterations ({})",
                    self.id, self.config.max_iterations
                );
                tracing::error!(agent_id = %self.id, %error);

                self.task.write().await.fail(&error);
                self.set_status(TaskAgentStatus::Failed(error.clone()))
                    .await;

                let _ = self
                    .context
                    .communication_hub
                    .broadcast(
                        self.id.clone(),
                        AgentMessage::TaskResult {
                            task_id: task_id.clone(),
                            success: false,
                            result: error.clone(),
                        },
                    )
                    .await;

                let _ = self
                    .context
                    .communication_hub
                    .unregister_agent(&self.id)
                    .await;
                self.context
                    .file_lock_manager
                    .release_all_locks(&self.id)
                    .await;

                let run_ended_at = Utc::now();
                let telemetry =
                    RunTelemetry::from_graph(&execution_graph, run_ended_at, false, total_cost_usd);
                return Ok(TaskAgentResult {
                    agent_id: self.id.clone(),
                    task_id,
                    success: false,
                    summary: error,
                    iterations,
                    replan_count: *self.replan_count.read().await,
                    budget_exhausted: false,
                    partial_output: None,
                    total_tokens_used,
                    total_cost_usd,
                    timed_out: false,
                    failure_category: Some(FailureCategory::IterationLimitExceeded),
                    execution_graph: execution_graph.clone(),
                    telemetry,
                    pre_execution_plan: pre_execution_plan.clone(),
                });
            }

            // ── Incoming messages (non-blocking) ────────────────────────────
            if let Some(envelope) = self
                .context
                .communication_hub
                .try_receive_message(&self.id)
                .await
                && let AgentMessage::HelpResponse {
                    request_id,
                    response,
                } = envelope.message
            {
                self.conversation_history
                    .write()
                    .await
                    .push(Message::user(format!(
                        "Response to help request {}: {}",
                        request_id, response
                    )));
            }

            // ── Budget: timeout ──────────────────────────────────────────────
            if let Some(secs) = self.config.timeout_secs
                && start_time.elapsed().as_secs() >= secs
            {
                let elapsed = start_time.elapsed().as_secs();
                let partial = self.last_assistant_text().await;
                let error = format!(
                    "Agent {} timed out after {}s (limit: {}s)",
                    self.id, elapsed, secs
                );
                tracing::error!(agent_id = %self.id, %error);
                self.task.write().await.fail(&error);
                self.set_status(TaskAgentStatus::Failed(error.clone()))
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .broadcast(
                        self.id.clone(),
                        AgentMessage::TaskResult {
                            task_id: task_id.clone(),
                            success: false,
                            result: error.clone(),
                        },
                    )
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .unregister_agent(&self.id)
                    .await;
                self.context
                    .file_lock_manager
                    .release_all_locks(&self.id)
                    .await;
                let run_ended_at = Utc::now();
                let telemetry =
                    RunTelemetry::from_graph(&execution_graph, run_ended_at, false, total_cost_usd);
                return Ok(TaskAgentResult {
                    agent_id: self.id.clone(),
                    task_id,
                    success: false,
                    summary: error,
                    iterations,
                    replan_count: *self.replan_count.read().await,
                    budget_exhausted: false,
                    partial_output: partial,
                    total_tokens_used,
                    total_cost_usd,
                    timed_out: true,
                    failure_category: Some(FailureCategory::WallClockTimeout),
                    execution_graph: execution_graph.clone(),
                    telemetry,
                    pre_execution_plan: pre_execution_plan.clone(),
                });
            }

            // ── Budget: token ceiling ────────────────────────────────────────
            if let Some(max) = self.config.max_total_tokens
                && total_tokens_used >= max
            {
                let partial = self.last_assistant_text().await;
                let error = format!(
                    "Agent {} exceeded token budget ({}/{} tokens)",
                    self.id, total_tokens_used, max
                );
                tracing::error!(agent_id = %self.id, %error);
                self.task.write().await.fail(&error);
                self.set_status(TaskAgentStatus::Failed(error.clone()))
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .broadcast(
                        self.id.clone(),
                        AgentMessage::TaskResult {
                            task_id: task_id.clone(),
                            success: false,
                            result: error.clone(),
                        },
                    )
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .unregister_agent(&self.id)
                    .await;
                self.context
                    .file_lock_manager
                    .release_all_locks(&self.id)
                    .await;
                let run_ended_at = Utc::now();
                let telemetry =
                    RunTelemetry::from_graph(&execution_graph, run_ended_at, false, total_cost_usd);
                return Ok(TaskAgentResult {
                    agent_id: self.id.clone(),
                    task_id,
                    success: false,
                    summary: error,
                    iterations,
                    replan_count: *self.replan_count.read().await,
                    budget_exhausted: true,
                    partial_output: partial,
                    total_tokens_used,
                    total_cost_usd,
                    timed_out: false,
                    failure_category: Some(FailureCategory::TokenBudgetExceeded),
                    execution_graph: execution_graph.clone(),
                    telemetry,
                    pre_execution_plan: pre_execution_plan.clone(),
                });
            }

            // ── Budget: cost ceiling ─────────────────────────────────────────
            if let Some(max) = self.config.max_cost_usd
                && total_cost_usd >= max
            {
                let partial = self.last_assistant_text().await;
                let error = format!(
                    "Agent {} exceeded cost budget (${:.6}/{:.6} USD)",
                    self.id, total_cost_usd, max
                );
                tracing::error!(agent_id = %self.id, %error);
                self.task.write().await.fail(&error);
                self.set_status(TaskAgentStatus::Failed(error.clone()))
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .broadcast(
                        self.id.clone(),
                        AgentMessage::TaskResult {
                            task_id: task_id.clone(),
                            success: false,
                            result: error.clone(),
                        },
                    )
                    .await;
                let _ = self
                    .context
                    .communication_hub
                    .unregister_agent(&self.id)
                    .await;
                self.context
                    .file_lock_manager
                    .release_all_locks(&self.id)
                    .await;
                let run_ended_at = Utc::now();
                let telemetry =
                    RunTelemetry::from_graph(&execution_graph, run_ended_at, false, total_cost_usd);
                return Ok(TaskAgentResult {
                    agent_id: self.id.clone(),
                    task_id,
                    success: false,
                    summary: error,
                    iterations,
                    replan_count: *self.replan_count.read().await,
                    budget_exhausted: true,
                    partial_output: partial,
                    total_tokens_used,
                    total_cost_usd,
                    timed_out: false,
                    failure_category: Some(FailureCategory::CostBudgetExceeded),
                    execution_graph: execution_graph.clone(),
                    telemetry,
                    pre_execution_plan: pre_execution_plan.clone(),
                });
            }

            // ── Goal re-validation ───────────────────────────────────────────
            if let Some(interval) = self.config.goal_revalidation_interval
                && interval > 0
                && iterations > 1
                && (iterations - 1).is_multiple_of(interval)
            {
                self.conversation_history
                    .write()
                    .await
                    .push(Message::user(format!(
                        "GOAL CHECK (iteration {}): Your original task was:\n\n\"{}\"\n\n\
                         Confirm you are still on track. Correct course if you have drifted.",
                        iterations, task_description
                    )));
            }

            // ── Hook B: on_before_provider_call ──────────────────────────────
            if let Some(ref hooks) = self.context.lifecycle_hooks {
                let conv_len = self.conversation_history.read().await.len();
                let iter_ctx = self.build_iteration_context(
                    iterations,
                    total_tokens_used,
                    total_cost_usd,
                    &start_time,
                    conv_len,
                );
                let mut history = self.conversation_history.write().await;
                let mut view = ConversationView::new(&mut history);
                hooks.on_before_provider_call(&iter_ctx, &mut view).await;
            }

            // ── Call provider ───────────────────────────────────────────────
            let response = self.call_provider().await?;

            // ── Accumulate token usage ───────────────────────────────────────
            total_tokens_used += response.usage.total_tokens as u64;
            let call_cost = response.usage.total_tokens as f64 * COST_PER_TOKEN;
            total_cost_usd += call_cost;

            // ── Emit billing UsageEvent::Tokens ─────────────────────────────
            #[cfg(feature = "telemetry")]
            if let Some(ref hook) = self.config.billing_hook {
                let event = brainwires_telemetry::UsageEvent::tokens(
                    self.id.clone(),
                    self.config.system_prompt.as_deref().unwrap_or("unknown"),
                    response.usage.total_tokens as u64,
                    call_cost,
                );
                if let Err(e) = hook.0.on_usage(&event).await {
                    tracing::warn!(agent_id = %self.id, error = %e, "billing hook error (tokens)");
                }
            }

            // ── Hook C: on_after_provider_call ───────────────────────────────
            if let Some(ref hooks) = self.context.lifecycle_hooks {
                let conv_len = self.conversation_history.read().await.len();
                let iter_ctx = self.build_iteration_context(
                    iterations,
                    total_tokens_used,
                    total_cost_usd,
                    &start_time,
                    conv_len,
                );
                hooks.on_after_provider_call(&iter_ctx, &response).await;
            }

            // ── Finalise step node ───────────────────────────────────────────
            execution_graph.finalize_step(
                step_idx,
                Utc::now(),
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.finish_reason.clone(),
            );

            // ── REPLAN detection ─────────────────────────────────────────────
            {
                let text = response.message.text().unwrap_or("").to_lowercase();
                if text.contains("replan") || text.contains("replanning") {
                    let mut count = self.replan_count.write().await;
                    *count += 1;
                    let c = *count;
                    drop(count);
                    self.set_status(TaskAgentStatus::Replanning(format!(
                        "attempt {}/{}",
                        c, self.config.max_replan_attempts
                    )))
                    .await;
                    if c > self.config.max_replan_attempts {
                        let error = format!(
                            "Agent {} exceeded max replan attempts ({}/{})",
                            self.id, c, self.config.max_replan_attempts
                        );
                        tracing::error!(agent_id = %self.id, %error);
                        self.task.write().await.fail(&error);
                        self.set_status(TaskAgentStatus::Failed(error.clone()))
                            .await;
                        let _ = self
                            .context
                            .communication_hub
                            .broadcast(
                                self.id.clone(),
                                AgentMessage::TaskResult {
                                    task_id: task_id.clone(),
                                    success: false,
                                    result: error.clone(),
                                },
                            )
                            .await;
                        let _ = self
                            .context
                            .communication_hub
                            .unregister_agent(&self.id)
                            .await;
                        self.context
                            .file_lock_manager
                            .release_all_locks(&self.id)
                            .await;
                        let run_ended_at = Utc::now();
                        let telemetry = RunTelemetry::from_graph(
                            &execution_graph,
                            run_ended_at,
                            false,
                            total_cost_usd,
                        );
                        return Ok(TaskAgentResult {
                            agent_id: self.id.clone(),
                            task_id,
                            success: false,
                            summary: error,
                            iterations,
                            replan_count: c,
                            budget_exhausted: false,
                            partial_output: None,
                            total_tokens_used,
                            total_cost_usd,
                            timed_out: false,
                            failure_category: Some(FailureCategory::MaxReplanAttemptsExceeded),
                            execution_graph: execution_graph.clone(),
                            telemetry,
                            pre_execution_plan: pre_execution_plan.clone(),
                        });
                    }
                }
            }

            let is_done = response
                .finish_reason
                .as_deref()
                .is_some_and(|r| r == "end_turn" || r == "stop");

            // ── Completion path ─────────────────────────────────────────────
            if is_done {
                let text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();

                // ── Hook F: on_before_completion ─────────────────────────────
                if let Some(ref hooks) = self.context.lifecycle_hooks {
                    let conv_len = self.conversation_history.read().await.len();
                    let iter_ctx = self.build_iteration_context(
                        iterations,
                        total_tokens_used,
                        total_cost_usd,
                        &start_time,
                        conv_len,
                    );
                    if !hooks.on_before_completion(&iter_ctx, &text).await {
                        continue; // Hook rejected completion
                    }
                }

                if let Some(result) = self
                    .attempt_validated_completion(
                        &text,
                        total_tokens_used,
                        total_cost_usd,
                        *self.replan_count.read().await,
                        execution_graph.clone(),
                        pre_execution_plan.clone(),
                        &intended_writes,
                    )
                    .await?
                {
                    // ── Hook: on_after_completion ────────────────────────────
                    if let Some(ref hooks) = self.context.lifecycle_hooks {
                        let conv_len = self.conversation_history.read().await.len();
                        let iter_ctx = self.build_iteration_context(
                            iterations,
                            total_tokens_used,
                            total_cost_usd,
                            &start_time,
                            conv_len,
                        );
                        hooks.on_after_completion(&iter_ctx, &result).await;
                    }
                    return Ok(result);
                }
                continue; // Validation failed — let the agent self-correct.
            }

            // ── Tool execution path ─────────────────────────────────────────
            let tool_uses = Self::extract_tool_uses(&response.message);

            if tool_uses.is_empty() {
                // No tools and no explicit completion signal — treat as done.
                let text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();

                // ── Hook F: on_before_completion (implicit) ──────────────────
                if let Some(ref hooks) = self.context.lifecycle_hooks {
                    let conv_len = self.conversation_history.read().await.len();
                    let iter_ctx = self.build_iteration_context(
                        iterations,
                        total_tokens_used,
                        total_cost_usd,
                        &start_time,
                        conv_len,
                    );
                    if !hooks.on_before_completion(&iter_ctx, &text).await {
                        continue; // Hook rejected completion
                    }
                }

                if let Some(result) = self
                    .attempt_validated_completion(
                        &text,
                        total_tokens_used,
                        total_cost_usd,
                        *self.replan_count.read().await,
                        execution_graph.clone(),
                        pre_execution_plan.clone(),
                        &intended_writes,
                    )
                    .await?
                {
                    // ── Hook: on_after_completion ────────────────────────────
                    if let Some(ref hooks) = self.context.lifecycle_hooks {
                        let conv_len = self.conversation_history.read().await.len();
                        let iter_ctx = self.build_iteration_context(
                            iterations,
                            total_tokens_used,
                            total_cost_usd,
                            &start_time,
                            conv_len,
                        );
                        hooks.on_after_completion(&iter_ctx, &result).await;
                    }
                    return Ok(result);
                }
                continue;
            }

            // Record the assistant's tool-use message in conversation history.
            self.conversation_history
                .write()
                .await
                .push(response.message.clone());

            for tool_use in &tool_uses {
                tracing::debug!(
                    agent_id = %self.id,
                    tool = %tool_use.name,
                    "executing tool"
                );

                // ── Hook D: on_before_tool_execution ─────────────────────────
                if let Some(ref hooks) = self.context.lifecycle_hooks {
                    let conv_len = self.conversation_history.read().await.len();
                    let iter_ctx = self.build_iteration_context(
                        iterations,
                        total_tokens_used,
                        total_cost_usd,
                        &start_time,
                        conv_len,
                    );
                    match hooks.on_before_tool_execution(&iter_ctx, tool_use).await {
                        ToolDecision::Execute => {} // proceed normally
                        ToolDecision::Override(result) => {
                            execution_graph.record_tool_call(
                                step_idx,
                                ToolCallRecord {
                                    tool_use_id: tool_use.id.clone(),
                                    tool_name: tool_use.name.clone(),
                                    is_error: result.is_error,
                                    executed_at: Utc::now(),
                                },
                            );
                            self.conversation_history
                                .write()
                                .await
                                .push(Self::tool_result_message(&result));
                            continue;
                        }
                        ToolDecision::Delegate(request) => {
                            match hooks.execute_delegation(&request).await {
                                Ok(delegation_result) => {
                                    let tool_result = ToolResult::success(
                                        tool_use.id.clone(),
                                        format!(
                                            "Delegated to sub-agent {}: {}",
                                            delegation_result.agent_id, delegation_result.output
                                        ),
                                    );
                                    execution_graph.record_tool_call(
                                        step_idx,
                                        ToolCallRecord {
                                            tool_use_id: tool_use.id.clone(),
                                            tool_name: tool_use.name.clone(),
                                            is_error: !delegation_result.success,
                                            executed_at: Utc::now(),
                                        },
                                    );
                                    self.conversation_history
                                        .write()
                                        .await
                                        .push(Self::tool_result_message(&tool_result));
                                }
                                Err(e) => {
                                    let tool_result = ToolResult::error(
                                        tool_use.id.clone(),
                                        format!("Delegation failed: {}", e),
                                    );
                                    execution_graph.record_tool_call(
                                        step_idx,
                                        ToolCallRecord {
                                            tool_use_id: tool_use.id.clone(),
                                            tool_name: tool_use.name.clone(),
                                            is_error: true,
                                            executed_at: Utc::now(),
                                        },
                                    );
                                    self.conversation_history
                                        .write()
                                        .await
                                        .push(Self::tool_result_message(&tool_result));
                                }
                            }
                            continue;
                        }
                    }
                }

                // ── Billing authorize hook (fail-closed) ─────────────────────
                // Ask the billing hook to authorize the pending tool call
                // before we dispatch it. When the hook enforces a budget and
                // returns `BudgetExhausted`, we reject the call outright
                // (unlike the fail-open `on_usage` advisory path below).
                #[cfg(feature = "telemetry")]
                if let Some(ref hook) = self.config.billing_hook {
                    let pending = brainwires_telemetry::UsageEvent::tool_call(
                        self.id.clone(),
                        tool_use.name.clone(),
                    );
                    if let Err(e) = hook.0.authorize(&pending).await {
                        tracing::warn!(
                            agent_id = %self.id,
                            tool = %tool_use.name,
                            error = %e,
                            "tool call rejected by billing authorize()"
                        );
                        execution_graph.record_tool_call(
                            step_idx,
                            ToolCallRecord {
                                tool_use_id: tool_use.id.clone(),
                                tool_name: tool_use.name.clone(),
                                is_error: true,
                                executed_at: Utc::now(),
                            },
                        );
                        let rejection = ToolResult::error(tool_use.id.clone(), e.to_string());
                        self.conversation_history
                            .write()
                            .await
                            .push(Self::tool_result_message(&rejection));
                        continue;
                    }
                }

                // ── Pre-execute hook ─────────────────────────────────────────
                if let Some(ref hook) = self.context.pre_execute_hook {
                    match hook.before_execute(tool_use, &tool_context).await {
                        Ok(PreHookDecision::Reject(reason)) => {
                            tracing::warn!(
                                agent_id = %self.id,
                                tool = %tool_use.name,
                                reason = %reason,
                                "tool call rejected by pre-execute hook"
                            );
                            execution_graph.record_tool_call(
                                step_idx,
                                ToolCallRecord {
                                    tool_use_id: tool_use.id.clone(),
                                    tool_name: tool_use.name.clone(),
                                    is_error: true,
                                    executed_at: Utc::now(),
                                },
                            );
                            let rejection = ToolResult::error(tool_use.id.clone(), reason);
                            self.conversation_history
                                .write()
                                .await
                                .push(Self::tool_result_message(&rejection));
                            continue;
                        }
                        Ok(PreHookDecision::Allow) => {}
                        Err(e) => {
                            tracing::error!(
                                agent_id = %self.id,
                                "pre-execute hook error: {}",
                                e
                            );
                        }
                    }
                }

                // ── File scope check (all tools with path params) ────────
                if let Some(ref allowed) = self.config.allowed_files {
                    let candidate_path = tool_use
                        .input
                        .get("path")
                        .or_else(|| tool_use.input.get("file_path"))
                        .and_then(|v| v.as_str());
                    if let Some(p) = candidate_path
                        && !Self::is_file_path_allowed(p, allowed)
                    {
                        tracing::warn!(
                            agent_id = %self.id,
                            path = %p,
                            tool = %tool_use.name,
                            "file scope violation (tool not in lock list)"
                        );
                        let result = ToolResult::error(
                            tool_use.id.clone(),
                            format!(
                                "File scope violation: '{}' is outside allowed paths: {:?}",
                                p, allowed
                            ),
                        );
                        self.conversation_history
                            .write()
                            .await
                            .push(Self::tool_result_message(&result));
                        continue;
                    }
                }

                let _tool_exec_start = std::time::Instant::now();
                let tool_result =
                    if let Some((path, lock_type)) = Self::get_lock_requirement(tool_use) {
                        // ── File scope whitelist check (lock-requiring tools) ─
                        if let Some(ref allowed) = self.config.allowed_files
                            && !Self::is_file_path_allowed(&path, allowed)
                        {
                            tracing::warn!(
                                agent_id = %self.id,
                                path = %path,
                                "file scope violation"
                            );
                            let result = ToolResult::error(
                                tool_use.id.clone(),
                                format!(
                                    "File scope violation: '{}' is outside allowed paths: {:?}",
                                    path, allowed
                                ),
                            );
                            self.conversation_history
                                .write()
                                .await
                                .push(Self::tool_result_message(&result));
                            continue;
                        }

                        self.set_status(TaskAgentStatus::WaitingForLock(path.clone()))
                            .await;

                        match self
                            .context
                            .file_lock_manager
                            .acquire_lock(&self.id, &path, lock_type)
                            .await
                        {
                            Ok(_guard) => {
                                self.set_status(TaskAgentStatus::Working(format!(
                                    "Executing {}",
                                    tool_use.name
                                )))
                                .await;
                                match self
                                    .context
                                    .tool_executor
                                    .execute(tool_use, &tool_context)
                                    .await
                                {
                                    Ok(r) => r,
                                    Err(e) => ToolResult::error(
                                        tool_use.id.clone(),
                                        format!("Tool execution failed: {}", e),
                                    ),
                                }
                                // _guard dropped here — lock released.
                            }
                            Err(e) => {
                                tracing::warn!(
                                    agent_id = %self.id,
                                    path = %path,
                                    "failed to acquire lock: {}",
                                    e
                                );
                                ToolResult::error(
                                    tool_use.id.clone(),
                                    format!("Could not acquire file lock: {}", e),
                                )
                            }
                        }
                    } else {
                        self.set_status(TaskAgentStatus::Working(format!(
                            "Executing {}",
                            tool_use.name
                        )))
                        .await;
                        match self
                            .context
                            .tool_executor
                            .execute(tool_use, &tool_context)
                            .await
                        {
                            Ok(r) => r,
                            Err(e) => ToolResult::error(
                                tool_use.id.clone(),
                                format!("Tool execution failed: {}", e),
                            ),
                        }
                    };

                // ── Record tool call in execution graph ──────────────────────
                execution_graph.record_tool_call(
                    step_idx,
                    ToolCallRecord {
                        tool_use_id: tool_use.id.clone(),
                        tool_name: tool_use.name.clone(),
                        is_error: tool_result.is_error,
                        executed_at: Utc::now(),
                    },
                );

                // ── Emit billing UsageEvent::ToolCall ───────────────────────
                #[cfg(feature = "telemetry")]
                if let Some(ref hook) = self.config.billing_hook {
                    let event = brainwires_telemetry::UsageEvent::tool_call(
                        self.id.clone(),
                        tool_use.name.clone(),
                    );
                    if let Err(e) = hook.0.on_usage(&event).await {
                        tracing::warn!(agent_id = %self.id, error = %e, "billing hook error (tool_call)");
                    }
                }

                // ── Emit analytics ToolCall event ────────────────────────────
                #[cfg(feature = "telemetry")]
                if let Some(ref collector) = self.config.analytics_collector {
                    collector.record(brainwires_telemetry::AnalyticsEvent::ToolCall {
                        session_id: None,
                        agent_id: Some(self.id.clone()),
                        tool_name: tool_use.name.clone(),
                        tool_use_id: tool_use.id.clone(),
                        is_error: tool_result.is_error,
                        duration_ms: Some(_tool_exec_start.elapsed().as_millis() as u64),
                        timestamp: Utc::now(),
                    });
                }

                // Track file in working set for file-write operations.
                if !tool_result.is_error
                    && Self::is_file_operation(&tool_use.name)
                    && let Some(fp) = Self::extract_file_path(tool_use)
                {
                    let tokens = estimate_tokens_from_size(
                        std::fs::metadata(&fp).ok().map(|m| m.len()).unwrap_or(0),
                    );
                    self.context.working_set.write().await.add(fp, tokens);
                }

                // Sanitize + wrap external tool results before injecting into
                // conversation history (input sanitization and instruction
                // hierarchy enforcement).
                let final_result = if EXTERNAL_CONTENT_TOOLS.contains(&tool_use.name.as_str())
                    && !tool_result.is_error
                {
                    ToolResult {
                        tool_use_id: tool_result.tool_use_id.clone(),
                        content: wrap_with_content_source(
                            &tool_result.content,
                            ContentSource::ExternalContent,
                        ),
                        is_error: false,
                    }
                } else {
                    tool_result.clone()
                };
                self.conversation_history
                    .write()
                    .await
                    .push(Self::tool_result_message(&final_result));

                // ── Hook E: on_after_tool_execution ──────────────────────────
                if let Some(ref hooks) = self.context.lifecycle_hooks {
                    let conv_len = self.conversation_history.read().await.len();
                    let iter_ctx = self.build_iteration_context(
                        iterations,
                        total_tokens_used,
                        total_cost_usd,
                        &start_time,
                        conv_len,
                    );
                    let mut history = self.conversation_history.write().await;
                    let mut view = ConversationView::new(&mut history);
                    hooks
                        .on_after_tool_execution(&iter_ctx, tool_use, &final_result, &mut view)
                        .await;
                }
            }

            // ── Loop detection ───────────────────────────────────────────────
            if let Some(ref ld) = self.config.loop_detection
                && ld.enabled
            {
                for tool_use in &tool_uses {
                    if recent_tool_names.len() == ld.window_size {
                        recent_tool_names.pop_front();
                    }
                    recent_tool_names.push_back(tool_use.name.clone());
                }
                if recent_tool_names.len() == ld.window_size
                    && recent_tool_names.iter().all(|n| n == &recent_tool_names[0])
                {
                    let stuck = recent_tool_names[0].clone();
                    let error = format!(
                        "Loop detected: '{}' called {} times consecutively. Aborting.",
                        stuck, ld.window_size
                    );
                    tracing::error!(agent_id = %self.id, %error);
                    self.conversation_history
                        .write()
                        .await
                        .push(Message::user(format!(
                            "SYSTEM: {error} Stop calling '{stuck}' and summarise progress."
                        )));
                    self.task.write().await.fail(&error);
                    self.set_status(TaskAgentStatus::Failed(error.clone()))
                        .await;
                    let _ = self
                        .context
                        .communication_hub
                        .broadcast(
                            self.id.clone(),
                            AgentMessage::TaskResult {
                                task_id: task_id.clone(),
                                success: false,
                                result: error.clone(),
                            },
                        )
                        .await;
                    let _ = self
                        .context
                        .communication_hub
                        .unregister_agent(&self.id)
                        .await;
                    self.context
                        .file_lock_manager
                        .release_all_locks(&self.id)
                        .await;
                    let run_ended_at = Utc::now();
                    let telemetry = RunTelemetry::from_graph(
                        &execution_graph,
                        run_ended_at,
                        false,
                        total_cost_usd,
                    );
                    return Ok(TaskAgentResult {
                        agent_id: self.id.clone(),
                        task_id,
                        success: false,
                        summary: error,
                        iterations,
                        replan_count: *self.replan_count.read().await,
                        budget_exhausted: false,
                        partial_output: None,
                        total_tokens_used,
                        total_cost_usd,
                        timed_out: false,
                        failure_category: Some(FailureCategory::LoopDetected),
                        execution_graph: execution_graph.clone(),
                        telemetry,
                        pre_execution_plan: pre_execution_plan.clone(),
                    });
                }
            }

            // ── Hook G: on_after_iteration + context pressure ────────────────
            if let Some(ref hooks) = self.context.lifecycle_hooks {
                let conv_len = self.conversation_history.read().await.len();
                let iter_ctx = self.build_iteration_context(
                    iterations,
                    total_tokens_used,
                    total_cost_usd,
                    &start_time,
                    conv_len,
                );

                // Context pressure check
                if let Some(budget) = self.config.context_budget_tokens {
                    let mut history = self.conversation_history.write().await;
                    let mut view = ConversationView::new(&mut history);
                    let est_tokens = view.estimated_tokens();
                    if est_tokens > budget {
                        hooks
                            .on_context_pressure(&iter_ctx, &mut view, est_tokens, budget)
                            .await;
                    }
                }

                // After-iteration hook
                let mut history = self.conversation_history.write().await;
                let mut view = ConversationView::new(&mut history);
                hooks.on_after_iteration(&iter_ctx, &mut view).await;
            }
        }
    }

    /// Build an [`IterationContext`] snapshot from current loop state.
    fn build_iteration_context<'a>(
        &'a self,
        iteration: u32,
        total_tokens_used: u64,
        total_cost_usd: f64,
        start_time: &std::time::Instant,
        conversation_len: usize,
    ) -> IterationContext<'a> {
        IterationContext {
            agent_id: &self.id,
            iteration,
            max_iterations: self.config.max_iterations,
            total_tokens_used,
            total_cost_usd,
            elapsed: start_time.elapsed(),
            conversation_len,
        }
    }

    /// Extract the most recent assistant text from conversation history, if any.
    async fn last_assistant_text(&self) -> Option<String> {
        self.conversation_history
            .read()
            .await
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .and_then(|m| m.text())
            .map(|t| t.to_string())
    }

    /// Emit an [`AnalyticsEvent::AgentRun`] event if an analytics collector is configured.
    ///
    /// This is a no-op when compiled without the `analytics` feature.
    #[cfg(feature = "telemetry")]
    fn maybe_emit_run_analytics(&self, result: &TaskAgentResult) {
        use brainwires_telemetry::AnalyticsEvent;
        if let Some(ref collector) = self.config.analytics_collector {
            let t = &result.telemetry;
            collector.record(AnalyticsEvent::AgentRun {
                session_id: None,
                agent_id: result.agent_id.clone(),
                task_id: result.task_id.clone(),
                prompt_hash: t.prompt_hash.clone(),
                success: t.success,
                total_iterations: t.total_iterations,
                total_tool_calls: t.total_tool_calls,
                tool_error_count: t.tool_error_count,
                tools_used: t.tools_used.clone(),
                total_prompt_tokens: t.total_prompt_tokens,
                total_completion_tokens: t.total_completion_tokens,
                total_cost_usd: t.total_cost_usd,
                duration_ms: t.duration_ms,
                failure_category: result.failure_category.as_ref().map(|fc| format!("{fc:?}")),
                timestamp: chrono::Utc::now(),
                compliance: None,
            });
        }
    }

    #[cfg(not(feature = "telemetry"))]
    #[inline(always)]
    fn maybe_emit_run_analytics(&self, _result: &TaskAgentResult) {}
}
