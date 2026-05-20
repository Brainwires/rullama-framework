//! Task Agent - Background agent that runs autonomously on a separate task
//!
//! Each TaskAgent has its own context and runs on a separate Tokio task,
//! executing a specific task and reporting results back via the communication hub.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::providers::Provider;
use crate::storage::{CachedEmbeddingProvider, MessageMetadata, MessageStore, VectorDatabase};
use crate::tools::ToolExecutor;
use crate::types::agent::{AgentContext, PermissionMode, Task};
use crate::types::message::{ChatResponse, ContentBlock, Message, MessageContent, Role};
use crate::types::provider::ChatOptions;
use crate::types::session_budget::SessionBudget;
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};
use crate::utils::context_builder::{ContextBuilder, ContextBuilderConfig};
use brainwires::agents::roles::AgentRole;
use brainwires::core::workflow_state::{
    FsWorkflowStateStore, SideEffectRecord, WorkflowCheckpoint, WorkflowStateStore,
};

use super::communication::{AgentMessage, CommunicationHub};
use super::file_locks::{FileLockManager, LockType};

/// Status of a task agent
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskAgentStatus {
    /// Agent is idle, not working on anything
    Idle,
    /// Agent is actively working
    Working(String),
    /// Agent is waiting for a file lock
    WaitingForLock(String),
    /// Agent is paused
    Paused(String),
    /// Agent has completed its task
    Completed(String),
    /// Agent has failed
    Failed(String),
}

impl std::fmt::Display for TaskAgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskAgentStatus::Idle => write!(f, "Idle"),
            TaskAgentStatus::Working(desc) => write!(f, "Working: {}", desc),
            TaskAgentStatus::WaitingForLock(path) => write!(f, "Waiting for lock: {}", path),
            TaskAgentStatus::Paused(reason) => write!(f, "Paused: {}", reason),
            TaskAgentStatus::Completed(summary) => write!(f, "Completed: {}", summary),
            TaskAgentStatus::Failed(error) => write!(f, "Failed: {}", error),
        }
    }
}

/// Result from a task agent execution
#[derive(Debug, Clone)]
pub struct TaskAgentResult {
    /// Agent ID
    pub agent_id: String,
    /// Task ID
    pub task_id: String,
    /// Whether the task completed successfully
    pub success: bool,
    /// Result summary
    pub summary: String,
    /// Number of iterations executed
    pub iterations: u32,
}

/// Configuration for a task agent
#[derive(Debug, Clone)]
pub struct TaskAgentConfig {
    /// Maximum iterations before giving up
    pub max_iterations: u32,
    /// Permission mode for tool execution
    pub permission_mode: PermissionMode,
    /// System prompt for the agent
    pub system_prompt: Option<String>,
    /// Temperature for AI calls
    pub temperature: f32,
    /// Max tokens for AI responses
    pub max_tokens: u32,
    /// Validation configuration (enforced quality checks)
    pub validation_config: Option<super::validation_loop::ValidationConfig>,
    /// MDAP configuration (Massively Decomposed Agentic Processes)
    pub mdap_config: Option<crate::mdap::MdapConfig>,
    /// Analytics collector — emit AgentRun and ToolCall events
    pub analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,

    /// Optional role that restricts which tools are presented to the model.
    ///
    /// Enforcement happens at provider-call time: the model only sees the tools
    /// permitted by the role, so it cannot accidentally (or intentionally) use
    /// out-of-scope tools. `None` grants full tool access (equivalent to
    /// `AgentRole::Execution`).
    pub role: Option<AgentRole>,

    // --- Budget limits (per-agent) ---
    /// Maximum total tokens (prompt + completion) this agent may consume.
    /// Checked after each provider call; the agent fails gracefully when exceeded.
    pub max_total_tokens: Option<u64>,
    /// Maximum cost in USD this agent may incur. Requires the provider to supply
    /// token-usage data and a matching entry in the pricing table.
    pub max_cost_usd: Option<f64>,
    /// Wall-clock timeout in seconds. The agent fails gracefully when exceeded.
    pub timeout_secs: Option<u64>,
    /// Optional session-level budget shared across all agents in a session.
    /// When set, both per-agent limits *and* the shared session cap are enforced.
    pub session_budget: Option<Arc<SessionBudget>>,
}

impl Default for TaskAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100, // High default to avoid artificial limits on complex tasks
            permission_mode: PermissionMode::Auto,
            system_prompt: None,
            temperature: 0.7,
            max_tokens: 4096, // Conservative limit to prevent corruption
            validation_config: Some(super::validation_loop::ValidationConfig::default()), // Enable validation by default
            mdap_config: None, // Disabled by default
            analytics_collector: crate::utils::logger::analytics_collector()
                .map(std::sync::Arc::new),
            role: None,
            max_total_tokens: None,
            max_cost_usd: None,
            timeout_secs: None,
            session_budget: None,
        }
    }
}

/// Task Agent - runs autonomously on a background task
pub struct TaskAgent {
    /// Unique agent ID
    id: String,
    /// The task this agent is working on
    task: Arc<RwLock<Task>>,
    /// AI provider
    provider: Arc<dyn Provider>,
    /// Tool executor
    tool_executor: ToolExecutor,
    /// Communication hub for messaging
    communication_hub: Arc<CommunicationHub>,
    /// File lock manager
    file_lock_manager: Arc<FileLockManager>,
    /// Current status
    status: Arc<RwLock<TaskAgentStatus>>,
    /// Configuration
    config: TaskAgentConfig,
    /// Agent context
    context: Arc<RwLock<AgentContext>>,
    /// Per-run shared registry of `(path -> SHA-256 of most recent write)`.
    ///
    /// Each iteration builds a fresh `ToolContext` (via
    /// `ToolContext::from_agent_context`), so the registry must live on the
    /// agent — not on `ToolContext` or `AgentContext` — to persist across
    /// iterations.  The file_ops `write_file` tool records hashes into this
    /// registry; the validation loop re-reads at finalisation to detect
    /// post-validation clobber by a concurrent writer.
    intended_writes: brainwires::core::IntendedWrites,
}

impl TaskAgent {
    /// Create a new task agent
    pub fn new(
        id: String,
        task: Task,
        provider: Arc<dyn Provider>,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
        context: AgentContext,
        config: TaskAgentConfig,
    ) -> Self {
        Self {
            id,
            task: Arc::new(RwLock::new(task)),
            provider,
            tool_executor: ToolExecutor::new(config.permission_mode),
            communication_hub,
            file_lock_manager,
            status: Arc::new(RwLock::new(TaskAgentStatus::Idle)),
            config,
            context: Arc::new(RwLock::new(context)),
            intended_writes: brainwires::core::IntendedWrites::new(),
        }
    }

    /// Get the agent ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the current status
    pub async fn status(&self) -> TaskAgentStatus {
        self.status.read().await.clone()
    }

    /// Get the task
    pub async fn task(&self) -> Task {
        self.task.read().await.clone()
    }

    /// Set the status and notify via communication hub
    async fn set_status(&self, status: TaskAgentStatus) {
        *self.status.write().await = status.clone();

        // Send status update
        let _ = self
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

    /// Check if a tool is a file operation that should update working set
    fn is_file_operation(tool_name: &str) -> bool {
        matches!(
            tool_name,
            "read_file" | "write_file" | "edit_file" | "append_to_file" | "delete_file"
        )
    }

    /// Extract file path from tool use
    fn extract_file_path(tool_use: &ToolUse) -> Option<std::path::PathBuf> {
        use std::path::PathBuf;

        // Check different parameter names tools use for file paths
        let path_str = tool_use
            .input
            .get("file_path")
            .or_else(|| tool_use.input.get("path"))
            .and_then(|v| v.as_str())?;

        Some(PathBuf::from(path_str))
    }

    /// Execute the task
    pub async fn execute(&self) -> Result<TaskAgentResult> {
        let task_id = {
            let task = self.task.read().await;
            task.id.clone()
        };

        let task_description = {
            let task = self.task.read().await;
            task.description.clone()
        };

        // Generate a trace ID for this execution. Written into AgentContext.metadata so
        // every ToolContext created from it automatically carries the trace ID, enabling
        // cross-system correlation in audit logs and OTel exporters.
        let trace_id = uuid::Uuid::new_v4();
        {
            let mut ctx = self.context.write().await;
            ctx.metadata
                .insert("trace_id".to_string(), trace_id.to_string());
        }

        tracing::info!(
            agent_id = %self.id,
            task_id = %task_id,
            %trace_id,
            task = %task_description,
            "TaskAgent starting execution"
        );

        // Register with communication hub
        if !self.communication_hub.is_registered(&self.id).await {
            self.communication_hub
                .register_agent(self.id.clone())
                .await?;
        }

        // Start the task
        {
            let mut task = self.task.write().await;
            task.start();
        }

        self.set_status(TaskAgentStatus::Working(task_description.clone()))
            .await;

        // Add initial user message
        {
            let mut context = self.context.write().await;
            let user_message = Message {
                role: Role::User,
                content: MessageContent::Text(task_description.clone()),
                name: None,
                metadata: None,
            };
            context.conversation_history.push(user_message);
        }

        let mut iterations = 0;
        let mut total_tokens_used: u64 = 0;
        let started_at = std::time::Instant::now();

        // Register this agent with the session budget (if any).
        if let Some(ref budget) = self.config.session_budget {
            budget.increment_agent_count();
        }

        // Initialise workflow checkpoint store and load any existing checkpoint.
        // A prior checkpoint means the agent crashed mid-run; we resume from where
        // it left off rather than re-executing already-completed tool calls.
        let workflow_store = match FsWorkflowStateStore::with_default_path() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(agent_id = %self.id, %e, "workflow state store unavailable");
                None
            }
        };
        let mut workflow_checkpoint: Option<WorkflowCheckpoint> =
            if let Some(ref s) = workflow_store {
                match s.load_checkpoint(&task_id).await {
                    Ok(Some(cp)) => {
                        tracing::info!(
                            agent_id = %self.id,
                            task_id = %task_id,
                            step = cp.step_index,
                            completed = cp.completed_tool_ids.len(),
                            "Resuming from prior workflow checkpoint"
                        );
                        Some(cp)
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(%e, "could not load workflow checkpoint");
                        None
                    }
                }
            } else {
                None
            };

        // Initialise context enhancement. Runs without gating so retrieval fires on
        // every call (task agents don't use compaction markers like the chat path).
        let message_store = Self::init_message_store().await;
        let context_builder = ContextBuilder::with_config(ContextBuilderConfig {
            use_gating: false,
            ..ContextBuilderConfig::default()
        });
        // Ensure the messages table exists before we start writing to it.
        if let Some(ref store) = message_store
            && let Err(e) = store.ensure_table().await
        {
            tracing::debug!("TaskAgent: MessageStore ensure_table failed (non-fatal) — {e}");
        }
        let mut persisted_up_to: usize = 0;

        loop {
            iterations += 1;

            tracing::debug!(
                agent_id = %self.id,
                iteration = iterations,
                max_iterations = self.config.max_iterations,
                "TaskAgent iteration starting"
            );

            // Update task iterations
            {
                let mut task = self.task.write().await;
                task.increment_iteration();
            }

            // Check wall-clock timeout
            if let Some(timeout_secs) = self.config.timeout_secs
                && started_at.elapsed().as_secs() >= timeout_secs
            {
                let error = format!("Agent {} timed out after {}s", self.id, timeout_secs);
                tracing::error!(agent_id = %self.id, timeout_secs, "TaskAgent timed out");
                return self.fail_agent(&task_id, &error, iterations).await;
            }

            // Check iteration limit
            if iterations >= self.config.max_iterations {
                let error = format!(
                    "Agent {} exceeded maximum iterations ({})",
                    self.id, self.config.max_iterations
                );

                tracing::error!(
                    agent_id = %self.id,
                    iterations = iterations,
                    max_iterations = self.config.max_iterations,
                    "TaskAgent exceeded max iterations"
                );

                {
                    let mut task = self.task.write().await;
                    task.fail(&error);
                }

                self.set_status(TaskAgentStatus::Failed(error.clone()))
                    .await;

                // Send task result
                let _ = self
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

                // Unregister
                let _ = self.communication_hub.unregister_agent(&self.id).await;

                // Release all locks
                self.file_lock_manager.release_all_locks(&self.id).await;

                return Ok(TaskAgentResult {
                    agent_id: self.id.clone(),
                    task_id,
                    success: false,
                    summary: error,
                    iterations,
                });
            }

            // Check for incoming messages (non-blocking)
            if let Some(envelope) = self.communication_hub.try_receive_message(&self.id).await {
                match envelope.message {
                    AgentMessage::HelpResponse {
                        request_id,
                        response,
                    } => {
                        // Add help response to context
                        let mut context = self.context.write().await;
                        context.conversation_history.push(Message {
                            role: Role::User,
                            content: MessageContent::Text(format!(
                                "Response to help request {}: {}",
                                request_id, response
                            )),
                            name: None,
                            metadata: None,
                        });
                    }
                    _ => {
                        // Handle other messages as needed
                    }
                }
            }

            // Call the AI provider
            let response = self
                .call_provider(message_store.as_ref(), &context_builder, &task_id)
                .await?;

            // --- Budget enforcement ---
            let call_tokens = response.usage.total_tokens as u64;
            total_tokens_used += call_tokens;

            // Per-agent token limit
            if let Some(limit) = self.config.max_total_tokens
                && total_tokens_used > limit
            {
                let error = format!(
                    "Agent {} exceeded token budget ({} > {})",
                    self.id, total_tokens_used, limit
                );
                tracing::warn!(agent_id = %self.id, total_tokens_used, limit, "token budget exceeded");
                return self.fail_agent(&task_id, &error, iterations).await;
            }

            // Session-level budget: record usage then check accumulated totals
            if let Some(ref budget) = self.config.session_budget {
                budget.record_run(call_tokens, 0.0);
                if let Err(e) = budget.check_limits() {
                    let error = format!("Agent {} stopped: session budget — {}", self.id, e);
                    tracing::warn!(agent_id = %self.id, %e, "session budget exceeded");
                    return self.fail_agent(&task_id, &error, iterations).await;
                }
            }
            // --- End budget enforcement ---

            // Check if task is complete
            if let Some(finish_reason) = &response.finish_reason
                && (finish_reason == "end_turn" || finish_reason == "stop")
            {
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();

                // VALIDATION: Run checks before allowing completion
                if let Some(validation_attempt) =
                    self.attempt_validated_completion(&message_text).await?
                {
                    // Clean up workflow checkpoint on successful completion.
                    if let Some(ref s) = workflow_store
                        && let Err(e) = s.delete_checkpoint(&task_id).await
                    {
                        tracing::debug!(%e, "checkpoint delete failed (non-fatal)");
                    }
                    return Ok(validation_attempt);
                }

                // Validation failed, continue looping to let agent fix issues
                continue;
            }

            // Process tool uses
            let tool_uses = self.extract_tool_uses(&response.message);

            if tool_uses.is_empty() {
                // No tool uses, treat as completion
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();

                // VALIDATION: Run checks before allowing completion
                if let Some(validation_attempt) =
                    self.attempt_validated_completion(&message_text).await?
                {
                    // Clean up workflow checkpoint on successful completion.
                    if let Some(ref s) = workflow_store
                        && let Err(e) = s.delete_checkpoint(&task_id).await
                    {
                        tracing::debug!(%e, "checkpoint delete failed (non-fatal)");
                    }
                    return Ok(validation_attempt);
                }

                // Validation failed, continue looping to let agent fix issues
                continue;
            }

            // Add assistant message to history
            {
                let mut context = self.context.write().await;
                context.conversation_history.push(response.message.clone());
            }

            // Execute tools
            let tool_context = {
                let context = self.context.read().await;
                let mut tc = ToolContext::from_agent_context(&context);
                // Share the per-run intended-writes registry so write_file
                // records hashes that the validation loop re-reads at
                // finalisation to detect post-validation clobber.
                tc.intended_writes = Some(self.intended_writes.clone());
                tc
            };

            for tool_use in tool_uses {
                tracing::debug!("[Agent {}] Processing tool: {}", self.id, tool_use.name);

                // Skip tool calls that were already completed in a prior run.
                if let Some(ref cp) = workflow_checkpoint
                    && cp.is_completed(&tool_use.id)
                {
                    tracing::info!(
                        agent_id = %self.id,
                        tool_use_id = %tool_use.id,
                        tool = %tool_use.name,
                        "Skipping already-completed tool call (crash-resume)"
                    );
                    continue;
                }

                // Determine if we need file locks
                let lock_needed = self.get_lock_requirement(&tool_use);
                tracing::debug!("[Agent {}] Lock needed: {:?}", self.id, lock_needed);

                if let Some((path, lock_type)) = lock_needed {
                    // Try to acquire lock
                    tracing::debug!("[Agent {}] Acquiring lock for path: {}", self.id, path);
                    self.set_status(TaskAgentStatus::WaitingForLock(path.clone()))
                        .await;

                    match self
                        .file_lock_manager
                        .acquire_lock(&self.id, &path, lock_type)
                        .await
                    {
                        Ok(_guard) => {
                            tracing::debug!(
                                "[Agent {}] Lock acquired, executing {}",
                                self.id,
                                tool_use.name
                            );
                            self.set_status(TaskAgentStatus::Working(format!(
                                "Executing {}",
                                tool_use.name
                            )))
                            .await;

                            // Execute the tool
                            tracing::debug!(
                                "[Agent {}] Calling tool_executor.execute for {}",
                                self.id,
                                tool_use.name
                            );
                            let _tool_start = std::time::Instant::now();
                            let result =
                                self.tool_executor.execute(&tool_use, &tool_context).await?;
                            if let Some(ref collector) = self.config.analytics_collector {
                                collector.record(brainwires_telemetry::AnalyticsEvent::ToolCall {
                                    session_id: None,
                                    agent_id: Some(self.id.clone()),
                                    tool_name: tool_use.name.clone(),
                                    tool_use_id: tool_use.id.clone(),
                                    is_error: result.is_error,
                                    duration_ms: Some(_tool_start.elapsed().as_millis() as u64),
                                    timestamp: chrono::Utc::now(),
                                });
                            }
                            tracing::debug!(
                                "[Agent {}] Tool {} returned: is_error={}",
                                self.id,
                                tool_use.name,
                                result.is_error
                            );

                            // Add tool result to context
                            tracing::debug!("[Agent {}] Acquiring context write lock", self.id);
                            let mut context = self.context.write().await;
                            tracing::debug!("[Agent {}] Context write lock acquired", self.id);
                            context.conversation_history.push(Message {
                                role: Role::User,
                                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                                    tool_use_id: result.tool_use_id.clone(),
                                    content: result.content.clone(),
                                    is_error: Some(result.is_error),
                                }]),
                                name: None,
                                metadata: None,
                            });
                            tracing::debug!("[Agent {}] Tool result added to context", self.id);

                            // Add file to working set for file operations
                            if !result.is_error
                                && Self::is_file_operation(&tool_use.name)
                                && let Some(file_path) = Self::extract_file_path(&tool_use)
                            {
                                let tokens = crate::types::working_set::estimate_tokens_from_size(
                                    std::fs::metadata(&file_path)
                                        .ok()
                                        .map(|m| m.len())
                                        .unwrap_or(0),
                                );
                                let path_display = file_path.display().to_string();
                                context.working_set.add(file_path, tokens);
                                tracing::debug!(
                                    "[Agent {}] Added {} to working set",
                                    self.id,
                                    path_display
                                );
                            }

                            // Persist checkpoint for this completed tool call (fire-and-forget).
                            if !result.is_error {
                                let target = Self::extract_file_path(&tool_use)
                                    .map(|p| p.display().to_string());
                                let reversible = matches!(
                                    tool_use.name.as_str(),
                                    "read_file" | "list_directory" | "search_code" | "glob"
                                );
                                let effect = SideEffectRecord::new(
                                    &tool_use.id,
                                    &tool_use.name,
                                    target,
                                    reversible,
                                );
                                if let Some(ref s) = workflow_store {
                                    if let Err(e) =
                                        s.mark_step_complete(&task_id, &tool_use.id, effect).await
                                    {
                                        tracing::debug!(%e, "checkpoint persist failed (non-fatal)");
                                    } else {
                                        // Keep in-memory checkpoint in sync.
                                        workflow_checkpoint
                                            .get_or_insert_with(|| {
                                                WorkflowCheckpoint::new(&task_id, &self.id)
                                            })
                                            .completed_tool_ids
                                            .insert(tool_use.id.clone());
                                    }
                                }
                            }

                            // Lock is released when guard is dropped
                        }
                        Err(e) => {
                            tracing::warn!("[Agent {}] Failed to acquire lock: {}", self.id, e);
                            // Could not acquire lock - report as tool error
                            let mut context = self.context.write().await;
                            context.conversation_history.push(Message {
                                role: Role::User,
                                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                                    tool_use_id: tool_use.id.clone(),
                                    content: format!("Could not acquire file lock: {}", e),
                                    is_error: Some(true),
                                }]),
                                name: None,
                                metadata: None,
                            });
                        }
                    }
                } else {
                    // No lock needed
                    self.set_status(TaskAgentStatus::Working(format!(
                        "Executing {}",
                        tool_use.name
                    )))
                    .await;

                    let _tool_start = std::time::Instant::now();
                    let result = self.tool_executor.execute(&tool_use, &tool_context).await?;
                    if let Some(ref collector) = self.config.analytics_collector {
                        collector.record(brainwires_telemetry::AnalyticsEvent::ToolCall {
                            session_id: None,
                            agent_id: Some(self.id.clone()),
                            tool_name: tool_use.name.clone(),
                            tool_use_id: tool_use.id.clone(),
                            is_error: result.is_error,
                            duration_ms: Some(_tool_start.elapsed().as_millis() as u64),
                            timestamp: chrono::Utc::now(),
                        });
                    }
                    let mut context = self.context.write().await;
                    context.conversation_history.push(Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                            tool_use_id: result.tool_use_id.clone(),
                            content: result.content.clone(),
                            is_error: Some(result.is_error),
                        }]),
                        name: None,
                        metadata: None,
                    });

                    // Add file to working set for file operations
                    if !result.is_error
                        && Self::is_file_operation(&tool_use.name)
                        && let Some(file_path) = Self::extract_file_path(&tool_use)
                    {
                        let tokens = crate::types::working_set::estimate_tokens_from_size(
                            std::fs::metadata(&file_path)
                                .ok()
                                .map(|m| m.len())
                                .unwrap_or(0),
                        );
                        context.working_set.add(file_path, tokens);
                        tracing::debug!("[Agent {}] Added file to working set", self.id);
                    }

                    // Persist checkpoint (fire-and-forget).
                    if !result.is_error {
                        let target =
                            Self::extract_file_path(&tool_use).map(|p| p.display().to_string());
                        let reversible = matches!(
                            tool_use.name.as_str(),
                            "read_file" | "list_directory" | "search_code" | "glob"
                        );
                        let effect =
                            SideEffectRecord::new(&tool_use.id, &tool_use.name, target, reversible);
                        if let Some(ref s) = workflow_store {
                            if let Err(e) =
                                s.mark_step_complete(&task_id, &tool_use.id, effect).await
                            {
                                tracing::debug!(%e, "checkpoint persist failed (non-fatal)");
                            } else {
                                workflow_checkpoint
                                    .get_or_insert_with(|| {
                                        WorkflowCheckpoint::new(&task_id, &self.id)
                                    })
                                    .completed_tool_ids
                                    .insert(tool_use.id.clone());
                            }
                        }
                    }
                }
            }

            // Persist new messages to MessageStore for future context retrieval.
            // Runs fire-and-forget — a failure here never aborts the agent.
            if let Some(ref store) = message_store {
                let history = self.context.read().await;
                let new_msgs = &history.conversation_history[persisted_up_to..];
                if !new_msgs.is_empty() {
                    Self::persist_messages(store, new_msgs, &task_id).await;
                    persisted_up_to = history.conversation_history.len();
                }
            }
        }
    }

    /// Attempt to complete task with validation checks
    /// Returns Some(result) if validation passed, None if failed (should retry)
    async fn attempt_validated_completion(
        &self,
        message_text: &str,
    ) -> Result<Option<TaskAgentResult>> {
        let task_id = {
            let task = self.task.read().await;
            task.id.clone()
        };

        // Check if validation is enabled
        if let Some(ref validation_config) = self.config.validation_config
            && validation_config.enabled
        {
            tracing::info!(
                "[Agent {}] Running validation checks before completion...",
                self.id
            );

            // Get working set files from context
            let working_set_files = {
                let context = self.context.read().await;
                context
                    .working_set
                    .file_paths()
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<String>>()
            };

            // Update validation config with working set files
            let mut config_with_ws = validation_config.clone();
            config_with_ws.working_set_files = working_set_files;
            // Inject the shared intended-writes registry so validation can
            // detect post-validation clobber by a concurrent writer.
            if config_with_ws.intended_writes.is_none() {
                config_with_ws.intended_writes = Some(self.intended_writes.clone());
            }

            tracing::debug!(
                "[Agent {}] Validating {} working set files",
                self.id,
                config_with_ws.working_set_files.len()
            );

            // Run validation
            match super::validation_loop::run_validation(&config_with_ws).await {
                Ok(validation_result) => {
                    if !validation_result.passed {
                        // Validation failed - inject feedback and continue
                        tracing::warn!(
                            "[Agent {}] Validation failed with {} issues",
                            self.id,
                            validation_result.issues.len()
                        );

                        let feedback =
                            super::validation_loop::format_validation_feedback(&validation_result);

                        // Add validation feedback to conversation history
                        {
                            let mut context = self.context.write().await;
                            context.conversation_history.push(Message {
                                role: Role::User,
                                content: MessageContent::Text(feedback),
                                name: None,
                                metadata: None,
                            });
                        }

                        // Return None to continue the loop
                        return Ok(None);
                    } else {
                        tracing::info!("[Agent {}] ✓ All validation checks passed!", self.id);
                    }
                }
                Err(e) => {
                    tracing::error!("[Agent {}] Validation error: {}", self.id, e);
                    // Continue anyway if validation itself fails
                }
            }
        }

        // Validation passed or disabled - complete the task
        {
            let mut task = self.task.write().await;
            task.complete(message_text);
        }

        self.set_status(TaskAgentStatus::Completed(message_text.to_string()))
            .await;

        // Send task result
        let _ = self
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

        // Unregister
        let _ = self.communication_hub.unregister_agent(&self.id).await;

        // Release all locks
        self.file_lock_manager.release_all_locks(&self.id).await;

        // Get iterations from context
        let iterations = {
            let task = self.task.read().await;
            task.iterations
        };

        Ok(Some(TaskAgentResult {
            agent_id: self.id.clone(),
            task_id,
            success: true,
            summary: message_text.to_string(),
            iterations,
        }))
    }

    /// Shared failure path: update task state, broadcast result, unregister, release locks.
    async fn fail_agent(
        &self,
        task_id: &str,
        error: &str,
        iterations: u32,
    ) -> Result<TaskAgentResult> {
        {
            let mut task = self.task.write().await;
            task.fail(error);
        }
        self.set_status(TaskAgentStatus::Failed(error.to_string()))
            .await;
        let _ = self
            .communication_hub
            .broadcast(
                self.id.clone(),
                AgentMessage::TaskResult {
                    task_id: task_id.to_string(),
                    success: false,
                    result: error.to_string(),
                },
            )
            .await;
        let _ = self.communication_hub.unregister_agent(&self.id).await;
        self.file_lock_manager.release_all_locks(&self.id).await;
        Ok(TaskAgentResult {
            agent_id: self.id.clone(),
            task_id: task_id.to_string(),
            success: false,
            summary: error.to_string(),
            iterations,
        })
    }

    /// Try to initialise a `MessageStore` backed by LanceDB.
    ///
    /// Returns `None` (with a debug log) if the DB or embedding provider cannot be
    /// created — the agent continues without context enhancement rather than failing.
    async fn init_message_store() -> Option<MessageStore> {
        let db_path = match crate::config::PlatformPaths::conversations_db_path() {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!("TaskAgent: skipping MessageStore — {e}");
                return None;
            }
        };
        let embeddings = match CachedEmbeddingProvider::new() {
            Ok(e) => Arc::new(e),
            Err(e) => {
                tracing::debug!("TaskAgent: skipping MessageStore (embeddings) — {e}");
                return None;
            }
        };
        let lance_client =
            match crate::storage::LanceDatabase::new(db_path.to_str().unwrap_or_default()).await {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    tracing::debug!("TaskAgent: skipping MessageStore (LanceDB) — {e}");
                    return None;
                }
            };
        if let Err(e) = lance_client.initialize(embeddings.dimension()).await {
            tracing::debug!("TaskAgent: skipping MessageStore (init) — {e}");
            return None;
        }
        Some(MessageStore::new(lance_client, embeddings))
    }

    /// Persist messages that haven't been stored yet.
    ///
    /// Runs fire-and-forget (errors are logged, not propagated) to avoid blocking
    /// the main agent loop on storage I/O.
    async fn persist_messages(store: &MessageStore, messages: &[Message], conversation_id: &str) {
        let batch: Vec<MessageMetadata> = messages
            .iter()
            .filter_map(|m| {
                let content = match &m.content {
                    MessageContent::Text(t) => t.clone(),
                    MessageContent::Blocks(_) => {
                        // Skip tool-result / tool-use blocks — they're noisy for retrieval
                        return None;
                    }
                };
                if content.trim().is_empty() {
                    return None;
                }
                Some(MessageMetadata {
                    message_id: uuid::Uuid::new_v4().to_string(),
                    conversation_id: conversation_id.to_string(),
                    role: format!("{:?}", m.role).to_lowercase(),
                    content,
                    token_count: None,
                    model_id: None,
                    images: None,
                    created_at: chrono::Utc::now().timestamp(),
                    expires_at: None,
                })
            })
            .collect();

        if !batch.is_empty()
            && let Err(e) = store.add_batch(batch).await
        {
            tracing::debug!("TaskAgent: MessageStore persist error (non-fatal) — {e}");
        }
    }

    /// Call the AI provider, optionally enhancing the message history with
    /// semantically retrieved context from `MessageStore`.
    async fn call_provider(
        &self,
        message_store: Option<&MessageStore>,
        context_builder: &ContextBuilder,
        conversation_id: &str,
    ) -> Result<ChatResponse> {
        let context = self.context.read().await;

        // Build system prompt via the framework registry, which handles role suffix injection.
        let system_prompt = self.config.system_prompt.clone().unwrap_or_else(|| {
            brainwires::agents::build_agent_prompt(
                brainwires::agents::AgentPromptKind::Reasoning {
                    agent_id: &self.id,
                    working_directory: &context.working_directory,
                },
                self.config.role,
            )
        });

        // Filter tools to only those permitted by the agent's role.
        let available_tools: Vec<_> = match self.config.role {
            Some(role) => role.filter_tools(&context.tools),
            None => context.tools.clone(),
        };

        // Build enhanced message list when a message store is available.
        // ContextBuilder uses use_gating=false so retrieval fires on every call
        // (unlike chat mode which gates on a compaction marker).
        let messages = if let Some(store) = message_store {
            let last_user_query = context
                .conversation_history
                .iter()
                .rev()
                .find(|m| m.role == Role::User)
                .and_then(|m| match &m.content {
                    MessageContent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .unwrap_or("");

            context_builder
                .build_full_context(
                    &context.conversation_history,
                    last_user_query,
                    store,
                    conversation_id,
                )
                .await
                .unwrap_or_else(|e| {
                    tracing::debug!("TaskAgent: context enhancement failed (non-fatal) — {e}");
                    context.conversation_history.clone()
                })
        } else {
            context.conversation_history.clone()
        };

        let options = ChatOptions {
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
            top_p: None,
            stop: None,
            system: Some(system_prompt),
            model: None,
            cache_strategy: Default::default(),
        };

        self.provider
            .chat(&messages, Some(&available_tools), &options)
            .await
    }

    /// Extract tool uses from a message
    fn extract_tool_uses(&self, message: &Message) -> Vec<ToolUse> {
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

    /// Determine if a tool needs a file lock and what type
    fn get_lock_requirement(&self, tool_use: &ToolUse) -> Option<(String, LockType)> {
        let name = tool_use.name.as_str();

        // Extract path from tool input
        let path = tool_use
            .input
            .get("path")
            .or_else(|| tool_use.input.get("file_path"));

        if let Some(path_value) = path {
            if let Some(path_str) = path_value.as_str() {
                match name {
                    // Read operations - shared lock
                    "read_file" | "list_directory" | "search_code" => {
                        Some((path_str.to_string(), LockType::Read))
                    }
                    // Write operations - exclusive lock
                    "write_file" | "edit_file" | "patch_file" | "delete_file"
                    | "create_directory" => Some((path_str.to_string(), LockType::Write)),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Spawn a task agent on a background Tokio task
pub fn spawn_task_agent(agent: Arc<TaskAgent>) -> tokio::task::JoinHandle<Result<TaskAgentResult>> {
    tokio::spawn(async move { agent.execute().await })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::agent::Task;
    use crate::types::message::{ChatResponse, Message, MessageContent, Role, StreamChunk, Usage};
    use crate::types::provider::ChatOptions;
    use crate::types::tool::Tool;
    use async_trait::async_trait;
    use futures::stream::BoxStream;

    /// Mock provider for testing
    struct MockProvider {
        responses: std::sync::Mutex<Vec<ChatResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }

        fn single_response(text: &str) -> Self {
            Self::new(vec![ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(text.to_string()),
                    name: None,
                    metadata: None,
                },
                finish_reason: Some("stop".to_string()),
                usage: Usage::default(),
            }])
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock-provider"
        }

        async fn chat(
            &self,
            _messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                anyhow::bail!("No more mock responses")
            }
            Ok(responses.remove(0))
        }

        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_task_agent_creation() {
        let provider = Arc::new(MockProvider::single_response("Done"));
        let hub = Arc::new(CommunicationHub::new());
        let lock_manager = Arc::new(FileLockManager::new());
        let task = Task::new("task-1", "Test task");
        let context = AgentContext::default();
        let config = TaskAgentConfig::default();

        let agent = TaskAgent::new(
            "agent-1".to_string(),
            task,
            provider,
            hub,
            lock_manager,
            context,
            config,
        );

        assert_eq!(agent.id(), "agent-1");
        assert_eq!(agent.status().await, TaskAgentStatus::Idle);
    }

    #[tokio::test]
    async fn test_task_agent_execution() {
        let provider = Arc::new(MockProvider::single_response("Task completed successfully"));
        let hub = Arc::new(CommunicationHub::new());
        let lock_manager = Arc::new(FileLockManager::new());
        let task = Task::new("task-1", "Test task");
        let context = AgentContext::default();
        let config = TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        };

        let agent = Arc::new(TaskAgent::new(
            "agent-1".to_string(),
            task,
            provider,
            hub,
            lock_manager,
            context,
            config,
        ));

        let result = agent.execute().await.unwrap();

        assert!(result.success);
        assert_eq!(result.agent_id, "agent-1");
        assert_eq!(result.task_id, "task-1");
        assert_eq!(result.iterations, 1);
    }

    #[tokio::test]
    async fn test_task_agent_status() {
        let status = TaskAgentStatus::Working("Processing data".to_string());
        assert_eq!(status.to_string(), "Working: Processing data");

        let status = TaskAgentStatus::Completed("All done".to_string());
        assert_eq!(status.to_string(), "Completed: All done");

        let status = TaskAgentStatus::Failed("Error occurred".to_string());
        assert_eq!(status.to_string(), "Failed: Error occurred");
    }

    #[tokio::test]
    async fn test_spawn_task_agent() {
        let provider = Arc::new(MockProvider::single_response("Done"));
        let hub = Arc::new(CommunicationHub::new());
        let lock_manager = Arc::new(FileLockManager::new());
        let task = Task::new("task-1", "Test task");
        let context = AgentContext::default();
        let config = TaskAgentConfig {
            validation_config: None,
            ..Default::default()
        };

        let agent = Arc::new(TaskAgent::new(
            "agent-1".to_string(),
            task,
            provider,
            hub,
            lock_manager,
            context,
            config,
        ));

        let handle = spawn_task_agent(agent);
        let result = handle.await.unwrap().unwrap();

        assert!(result.success);
    }
}
