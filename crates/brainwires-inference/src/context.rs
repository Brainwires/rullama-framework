//! Agent Context - environment for autonomous task execution
//!
//! [`AgentContext`] bundles the stable environment that a [`TaskAgent`][super::task_agent::TaskAgent]
//! operates in: working directory, tool executor, inter-agent communication,
//! file lock coordination, and the working set of files currently in context.
//!
//! Conversation history and tool definitions are maintained *internally* by
//! the agent; they are not part of this context.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use brainwires_core::WorkingSet;
use brainwires_tool_runtime::{ToolExecutor, ToolPreHook};

use crate::agent_hooks::AgentLifecycleHooks;
use brainwires_agent::communication::CommunicationHub;
use brainwires_agent::file_locks::FileLockManager;

/// Environment context for a task agent.
///
/// Pass this to [`TaskAgent::new`][super::task_agent::TaskAgent::new] at
/// construction time. All fields are cheaply cloneable via `Arc`.
#[non_exhaustive]
pub struct AgentContext {
    /// Working directory used for resolving relative file paths.
    pub working_directory: String,

    /// Executes tools on behalf of the agent.
    pub tool_executor: Arc<dyn ToolExecutor>,

    /// Inter-agent message bus.
    pub communication_hub: Arc<CommunicationHub>,

    /// Coordinates exclusive/shared file access across concurrent agents.
    pub file_lock_manager: Arc<FileLockManager>,

    /// Tracks files currently loaded into the agent's context window.
    pub working_set: Arc<RwLock<WorkingSet>>,

    /// Application-specific metadata passed through to tools.
    pub metadata: HashMap<String, String>,

    /// Optional pre-execution hook for semantic tool validation.
    ///
    /// When set, the hook is called before every tool execution. Returning
    /// [`PreHookDecision::Reject`](brainwires_tool_runtime::PreHookDecision::Reject) causes the tool call to be skipped and
    /// the rejection message injected as a `ToolResult::error`.
    pub pre_execute_hook: Option<Arc<dyn ToolPreHook>>,

    /// Optional lifecycle hooks for granular loop control.
    ///
    /// When set, the agent loop calls these hooks at every phase: iteration
    /// boundaries, provider calls, tool execution, completion, and context
    /// management. All hook methods have default no-op implementations.
    ///
    /// See [`AgentLifecycleHooks`] for the full hook surface.
    pub lifecycle_hooks: Option<Arc<dyn AgentLifecycleHooks>>,
}

impl AgentContext {
    /// Create a new agent context with the given environment.
    ///
    /// A fresh, empty [`WorkingSet`] is created automatically. Use
    /// [`AgentContext::with_working_set`] to supply a pre-populated one.
    pub fn new(
        working_directory: impl Into<String>,
        tool_executor: Arc<dyn ToolExecutor>,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
    ) -> Self {
        Self {
            working_directory: working_directory.into(),
            tool_executor,
            communication_hub,
            file_lock_manager,
            working_set: Arc::new(RwLock::new(WorkingSet::new())),
            metadata: HashMap::new(),
            pre_execute_hook: None,
            lifecycle_hooks: None,
        }
    }

    /// Create a context that shares an existing [`WorkingSet`].
    pub fn with_working_set(
        working_directory: impl Into<String>,
        tool_executor: Arc<dyn ToolExecutor>,
        communication_hub: Arc<CommunicationHub>,
        file_lock_manager: Arc<FileLockManager>,
        working_set: Arc<RwLock<WorkingSet>>,
    ) -> Self {
        Self {
            working_directory: working_directory.into(),
            tool_executor,
            communication_hub,
            file_lock_manager,
            working_set,
            metadata: HashMap::new(),
            pre_execute_hook: None,
            lifecycle_hooks: None,
        }
    }

    /// Add application-specific metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set a pre-execution hook for semantic tool validation.
    pub fn with_pre_execute_hook(mut self, hook: Arc<dyn ToolPreHook>) -> Self {
        self.pre_execute_hook = Some(hook);
        self
    }

    /// Set lifecycle hooks for granular loop control.
    pub fn with_lifecycle_hooks(mut self, hooks: Arc<dyn AgentLifecycleHooks>) -> Self {
        self.lifecycle_hooks = Some(hooks);
        self
    }
}
