//! Tool Executor trait
//!
//! Defines the [`ToolExecutor`] trait for abstracted tool execution.
//! Framework crates like `brainwires-agent` depend on this trait
//! to call tools without coupling to any concrete implementation.

use anyhow::Result;
use async_trait::async_trait;

use brainwires_core::{Tool, ToolContext, ToolResult, ToolUse};

/// Trait for executing tools in an agent context.
///
/// Implement this on your tool executor to integrate with framework agents
/// like `TaskAgent`. The trait is object-safe and can be used as
/// `Arc<dyn ToolExecutor>`.
///
/// # Example
///
/// ```rust,ignore
/// use brainwires_tool_runtime::ToolExecutor;
/// use brainwires_core::{Tool, ToolContext, ToolResult, ToolUse};
/// use async_trait::async_trait;
///
/// struct MyExecutor;
///
/// #[async_trait]
/// impl ToolExecutor for MyExecutor {
///     async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> anyhow::Result<ToolResult> {
///         Ok(ToolResult::success(tool_use.id.clone(), "done".to_string()))
///     }
///
///     fn available_tools(&self) -> Vec<Tool> {
///         vec![]
///     }
/// }
/// ```
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool and return its result.
    ///
    /// The `tool_use` contains the tool name and input parameters.
    /// The `context` provides working directory and execution metadata.
    async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult>;

    /// Return the list of tools available for the AI to invoke.
    fn available_tools(&self) -> Vec<Tool>;
}

/// Decision returned by a [`ToolPreHook`] before a tool call.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PreHookDecision {
    /// Allow the tool call to proceed normally.
    Allow,
    /// Reject the call; inject this message as `ToolResult::error`.
    Reject(String),
}

/// Pluggable pre-execution hook for semantic tool validation.
///
/// Implement this to intercept tool calls before execution and validate
/// call intent against current agent state (not just JSON schema).
/// Hook is set via `AgentContext::with_pre_execute_hook()`.
#[async_trait]
pub trait ToolPreHook: Send + Sync {
    /// Called before tool execution to validate or reject the call.
    async fn before_execute(
        &self,
        tool_use: &ToolUse,
        context: &ToolContext,
    ) -> Result<PreHookDecision>;
}
