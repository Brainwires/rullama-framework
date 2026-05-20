//! Built-in tool executor — single dispatch point for all framework tools.
//!
//! [`BuiltinToolExecutor`] eliminates the need for every consumer (agent-chat,
//! gateway, etc.) to reimplement a `dispatch_tool()` match statement.
//! Construct one with a [`ToolRegistry`] and a [`ToolContext`], then call
//! [`execute`](BuiltinToolExecutor::execute) or use it through the
//! [`ToolExecutor`] trait.

use anyhow::Result;
use async_trait::async_trait;

use brainwires_core::{Tool, ToolContext, ToolResult, ToolUse};

use brainwires_tool_runtime::{ToolExecutor, ToolRegistry, ToolSearchTool};

/// Concrete executor that dispatches tool calls to the built-in tool modules
/// registered in a [`ToolRegistry`].
///
/// # Example
///
/// ```rust,ignore
/// use brainwires_tool_builtins::BuiltinToolExecutor;
/// use brainwires_tool_runtime::ToolRegistry;
/// use brainwires_core::ToolContext;
///
/// let registry = brainwires_tool_builtins::registry_with_builtins();
/// let context = ToolContext::default();
/// let executor = BuiltinToolExecutor::new(registry, context);
///
/// // Check available tools
/// assert!(executor.has_tool("execute_command"));
///
/// // Execute via the ToolExecutor trait
/// // let result = executor.execute(&tool_use, &context).await?;
/// ```
pub struct BuiltinToolExecutor {
    registry: ToolRegistry,
    context: ToolContext,
}

impl BuiltinToolExecutor {
    /// Create a new executor backed by the given registry and default context.
    pub fn new(registry: ToolRegistry, context: ToolContext) -> Self {
        Self { registry, context }
    }

    /// Execute a tool by name, dispatching to the correct handler.
    ///
    /// This is the standalone entry-point that mirrors the old
    /// `dispatch_tool()` function from agent-chat.
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        tool_use_id: &str,
        input: &serde_json::Value,
    ) -> ToolResult {
        self.dispatch(tool_use_id, tool_name, input, &self.context)
            .await
    }

    /// Get all tool definitions (for sending to the provider).
    pub fn tools(&self) -> Vec<Tool> {
        self.registry.get_all().to_vec()
    }

    /// Check if a tool exists in the registry.
    pub fn has_tool(&self, name: &str) -> bool {
        self.registry.get(name).is_some()
    }

    /// Return a reference to the underlying registry.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Return a reference to the default context.
    pub fn context(&self) -> &ToolContext {
        &self.context
    }

    /// Core dispatch logic — routes a tool call to the correct handler module.
    async fn dispatch(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        input: &serde_json::Value,
        context: &ToolContext,
    ) -> ToolResult {
        // Always-available tools
        if tool_name == "search_tools" {
            return ToolSearchTool::execute(tool_use_id, tool_name, input, context, &self.registry);
        }

        // Native-only tools
        #[cfg(feature = "native")]
        {
            match tool_name {
                // Bash / shell execution
                "bash" | "execute_command" => {
                    return crate::BashTool::execute(tool_use_id, tool_name, input, context);
                }

                // File operations
                "read_file" | "write_file" | "edit_file" | "patch_file" | "list_directory"
                | "delete_file" | "create_directory" | "file_search" => {
                    return crate::FileOpsTool::execute(tool_use_id, tool_name, input, context);
                }

                // Git operations
                "git_status" | "git_diff" | "git_log" | "git_stage" | "git_commit" | "git_push"
                | "git_pull" | "git_branch" | "git_checkout" | "git_stash" | "git_reset"
                | "git_show" | "git_blame" => {
                    return crate::GitTool::execute(tool_use_id, tool_name, input, context);
                }

                // Code / file search
                "search_code" | "search_files" => {
                    return crate::SearchTool::execute(tool_use_id, tool_name, input, context);
                }

                // Validation
                "check_duplicates" | "verify_build" | "check_syntax" => {
                    return brainwires_tool_runtime::ValidationTool::execute(
                        tool_use_id,
                        tool_name,
                        input,
                        context,
                    )
                    .await;
                }

                // Web fetching
                "fetch_url" => {
                    return crate::WebTool::execute(tool_use_id, tool_name, input, context).await;
                }

                _ => {}
            }
        }

        // Feature-gated: orchestrator
        #[cfg(any(feature = "orchestrator", feature = "orchestrator-wasm"))]
        {
            if tool_name == "execute_script" {
                let orchestrator = brainwires_tool_runtime::OrchestratorTool::new();
                return orchestrator
                    .execute(tool_use_id, tool_name, input, context)
                    .await;
            }
        }

        // Feature-gated: code execution / interpreters
        #[cfg(feature = "interpreters")]
        {
            if tool_name == "execute_code" {
                return crate::CodeExecTool::execute(tool_use_id, tool_name, input, context).await;
            }
        }

        // Feature-gated: semantic search / RAG
        #[cfg(feature = "rag")]
        {
            match tool_name {
                "index_codebase"
                | "query_codebase"
                | "search_with_filters"
                | "get_rag_statistics"
                | "clear_rag_index"
                | "search_git_history" => {
                    return crate::SemanticSearchTool::execute(
                        tool_use_id,
                        tool_name,
                        input,
                        context,
                    )
                    .await;
                }
                _ => {}
            }
        }

        // Feature-gated: browser automation via Thalora subprocess
        #[cfg(feature = "browser")]
        {
            match tool_name {
                "browser_read_url" | "browser_navigate" | "browser_click" | "browser_fill"
                | "browser_eval" | "browser_screenshot" | "browser_search" => {
                    return crate::BrowserTool::execute(tool_use_id, tool_name, input, context)
                        .await;
                }
                _ => {}
            }
        }

        // Unknown tool — return an error result
        ToolResult::error(
            tool_use_id.to_string(),
            format!("Unknown tool: {tool_name}"),
        )
    }
}

#[async_trait]
impl ToolExecutor for BuiltinToolExecutor {
    async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult> {
        Ok(self
            .dispatch(&tool_use.id, &tool_use.name, &tool_use.input, context)
            .await)
    }

    fn available_tools(&self) -> Vec<Tool> {
        self.tools()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ToolInputSchema;
    use std::collections::HashMap;

    fn make_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("A {} tool", name),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            ..Default::default()
        }
    }

    fn make_executor_with(names: &[&str]) -> BuiltinToolExecutor {
        let mut registry = ToolRegistry::new();
        for name in names {
            registry.register(make_tool(name));
        }
        let context = ToolContext::default();
        BuiltinToolExecutor::new(registry, context)
    }

    #[test]
    fn test_new_creates_successfully() {
        let executor = make_executor_with(&["read_file", "execute_command"]);
        assert_eq!(executor.tools().len(), 2);
    }

    #[test]
    fn test_has_tool_returns_true_for_registered() {
        let executor = make_executor_with(&["read_file", "execute_command"]);
        assert!(executor.has_tool("read_file"));
        assert!(executor.has_tool("execute_command"));
    }

    #[test]
    fn test_has_tool_returns_false_for_unknown() {
        let executor = make_executor_with(&["read_file"]);
        assert!(!executor.has_tool("nonexistent_tool"));
        assert!(!executor.has_tool(""));
    }

    #[test]
    fn test_tools_returns_registered_tools() {
        let executor = make_executor_with(&["read_file", "write_file", "git_status"]);
        let tools = executor.tools();
        assert_eq!(tools.len(), 3);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"git_status"));
    }

    #[test]
    fn test_available_tools_matches_tools() {
        let executor = make_executor_with(&["read_file", "execute_command"]);
        let tools = executor.tools();
        let available = executor.available_tools();
        assert_eq!(tools.len(), available.len());
    }

    #[tokio::test]
    async fn test_unknown_tool_returns_error() {
        let executor = make_executor_with(&["read_file"]);
        let result = executor
            .execute_tool("totally_fake_tool", "test-id-1", &serde_json::json!({}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown tool"));
        assert!(result.content.contains("totally_fake_tool"));
    }

    #[tokio::test]
    async fn test_unknown_tool_via_trait() {
        let executor = make_executor_with(&["read_file"]);
        let tool_use = ToolUse {
            id: "test-id-2".to_string(),
            name: "nonexistent".to_string(),
            input: serde_json::json!({}),
        };
        let result = executor
            .execute(&tool_use, &executor.context)
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Unknown tool"));
    }

    #[test]
    fn test_empty_registry() {
        let executor = make_executor_with(&[]);
        assert_eq!(executor.tools().len(), 0);
        assert!(!executor.has_tool("anything"));
    }

    #[test]
    fn test_with_builtins_registry() {
        let registry = crate::registry_with_builtins();
        let tool_count = registry.len();
        let context = ToolContext::default();
        let executor = BuiltinToolExecutor::new(registry, context);
        assert_eq!(executor.tools().len(), tool_count);
        // search_tools is always available
        assert!(executor.has_tool("search_tools"));
    }
}
