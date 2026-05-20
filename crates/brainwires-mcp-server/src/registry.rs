use anyhow::Result;
use async_trait::async_trait;
use brainwires_mcp_client::CallToolResult;
use serde_json::Value;

use crate::connection::RequestContext;
use crate::error::AgentNetworkError;

/// Definition of an MCP tool.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for tool input.
    pub input_schema: Value,
}

/// Trait for tool execution handlers.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given arguments.
    async fn call(&self, args: Value, ctx: &RequestContext) -> Result<CallToolResult>;
}

struct RegisteredTool {
    def: McpToolDef,
    handler: Box<dyn ToolHandler>,
}

/// Registry of MCP tools with their handlers.
pub struct McpToolRegistry {
    tools: Vec<RegisteredTool>,
}

impl McpToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Register a tool with its handler.
    pub fn register(
        &mut self,
        name: &str,
        description: &str,
        input_schema: Value,
        handler: impl ToolHandler + 'static,
    ) {
        self.tools.push(RegisteredTool {
            def: McpToolDef {
                name: name.to_string(),
                description: description.to_string(),
                input_schema,
            },
            handler: Box::new(handler),
        });
    }

    /// List all registered tool definitions (by reference).
    pub fn list_tools(&self) -> Vec<&McpToolDef> {
        self.tools.iter().map(|t| &t.def).collect()
    }

    /// List all registered tool definitions (cloned).
    pub fn list_tool_defs(&self) -> Vec<McpToolDef> {
        self.tools.iter().map(|t| t.def.clone()).collect()
    }

    /// Dispatch a tool call to its registered handler.
    pub async fn dispatch(
        &self,
        name: &str,
        args: Value,
        ctx: &RequestContext,
    ) -> Result<CallToolResult> {
        for tool in &self.tools {
            if tool.def.name == name {
                return tool.handler.call(args, ctx).await;
            }
        }
        Err(AgentNetworkError::ToolNotFound(name.to_string()).into())
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.def.name == name)
    }
}

impl Default for McpToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoHandler;

    #[async_trait]
    impl ToolHandler for EchoHandler {
        async fn call(&self, _args: Value, _ctx: &RequestContext) -> Result<CallToolResult> {
            Ok(CallToolResult::success(vec![]))
        }
    }

    #[test]
    fn test_registry_register_and_list() {
        let mut registry = McpToolRegistry::new();
        registry.register("echo", "Echo tool", json!({"type": "object"}), EchoHandler);

        let tools = registry.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
    }

    #[test]
    fn test_registry_has_tool() {
        let mut registry = McpToolRegistry::new();
        registry.register("test", "Test tool", json!({"type": "object"}), EchoHandler);

        assert!(registry.has_tool("test"));
        assert!(!registry.has_tool("nonexistent"));
    }
}
