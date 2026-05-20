//! Tool Orchestrator - Rhai-based tool orchestration for AI agents
//!
//! Implements Anthropic's "Programmatic Tool Calling" pattern for token-efficient
//! tool orchestration. Instead of sequential tool calls, AI writes Rhai scripts
//! that orchestrate multiple tools, returning only the final result.
//!
//! ## Features
//!
//! This module supports multiple build targets via feature flags:
//!
//! - **`orchestrator`** (default for native) - Thread-safe Rust library with `Arc`/`Mutex`
//! - **`orchestrator-wasm`** - WebAssembly bindings for browser/Node.js via `wasm-bindgen`
//!
//! ## Benefits
//!
//! - **37% token reduction** - intermediate results don't pollute context
//! - **Parallel execution** - multiple tools in one pass
//! - **Complex orchestration** - loops, conditionals, data processing

// When both features are active (e.g. --all-features), native `orchestrator` takes priority.

pub mod engine;
pub mod sandbox;
pub mod types;

pub use engine::{ToolExecutor, ToolOrchestrator, dynamic_to_json};
pub use sandbox::{
    // Default limit constants
    DEFAULT_MAX_ARRAY_SIZE,
    DEFAULT_MAX_MAP_SIZE,
    DEFAULT_MAX_OPERATIONS,
    DEFAULT_MAX_STRING_SIZE,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_TIMEOUT_MS,
    // Profile constants
    EXTENDED_MAX_OPERATIONS,
    EXTENDED_MAX_TOOL_CALLS,
    EXTENDED_TIMEOUT_MS,
    ExecutionLimits,
    QUICK_MAX_OPERATIONS,
    QUICK_MAX_TOOL_CALLS,
    QUICK_TIMEOUT_MS,
};
pub use types::{OrchestratorError, OrchestratorResult, ToolCall};

// ── OrchestratorTool wrapper ───────────────────────────────────────────────
//
// High-level tool wrapper that integrates the Rhai orchestrator with the
// brainwires tool system.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Orchestrator tool for executing Rhai scripts with access to registered tools
pub struct OrchestratorTool {
    /// The underlying Rhai orchestrator
    orchestrator: Arc<RwLock<ToolOrchestrator>>,
}

impl OrchestratorTool {
    /// Create a new OrchestratorTool
    pub fn new() -> Self {
        Self {
            orchestrator: Arc::new(RwLock::new(ToolOrchestrator::new())),
        }
    }

    /// Get the tool definition for execute_script
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::execute_script_tool()]
    }

    fn execute_script_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "script".to_string(),
            json!({
                "type": "string",
                "description": "The Rhai script to execute. Use registered tool functions (e.g., read_file(path), search_code(pattern)) and return a final value."
            }),
        );
        properties.insert(
            "max_operations".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum operations allowed (default: 100000)",
                "default": 100000
            }),
        );
        properties.insert(
            "max_tool_calls".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum tool calls allowed (default: 50)",
                "default": 50
            }),
        );
        properties.insert(
            "timeout_ms".to_string(),
            json!({
                "type": "integer",
                "description": "Timeout in milliseconds (default: 30000)",
                "default": 30000
            }),
        );

        Tool {
            name: "execute_script".to_string(),
            description:
                r#"PRIMARY TOOL: Execute a Rhai script for programmatic tool orchestration.

This is the preferred way to interact with tools. Write Rhai scripts to orchestrate
multiple tool calls efficiently, with intermediate results staying out of the context window.

Benefits:
- 37% token reduction vs sequential tool calls
- Loops, conditionals, and data transformation
- Batch operations in a single execution
- Only final result enters context

Available tools can be discovered via `search_tools`. All tools are callable as functions.

Rhai Syntax Quick Reference:
- Variables: `let x = 42;`
- Strings: `let s = "hello";` or template: `` `Hello ${name}` ``
- Arrays: `let arr = [1, 2, 3];`
- Objects: `let obj = #{ key: "value" };`
- Loops: `for item in items { ... }`
- Conditionals: `if condition { ... } else { ... }`

Example - Find and count TODOs:
```rhai
let files = list_directory("src");
let count = 0;
for file in files {
    if file.ends_with(".rs") {
        let content = read_file(file);
        count += content.matches("TODO").len();
    }
}
`Found ${count} TODO comments`
```"#
                    .to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["script".to_string()]),
            requires_approval: false,
            defer_loading: false,
            ..Default::default()
        }
    }

    /// Register a tool executor function
    pub async fn register_executor<F>(&self, name: impl Into<String>, executor: F)
    where
        F: Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    {
        let mut orchestrator = self.orchestrator.write().await;
        orchestrator.register_executor(name, executor);
    }

    /// Execute the orchestrator tool
    pub async fn execute(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "execute_script" => self.execute_script(input).await,
            _ => Err(anyhow::anyhow!("Unknown orchestrator tool: {}", tool_name)),
        };

        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Script execution failed: {}", e),
            ),
        }
    }

    async fn execute_script(&self, input: &Value) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        struct Input {
            script: String,
            #[serde(default = "default_max_ops")]
            max_operations: u64,
            #[serde(default = "default_max_calls")]
            max_tool_calls: usize,
            #[serde(default = "default_timeout")]
            timeout_ms: u64,
        }

        fn default_max_ops() -> u64 {
            100_000
        }
        fn default_max_calls() -> usize {
            50
        }
        fn default_timeout() -> u64 {
            30_000
        }

        let params: Input = serde_json::from_value(input.clone())?;

        let limits = ExecutionLimits::default()
            .with_max_operations(params.max_operations)
            .with_max_tool_calls(params.max_tool_calls)
            .with_timeout_ms(params.timeout_ms);

        let orchestrator = self.orchestrator.read().await;
        let result = orchestrator.execute(&params.script, limits)?;

        if result.success {
            let mut output = result.output;
            if !result.tool_calls.is_empty() {
                output.push_str(&format!(
                    "\n\n--- Script executed {} tool call(s) in {}ms ---",
                    result.tool_calls.len(),
                    result.execution_time_ms
                ));
            }
            Ok(output)
        } else {
            Err(anyhow::anyhow!(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }

    /// Get the underlying orchestrator for direct access
    pub fn orchestrator(&self) -> Arc<RwLock<ToolOrchestrator>> {
        Arc::clone(&self.orchestrator)
    }
}

impl Default for OrchestratorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = OrchestratorTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "execute_script");
        assert!(!tools[0].requires_approval);
        assert!(!tools[0].defer_loading);
    }

    #[test]
    fn test_execute_script_tool_definition() {
        let tool = OrchestratorTool::execute_script_tool();
        assert_eq!(tool.name, "execute_script");
        assert!(tool.description.contains("Rhai"));
        assert!(tool.description.contains("programmatic"));
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator_tool = OrchestratorTool::new();
        let orchestrator = orchestrator_tool.orchestrator.read().await;
        assert!(orchestrator.registered_tools().is_empty());
    }

    #[tokio::test]
    async fn test_register_executor() {
        let orchestrator_tool = OrchestratorTool::new();
        orchestrator_tool
            .register_executor("test_tool", |_| Ok("success".to_string()))
            .await;

        let orchestrator = orchestrator_tool.orchestrator.read().await;
        assert!(orchestrator.registered_tools().contains(&"test_tool"));
    }

    #[tokio::test]
    async fn test_execute_simple_script() {
        let orchestrator_tool = OrchestratorTool::new();
        let context = ToolContext::default();

        let input = json!({
            "script": "let x = 1 + 2; x"
        });

        let result = orchestrator_tool
            .execute("test-id", "execute_script", &input, &context)
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("3"));
    }

    #[tokio::test]
    async fn test_execute_with_tool() {
        let orchestrator_tool = OrchestratorTool::new();
        orchestrator_tool
            .register_executor("greet", |input| {
                let name = input.as_str().unwrap_or("world");
                Ok(format!("Hello, {}!", name))
            })
            .await;

        let context = ToolContext::default();
        let input = json!({
            "script": r#"greet("Claude")"#
        });

        let result = orchestrator_tool
            .execute("test-id", "execute_script", &input, &context)
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("Hello, Claude!"));
    }

    #[tokio::test]
    async fn test_execute_unknown_tool_name() {
        let orchestrator_tool = OrchestratorTool::new();
        let context = ToolContext::default();

        let result = orchestrator_tool
            .execute("test-id", "unknown_tool", &json!({}), &context)
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown orchestrator tool"));
    }
}
