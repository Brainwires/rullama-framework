//! Agent Management Tools for MCP
//!
//! Provides MCP tools for spawning and managing task agents

use brainwires_core::{Tool, ToolInputSchema};
use serde_json::json;
use std::collections::HashMap;

/// Registry of agent management tools
pub struct AgentToolRegistry {
    tools: Vec<Tool>,
}

impl AgentToolRegistry {
    /// Create a new agent tool registry
    pub fn new() -> Self {
        let tools = vec![
            Tool {
                name: "agent_spawn".to_string(),
                description: "Spawn a new task agent to work on a subtask autonomously. \
                              The agent will execute the task in the background and report results. \
                              Useful for breaking down large workloads hierarchically."
                    .to_string(),
                input_schema: ToolInputSchema::object(
                    {
                        let mut props = HashMap::new();
                        props.insert("description".to_string(), json!({
                            "type": "string",
                            "description": "Description of the task for the agent to execute"
                        }));
                        props.insert("working_directory".to_string(), json!({
                            "type": "string",
                            "description": "Optional working directory for file operations. If not specified, uses the MCP server's current directory."
                        }));
                        props.insert("max_iterations".to_string(), json!({
                            "type": "integer",
                            "description": "Optional maximum number of iterations before the agent stops (default: 100). Set lower for simple tasks or higher for complex ones."
                        }));
                        props.insert("enable_validation".to_string(), json!({
                            "type": "boolean",
                            "description": "Enable automatic validation checks before completion (default: true). Validates syntax, duplicates, and build success."
                        }));
                        props.insert("build_type".to_string(), json!({
                            "type": "string",
                            "enum": ["npm", "cargo", "typescript"],
                            "description": "Optional build type for validation (npm, cargo, or typescript). If specified, agent must pass build before completing."
                        }));
                        props.insert("enable_mdap".to_string(), json!({
                            "type": "boolean",
                            "description": "Enable MDAP (Massively Decomposed Agentic Processes) for zero-error execution through task decomposition and multi-agent voting (default: false)"
                        }));
                        props.insert("mdap_k".to_string(), json!({
                            "type": "integer",
                            "description": "Vote margin threshold for MDAP (default: 3). Higher values = more reliability but higher cost. Range: 1-7."
                        }));
                        props.insert("mdap_target_success".to_string(), json!({
                            "type": "number",
                            "description": "Target success rate for MDAP (default: 0.95). Range: 0.90-0.99."
                        }));
                        props.insert("mdap_preset".to_string(), json!({
                            "type": "string",
                            "enum": ["default", "high_reliability", "cost_optimized"],
                            "description": "MDAP preset: 'default' (k=3, 95%), 'high_reliability' (k=5, 99%), 'cost_optimized' (k=2, 90%)"
                        }));
                        props
                    },
                    vec!["description".to_string()],
                ),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_list".to_string(),
                description: "List all currently running task agents and their status"
                    .to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_status".to_string(),
                description: "Get detailed status of a specific task agent".to_string(),
                input_schema: ToolInputSchema::object(
                    {
                        let mut props = HashMap::new();
                        props.insert("agent_id".to_string(), json!({
                            "type": "string",
                            "description": "ID of the agent to query"
                        }));
                        props
                    },
                    vec!["agent_id".to_string()],
                ),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_stop".to_string(),
                description: "Stop a running task agent".to_string(),
                input_schema: ToolInputSchema::object(
                    {
                        let mut props = HashMap::new();
                        props.insert("agent_id".to_string(), json!({
                            "type": "string",
                            "description": "ID of the agent to stop"
                        }));
                        props
                    },
                    vec!["agent_id".to_string()],
                ),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_await".to_string(),
                description: "Wait for a task agent to complete and return its result. \
                              Unlike agent_status which returns immediately, this tool blocks \
                              until the agent finishes (completes or fails) and returns the final result."
                    .to_string(),
                input_schema: ToolInputSchema::object(
                    {
                        let mut props = HashMap::new();
                        props.insert("agent_id".to_string(), json!({
                            "type": "string",
                            "description": "ID of the agent to wait for"
                        }));
                        props.insert("timeout_secs".to_string(), json!({
                            "type": "integer",
                            "description": "Optional timeout in seconds. If not provided, waits indefinitely."
                        }));
                        props
                    },
                    vec!["agent_id".to_string()],
                ),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_pool_stats".to_string(),
                description: "Get statistics about the agent pool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "agent_file_locks".to_string(),
                description: "List all currently held file locks by agents".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            // Self-improvement tools
            Tool {
                name: "self_improve_start".to_string(),
                description: "Start an autonomous self-improvement loop that analyzes the codebase \
                              and spawns agents to fix issues (clippy warnings, TODOs, missing docs, \
                              dead code, test gaps, code smells)"
                    .to_string(),
                input_schema: ToolInputSchema::object(
                    {
                        let mut props = HashMap::new();
                        props.insert("max_cycles".to_string(), json!({
                            "type": "integer",
                            "description": "Maximum number of improvement cycles (default: 10)"
                        }));
                        props.insert("max_budget".to_string(), json!({
                            "type": "number",
                            "description": "Maximum budget in dollars (default: 10.0)"
                        }));
                        props.insert("dry_run".to_string(), json!({
                            "type": "boolean",
                            "description": "List tasks without executing (default: false)"
                        }));
                        props.insert("strategies".to_string(), json!({
                            "type": "string",
                            "description": "Comma-separated list of strategies: clippy,todo_scanner,doc_gaps,test_coverage,refactoring,dead_code (empty = all)"
                        }));
                        props.insert("no_bridge".to_string(), json!({
                            "type": "boolean",
                            "description": "Disable MCP bridge execution path (default: false)"
                        }));
                        props.insert("no_direct".to_string(), json!({
                            "type": "boolean",
                            "description": "Disable direct agent execution path (default: false)"
                        }));
                        props
                    },
                    vec![],
                ),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "self_improve_status".to_string(),
                description: "Get the status of a running self-improvement session"
                    .to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "self_improve_stop".to_string(),
                description: "Stop a running self-improvement session"
                    .to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
        ];

        Self { tools }
    }

    /// Get all agent management tools
    pub fn get_tools(&self) -> &[Tool] {
        &self.tools
    }
}

impl Default for AgentToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AgentToolRegistry::new();
        assert_eq!(registry.get_tools().len(), 10, "Should have 10 agent tools");
    }

    #[test]
    fn test_default_creation() {
        let registry = AgentToolRegistry::default();
        assert_eq!(registry.get_tools().len(), 10);
    }

    #[test]
    fn test_agent_spawn_tool() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        let spawn_tool = tools
            .iter()
            .find(|t| t.name == "agent_spawn")
            .expect("agent_spawn tool should exist");

        assert_eq!(spawn_tool.name, "agent_spawn");
        assert!(spawn_tool.description.contains("autonomous"));
        assert!(!spawn_tool.requires_approval);

        // Check schema structure
        assert_eq!(spawn_tool.input_schema.schema_type, "object");
        assert!(spawn_tool.input_schema.properties.is_some());
        let props = spawn_tool.input_schema.properties.as_ref().unwrap();
        assert!(props.contains_key("description"));

        assert!(spawn_tool.input_schema.required.is_some());
        let required = spawn_tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"description".to_string()));
    }

    #[test]
    fn test_agent_list_tool() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        let list_tool = tools
            .iter()
            .find(|t| t.name == "agent_list")
            .expect("agent_list tool should exist");

        assert_eq!(list_tool.name, "agent_list");
        assert!(list_tool.description.contains("running"));
        assert!(!list_tool.requires_approval);

        // Should have empty properties
        assert_eq!(list_tool.input_schema.schema_type, "object");
        let props = list_tool.input_schema.properties.as_ref().unwrap();
        assert!(props.is_empty());
    }

    #[test]
    fn test_agent_status_tool() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        let status_tool = tools
            .iter()
            .find(|t| t.name == "agent_status")
            .expect("agent_status tool should exist");

        assert_eq!(status_tool.name, "agent_status");
        assert!(status_tool.description.contains("status"));
        assert!(!status_tool.requires_approval);

        // Should require agent_id parameter
        let props = status_tool.input_schema.properties.as_ref().unwrap();
        assert!(props.contains_key("agent_id"));

        let required = status_tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"agent_id".to_string()));
    }

    #[test]
    fn test_agent_stop_tool() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        let stop_tool = tools
            .iter()
            .find(|t| t.name == "agent_stop")
            .expect("agent_stop tool should exist");

        assert_eq!(stop_tool.name, "agent_stop");
        assert!(stop_tool.description.contains("Stop"));
        assert!(!stop_tool.requires_approval);

        // Should require agent_id parameter
        let props = stop_tool.input_schema.properties.as_ref().unwrap();
        assert!(props.contains_key("agent_id"));

        let required = stop_tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"agent_id".to_string()));
    }

    #[test]
    fn test_agent_await_tool() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        let await_tool = tools
            .iter()
            .find(|t| t.name == "agent_await")
            .expect("agent_await tool should exist");

        assert_eq!(await_tool.name, "agent_await");
        assert!(await_tool.description.contains("Wait"));
        assert!(await_tool.description.contains("complete"));
        assert!(!await_tool.requires_approval);

        // Should require agent_id parameter
        let props = await_tool.input_schema.properties.as_ref().unwrap();
        assert!(props.contains_key("agent_id"));
        assert!(props.contains_key("timeout_secs"));

        let required = await_tool.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"agent_id".to_string()));
        // timeout_secs is optional
        assert!(!required.contains(&"timeout_secs".to_string()));
    }

    #[test]
    fn test_all_tools_have_descriptions() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        for tool in tools {
            assert!(
                !tool.description.is_empty(),
                "Tool {} should have a description",
                tool.name
            );
        }
    }

    #[test]
    fn test_all_tools_have_object_schemas() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        for tool in tools {
            assert_eq!(
                tool.input_schema.schema_type, "object",
                "Tool {} should have object schema",
                tool.name
            );
        }
    }

    #[test]
    fn test_tool_names_are_prefixed() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        for tool in tools {
            assert!(
                tool.name.starts_with("agent_") || tool.name.starts_with("self_improve_"),
                "Tool {} should be prefixed with 'agent_' or 'self_improve_'",
                tool.name
            );
        }
    }

    #[test]
    fn test_no_approval_required() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        for tool in tools {
            assert!(
                !tool.requires_approval,
                "Tool {} should not require approval for autonomous operation",
                tool.name
            );
        }
    }

    #[test]
    fn test_schema_serialization() {
        let registry = AgentToolRegistry::new();
        let tools = registry.get_tools();

        for tool in tools {
            let serialized = serde_json::to_value(&tool.input_schema);
            assert!(
                serialized.is_ok(),
                "Tool {} schema should serialize to JSON",
                tool.name
            );

            let value = serialized.unwrap();
            assert!(value.is_object());
            assert_eq!(value["type"], "object");
        }
    }
}
