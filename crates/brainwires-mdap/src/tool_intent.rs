//! Tool Intent Types for MDAP Microagents
//!
//! This module defines structured types for expressing tool intent in microagent outputs.
//! Instead of executing tools directly (which would break stateless execution and voting),
//! microagents express their intent to call tools, which are then executed after voting
//! consensus is achieved.
//!
//! # Design Rationale
//!
//! The MAKER paper's guarantees require:
//! - Stateless microagent execution
//! - Deterministic outputs for voting consensus
//! - No side effects during the voting loop
//!
//! By separating intent (deterministic) from execution (non-deterministic), we preserve
//! these guarantees while enabling practical tool use.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::microagent::SubtaskOutput;

/// Schema describing a tool's interface for intent expression
///
/// This is a simplified schema for describing tools to microagents.
/// It contains just enough information for the LLM to express intent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// Parameter descriptions (name -> description)
    #[serde(default)]
    pub parameters: HashMap<String, String>,
    /// Required parameters
    #[serde(default)]
    pub required: Vec<String>,
    /// Tool category
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<ToolCategory>,
}

impl From<brainwires_core::Tool> for ToolSchema {
    fn from(tool: brainwires_core::Tool) -> Self {
        let mut schema = Self::new(&tool.name, &tool.description);

        // Extract parameters from input_schema
        if let Some(props) = &tool.input_schema.properties {
            for (name, value) in props {
                let desc = value
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No description")
                    .to_string();
                schema.parameters.insert(name.clone(), desc);
            }
        }

        if let Some(required) = &tool.input_schema.required {
            schema.required = required.clone();
        }

        schema
    }
}

impl ToolSchema {
    /// Create a new tool schema
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: HashMap::new(),
            required: Vec::new(),
            category: None,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.parameters.insert(name.into(), description.into());
        self
    }

    /// Add a required parameter
    pub fn with_required_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let name = name.into();
        self.parameters.insert(name.clone(), description.into());
        self.required.push(name);
        self
    }

    /// Set category
    pub fn with_category(mut self, category: ToolCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Format as a string for inclusion in prompts
    pub fn to_prompt_format(&self) -> String {
        let mut result = format!("- **{}**: {}\n", self.name, self.description);
        if !self.parameters.is_empty() {
            result.push_str("  Parameters:\n");
            for (name, desc) in &self.parameters {
                let required = if self.required.contains(name) {
                    " (required)"
                } else {
                    ""
                };
                result.push_str(&format!("    - {}{}: {}\n", name, required, desc));
            }
        }
        result
    }
}

/// A tool call intent that can be voted on
///
/// This represents what tool the microagent wants to call, without actually
/// executing it. The intent is deterministic and can be compared for voting.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolIntent {
    /// Tool name to call (e.g., "read_file", "search_files")
    pub tool_name: String,
    /// Tool arguments as JSON
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Why this tool is needed (for debugging/logging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

impl ToolIntent {
    /// Create a new tool intent
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            rationale: None,
        }
    }

    /// Create a tool intent with rationale
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }

    /// Check if this intent matches a tool category
    pub fn matches_category(&self, category: &ToolCategory) -> bool {
        category.contains_tool(&self.tool_name)
    }
}

/// Extended subtask output that may include tool intent
///
/// When a microagent needs to use a tool, it outputs this structure
/// with both the regular output and the tool intent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtaskOutputWithIntent {
    /// Base subtask output (the regular output)
    pub output: SubtaskOutput,
    /// Optional tool intent (if the subtask needs a tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_intent: Option<ToolIntent>,
    /// Whether the output is complete or waiting for tool result
    #[serde(default)]
    pub awaiting_tool_result: bool,
}

impl SubtaskOutputWithIntent {
    /// Create from a regular subtask output (no tool intent)
    pub fn from_output(output: SubtaskOutput) -> Self {
        Self {
            output,
            tool_intent: None,
            awaiting_tool_result: false,
        }
    }

    /// Create with a tool intent
    pub fn with_tool_intent(output: SubtaskOutput, intent: ToolIntent) -> Self {
        Self {
            output,
            tool_intent: Some(intent),
            awaiting_tool_result: true,
        }
    }

    /// Check if this output has a pending tool intent
    pub fn has_tool_intent(&self) -> bool {
        self.tool_intent.is_some()
    }

    /// Mark the tool result as received
    pub fn mark_tool_complete(mut self) -> Self {
        self.awaiting_tool_result = false;
        self
    }
}

/// Tool categories for permission control
///
/// These categories group tools by their risk level and side effects.
/// Microagents are restricted to read-only categories by default.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Read files (read_file, etc.)
    FileRead,
    /// Write files (write_file, edit_file, etc.)
    FileWrite,
    /// Search operations (search_files, grep, etc.)
    Search,
    /// Semantic/RAG search
    SemanticSearch,
    /// Shell command execution
    Bash,
    /// Git operations
    Git,
    /// Web requests
    Web,
    /// Agent spawning
    AgentPool,
    /// Task management
    TaskManager,
    /// MCP tools (dynamic)
    Mcp,
    /// Custom category
    Custom(String),
}

impl ToolCategory {
    /// Check if a tool name belongs to this category
    pub fn contains_tool(&self, tool_name: &str) -> bool {
        match self {
            ToolCategory::FileRead => {
                matches!(tool_name, "read_file" | "file_read" | "get_file_contents")
            }
            ToolCategory::FileWrite => matches!(
                tool_name,
                "write_file" | "edit_file" | "delete_file" | "create_directory" | "file_write"
            ),
            ToolCategory::Search => matches!(
                tool_name,
                "search_files" | "grep" | "find_files" | "glob" | "file_search"
            ),
            ToolCategory::SemanticSearch => matches!(
                tool_name,
                "semantic_search" | "query_codebase" | "rag_search"
            ),
            ToolCategory::Bash => matches!(
                tool_name,
                "bash" | "execute_command" | "shell" | "run_command"
            ),
            ToolCategory::Git => matches!(
                tool_name,
                "git" | "git_status" | "git_diff" | "git_commit" | "git_log"
            ),
            ToolCategory::Web => matches!(
                tool_name,
                "web_search" | "fetch_url" | "browse" | "http_request"
            ),
            ToolCategory::AgentPool => {
                matches!(tool_name, "spawn_agent" | "agent_pool" | "create_agent")
            }
            ToolCategory::TaskManager => {
                matches!(tool_name, "create_task" | "update_task" | "task_manager")
            }
            ToolCategory::Mcp => tool_name.starts_with("mcp_") || tool_name.starts_with("mcp__"),
            ToolCategory::Custom(prefix) => tool_name.starts_with(prefix),
        }
    }

    /// Get all read-only categories (safe for microagents)
    pub fn read_only_categories() -> HashSet<ToolCategory> {
        HashSet::from([
            ToolCategory::FileRead,
            ToolCategory::Search,
            ToolCategory::SemanticSearch,
        ])
    }

    /// Get all categories that produce side effects
    pub fn side_effect_categories() -> HashSet<ToolCategory> {
        HashSet::from([
            ToolCategory::FileWrite,
            ToolCategory::Bash,
            ToolCategory::Git,
            ToolCategory::Web,
            ToolCategory::AgentPool,
            ToolCategory::TaskManager,
        ])
    }
}

/// Result of parsing tool intent from microagent output
#[derive(Clone, Debug)]
pub enum IntentParseResult {
    /// No tool intent found (regular output)
    NoIntent(SubtaskOutput),
    /// Tool intent found and parsed
    WithIntent(SubtaskOutputWithIntent),
    /// Failed to parse intent
    ParseError(String),
}

/// Parse tool intent from a microagent's text response
///
/// Looks for JSON blocks containing tool_intent fields.
pub fn parse_tool_intent(subtask_id: &str, response_text: &str) -> IntentParseResult {
    // Try to find JSON block with tool_intent
    if let Some(intent) = extract_tool_intent_json(response_text) {
        match serde_json::from_value::<ToolIntent>(intent.clone()) {
            Ok(tool_intent) => {
                // Extract the non-tool output (everything except the JSON block)
                let output_text = remove_json_block(response_text);
                let output = SubtaskOutput::new(
                    subtask_id,
                    serde_json::json!({
                        "text": output_text.trim(),
                        "awaiting_tool": true,
                    }),
                );
                IntentParseResult::WithIntent(SubtaskOutputWithIntent::with_tool_intent(
                    output,
                    tool_intent,
                ))
            }
            Err(e) => IntentParseResult::ParseError(format!("Failed to parse tool intent: {}", e)),
        }
    } else {
        // No tool intent, regular output
        let output = SubtaskOutput::new(subtask_id, serde_json::json!({ "text": response_text }));
        IntentParseResult::NoIntent(output)
    }
}

/// Extract tool_intent JSON from response text
fn extract_tool_intent_json(text: &str) -> Option<serde_json::Value> {
    // Look for ```json blocks first
    if let Some(json_block) = extract_json_code_block(text)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_block)
    {
        if value.get("tool_intent").is_some() {
            return value.get("tool_intent").cloned();
        }
        // Check if the whole block is a tool intent
        if value.get("tool_name").is_some() {
            return Some(value);
        }
    }

    // Look for inline JSON with tool_intent
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('{')
            && trimmed.ends_with('}')
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
        {
            if value.get("tool_intent").is_some() {
                return value.get("tool_intent").cloned();
            }
            if value.get("tool_name").is_some() {
                return Some(value);
            }
        }
    }

    None
}

/// Extract JSON code block from markdown
fn extract_json_code_block(text: &str) -> Option<String> {
    let start_markers = ["```json", "```JSON"];
    let end_marker = "```";

    for start in start_markers {
        if let Some(start_idx) = text.find(start) {
            let content_start = start_idx + start.len();
            if let Some(end_idx) = text[content_start..].find(end_marker) {
                return Some(
                    text[content_start..content_start + end_idx]
                        .trim()
                        .to_string(),
                );
            }
        }
    }

    None
}

/// Remove JSON block from text, leaving other content
fn remove_json_block(text: &str) -> String {
    let start_markers = ["```json", "```JSON"];
    let end_marker = "```";

    let mut result = text.to_string();

    for start in start_markers {
        if let Some(start_idx) = result.find(start) {
            let content_start = start_idx + start.len();
            if let Some(end_idx) = result[content_start..].find(end_marker) {
                let block_end = content_start + end_idx + end_marker.len();
                result = format!("{}{}", &result[..start_idx], &result[block_end..]);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_intent_creation() {
        let intent = ToolIntent::new("read_file", serde_json::json!({"path": "/test.txt"}))
            .with_rationale("Need to read configuration");

        assert_eq!(intent.tool_name, "read_file");
        assert_eq!(
            intent.rationale,
            Some("Need to read configuration".to_string())
        );
    }

    #[test]
    fn test_tool_category_matching() {
        assert!(ToolCategory::FileRead.contains_tool("read_file"));
        assert!(ToolCategory::FileWrite.contains_tool("write_file"));
        assert!(ToolCategory::Search.contains_tool("grep"));
        assert!(ToolCategory::Mcp.contains_tool("mcp__brainwires-rag__query"));
        assert!(!ToolCategory::FileRead.contains_tool("bash"));
    }

    #[test]
    fn test_parse_tool_intent_with_json_block() {
        let response = r#"I need to read a file first.

```json
{
    "tool_name": "read_file",
    "arguments": {"path": "/test.txt"},
    "rationale": "Check contents"
}
```
"#;

        match parse_tool_intent("task-1", response) {
            IntentParseResult::WithIntent(output) => {
                assert!(output.has_tool_intent());
                let intent = output.tool_intent.unwrap();
                assert_eq!(intent.tool_name, "read_file");
            }
            _ => panic!("Expected WithIntent result"),
        }
    }

    #[test]
    fn test_parse_no_intent() {
        let response = "This is just a regular response without any tool calls.";

        match parse_tool_intent("task-1", response) {
            IntentParseResult::NoIntent(output) => {
                assert_eq!(output.subtask_id, "task-1");
            }
            _ => panic!("Expected NoIntent result"),
        }
    }

    #[test]
    fn test_read_only_categories() {
        let read_only = ToolCategory::read_only_categories();
        assert!(read_only.contains(&ToolCategory::FileRead));
        assert!(read_only.contains(&ToolCategory::Search));
        assert!(!read_only.contains(&ToolCategory::FileWrite));
        assert!(!read_only.contains(&ToolCategory::Bash));
    }
}
