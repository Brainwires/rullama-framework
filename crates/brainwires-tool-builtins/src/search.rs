use anyhow::Result;
use ignore::WalkBuilder;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Regex-based code pattern search tool
pub struct SearchTool;

impl SearchTool {
    /// Return tool definitions for code search.
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::search_code_tool()]
    }

    fn search_code_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "pattern".to_string(),
            json!({"type": "string", "description": "Regex pattern to search for"}),
        );
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to search in", "default": "."}),
        );
        Tool {
            name: "search_code".to_string(),
            description: "Search for code patterns in files using regex.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["pattern".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    /// Execute a search tool by name.
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "search_code" => Self::search_code(input, context),
            _ => Err(anyhow::anyhow!("Unknown search tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(tool_use_id.to_string(), format!("Search failed: {}", e)),
        }
    }

    fn search_code(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            pattern: String,
            #[serde(default = "default_path")]
            path: String,
        }
        fn default_path() -> String {
            ".".to_string()
        }

        let params: Input = serde_json::from_value(input.clone())?;
        let regex = Regex::new(&params.pattern)?;
        let search_path = if params.path == "." {
            &context.working_directory
        } else {
            &params.path
        };

        let mut matches = Vec::new();
        for entry in WalkBuilder::new(search_path).build() {
            let entry = entry?;
            if entry.path().is_file()
                && let Ok(content) = fs::read_to_string(entry.path())
            {
                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        matches.push(format!(
                            "{}:{} - {}",
                            entry.path().display(),
                            line_num + 1,
                            line.trim()
                        ));
                        if matches.len() >= 100 {
                            break;
                        }
                    }
                }
            }
        }
        Ok(format!(
            "Search Results:\nPattern: {}\nMatches: {}\n\n{}",
            params.pattern,
            matches.len(),
            matches.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> ToolContext {
        ToolContext {
            working_directory: std::env::current_dir()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_tools() {
        let tools = SearchTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search_code");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let context = create_test_context();
        let input = json!({"pattern": "test"});
        let result = SearchTool::execute("1", "unknown_tool", &input, &context);
        assert!(result.is_error);
    }
}
