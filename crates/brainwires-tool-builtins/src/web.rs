use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Web fetching tool implementation
pub struct WebTool;

impl WebTool {
    /// Return tool definitions for web operations.
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::fetch_url_tool()]
    }

    fn fetch_url_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "url".to_string(),
            json!({"type": "string", "description": "URL to fetch"}),
        );
        Tool {
            name: "fetch_url".to_string(),
            description: "Fetch content from a URL on the internet.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["url".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    /// Execute a web tool by name.
    #[tracing::instrument(name = "tool.execute", skip(input, _context), fields(tool_name))]
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "fetch_url" => Self::fetch_url(input).await,
            _ => Err(anyhow::anyhow!("Unknown web tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Web operation failed: {}", e),
            ),
        }
    }

    async fn fetch_url(input: &Value) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            url: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let client = Client::new();
        let response = client.get(&params.url).send().await?;
        let text = response.text().await?;
        Ok(format!(
            "URL: {}\nContent length: {} bytes\n\n{}",
            params.url,
            text.len(),
            text
        ))
    }

    /// Fetch URL content (helper for orchestrator integration)
    pub async fn fetch_url_content(url: &str) -> Result<String> {
        let client = Client::new();
        let response = client.get(url).send().await?;
        let text = response.text().await?;
        Ok(format!(
            "URL: {}\nContent length: {} bytes\n\n{}",
            url,
            text.len(),
            text
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> ToolContext {
        ToolContext {
            working_directory: ".".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_tools() {
        let tools = WebTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "fetch_url");
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let context = create_test_context();
        let input = json!({"url": "https://example.com"});
        let result = WebTool::execute("1", "unknown_tool", &input, &context).await;
        assert!(result.is_error);
    }
}
