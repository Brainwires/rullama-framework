//! Tool Search - Meta-tool for discovering available tools dynamically

use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::ToolRegistry;
use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Search mode for tool discovery
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Keyword-based search (default).
    #[default]
    Keyword,
    /// Regex-based search.
    Regex,
    /// Semantic embedding-based search (requires `rag` feature).
    Semantic,
}

/// Meta-tool for discovering available tools dynamically.
pub struct ToolSearchTool;

impl ToolSearchTool {
    /// Return tool definitions for tool search.
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::search_tools_tool()]
    }

    fn search_tools_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "query".to_string(),
            json!({"type": "string", "description": "Search query to find relevant tools"}),
        );
        properties.insert("mode".to_string(), json!({"type": "string", "enum": ["keyword", "regex", "semantic"], "description": "Search mode: keyword (substring match), regex (pattern match), or semantic (embedding similarity, requires rag feature)", "default": "keyword"}));
        properties.insert(
            "include_deferred".to_string(),
            json!({"type": "boolean", "description": "Include deferred tools", "default": true}),
        );
        properties.insert(
            "limit".to_string(),
            json!({"type": "integer", "description": "Maximum number of results to return (semantic mode only)", "default": 10}),
        );
        properties.insert(
            "min_score".to_string(),
            json!({"type": "number", "description": "Minimum similarity score 0.0-1.0 (semantic mode only)", "default": 0.3}),
        );
        Tool {
            name: "search_tools".to_string(),
            description: "Search for available tools by name or description.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["query".to_string()]),
            requires_approval: false,
            defer_loading: false,
            ..Default::default()
        }
    }

    /// Execute the tool search tool by name.
    #[tracing::instrument(
        name = "tool.execute",
        skip(input, _context, registry),
        fields(tool_name)
    )]
    pub fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &ToolContext,
        registry: &ToolRegistry,
    ) -> ToolResult {
        let result = match tool_name {
            "search_tools" => Self::search_tools(input, registry),
            _ => Err(anyhow::anyhow!("Unknown tool search tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Tool search failed: {}", e),
            ),
        }
    }

    fn search_tools(input: &Value, registry: &ToolRegistry) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        #[allow(dead_code)] // limit and min_score are used only with the `rag` feature
        struct Input {
            query: String,
            #[serde(default)]
            mode: SearchMode,
            #[serde(default = "dt")]
            include_deferred: bool,
            #[serde(default = "default_limit")]
            limit: usize,
            #[serde(default = "default_min_score")]
            min_score: f32,
        }
        fn dt() -> bool {
            true
        }
        fn default_limit() -> usize {
            10
        }
        fn default_min_score() -> f32 {
            0.3
        }

        let params: Input = serde_json::from_value(input.clone())?;

        // Handle semantic mode separately
        #[cfg(feature = "rag")]
        if params.mode == SearchMode::Semantic {
            return Self::search_tools_semantic(
                &params.query,
                registry,
                params.include_deferred,
                params.limit,
                params.min_score,
            );
        }

        #[cfg(not(feature = "rag"))]
        if params.mode == SearchMode::Semantic {
            return Err(anyhow::anyhow!(
                "Semantic search mode requires the 'rag' feature to be enabled. Use 'keyword' or 'regex' mode instead."
            ));
        }

        if params.mode == SearchMode::Regex && params.query.len() > 200 {
            return Err(anyhow::anyhow!(
                "Regex pattern exceeds maximum length of 200 characters (got {})",
                params.query.len()
            ));
        }

        let regex =
            if params.mode == SearchMode::Regex {
                Some(Regex::new(&params.query).map_err(|e| {
                    anyhow::anyhow!("Invalid regex pattern '{}': {}", params.query, e)
                })?)
            } else {
                None
            };

        let query_lower = params.query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let matching_tools: Vec<&Tool> = registry
            .get_all()
            .iter()
            .filter(|tool| {
                if tool.defer_loading && !params.include_deferred {
                    return false;
                }
                let search_text = format!("{} {}", tool.name, tool.description);
                match &regex {
                    Some(re) => re.is_match(&search_text),
                    None => {
                        let name_lower = tool.name.to_lowercase();
                        let desc_lower = tool.description.to_lowercase();
                        query_terms
                            .iter()
                            .any(|term| name_lower.contains(term) || desc_lower.contains(term))
                    }
                }
            })
            .collect();

        if matching_tools.is_empty() {
            return Ok(format!(
                "No tools found matching query: \"{}\"",
                params.query
            ));
        }

        let mut result = format!(
            "Found {} tools matching \"{}\":\n\n",
            matching_tools.len(),
            params.query
        );
        for tool in matching_tools {
            Self::format_tool(&mut result, tool, None);
        }
        Ok(result)
    }

    /// Format a single tool entry for output.
    fn format_tool(result: &mut String, tool: &Tool, score: Option<f32>) {
        result.push_str(&format!("## {}\n", tool.name));
        if let Some(s) = score {
            result.push_str(&format!("**Similarity:** {:.2}\n", s));
        }
        result.push_str(&format!("**Description:** {}\n", tool.description));
        if let Some(props) = &tool.input_schema.properties {
            result.push_str("**Parameters:**\n");
            for (name, schema) in props {
                let desc = schema
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No description");
                let ptype = schema
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                result.push_str(&format!("  - `{}` ({}): {}\n", name, ptype, desc));
            }
        }
        result.push('\n');
    }

    /// Semantic search using embedding similarity.
    #[cfg(feature = "rag")]
    fn search_tools_semantic(
        query: &str,
        registry: &ToolRegistry,
        include_deferred: bool,
        limit: usize,
        min_score: f32,
    ) -> anyhow::Result<String> {
        use crate::tool_embedding::ToolEmbeddingIndex;
        use std::sync::OnceLock;

        // Cache the embedding index; rebuild if tool count changes.
        static CACHED_INDEX: OnceLock<(usize, ToolEmbeddingIndex)> = OnceLock::new();

        let tools: Vec<&Tool> = registry
            .get_all()
            .iter()
            .filter(|t| include_deferred || !t.defer_loading)
            .collect();

        // Build tool pairs for embedding
        let tool_pairs: Vec<(String, String)> = tools
            .iter()
            .map(|t| (t.name.clone(), t.description.clone()))
            .collect();

        // Use cached index if tool count hasn't changed, otherwise build new one.
        // OnceLock means first call builds, subsequent calls reuse.
        // If tools change (e.g., MCP tools added), the count won't match and we
        // fall through to building a fresh index.
        let index = CACHED_INDEX.get_or_init(|| {
            let idx = ToolEmbeddingIndex::build(&tool_pairs)
                .expect("Failed to build tool embedding index");
            (tool_pairs.len(), idx)
        });

        // If tool count changed, we need a fresh index but can't replace OnceLock.
        // In that case, build an ad-hoc index.
        let search_results = if index.0 != tool_pairs.len() {
            let fresh_index = ToolEmbeddingIndex::build(&tool_pairs)?;
            fresh_index.search(query, limit, min_score)?
        } else {
            index.1.search(query, limit, min_score)?
        };

        if search_results.is_empty() {
            return Ok(format!(
                "No tools found semantically matching query: \"{}\" (min_score: {:.2})",
                query, min_score
            ));
        }

        let mut result = format!(
            "Found {} tools semantically matching \"{}\":\n\n",
            search_results.len(),
            query
        );

        for (tool_name, score) in &search_results {
            if let Some(tool) = registry.get(tool_name) {
                Self::format_tool(&mut result, tool, Some(*score));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = ToolSearchTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search_tools");
    }

    #[test]
    fn test_search_mode_default() {
        let mode = SearchMode::default();
        assert_eq!(mode, SearchMode::Keyword);
    }
}
