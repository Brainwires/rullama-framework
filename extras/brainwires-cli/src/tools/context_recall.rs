use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

use crate::storage::{CachedEmbeddingProvider, LanceDatabase, MessageStore, VectorDatabase};
use crate::types::tool::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Tool for recalling context from conversation history
///
/// This tool allows agents to search through the full conversation history
/// for specific details that may have been compacted out of the current
/// context window. Uses semantic search with embeddings.
pub struct ContextRecallTool;

impl ContextRecallTool {
    /// Get the recall_context tool definition
    pub fn get_tools() -> Vec<Tool> {
        let mut properties = HashMap::new();

        properties.insert(
            "query".to_string(),
            json!({
                "type": "string",
                "description": "What to search for in conversation history (e.g., 'authentication discussion', 'database schema decisions')"
            }),
        );

        properties.insert(
            "limit".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5
            }),
        );

        properties.insert(
            "min_score".to_string(),
            json!({
                "type": "number",
                "description": "Minimum relevance score from 0.0 to 1.0 (default: 0.6)",
                "default": 0.6
            }),
        );

        properties.insert(
            "cross_conversation".to_string(),
            json!({
                "type": "boolean",
                "description": "Search across all conversations, not just the current one (default: false)",
                "default": false
            }),
        );

        vec![Tool {
            name: "recall_context".to_string(),
            description: "Search through the full conversation history for specific details, \
                         decisions, or context that may have been compacted out of the current \
                         context window. Use this when you need to recall earlier discussion \
                         details, decisions made, code snippets discussed, or any information \
                         from earlier in the conversation that is no longer in the active context."
                .to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["query".to_string()]),
            requires_approval: false,
            defer_loading: true, // Context recall is deferred
            ..Default::default()
        }]
    }

    /// Execute the recall_context tool
    pub async fn execute(
        tool_use_id: &str,
        _tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return ToolResult::error(
                    tool_use_id.to_string(),
                    "Missing required 'query' parameter".to_string(),
                );
            }
        };

        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let min_score = input
            .get("min_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6) as f32;

        let cross_conversation = input
            .get("cross_conversation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Get conversation_id from context metadata
        let conversation_id = context.metadata.get("conversation_id").map(|s| s.as_str());

        match Self::search_history(query, limit, min_score, cross_conversation, conversation_id)
            .await
        {
            Ok(results) => {
                if results.is_empty() {
                    ToolResult::success(
                        tool_use_id.to_string(),
                        format!(
                            "No relevant context found for query: '{}'\n\nTry:\n- Using different keywords\n- Lowering the min_score threshold\n- Setting cross_conversation: true to search all conversations",
                            query
                        ),
                    )
                } else {
                    let mut output = format!("Found {} relevant messages:\n\n", results.len());

                    for (i, (message, score)) in results.iter().enumerate() {
                        output.push_str(&format!(
                            "{}. [Score: {:.2}] [{}]\n",
                            i + 1,
                            score,
                            message.role
                        ));

                        // Truncate content if too long
                        let content = if message.content.len() > 500 {
                            format!("{}...", &message.content[..500])
                        } else {
                            message.content.clone()
                        };

                        output.push_str(&format!("   {}\n\n", content));
                    }

                    ToolResult::success(tool_use_id.to_string(), output)
                }
            }
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Failed to search conversation history: {}", e),
            ),
        }
    }

    /// Search conversation history using semantic search
    async fn search_history(
        query: &str,
        limit: usize,
        min_score: f32,
        cross_conversation: bool,
        conversation_id: Option<&str>,
    ) -> anyhow::Result<Vec<(crate::storage::MessageMetadata, f32)>> {
        use crate::utils::paths::PlatformPaths;

        // Get the database path using XDG-compliant paths
        let db_path = PlatformPaths::conversations_db_path()?;

        // Ensure the data directory exists
        PlatformPaths::ensure_data_dir()?;

        let client: Arc<LanceDatabase> =
            Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await?);
        let embeddings: Arc<CachedEmbeddingProvider> = Arc::new(CachedEmbeddingProvider::new()?);

        // Initialize tables if needed
        client.initialize(embeddings.dimension()).await?;

        let message_store = MessageStore::new(client, embeddings);

        // Search based on scope
        if cross_conversation || conversation_id.is_none() {
            // Search across all conversations
            message_store.search(query, limit, min_score).await
        } else if let Some(conv_id) = conversation_id {
            // Search within current conversation only
            message_store
                .search_conversation(conv_id, query, limit, min_score)
                .await
        } else {
            message_store.search(query, limit, min_score).await
        }
    }

    // ============ Helper method for orchestrator integration ============

    /// Execute context recall search (for orchestrator)
    pub async fn execute_recall(query: &str) -> Result<String, String> {
        // Default to searching across all conversations with default limits
        let results = Self::search_history(query, 5, 0.6, true, None)
            .await
            .map_err(|e| e.to_string())?;

        if results.is_empty() {
            return Ok(format!("No relevant context found for query: '{}'", query));
        }

        let mut output = format!("Found {} relevant messages:\n\n", results.len());
        for (i, (message, score)) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. [Score: {:.2}] [{}]\n",
                i + 1,
                score,
                message.role
            ));

            // Truncate content if too long
            let content = if message.content.len() > 500 {
                format!("{}...", &message.content[..500])
            } else {
                message.content.clone()
            };

            output.push_str(&format!("   {}\n\n", content));
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = ContextRecallTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "recall_context");
    }

    #[test]
    fn test_tool_definition() {
        let tools = ContextRecallTool::get_tools();
        let tool = &tools[0];

        assert!(tool.description.contains("conversation history"));
        assert!(!tool.requires_approval);

        // Check required parameters
        if let Some(required) = &tool.input_schema.required {
            assert!(required.contains(&"query".to_string()));
        }
    }

    #[test]
    fn test_tool_has_all_properties() {
        let tools = ContextRecallTool::get_tools();
        let tool = &tools[0];

        if let Some(props) = &tool.input_schema.properties {
            assert!(props.contains_key("query"));
            assert!(props.contains_key("limit"));
            assert!(props.contains_key("min_score"));
            assert!(props.contains_key("cross_conversation"));
        }
    }
}
