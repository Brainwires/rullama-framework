//! Semantic Search Tool - RAG-powered codebase search
//!
//! Provides semantic code search using vector embeddings via the brainwires-rag crate.
//! Supports indexing, querying, filtered search, statistics, and git history search.
//!
//! Requires the `rag` feature flag.

use anyhow::Result;
use brainwires_rag::{
    AdvancedSearchRequest, IndexRequest, QueryRequest, RagClient, SearchGitHistoryRequest,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Global RAG client instance (lazy initialized)
static RAG_CLIENT: OnceCell<Arc<RagClient>> = OnceCell::const_new();

/// Get or initialize the RAG client
async fn get_rag_client() -> Result<Arc<RagClient>> {
    RAG_CLIENT
        .get_or_try_init(|| async {
            let client = RagClient::new().await?;
            Ok(Arc::new(client))
        })
        .await
        .map(Arc::clone)
}

/// Tool definitions and executor for semantic codebase search powered by RAG.
pub struct SemanticSearchTool;

impl SemanticSearchTool {
    /// Get all semantic search tool definitions
    pub fn get_tools() -> Vec<Tool> {
        let mut index_properties = HashMap::new();
        index_properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the codebase directory to index"}),
        );
        index_properties.insert(
            "project".to_string(),
            json!({"type": "string", "description": "Optional project name for multi-project support"}),
        );
        index_properties.insert(
            "include_patterns".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "Optional glob patterns to include", "default": []}),
        );
        index_properties.insert(
            "exclude_patterns".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "Optional glob patterns to exclude", "default": []}),
        );
        index_properties.insert(
            "max_file_size".to_string(),
            json!({"type": "integer", "description": "Maximum file size in bytes to index (default: 1MB)", "default": 1048576}),
        );

        vec![
            Tool {
                name: "index_codebase".to_string(),
                description: "Index a codebase directory for semantic search using embeddings. Automatically performs full or incremental indexing. After indexing, use query_codebase to search.".to_string(),
                input_schema: ToolInputSchema::object(index_properties, vec!["path".to_string()]),
                requires_approval: false,
                defer_loading: true,
                ..Default::default()
            },
            Tool {
                name: "query_codebase".to_string(),
                description: "Search the indexed codebase using semantic search. Returns relevant code chunks ranked by similarity.".to_string(),
                input_schema: {
                    let mut properties = HashMap::new();
                    properties.insert("query".to_string(), json!({"type": "string", "description": "The search query"}));
                    properties.insert("project".to_string(), json!({"type": "string", "description": "Optional project name to filter by"}));
                    properties.insert("limit".to_string(), json!({"type": "integer", "description": "Number of results (default: 10)", "default": 10}));
                    properties.insert("min_score".to_string(), json!({"type": "number", "description": "Minimum similarity score 0-1 (default: 0.7)", "default": 0.7}));
                    properties.insert("hybrid".to_string(), json!({"type": "boolean", "description": "Enable hybrid search (vector + keyword) (default: true)", "default": true}));
                    ToolInputSchema::object(properties, vec!["query".to_string()])
                },
                requires_approval: false,
                defer_loading: true,
                ..Default::default()
            },
            Tool {
                name: "search_with_filters".to_string(),
                description: "Advanced semantic search with filters for file type, language, and path patterns.".to_string(),
                input_schema: {
                    let mut properties = HashMap::new();
                    properties.insert("query".to_string(), json!({"type": "string", "description": "The search query"}));
                    properties.insert("project".to_string(), json!({"type": "string", "description": "Optional project name"}));
                    properties.insert("limit".to_string(), json!({"type": "integer", "description": "Number of results (default: 10)", "default": 10}));
                    properties.insert("min_score".to_string(), json!({"type": "number", "description": "Minimum score (default: 0.7)", "default": 0.7}));
                    properties.insert("file_extensions".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Filter by extensions", "default": []}));
                    properties.insert("languages".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Filter by languages", "default": []}));
                    properties.insert("path_patterns".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Filter by path patterns", "default": []}));
                    ToolInputSchema::object(properties, vec!["query".to_string()])
                },
                requires_approval: false,
                defer_loading: true,
                ..Default::default()
            },
            Tool {
                name: "get_rag_statistics".to_string(),
                description: "Get statistics about the indexed codebase (file counts, chunk counts, languages).".to_string(),
                input_schema: {
                    let mut properties = HashMap::new();
                    properties.insert("project".to_string(), json!({"type": "string", "description": "Optional project name"}));
                    ToolInputSchema::object(properties, vec![])
                },
                requires_approval: false,
                defer_loading: true,
                ..Default::default()
            },
            Tool {
                name: "clear_rag_index".to_string(),
                description: "Clear all indexed data from the vector database. Use before reindexing from scratch.".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: true,
                defer_loading: true,
                ..Default::default()
            },
            Tool {
                name: "search_git_history".to_string(),
                description: "Search git commit history using semantic search with on-demand indexing.".to_string(),
                input_schema: {
                    let mut properties = HashMap::new();
                    properties.insert("query".to_string(), json!({"type": "string", "description": "The search query"}));
                    properties.insert("path".to_string(), json!({"type": "string", "description": "Path to the git repository (default: .)", "default": "."}));
                    properties.insert("project".to_string(), json!({"type": "string", "description": "Optional project name"}));
                    properties.insert("branch".to_string(), json!({"type": "string", "description": "Optional branch name"}));
                    properties.insert("max_commits".to_string(), json!({"type": "integer", "description": "Max commits to index (default: 10)", "default": 10}));
                    properties.insert("limit".to_string(), json!({"type": "integer", "description": "Number of results (default: 10)", "default": 10}));
                    properties.insert("min_score".to_string(), json!({"type": "number", "description": "Minimum score (default: 0.7)", "default": 0.7}));
                    properties.insert("author".to_string(), json!({"type": "string", "description": "Filter by author (regex)"}));
                    properties.insert("since".to_string(), json!({"type": "string", "description": "Filter since date (ISO 8601)"}));
                    properties.insert("until".to_string(), json!({"type": "string", "description": "Filter until date (ISO 8601)"}));
                    properties.insert("file_pattern".to_string(), json!({"type": "string", "description": "Filter by file path pattern (regex)"}));
                    ToolInputSchema::object(properties, vec!["query".to_string()])
                },
                requires_approval: false,
                defer_loading: true,
                ..Default::default()
            },
        ]
    }

    /// Execute a semantic search tool
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "index_codebase" => Self::index_codebase(input).await,
            "query_codebase" => Self::query_codebase(input).await,
            "search_with_filters" => Self::search_with_filters(input).await,
            "get_rag_statistics" => Self::get_statistics(input).await,
            "clear_rag_index" => Self::clear_index(input).await,
            "search_git_history" => Self::search_git_history(input).await,
            _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
        };

        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Semantic search operation failed: {}", e),
            ),
        }
    }

    async fn index_codebase(input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?
            .to_string();

        let request = IndexRequest {
            path,
            project: input["project"].as_str().map(|s| s.to_string()),
            include_patterns: input["include_patterns"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            exclude_patterns: input["exclude_patterns"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            max_file_size: input["max_file_size"].as_u64().unwrap_or(1_048_576) as usize,
        };

        let response = client.index_codebase(request).await?;

        Ok(format!(
            "Indexed {} files, {} chunks in {}ms (mode: {:?})",
            response.files_indexed, response.chunks_created, response.duration_ms, response.mode
        ))
    }

    async fn query_codebase(input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let request = QueryRequest {
            query: input["query"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'query' parameter"))?
                .to_string(),
            path: input["path"].as_str().map(|s| s.to_string()),
            project: input["project"].as_str().map(|s| s.to_string()),
            limit: input["limit"].as_u64().unwrap_or(10) as usize,
            min_score: input["min_score"].as_f64().unwrap_or(0.7) as f32,
            hybrid: input["hybrid"].as_bool().unwrap_or(true),
        };

        let response = client.query_codebase(request).await?;

        let mut output = format!("Found {} results:\n\n", response.results.len());
        for (i, result) in response.results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} (score: {:.3})\n",
                i + 1,
                result.file_path,
                result.score
            ));
            output.push_str(&format!(
                "   Lines {}-{}\n",
                result.start_line, result.end_line
            ));
            output.push_str(&format!(
                "   {}\n\n",
                result.content.lines().next().unwrap_or("")
            ));
        }

        Ok(output)
    }

    async fn search_with_filters(input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let request = AdvancedSearchRequest {
            query: input["query"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'query' parameter"))?
                .to_string(),
            path: input["path"].as_str().map(|s| s.to_string()),
            project: input["project"].as_str().map(|s| s.to_string()),
            limit: input["limit"].as_u64().unwrap_or(10) as usize,
            min_score: input["min_score"].as_f64().unwrap_or(0.7) as f32,
            file_extensions: input["file_extensions"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            languages: input["languages"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            path_patterns: input["path_patterns"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
        };

        let response = client.search_with_filters(request).await?;

        let mut output = format!("Found {} filtered results:\n\n", response.results.len());
        for (i, result) in response.results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} (score: {:.3})\n",
                i + 1,
                result.file_path,
                result.score
            ));
            output.push_str(&format!("   Language: {}\n", result.language));
            output.push_str(&format!(
                "   Lines {}-{}\n\n",
                result.start_line, result.end_line
            ));
        }

        Ok(output)
    }

    async fn get_statistics(_input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let response = client.get_statistics().await?;

        let mut output = String::from("RAG Index Statistics:\n");
        output.push_str(&format!("  Total chunks: {}\n", response.total_chunks));
        output.push_str(&format!("  Total files: {}\n\n", response.total_files));

        if !response.language_breakdown.is_empty() {
            output.push_str("Languages:\n");
            for lang_stat in &response.language_breakdown {
                output.push_str(&format!(
                    "  {}: {} files, {} chunks\n",
                    lang_stat.language, lang_stat.file_count, lang_stat.chunk_count
                ));
            }
        }

        Ok(output)
    }

    async fn clear_index(_input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let response = client.clear_index().await?;

        Ok(format!("Cleared index: {}", response.message))
    }

    async fn search_git_history(input: &Value) -> Result<String> {
        let client = get_rag_client().await?;

        let request = SearchGitHistoryRequest {
            query: input["query"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing 'query' parameter"))?
                .to_string(),
            path: input["path"].as_str().unwrap_or(".").to_string(),
            project: input["project"].as_str().map(|s| s.to_string()),
            branch: input["branch"].as_str().map(|s| s.to_string()),
            max_commits: input["max_commits"].as_u64().unwrap_or(10) as usize,
            limit: input["limit"].as_u64().unwrap_or(10) as usize,
            min_score: input["min_score"].as_f64().unwrap_or(0.7) as f32,
            author: input["author"].as_str().map(|s| s.to_string()),
            since: input["since"].as_str().map(|s| s.to_string()),
            until: input["until"].as_str().map(|s| s.to_string()),
            file_pattern: input["file_pattern"].as_str().map(|s| s.to_string()),
        };

        let response = client.search_git_history(request).await?;

        let mut output = format!("Found {} commits:\n\n", response.results.len());
        for (i, result) in response.results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} (score: {:.3})\n",
                i + 1,
                &result.commit_hash[..8],
                result.score
            ));
            output.push_str(&format!("   Author: {}\n", result.author));
            output.push_str(&format!("   Date: {}\n", result.commit_date));
            output.push_str(&format!("   Message: {}\n\n", result.commit_message));
        }

        Ok(output)
    }

    // ============ Helper methods for orchestrator integration ============

    /// Execute a query against the indexed codebase (for orchestrator)
    pub async fn execute_query(
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<String, String> {
        let input = json!({
            "query": query,
            "limit": limit,
            "min_score": min_score,
            "hybrid": true
        });
        Self::query_codebase(&input)
            .await
            .map_err(|e| e.to_string())
    }

    /// Index a codebase directory (for orchestrator)
    pub async fn execute_index(path: &str) -> Result<String, String> {
        let input = json!({ "path": path });
        Self::index_codebase(&input)
            .await
            .map_err(|e| e.to_string())
    }

    /// Execute filtered search (for orchestrator)
    pub async fn execute_filtered_search(input: &Value) -> Result<String, String> {
        Self::search_with_filters(input)
            .await
            .map_err(|e| e.to_string())
    }

    /// Get RAG statistics (for orchestrator)
    pub async fn execute_get_stats() -> Result<String, String> {
        Self::get_statistics(&Value::Null)
            .await
            .map_err(|e| e.to_string())
    }

    /// Search git history (for orchestrator)
    pub async fn execute_git_history_search(input: &Value) -> Result<String, String> {
        Self::search_git_history(input)
            .await
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = SemanticSearchTool::get_tools();
        assert_eq!(tools.len(), 6);

        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"index_codebase"));
        assert!(tool_names.contains(&"query_codebase"));
        assert!(tool_names.contains(&"search_with_filters"));
        assert!(tool_names.contains(&"get_rag_statistics"));
        assert!(tool_names.contains(&"clear_rag_index"));
        assert!(tool_names.contains(&"search_git_history"));
    }

    #[test]
    fn test_index_codebase_tool_definition() {
        let tools = SemanticSearchTool::get_tools();
        let index_tool = tools.iter().find(|t| t.name == "index_codebase").unwrap();

        assert!(index_tool.description.contains("Index"));
        assert!(index_tool.description.contains("semantic"));
        assert!(!index_tool.requires_approval);
        assert!(index_tool.defer_loading);
    }

    #[test]
    fn test_query_codebase_tool_definition() {
        let tools = SemanticSearchTool::get_tools();
        let query_tool = tools.iter().find(|t| t.name == "query_codebase").unwrap();

        assert!(query_tool.description.contains("Search"));
        assert!(!query_tool.requires_approval);
    }

    #[test]
    fn test_clear_index_requires_approval() {
        let tools = SemanticSearchTool::get_tools();
        let clear_tool = tools.iter().find(|t| t.name == "clear_rag_index").unwrap();
        assert!(clear_tool.requires_approval);
    }

    #[test]
    fn test_all_tools_have_descriptions() {
        let tools = SemanticSearchTool::get_tools();
        for tool in tools {
            assert!(
                !tool.description.is_empty(),
                "Tool {} should have a description",
                tool.name
            );
        }
    }

    #[test]
    fn test_index_codebase_required_fields() {
        let tools = SemanticSearchTool::get_tools();
        let index_tool = tools.iter().find(|t| t.name == "index_codebase").unwrap();

        if let Some(ref required) = index_tool.input_schema.required {
            assert!(required.contains(&"path".to_string()));
        }
    }
}
