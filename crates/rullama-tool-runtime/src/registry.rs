//! Tool Registry - Composable container for tool definitions
//!
//! Provides a `ToolRegistry` that stores tool definitions and supports
//! deferred loading, category filtering, and search.

use rullama_core::Tool;

/// Tool categories for filtering tools by purpose
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    /// File operation tools.
    FileOps,
    /// Code search tools.
    Search,
    /// Semantic/RAG search tools.
    SemanticSearch,
    /// Git version control tools.
    Git,
    /// Task management tools.
    TaskManager,
    /// Agent pool management tools.
    AgentPool,
    /// Web fetching tools.
    Web,
    /// Web search tools.
    WebSearch,
    /// Shell command execution tools.
    Bash,
    /// Planning tools.
    Planning,
    /// Context recall tools.
    Context,
    /// Script orchestrator tools.
    Orchestrator,
    /// Code execution/interpreter tools.
    CodeExecution,
    /// Session task tools.
    SessionTask,
    /// Validation tools.
    Validation,
}

/// Composable tool registry — stores and queries tool definitions.
///
/// This registry is empty by default; callers compose it by registering
/// tools from whichever modules they need. The umbrella `rullama-tools`
/// crate provides a `registry_with_builtins()` helper that pre-populates
/// one with every concrete builtin (`BashTool`, `FileOpsTool`, ...).
///
/// # Example
/// ```ignore
/// use rullama_tool_runtime::ToolRegistry;
/// // From rullama-tool-builtins (concrete tools + registry helper):
/// // use rullama_tool_builtins::{registry_with_builtins, BashTool};
///
/// let mut registry = ToolRegistry::new();
/// // registry.register_tools(BashTool::get_tools());
/// ```
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    /// Always-available tools — currently just the meta `tool_search`. The
    /// concrete builtins are not in this crate; use
    /// `rullama_tool_builtins::registry_with_builtins()` for a pre-populated
    /// registry, or call [`Self::register_tools`] manually.
    pub fn with_runtime_meta_tools() -> Self {
        let mut registry = Self::new();
        registry.register_tools(crate::ToolSearchTool::get_tools());
        registry
    }

    /// Register a single tool
    pub fn register(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    /// Register multiple tools at once
    pub fn register_tools(&mut self, tools: Vec<Tool>) {
        self.tools.extend(tools);
    }

    /// Get all registered tools
    pub fn get_all(&self) -> &[Tool] {
        &self.tools
    }

    /// Get all tools including additional external tools (e.g., MCP tools)
    pub fn get_all_with_extra(&self, extra: &[Tool]) -> Vec<Tool> {
        let mut all = self.tools.clone();
        all.extend(extra.iter().cloned());
        all
    }

    /// Look up a tool by name
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Get tools that should be loaded initially (defer_loading = false)
    pub fn get_initial_tools(&self) -> Vec<&Tool> {
        self.tools.iter().filter(|t| !t.defer_loading).collect()
    }

    /// Get only deferred tools (defer_loading = true)
    pub fn get_deferred_tools(&self) -> Vec<&Tool> {
        self.tools.iter().filter(|t| t.defer_loading).collect()
    }

    /// Search tools by query string matching name and description
    pub fn search_tools(&self, query: &str) -> Vec<&Tool> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        self.tools
            .iter()
            .filter(|tool| {
                let name_lower = tool.name.to_lowercase();
                let desc_lower = tool.description.to_lowercase();
                query_terms
                    .iter()
                    .any(|term| name_lower.contains(term) || desc_lower.contains(term))
            })
            .collect()
    }

    /// Get tools by category
    pub fn get_by_category(&self, category: ToolCategory) -> Vec<&Tool> {
        let names: &[&str] = match category {
            ToolCategory::FileOps => &[
                "read_file",
                "write_file",
                "edit_file",
                "patch_file",
                "list_directory",
                "search_files",
                "delete_file",
                "create_directory",
            ],
            ToolCategory::Search => &["search_code", "search_files"],
            ToolCategory::SemanticSearch => &[
                "index_codebase",
                "query_codebase",
                "search_with_filters",
                "get_rag_statistics",
                "clear_rag_index",
                "search_git_history",
            ],
            ToolCategory::Git => &[
                "git_status",
                "git_diff",
                "git_log",
                "git_stage",
                "git_unstage",
                "git_commit",
                "git_push",
                "git_pull",
                "git_fetch",
                "git_discard",
                "git_branch",
            ],
            ToolCategory::TaskManager => &[
                "task_create",
                "task_start",
                "task_complete",
                "task_list",
                "task_skip",
                "task_add",
                "task_block",
                "task_depends",
                "task_ready",
                "task_time",
            ],
            ToolCategory::AgentPool => &[
                "agent_spawn",
                "agent_status",
                "agent_list",
                "agent_stop",
                "agent_await",
            ],
            ToolCategory::Web => &["fetch_url"],
            ToolCategory::WebSearch => &["web_search", "web_browse", "web_scrape"],
            ToolCategory::Bash => &["execute_command"],
            ToolCategory::Planning => &["plan_task"],
            ToolCategory::Context => &["recall_context"],
            ToolCategory::Orchestrator => &["execute_script"],
            ToolCategory::CodeExecution => &["execute_code"],
            ToolCategory::SessionTask => &["task_list_write"],
            ToolCategory::Validation => &["check_duplicates", "verify_build", "check_syntax"],
        };

        self.tools
            .iter()
            .filter(|t| names.contains(&t.name.as_str()))
            .collect()
    }

    /// Get all tools including MCP tools
    pub fn get_all_with_mcp(&self, mcp_tools: &[Tool]) -> Vec<Tool> {
        self.get_all_with_extra(mcp_tools)
    }

    /// Core tool names used for basic project exploration. Exposed so callers
    /// can extend the default set with extras from config without forking the
    /// list. Keep alphabetised so the serialised tools array is a stable
    /// prefix — that is what makes the Anthropic prompt cache break points
    /// in `rullama_provider::anthropic` actually land cache hits.
    pub const CORE_TOOL_NAMES: &'static [&'static str] = &[
        "edit_file",
        "execute_command",
        "git_commit",
        "git_diff",
        "git_log",
        "git_stage",
        "git_status",
        "index_codebase",
        "list_directory",
        "query_codebase",
        "read_file",
        "search_code",
        "search_tools",
        "write_file",
    ];

    /// Get core tools for basic project exploration, returned in the
    /// canonical order defined by `CORE_TOOL_NAMES` so the resulting `tools`
    /// array is byte-stable across turns.
    pub fn get_core(&self) -> Vec<&Tool> {
        Self::CORE_TOOL_NAMES
            .iter()
            .filter_map(|name| self.tools.iter().find(|t| t.name == *name))
            .collect()
    }

    /// Get core tools plus any extras named by `extra_names` (deduplicated,
    /// extras appended after core in the order given). Unknown names are
    /// silently skipped.
    pub fn get_core_with_extras(&self, extra_names: &[String]) -> Vec<&Tool> {
        let mut out = self.get_core();
        for name in extra_names {
            if Self::CORE_TOOL_NAMES.contains(&name.as_str()) {
                continue; // already in core
            }
            if let Some(tool) = self.tools.iter().find(|t| t.name == *name) {
                out.push(tool);
            }
        }
        out
    }

    /// Get primary meta-tools (always available)
    pub fn get_primary(&self) -> Vec<&Tool> {
        let primary_names = ["execute_script", "search_tools"];
        self.tools
            .iter()
            .filter(|t| primary_names.contains(&t.name.as_str()))
            .collect()
    }

    /// Search tools by semantic similarity using embeddings.
    ///
    /// Returns tools with their similarity scores, sorted by relevance.
    /// Requires the `rag` feature to be enabled.
    #[cfg(feature = "rag")]
    pub fn semantic_search_tools(
        &self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> anyhow::Result<Vec<(&Tool, f32)>> {
        let tool_pairs: Vec<(String, String)> = self
            .tools
            .iter()
            .map(|t| (t.name.clone(), t.description.clone()))
            .collect();

        let index = crate::tool_embedding::ToolEmbeddingIndex::build(&tool_pairs)?;
        let results = index.search(query, limit, min_score)?;

        Ok(results
            .into_iter()
            .filter_map(|(name, score)| self.get(&name).map(|tool| (tool, score)))
            .collect())
    }

    /// Return a filtered view containing only the named tools.
    ///
    /// Useful when constructing a provider call for a constrained agent role —
    /// the caller supplies the allow-list (e.g. from `AgentRole::allowed_tools`)
    /// and receives only the matching `Tool` definitions.
    ///
    /// Tools not present in the registry are silently skipped, so the list may
    /// be shorter than `allow` if some tools are not registered.
    pub fn filtered_view(&self, allow: &[&str]) -> Vec<Tool> {
        self.tools
            .iter()
            .filter(|t| allow.contains(&t.name.as_str()))
            .cloned()
            .collect()
    }

    /// Total number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rullama_core::ToolInputSchema;
    use std::collections::HashMap;

    fn make_tool(name: &str, defer: bool) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("A {} tool", name),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            requires_approval: false,
            defer_loading: defer,
            ..Default::default()
        }
    }

    #[test]
    fn test_new_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_single() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("test_tool", false));
        assert_eq!(registry.len(), 1);
        assert!(registry.get("test_tool").is_some());
    }

    #[test]
    fn test_register_multiple() {
        let mut registry = ToolRegistry::new();
        registry.register_tools(vec![make_tool("tool1", false), make_tool("tool2", false)]);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_get_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("my_tool", false));

        assert!(registry.get("my_tool").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_get_core_preserves_canonical_order() {
        // Build a registry with the canonical core tools inserted in reverse,
        // to prove get_core() returns them in CORE_TOOL_NAMES order regardless
        // of insertion order — that's what gives the API request body a stable
        // prefix for prompt-cache hits.
        let mut registry = ToolRegistry::new();
        for name in ToolRegistry::CORE_TOOL_NAMES.iter().rev() {
            registry.register(make_tool(name, false));
        }

        let core_names: Vec<&str> = registry
            .get_core()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let expected: Vec<&str> = ToolRegistry::CORE_TOOL_NAMES.to_vec();
        assert_eq!(core_names, expected);
    }

    #[test]
    fn test_get_core_with_extras_appends_unknown_core() {
        let mut registry = ToolRegistry::new();
        for name in ToolRegistry::CORE_TOOL_NAMES {
            registry.register(make_tool(name, false));
        }
        registry.register(make_tool("extra_one", false));
        registry.register(make_tool("extra_two", false));

        // "read_file" is already core — must not duplicate; unknown names
        // silently skipped.
        let extras = vec![
            "extra_one".to_string(),
            "read_file".to_string(),
            "does_not_exist".to_string(),
            "extra_two".to_string(),
        ];
        let names: Vec<&str> = registry
            .get_core_with_extras(&extras)
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        let mut expected: Vec<&str> = ToolRegistry::CORE_TOOL_NAMES.to_vec();
        expected.push("extra_one");
        expected.push("extra_two");
        assert_eq!(names, expected);
    }

    #[test]
    fn test_initial_vs_deferred() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("initial", false));
        registry.register(make_tool("deferred", true));

        assert_eq!(registry.get_initial_tools().len(), 1);
        assert_eq!(registry.get_initial_tools()[0].name, "initial");

        assert_eq!(registry.get_deferred_tools().len(), 1);
        assert_eq!(registry.get_deferred_tools()[0].name, "deferred");
    }

    #[test]
    fn test_search_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Tool {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            ..Default::default()
        });
        registry.register(Tool {
            name: "write_file".to_string(),
            description: "Write content to a file".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            ..Default::default()
        });
        registry.register(Tool {
            name: "execute_command".to_string(),
            description: "Execute a bash command".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            ..Default::default()
        });

        let results = registry.search_tools("file");
        assert_eq!(results.len(), 2);

        let results = registry.search_tools("bash");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_get_all_with_extra() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("builtin", false));

        let extra = vec![make_tool("mcp_tool", false)];
        let all = registry.get_all_with_extra(&extra);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn filtered_view_returns_only_named_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", false));
        registry.register(make_tool("write_file", false));
        registry.register(make_tool("execute_command", false));

        let view = registry.filtered_view(&["read_file", "execute_command"]);
        assert_eq!(view.len(), 2);
        let names: Vec<&str> = view.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"execute_command"));
        assert!(!names.contains(&"write_file"));
    }

    #[test]
    fn filtered_view_unknown_names_are_silently_skipped() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", false));

        // "nonexistent" is not in the registry — must not panic, just ignored
        let view = registry.filtered_view(&["read_file", "nonexistent"]);
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].name, "read_file");
    }

    #[test]
    fn filtered_view_empty_allow_list_returns_empty() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", false));

        let view = registry.filtered_view(&[]);
        assert!(view.is_empty());
    }
}
