//! Smart Tool Router
//!
//! Analyzes user queries to determine which tool categories are relevant.
//! Uses pure keyword-based pattern matching (no AI/inference dependencies).
//!
//! For inference-enhanced routing, use the CLI's `smart_router` wrapper which
//! adds `LocalRouter` integration on top of these functions.

use crate::{ToolCategory, ToolRegistry};
use rullama_core::{Message, Tool};
use std::collections::HashSet;

/// Keyword patterns for each tool category
struct CategoryPatterns {
    category: ToolCategory,
    keywords: &'static [&'static str],
}

const CATEGORY_PATTERNS: &[CategoryPatterns] = &[
    CategoryPatterns {
        category: ToolCategory::FileOps,
        keywords: &[
            "file",
            "read",
            "write",
            "edit",
            "create",
            "delete",
            "directory",
            "folder",
            "list",
            "content",
            "save",
            "open",
            "path",
            "rename",
            "move",
            "copy",
            "mkdir",
            "touch",
            "cat",
            "ls",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Search,
        keywords: &[
            "search",
            "find",
            "grep",
            "look",
            "where",
            "locate",
            "pattern",
            "match",
            "regex",
            "occurrence",
            "rg",
            "ripgrep",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::SemanticSearch,
        keywords: &[
            "semantic",
            "meaning",
            "similar",
            "related",
            "understand",
            "codebase",
            "index",
            "rag",
            "embedding",
            "concept",
            "query",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Git,
        keywords: &[
            "git",
            "commit",
            "diff",
            "branch",
            "merge",
            "push",
            "pull",
            "clone",
            "status",
            "log",
            "history",
            "version",
            "repository",
            "repo",
            "checkout",
            "stash",
            "rebase",
            "cherry-pick",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::TaskManager,
        keywords: &[
            "task",
            "todo",
            "progress",
            "complete",
            "pending",
            "assign",
            "track",
            "subtask",
            "dependency",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::AgentPool,
        keywords: &[
            "agent",
            "spawn",
            "parallel",
            "concurrent",
            "worker",
            "pool",
            "background",
            "async",
            "thread",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Web,
        keywords: &[
            "url", "fetch", "http", "api", "endpoint", "request", "download", "curl", "get", "post",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::WebSearch,
        keywords: &[
            "web",
            "search",
            "google",
            "browse",
            "scrape",
            "internet",
            "online",
            "website",
            "page",
            "html",
            "duckduckgo",
            "bing",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Bash,
        keywords: &[
            "run", "execute", "command", "shell", "bash", "terminal", "script", "npm", "cargo",
            "pip", "make", "build", "install", "test", "compile", "yarn", "pnpm", "docker",
            "kubectl",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Planning,
        keywords: &[
            "plan",
            "design",
            "architect",
            "strategy",
            "approach",
            "implement",
            "roadmap",
            "outline",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Context,
        keywords: &[
            "context",
            "remember",
            "recall",
            "previous",
            "earlier",
            "mentioned",
            "before",
            "history",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::Orchestrator,
        keywords: &[
            "batch",
            "loop",
            "iterate",
            "for each",
            "every",
            "all files",
            "multiple",
            "process all",
            "count all",
            "script",
            "orchestrate",
            "workflow",
            "automate",
            "chain",
            "pipeline",
            "sequential",
            "conditional",
            "each file",
        ],
    },
    CategoryPatterns {
        category: ToolCategory::CodeExecution,
        keywords: &[
            "execute code",
            "run code",
            "python",
            "javascript",
            "compile",
            "interpreter",
            "sandbox",
            "piston",
            "run script",
        ],
    },
];

/// Get context for analysis based on adaptive window
/// - Short prompts (< 50 chars): analyze last 3 messages
/// - Medium prompts (50-150 chars): analyze last 2 messages
/// - Detailed prompts (> 150 chars): analyze current message only
pub fn get_context_for_analysis(messages: &[Message]) -> String {
    let last_msg = messages.last().and_then(|m| m.text()).unwrap_or("");
    let msg_len = last_msg.len();

    if msg_len > 150 {
        // Detailed prompt - current only
        last_msg.to_string()
    } else if msg_len > 50 {
        // Medium prompt - last 2
        messages
            .iter()
            .rev()
            .take(2)
            .filter_map(|m| m.text())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        // Short prompt - last 3
        messages
            .iter()
            .rev()
            .take(3)
            .filter_map(|m| m.text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Analyze a query and return relevant tool categories
pub fn analyze_query(query: &str) -> Vec<ToolCategory> {
    let query_lower = query.to_lowercase();
    let words: HashSet<&str> = query_lower.split_whitespace().collect();

    let mut matched_categories = Vec::new();

    for pattern in CATEGORY_PATTERNS {
        let has_match = pattern.keywords.iter().any(|kw| {
            // Check exact word match or substring
            words.contains(kw) || query_lower.contains(kw)
        });

        if has_match {
            matched_categories.push(pattern.category);
        }
    }

    // If no specific categories matched, return default set
    if matched_categories.is_empty() {
        // Default to FileOps, Search, and Bash (most commonly needed)
        return vec![
            ToolCategory::FileOps,
            ToolCategory::Search,
            ToolCategory::Bash,
        ];
    }

    // Always include FileOps as it's almost always useful
    if !matched_categories.contains(&ToolCategory::FileOps) {
        matched_categories.push(ToolCategory::FileOps);
    }

    matched_categories
}

/// Analyze conversation messages and return relevant tool categories
pub fn analyze_messages(messages: &[Message]) -> Vec<ToolCategory> {
    let context = get_context_for_analysis(messages);
    analyze_query(&context)
}

/// Get tools for the given categories from the registry
pub fn get_tools_for_categories(registry: &ToolRegistry, categories: &[ToolCategory]) -> Vec<Tool> {
    let mut tools = Vec::new();
    let mut seen_names = HashSet::new();

    for category in categories {
        for tool in registry.get_by_category(*category) {
            if seen_names.insert(tool.name.clone()) {
                tools.push(tool.clone());
            }
        }
    }

    tools
}

/// Get smart-routed tools for the given messages, picking from `registry`.
///
/// Callers that want to route over the full set of rullama builtins can
/// pass `rullama_tool_builtins::registry_with_builtins()`.
pub fn get_smart_tools(messages: &[Message], registry: &ToolRegistry) -> Vec<Tool> {
    let categories = analyze_messages(messages);
    get_tools_for_categories(registry, &categories)
}

/// Get smart-routed tools including MCP tools that match detected categories.
///
/// Same shape as [`get_smart_tools`]; the caller-supplied registry replaces
/// the previous hardcoded `ToolRegistry::with_builtins()`.
pub fn get_smart_tools_with_mcp(
    messages: &[Message],
    registry: &ToolRegistry,
    mcp_tools: &[Tool],
) -> Vec<Tool> {
    let categories = analyze_messages(messages);

    let mut tools = get_tools_for_categories(registry, &categories);
    let mut seen_names: HashSet<String> = tools.iter().map(|t| t.name.clone()).collect();

    // Add MCP tools that match any of the detected categories
    for mcp_tool in mcp_tools {
        if !seen_names.contains(&mcp_tool.name)
            && mcp_tool_matches_categories(mcp_tool, &categories)
        {
            tools.push(mcp_tool.clone());
            seen_names.insert(mcp_tool.name.clone());
        }
    }

    tools
}

/// Check if an MCP tool matches any of the detected categories based on keywords
fn mcp_tool_matches_categories(tool: &Tool, categories: &[ToolCategory]) -> bool {
    let text = format!("{} {}", tool.name, tool.description).to_lowercase();

    for category in categories {
        let matches = match category {
            ToolCategory::FileOps => {
                text.contains("file")
                    || text.contains("read")
                    || text.contains("write")
                    || text.contains("directory")
                    || text.contains("path")
                    || text.contains("folder")
            }
            ToolCategory::Search => {
                text.contains("search")
                    || text.contains("find")
                    || text.contains("query")
                    || text.contains("lookup")
                    || text.contains("grep")
            }
            ToolCategory::SemanticSearch => {
                text.contains("semantic")
                    || text.contains("embedding")
                    || text.contains("rag")
                    || text.contains("vector")
                    || text.contains("similarity")
            }
            ToolCategory::Git => {
                text.contains("git")
                    || text.contains("commit")
                    || text.contains("branch")
                    || text.contains("pull")
                    || text.contains("push")
                    || text.contains("repository")
            }
            ToolCategory::Web => {
                text.contains("http")
                    || text.contains("url")
                    || text.contains("api")
                    || text.contains("fetch")
                    || text.contains("request")
            }
            ToolCategory::WebSearch => {
                text.contains("web")
                    || text.contains("browse")
                    || text.contains("scrape")
                    || text.contains("google")
            }
            ToolCategory::Bash => {
                text.contains("shell")
                    || text.contains("exec")
                    || text.contains("command")
                    || text.contains("run")
                    || text.contains("terminal")
            }
            ToolCategory::TaskManager => {
                text.contains("task")
                    || text.contains("todo")
                    || text.contains("issue")
                    || text.contains("ticket")
            }
            ToolCategory::AgentPool => {
                text.contains("agent")
                    || text.contains("spawn")
                    || text.contains("worker")
                    || text.contains("parallel")
            }
            ToolCategory::Planning => {
                text.contains("plan") || text.contains("design") || text.contains("architect")
            }
            ToolCategory::Context => {
                text.contains("context") || text.contains("recall") || text.contains("memory")
            }
            ToolCategory::Orchestrator => {
                text.contains("script")
                    || text.contains("orchestrat")
                    || text.contains("automat")
                    || text.contains("workflow")
                    || text.contains("batch")
            }
            ToolCategory::CodeExecution => {
                text.contains("execute")
                    || text.contains("run")
                    || text.contains("code")
                    || text.contains("python")
                    || text.contains("javascript")
                    || text.contains("compile")
            }
            ToolCategory::SessionTask => {
                // Session tasks are internal-only, never match external MCP tools
                false
            }
            ToolCategory::Validation => {
                text.contains("valid")
                    || text.contains("check")
                    || text.contains("verify")
                    || text.contains("lint")
                    || text.contains("build")
                    || text.contains("syntax")
                    || text.contains("duplicate")
                    || text.contains("test")
            }
        };
        if matches {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_git_query() {
        let categories = analyze_query("Show me the git diff");
        assert!(categories.contains(&ToolCategory::Git));
    }

    #[test]
    fn test_analyze_file_query() {
        let categories = analyze_query("Read the config file");
        assert!(categories.contains(&ToolCategory::FileOps));
    }

    #[test]
    fn test_analyze_search_query() {
        let categories = analyze_query("Find all functions named handle");
        assert!(categories.contains(&ToolCategory::Search));
    }

    #[test]
    fn test_analyze_web_search_query() {
        let categories = analyze_query("Search the web for Rust best practices");
        assert!(categories.contains(&ToolCategory::WebSearch));
    }

    #[test]
    fn test_analyze_bash_query() {
        let categories = analyze_query("Run cargo build");
        assert!(categories.contains(&ToolCategory::Bash));
    }

    #[test]
    fn test_default_categories() {
        let categories = analyze_query("Hello, how are you?");
        assert!(!categories.is_empty());
        assert!(categories.contains(&ToolCategory::FileOps));
        assert!(categories.contains(&ToolCategory::Search));
        assert!(categories.contains(&ToolCategory::Bash));
    }

    #[test]
    fn test_fileops_always_included() {
        let categories = analyze_query("Show me the git status");
        // FileOps should be added even though only Git was matched
        assert!(categories.contains(&ToolCategory::FileOps));
        assert!(categories.contains(&ToolCategory::Git));
    }

    #[test]
    fn test_multiple_categories() {
        let categories = analyze_query("Search for files and run the tests");
        assert!(categories.contains(&ToolCategory::Search));
        assert!(categories.contains(&ToolCategory::Bash));
    }
}
