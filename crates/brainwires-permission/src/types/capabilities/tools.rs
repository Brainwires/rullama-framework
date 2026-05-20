use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Tool execution capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    /// Tool categories allowed
    #[serde(default = "default_allowed_categories")]
    pub allowed_categories: HashSet<ToolCategory>,

    /// Specific tools denied (overrides category allows)
    #[serde(default)]
    pub denied_tools: HashSet<String>,

    /// Specific tools allowed (if not using categories)
    #[serde(default)]
    pub allowed_tools: Option<HashSet<String>>,

    /// Require approval for these tools regardless of trust
    #[serde(default)]
    pub always_approve: HashSet<String>,
}

fn default_allowed_categories() -> HashSet<ToolCategory> {
    let mut set = HashSet::new();
    set.insert(ToolCategory::FileRead);
    set.insert(ToolCategory::Search);
    set.insert(ToolCategory::Web);
    set
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            allowed_categories: default_allowed_categories(),
            denied_tools: HashSet::new(),
            allowed_tools: None,
            always_approve: HashSet::new(),
        }
    }
}

impl ToolCapabilities {
    /// Create full access tool capabilities
    pub fn full() -> Self {
        let mut categories = HashSet::new();
        categories.insert(ToolCategory::FileRead);
        categories.insert(ToolCategory::FileWrite);
        categories.insert(ToolCategory::Search);
        categories.insert(ToolCategory::Git);
        categories.insert(ToolCategory::GitDestructive);
        categories.insert(ToolCategory::Bash);
        categories.insert(ToolCategory::Web);
        categories.insert(ToolCategory::CodeExecution);
        categories.insert(ToolCategory::AgentSpawn);
        categories.insert(ToolCategory::Planning);
        categories.insert(ToolCategory::System);

        Self {
            allowed_categories: categories,
            denied_tools: HashSet::new(),
            allowed_tools: None,
            always_approve: HashSet::new(),
        }
    }
}

/// Tool categories for permission grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Read file operations: read_file, list_directory, search_files
    FileRead,
    /// Write file operations: write_file, edit_file, patch_file, delete_file
    FileWrite,
    /// Search operations: search_code, semantic search, RAG
    Search,
    /// Git operations: status, diff, log, add, commit, push, pull
    Git,
    /// Destructive git operations: force push, hard reset, rebase
    GitDestructive,
    /// Shell command execution
    Bash,
    /// Web operations: fetch_url, web_search, web_scrape
    Web,
    /// Code execution in sandboxed environment
    CodeExecution,
    /// Agent spawning and management
    AgentSpawn,
    /// Planning and task management
    Planning,
    /// System-level operations
    System,
}
