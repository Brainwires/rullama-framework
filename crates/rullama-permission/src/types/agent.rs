//! AgentCapabilities and CapabilityProfile types with all profile constructors and logic.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::capabilities::{
    FilesystemCapabilities, GitCapabilities, GitOperation, NetworkCapabilities, ResourceQuotas,
    SpawningCapabilities, ToolCapabilities, ToolCategory,
};
use super::path_pattern::PathPattern;

/// Agent capabilities - explicit permissions granted to an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Unique capability set ID for auditing
    #[serde(default = "default_capability_id")]
    pub capability_id: String,

    /// File system capabilities
    #[serde(default)]
    pub filesystem: FilesystemCapabilities,

    /// Tool execution capabilities
    #[serde(default)]
    pub tools: ToolCapabilities,

    /// Network capabilities
    #[serde(default)]
    pub network: NetworkCapabilities,

    /// Agent spawning capabilities
    #[serde(default)]
    pub spawning: SpawningCapabilities,

    /// Git operation capabilities
    #[serde(default)]
    pub git: GitCapabilities,

    /// Resource quota limits
    #[serde(default)]
    pub quotas: ResourceQuotas,
}

fn default_capability_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            capability_id: default_capability_id(),
            filesystem: FilesystemCapabilities::default(),
            tools: ToolCapabilities::default(),
            network: NetworkCapabilities::default(),
            spawning: SpawningCapabilities::default(),
            git: GitCapabilities::default(),
            quotas: ResourceQuotas::default(),
        }
    }
}

impl AgentCapabilities {
    /// Check if a tool is allowed by the current capabilities
    pub fn allows_tool(&self, tool_name: &str) -> bool {
        // Check explicit deny list first
        if self.tools.denied_tools.contains(tool_name) {
            return false;
        }

        // Check explicit allow list if specified
        if let Some(ref allowed) = self.tools.allowed_tools {
            return allowed.contains(tool_name);
        }

        // Fall back to category-based check
        let category = Self::categorize_tool(tool_name);
        self.tools.allowed_categories.contains(&category)
    }

    /// Check if a tool requires explicit approval
    pub fn requires_approval(&self, tool_name: &str) -> bool {
        self.tools.always_approve.contains(tool_name)
    }

    /// Categorize a tool by name into a ToolCategory
    pub fn categorize_tool(tool_name: &str) -> ToolCategory {
        match tool_name {
            // File read operations
            "read_file" | "list_directory" | "search_files" => ToolCategory::FileRead,

            // File write operations
            "write_file" | "edit_file" | "patch_file" | "delete_file" | "create_directory" => {
                ToolCategory::FileWrite
            }

            // Search operations
            "search_code"
            | "index_codebase"
            | "query_codebase"
            | "search_with_filters"
            | "get_rag_statistics"
            | "clear_rag_index"
            | "search_git_history" => ToolCategory::Search,

            // Git operations - check for destructive operations first
            name if name.starts_with("git_") => {
                if name.contains("force")
                    || name.contains("reset")
                    || name.contains("rebase")
                    || name.contains("delete_branch")
                {
                    ToolCategory::GitDestructive
                } else {
                    ToolCategory::Git
                }
            }

            // Bash/shell operations
            "execute_command" => ToolCategory::Bash,

            // Web operations
            "fetch_url" | "web_search" | "web_browse" | "web_scrape" => ToolCategory::Web,

            // Code execution
            "execute_code" | "execute_script" => ToolCategory::CodeExecution,

            // Agent operations
            "agent_spawn" | "agent_stop" | "agent_status" | "agent_list" | "agent_pool_stats"
            | "agent_file_locks" => ToolCategory::AgentSpawn,

            // Planning/task operations
            "plan_task" | "task_create" | "task_add_subtask" | "task_start" | "task_complete"
            | "task_fail" | "task_list" | "task_get" => ToolCategory::Planning,

            // MCP tools
            name if name.starts_with("mcp_") => ToolCategory::System,

            // Context operations
            "recall_context" | "search_tools" => ToolCategory::Search,

            // Default to System for unknown tools
            _ => ToolCategory::System,
        }
    }

    /// Check if a file path is allowed for reading
    pub fn allows_read(&self, path: &str) -> bool {
        // Check denied paths first
        for denied in &self.filesystem.denied_paths {
            if denied.matches(path) {
                return false;
            }
        }

        // Check if any read path matches
        for allowed in &self.filesystem.read_paths {
            if allowed.matches(path) {
                return true;
            }
        }

        false
    }

    /// Check if a file path is allowed for writing
    pub fn allows_write(&self, path: &str) -> bool {
        // Check denied paths first
        for denied in &self.filesystem.denied_paths {
            if denied.matches(path) {
                return false;
            }
        }

        // Check if any write path matches
        for allowed in &self.filesystem.write_paths {
            if allowed.matches(path) {
                return true;
            }
        }

        false
    }

    /// Check if a domain is allowed for network access
    pub fn allows_domain(&self, domain: &str) -> bool {
        // Check denied domains first
        for denied in &self.network.denied_domains {
            if Self::domain_matches(denied, domain) {
                return false;
            }
        }

        // If allow_all is set, allow everything not denied
        if self.network.allow_all {
            return true;
        }

        // Check allowed domains
        for allowed in &self.network.allowed_domains {
            if Self::domain_matches(allowed, domain) {
                return true;
            }
        }

        false
    }

    /// Check if a git operation is allowed
    pub fn allows_git_op(&self, op: GitOperation) -> bool {
        // Check for destructive operations
        if op.is_destructive() && !self.git.can_destructive {
            return false;
        }

        // Check force push
        if op == GitOperation::ForcePush && !self.git.can_force_push {
            return false;
        }

        self.git.allowed_ops.contains(&op)
    }

    /// Check if spawning agents is allowed
    pub fn can_spawn_agent(&self, current_children: u32, current_depth: u32) -> bool {
        if !self.spawning.can_spawn {
            return false;
        }

        if current_children >= self.spawning.max_children {
            return false;
        }

        if current_depth >= self.spawning.max_depth {
            return false;
        }

        true
    }

    /// Simple domain matching with wildcard support
    fn domain_matches(pattern: &str, domain: &str) -> bool {
        if pattern.starts_with("*.") {
            let suffix = &pattern[1..]; // Keep the dot
            domain.ends_with(suffix) || domain == &pattern[2..]
        } else {
            pattern == domain
        }
    }
}

// ── Capability Profiles ──────────────────────────────────────────────

/// Capability profile names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityProfile {
    /// Read-only exploration - safe for untrusted agents
    ReadOnly,
    /// Standard development - balanced safety and utility
    StandardDev,
    /// Full access - for trusted orchestrators
    FullAccess,
    /// Custom profile loaded from config
    Custom,
}

impl CapabilityProfile {
    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "read_only" | "readonly" | "read-only" => Some(Self::ReadOnly),
            "standard_dev" | "standarddev" | "standard-dev" | "standard" => Some(Self::StandardDev),
            "full_access" | "fullaccess" | "full-access" | "full" => Some(Self::FullAccess),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::StandardDev => "standard_dev",
            Self::FullAccess => "full_access",
            Self::Custom => "custom",
        }
    }
}

impl AgentCapabilities {
    /// Read-only exploration - safe for untrusted agents
    ///
    /// This profile allows:
    /// - Reading all files (except secrets)
    /// - Search operations
    /// - Read-only git operations
    /// - No network access
    /// - No spawning
    /// - Conservative quotas
    pub fn read_only() -> Self {
        Self {
            capability_id: uuid::Uuid::new_v4().to_string(),
            filesystem: FilesystemCapabilities {
                read_paths: vec![PathPattern::new("**/*")],
                write_paths: vec![],
                denied_paths: vec![
                    PathPattern::new("**/.env*"),
                    PathPattern::new("**/*credentials*"),
                    PathPattern::new("**/*secret*"),
                    PathPattern::new("**/*.pem"),
                    PathPattern::new("**/*.key"),
                ],
                follow_symlinks: false,
                access_hidden: false,
                can_delete: false,
                can_create_dirs: false,
                max_write_size: None,
            },
            tools: ToolCapabilities {
                allowed_categories: {
                    let mut cats = HashSet::new();
                    cats.insert(ToolCategory::FileRead);
                    cats.insert(ToolCategory::Search);
                    cats
                },
                denied_tools: HashSet::new(),
                allowed_tools: None,
                always_approve: HashSet::new(),
            },
            network: NetworkCapabilities::disabled(),
            spawning: SpawningCapabilities::disabled(),
            git: GitCapabilities::read_only(),
            quotas: ResourceQuotas::conservative(),
        }
    }

    /// Standard development - balanced safety and utility
    ///
    /// This profile allows:
    /// - Reading all files (except secrets)
    /// - Writing to src/, tests/, docs/
    /// - File read/write, search, git, planning tools
    /// - Network access to common dev domains
    /// - Limited agent spawning
    /// - Standard quotas
    pub fn standard_dev() -> Self {
        Self {
            capability_id: uuid::Uuid::new_v4().to_string(),
            filesystem: FilesystemCapabilities {
                read_paths: vec![PathPattern::new("**/*")],
                write_paths: vec![
                    PathPattern::new("src/**"),
                    PathPattern::new("tests/**"),
                    PathPattern::new("docs/**"),
                    PathPattern::new("scripts/**"),
                    PathPattern::new("*.toml"),
                    PathPattern::new("*.json"),
                    PathPattern::new("*.yaml"),
                    PathPattern::new("*.yml"),
                    PathPattern::new("*.md"),
                    PathPattern::new("Makefile"),
                    PathPattern::new(".gitignore"),
                ],
                denied_paths: vec![
                    PathPattern::new("**/.env*"),
                    PathPattern::new("**/*credentials*"),
                    PathPattern::new("**/*secret*"),
                    PathPattern::new("**/node_modules/**"),
                    PathPattern::new("**/target/**"),
                    PathPattern::new("**/.git/**"),
                ],
                follow_symlinks: true,
                access_hidden: true,
                can_delete: true,
                can_create_dirs: true,
                max_write_size: Some(1024 * 1024), // 1MB
            },
            tools: ToolCapabilities {
                allowed_categories: {
                    let mut cats = HashSet::new();
                    cats.insert(ToolCategory::FileRead);
                    cats.insert(ToolCategory::FileWrite);
                    cats.insert(ToolCategory::Search);
                    cats.insert(ToolCategory::Git);
                    cats.insert(ToolCategory::Planning);
                    cats.insert(ToolCategory::Web);
                    cats
                },
                denied_tools: {
                    let mut denied = HashSet::new();
                    denied.insert("execute_code".to_string());
                    denied
                },
                allowed_tools: None,
                always_approve: {
                    let mut approve = HashSet::new();
                    approve.insert("delete_file".to_string());
                    approve.insert("execute_command".to_string());
                    approve
                },
            },
            network: NetworkCapabilities {
                allowed_domains: vec![
                    "github.com".to_string(),
                    "*.github.com".to_string(),
                    "docs.rs".to_string(),
                    "crates.io".to_string(),
                    "npmjs.com".to_string(),
                    "*.npmjs.com".to_string(),
                    "pypi.org".to_string(),
                    "stackoverflow.com".to_string(),
                ],
                denied_domains: vec![],
                allow_all: false,
                rate_limit: Some(60),
                allow_api_calls: true,
                max_response_size: Some(10 * 1024 * 1024), // 10MB
            },
            spawning: SpawningCapabilities {
                can_spawn: true,
                max_children: 3,
                max_depth: 2,
                can_elevate: false,
            },
            git: GitCapabilities::standard(),
            quotas: ResourceQuotas::standard(),
        }
    }

    /// Full access - for trusted orchestrators
    ///
    /// This profile allows:
    /// - Full filesystem access
    /// - All tools including bash and code execution
    /// - Full network access
    /// - Full spawning capabilities
    /// - Generous quotas
    pub fn full_access() -> Self {
        Self {
            capability_id: uuid::Uuid::new_v4().to_string(),
            filesystem: FilesystemCapabilities::full(),
            tools: ToolCapabilities::full(),
            network: NetworkCapabilities::full(),
            spawning: SpawningCapabilities::full(),
            git: GitCapabilities::full(),
            quotas: ResourceQuotas::generous(),
        }
    }

    /// Create capabilities from a profile name
    pub fn from_profile(profile: CapabilityProfile) -> Self {
        match profile {
            CapabilityProfile::ReadOnly => Self::read_only(),
            CapabilityProfile::StandardDev => Self::standard_dev(),
            CapabilityProfile::FullAccess => Self::full_access(),
            CapabilityProfile::Custom => Self::default(),
        }
    }

    /// Create a child capability set that is a subset of the parent
    ///
    /// Child capabilities can never exceed parent capabilities.
    pub fn derive_child(&self) -> Self {
        // Child inherits parent capabilities but with reduced spawning depth
        let mut child = self.clone();
        child.capability_id = uuid::Uuid::new_v4().to_string();

        // Reduce spawning depth
        if child.spawning.max_depth > 0 {
            child.spawning.max_depth -= 1;
        }

        // Disable elevation for children
        child.spawning.can_elevate = false;

        child
    }

    /// Merge capabilities, taking the more restrictive option for each field
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            capability_id: uuid::Uuid::new_v4().to_string(),
            filesystem: FilesystemCapabilities {
                // Intersection of allowed paths
                read_paths: self
                    .filesystem
                    .read_paths
                    .iter()
                    .filter(|p| {
                        other
                            .filesystem
                            .read_paths
                            .iter()
                            .any(|op| op.pattern() == p.pattern())
                    })
                    .cloned()
                    .collect(),
                write_paths: self
                    .filesystem
                    .write_paths
                    .iter()
                    .filter(|p| {
                        other
                            .filesystem
                            .write_paths
                            .iter()
                            .any(|op| op.pattern() == p.pattern())
                    })
                    .cloned()
                    .collect(),
                // Union of denied paths (more restrictive)
                denied_paths: {
                    let mut denied = self.filesystem.denied_paths.clone();
                    for p in &other.filesystem.denied_paths {
                        if !denied.iter().any(|dp| dp.pattern() == p.pattern()) {
                            denied.push(p.clone());
                        }
                    }
                    denied
                },
                follow_symlinks: self.filesystem.follow_symlinks
                    && other.filesystem.follow_symlinks,
                access_hidden: self.filesystem.access_hidden && other.filesystem.access_hidden,
                can_delete: self.filesystem.can_delete && other.filesystem.can_delete,
                can_create_dirs: self.filesystem.can_create_dirs
                    && other.filesystem.can_create_dirs,
                max_write_size: match (
                    self.filesystem.max_write_size,
                    other.filesystem.max_write_size,
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
            },
            tools: ToolCapabilities {
                // Intersection of allowed categories
                allowed_categories: self
                    .tools
                    .allowed_categories
                    .intersection(&other.tools.allowed_categories)
                    .cloned()
                    .collect(),
                // Union of denied tools
                denied_tools: self
                    .tools
                    .denied_tools
                    .union(&other.tools.denied_tools)
                    .cloned()
                    .collect(),
                allowed_tools: match (&self.tools.allowed_tools, &other.tools.allowed_tools) {
                    (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect()),
                    (Some(a), None) => Some(a.clone()),
                    (None, Some(b)) => Some(b.clone()),
                    (None, None) => None,
                },
                // Union of tools requiring approval
                always_approve: self
                    .tools
                    .always_approve
                    .union(&other.tools.always_approve)
                    .cloned()
                    .collect(),
            },
            network: NetworkCapabilities {
                allowed_domains: self
                    .network
                    .allowed_domains
                    .iter()
                    .filter(|d| {
                        other.network.allowed_domains.contains(d) || other.network.allow_all
                    })
                    .cloned()
                    .collect(),
                denied_domains: {
                    let mut denied = self.network.denied_domains.clone();
                    denied.extend(other.network.denied_domains.iter().cloned());
                    denied.sort();
                    denied.dedup();
                    denied
                },
                allow_all: self.network.allow_all && other.network.allow_all,
                rate_limit: match (self.network.rate_limit, other.network.rate_limit) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
                allow_api_calls: self.network.allow_api_calls && other.network.allow_api_calls,
                max_response_size: match (
                    self.network.max_response_size,
                    other.network.max_response_size,
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
            },
            spawning: SpawningCapabilities {
                can_spawn: self.spawning.can_spawn && other.spawning.can_spawn,
                max_children: self.spawning.max_children.min(other.spawning.max_children),
                max_depth: self.spawning.max_depth.min(other.spawning.max_depth),
                can_elevate: self.spawning.can_elevate && other.spawning.can_elevate,
            },
            git: GitCapabilities {
                allowed_ops: self
                    .git
                    .allowed_ops
                    .intersection(&other.git.allowed_ops)
                    .cloned()
                    .collect(),
                protected_branches: {
                    let mut branches = self.git.protected_branches.clone();
                    branches.extend(other.git.protected_branches.iter().cloned());
                    branches.sort();
                    branches.dedup();
                    branches
                },
                can_force_push: self.git.can_force_push && other.git.can_force_push,
                can_destructive: self.git.can_destructive && other.git.can_destructive,
                require_pr_branches: {
                    let mut branches = self.git.require_pr_branches.clone();
                    branches.extend(other.git.require_pr_branches.iter().cloned());
                    branches.sort();
                    branches.dedup();
                    branches
                },
            },
            quotas: ResourceQuotas {
                max_execution_time: match (
                    self.quotas.max_execution_time,
                    other.quotas.max_execution_time,
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
                max_memory: match (self.quotas.max_memory, other.quotas.max_memory) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
                max_tokens: match (self.quotas.max_tokens, other.quotas.max_tokens) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
                max_tool_calls: match (self.quotas.max_tool_calls, other.quotas.max_tool_calls) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
                max_files_modified: match (
                    self.quotas.max_files_modified,
                    other.quotas.max_files_modified,
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                },
            },
        }
    }
}
