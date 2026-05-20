//! Permission configuration loading
//!
//! Handles loading and parsing of permissions.toml configuration files.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::profiles::CapabilityProfile;
use super::types::{AgentCapabilities, GitOperation, PathPattern, ToolCategory};

/// Root configuration structure for permissions.toml
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermissionsConfig {
    /// Default settings
    #[serde(default)]
    pub default: DefaultConfig,

    /// Filesystem capabilities
    #[serde(default)]
    pub filesystem: FilesystemConfig,

    /// Tool capabilities
    #[serde(default)]
    pub tools: ToolsConfig,

    /// Network capabilities
    #[serde(default)]
    pub network: NetworkConfig,

    /// Spawning capabilities
    #[serde(default)]
    pub spawning: SpawningConfig,

    /// Git capabilities
    #[serde(default)]
    pub git: GitConfig,

    /// Resource quotas
    #[serde(default)]
    pub quotas: QuotasConfig,

    /// Policy rules
    #[serde(default)]
    pub policies: PoliciesConfig,
}

/// Default configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultConfig {
    /// Base profile to start from
    #[serde(default = "default_profile")]
    pub profile: String,
}

fn default_profile() -> String {
    "standard_dev".to_string()
}

impl Default for DefaultConfig {
    fn default() -> Self {
        Self {
            profile: default_profile(),
        }
    }
}

/// Filesystem configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FilesystemConfig {
    /// Allowed read paths
    #[serde(default)]
    pub read_paths: Option<Vec<String>>,

    /// Allowed write paths
    #[serde(default)]
    pub write_paths: Option<Vec<String>>,

    /// Denied paths
    #[serde(default)]
    pub denied_paths: Option<Vec<String>>,

    /// Can follow symlinks
    #[serde(default)]
    pub follow_symlinks: Option<bool>,

    /// Can access hidden files
    #[serde(default)]
    pub access_hidden: Option<bool>,

    /// Maximum write size (e.g., "1MB", "512KB")
    #[serde(default)]
    pub max_write_size: Option<String>,

    /// Can delete files
    #[serde(default)]
    pub can_delete: Option<bool>,

    /// Can create directories
    #[serde(default)]
    pub can_create_dirs: Option<bool>,
}

/// Tools configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolsConfig {
    /// Allowed tool categories
    #[serde(default)]
    pub allowed_categories: Option<Vec<String>>,

    /// Denied tools
    #[serde(default)]
    pub denied_tools: Option<Vec<String>>,

    /// Tools requiring approval
    #[serde(default)]
    pub always_approve: Option<Vec<String>>,
}

/// Network configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkConfig {
    /// Allowed domains
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,

    /// Denied domains
    #[serde(default)]
    pub denied_domains: Option<Vec<String>>,

    /// Allow all domains
    #[serde(default)]
    pub allow_all: Option<bool>,

    /// Rate limit (requests per minute)
    #[serde(default)]
    pub rate_limit: Option<u32>,

    /// Allow API calls
    #[serde(default)]
    pub allow_api_calls: Option<bool>,
}

/// Spawning configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpawningConfig {
    /// Enable spawning
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Maximum children
    #[serde(default)]
    pub max_children: Option<u32>,

    /// Maximum depth
    #[serde(default)]
    pub max_depth: Option<u32>,

    /// Can elevate privileges
    #[serde(default)]
    pub can_elevate: Option<bool>,
}

/// Git configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GitConfig {
    /// Allowed operations
    #[serde(default)]
    pub allowed_ops: Option<Vec<String>>,

    /// Protected branches
    #[serde(default)]
    pub protected_branches: Option<Vec<String>>,

    /// Can force push
    #[serde(default)]
    pub can_force_push: Option<bool>,

    /// Can perform destructive operations
    #[serde(default)]
    pub can_destructive: Option<bool>,

    /// Branches requiring PR
    #[serde(default)]
    pub require_pr_branches: Option<Vec<String>>,
}

/// Quotas configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuotasConfig {
    /// Maximum execution time (e.g., "30m", "1h")
    #[serde(default)]
    pub max_execution_time: Option<String>,

    /// Maximum tool calls
    #[serde(default)]
    pub max_tool_calls: Option<u32>,

    /// Maximum files modified
    #[serde(default)]
    pub max_files_modified: Option<u32>,

    /// Maximum tokens
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

/// Policies configuration section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoliciesConfig {
    /// Policy rules
    #[serde(default)]
    pub rules: Vec<PolicyRuleConfig>,
}

/// Individual policy rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRuleConfig {
    /// Rule name
    pub name: String,

    /// Priority (higher = checked first)
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Conditions
    #[serde(default)]
    pub conditions: Vec<PolicyConditionConfig>,

    /// Action to take
    pub action: String,

    /// Enforcement mode
    #[serde(default = "default_enforcement")]
    pub enforcement: String,
}

fn default_priority() -> u32 {
    50
}

fn default_enforcement() -> String {
    "Coercive".to_string()
}

/// Policy condition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PolicyConditionConfig {
    /// Match by file path pattern.
    FilePath {
        /// The file path pattern.
        file_path: String,
    },
    /// Match by tool name.
    Tool {
        /// The tool name.
        tool: String,
    },
    /// Match by tool category.
    ToolCategory {
        /// The tool category name.
        tool_category: String,
    },
    /// Match by git operation.
    GitOp {
        /// The git operation name.
        git_op: String,
    },
    /// Match by trust level.
    TrustLevel {
        /// Trust level condition.
        trust_level: TrustLevelConditionConfig,
    },
    /// Match by network domain.
    Domain {
        /// The domain name.
        domain: String,
    },
}

/// Trust level condition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustLevelConditionConfig {
    /// Minimum trust level (inclusive).
    #[serde(default)]
    pub at_least: Option<String>,
    /// Maximum trust level (inclusive).
    #[serde(default)]
    pub at_most: Option<String>,
    /// Exact trust level required.
    #[serde(default)]
    pub exactly: Option<String>,
}

/// Type alias for policy rule (used by policy engine)
pub type PolicyRule = PolicyRuleConfig;

/// Simplified policy condition for policy engine
#[derive(Debug, Clone, Default)]
pub struct PolicyCondition {
    /// Tool name to match.
    pub tool: Option<String>,
    /// Tool category to match.
    pub tool_category: Option<String>,
    /// File path pattern to match.
    pub file_path: Option<String>,
    /// Network domain to match.
    pub domain: Option<String>,
    /// Git operation to match.
    pub git_op: Option<String>,
    /// Minimum trust level required.
    pub min_trust_level: Option<u8>,
}

impl From<&PolicyConditionConfig> for PolicyCondition {
    fn from(config: &PolicyConditionConfig) -> Self {
        match config {
            PolicyConditionConfig::Tool { tool } => PolicyCondition {
                tool: Some(tool.clone()),
                ..Default::default()
            },
            PolicyConditionConfig::ToolCategory { tool_category } => PolicyCondition {
                tool_category: Some(tool_category.clone()),
                ..Default::default()
            },
            PolicyConditionConfig::FilePath { file_path } => PolicyCondition {
                file_path: Some(file_path.clone()),
                ..Default::default()
            },
            PolicyConditionConfig::Domain { domain } => PolicyCondition {
                domain: Some(domain.clone()),
                ..Default::default()
            },
            PolicyConditionConfig::GitOp { git_op } => PolicyCondition {
                git_op: Some(git_op.clone()),
                ..Default::default()
            },
            PolicyConditionConfig::TrustLevel { trust_level } => {
                let level =
                    trust_level
                        .at_least
                        .as_ref()
                        .and_then(|s| match s.to_lowercase().as_str() {
                            "untrusted" => Some(0),
                            "low" => Some(1),
                            "medium" => Some(2),
                            "high" => Some(3),
                            "system" => Some(4),
                            _ => s.parse().ok(),
                        });
                PolicyCondition {
                    min_trust_level: level,
                    ..Default::default()
                }
            }
        }
    }
}

/// Extended PolicyRuleConfig methods for policy engine integration
impl PolicyRuleConfig {
    /// Get conditions as PolicyCondition structs
    pub fn get_conditions(&self) -> Vec<PolicyCondition> {
        self.conditions.iter().map(PolicyCondition::from).collect()
    }
}

impl PermissionsConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read permissions config from {:?}", path))?;

        toml::from_str(&contents)
            .with_context(|| format!("Failed to parse permissions config from {:?}", path))
    }

    /// Load configuration from the default path, or return default config if not found
    #[cfg(feature = "native")]
    pub fn load_or_default() -> Self {
        match default_permissions_path() {
            Ok(path) if path.exists() => Self::load(&path).unwrap_or_default(),
            _ => Self::default(),
        }
    }

    /// Convert to AgentCapabilities
    pub fn to_capabilities(&self) -> AgentCapabilities {
        // Start with base profile
        let profile = CapabilityProfile::parse(&self.default.profile)
            .unwrap_or(CapabilityProfile::StandardDev);
        let mut caps = AgentCapabilities::from_profile(profile);

        // Apply filesystem overrides
        if let Some(ref paths) = self.filesystem.read_paths {
            caps.filesystem.read_paths = paths.iter().map(|p| PathPattern::new(p)).collect();
        }
        if let Some(ref paths) = self.filesystem.write_paths {
            caps.filesystem.write_paths = paths.iter().map(|p| PathPattern::new(p)).collect();
        }
        if let Some(ref paths) = self.filesystem.denied_paths {
            caps.filesystem.denied_paths = paths.iter().map(|p| PathPattern::new(p)).collect();
        }
        if let Some(v) = self.filesystem.follow_symlinks {
            caps.filesystem.follow_symlinks = v;
        }
        if let Some(v) = self.filesystem.access_hidden {
            caps.filesystem.access_hidden = v;
        }
        if let Some(ref size) = self.filesystem.max_write_size {
            caps.filesystem.max_write_size = parse_size(size);
        }
        if let Some(v) = self.filesystem.can_delete {
            caps.filesystem.can_delete = v;
        }
        if let Some(v) = self.filesystem.can_create_dirs {
            caps.filesystem.can_create_dirs = v;
        }

        // Apply tools overrides
        if let Some(ref cats) = self.tools.allowed_categories {
            caps.tools.allowed_categories =
                cats.iter().filter_map(|c| parse_tool_category(c)).collect();
        }
        if let Some(ref tools) = self.tools.denied_tools {
            caps.tools.denied_tools = tools.iter().cloned().collect();
        }
        if let Some(ref tools) = self.tools.always_approve {
            caps.tools.always_approve = tools.iter().cloned().collect();
        }

        // Apply network overrides
        if let Some(ref domains) = self.network.allowed_domains {
            caps.network.allowed_domains = domains.clone();
        }
        if let Some(ref domains) = self.network.denied_domains {
            caps.network.denied_domains = domains.clone();
        }
        if let Some(v) = self.network.allow_all {
            caps.network.allow_all = v;
        }
        if let Some(v) = self.network.rate_limit {
            caps.network.rate_limit = Some(v);
        }
        if let Some(v) = self.network.allow_api_calls {
            caps.network.allow_api_calls = v;
        }

        // Apply spawning overrides
        if let Some(v) = self.spawning.enabled {
            caps.spawning.can_spawn = v;
        }
        if let Some(v) = self.spawning.max_children {
            caps.spawning.max_children = v;
        }
        if let Some(v) = self.spawning.max_depth {
            caps.spawning.max_depth = v;
        }
        if let Some(v) = self.spawning.can_elevate {
            caps.spawning.can_elevate = v;
        }

        // Apply git overrides
        if let Some(ref ops) = self.git.allowed_ops {
            caps.git.allowed_ops = ops.iter().filter_map(|o| parse_git_operation(o)).collect();
        }
        if let Some(ref branches) = self.git.protected_branches {
            caps.git.protected_branches = branches.clone();
        }
        if let Some(v) = self.git.can_force_push {
            caps.git.can_force_push = v;
        }
        if let Some(v) = self.git.can_destructive {
            caps.git.can_destructive = v;
        }
        if let Some(ref branches) = self.git.require_pr_branches {
            caps.git.require_pr_branches = branches.clone();
        }

        // Apply quota overrides
        if let Some(ref time) = self.quotas.max_execution_time {
            caps.quotas.max_execution_time = parse_duration(time);
        }
        if let Some(v) = self.quotas.max_tool_calls {
            caps.quotas.max_tool_calls = Some(v);
        }
        if let Some(v) = self.quotas.max_files_modified {
            caps.quotas.max_files_modified = Some(v);
        }
        if let Some(v) = self.quotas.max_tokens {
            caps.quotas.max_tokens = Some(v);
        }

        caps
    }

    /// Generate default TOML configuration content
    pub fn default_toml() -> String {
        r#"# Brainwires Permission Configuration
# Location: ~/.brainwires/permissions.toml

[default]
profile = "standard_dev"  # read_only | standard_dev | full_access | custom

[filesystem]
read_paths = ["**/*"]
write_paths = ["src/**", "tests/**", "docs/**"]
denied_paths = ["**/.env*", "**/secrets/**"]
follow_symlinks = true
access_hidden = true
max_write_size = "1MB"

[tools]
allowed_categories = ["FileRead", "FileWrite", "Search", "Git", "Planning"]
denied_tools = ["execute_code"]
always_approve = ["delete_file", "execute_command"]

[network]
allowed_domains = ["github.com", "*.github.com", "docs.rs", "crates.io"]
rate_limit = 60

[spawning]
enabled = true
max_children = 3
max_depth = 2

[git]
allowed_ops = ["Status", "Diff", "Log", "Add", "Commit", "Push", "Pull"]
protected_branches = ["main", "master"]
can_force_push = false

[quotas]
max_execution_time = "30m"
max_tool_calls = 500
max_files_modified = 50

[[policies.rules]]
name = "protect_secrets"
priority = 100
conditions = [
    { file_path = "**/.env*" },
    { file_path = "**/*secret*" },
]
action = "Deny"
enforcement = "Coercive"

[[policies.rules]]
name = "approve_destructive_git"
priority = 90
conditions = [
    { git_op = "Reset" },
    { git_op = "Rebase" },
]
action = "RequireApproval"
enforcement = "Coercive"
"#
        .to_string()
    }
}

/// Get the default permissions file path
/// Uses ~/.brainwires/permissions.toml
#[cfg(feature = "native")]
pub fn default_permissions_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Failed to get home directory"))?;
    Ok(home.join(".brainwires").join("permissions.toml"))
}

/// Ensure the .brainwires directory exists
#[cfg(feature = "native")]
pub fn ensure_permissions_dir() -> Result<PathBuf> {
    let dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Failed to get home directory"))?
        .join(".brainwires");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Parse a size string like "1MB" or "512KB" into bytes
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim().to_uppercase();
    let (num, unit) = if s.ends_with("GB") {
        (s.trim_end_matches("GB").trim(), 1024 * 1024 * 1024)
    } else if s.ends_with("MB") {
        (s.trim_end_matches("MB").trim(), 1024 * 1024)
    } else if s.ends_with("KB") {
        (s.trim_end_matches("KB").trim(), 1024)
    } else if s.ends_with('B') {
        (s.trim_end_matches('B').trim(), 1)
    } else {
        (s.as_str(), 1)
    };

    num.parse::<u64>().ok().map(|n| n * unit)
}

/// Parse a duration string like "30m" or "1h" into seconds
fn parse_duration(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    let (num, unit) = if s.ends_with('h') {
        (s.trim_end_matches('h').trim(), 3600)
    } else if s.ends_with('m') {
        (s.trim_end_matches('m').trim(), 60)
    } else if s.ends_with('s') {
        (s.trim_end_matches('s').trim(), 1)
    } else {
        (s.as_str(), 1)
    };

    num.parse::<u64>().ok().map(|n| n * unit)
}

/// Parse a tool category string
pub fn parse_tool_category(s: &str) -> Option<ToolCategory> {
    match s.to_lowercase().replace('_', "").as_str() {
        "fileread" => Some(ToolCategory::FileRead),
        "filewrite" => Some(ToolCategory::FileWrite),
        "search" => Some(ToolCategory::Search),
        "git" => Some(ToolCategory::Git),
        "gitdestructive" => Some(ToolCategory::GitDestructive),
        "bash" => Some(ToolCategory::Bash),
        "web" => Some(ToolCategory::Web),
        "codeexecution" => Some(ToolCategory::CodeExecution),
        "agentspawn" => Some(ToolCategory::AgentSpawn),
        "planning" => Some(ToolCategory::Planning),
        "system" => Some(ToolCategory::System),
        _ => None,
    }
}

/// Parse a git operation string
pub fn parse_git_operation(s: &str) -> Option<GitOperation> {
    match s.to_lowercase().as_str() {
        "status" => Some(GitOperation::Status),
        "diff" => Some(GitOperation::Diff),
        "log" => Some(GitOperation::Log),
        "add" => Some(GitOperation::Add),
        "commit" => Some(GitOperation::Commit),
        "push" => Some(GitOperation::Push),
        "pull" => Some(GitOperation::Pull),
        "fetch" => Some(GitOperation::Fetch),
        "branch" => Some(GitOperation::Branch),
        "checkout" => Some(GitOperation::Checkout),
        "merge" => Some(GitOperation::Merge),
        "rebase" => Some(GitOperation::Rebase),
        "reset" => Some(GitOperation::Reset),
        "stash" => Some(GitOperation::Stash),
        "tag" => Some(GitOperation::Tag),
        "forcepush" | "force_push" | "force-push" => Some(GitOperation::ForcePush),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("1MB"), Some(1024 * 1024));
        assert_eq!(parse_size("512KB"), Some(512 * 1024));
        assert_eq!(parse_size("1GB"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_size("100B"), Some(100));
        assert_eq!(parse_size("100"), Some(100));
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("30m"), Some(30 * 60));
        assert_eq!(parse_duration("1h"), Some(3600));
        assert_eq!(parse_duration("90s"), Some(90));
        assert_eq!(parse_duration("120"), Some(120));
    }

    #[test]
    fn test_parse_tool_category() {
        assert_eq!(
            parse_tool_category("FileRead"),
            Some(ToolCategory::FileRead)
        );
        assert_eq!(
            parse_tool_category("file_read"),
            Some(ToolCategory::FileRead)
        );
        assert_eq!(parse_tool_category("Git"), Some(ToolCategory::Git));
        assert_eq!(parse_tool_category("invalid"), None);
    }

    #[test]
    fn test_parse_git_operation() {
        assert_eq!(parse_git_operation("Status"), Some(GitOperation::Status));
        assert_eq!(parse_git_operation("push"), Some(GitOperation::Push));
        assert_eq!(
            parse_git_operation("ForcePush"),
            Some(GitOperation::ForcePush)
        );
        assert_eq!(
            parse_git_operation("force_push"),
            Some(GitOperation::ForcePush)
        );
    }

    #[test]
    fn test_default_toml_parses() {
        let toml_str = PermissionsConfig::default_toml();
        let config: Result<PermissionsConfig, _> = toml::from_str(&toml_str);
        assert!(
            config.is_ok(),
            "Default TOML should parse: {:?}",
            config.err()
        );
    }

    #[test]
    fn test_config_to_capabilities() {
        let config = PermissionsConfig::default();
        let caps = config.to_capabilities();

        // Should be standard_dev by default
        assert!(caps.allows_tool("read_file"));
        assert!(caps.allows_tool("write_file"));
    }
}
