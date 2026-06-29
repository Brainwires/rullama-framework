//! Git Worktree Management for Agent Isolation
//!
//! This module provides Git worktree management to allow agents to work in
//! isolation without interfering with each other's changes. Each agent can
//! have its own worktree, enabling parallel development work.
//!
//! # Key Concepts
//!
//! - **Worktree**: A separate working directory linked to the same Git repository
//! - **WorktreeManager**: Manages creation, tracking, and cleanup of worktrees
//! - **AgentWorktree**: Associates a worktree with a specific agent
//!
//! # Use Cases
//!
//! - Multiple agents working on different features simultaneously
//! - Isolation of experimental changes
//! - Safe build/test environments without affecting main working directory

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

const WORKTREE_MAX_AGE_SECS: u64 = 86_400;

/// Manages Git worktrees for agent isolation
pub struct WorktreeManager {
    /// The main repository path
    repo_path: PathBuf,
    /// Base directory for worktrees
    worktree_base: PathBuf,
    /// Active worktrees by agent ID
    worktrees: RwLock<HashMap<String, AgentWorktree>>,
    /// Configuration
    config: WorktreeConfig,
}

/// Configuration for worktree management
#[derive(Debug, Clone)]
pub struct WorktreeConfig {
    /// Maximum age before a worktree is considered stale
    pub max_age: Duration,
    /// Whether to auto-cleanup stale worktrees
    pub auto_cleanup: bool,
    /// Prefix for worktree directory names
    pub prefix: String,
    /// Maximum number of worktrees allowed
    pub max_worktrees: usize,
}

impl Default for WorktreeConfig {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(WORKTREE_MAX_AGE_SECS),
            auto_cleanup: true,
            prefix: "agent-wt-".to_string(),
            max_worktrees: 10,
        }
    }
}

/// A worktree associated with an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWorktree {
    /// Agent that owns this worktree
    pub agent_id: String,
    /// Path to the worktree
    pub path: PathBuf,
    /// Branch name in this worktree
    pub branch: String,
    /// When the worktree was created
    #[serde(skip, default = "Instant::now")]
    pub created_at: Instant,
    /// When the worktree was last accessed
    #[serde(skip, default = "Instant::now")]
    pub last_accessed: Instant,
    /// Whether the worktree has uncommitted changes
    pub has_changes: bool,
    /// Purpose/description of this worktree
    pub purpose: String,
}

impl AgentWorktree {
    /// Check if the worktree is stale
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.last_accessed.elapsed() > max_age
    }

    /// Get the age of this worktree
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Result of worktree operations
#[derive(Debug)]
pub enum WorktreeResult {
    /// Worktree created successfully
    Created(AgentWorktree),
    /// Worktree already exists
    AlreadyExists(AgentWorktree),
    /// Worktree removed successfully
    Removed {
        /// Path of the removed worktree.
        path: PathBuf,
    },
    /// Operation failed
    Error(String),
}

impl WorktreeResult {
    /// Check if the operation was successful
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            WorktreeResult::Created(_)
                | WorktreeResult::AlreadyExists(_)
                | WorktreeResult::Removed { .. }
        )
    }

    /// Get the worktree if available
    pub fn worktree(&self) -> Option<&AgentWorktree> {
        match self {
            WorktreeResult::Created(wt) | WorktreeResult::AlreadyExists(wt) => Some(wt),
            _ => None,
        }
    }
}

/// Information about a Git worktree from `git worktree list`
#[derive(Debug, Clone)]
pub struct GitWorktreeInfo {
    /// Path to the worktree
    pub path: PathBuf,
    /// HEAD commit
    pub head: String,
    /// Branch name (if any)
    pub branch: Option<String>,
    /// Whether this is bare
    pub bare: bool,
}

impl WorktreeManager {
    /// Create a new worktree manager
    pub fn new(repo_path: impl Into<PathBuf>) -> Self {
        let repo_path = repo_path.into();
        let worktree_base = repo_path.join(".worktrees");

        Self {
            repo_path,
            worktree_base,
            worktrees: RwLock::new(HashMap::new()),
            config: WorktreeConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(repo_path: impl Into<PathBuf>, config: WorktreeConfig) -> Self {
        let repo_path = repo_path.into();
        let worktree_base = repo_path.join(".worktrees");

        Self {
            repo_path,
            worktree_base,
            worktrees: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Set a custom base directory for worktrees
    pub fn with_worktree_base(mut self, base: impl Into<PathBuf>) -> Self {
        self.worktree_base = base.into();
        self
    }

    /// Create or get a worktree for an agent
    pub async fn get_or_create_worktree(
        &self,
        agent_id: &str,
        branch: &str,
        purpose: &str,
    ) -> WorktreeResult {
        // Check if agent already has a worktree
        {
            let worktrees = self.worktrees.read().await;
            if let Some(existing) = worktrees.get(agent_id) {
                return WorktreeResult::AlreadyExists(existing.clone());
            }
        }

        // Check worktree limit
        let worktrees = self.worktrees.read().await;
        if worktrees.len() >= self.config.max_worktrees {
            drop(worktrees);

            // Try cleanup if auto-cleanup is enabled
            if self.config.auto_cleanup {
                self.cleanup_stale_worktrees().await;

                let worktrees = self.worktrees.read().await;
                if worktrees.len() >= self.config.max_worktrees {
                    return WorktreeResult::Error(format!(
                        "Maximum worktrees ({}) reached",
                        self.config.max_worktrees
                    ));
                }
            } else {
                return WorktreeResult::Error(format!(
                    "Maximum worktrees ({}) reached",
                    self.config.max_worktrees
                ));
            }
        } else {
            drop(worktrees);
        }

        // Create the worktree
        self.create_worktree(agent_id, branch, purpose).await
    }

    /// Create a new worktree for an agent
    async fn create_worktree(&self, agent_id: &str, branch: &str, purpose: &str) -> WorktreeResult {
        // Ensure base directory exists
        if let Err(e) = std::fs::create_dir_all(&self.worktree_base) {
            return WorktreeResult::Error(format!("Failed to create worktree base: {}", e));
        }

        // Generate unique worktree path
        let worktree_name = format!(
            "{}{}",
            self.config.prefix,
            agent_id.replace(['/', '\\', ' '], "-")
        );
        let worktree_path = self.worktree_base.join(&worktree_name);

        // Check if branch exists
        let branch_exists = self.branch_exists(branch);

        // Build git worktree command
        let mut cmd = Command::new("git");
        cmd.current_dir(&self.repo_path).arg("worktree").arg("add");

        if branch_exists {
            // Checkout existing branch
            cmd.arg(&worktree_path).arg(branch);
        } else {
            // Create new branch from current HEAD
            cmd.arg("-b").arg(branch).arg(&worktree_path);
        }

        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => {
                return WorktreeResult::Error(format!("Failed to run git worktree add: {}", e));
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return WorktreeResult::Error(format!("git worktree add failed: {}", stderr));
        }

        // Create the worktree record
        let worktree = AgentWorktree {
            agent_id: agent_id.to_string(),
            path: worktree_path,
            branch: branch.to_string(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            has_changes: false,
            purpose: purpose.to_string(),
        };

        // Store in our tracking map
        self.worktrees
            .write()
            .await
            .insert(agent_id.to_string(), worktree.clone());

        WorktreeResult::Created(worktree)
    }

    /// Check if a branch exists
    fn branch_exists(&self, branch: &str) -> bool {
        let output = Command::new("git")
            .current_dir(&self.repo_path)
            .args(["rev-parse", "--verify", &format!("refs/heads/{}", branch)])
            .output();

        matches!(output, Ok(o) if o.status.success())
    }

    /// Get an agent's worktree
    pub async fn get_worktree(&self, agent_id: &str) -> Option<AgentWorktree> {
        let mut worktrees = self.worktrees.write().await;
        if let Some(worktree) = worktrees.get_mut(agent_id) {
            // Update last accessed time
            worktree.last_accessed = Instant::now();
            Some(worktree.clone())
        } else {
            None
        }
    }

    /// Remove an agent's worktree
    pub async fn remove_worktree(&self, agent_id: &str, force: bool) -> WorktreeResult {
        let worktree = {
            let worktrees = self.worktrees.read().await;
            worktrees.get(agent_id).cloned()
        };

        let worktree = match worktree {
            Some(wt) => wt,
            None => {
                return WorktreeResult::Error(format!("No worktree found for agent {}", agent_id));
            }
        };

        // Check for uncommitted changes
        if !force && self.has_uncommitted_changes(&worktree.path) {
            return WorktreeResult::Error(
                "Worktree has uncommitted changes. Use force=true to remove anyway.".to_string(),
            );
        }

        // Remove the worktree
        let mut cmd = Command::new("git");
        cmd.current_dir(&self.repo_path)
            .args(["worktree", "remove"]);

        if force {
            cmd.arg("--force");
        }

        cmd.arg(&worktree.path);

        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => {
                return WorktreeResult::Error(format!("Failed to run git worktree remove: {}", e));
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return WorktreeResult::Error(format!("git worktree remove failed: {}", stderr));
        }

        // Remove from tracking
        self.worktrees.write().await.remove(agent_id);

        WorktreeResult::Removed {
            path: worktree.path,
        }
    }

    /// Check if a worktree has uncommitted changes
    fn has_uncommitted_changes(&self, worktree_path: &Path) -> bool {
        let output = Command::new("git")
            .current_dir(worktree_path)
            .args(["status", "--porcelain"])
            .output();

        match output {
            Ok(o) if o.status.success() => !o.stdout.is_empty(),
            _ => false, // If we can't check, assume no changes
        }
    }

    /// Cleanup stale worktrees
    pub async fn cleanup_stale_worktrees(&self) -> Vec<String> {
        let mut removed = Vec::new();
        let stale_agents: Vec<String> = {
            let worktrees = self.worktrees.read().await;
            worktrees
                .iter()
                .filter(|(_, wt)| wt.is_stale(self.config.max_age) && !wt.has_changes)
                .map(|(id, _)| id.clone())
                .collect()
        };

        for agent_id in stale_agents {
            if let WorktreeResult::Removed { .. } = self.remove_worktree(&agent_id, false).await {
                removed.push(agent_id);
            }
        }

        removed
    }

    /// List all worktrees (both tracked and untracked)
    pub async fn list_all_worktrees(&self) -> Result<Vec<GitWorktreeInfo>, String> {
        let output = Command::new("git")
            .current_dir(&self.repo_path)
            .args(["worktree", "list", "--porcelain"])
            .output()
            .map_err(|e| format!("Failed to run git worktree list: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git worktree list failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_worktree_list(&stdout))
    }

    /// Parse the porcelain output of `git worktree list`
    fn parse_worktree_list(output: &str) -> Vec<GitWorktreeInfo> {
        let mut worktrees = Vec::new();
        let mut current: Option<GitWorktreeInfo> = None;

        for line in output.lines() {
            if line.starts_with("worktree ") {
                // Save previous entry if any
                if let Some(wt) = current.take() {
                    worktrees.push(wt);
                }
                // Start new entry
                current = Some(GitWorktreeInfo {
                    path: PathBuf::from(line.trim_start_matches("worktree ")),
                    head: String::new(),
                    branch: None,
                    bare: false,
                });
            } else if let Some(ref mut wt) = current {
                if line.starts_with("HEAD ") {
                    wt.head = line.trim_start_matches("HEAD ").to_string();
                } else if line.starts_with("branch refs/heads/") {
                    wt.branch = Some(line.trim_start_matches("branch refs/heads/").to_string());
                } else if line == "bare" {
                    wt.bare = true;
                }
            }
        }

        // Don't forget the last entry
        if let Some(wt) = current {
            worktrees.push(wt);
        }

        worktrees
    }

    /// Get tracked worktrees
    pub async fn list_tracked_worktrees(&self) -> Vec<AgentWorktree> {
        self.worktrees.read().await.values().cloned().collect()
    }

    /// Sync tracking with actual git worktrees
    pub async fn sync_with_git(&self) -> Result<SyncResult, String> {
        let git_worktrees = self.list_all_worktrees().await?;
        let mut tracked = self.worktrees.write().await;

        let mut added = 0;
        let mut removed = 0;

        // Find worktrees that exist in git but not in tracking
        for git_wt in &git_worktrees {
            // Skip the main worktree (no branch usually) and bare repos
            if git_wt.bare || git_wt.branch.is_none() {
                continue;
            }

            // Check if the path contains our prefix
            let path_str = git_wt.path.to_string_lossy();
            if path_str.contains(&self.config.prefix) {
                // Try to extract agent ID from path
                if let Some(name) = git_wt.path.file_name() {
                    let name_str = name.to_string_lossy();
                    if let Some(agent_id) = name_str.strip_prefix(&self.config.prefix)
                        && !tracked.contains_key(agent_id)
                    {
                        tracked.insert(
                            agent_id.to_string(),
                            AgentWorktree {
                                agent_id: agent_id.to_string(),
                                path: git_wt.path.clone(),
                                branch: git_wt.branch.clone().unwrap_or_default(),
                                created_at: Instant::now(),
                                last_accessed: Instant::now(),
                                has_changes: false,
                                purpose: "Discovered via sync".to_string(),
                            },
                        );
                        added += 1;
                    }
                }
            }
        }

        // Find tracked worktrees that no longer exist
        let git_paths: std::collections::HashSet<_> =
            git_worktrees.iter().map(|wt| &wt.path).collect();
        let to_remove: Vec<_> = tracked
            .iter()
            .filter(|(_, wt)| !git_paths.contains(&wt.path))
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            tracked.remove(&id);
            removed += 1;
        }

        Ok(SyncResult { added, removed })
    }

    /// Update the has_changes flag for a worktree
    pub async fn update_changes_status(&self, agent_id: &str) -> bool {
        let mut worktrees = self.worktrees.write().await;
        if let Some(worktree) = worktrees.get_mut(agent_id) {
            worktree.has_changes = self.has_uncommitted_changes(&worktree.path);
            worktree.has_changes
        } else {
            false
        }
    }

    /// Get the working directory for an agent
    pub async fn get_working_directory(&self, agent_id: &str) -> PathBuf {
        let worktrees = self.worktrees.read().await;
        if let Some(worktree) = worktrees.get(agent_id) {
            worktree.path.clone()
        } else {
            self.repo_path.clone()
        }
    }

    /// Get statistics about worktrees
    pub async fn get_stats(&self) -> WorktreeStats {
        let worktrees = self.worktrees.read().await;

        WorktreeStats {
            total_tracked: worktrees.len(),
            with_changes: worktrees.values().filter(|wt| wt.has_changes).count(),
            stale: worktrees
                .values()
                .filter(|wt| wt.is_stale(self.config.max_age))
                .count(),
            max_allowed: self.config.max_worktrees,
        }
    }
}

/// Result of syncing tracking with git
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of worktrees added to tracking
    pub added: usize,
    /// Number of worktrees removed from tracking
    pub removed: usize,
}

/// Statistics about worktrees
#[derive(Debug, Clone)]
pub struct WorktreeStats {
    /// Total tracked worktrees
    pub total_tracked: usize,
    /// Worktrees with uncommitted changes
    pub with_changes: usize,
    /// Stale worktrees
    pub stale: usize,
    /// Maximum allowed worktrees
    pub max_allowed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_worktree_config_default() {
        let config = WorktreeConfig::default();
        assert_eq!(config.max_worktrees, 10);
        assert!(config.auto_cleanup);
        assert_eq!(config.prefix, "agent-wt-");
    }

    #[test]
    fn test_agent_worktree_staleness() {
        let worktree = AgentWorktree {
            agent_id: "test-agent".to_string(),
            path: PathBuf::from("/tmp/test"),
            branch: "feature".to_string(),
            created_at: Instant::now() - Duration::from_secs(3600),
            last_accessed: Instant::now() - Duration::from_secs(3600),
            has_changes: false,
            purpose: "test".to_string(),
        };

        // Should be stale after 30 minutes
        assert!(worktree.is_stale(Duration::from_secs(1800)));
        // Should not be stale after 2 hours
        assert!(!worktree.is_stale(Duration::from_secs(7200)));
    }

    #[test]
    fn test_parse_worktree_list() {
        let output = r#"worktree /home/user/repo
HEAD abc123
branch refs/heads/main

worktree /home/user/repo/.worktrees/feature
HEAD def456
branch refs/heads/feature
"#;

        let worktrees = WorktreeManager::parse_worktree_list(output);
        assert_eq!(worktrees.len(), 2);

        assert_eq!(worktrees[0].path, PathBuf::from("/home/user/repo"));
        assert_eq!(worktrees[0].head, "abc123");
        assert_eq!(worktrees[0].branch, Some("main".to_string()));

        assert_eq!(
            worktrees[1].path,
            PathBuf::from("/home/user/repo/.worktrees/feature")
        );
        assert_eq!(worktrees[1].branch, Some("feature".to_string()));
    }

    #[test]
    fn test_worktree_result_success() {
        let worktree = AgentWorktree {
            agent_id: "test".to_string(),
            path: PathBuf::from("/tmp/test"),
            branch: "main".to_string(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            has_changes: false,
            purpose: "test".to_string(),
        };

        let created = WorktreeResult::Created(worktree.clone());
        assert!(created.is_success());
        assert!(created.worktree().is_some());

        let exists = WorktreeResult::AlreadyExists(worktree);
        assert!(exists.is_success());

        let removed = WorktreeResult::Removed {
            path: PathBuf::from("/tmp/test"),
        };
        assert!(removed.is_success());

        let error = WorktreeResult::Error("test error".to_string());
        assert!(!error.is_success());
        assert!(error.worktree().is_none());
    }

    #[tokio::test]
    async fn test_worktree_manager_creation() {
        let temp_dir = env::temp_dir().join("test-worktree-manager");
        let manager = WorktreeManager::new(&temp_dir);

        assert_eq!(manager.repo_path, temp_dir);
        assert_eq!(manager.worktree_base, temp_dir.join(".worktrees"));
    }

    #[tokio::test]
    async fn test_worktree_stats() {
        let temp_dir = env::temp_dir().join("test-worktree-stats");
        let manager = WorktreeManager::new(&temp_dir);

        // Add some test worktrees directly to the tracking map
        {
            let mut worktrees = manager.worktrees.write().await;
            worktrees.insert(
                "agent-1".to_string(),
                AgentWorktree {
                    agent_id: "agent-1".to_string(),
                    path: PathBuf::from("/tmp/wt1"),
                    branch: "feature-1".to_string(),
                    created_at: Instant::now(),
                    last_accessed: Instant::now(),
                    has_changes: false,
                    purpose: "test".to_string(),
                },
            );
            worktrees.insert(
                "agent-2".to_string(),
                AgentWorktree {
                    agent_id: "agent-2".to_string(),
                    path: PathBuf::from("/tmp/wt2"),
                    branch: "feature-2".to_string(),
                    created_at: Instant::now(),
                    last_accessed: Instant::now(),
                    has_changes: true, // Has changes
                    purpose: "test".to_string(),
                },
            );
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_tracked, 2);
        assert_eq!(stats.with_changes, 1);
        assert_eq!(stats.max_allowed, 10);
    }

    #[tokio::test]
    async fn test_get_working_directory() {
        let temp_dir = env::temp_dir().join("test-working-dir");
        let manager = WorktreeManager::new(&temp_dir);

        // Without worktree, should return repo path
        let dir = manager.get_working_directory("unknown-agent").await;
        assert_eq!(dir, temp_dir);

        // Add a worktree
        {
            let mut worktrees = manager.worktrees.write().await;
            worktrees.insert(
                "agent-1".to_string(),
                AgentWorktree {
                    agent_id: "agent-1".to_string(),
                    path: PathBuf::from("/tmp/agent-1-worktree"),
                    branch: "feature".to_string(),
                    created_at: Instant::now(),
                    last_accessed: Instant::now(),
                    has_changes: false,
                    purpose: "test".to_string(),
                },
            );
        }

        // With worktree, should return worktree path
        let dir = manager.get_working_directory("agent-1").await;
        assert_eq!(dir, PathBuf::from("/tmp/agent-1-worktree"));
    }

    #[tokio::test]
    async fn test_list_tracked_worktrees() {
        let temp_dir = env::temp_dir().join("test-list-tracked");
        let manager = WorktreeManager::new(&temp_dir);

        // Add test worktrees
        {
            let mut worktrees = manager.worktrees.write().await;
            for i in 0..3 {
                worktrees.insert(
                    format!("agent-{}", i),
                    AgentWorktree {
                        agent_id: format!("agent-{}", i),
                        path: PathBuf::from(format!("/tmp/wt{}", i)),
                        branch: format!("feature-{}", i),
                        created_at: Instant::now(),
                        last_accessed: Instant::now(),
                        has_changes: false,
                        purpose: "test".to_string(),
                    },
                );
            }
        }

        let tracked = manager.list_tracked_worktrees().await;
        assert_eq!(tracked.len(), 3);
    }

    #[test]
    fn test_worktree_age() {
        let worktree = AgentWorktree {
            agent_id: "test".to_string(),
            path: PathBuf::from("/tmp/test"),
            branch: "main".to_string(),
            created_at: Instant::now() - Duration::from_secs(120),
            last_accessed: Instant::now(),
            has_changes: false,
            purpose: "test".to_string(),
        };

        let age = worktree.age();
        assert!(age >= Duration::from_secs(119)); // Allow for slight timing variations
    }
}
