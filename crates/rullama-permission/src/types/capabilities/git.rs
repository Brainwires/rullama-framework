use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Git operation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCapabilities {
    /// Allowed operations
    #[serde(default = "default_git_ops")]
    pub allowed_ops: HashSet<GitOperation>,

    /// Protected branches (cannot push directly)
    #[serde(default)]
    pub protected_branches: Vec<String>,

    /// Can force push (dangerous)
    #[serde(default)]
    pub can_force_push: bool,

    /// Can perform destructive operations
    #[serde(default)]
    pub can_destructive: bool,

    /// Require PR for these branches
    #[serde(default)]
    pub require_pr_branches: Vec<String>,
}

fn default_git_ops() -> HashSet<GitOperation> {
    let mut ops = HashSet::new();
    ops.insert(GitOperation::Status);
    ops.insert(GitOperation::Diff);
    ops.insert(GitOperation::Log);
    ops
}

impl Default for GitCapabilities {
    fn default() -> Self {
        Self {
            allowed_ops: default_git_ops(),
            protected_branches: vec!["main".to_string(), "master".to_string()],
            can_force_push: false,
            can_destructive: false,
            require_pr_branches: Vec::new(),
        }
    }
}

impl GitCapabilities {
    /// Create read-only git capabilities
    pub fn read_only() -> Self {
        let mut ops = HashSet::new();
        ops.insert(GitOperation::Status);
        ops.insert(GitOperation::Diff);
        ops.insert(GitOperation::Log);
        ops.insert(GitOperation::Fetch);

        Self {
            allowed_ops: ops,
            protected_branches: vec!["main".to_string(), "master".to_string()],
            can_force_push: false,
            can_destructive: false,
            require_pr_branches: Vec::new(),
        }
    }

    /// Create standard git capabilities
    pub fn standard() -> Self {
        let mut ops = HashSet::new();
        ops.insert(GitOperation::Status);
        ops.insert(GitOperation::Diff);
        ops.insert(GitOperation::Log);
        ops.insert(GitOperation::Add);
        ops.insert(GitOperation::Commit);
        ops.insert(GitOperation::Push);
        ops.insert(GitOperation::Pull);
        ops.insert(GitOperation::Fetch);
        ops.insert(GitOperation::Branch);
        ops.insert(GitOperation::Checkout);
        ops.insert(GitOperation::Stash);

        Self {
            allowed_ops: ops,
            protected_branches: vec!["main".to_string(), "master".to_string()],
            can_force_push: false,
            can_destructive: false,
            require_pr_branches: Vec::new(),
        }
    }

    /// Create full git capabilities
    pub fn full() -> Self {
        let mut ops = HashSet::new();
        ops.insert(GitOperation::Status);
        ops.insert(GitOperation::Diff);
        ops.insert(GitOperation::Log);
        ops.insert(GitOperation::Add);
        ops.insert(GitOperation::Commit);
        ops.insert(GitOperation::Push);
        ops.insert(GitOperation::Pull);
        ops.insert(GitOperation::Fetch);
        ops.insert(GitOperation::Branch);
        ops.insert(GitOperation::Checkout);
        ops.insert(GitOperation::Merge);
        ops.insert(GitOperation::Rebase);
        ops.insert(GitOperation::Reset);
        ops.insert(GitOperation::Stash);
        ops.insert(GitOperation::Tag);
        ops.insert(GitOperation::ForcePush);

        Self {
            allowed_ops: ops,
            protected_branches: Vec::new(),
            can_force_push: true,
            can_destructive: true,
            require_pr_branches: Vec::new(),
        }
    }
}

/// Git operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GitOperation {
    /// View working tree status.
    Status,
    /// Show changes between commits.
    Diff,
    /// View commit history.
    Log,
    /// Stage changes.
    Add,
    /// Create a commit.
    Commit,
    /// Push to remote.
    Push,
    /// Pull from remote.
    Pull,
    /// Fetch from remote.
    Fetch,
    /// Branch operations.
    Branch,
    /// Switch branches.
    Checkout,
    /// Merge branches.
    Merge,
    /// Rebase commits.
    Rebase,
    /// Reset to a previous state.
    Reset,
    /// Stash changes.
    Stash,
    /// Tag a commit.
    Tag,
    /// Force push to remote.
    ForcePush,
}

impl GitOperation {
    /// Check if this operation is destructive
    pub fn is_destructive(&self) -> bool {
        matches!(
            self,
            GitOperation::Rebase
                | GitOperation::Reset
                | GitOperation::ForcePush
                | GitOperation::Merge
        )
    }
}
