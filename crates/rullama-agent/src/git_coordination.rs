//! Git operation coordination for multi-agent systems
//!
//! Maps git tool operations to their required resource locks and
//! coordinates with the resource lock manager to ensure safe concurrent access.
//!
//! ## Lock Requirements by Git Operation
//!
//! | Git Tool | Required Locks | Notes |
//! |----------|---------------|-------|
//! | git_status, git_diff, git_log, git_search, git_fetch | None | Read-only |
//! | git_stage, git_unstage | GitIndex | Modifies staging area |
//! | git_commit | GitIndex, GitCommit | Creates commit |
//! | git_push | GitRemoteWrite | Writes to remote |
//! | git_pull | GitRemoteMerge, GitIndex | Reads remote, modifies working tree |
//! | git_branch | GitBranch | Branch operations |
//! | git_discard | GitDestructive, GitIndex | Dangerous: loses changes |

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::communication::{AgentMessage, CommunicationHub, GitOperationType};
use crate::resource_checker::ResourceChecker;
use crate::resource_locks::{ResourceLockGuard, ResourceLockManager, ResourceScope, ResourceType};

/// Git tool name constants
pub mod git_tools {
    /// Git status tool name.
    pub const STATUS: &str = "git_status";
    /// Git diff tool name.
    pub const DIFF: &str = "git_diff";
    /// Git log tool name.
    pub const LOG: &str = "git_log";
    /// Git search tool name.
    pub const SEARCH: &str = "git_search";
    /// Git fetch tool name.
    pub const FETCH: &str = "git_fetch";
    /// Git stage tool name.
    pub const STAGE: &str = "git_stage";
    /// Git unstage tool name.
    pub const UNSTAGE: &str = "git_unstage";
    /// Git commit tool name.
    pub const COMMIT: &str = "git_commit";
    /// Git push tool name.
    pub const PUSH: &str = "git_push";
    /// Git pull tool name.
    pub const PULL: &str = "git_pull";
    /// Git branch tool name.
    pub const BRANCH: &str = "git_branch";
    /// Git discard tool name.
    pub const DISCARD: &str = "git_discard";
}

/// Lock requirements for a git operation
#[derive(Debug, Clone)]
pub struct GitLockRequirements {
    /// Primary resource types needed
    pub resource_types: Vec<ResourceType>,
    /// Whether to check for file write conflicts
    pub check_file_conflicts: bool,
    /// Whether to check for build conflicts
    pub check_build_conflicts: bool,
    /// Git operation type for messaging
    pub operation_type: GitOperationType,
    /// Human-readable description
    pub description: &'static str,
}

impl GitLockRequirements {
    /// Returns true if no locks are needed (read-only operation)
    pub fn is_read_only(&self) -> bool {
        self.resource_types.is_empty()
    }
}

/// Get the lock requirements for a git tool
pub fn get_lock_requirements(tool_name: &str) -> GitLockRequirements {
    match tool_name {
        // Read-only operations - no locks needed
        git_tools::STATUS | git_tools::DIFF | git_tools::LOG | git_tools::SEARCH => {
            GitLockRequirements {
                resource_types: vec![],
                check_file_conflicts: false,
                check_build_conflicts: false,
                operation_type: GitOperationType::ReadOnly,
                description: "Read-only git operation",
            }
        }
        git_tools::FETCH => GitLockRequirements {
            resource_types: vec![],
            check_file_conflicts: false,
            check_build_conflicts: false,
            operation_type: GitOperationType::ReadOnly,
            description: "Fetch from remote",
        },

        // Staging operations
        git_tools::STAGE | git_tools::UNSTAGE => GitLockRequirements {
            resource_types: vec![ResourceType::GitIndex],
            check_file_conflicts: true, // Wait for files being edited
            check_build_conflicts: false,
            operation_type: GitOperationType::Staging,
            description: "Staging area modification",
        },

        // Commit operation
        git_tools::COMMIT => GitLockRequirements {
            resource_types: vec![ResourceType::GitIndex, ResourceType::GitCommit],
            check_file_conflicts: true,  // Wait for files being edited
            check_build_conflicts: true, // Wait for builds to complete
            operation_type: GitOperationType::Commit,
            description: "Create commit",
        },

        // Remote write operation
        git_tools::PUSH => GitLockRequirements {
            resource_types: vec![ResourceType::GitRemoteWrite],
            check_file_conflicts: false,
            check_build_conflicts: true, // Don't push during active build
            operation_type: GitOperationType::RemoteWrite,
            description: "Push to remote",
        },

        // Remote merge operation
        git_tools::PULL => GitLockRequirements {
            resource_types: vec![ResourceType::GitRemoteMerge, ResourceType::GitIndex],
            check_file_conflicts: true, // Wait for files being edited (pull modifies working tree)
            check_build_conflicts: true, // Don't pull during active build
            operation_type: GitOperationType::RemoteMerge,
            description: "Pull from remote",
        },

        // Branch operation
        git_tools::BRANCH => GitLockRequirements {
            resource_types: vec![ResourceType::GitBranch],
            check_file_conflicts: false,
            check_build_conflicts: false,
            operation_type: GitOperationType::Branch,
            description: "Branch operation",
        },

        // Destructive operation
        git_tools::DISCARD => GitLockRequirements {
            resource_types: vec![ResourceType::GitDestructive, ResourceType::GitIndex],
            check_file_conflicts: true,  // Wait for files being edited
            check_build_conflicts: true, // Wait for builds
            operation_type: GitOperationType::Destructive,
            description: "Discard changes (destructive)",
        },

        // Unknown operation - treat as read-only for safety
        _ => GitLockRequirements {
            resource_types: vec![],
            check_file_conflicts: false,
            check_build_conflicts: false,
            operation_type: GitOperationType::ReadOnly,
            description: "Unknown git operation",
        },
    }
}

/// Coordinator for git operations across agents
pub struct GitCoordinator {
    resource_locks: Arc<ResourceLockManager>,
    resource_checker: Option<Arc<ResourceChecker>>,
    communication_hub: Option<Arc<CommunicationHub>>,
    project_root: PathBuf,
}

impl GitCoordinator {
    /// Create a new git coordinator
    pub fn new(resource_locks: Arc<ResourceLockManager>, project_root: PathBuf) -> Self {
        Self {
            resource_locks,
            resource_checker: None,
            communication_hub: None,
            project_root,
        }
    }

    /// Create a git coordinator with full integration
    pub fn with_full_integration(
        resource_locks: Arc<ResourceLockManager>,
        resource_checker: Arc<ResourceChecker>,
        communication_hub: Arc<CommunicationHub>,
        project_root: PathBuf,
    ) -> Self {
        Self {
            resource_locks,
            resource_checker: Some(resource_checker),
            communication_hub: Some(communication_hub),
            project_root,
        }
    }

    /// Get the project scope for this coordinator
    pub fn project_scope(&self) -> ResourceScope {
        ResourceScope::Project(self.project_root.clone())
    }

    /// Acquire all locks needed for a git operation
    ///
    /// Returns a vector of lock guards that must be held during the operation.
    /// The guards are released when dropped.
    #[tracing::instrument(name = "agent.git.acquire", skip(self))]
    pub async fn acquire_for_git_op(
        &self,
        agent_id: &str,
        tool_name: &str,
    ) -> Result<GitOperationLocks> {
        let requirements = get_lock_requirements(tool_name);

        // If read-only, no locks needed
        if requirements.is_read_only() {
            return Ok(GitOperationLocks {
                guards: vec![],
                operation_type: requirements.operation_type,
                description: requirements.description.to_string(),
            });
        }

        let scope = self.project_scope();

        // Check for cross-resource conflicts if resource checker is available
        if let Some(checker) = &self.resource_checker {
            // Check file conflicts
            if requirements.check_file_conflicts {
                let git_op_type = match requirements.operation_type {
                    GitOperationType::Staging => ResourceType::GitIndex,
                    GitOperationType::Commit => ResourceType::GitCommit,
                    GitOperationType::RemoteWrite => ResourceType::GitRemoteWrite,
                    GitOperationType::RemoteMerge => ResourceType::GitRemoteMerge,
                    GitOperationType::Branch => ResourceType::GitBranch,
                    GitOperationType::Destructive => ResourceType::GitDestructive,
                    GitOperationType::ReadOnly => ResourceType::GitIndex, // fallback
                };

                let conflict_check = checker
                    .can_start_git_operation(git_op_type, &scope, agent_id)
                    .await;

                if conflict_check.is_blocked() {
                    let conflicts: Vec<String> = conflict_check
                        .conflicts()
                        .iter()
                        .map(|c| format!("{}: {} by {}", c.resource, c.status, c.holder_agent))
                        .collect();

                    return Err(anyhow::anyhow!(
                        "Git operation blocked by conflicts: {}",
                        conflicts.join(", ")
                    ));
                }
            }
        }

        // Broadcast operation start if communication hub is available
        if let Some(hub) = &self.communication_hub {
            let _ = hub
                .broadcast(
                    agent_id.to_string(),
                    AgentMessage::GitOperationStarted {
                        agent_id: agent_id.to_string(),
                        git_op: requirements.operation_type,
                        branch: None, // Could be enhanced to include branch info
                        description: requirements.description.to_string(),
                    },
                )
                .await;
        }

        // Acquire all required locks
        let mut guards = Vec::new();
        for resource_type in &requirements.resource_types {
            let guard = self
                .resource_locks
                .acquire_resource(
                    agent_id,
                    *resource_type,
                    scope.clone(),
                    requirements.description,
                )
                .await?;
            guards.push(guard);
        }

        Ok(GitOperationLocks {
            guards,
            operation_type: requirements.operation_type,
            description: requirements.description.to_string(),
        })
    }

    /// Check if a git operation can be performed without blocking
    pub async fn can_perform_git_op(&self, agent_id: &str, tool_name: &str) -> bool {
        let requirements = get_lock_requirements(tool_name);

        // Read-only operations can always proceed
        if requirements.is_read_only() {
            return true;
        }

        let scope = self.project_scope();

        // Check resource locks
        for resource_type in &requirements.resource_types {
            if !self
                .resource_locks
                .can_acquire(agent_id, *resource_type, &scope)
                .await
            {
                return false;
            }
        }

        // Check cross-resource conflicts
        if let Some(checker) = &self.resource_checker
            && requirements.check_file_conflicts
        {
            let git_op_type = match requirements.operation_type {
                GitOperationType::Staging => ResourceType::GitIndex,
                GitOperationType::Commit => ResourceType::GitCommit,
                GitOperationType::RemoteWrite => ResourceType::GitRemoteWrite,
                GitOperationType::RemoteMerge => ResourceType::GitRemoteMerge,
                GitOperationType::Branch => ResourceType::GitBranch,
                GitOperationType::Destructive => ResourceType::GitDestructive,
                GitOperationType::ReadOnly => return true,
            };

            let check = checker
                .can_start_git_operation(git_op_type, &scope, agent_id)
                .await;

            if check.is_blocked() {
                return false;
            }
        }

        true
    }

    /// Broadcast that a git operation has completed
    pub async fn broadcast_completion(
        &self,
        agent_id: &str,
        operation_type: GitOperationType,
        success: bool,
        summary: &str,
    ) {
        if let Some(hub) = &self.communication_hub {
            let _ = hub
                .broadcast(
                    agent_id.to_string(),
                    AgentMessage::GitOperationCompleted {
                        agent_id: agent_id.to_string(),
                        git_op: operation_type,
                        success,
                        summary: summary.to_string(),
                    },
                )
                .await;
        }
    }
}

/// Holds locks for a git operation
///
/// The locks are released when this struct is dropped.
pub struct GitOperationLocks {
    guards: Vec<ResourceLockGuard>,
    /// Type of git operation.
    pub operation_type: GitOperationType,
    /// Human-readable description.
    pub description: String,
}

impl GitOperationLocks {
    /// Returns true if this represents a read-only operation (no locks held)
    pub fn is_read_only(&self) -> bool {
        self.guards.is_empty()
    }

    /// Get the number of locks held
    pub fn lock_count(&self) -> usize {
        self.guards.len()
    }
}

/// Helper trait for git operations with automatic lock management
#[async_trait::async_trait]
pub trait GitOperationRunner {
    /// Run a git operation with automatic lock acquisition and release
    async fn run_with_locks<F, T>(
        &self,
        agent_id: &str,
        tool_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send,
        T: Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_only_operations() {
        assert!(get_lock_requirements(git_tools::STATUS).is_read_only());
        assert!(get_lock_requirements(git_tools::DIFF).is_read_only());
        assert!(get_lock_requirements(git_tools::LOG).is_read_only());
        assert!(get_lock_requirements(git_tools::SEARCH).is_read_only());
        assert!(get_lock_requirements(git_tools::FETCH).is_read_only());
    }

    #[test]
    fn test_staging_operations() {
        let stage_req = get_lock_requirements(git_tools::STAGE);
        assert!(!stage_req.is_read_only());
        assert!(stage_req.resource_types.contains(&ResourceType::GitIndex));
        assert!(matches!(
            stage_req.operation_type,
            GitOperationType::Staging
        ));

        let unstage_req = get_lock_requirements(git_tools::UNSTAGE);
        assert!(!unstage_req.is_read_only());
        assert!(unstage_req.resource_types.contains(&ResourceType::GitIndex));
    }

    #[test]
    fn test_commit_operation() {
        let req = get_lock_requirements(git_tools::COMMIT);
        assert!(!req.is_read_only());
        assert!(req.resource_types.contains(&ResourceType::GitIndex));
        assert!(req.resource_types.contains(&ResourceType::GitCommit));
        assert!(req.check_file_conflicts);
        assert!(req.check_build_conflicts);
        assert!(matches!(req.operation_type, GitOperationType::Commit));
    }

    #[test]
    fn test_push_operation() {
        let req = get_lock_requirements(git_tools::PUSH);
        assert!(!req.is_read_only());
        assert!(req.resource_types.contains(&ResourceType::GitRemoteWrite));
        assert!(!req.check_file_conflicts);
        assert!(req.check_build_conflicts);
        assert!(matches!(req.operation_type, GitOperationType::RemoteWrite));
    }

    #[test]
    fn test_pull_operation() {
        let req = get_lock_requirements(git_tools::PULL);
        assert!(!req.is_read_only());
        assert!(req.resource_types.contains(&ResourceType::GitRemoteMerge));
        assert!(req.resource_types.contains(&ResourceType::GitIndex));
        assert!(req.check_file_conflicts);
        assert!(req.check_build_conflicts);
        assert!(matches!(req.operation_type, GitOperationType::RemoteMerge));
    }

    #[test]
    fn test_destructive_operation() {
        let req = get_lock_requirements(git_tools::DISCARD);
        assert!(!req.is_read_only());
        assert!(req.resource_types.contains(&ResourceType::GitDestructive));
        assert!(req.resource_types.contains(&ResourceType::GitIndex));
        assert!(req.check_file_conflicts);
        assert!(req.check_build_conflicts);
        assert!(matches!(req.operation_type, GitOperationType::Destructive));
    }

    #[test]
    fn test_unknown_operation() {
        let req = get_lock_requirements("unknown_git_tool");
        assert!(req.is_read_only());
    }

    #[tokio::test]
    async fn test_coordinator_read_only_no_locks() {
        let resource_locks = Arc::new(ResourceLockManager::new());
        let coordinator = GitCoordinator::new(resource_locks, PathBuf::from("/test/project"));

        let locks = coordinator
            .acquire_for_git_op("agent-1", git_tools::STATUS)
            .await
            .unwrap();

        assert!(locks.is_read_only());
        assert_eq!(locks.lock_count(), 0);
    }

    #[tokio::test]
    async fn test_coordinator_staging_acquires_index_lock() {
        let resource_locks = Arc::new(ResourceLockManager::new());
        let coordinator =
            GitCoordinator::new(resource_locks.clone(), PathBuf::from("/test/project"));

        let locks = coordinator
            .acquire_for_git_op("agent-1", git_tools::STAGE)
            .await
            .unwrap();

        assert!(!locks.is_read_only());
        assert_eq!(locks.lock_count(), 1);
        assert!(matches!(locks.operation_type, GitOperationType::Staging));

        // Verify the lock is held
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));
        assert!(
            !resource_locks
                .can_acquire("agent-2", ResourceType::GitIndex, &scope)
                .await
        );
    }

    #[tokio::test]
    async fn test_coordinator_commit_acquires_multiple_locks() {
        let resource_locks = Arc::new(ResourceLockManager::new());
        let coordinator =
            GitCoordinator::new(resource_locks.clone(), PathBuf::from("/test/project"));

        let locks = coordinator
            .acquire_for_git_op("agent-1", git_tools::COMMIT)
            .await
            .unwrap();

        assert!(!locks.is_read_only());
        assert_eq!(locks.lock_count(), 2); // GitIndex + GitCommit
        assert!(matches!(locks.operation_type, GitOperationType::Commit));
    }

    #[tokio::test]
    async fn test_can_perform_git_op() {
        let resource_locks = Arc::new(ResourceLockManager::new());
        let coordinator =
            GitCoordinator::new(resource_locks.clone(), PathBuf::from("/test/project"));

        // Read-only should always be allowed
        assert!(
            coordinator
                .can_perform_git_op("agent-1", git_tools::STATUS)
                .await
        );

        // Staging should be allowed when no conflicts
        assert!(
            coordinator
                .can_perform_git_op("agent-1", git_tools::STAGE)
                .await
        );

        // Agent 1 acquires the index lock
        let _locks = coordinator
            .acquire_for_git_op("agent-1", git_tools::STAGE)
            .await
            .unwrap();

        // Agent 2 should not be able to stage
        assert!(
            !coordinator
                .can_perform_git_op("agent-2", git_tools::STAGE)
                .await
        );

        // But agent 1 can (idempotent)
        assert!(
            coordinator
                .can_perform_git_op("agent-1", git_tools::STAGE)
                .await
        );
    }
}
