//! Cross-resource conflict detection
//!
//! Provides bidirectional checking between file locks and resource locks:
//! - Builds should wait if source files are being edited
//! - File writes should wait if build/test is in progress
//!
//! This ensures consistency and prevents race conditions between
//! file editing and build/test operations.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::communication::{ConflictInfo, ConflictType};
use crate::file_locks::{FileLockManager, LockType};
use crate::operation_tracker::OperationTracker;
use crate::resource_locks::{ResourceLockManager, ResourceScope, ResourceType};

/// Result of checking for conflicts before an operation
#[derive(Debug, Clone)]
pub enum ConflictCheck {
    /// No conflicts, operation can proceed
    Clear,
    /// Operation is blocked by active conflicts - must wait
    Blocked(Vec<Conflict>),
    /// Conflicts exist but are warnings only - can proceed with caution
    Warning(Vec<Conflict>),
}

impl ConflictCheck {
    /// Returns true if the operation can proceed (Clear or Warning)
    pub fn can_proceed(&self) -> bool {
        matches!(self, ConflictCheck::Clear | ConflictCheck::Warning(_))
    }

    /// Returns true if the operation is blocked
    pub fn is_blocked(&self) -> bool {
        matches!(self, ConflictCheck::Blocked(_))
    }

    /// Get all conflicts (blocking or warning)
    pub fn conflicts(&self) -> &[Conflict] {
        match self {
            ConflictCheck::Clear => &[],
            ConflictCheck::Blocked(c) | ConflictCheck::Warning(c) => c,
        }
    }
}

/// Information about a detected conflict
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Type of conflict
    pub conflict_type: ResourceConflictType,
    /// Agent holding the conflicting resource
    pub holder_agent: String,
    /// Resource identifier (path or scope description)
    pub resource: String,
    /// When the conflict started
    pub started_at: Instant,
    /// Current status of the blocking operation
    pub status: String,
    /// Description of what the holder is doing
    pub description: String,
}

impl Conflict {
    /// Convert to the communication ConflictInfo type for messaging
    pub fn to_conflict_info(&self) -> ConflictInfo {
        ConflictInfo {
            conflict_type: match &self.conflict_type {
                ResourceConflictType::FileWriteBlocksBuild { path } => {
                    ConflictType::FileWriteBlocksBuild { path: path.clone() }
                }
                ResourceConflictType::BuildBlocksFileWrite => ConflictType::BuildBlocksFileWrite,
                ResourceConflictType::TestBlocksFileWrite => ConflictType::TestBlocksFileWrite,
                ResourceConflictType::GitBlocksFileWrite => ConflictType::GitBlocksFileWrite,
                ResourceConflictType::FileWriteBlocksGit { path } => {
                    ConflictType::FileWriteBlocksGit { path: path.clone() }
                }
                ResourceConflictType::BuildBlocksGit => ConflictType::BuildBlocksGit,
            },
            holder_agent: self.holder_agent.clone(),
            resource: self.resource.clone(),
            duration_secs: self.started_at.elapsed().as_secs(),
            status: self.status.clone(),
        }
    }
}

/// Types of cross-resource conflicts
#[derive(Debug, Clone)]
pub enum ResourceConflictType {
    /// File write lock blocks a build operation
    FileWriteBlocksBuild {
        /// Path of the locked file.
        path: PathBuf,
    },
    /// Build in progress blocks file write
    BuildBlocksFileWrite,
    /// Test in progress blocks file write
    TestBlocksFileWrite,
    /// Git operation blocks file write
    GitBlocksFileWrite,
    /// File write lock blocks git operation
    FileWriteBlocksGit {
        /// Path of the locked file.
        path: PathBuf,
    },
    /// Build in progress blocks git operation
    BuildBlocksGit,
}

/// Proposed operation to check for conflicts
#[derive(Debug, Clone)]
pub enum ProposedOperation {
    /// File write operation
    FileWrite {
        /// Path of the file to write.
        path: PathBuf,
        /// Agent performing the write.
        agent_id: String,
    },
    /// Build operation
    Build {
        /// Scope of the build.
        scope: ResourceScope,
        /// Agent performing the build.
        agent_id: String,
    },
    /// Test operation
    Test {
        /// Scope of the test.
        scope: ResourceScope,
        /// Agent performing the test.
        agent_id: String,
    },
    /// Git staging operation
    GitStaging {
        /// Scope of the staging operation.
        scope: ResourceScope,
        /// Agent performing the staging.
        agent_id: String,
    },
    /// Git commit operation
    GitCommit {
        /// Scope of the commit operation.
        scope: ResourceScope,
        /// Agent performing the commit.
        agent_id: String,
    },
    /// Git push operation
    GitPush {
        /// Scope of the push operation.
        scope: ResourceScope,
        /// Agent performing the push.
        agent_id: String,
    },
    /// Git pull operation
    GitPull {
        /// Scope of the pull operation.
        scope: ResourceScope,
        /// Agent performing the pull.
        agent_id: String,
    },
}

impl ProposedOperation {
    /// Get the agent ID for this proposed operation.
    pub fn agent_id(&self) -> &str {
        match self {
            ProposedOperation::FileWrite { agent_id, .. }
            | ProposedOperation::Build { agent_id, .. }
            | ProposedOperation::Test { agent_id, .. }
            | ProposedOperation::GitStaging { agent_id, .. }
            | ProposedOperation::GitCommit { agent_id, .. }
            | ProposedOperation::GitPush { agent_id, .. }
            | ProposedOperation::GitPull { agent_id, .. } => agent_id,
        }
    }
}

/// Cross-resource conflict checker
///
/// Checks for conflicts between:
/// - File locks and build/test operations
/// - Build/test operations and file writes
/// - Git operations and file/build operations
pub struct ResourceChecker {
    file_locks: Arc<FileLockManager>,
    resource_locks: Arc<ResourceLockManager>,
    _operation_tracker: Option<Arc<OperationTracker>>,
    /// File patterns to check for build conflicts (e.g., "src/**/*.rs")
    source_patterns: Vec<String>,
}

impl ResourceChecker {
    /// Create a new resource checker
    pub fn new(file_locks: Arc<FileLockManager>, resource_locks: Arc<ResourceLockManager>) -> Self {
        Self {
            file_locks,
            resource_locks,
            _operation_tracker: None,
            source_patterns: default_source_patterns(),
        }
    }

    /// Create a resource checker with operation tracker integration
    pub fn with_operation_tracker(
        file_locks: Arc<FileLockManager>,
        resource_locks: Arc<ResourceLockManager>,
        operation_tracker: Arc<OperationTracker>,
    ) -> Self {
        Self {
            file_locks,
            resource_locks,
            _operation_tracker: Some(operation_tracker),
            source_patterns: default_source_patterns(),
        }
    }

    /// Set custom source file patterns for build conflict detection
    pub fn with_source_patterns(mut self, patterns: Vec<String>) -> Self {
        self.source_patterns = patterns;
        self
    }

    /// Check if a build can start (no active file write locks in project)
    ///
    /// Returns `Clear` if no source files are being edited,
    /// or `Blocked` with details of which files are locked.
    pub async fn can_start_build(&self, scope: &ResourceScope, agent_id: &str) -> ConflictCheck {
        let file_locks = self.file_locks.list_locks().await;

        let mut conflicts = Vec::new();

        for (path, lock_info) in file_locks {
            // Skip if same agent
            if lock_info.agent_id == agent_id {
                continue;
            }

            // Only check write locks
            if lock_info.lock_type != LockType::Write {
                continue;
            }

            // Check if this file is within the scope and is a source file
            if self.is_in_scope(&path, scope) && self.is_source_file(&path) {
                conflicts.push(Conflict {
                    conflict_type: ResourceConflictType::FileWriteBlocksBuild {
                        path: path.clone(),
                    },
                    holder_agent: lock_info.agent_id.clone(),
                    resource: path.display().to_string(),
                    started_at: lock_info.acquired_at,
                    status: "File locked for editing".to_string(),
                    description: format!("Write lock on {}", path.display()),
                });
            }
        }

        if conflicts.is_empty() {
            ConflictCheck::Clear
        } else {
            ConflictCheck::Blocked(conflicts)
        }
    }

    /// Check if a file can be written (no active build/test in project)
    ///
    /// Returns `Clear` if no build/test is running,
    /// or `Blocked` with details of the blocking operation.
    pub async fn can_write_file(&self, path: &Path, agent_id: &str) -> ConflictCheck {
        let resource_locks = self.resource_locks.list_locks().await;

        let mut conflicts = Vec::new();

        // Determine the scope from the file path
        let file_scope = self.scope_for_path(path);

        for lock_info in resource_locks {
            // Skip if same agent
            if lock_info.agent_id == agent_id {
                continue;
            }

            // Check if the lock scope overlaps with the file's scope
            if !self.scopes_overlap(&lock_info.scope, &file_scope) {
                continue;
            }

            // Only source files are affected by builds/tests
            if !self.is_source_file(path) {
                continue;
            }

            // Check for build/test conflicts
            match lock_info.resource_type {
                ResourceType::Build | ResourceType::BuildTest => {
                    conflicts.push(Conflict {
                        conflict_type: ResourceConflictType::BuildBlocksFileWrite,
                        holder_agent: lock_info.agent_id.clone(),
                        resource: format!("{} ({})", lock_info.resource_type, lock_info.scope),
                        started_at: lock_info.acquired_at,
                        status: lock_info.status.clone(),
                        description: lock_info.description.clone(),
                    });
                }
                ResourceType::Test => {
                    conflicts.push(Conflict {
                        conflict_type: ResourceConflictType::TestBlocksFileWrite,
                        holder_agent: lock_info.agent_id.clone(),
                        resource: format!("{} ({})", lock_info.resource_type, lock_info.scope),
                        started_at: lock_info.acquired_at,
                        status: lock_info.status.clone(),
                        description: lock_info.description.clone(),
                    });
                }
                ResourceType::GitIndex
                | ResourceType::GitCommit
                | ResourceType::GitRemoteWrite
                | ResourceType::GitRemoteMerge
                | ResourceType::GitBranch
                | ResourceType::GitDestructive => {
                    // Git operations that modify the working tree block file writes
                    if lock_info.resource_type == ResourceType::GitRemoteMerge
                        || lock_info.resource_type == ResourceType::GitDestructive
                    {
                        conflicts.push(Conflict {
                            conflict_type: ResourceConflictType::GitBlocksFileWrite,
                            holder_agent: lock_info.agent_id.clone(),
                            resource: format!("{} ({})", lock_info.resource_type, lock_info.scope),
                            started_at: lock_info.acquired_at,
                            status: lock_info.status.clone(),
                            description: lock_info.description.clone(),
                        });
                    }
                }
            }
        }

        if conflicts.is_empty() {
            ConflictCheck::Clear
        } else {
            ConflictCheck::Blocked(conflicts)
        }
    }

    /// Check if a git operation can start
    ///
    /// Git operations are blocked by:
    /// - Active file write locks (for operations that read working tree)
    /// - Active builds (for commit/push operations)
    pub async fn can_start_git_operation(
        &self,
        git_op: ResourceType,
        scope: &ResourceScope,
        agent_id: &str,
    ) -> ConflictCheck {
        let mut conflicts = Vec::new();

        // Check file locks for git operations that read the working tree
        if matches!(
            git_op,
            ResourceType::GitIndex | ResourceType::GitCommit | ResourceType::GitRemoteWrite
        ) {
            let file_locks = self.file_locks.list_locks().await;

            for (path, lock_info) in file_locks {
                if lock_info.agent_id == agent_id {
                    continue;
                }
                if lock_info.lock_type != LockType::Write {
                    continue;
                }
                if self.is_in_scope(&path, scope) && self.is_source_file(&path) {
                    conflicts.push(Conflict {
                        conflict_type: ResourceConflictType::FileWriteBlocksGit {
                            path: path.clone(),
                        },
                        holder_agent: lock_info.agent_id.clone(),
                        resource: path.display().to_string(),
                        started_at: lock_info.acquired_at,
                        status: "File locked for editing".to_string(),
                        description: format!("Write lock on {}", path.display()),
                    });
                }
            }
        }

        // Check for build conflicts for commit/push operations
        if matches!(
            git_op,
            ResourceType::GitCommit | ResourceType::GitRemoteWrite
        ) {
            let resource_locks = self.resource_locks.list_locks().await;

            for lock_info in resource_locks {
                if lock_info.agent_id == agent_id {
                    continue;
                }
                if !self.scopes_overlap(&lock_info.scope, scope) {
                    continue;
                }

                if matches!(
                    lock_info.resource_type,
                    ResourceType::Build | ResourceType::Test | ResourceType::BuildTest
                ) {
                    conflicts.push(Conflict {
                        conflict_type: ResourceConflictType::BuildBlocksGit,
                        holder_agent: lock_info.agent_id.clone(),
                        resource: format!("{} ({})", lock_info.resource_type, lock_info.scope),
                        started_at: lock_info.acquired_at,
                        status: lock_info.status.clone(),
                        description: lock_info.description.clone(),
                    });
                }
            }
        }

        if conflicts.is_empty() {
            ConflictCheck::Clear
        } else {
            ConflictCheck::Blocked(conflicts)
        }
    }

    /// Check all conflicts for a proposed operation
    pub async fn check_conflicts(&self, operation: &ProposedOperation) -> ConflictCheck {
        match operation {
            ProposedOperation::FileWrite { path, agent_id } => {
                self.can_write_file(path, agent_id).await
            }
            ProposedOperation::Build { scope, agent_id } => {
                self.can_start_build(scope, agent_id).await
            }
            ProposedOperation::Test { scope, agent_id } => {
                // Same checks as build
                self.can_start_build(scope, agent_id).await
            }
            ProposedOperation::GitStaging { scope, agent_id } => {
                self.can_start_git_operation(ResourceType::GitIndex, scope, agent_id)
                    .await
            }
            ProposedOperation::GitCommit { scope, agent_id } => {
                self.can_start_git_operation(ResourceType::GitCommit, scope, agent_id)
                    .await
            }
            ProposedOperation::GitPush { scope, agent_id } => {
                self.can_start_git_operation(ResourceType::GitRemoteWrite, scope, agent_id)
                    .await
            }
            ProposedOperation::GitPull { scope, agent_id } => {
                self.can_start_git_operation(ResourceType::GitRemoteMerge, scope, agent_id)
                    .await
            }
        }
    }

    /// Get all current conflicts that would block a build
    pub async fn get_build_blockers(&self, scope: &ResourceScope, agent_id: &str) -> Vec<Conflict> {
        match self.can_start_build(scope, agent_id).await {
            ConflictCheck::Blocked(conflicts) => conflicts,
            _ => Vec::new(),
        }
    }

    /// Get all current conflicts that would block a file write
    pub async fn get_file_write_blockers(&self, path: &Path, agent_id: &str) -> Vec<Conflict> {
        match self.can_write_file(path, agent_id).await {
            ConflictCheck::Blocked(conflicts) => conflicts,
            _ => Vec::new(),
        }
    }

    // === Helper methods ===

    /// Check if a path is within a resource scope
    fn is_in_scope(&self, path: &Path, scope: &ResourceScope) -> bool {
        match scope {
            ResourceScope::Global => true,
            ResourceScope::Project(project_path) => path.starts_with(project_path),
        }
    }

    /// Determine the scope for a file path
    fn scope_for_path(&self, path: &Path) -> ResourceScope {
        // Try to find the project root by looking for common markers
        let mut current = path.parent();
        while let Some(dir) = current {
            if dir.join("Cargo.toml").exists()
                || dir.join("package.json").exists()
                || dir.join(".git").exists()
            {
                return ResourceScope::Project(dir.to_path_buf());
            }
            current = dir.parent();
        }
        ResourceScope::Global
    }

    /// Check if two scopes overlap
    fn scopes_overlap(&self, scope1: &ResourceScope, scope2: &ResourceScope) -> bool {
        match (scope1, scope2) {
            (ResourceScope::Global, _) | (_, ResourceScope::Global) => true,
            (ResourceScope::Project(p1), ResourceScope::Project(p2)) => {
                p1.starts_with(p2) || p2.starts_with(p1)
            }
        }
    }

    /// Check if a file is a source file that should block builds
    fn is_source_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check against source patterns
        for pattern in &self.source_patterns {
            if matches_pattern(&path_str, pattern) {
                return true;
            }
        }

        // Default: check common source extensions
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(
                ext.as_str(),
                "rs" | "ts"
                    | "tsx"
                    | "js"
                    | "jsx"
                    | "py"
                    | "go"
                    | "java"
                    | "c"
                    | "cpp"
                    | "h"
                    | "hpp"
                    | "cs"
                    | "swift"
                    | "kt"
                    | "scala"
                    | "rb"
                    | "php"
            )
        } else {
            false
        }
    }
}

/// Default source file patterns for build conflict detection
fn default_source_patterns() -> Vec<String> {
    vec![
        "src/**/*".to_string(),
        "lib/**/*".to_string(),
        "crates/**/*".to_string(),
        "packages/**/*".to_string(),
        "app/**/*".to_string(),
    ]
}

/// Simple glob-style pattern matching
fn matches_pattern(path: &str, pattern: &str) -> bool {
    if pattern.contains("**") {
        // Handle ** (match any depth)
        let parts: Vec<&str> = pattern.split("**").collect();
        if parts.len() == 2 {
            let prefix = parts[0].trim_end_matches('/');
            let suffix = parts[1].trim_start_matches('/');

            let has_prefix = prefix.is_empty() || path.starts_with(prefix);
            let has_suffix = suffix.is_empty()
                || suffix == "*"
                || path.ends_with(suffix.trim_start_matches('*'));

            return has_prefix && has_suffix;
        }
    }

    if pattern.contains('*') {
        // Handle single * (match within directory)
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return path.starts_with(parts[0]) && path.ends_with(parts[1]);
        }
    }

    // Exact match
    path == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern("src/main.rs", "src/**/*"));
        assert!(matches_pattern("src/lib/utils.rs", "src/**/*"));
        assert!(matches_pattern("crates/foo/src/lib.rs", "crates/**/*"));
        assert!(!matches_pattern("target/debug/main", "src/**/*"));
    }

    #[test]
    fn test_is_source_file() {
        let checker = ResourceChecker::new(
            Arc::new(FileLockManager::new()),
            Arc::new(ResourceLockManager::new()),
        );

        assert!(checker.is_source_file(Path::new("src/main.rs")));
        assert!(checker.is_source_file(Path::new("lib/index.ts")));
        assert!(checker.is_source_file(Path::new("app.py")));
        assert!(!checker.is_source_file(Path::new("README.md")));
        assert!(!checker.is_source_file(Path::new("Cargo.toml")));
    }

    #[test]
    fn test_scopes_overlap() {
        let checker = ResourceChecker::new(
            Arc::new(FileLockManager::new()),
            Arc::new(ResourceLockManager::new()),
        );

        // Global overlaps with everything
        assert!(checker.scopes_overlap(&ResourceScope::Global, &ResourceScope::Global));
        assert!(checker.scopes_overlap(
            &ResourceScope::Global,
            &ResourceScope::Project(PathBuf::from("/foo"))
        ));

        // Project scopes
        assert!(checker.scopes_overlap(
            &ResourceScope::Project(PathBuf::from("/foo")),
            &ResourceScope::Project(PathBuf::from("/foo"))
        ));
        assert!(checker.scopes_overlap(
            &ResourceScope::Project(PathBuf::from("/foo")),
            &ResourceScope::Project(PathBuf::from("/foo/bar"))
        ));
        assert!(!checker.scopes_overlap(
            &ResourceScope::Project(PathBuf::from("/foo")),
            &ResourceScope::Project(PathBuf::from("/baz"))
        ));
    }

    #[tokio::test]
    async fn test_can_start_build_no_conflicts() {
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());
        let checker = ResourceChecker::new(file_locks, resource_locks);

        let scope = ResourceScope::Project(PathBuf::from("/test/project"));
        let result = checker.can_start_build(&scope, "agent-1").await;

        assert!(matches!(result, ConflictCheck::Clear));
    }

    #[tokio::test]
    async fn test_can_write_file_no_conflicts() {
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());
        let checker = ResourceChecker::new(file_locks, resource_locks);

        let result = checker
            .can_write_file(Path::new("/test/project/src/main.rs"), "agent-1")
            .await;

        assert!(matches!(result, ConflictCheck::Clear));
    }

    #[tokio::test]
    async fn test_build_blocked_by_file_write() {
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());

        // Agent 2 acquires a write lock on a source file
        file_locks
            .acquire_lock("agent-2", "/test/project/src/main.rs", LockType::Write)
            .await
            .unwrap();

        let checker = ResourceChecker::new(file_locks, resource_locks);

        let scope = ResourceScope::Project(PathBuf::from("/test/project"));
        let result = checker.can_start_build(&scope, "agent-1").await;

        assert!(result.is_blocked());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(
            conflicts[0].conflict_type,
            ResourceConflictType::FileWriteBlocksBuild { .. }
        ));
    }

    #[tokio::test]
    async fn test_file_write_blocked_by_build() {
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());

        // Agent 2 acquires a build lock
        resource_locks
            .acquire_resource(
                "agent-2",
                ResourceType::Build,
                ResourceScope::Project(PathBuf::from("/test/project")),
                "cargo build",
            )
            .await
            .unwrap();

        let checker = ResourceChecker::new(file_locks, resource_locks);

        let result = checker
            .can_write_file(Path::new("/test/project/src/main.rs"), "agent-1")
            .await;

        assert!(result.is_blocked());
        let conflicts = result.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(
            conflicts[0].conflict_type,
            ResourceConflictType::BuildBlocksFileWrite
        ));
    }

    #[tokio::test]
    async fn test_same_agent_no_conflict() {
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());

        // Same agent has file lock and wants to build
        file_locks
            .acquire_lock("agent-1", "/test/project/src/main.rs", LockType::Write)
            .await
            .unwrap();

        let checker = ResourceChecker::new(file_locks, resource_locks);

        let scope = ResourceScope::Project(PathBuf::from("/test/project"));
        let result = checker.can_start_build(&scope, "agent-1").await;

        // Same agent should not conflict with itself
        assert!(matches!(result, ConflictCheck::Clear));
    }
}
