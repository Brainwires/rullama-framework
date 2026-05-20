//! Resource locking system for build/test/git coordination
//!
//! Provides exclusive locks for build, test, and git operations to prevent
//! concurrent operations that could interfere with each other.
//!
//! Key features:
//! - Liveness-based validation (no fixed timeouts)
//! - Integration with OperationTracker for heartbeat monitoring
//! - Git-specific resource types for fine-grained control
//! - Wait queue integration for coordinated access

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast};

use crate::operation_tracker::{OperationHandle, OperationTracker};
use crate::wait_queue::WaitQueue;

/// Type of resource lock
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Build lock - prevents concurrent builds
    Build,
    /// Test lock - prevents concurrent test runs
    Test,
    /// Combined build+test lock (for commands that do both)
    BuildTest,
    /// Git index/staging area operations
    GitIndex,
    /// Git commit operations
    GitCommit,
    /// Git remote write operations (push)
    GitRemoteWrite,
    /// Git remote merge operations (pull)
    GitRemoteMerge,
    /// Git branch operations (create, switch, delete)
    GitBranch,
    /// Git destructive operations (discard, reset)
    GitDestructive,
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceType::Build => write!(f, "Build"),
            ResourceType::Test => write!(f, "Test"),
            ResourceType::BuildTest => write!(f, "BuildTest"),
            ResourceType::GitIndex => write!(f, "GitIndex"),
            ResourceType::GitCommit => write!(f, "GitCommit"),
            ResourceType::GitRemoteWrite => write!(f, "GitRemoteWrite"),
            ResourceType::GitRemoteMerge => write!(f, "GitRemoteMerge"),
            ResourceType::GitBranch => write!(f, "GitBranch"),
            ResourceType::GitDestructive => write!(f, "GitDestructive"),
        }
    }
}

impl ResourceType {
    /// Check if this resource type conflicts with another
    pub fn conflicts_with(&self, other: &ResourceType) -> bool {
        use ResourceType::*;
        match (self, other) {
            // Same type always conflicts
            (a, b) if a == b => true,

            // BuildTest conflicts with both Build and Test
            (BuildTest, Build) | (Build, BuildTest) => true,
            (BuildTest, Test) | (Test, BuildTest) => true,

            // Git operations that modify index conflict with each other
            (GitIndex, GitCommit) | (GitCommit, GitIndex) => true,
            (GitIndex, GitRemoteMerge) | (GitRemoteMerge, GitIndex) => true,
            (GitIndex, GitDestructive) | (GitDestructive, GitIndex) => true,

            // Commit conflicts with destructive operations
            (GitCommit, GitDestructive) | (GitDestructive, GitCommit) => true,

            // Build/Test conflicts with git operations that change code
            (Build, GitRemoteMerge) | (GitRemoteMerge, Build) => true,
            (Test, GitRemoteMerge) | (GitRemoteMerge, Test) => true,
            (Build, GitDestructive) | (GitDestructive, Build) => true,
            (Test, GitDestructive) | (GitDestructive, Test) => true,

            _ => false,
        }
    }

    /// Check if this is a git-related resource type
    pub fn is_git(&self) -> bool {
        matches!(
            self,
            ResourceType::GitIndex
                | ResourceType::GitCommit
                | ResourceType::GitRemoteWrite
                | ResourceType::GitRemoteMerge
                | ResourceType::GitBranch
                | ResourceType::GitDestructive
        )
    }

    /// Check if this is a build/test resource type
    pub fn is_build_test(&self) -> bool {
        matches!(
            self,
            ResourceType::Build | ResourceType::Test | ResourceType::BuildTest
        )
    }
}

/// Scope of a resource lock
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceScope {
    /// Global lock across all projects
    Global,
    /// Project-specific lock (based on project root path)
    Project(PathBuf),
}

impl std::fmt::Display for ResourceScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceScope::Global => write!(f, "Global"),
            ResourceScope::Project(path) => write!(f, "Project({})", path.display()),
        }
    }
}

/// Information about a held resource lock
#[derive(Debug, Clone)]
pub struct ResourceLockInfo {
    /// ID of the agent holding the lock
    pub agent_id: String,
    /// Type of resource locked
    pub resource_type: ResourceType,
    /// Scope of the lock
    pub scope: ResourceScope,
    /// When the lock was acquired
    pub acquired_at: Instant,
    /// Operation ID for liveness tracking (replaces timeout)
    pub operation_id: Option<String>,
    /// Description of the operation
    pub description: String,
    /// Current status message
    pub status: String,
}

impl ResourceLockInfo {
    /// Get elapsed time since lock was acquired
    pub fn elapsed(&self) -> Duration {
        self.acquired_at.elapsed()
    }
}

/// Key for resource lock storage
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceKey {
    resource_type: ResourceType,
    scope: ResourceScope,
}

/// Guard that releases a resource lock when dropped
pub struct ResourceLockGuard {
    manager: Arc<ResourceLockManager>,
    agent_id: String,
    resource_type: ResourceType,
    scope: ResourceScope,
}

impl Drop for ResourceLockGuard {
    fn drop(&mut self) {
        let manager = self.manager.clone();
        let agent_id = self.agent_id.clone();
        let resource_type = self.resource_type;
        let scope = self.scope.clone();

        // Spawn a task to release the lock asynchronously
        tokio::spawn(async move {
            if let Err(e) = manager
                .release_resource_internal(&agent_id, resource_type, &scope)
                .await
            {
                eprintln!("Warning: Failed to release resource lock on drop: {}", e);
            }
        });
    }
}

/// Notification events for lock state changes
#[derive(Debug, Clone)]
pub enum LockNotification {
    /// Lock was acquired
    Acquired {
        /// Agent that acquired the lock.
        agent_id: String,
        /// Type of resource locked.
        resource_type: ResourceType,
        /// Scope of the lock.
        scope: ResourceScope,
    },
    /// Lock was released
    Released {
        /// Agent that released the lock.
        agent_id: String,
        /// Type of resource unlocked.
        resource_type: ResourceType,
        /// Scope of the lock.
        scope: ResourceScope,
    },
    /// Lock became stale (holder stopped sending heartbeats)
    Stale {
        /// Agent whose lock became stale.
        agent_id: String,
        /// Type of stale resource lock.
        resource_type: ResourceType,
        /// Scope of the stale lock.
        scope: ResourceScope,
    },
}

/// Manages resource locks (build/test/git) across multiple agents
///
/// Uses liveness-based validation instead of fixed timeouts:
/// - Integrates with OperationTracker for heartbeat monitoring
/// - Locks are valid as long as the holder is alive
/// - Supports wait queue for coordinated access
pub struct ResourceLockManager {
    /// Map of resource keys to their lock info
    locks: RwLock<HashMap<ResourceKey, ResourceLockInfo>>,
    /// Operation tracker for liveness checking
    operation_tracker: Option<Arc<OperationTracker>>,
    /// Wait queue for coordinated access
    wait_queue: Option<Arc<WaitQueue>>,
    /// Notification channel for lock events
    event_sender: broadcast::Sender<LockNotification>,
}

impl ResourceLockManager {
    /// Create a new resource lock manager
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(256);
        Self {
            locks: RwLock::new(HashMap::new()),
            operation_tracker: None,
            wait_queue: None,
            event_sender,
        }
    }

    /// Create a resource lock manager with operation tracker integration
    pub fn with_operation_tracker(operation_tracker: Arc<OperationTracker>) -> Self {
        let (event_sender, _) = broadcast::channel(256);
        Self {
            locks: RwLock::new(HashMap::new()),
            operation_tracker: Some(operation_tracker),
            wait_queue: None,
            event_sender,
        }
    }

    /// Create a fully integrated resource lock manager
    pub fn with_full_integration(
        operation_tracker: Arc<OperationTracker>,
        wait_queue: Arc<WaitQueue>,
    ) -> Self {
        let (event_sender, _) = broadcast::channel(256);
        Self {
            locks: RwLock::new(HashMap::new()),
            operation_tracker: Some(operation_tracker),
            wait_queue: Some(wait_queue),
            event_sender,
        }
    }

    /// Subscribe to lock notifications
    pub fn subscribe(&self) -> broadcast::Receiver<LockNotification> {
        self.event_sender.subscribe()
    }

    /// Get the operation tracker if configured
    pub fn operation_tracker(&self) -> Option<&Arc<OperationTracker>> {
        self.operation_tracker.as_ref()
    }

    /// Get the wait queue if configured
    pub fn wait_queue(&self) -> Option<&Arc<WaitQueue>> {
        self.wait_queue.as_ref()
    }

    /// Acquire a resource lock
    ///
    /// Returns a ResourceLockGuard that automatically releases the lock when dropped.
    pub async fn acquire_resource(
        self: &Arc<Self>,
        agent_id: &str,
        resource_type: ResourceType,
        scope: ResourceScope,
        description: &str,
    ) -> Result<ResourceLockGuard> {
        let mut locks = self.locks.write().await;

        // Clean up stale locks first (based on liveness)
        self.cleanup_stale_internal(&mut locks).await;

        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        // Check for existing lock
        if let Some(existing) = locks.get(&key) {
            if existing.agent_id != agent_id {
                // Check if the existing lock is still valid
                if self.is_lock_alive_internal(existing).await {
                    return Err(anyhow!(
                        "Resource {} ({}) is locked by agent {} ({})",
                        resource_type,
                        scope,
                        existing.agent_id,
                        existing.description
                    ));
                }
                // Lock holder is dead, remove and continue
                locks.remove(&key);
            } else {
                // Same agent already has the lock, return success (idempotent)
                return Ok(ResourceLockGuard {
                    manager: Arc::clone(self),
                    agent_id: agent_id.to_string(),
                    resource_type,
                    scope,
                });
            }
        }

        // Check for conflicting locks using the new conflict detection
        self.check_conflicts_internal(&locks, agent_id, resource_type, &scope)
            .await?;

        // Acquire the lock
        locks.insert(
            key,
            ResourceLockInfo {
                agent_id: agent_id.to_string(),
                resource_type,
                scope: scope.clone(),
                acquired_at: Instant::now(),
                operation_id: None,
                description: description.to_string(),
                status: "Starting".to_string(),
            },
        );

        // Send notification
        let _ = self.event_sender.send(LockNotification::Acquired {
            agent_id: agent_id.to_string(),
            resource_type,
            scope: scope.clone(),
        });

        Ok(ResourceLockGuard {
            manager: Arc::clone(self),
            agent_id: agent_id.to_string(),
            resource_type,
            scope,
        })
    }

    /// Acquire a resource lock with operation tracking
    ///
    /// This method creates an OperationHandle that:
    /// - Automatically sends heartbeats
    /// - Can have a process attached for liveness monitoring
    /// - Signals completion when dropped
    pub async fn acquire_with_operation(
        self: &Arc<Self>,
        agent_id: &str,
        resource_type: ResourceType,
        scope: ResourceScope,
        description: &str,
    ) -> Result<(ResourceLockGuard, Option<OperationHandle>)> {
        // First acquire the lock
        let guard = self
            .acquire_resource(agent_id, resource_type, scope.clone(), description)
            .await?;

        // If we have an operation tracker, start tracking the operation
        let operation_handle = if let Some(tracker) = &self.operation_tracker {
            let handle = tracker
                .start_operation(agent_id, resource_type, scope.clone(), description)
                .await?;

            // Update the lock with the operation ID
            let mut locks = self.locks.write().await;
            let key = ResourceKey {
                resource_type,
                scope: scope.clone(),
            };
            if let Some(lock_info) = locks.get_mut(&key) {
                lock_info.operation_id = Some(handle.operation_id().to_string());
            }

            Some(handle)
        } else {
            None
        };

        Ok((guard, operation_handle))
    }

    /// Check for conflicting locks
    async fn check_conflicts_internal(
        &self,
        locks: &HashMap<ResourceKey, ResourceLockInfo>,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Result<()> {
        // Check all existing locks for conflicts
        for (key, existing) in locks.iter() {
            if &key.scope != scope {
                continue; // Different scope, no conflict
            }
            if existing.agent_id == agent_id {
                continue; // Same agent, no conflict
            }
            if !self.is_lock_alive_internal(existing).await {
                continue; // Lock holder is dead, will be cleaned up
            }

            // Check if resource types conflict
            if resource_type.conflicts_with(&key.resource_type) {
                return Err(anyhow!(
                    "Cannot acquire {} lock: {} is locked by agent {} ({})",
                    resource_type,
                    key.resource_type,
                    existing.agent_id,
                    existing.description
                ));
            }
        }

        Ok(())
    }

    /// Check if a lock is still alive (holder is active)
    async fn is_lock_alive_internal(&self, lock_info: &ResourceLockInfo) -> bool {
        // If we have an operation tracker and the lock has an operation ID, check liveness
        if let (Some(tracker), Some(op_id)) = (&self.operation_tracker, &lock_info.operation_id) {
            return tracker.is_alive(op_id).await;
        }
        // Without operation tracking, assume lock is valid
        true
    }

    /// Clean up stale locks (holders that are no longer alive)
    async fn cleanup_stale_internal(
        &self,
        locks: &mut HashMap<ResourceKey, ResourceLockInfo>,
    ) -> usize {
        let mut stale_keys = Vec::new();

        for (key, info) in locks.iter() {
            if !self.is_lock_alive_internal(info).await {
                stale_keys.push(key.clone());
                let _ = self.event_sender.send(LockNotification::Stale {
                    agent_id: info.agent_id.clone(),
                    resource_type: info.resource_type,
                    scope: info.scope.clone(),
                });
            }
        }

        let count = stale_keys.len();
        for key in stale_keys {
            // Notify wait queue if configured
            if let Some(wait_queue) = &self.wait_queue {
                let resource_key = format!("{}:{}", key.resource_type, key.scope);
                let _ = wait_queue.notify_released(&resource_key).await;
            }
            locks.remove(&key);
        }

        count
    }

    /// Release a specific resource lock
    pub async fn release_resource(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Result<()> {
        self.release_resource_internal(agent_id, resource_type, scope)
            .await
    }

    /// Internal release implementation
    async fn release_resource_internal(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Result<()> {
        let mut locks = self.locks.write().await;

        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        if let Some(existing) = locks.get(&key) {
            if existing.agent_id == agent_id {
                locks.remove(&key);

                // Send notification
                let _ = self.event_sender.send(LockNotification::Released {
                    agent_id: agent_id.to_string(),
                    resource_type,
                    scope: scope.clone(),
                });

                // Notify wait queue if configured
                if let Some(wait_queue) = &self.wait_queue {
                    let resource_key = format!("{}:{}", resource_type, scope);
                    let _ = wait_queue.notify_released(&resource_key).await;
                }

                Ok(())
            } else {
                Err(anyhow!(
                    "Resource {} ({}) is locked by agent {}, not {}",
                    resource_type,
                    scope,
                    existing.agent_id,
                    agent_id
                ))
            }
        } else {
            Err(anyhow!(
                "No lock found for resource {} ({})",
                resource_type,
                scope
            ))
        }
    }

    /// Release all locks held by an agent
    pub async fn release_all_for_agent(&self, agent_id: &str) -> usize {
        let mut locks = self.locks.write().await;
        let original_len = locks.len();
        locks.retain(|_, info| info.agent_id != agent_id);
        original_len - locks.len()
    }

    /// Check if a resource can be acquired by an agent
    pub async fn can_acquire(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> bool {
        let locks = self.locks.read().await;

        // Check all existing locks for conflicts
        for (key, existing) in locks.iter() {
            if &key.scope != scope {
                continue; // Different scope, no conflict
            }
            if existing.agent_id == agent_id {
                continue; // Same agent, no conflict
            }
            if !self.is_lock_alive_internal(existing).await {
                continue; // Lock holder is dead, will be cleaned up
            }

            // Check if resource types conflict
            if resource_type.conflicts_with(&key.resource_type) {
                return false;
            }
        }

        true
    }

    /// Get detailed information about what's blocking acquisition
    pub async fn get_blocking_locks(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Vec<ResourceLockInfo> {
        let locks = self.locks.read().await;
        let mut blocking = Vec::new();

        for (key, existing) in locks.iter() {
            if &key.scope != scope {
                continue;
            }
            if existing.agent_id == agent_id {
                continue;
            }
            if !self.is_lock_alive_internal(existing).await {
                continue;
            }
            if resource_type.conflicts_with(&key.resource_type) {
                blocking.push(existing.clone());
            }
        }

        blocking
    }

    /// Query the detailed status of a lock
    pub async fn query_lock_status(
        &self,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Option<LockStatus> {
        let locks = self.locks.read().await;
        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        if let Some(info) = locks.get(&key) {
            let is_alive = self.is_lock_alive_internal(info).await;
            let operation_status = if let (Some(tracker), Some(op_id)) =
                (&self.operation_tracker, &info.operation_id)
            {
                tracker.get_status(op_id).await
            } else {
                None
            };

            Some(LockStatus {
                agent_id: info.agent_id.clone(),
                resource_type: info.resource_type,
                scope: info.scope.clone(),
                acquired_at_secs_ago: info.elapsed().as_secs(),
                is_alive,
                description: info.description.clone(),
                status: info.status.clone(),
                operation_id: info.operation_id.clone(),
                operation_status,
            })
        } else {
            None
        }
    }

    /// Check if a resource is currently locked
    pub async fn check_lock(
        &self,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Option<ResourceLockInfo> {
        let locks = self.locks.read().await;

        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        locks.get(&key).cloned()
    }

    /// Force release a lock (admin operation)
    pub async fn force_release(
        &self,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Result<()> {
        let mut locks = self.locks.write().await;

        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        if locks.remove(&key).is_some() {
            Ok(())
        } else {
            Err(anyhow!(
                "No lock found for resource {} ({})",
                resource_type,
                scope
            ))
        }
    }

    /// Get all currently held locks
    pub async fn list_locks(&self) -> Vec<ResourceLockInfo> {
        let locks = self.locks.read().await;
        locks.values().cloned().collect()
    }

    /// Get locks held by a specific agent
    pub async fn locks_for_agent(&self, agent_id: &str) -> Vec<ResourceLockInfo> {
        let locks = self.locks.read().await;
        locks
            .values()
            .filter(|info| info.agent_id == agent_id)
            .cloned()
            .collect()
    }

    /// Clean up stale locks (public version)
    pub async fn cleanup_stale(&self) -> usize {
        let mut locks = self.locks.write().await;
        self.cleanup_stale_internal(&mut locks).await
    }

    /// Get statistics about current locks
    pub async fn stats(&self) -> ResourceLockStats {
        let locks = self.locks.read().await;

        let mut build_locks = 0;
        let mut test_locks = 0;
        let mut buildtest_locks = 0;
        let mut git_locks = 0;

        for info in locks.values() {
            match info.resource_type {
                ResourceType::Build => build_locks += 1,
                ResourceType::Test => test_locks += 1,
                ResourceType::BuildTest => buildtest_locks += 1,
                ResourceType::GitIndex
                | ResourceType::GitCommit
                | ResourceType::GitRemoteWrite
                | ResourceType::GitRemoteMerge
                | ResourceType::GitBranch
                | ResourceType::GitDestructive => git_locks += 1,
            }
        }

        ResourceLockStats {
            total_locks: locks.len(),
            build_locks,
            test_locks,
            buildtest_locks,
            git_locks,
        }
    }

    /// Update the status of a held lock
    pub async fn update_lock_status(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
        status: &str,
    ) -> Result<()> {
        let mut locks = self.locks.write().await;
        let key = ResourceKey {
            resource_type,
            scope: scope.clone(),
        };

        if let Some(info) = locks.get_mut(&key) {
            if info.agent_id == agent_id {
                info.status = status.to_string();
                Ok(())
            } else {
                Err(anyhow!(
                    "Lock is held by agent {}, not {}",
                    info.agent_id,
                    agent_id
                ))
            }
        } else {
            Err(anyhow!(
                "No lock found for resource {} ({})",
                resource_type,
                scope
            ))
        }
    }
}

impl Default for ResourceLockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about current resource locks
#[derive(Debug, Clone)]
pub struct ResourceLockStats {
    /// Total number of locks
    pub total_locks: usize,
    /// Number of build locks
    pub build_locks: usize,
    /// Number of test locks
    pub test_locks: usize,
    /// Number of build+test locks
    pub buildtest_locks: usize,
    /// Number of git-related locks
    pub git_locks: usize,
}

/// Detailed status of a lock (for querying)
#[derive(Debug, Clone)]
pub struct LockStatus {
    /// Agent holding the lock
    pub agent_id: String,
    /// Type of resource locked
    pub resource_type: ResourceType,
    /// Scope of the lock
    pub scope: ResourceScope,
    /// Seconds since lock was acquired
    pub acquired_at_secs_ago: u64,
    /// Whether the lock holder is still alive
    pub is_alive: bool,
    /// Description of the operation
    pub description: String,
    /// Current status message
    pub status: String,
    /// Operation ID if tracked
    pub operation_id: Option<String>,
    /// Detailed operation status if available
    pub operation_status: Option<super::operation_tracker::OperationStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_build_lock() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        assert!(
            manager
                .check_lock(ResourceType::Build, &scope)
                .await
                .is_some()
        );

        drop(guard);
        // Give the async drop task time to run
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(
            manager
                .check_lock(ResourceType::Build, &scope)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_build_lock_blocks_other_agent() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        let result = manager
            .acquire_resource("agent-2", ResourceType::Build, scope.clone(), "cargo build")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_same_agent_reacquire() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard1 = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        // Same agent can reacquire (idempotent)
        let _guard2 = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        assert!(
            manager
                .check_lock(ResourceType::Build, &scope)
                .await
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_buildtest_blocks_build_and_test() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard = manager
            .acquire_resource(
                "agent-1",
                ResourceType::BuildTest,
                scope.clone(),
                "cargo build && cargo test",
            )
            .await
            .unwrap();

        // BuildTest should block Build
        let result = manager
            .acquire_resource("agent-2", ResourceType::Build, scope.clone(), "cargo build")
            .await;
        assert!(result.is_err());

        // BuildTest should block Test
        let result = manager
            .acquire_resource("agent-2", ResourceType::Test, scope.clone(), "cargo test")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_build_blocks_buildtest() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        // Build should block BuildTest
        let result = manager
            .acquire_resource(
                "agent-2",
                ResourceType::BuildTest,
                scope.clone(),
                "cargo build && cargo test",
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_different_scopes_independent() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope1 = ResourceScope::Project(PathBuf::from("/test/project1"));
        let scope2 = ResourceScope::Project(PathBuf::from("/test/project2"));

        let _guard1 = manager
            .acquire_resource(
                "agent-1",
                ResourceType::Build,
                scope1.clone(),
                "cargo build",
            )
            .await
            .unwrap();

        // Different project should work
        let _guard2 = manager
            .acquire_resource(
                "agent-2",
                ResourceType::Build,
                scope2.clone(),
                "cargo build",
            )
            .await
            .unwrap();

        assert!(
            manager
                .check_lock(ResourceType::Build, &scope1)
                .await
                .is_some()
        );
        assert!(
            manager
                .check_lock(ResourceType::Build, &scope2)
                .await
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_release_all_for_agent() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let guard1 = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();
        let guard2 = manager
            .acquire_resource("agent-1", ResourceType::Test, scope.clone(), "cargo test")
            .await
            .unwrap();

        // Forget guards to prevent auto-release
        std::mem::forget(guard1);
        std::mem::forget(guard2);

        let released = manager.release_all_for_agent("agent-1").await;
        assert_eq!(released, 2);

        assert!(
            manager
                .check_lock(ResourceType::Build, &scope)
                .await
                .is_none()
        );
        assert!(
            manager
                .check_lock(ResourceType::Test, &scope)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_can_acquire() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        // No locks - can acquire
        assert!(
            manager
                .can_acquire("agent-1", ResourceType::Build, &scope)
                .await
        );

        let _guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        // Same agent can acquire
        assert!(
            manager
                .can_acquire("agent-1", ResourceType::Build, &scope)
                .await
        );

        // Other agent cannot
        assert!(
            !manager
                .can_acquire("agent-2", ResourceType::Build, &scope)
                .await
        );
    }

    #[tokio::test]
    async fn test_stats() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard1 = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_resource(
                "agent-2",
                ResourceType::Test,
                ResourceScope::Global,
                "cargo test",
            )
            .await
            .unwrap();

        let stats = manager.stats().await;
        assert_eq!(stats.total_locks, 2);
        assert_eq!(stats.build_locks, 1);
        assert_eq!(stats.test_locks, 1);
        assert_eq!(stats.buildtest_locks, 0);
    }

    #[tokio::test]
    async fn test_git_resource_types() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        // GitIndex lock
        let _guard = manager
            .acquire_resource(
                "agent-1",
                ResourceType::GitIndex,
                scope.clone(),
                "git stage",
            )
            .await
            .unwrap();

        // GitCommit should conflict with GitIndex
        let result = manager
            .acquire_resource(
                "agent-2",
                ResourceType::GitCommit,
                scope.clone(),
                "git commit",
            )
            .await;
        assert!(result.is_err());

        // GitRemoteWrite should NOT conflict with GitIndex
        let _guard2 = manager
            .acquire_resource(
                "agent-2",
                ResourceType::GitRemoteWrite,
                scope.clone(),
                "git push",
            )
            .await
            .unwrap();

        let stats = manager.stats().await;
        assert_eq!(stats.git_locks, 2);
    }

    #[tokio::test]
    async fn test_resource_type_conflicts() {
        // Test the conflicts_with method
        assert!(ResourceType::Build.conflicts_with(&ResourceType::Build));
        assert!(ResourceType::Build.conflicts_with(&ResourceType::BuildTest));
        assert!(ResourceType::BuildTest.conflicts_with(&ResourceType::Build));
        assert!(ResourceType::BuildTest.conflicts_with(&ResourceType::Test));

        // Git conflicts
        assert!(ResourceType::GitIndex.conflicts_with(&ResourceType::GitCommit));
        assert!(ResourceType::GitIndex.conflicts_with(&ResourceType::GitRemoteMerge));
        assert!(ResourceType::GitIndex.conflicts_with(&ResourceType::GitDestructive));

        // Non-conflicts
        assert!(!ResourceType::Build.conflicts_with(&ResourceType::Test));
        assert!(!ResourceType::GitRemoteWrite.conflicts_with(&ResourceType::GitBranch));
    }

    #[tokio::test]
    async fn test_get_blocking_locks() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        let blocking = manager
            .get_blocking_locks("agent-2", ResourceType::BuildTest, &scope)
            .await;

        assert_eq!(blocking.len(), 1);
        assert_eq!(blocking[0].agent_id, "agent-1");
        assert_eq!(blocking[0].resource_type, ResourceType::Build);
    }

    #[tokio::test]
    async fn test_update_lock_status() {
        let manager = Arc::new(ResourceLockManager::new());
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _guard = manager
            .acquire_resource("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        // Update status
        manager
            .update_lock_status("agent-1", ResourceType::Build, &scope, "Compiling crate...")
            .await
            .unwrap();

        let lock = manager
            .check_lock(ResourceType::Build, &scope)
            .await
            .unwrap();
        assert_eq!(lock.status, "Compiling crate...");

        // Other agent cannot update
        let result = manager
            .update_lock_status("agent-2", ResourceType::Build, &scope, "Hacking...")
            .await;
        assert!(result.is_err());
    }
}
