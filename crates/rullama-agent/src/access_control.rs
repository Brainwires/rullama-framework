//! Unified access control manager for inter-agent coordination
//!
//! Provides a single interface for managing file locks, resource locks (build/test),
//! and read-before-write enforcement.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::file_locks::{FileLockManager, LockGuard, LockType};
use crate::resource_locks::{ResourceLockGuard, ResourceLockManager, ResourceScope, ResourceType};

const DEFAULT_INITIAL_DELAY_MS: u64 = 100;
const DEFAULT_MAX_RETRIES: u32 = 5;
const DEFAULT_MAX_DELAY_SECS: u64 = 5;
const PERSISTENT_LOCK_TIMEOUT_SECS: u64 = 300;
const FILE_LOCK_BACKOFF_INITIAL_MS: u64 = 50;
const FILE_LOCK_BACKOFF_MAX_MS: u64 = 500;
const RESOURCE_LOCK_BACKOFF_INITIAL_MS: u64 = 100;
const RESOURCE_LOCK_BACKOFF_MAX_SECS: u64 = 1;

/// Trait for persistent lock storage (inter-process coordination)
///
/// Implement this trait to enable cross-process lock coordination.
/// The default implementation (no-op) is used when no persistent store is configured.
#[async_trait::async_trait]
pub trait LockPersistence: Send + Sync {
    /// Try to acquire a persistent lock
    ///
    /// Returns `Ok(true)` if the lock was acquired, `Ok(false)` if it's already held.
    async fn try_acquire(
        &self,
        lock_type: &str,
        resource_path: &str,
        agent_id: &str,
        timeout: Option<Duration>,
    ) -> Result<bool>;

    /// Release a persistent lock
    async fn release(&self, lock_type: &str, resource_path: &str, agent_id: &str) -> Result<()>;

    /// Release all locks held by an agent
    async fn release_all_for_agent(&self, agent_id: &str) -> Result<usize>;

    /// Cleanup stale locks (e.g., from crashed processes)
    async fn cleanup_stale(&self) -> Result<usize>;
}

/// Strategy for handling lock contention
#[derive(Debug, Clone)]
pub enum ContentionStrategy {
    /// Fail immediately if lock is unavailable
    FailFast,
    /// Wait up to the specified duration for the lock
    WaitWithTimeout(Duration),
    /// Retry with exponential backoff
    RetryWithBackoff {
        /// Initial delay between retries.
        initial_delay: Duration,
        /// Maximum number of retries.
        max_retries: u32,
        /// Maximum delay cap.
        max_delay: Duration,
    },
}

impl Default for ContentionStrategy {
    fn default() -> Self {
        ContentionStrategy::RetryWithBackoff {
            initial_delay: Duration::from_millis(DEFAULT_INITIAL_DELAY_MS),
            max_retries: DEFAULT_MAX_RETRIES,
            max_delay: Duration::from_secs(DEFAULT_MAX_DELAY_SECS),
        }
    }
}

/// Bundle of locks acquired for a single operation
pub struct LockBundle {
    /// File lock guard (if applicable)
    pub file_lock: Option<LockGuard>,
    /// Resource lock guard (if applicable)
    pub resource_lock: Option<ResourceLockGuard>,
}

impl LockBundle {
    /// Create an empty lock bundle
    pub fn empty() -> Self {
        Self {
            file_lock: None,
            resource_lock: None,
        }
    }

    /// Check if this bundle contains any locks
    pub fn has_locks(&self) -> bool {
        self.file_lock.is_some() || self.resource_lock.is_some()
    }
}

/// Unified access control manager
pub struct AccessControlManager {
    /// File lock manager (in-memory, intra-process)
    file_locks: Arc<FileLockManager>,
    /// Resource lock manager (build/test, in-memory, intra-process)
    resource_locks: Arc<ResourceLockManager>,
    /// Strategy for handling contention
    contention_strategy: ContentionStrategy,
    /// Tracking of files read by each agent (for read-before-write enforcement)
    read_tracking: RwLock<HashMap<String, HashSet<PathBuf>>>,
    /// Project root for determining resource scope
    project_root: PathBuf,
    /// Optional persistent lock store for inter-process coordination
    lock_store: Option<Arc<dyn LockPersistence>>,
}

impl AccessControlManager {
    /// Create a new access control manager
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            file_locks: Arc::new(FileLockManager::new()),
            resource_locks: Arc::new(ResourceLockManager::new()),
            contention_strategy: ContentionStrategy::default(),
            read_tracking: RwLock::new(HashMap::new()),
            project_root,
            lock_store: None,
        }
    }

    /// Create with custom managers (for testing or sharing)
    pub fn with_managers(
        file_locks: Arc<FileLockManager>,
        resource_locks: Arc<ResourceLockManager>,
        project_root: PathBuf,
    ) -> Self {
        Self {
            file_locks,
            resource_locks,
            contention_strategy: ContentionStrategy::default(),
            read_tracking: RwLock::new(HashMap::new()),
            project_root,
            lock_store: None,
        }
    }

    /// Set the contention strategy
    pub fn with_strategy(mut self, strategy: ContentionStrategy) -> Self {
        self.contention_strategy = strategy;
        self
    }

    /// Enable inter-process locking with a persistent lock store
    pub fn with_lock_persistence(mut self, lock_store: Arc<dyn LockPersistence>) -> Self {
        self.lock_store = Some(lock_store);
        self
    }

    /// Get the lock store (if configured)
    pub fn lock_store(&self) -> Option<&Arc<dyn LockPersistence>> {
        self.lock_store.as_ref()
    }

    /// Get a reference to the file lock manager
    pub fn file_locks(&self) -> &Arc<FileLockManager> {
        &self.file_locks
    }

    /// Get a reference to the resource lock manager
    pub fn resource_locks(&self) -> &Arc<ResourceLockManager> {
        &self.resource_locks
    }

    /// Track that an agent has read a file
    pub async fn track_file_read(&self, agent_id: &str, path: &Path) {
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut tracking = self.read_tracking.write().await;
        tracking
            .entry(agent_id.to_string())
            .or_default()
            .insert(canonical_path);
    }

    /// Check if an agent has read a file
    pub async fn has_read_file(&self, agent_id: &str, path: &Path) -> bool {
        let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let tracking = self.read_tracking.read().await;
        tracking
            .get(agent_id)
            .map(|files| files.contains(&canonical_path))
            .unwrap_or(false)
    }

    /// Validate that a write operation is allowed (file must have been read first)
    pub async fn validate_write(&self, agent_id: &str, path: &Path) -> Result<()> {
        // New files don't need to be read first
        if !path.exists() {
            return Ok(());
        }

        if !self.has_read_file(agent_id, path).await {
            return Err(anyhow!(
                "Must read file before writing: {}. Use read_file first.",
                path.display()
            ));
        }
        Ok(())
    }

    /// Clear read tracking for an agent (call on agent shutdown)
    pub async fn clear_tracking_for_agent(&self, agent_id: &str) {
        let mut tracking = self.read_tracking.write().await;
        tracking.remove(agent_id);
    }

    /// Get the lock requirement for a file operation
    pub fn get_file_lock_requirement(
        tool_name: &str,
        input: &Value,
    ) -> Option<(PathBuf, LockType)> {
        let path_str = input
            .get("path")
            .or_else(|| input.get("file_path"))
            .and_then(|v| v.as_str())?;

        let path = PathBuf::from(path_str);

        let lock_type = match tool_name {
            // Read operations - shared lock
            "read_file" | "list_directory" | "search_files" => LockType::Read,
            // Write operations - exclusive lock
            "write_file" | "edit_file" | "patch_file" | "delete_file" | "create_directory" => {
                LockType::Write
            }
            _ => return None,
        };

        Some((path, lock_type))
    }

    /// Detect if a bash command is a build command
    pub fn detect_build_command(command: &str) -> bool {
        let build_patterns = [
            "cargo build",
            "cargo b ",
            "cargo b\n",
            "cargo b$",
            "make ",
            "make\n",
            "make$",
            "cmake",
            "npm run build",
            "npm build",
            "yarn build",
            "pnpm build",
            "go build",
            "mvn compile",
            "mvn package",
            "gradle build",
            "gradle assemble",
            "msbuild",
            "dotnet build",
            "rustc ",
            "gcc ",
            "g++ ",
            "clang ",
            "clang++ ",
            "javac ",
            "tsc ",
            "webpack",
            "vite build",
            "rollup",
            "esbuild",
        ];

        let cmd_lower = command.to_lowercase();
        build_patterns
            .iter()
            .any(|p| cmd_lower.contains(&p.to_lowercase()))
    }

    /// Detect if a bash command is a test command
    pub fn detect_test_command(command: &str) -> bool {
        let test_patterns = [
            "cargo test",
            "cargo t ",
            "cargo t\n",
            "cargo t$",
            "npm test",
            "npm run test",
            "yarn test",
            "pnpm test",
            "go test",
            "pytest",
            "python -m pytest",
            "jest",
            "mocha",
            "vitest",
            "mvn test",
            "gradle test",
            "dotnet test",
            "rspec",
            "bundle exec rspec",
            "phpunit",
            "mix test",
            "elixir.*test",
        ];

        let cmd_lower = command.to_lowercase();
        test_patterns
            .iter()
            .any(|p| cmd_lower.contains(&p.to_lowercase()))
    }

    /// Get resource lock requirement for a bash command
    pub fn get_resource_requirement(&self, command: &str) -> Option<(ResourceType, ResourceScope)> {
        let is_build = Self::detect_build_command(command);
        let is_test = Self::detect_test_command(command);

        let resource_type = match (is_build, is_test) {
            (true, true) => ResourceType::BuildTest,
            (true, false) => ResourceType::Build,
            (false, true) => ResourceType::Test,
            (false, false) => return None,
        };

        // Use project scope based on project root
        let scope = ResourceScope::Project(self.project_root.clone());

        Some((resource_type, scope))
    }

    /// Acquire all necessary locks for a tool operation
    pub async fn acquire_for_tool(
        self: &Arc<Self>,
        agent_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> Result<LockBundle> {
        let mut bundle = LockBundle::empty();

        // Handle file operations
        if let Some((path, lock_type)) = Self::get_file_lock_requirement(tool_name, input) {
            // For write operations, validate read-before-write
            if lock_type == LockType::Write {
                self.validate_write(agent_id, &path).await?;
            }

            let file_lock = self
                .acquire_file_lock_with_retry(agent_id, &path, lock_type)
                .await?;
            bundle.file_lock = Some(file_lock);
        }

        // Handle bash commands (build/test)
        if tool_name == "execute_command"
            && let Some(command) = input.get("command").and_then(|v| v.as_str())
            && let Some((resource_type, scope)) = self.get_resource_requirement(command)
        {
            let resource_lock = self
                .acquire_resource_lock_with_retry(agent_id, resource_type, scope)
                .await?;
            bundle.resource_lock = Some(resource_lock);
        }

        Ok(bundle)
    }

    /// Convert LockType to string for persistent storage
    fn lock_type_to_string(lock_type: LockType) -> &'static str {
        match lock_type {
            LockType::Read => "file_read",
            LockType::Write => "file_write",
        }
    }

    /// Convert ResourceType to string for persistent storage
    fn resource_type_to_string(resource_type: ResourceType) -> &'static str {
        match resource_type {
            ResourceType::Build => "build",
            ResourceType::Test => "test",
            ResourceType::BuildTest => "build_test",
            ResourceType::GitIndex => "git_index",
            ResourceType::GitCommit => "git_commit",
            ResourceType::GitRemoteWrite => "git_remote_write",
            ResourceType::GitRemoteMerge => "git_remote_merge",
            ResourceType::GitBranch => "git_branch",
            ResourceType::GitDestructive => "git_destructive",
        }
    }

    /// Try to acquire persistent lock (if lock_store is configured)
    async fn try_acquire_persistent_lock(
        &self,
        agent_id: &str,
        lock_type_str: &str,
        resource_path: &str,
    ) -> Result<bool> {
        if let Some(store) = &self.lock_store {
            store
                .try_acquire(
                    lock_type_str,
                    resource_path,
                    agent_id,
                    Some(Duration::from_secs(PERSISTENT_LOCK_TIMEOUT_SECS)),
                )
                .await
        } else {
            // No persistent store, always succeed
            Ok(true)
        }
    }

    /// Release persistent lock (if lock_store is configured)
    async fn release_persistent_lock(
        &self,
        agent_id: &str,
        lock_type_str: &str,
        resource_path: &str,
    ) -> Result<()> {
        if let Some(store) = &self.lock_store {
            store
                .release(lock_type_str, resource_path, agent_id)
                .await?;
        }
        Ok(())
    }

    /// Acquire a file lock with retry based on contention strategy
    async fn acquire_file_lock_with_retry(
        &self,
        agent_id: &str,
        path: &Path,
        lock_type: LockType,
    ) -> Result<LockGuard> {
        let lock_type_str = Self::lock_type_to_string(lock_type);
        let resource_path = path.to_string_lossy().to_string();

        match &self.contention_strategy {
            ContentionStrategy::FailFast => {
                // Try persistent lock first
                if !self
                    .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                    .await?
                {
                    return Err(anyhow!(
                        "File {} is locked by another process",
                        path.display()
                    ));
                }

                // Then acquire in-memory lock
                match self
                    .file_locks
                    .acquire_lock(agent_id, path, lock_type)
                    .await
                {
                    Ok(guard) => Ok(guard),
                    Err(e) => {
                        // Release persistent lock on failure
                        let _ = self
                            .release_persistent_lock(agent_id, lock_type_str, &resource_path)
                            .await;
                        Err(e)
                    }
                }
            }
            ContentionStrategy::WaitWithTimeout(timeout) => {
                let deadline = tokio::time::Instant::now() + *timeout;
                let mut delay = Duration::from_millis(FILE_LOCK_BACKOFF_INITIAL_MS);

                loop {
                    // Try persistent lock first
                    if self
                        .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                        .await?
                    {
                        // Then try in-memory lock
                        match self
                            .file_locks
                            .acquire_lock(agent_id, path, lock_type)
                            .await
                        {
                            Ok(guard) => return Ok(guard),
                            Err(e) => {
                                // Release persistent lock and retry
                                let _ = self
                                    .release_persistent_lock(
                                        agent_id,
                                        lock_type_str,
                                        &resource_path,
                                    )
                                    .await;
                                if tokio::time::Instant::now() >= deadline {
                                    return Err(anyhow!(
                                        "Timeout waiting for file lock on {}: {}",
                                        path.display(),
                                        e
                                    ));
                                }
                            }
                        }
                    } else if tokio::time::Instant::now() >= deadline {
                        return Err(anyhow!(
                            "Timeout waiting for file lock on {} (held by another process)",
                            path.display()
                        ));
                    }

                    tokio::time::sleep(delay).await;
                    delay =
                        std::cmp::min(delay * 2, Duration::from_millis(FILE_LOCK_BACKOFF_MAX_MS));
                }
            }
            ContentionStrategy::RetryWithBackoff {
                initial_delay,
                max_retries,
                max_delay,
            } => {
                let mut delay = *initial_delay;
                let mut attempts = 0;

                loop {
                    // Try persistent lock first
                    if self
                        .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                        .await?
                    {
                        // Then try in-memory lock
                        match self
                            .file_locks
                            .acquire_lock(agent_id, path, lock_type)
                            .await
                        {
                            Ok(guard) => return Ok(guard),
                            Err(e) => {
                                // Release persistent lock and retry
                                let _ = self
                                    .release_persistent_lock(
                                        agent_id,
                                        lock_type_str,
                                        &resource_path,
                                    )
                                    .await;
                                attempts += 1;
                                if attempts > *max_retries {
                                    return Err(anyhow!(
                                        "Failed to acquire file lock on {} after {} attempts: {}",
                                        path.display(),
                                        max_retries,
                                        e
                                    ));
                                }
                                tracing::debug!(
                                    "Lock contention on {}, attempt {}/{}, waiting {:?}",
                                    path.display(),
                                    attempts,
                                    max_retries,
                                    delay
                                );
                            }
                        }
                    } else {
                        attempts += 1;
                        if attempts > *max_retries {
                            return Err(anyhow!(
                                "Failed to acquire file lock on {} after {} attempts (held by another process)",
                                path.display(),
                                max_retries
                            ));
                        }
                        tracing::debug!(
                            "Lock contention on {} (inter-process), attempt {}/{}, waiting {:?}",
                            path.display(),
                            attempts,
                            max_retries,
                            delay
                        );
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay * 2, *max_delay);
                }
            }
        }
    }

    /// Get the resource path string for a scope
    fn scope_to_resource_path(scope: &ResourceScope) -> String {
        match scope {
            ResourceScope::Global => "global".to_string(),
            ResourceScope::Project(path) => path.to_string_lossy().to_string(),
        }
    }

    /// Acquire a resource lock with retry based on contention strategy
    async fn acquire_resource_lock_with_retry(
        &self,
        agent_id: &str,
        resource_type: ResourceType,
        scope: ResourceScope,
    ) -> Result<ResourceLockGuard> {
        let lock_type_str = Self::resource_type_to_string(resource_type);
        let resource_path = Self::scope_to_resource_path(&scope);

        match &self.contention_strategy {
            ContentionStrategy::FailFast => {
                // Try persistent lock first
                if !self
                    .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                    .await?
                {
                    return Err(anyhow!("{} lock is held by another process", resource_type));
                }

                // Then acquire in-memory lock
                let description = format!("{} lock", resource_type);
                match self
                    .resource_locks
                    .acquire_resource(agent_id, resource_type, scope, &description)
                    .await
                {
                    Ok(guard) => Ok(guard),
                    Err(e) => {
                        // Release persistent lock on failure
                        let _ = self
                            .release_persistent_lock(agent_id, lock_type_str, &resource_path)
                            .await;
                        Err(e)
                    }
                }
            }
            ContentionStrategy::WaitWithTimeout(timeout) => {
                let deadline = tokio::time::Instant::now() + *timeout;
                let mut delay = Duration::from_millis(RESOURCE_LOCK_BACKOFF_INITIAL_MS);
                let description = format!("{} lock", resource_type);

                loop {
                    // Try persistent lock first
                    if self
                        .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                        .await?
                    {
                        // Then try in-memory lock
                        match self
                            .resource_locks
                            .acquire_resource(agent_id, resource_type, scope.clone(), &description)
                            .await
                        {
                            Ok(guard) => return Ok(guard),
                            Err(e) => {
                                // Release persistent lock and retry
                                let _ = self
                                    .release_persistent_lock(
                                        agent_id,
                                        lock_type_str,
                                        &resource_path,
                                    )
                                    .await;
                                if tokio::time::Instant::now() >= deadline {
                                    return Err(anyhow!(
                                        "Timeout waiting for {} lock: {}",
                                        resource_type,
                                        e
                                    ));
                                }
                            }
                        }
                    } else if tokio::time::Instant::now() >= deadline {
                        return Err(anyhow!(
                            "Timeout waiting for {} lock (held by another process)",
                            resource_type
                        ));
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(
                        delay * 2,
                        Duration::from_secs(RESOURCE_LOCK_BACKOFF_MAX_SECS),
                    );
                }
            }
            ContentionStrategy::RetryWithBackoff {
                initial_delay,
                max_retries,
                max_delay,
            } => {
                let mut delay = *initial_delay;
                let mut attempts = 0;
                let description = format!("{} lock", resource_type);

                loop {
                    // Try persistent lock first
                    if self
                        .try_acquire_persistent_lock(agent_id, lock_type_str, &resource_path)
                        .await?
                    {
                        // Then try in-memory lock
                        match self
                            .resource_locks
                            .acquire_resource(agent_id, resource_type, scope.clone(), &description)
                            .await
                        {
                            Ok(guard) => return Ok(guard),
                            Err(e) => {
                                // Release persistent lock and retry
                                let _ = self
                                    .release_persistent_lock(
                                        agent_id,
                                        lock_type_str,
                                        &resource_path,
                                    )
                                    .await;
                                attempts += 1;
                                if attempts > *max_retries {
                                    return Err(anyhow!(
                                        "Failed to acquire {} lock after {} attempts: {}",
                                        resource_type,
                                        max_retries,
                                        e
                                    ));
                                }
                                tracing::debug!(
                                    "{} lock contention, attempt {}/{}, waiting {:?}",
                                    resource_type,
                                    attempts,
                                    max_retries,
                                    delay
                                );
                            }
                        }
                    } else {
                        attempts += 1;
                        if attempts > *max_retries {
                            return Err(anyhow!(
                                "Failed to acquire {} lock after {} attempts (held by another process)",
                                resource_type,
                                max_retries
                            ));
                        }
                        tracing::debug!(
                            "{} lock contention (inter-process), attempt {}/{}, waiting {:?}",
                            resource_type,
                            attempts,
                            max_retries,
                            delay
                        );
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay * 2, *max_delay);
                }
            }
        }
    }

    /// Release all locks and tracking for an agent (call on agent shutdown)
    pub async fn cleanup_agent(&self, agent_id: &str) -> (usize, usize, usize) {
        let file_locks_released = self.file_locks.release_all_locks(agent_id).await;
        let resource_locks_released = self.resource_locks.release_all_for_agent(agent_id).await;

        // Also release persistent locks
        let persistent_locks_released = if let Some(store) = &self.lock_store {
            store.release_all_for_agent(agent_id).await.unwrap_or(0)
        } else {
            0
        };

        self.clear_tracking_for_agent(agent_id).await;
        (
            file_locks_released,
            resource_locks_released,
            persistent_locks_released,
        )
    }

    /// Cleanup stale persistent locks (call on startup)
    pub async fn cleanup_stale_locks(&self) -> Result<usize> {
        if let Some(store) = &self.lock_store {
            store.cleanup_stale().await
        } else {
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn create_manager() -> Arc<AccessControlManager> {
        Arc::new(AccessControlManager::new(PathBuf::from("/test/project")))
    }

    #[tokio::test]
    async fn test_track_file_read() {
        let manager = create_manager();

        let path = PathBuf::from("/test/file.txt");
        assert!(!manager.has_read_file("agent-1", &path).await);

        manager.track_file_read("agent-1", &path).await;
        assert!(manager.has_read_file("agent-1", &path).await);

        // Different agent hasn't read it
        assert!(!manager.has_read_file("agent-2", &path).await);
    }

    #[tokio::test]
    async fn test_validate_write_requires_read() {
        let manager = create_manager();

        // Create a temp file
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "test content").unwrap();

        // Write without reading should fail
        let result = manager.validate_write("agent-1", &file_path).await;
        assert!(result.is_err());

        // After reading, write should succeed
        manager.track_file_read("agent-1", &file_path).await;
        let result = manager.validate_write("agent-1", &file_path).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_write_allows_new_files() {
        let manager = create_manager();

        // New file that doesn't exist should be allowed
        let path = PathBuf::from("/nonexistent/new_file.txt");
        let result = manager.validate_write("agent-1", &path).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_file_lock_requirement() {
        // Read operations
        let input = serde_json::json!({"path": "/test/file.txt"});
        let req = AccessControlManager::get_file_lock_requirement("read_file", &input);
        assert!(matches!(req, Some((_, LockType::Read))));

        // Write operations
        let req = AccessControlManager::get_file_lock_requirement("write_file", &input);
        assert!(matches!(req, Some((_, LockType::Write))));

        let req = AccessControlManager::get_file_lock_requirement("edit_file", &input);
        assert!(matches!(req, Some((_, LockType::Write))));

        // Unknown tool
        let req = AccessControlManager::get_file_lock_requirement("unknown_tool", &input);
        assert!(req.is_none());
    }

    #[tokio::test]
    async fn test_detect_build_command() {
        assert!(AccessControlManager::detect_build_command("cargo build"));
        assert!(AccessControlManager::detect_build_command(
            "cargo build --release"
        ));
        assert!(AccessControlManager::detect_build_command("npm run build"));
        assert!(AccessControlManager::detect_build_command("make all"));
        assert!(AccessControlManager::detect_build_command(
            "gcc -o main main.c"
        ));

        assert!(!AccessControlManager::detect_build_command("ls -la"));
        assert!(!AccessControlManager::detect_build_command("cargo test"));
        assert!(!AccessControlManager::detect_build_command("echo hello"));
    }

    #[tokio::test]
    async fn test_detect_test_command() {
        assert!(AccessControlManager::detect_test_command("cargo test"));
        assert!(AccessControlManager::detect_test_command(
            "cargo test --release"
        ));
        assert!(AccessControlManager::detect_test_command("npm test"));
        assert!(AccessControlManager::detect_test_command("pytest"));
        assert!(AccessControlManager::detect_test_command("jest"));

        assert!(!AccessControlManager::detect_test_command("ls -la"));
        assert!(!AccessControlManager::detect_test_command("cargo build"));
        assert!(!AccessControlManager::detect_test_command("echo hello"));
    }

    #[tokio::test]
    async fn test_get_resource_requirement() {
        let manager = create_manager();

        // Build command
        let req = manager.get_resource_requirement("cargo build");
        assert!(matches!(req, Some((ResourceType::Build, _))));

        // Test command
        let req = manager.get_resource_requirement("cargo test");
        assert!(matches!(req, Some((ResourceType::Test, _))));

        // Build + test (cargo test with build)
        let req = manager.get_resource_requirement("cargo build && cargo test");
        assert!(matches!(req, Some((ResourceType::BuildTest, _))));

        // Neither
        let req = manager.get_resource_requirement("ls -la");
        assert!(req.is_none());
    }

    #[tokio::test]
    async fn test_acquire_for_tool_file_operation() {
        let manager = create_manager();

        // Read operation should succeed without prior read
        let input = serde_json::json!({"path": "/test/file.txt"});
        let result = manager
            .acquire_for_tool("agent-1", "read_file", &input)
            .await;
        assert!(result.is_ok());
        let bundle = result.unwrap();
        assert!(bundle.file_lock.is_some());
        assert!(bundle.resource_lock.is_none());
    }

    #[tokio::test]
    async fn test_acquire_for_tool_build_command() {
        let manager = create_manager();

        let input = serde_json::json!({"command": "cargo build"});
        let result = manager
            .acquire_for_tool("agent-1", "execute_command", &input)
            .await;
        assert!(result.is_ok());
        let bundle = result.unwrap();
        assert!(bundle.file_lock.is_none());
        assert!(bundle.resource_lock.is_some());
    }

    #[tokio::test]
    async fn test_cleanup_agent() {
        let manager = create_manager();

        // Acquire some locks
        let input = serde_json::json!({"path": "/test/file.txt"});
        let bundle = manager
            .acquire_for_tool("agent-1", "read_file", &input)
            .await
            .unwrap();

        // Forget the bundle to prevent auto-release
        std::mem::forget(bundle);

        // Track a file read
        manager
            .track_file_read("agent-1", &PathBuf::from("/test/file.txt"))
            .await;

        // Cleanup
        let (file_released, _resource_released, _persistent_released) =
            manager.cleanup_agent("agent-1").await;
        assert_eq!(file_released, 1);

        // Tracking should be cleared
        assert!(
            !manager
                .has_read_file("agent-1", &PathBuf::from("/test/file.txt"))
                .await
        );
    }

    #[tokio::test]
    async fn test_clear_tracking_for_agent() {
        let manager = create_manager();

        manager
            .track_file_read("agent-1", &PathBuf::from("/test/file1.txt"))
            .await;
        manager
            .track_file_read("agent-1", &PathBuf::from("/test/file2.txt"))
            .await;
        manager
            .track_file_read("agent-2", &PathBuf::from("/test/file1.txt"))
            .await;

        manager.clear_tracking_for_agent("agent-1").await;

        assert!(
            !manager
                .has_read_file("agent-1", &PathBuf::from("/test/file1.txt"))
                .await
        );
        assert!(
            !manager
                .has_read_file("agent-1", &PathBuf::from("/test/file2.txt"))
                .await
        );
        // agent-2 tracking should remain
        assert!(
            manager
                .has_read_file("agent-2", &PathBuf::from("/test/file1.txt"))
                .await
        );
    }
}
