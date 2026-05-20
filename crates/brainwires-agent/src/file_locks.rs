//! File locking system for multi-agent coordination
//!
//! Provides a mechanism for agents to "checkout" files, preventing concurrent
//! modifications and ensuring consistency across background task agents.

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const DEFAULT_LOCK_TIMEOUT_SECS: u64 = 300;
const LOCK_POLL_INTERVAL_MS: u64 = 50;

/// Type of file lock
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    /// Shared read lock - multiple agents can hold simultaneously
    Read,
    /// Exclusive write lock - only one agent can hold
    Write,
}

/// Information about a held lock
#[derive(Debug, Clone)]
pub struct LockInfo {
    /// ID of the agent holding the lock
    pub agent_id: String,
    /// Type of lock
    pub lock_type: LockType,
    /// When the lock was acquired
    pub acquired_at: Instant,
    /// Optional timeout for auto-release
    pub timeout: Option<Duration>,
}

impl LockInfo {
    /// Check if the lock has expired
    pub fn is_expired(&self) -> bool {
        if let Some(timeout) = self.timeout {
            self.acquired_at.elapsed() > timeout
        } else {
            false
        }
    }

    /// Get remaining time before timeout
    pub fn time_remaining(&self) -> Option<Duration> {
        self.timeout.map(|timeout| {
            let elapsed = self.acquired_at.elapsed();
            if elapsed >= timeout {
                Duration::ZERO
            } else {
                timeout - elapsed
            }
        })
    }
}

/// Internal lock state for a file
#[derive(Debug, Clone, Default)]
struct FileLockState {
    /// Write lock (exclusive)
    write_lock: Option<LockInfo>,
    /// Read locks (shared)
    read_locks: Vec<LockInfo>,
}

/// Guard that releases a lock when dropped
pub struct LockGuard {
    manager: Arc<FileLockManager>,
    agent_id: String,
    path: PathBuf,
    lock_type: LockType,
}

impl std::fmt::Debug for LockGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockGuard")
            .field("agent_id", &self.agent_id)
            .field("path", &self.path)
            .field("lock_type", &self.lock_type)
            .finish()
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        // Use blocking release since we're in Drop
        let manager = self.manager.clone();
        let agent_id = self.agent_id.clone();
        let path = self.path.clone();
        let lock_type = self.lock_type;

        // Spawn a task to release the lock asynchronously
        tokio::spawn(async move {
            if let Err(e) = manager
                .release_lock_internal(&agent_id, &path, lock_type)
                .await
            {
                eprintln!("Warning: Failed to release lock on drop: {}", e);
            }
        });
    }
}

/// Manages file locks across multiple agents
pub struct FileLockManager {
    /// Map of file paths to their lock states
    locks: RwLock<HashMap<PathBuf, FileLockState>>,
    /// Default timeout for locks
    default_timeout: Option<Duration>,
    /// Waiting agents: agent_id -> set of paths they're waiting for
    /// Used for deadlock detection
    waiting: RwLock<HashMap<String, HashSet<PathBuf>>>,
}

impl FileLockManager {
    /// Create a new file lock manager
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            default_timeout: Some(Duration::from_secs(DEFAULT_LOCK_TIMEOUT_SECS)),
            waiting: RwLock::new(HashMap::new()),
        }
    }

    /// Create a file lock manager with a custom default timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            default_timeout: Some(timeout),
            waiting: RwLock::new(HashMap::new()),
        }
    }

    /// Create a file lock manager with no default timeout
    pub fn without_timeout() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            default_timeout: None,
            waiting: RwLock::new(HashMap::new()),
        }
    }

    /// Acquire a lock on a file
    ///
    /// Returns a LockGuard that automatically releases the lock when dropped.
    #[tracing::instrument(name = "agent.lock.acquire", skip_all, fields(agent_id, lock_type = ?lock_type))]
    pub async fn acquire_lock(
        self: &Arc<Self>,
        agent_id: &str,
        path: impl AsRef<Path>,
        lock_type: LockType,
    ) -> Result<LockGuard> {
        self.acquire_lock_with_timeout(agent_id, path, lock_type, self.default_timeout)
            .await
    }

    /// Acquire a lock with a specific timeout
    #[tracing::instrument(name = "agent.lock.acquire_timeout", skip_all, fields(agent_id, lock_type = ?lock_type))]
    pub async fn acquire_lock_with_timeout(
        self: &Arc<Self>,
        agent_id: &str,
        path: impl AsRef<Path>,
        lock_type: LockType,
        timeout: Option<Duration>,
    ) -> Result<LockGuard> {
        let path = path.as_ref().to_path_buf();
        let mut locks = self.locks.write().await;

        // Clean up expired locks first
        self.cleanup_expired_internal(&mut locks);

        let state = locks.entry(path.clone()).or_default();

        match lock_type {
            LockType::Read => {
                // Check for write lock
                if let Some(write_lock) = &state.write_lock
                    && write_lock.agent_id != agent_id
                {
                    return Err(anyhow!(
                        "File {} is write-locked by agent {}",
                        path.display(),
                        write_lock.agent_id
                    ));
                }

                // Add read lock
                state.read_locks.push(LockInfo {
                    agent_id: agent_id.to_string(),
                    lock_type: LockType::Read,
                    acquired_at: Instant::now(),
                    timeout,
                });
            }
            LockType::Write => {
                // Check for existing write lock
                if let Some(write_lock) = &state.write_lock {
                    if write_lock.agent_id != agent_id {
                        return Err(anyhow!(
                            "File {} is already write-locked by agent {}",
                            path.display(),
                            write_lock.agent_id
                        ));
                    }
                    // Same agent already has write lock, return success
                    return Ok(LockGuard {
                        manager: Arc::clone(self),
                        agent_id: agent_id.to_string(),
                        path,
                        lock_type,
                    });
                }

                // Check for read locks by other agents
                let other_readers: Vec<_> = state
                    .read_locks
                    .iter()
                    .filter(|lock| lock.agent_id != agent_id)
                    .map(|lock| lock.agent_id.clone())
                    .collect();

                if !other_readers.is_empty() {
                    return Err(anyhow!(
                        "File {} has read locks from agents: {:?}",
                        path.display(),
                        other_readers
                    ));
                }

                // Set write lock
                state.write_lock = Some(LockInfo {
                    agent_id: agent_id.to_string(),
                    lock_type: LockType::Write,
                    acquired_at: Instant::now(),
                    timeout,
                });
            }
        }

        Ok(LockGuard {
            manager: Arc::clone(self),
            agent_id: agent_id.to_string(),
            path,
            lock_type,
        })
    }

    /// Acquire a lock with waiting and timeout
    ///
    /// This method will wait up to `wait_timeout` for the lock to become available.
    /// It includes deadlock detection to prevent circular wait scenarios.
    #[tracing::instrument(name = "agent.lock.acquire_wait", skip_all, fields(agent_id, lock_type = ?lock_type))]
    pub async fn acquire_with_wait(
        self: &Arc<Self>,
        agent_id: &str,
        path: impl AsRef<Path>,
        lock_type: LockType,
        wait_timeout: Duration,
    ) -> Result<LockGuard> {
        let path = path.as_ref().to_path_buf();
        let deadline = Instant::now() + wait_timeout;
        let poll_interval = Duration::from_millis(LOCK_POLL_INTERVAL_MS);

        loop {
            // Check for deadlock before waiting
            if self.would_deadlock(agent_id, &path).await {
                return Err(anyhow!(
                    "Deadlock detected: agent {} waiting for {} would create circular dependency",
                    agent_id,
                    path.display()
                ));
            }

            // Try to acquire the lock
            match self
                .acquire_lock_with_timeout(agent_id, &path, lock_type, self.default_timeout)
                .await
            {
                Ok(guard) => {
                    // Successfully acquired - remove from waiting set
                    self.stop_waiting(agent_id, &path).await;
                    return Ok(guard);
                }
                Err(_) if Instant::now() < deadline => {
                    // Record that we're waiting for this path
                    self.start_waiting(agent_id, &path).await;

                    // Clean up expired locks that might be blocking us
                    self.cleanup_expired().await;

                    // Wait before retrying
                    tokio::time::sleep(poll_interval).await;
                }
                Err(e) => {
                    // Timeout or other error
                    self.stop_waiting(agent_id, &path).await;
                    return Err(anyhow!(
                        "Lock acquisition timeout after {:?}: {}",
                        wait_timeout,
                        e
                    ));
                }
            }
        }
    }

    /// Check if acquiring a lock would cause a deadlock
    ///
    /// Uses cycle detection in the wait-for graph.
    async fn would_deadlock(&self, agent_id: &str, target_path: &Path) -> bool {
        let locks = self.locks.read().await;
        let waiting = self.waiting.read().await;

        // Find who currently holds the lock on target_path
        let current_holders = if let Some(state) = locks.get(target_path) {
            let mut holders = HashSet::new();
            if let Some(write_lock) = &state.write_lock {
                holders.insert(write_lock.agent_id.clone());
            }
            for read_lock in &state.read_locks {
                holders.insert(read_lock.agent_id.clone());
            }
            holders
        } else {
            return false; // No one holds the lock
        };

        // If we already hold the lock, no deadlock
        if current_holders.contains(agent_id) {
            return false;
        }

        // DFS to detect cycle: would any holder eventually wait for us?
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for holder in current_holders {
            stack.push(holder);
        }

        while let Some(current) = stack.pop() {
            if current == agent_id {
                return true; // Cycle detected
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            // Find what paths this agent is waiting for
            if let Some(waiting_for) = waiting.get(&current) {
                // Find who holds those paths
                for waiting_path in waiting_for {
                    if let Some(state) = locks.get(waiting_path) {
                        if let Some(write_lock) = &state.write_lock
                            && !visited.contains(&write_lock.agent_id)
                        {
                            stack.push(write_lock.agent_id.clone());
                        }
                        for read_lock in &state.read_locks {
                            if !visited.contains(&read_lock.agent_id) {
                                stack.push(read_lock.agent_id.clone());
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// Record that an agent is waiting for a path
    async fn start_waiting(&self, agent_id: &str, path: &Path) {
        let mut waiting = self.waiting.write().await;
        waiting
            .entry(agent_id.to_string())
            .or_insert_with(HashSet::new)
            .insert(path.to_path_buf());
    }

    /// Remove an agent from the waiting set for a path
    async fn stop_waiting(&self, agent_id: &str, path: &Path) {
        let mut waiting = self.waiting.write().await;
        if let Some(paths) = waiting.get_mut(agent_id) {
            paths.remove(path);
            if paths.is_empty() {
                waiting.remove(agent_id);
            }
        }
    }

    /// Clear all waiting entries for an agent (e.g., when agent exits)
    pub async fn clear_waiting(&self, agent_id: &str) {
        let mut waiting = self.waiting.write().await;
        waiting.remove(agent_id);
    }

    /// Get all agents currently waiting for locks
    pub async fn get_waiting_agents(&self) -> HashMap<String, Vec<PathBuf>> {
        let waiting = self.waiting.read().await;
        waiting
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    }

    /// Release a specific lock
    #[tracing::instrument(name = "agent.lock.release", skip_all, fields(agent_id, lock_type = ?lock_type))]
    pub async fn release_lock(
        &self,
        agent_id: &str,
        path: impl AsRef<Path>,
        lock_type: LockType,
    ) -> Result<()> {
        self.release_lock_internal(agent_id, path.as_ref(), lock_type)
            .await
    }

    /// Internal release implementation
    async fn release_lock_internal(
        &self,
        agent_id: &str,
        path: &Path,
        lock_type: LockType,
    ) -> Result<()> {
        let mut locks = self.locks.write().await;

        if let Some(state) = locks.get_mut(path) {
            match lock_type {
                LockType::Read => {
                    // Remove matching read lock
                    let original_len = state.read_locks.len();
                    state.read_locks.retain(|lock| lock.agent_id != agent_id);

                    if state.read_locks.len() == original_len {
                        return Err(anyhow!(
                            "No read lock found for agent {} on {}",
                            agent_id,
                            path.display()
                        ));
                    }
                }
                LockType::Write => {
                    // Remove write lock if it belongs to this agent
                    if let Some(write_lock) = &state.write_lock {
                        if write_lock.agent_id == agent_id {
                            state.write_lock = None;
                        } else {
                            return Err(anyhow!(
                                "Write lock on {} belongs to agent {}, not {}",
                                path.display(),
                                write_lock.agent_id,
                                agent_id
                            ));
                        }
                    } else {
                        return Err(anyhow!("No write lock found on {}", path.display()));
                    }
                }
            }

            // Clean up empty state
            if state.write_lock.is_none() && state.read_locks.is_empty() {
                locks.remove(path);
            }
        } else {
            return Err(anyhow!("No locks found for {}", path.display()));
        }

        Ok(())
    }

    /// Release all locks held by an agent
    #[tracing::instrument(name = "agent.lock.release_all", skip(self))]
    pub async fn release_all_locks(&self, agent_id: &str) -> usize {
        let mut locks = self.locks.write().await;
        let mut released = 0;

        for state in locks.values_mut() {
            // Release write lock
            if let Some(write_lock) = &state.write_lock
                && write_lock.agent_id == agent_id
            {
                state.write_lock = None;
                released += 1;
            }

            // Release read locks
            let original_len = state.read_locks.len();
            state.read_locks.retain(|lock| lock.agent_id != agent_id);
            released += original_len - state.read_locks.len();
        }

        // Clean up empty entries
        locks.retain(|_, state| state.write_lock.is_some() || !state.read_locks.is_empty());

        released
    }

    /// Check if a file is locked
    pub async fn check_lock(&self, path: impl AsRef<Path>) -> Option<LockInfo> {
        let locks = self.locks.read().await;

        if let Some(state) = locks.get(path.as_ref()) {
            // Return write lock if present, otherwise first read lock
            if let Some(write_lock) = &state.write_lock {
                return Some(write_lock.clone());
            }
            if let Some(read_lock) = state.read_locks.first() {
                return Some(read_lock.clone());
            }
        }

        None
    }

    /// Check if a file is locked by a specific agent
    pub async fn is_locked_by(&self, path: impl AsRef<Path>, agent_id: &str) -> bool {
        let locks = self.locks.read().await;

        if let Some(state) = locks.get(path.as_ref()) {
            if let Some(write_lock) = &state.write_lock
                && write_lock.agent_id == agent_id
            {
                return true;
            }
            if state
                .read_locks
                .iter()
                .any(|lock| lock.agent_id == agent_id)
            {
                return true;
            }
        }

        false
    }

    /// Check if a file can be locked with a specific type by an agent
    pub async fn can_acquire(
        &self,
        path: impl AsRef<Path>,
        agent_id: &str,
        lock_type: LockType,
    ) -> bool {
        let locks = self.locks.read().await;

        if let Some(state) = locks.get(path.as_ref()) {
            match lock_type {
                LockType::Read => {
                    // Can read if no write lock or own write lock
                    if let Some(write_lock) = &state.write_lock {
                        return write_lock.agent_id == agent_id;
                    }
                    true
                }
                LockType::Write => {
                    // Can write if no other agent has any lock
                    if let Some(write_lock) = &state.write_lock
                        && write_lock.agent_id != agent_id
                    {
                        return false;
                    }
                    !state
                        .read_locks
                        .iter()
                        .any(|lock| lock.agent_id != agent_id)
                }
            }
        } else {
            true
        }
    }

    /// Force release a lock (admin operation)
    pub async fn force_release(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut locks = self.locks.write().await;

        if locks.remove(path.as_ref()).is_some() {
            Ok(())
        } else {
            Err(anyhow!("No locks found for {}", path.as_ref().display()))
        }
    }

    /// Get all currently held locks
    pub async fn list_locks(&self) -> Vec<(PathBuf, LockInfo)> {
        let locks = self.locks.read().await;
        let mut result = Vec::new();

        for (path, state) in locks.iter() {
            if let Some(write_lock) = &state.write_lock {
                result.push((path.clone(), write_lock.clone()));
            }
            for read_lock in &state.read_locks {
                result.push((path.clone(), read_lock.clone()));
            }
        }

        result
    }

    /// Get locks held by a specific agent
    pub async fn locks_for_agent(&self, agent_id: &str) -> Vec<(PathBuf, LockInfo)> {
        let locks = self.locks.read().await;
        let mut result = Vec::new();

        for (path, state) in locks.iter() {
            if let Some(write_lock) = &state.write_lock
                && write_lock.agent_id == agent_id
            {
                result.push((path.clone(), write_lock.clone()));
            }
            for read_lock in &state.read_locks {
                if read_lock.agent_id == agent_id {
                    result.push((path.clone(), read_lock.clone()));
                }
            }
        }

        result
    }

    /// Clean up expired locks
    pub async fn cleanup_expired(&self) -> usize {
        let mut locks = self.locks.write().await;
        self.cleanup_expired_internal(&mut locks)
    }

    /// Internal cleanup implementation
    fn cleanup_expired_internal(&self, locks: &mut HashMap<PathBuf, FileLockState>) -> usize {
        let mut cleaned = 0;

        for state in locks.values_mut() {
            // Clean expired write lock
            if let Some(write_lock) = &state.write_lock
                && write_lock.is_expired()
            {
                state.write_lock = None;
                cleaned += 1;
            }

            // Clean expired read locks
            let original_len = state.read_locks.len();
            state.read_locks.retain(|lock| !lock.is_expired());
            cleaned += original_len - state.read_locks.len();
        }

        // Remove empty entries
        locks.retain(|_, state| state.write_lock.is_some() || !state.read_locks.is_empty());

        cleaned
    }

    /// Get statistics about current locks
    pub async fn stats(&self) -> LockStats {
        let locks = self.locks.read().await;

        let mut total_files = 0;
        let mut total_write_locks = 0;
        let mut total_read_locks = 0;

        for state in locks.values() {
            total_files += 1;
            if state.write_lock.is_some() {
                total_write_locks += 1;
            }
            total_read_locks += state.read_locks.len();
        }

        LockStats {
            total_files,
            total_write_locks,
            total_read_locks,
        }
    }
}

impl Default for FileLockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about current locks
#[derive(Debug, Clone)]
pub struct LockStats {
    /// Number of files with locks
    pub total_files: usize,
    /// Number of write locks
    pub total_write_locks: usize,
    /// Number of read locks
    pub total_read_locks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_write_lock() {
        let manager = Arc::new(FileLockManager::new());
        let guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        assert_eq!(guard.lock_type, LockType::Write);
        assert!(manager.is_locked_by("/test/file.txt", "agent-1").await);
    }

    #[tokio::test]
    async fn test_acquire_read_lock() {
        let manager = Arc::new(FileLockManager::new());
        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Read)
            .await
            .unwrap();

        assert!(manager.is_locked_by("/test/file.txt", "agent-1").await);
    }

    #[tokio::test]
    async fn test_multiple_read_locks() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Read)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Read)
            .await
            .unwrap();

        assert!(manager.is_locked_by("/test/file.txt", "agent-1").await);
        assert!(manager.is_locked_by("/test/file.txt", "agent-2").await);
    }

    #[tokio::test]
    async fn test_write_lock_blocks_other_write() {
        let manager = Arc::new(FileLockManager::new());

        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        let result = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Write)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_lock_blocks_read() {
        let manager = Arc::new(FileLockManager::new());

        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        let result = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Read)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_lock_blocks_write() {
        let manager = Arc::new(FileLockManager::new());

        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Read)
            .await
            .unwrap();

        let result = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Write)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_same_agent_reacquire_write() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        // Same agent can reacquire their own write lock
        assert!(manager.is_locked_by("/test/file.txt", "agent-1").await);
    }

    #[tokio::test]
    async fn test_release_all_locks() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file1.txt", LockType::Write)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-1", "/test/file2.txt", LockType::Read)
            .await
            .unwrap();

        // Forget guards to prevent auto-release
        std::mem::forget(_guard1);
        std::mem::forget(_guard2);

        let released = manager.release_all_locks("agent-1").await;
        assert_eq!(released, 2);
    }

    #[tokio::test]
    async fn test_lock_stats() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file1.txt", LockType::Write)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-2", "/test/file2.txt", LockType::Read)
            .await
            .unwrap();
        let _guard3 = manager
            .acquire_lock("agent-3", "/test/file2.txt", LockType::Read)
            .await
            .unwrap();

        let stats = manager.stats().await;
        assert_eq!(stats.total_files, 2);
        assert_eq!(stats.total_write_locks, 1);
        assert_eq!(stats.total_read_locks, 2);
    }

    #[tokio::test]
    async fn test_can_acquire() {
        let manager = Arc::new(FileLockManager::new());

        // No locks - can acquire anything
        assert!(
            manager
                .can_acquire("/test/file.txt", "agent-1", LockType::Write)
                .await
        );
        assert!(
            manager
                .can_acquire("/test/file.txt", "agent-1", LockType::Read)
                .await
        );

        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        // Same agent can acquire
        assert!(
            manager
                .can_acquire("/test/file.txt", "agent-1", LockType::Write)
                .await
        );
        assert!(
            manager
                .can_acquire("/test/file.txt", "agent-1", LockType::Read)
                .await
        );

        // Other agent cannot
        assert!(
            !manager
                .can_acquire("/test/file.txt", "agent-2", LockType::Write)
                .await
        );
        assert!(
            !manager
                .can_acquire("/test/file.txt", "agent-2", LockType::Read)
                .await
        );
    }

    #[tokio::test]
    async fn test_expired_lock_cleanup() {
        let manager = Arc::new(FileLockManager::new());

        // Acquire lock with very short timeout
        let _guard = manager
            .acquire_lock_with_timeout(
                "agent-1",
                "/test/file.txt",
                LockType::Write,
                Some(Duration::from_millis(1)),
            )
            .await
            .unwrap();

        // Forget guard to prevent auto-release
        std::mem::forget(_guard);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cleanup should remove expired lock
        let cleaned = manager.cleanup_expired().await;
        assert_eq!(cleaned, 1);

        // Now another agent can acquire
        let result = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Write)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_force_release() {
        let manager = Arc::new(FileLockManager::new());

        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        // Forget guard
        std::mem::forget(_guard);

        // Force release
        manager.force_release("/test/file.txt").await.unwrap();

        // Another agent can now acquire
        let result = manager
            .acquire_lock("agent-2", "/test/file.txt", LockType::Write)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_locks() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file1.txt", LockType::Write)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-2", "/test/file2.txt", LockType::Read)
            .await
            .unwrap();

        let locks = manager.list_locks().await;
        assert_eq!(locks.len(), 2);
    }

    #[tokio::test]
    async fn test_locks_for_agent() {
        let manager = Arc::new(FileLockManager::new());

        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file1.txt", LockType::Write)
            .await
            .unwrap();
        let _guard2 = manager
            .acquire_lock("agent-1", "/test/file2.txt", LockType::Read)
            .await
            .unwrap();
        let _guard3 = manager
            .acquire_lock("agent-2", "/test/file3.txt", LockType::Write)
            .await
            .unwrap();

        let agent1_locks = manager.locks_for_agent("agent-1").await;
        assert_eq!(agent1_locks.len(), 2);

        let agent2_locks = manager.locks_for_agent("agent-2").await;
        assert_eq!(agent2_locks.len(), 1);
    }

    #[tokio::test]
    async fn test_acquire_with_wait_success() {
        let manager = Arc::new(FileLockManager::new());

        // Acquire initial lock
        let guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        // Spawn task to release after delay
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(guard);
            // Give time for the drop to process
            tokio::time::sleep(Duration::from_millis(10)).await;
            manager_clone.cleanup_expired().await;
        });

        // Agent 2 should wait and eventually acquire
        let result = manager
            .acquire_with_wait(
                "agent-2",
                "/test/file.txt",
                LockType::Write,
                Duration::from_millis(500),
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_acquire_with_wait_timeout() {
        let manager = Arc::new(FileLockManager::new());

        // Acquire lock that won't be released
        let _guard = manager
            .acquire_lock("agent-1", "/test/file.txt", LockType::Write)
            .await
            .unwrap();

        // Agent 2 should timeout
        let result = manager
            .acquire_with_wait(
                "agent-2",
                "/test/file.txt",
                LockType::Write,
                Duration::from_millis(100),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout"));
    }

    #[tokio::test]
    async fn test_deadlock_detection() {
        let manager = Arc::new(FileLockManager::new());

        // Agent 1 holds lock on file1
        let _guard1 = manager
            .acquire_lock("agent-1", "/test/file1.txt", LockType::Write)
            .await
            .unwrap();

        // Agent 2 holds lock on file2
        let _guard2 = manager
            .acquire_lock("agent-2", "/test/file2.txt", LockType::Write)
            .await
            .unwrap();

        // Simulate agent 1 waiting for file2
        manager
            .start_waiting("agent-1", std::path::Path::new("/test/file2.txt"))
            .await;

        // Agent 2 trying to acquire file1 would create a deadlock
        assert!(
            manager
                .would_deadlock("agent-2", std::path::Path::new("/test/file1.txt"))
                .await
        );

        // But agent 3 trying to acquire file1 would NOT create a deadlock
        assert!(
            !manager
                .would_deadlock("agent-3", std::path::Path::new("/test/file1.txt"))
                .await
        );
    }

    #[tokio::test]
    async fn test_waiting_agents() {
        let manager = Arc::new(FileLockManager::new());

        // Record some waiting agents
        manager
            .start_waiting("agent-1", std::path::Path::new("/test/file1.txt"))
            .await;
        manager
            .start_waiting("agent-1", std::path::Path::new("/test/file2.txt"))
            .await;
        manager
            .start_waiting("agent-2", std::path::Path::new("/test/file1.txt"))
            .await;

        let waiting = manager.get_waiting_agents().await;
        assert_eq!(waiting.len(), 2);
        assert_eq!(waiting.get("agent-1").map(|v| v.len()), Some(2));
        assert_eq!(waiting.get("agent-2").map(|v| v.len()), Some(1));

        // Clear agent 1's waiting
        manager.clear_waiting("agent-1").await;

        let waiting = manager.get_waiting_agents().await;
        assert_eq!(waiting.len(), 1);
        assert!(!waiting.contains_key("agent-1"));
    }
}
