//! SQLite-backed persistent lock storage for inter-process coordination
//!
//! Enables multiple brainwires-cli instances to coordinate file access
//! and build/test operations through a shared SQLite database.
//!
//! SQLite provides ACID compliance and immediate consistency, making it
//! ideal for lock coordination where eventual consistency would cause bugs.

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

/// Record representing a lock in the database
#[derive(Debug, Clone)]
pub struct LockRecord {
    /// Unique lock identifier
    pub lock_id: String,
    /// Type of lock: "file_read", "file_write", "build", "test", "build_test"
    pub lock_type: String,
    /// Resource being locked (file path or project path)
    pub resource_path: String,
    /// ID of the agent holding the lock
    pub agent_id: String,
    /// Process ID for stale lock detection
    pub process_id: i32,
    /// When the lock was acquired (Unix timestamp in milliseconds)
    pub acquired_at: i64,
    /// When the lock expires (optional, Unix timestamp in milliseconds)
    pub expires_at: Option<i64>,
    /// Hostname of the machine holding the lock
    pub hostname: String,
}

/// SQLite-backed persistent lock storage
pub struct LockStore {
    /// SQLite connection (wrapped in Mutex for thread safety)
    conn: Mutex<Connection>,
    /// Current process ID (cached for efficiency)
    current_pid: i32,
    /// Current hostname (cached for efficiency)
    current_hostname: String,
}

impl LockStore {
    /// Create a new lock store with default database path (~/.brainwires/locks.db)
    pub async fn new_default() -> Result<Self> {
        let db_path = Self::default_db_path()?;
        Self::new_with_path(&db_path).await
    }

    /// Get the default database path
    fn default_db_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        let brainwires_dir = home.join(".brainwires");
        std::fs::create_dir_all(&brainwires_dir)
            .context("Failed to create ~/.brainwires directory")?;
        Ok(brainwires_dir.join("locks.db"))
    }

    /// Create a new lock store with a custom database path
    pub async fn new_with_path(db_path: &PathBuf) -> Result<Self> {
        let current_pid = std::process::id() as i32;
        let current_hostname = gethostname::gethostname().to_string_lossy().to_string();

        // Open SQLite connection with WAL mode for better concurrent access
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open lock database at {:?}", db_path))?;

        // Enable WAL mode for better concurrent read/write performance
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA busy_timeout=5000;
             PRAGMA synchronous=NORMAL;",
        )
        .context("Failed to configure SQLite")?;

        let store = Self {
            conn: Mutex::new(conn),
            current_pid,
            current_hostname,
        };

        // Ensure the locks table exists
        store.ensure_table()?;

        Ok(store)
    }

    /// Ensure the locks table exists
    fn ensure_table(&self) -> Result<()> {
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS locks (
                lock_id TEXT PRIMARY KEY,
                lock_type TEXT NOT NULL,
                resource_path TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                acquired_at INTEGER NOT NULL,
                expires_at INTEGER,
                hostname TEXT NOT NULL
            )",
            [],
        )
        .context("Failed to create locks table")?;

        // Create index for faster queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_locks_agent ON locks(agent_id, process_id, hostname)",
            [],
        )
        .context("Failed to create locks index")?;

        Ok(())
    }

    /// Generate a unique lock ID
    fn generate_lock_id(lock_type: &str, resource_path: &str) -> String {
        format!("{}:{}", lock_type, resource_path)
    }

    /// Try to acquire a lock. Returns true if acquired, false if already held by another.
    pub async fn try_acquire(
        &self,
        lock_type: &str,
        resource_path: &str,
        agent_id: &str,
        timeout: Option<Duration>,
    ) -> Result<bool> {
        let lock_id = Self::generate_lock_id(lock_type, resource_path);
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        // Check if lock already exists
        let existing: Option<LockRecord> = conn
            .query_row(
                "SELECT lock_id, lock_type, resource_path, agent_id, process_id,
                        acquired_at, expires_at, hostname
                 FROM locks WHERE lock_id = ?",
                [&lock_id],
                |row| {
                    Ok(LockRecord {
                        lock_id: row.get(0)?,
                        lock_type: row.get(1)?,
                        resource_path: row.get(2)?,
                        agent_id: row.get(3)?,
                        process_id: row.get(4)?,
                        acquired_at: row.get(5)?,
                        expires_at: row.get(6)?,
                        hostname: row.get(7)?,
                    })
                },
            )
            .ok();

        if let Some(ref existing) = existing {
            // If held by same agent in same process, allow (idempotent)
            if existing.agent_id == agent_id
                && existing.process_id == self.current_pid
                && existing.hostname == self.current_hostname
            {
                return Ok(true);
            }

            // Check if the lock is stale
            if self.is_lock_stale(existing) {
                // Remove stale lock and proceed
                conn.execute("DELETE FROM locks WHERE lock_id = ?", [&lock_id])
                    .context("Failed to remove stale lock")?;
            } else {
                // Lock is held by another active process
                return Ok(false);
            }
        }

        // Acquire the lock
        let now = Utc::now().timestamp_millis();
        let expires_at = timeout.map(|t| now + t.as_millis() as i64);

        conn.execute(
            "INSERT OR REPLACE INTO locks
             (lock_id, lock_type, resource_path, agent_id, process_id, acquired_at, expires_at, hostname)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                lock_id,
                lock_type,
                resource_path,
                agent_id,
                self.current_pid,
                now,
                expires_at,
                self.current_hostname,
            ],
        )
        .context("Failed to acquire lock")?;

        Ok(true)
    }

    /// Release a lock
    pub async fn release(
        &self,
        lock_type: &str,
        resource_path: &str,
        agent_id: &str,
    ) -> Result<bool> {
        let lock_id = Self::generate_lock_id(lock_type, resource_path);
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        // Delete only if owned by this agent in this process
        let deleted = conn.execute(
            "DELETE FROM locks WHERE lock_id = ? AND agent_id = ? AND process_id = ? AND hostname = ?",
            params![lock_id, agent_id, self.current_pid, self.current_hostname],
        ).context("Failed to release lock")?;

        Ok(deleted > 0)
    }

    /// Release all locks held by a specific agent in the current process
    pub async fn release_all_for_agent(&self, agent_id: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        let deleted = conn
            .execute(
                "DELETE FROM locks WHERE agent_id = ? AND process_id = ? AND hostname = ?",
                params![agent_id, self.current_pid, self.current_hostname],
            )
            .context("Failed to release agent locks")?;

        Ok(deleted)
    }

    /// Check if a lock is held and by whom
    pub async fn is_locked(
        &self,
        lock_type: &str,
        resource_path: &str,
    ) -> Result<Option<LockRecord>> {
        let lock_id = Self::generate_lock_id(lock_type, resource_path);
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        conn.query_row(
            "SELECT lock_id, lock_type, resource_path, agent_id, process_id,
                    acquired_at, expires_at, hostname
             FROM locks WHERE lock_id = ?",
            [&lock_id],
            |row| {
                Ok(LockRecord {
                    lock_id: row.get(0)?,
                    lock_type: row.get(1)?,
                    resource_path: row.get(2)?,
                    agent_id: row.get(3)?,
                    process_id: row.get(4)?,
                    acquired_at: row.get(5)?,
                    expires_at: row.get(6)?,
                    hostname: row.get(7)?,
                })
            },
        )
        .optional()
        .context("Failed to check lock status")
    }

    /// Cleanup expired and stale locks
    pub async fn cleanup_stale(&self) -> Result<usize> {
        let now = Utc::now().timestamp_millis();
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        // First, delete expired locks
        let expired_count = conn
            .execute(
                "DELETE FROM locks WHERE expires_at IS NOT NULL AND expires_at < ?",
                [now],
            )
            .context("Failed to cleanup expired locks")?;

        // Then, get remaining locks to check for dead processes
        let mut stmt = conn
            .prepare(
                "SELECT lock_id, lock_type, resource_path, agent_id, process_id,
                        acquired_at, expires_at, hostname
                 FROM locks WHERE hostname = ?",
            )
            .context("Failed to prepare stale lock query")?;

        let locks: Vec<LockRecord> = stmt
            .query_map([&self.current_hostname], |row| {
                Ok(LockRecord {
                    lock_id: row.get(0)?,
                    lock_type: row.get(1)?,
                    resource_path: row.get(2)?,
                    agent_id: row.get(3)?,
                    process_id: row.get(4)?,
                    acquired_at: row.get(5)?,
                    expires_at: row.get(6)?,
                    hostname: row.get(7)?,
                })
            })
            .context("Failed to query locks")?
            .filter_map(|r| r.ok())
            .collect();

        drop(stmt);

        // Delete locks from dead processes
        let mut stale_count = 0;
        for lock in locks {
            if !Self::is_process_alive(lock.process_id) {
                conn.execute("DELETE FROM locks WHERE lock_id = ?", [&lock.lock_id])
                    .ok();
                stale_count += 1;
            }
        }

        Ok(expired_count + stale_count)
    }

    /// List all active locks
    pub async fn list_locks(&self) -> Result<Vec<LockRecord>> {
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT lock_id, lock_type, resource_path, agent_id, process_id,
                        acquired_at, expires_at, hostname
                 FROM locks",
            )
            .context("Failed to prepare list locks query")?;

        let locks = stmt
            .query_map([], |row| {
                Ok(LockRecord {
                    lock_id: row.get(0)?,
                    lock_type: row.get(1)?,
                    resource_path: row.get(2)?,
                    agent_id: row.get(3)?,
                    process_id: row.get(4)?,
                    acquired_at: row.get(5)?,
                    expires_at: row.get(6)?,
                    hostname: row.get(7)?,
                })
            })
            .context("Failed to query locks")?
            .filter_map(|r| r.ok())
            .collect();

        Ok(locks)
    }

    /// Force release a lock by ID (admin operation)
    pub async fn force_release(&self, lock_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("SQLite connection lock poisoned");
        conn.execute("DELETE FROM locks WHERE lock_id = ?", [lock_id])
            .context("Failed to force release lock")?;
        Ok(())
    }

    /// Check if a lock is stale (expired or from dead process)
    fn is_lock_stale(&self, lock: &LockRecord) -> bool {
        let now = Utc::now().timestamp_millis();

        // Check if expired
        if let Some(expires_at) = lock.expires_at
            && now > expires_at
        {
            return true;
        }

        // Check if process is dead (only if same hostname)
        if lock.hostname == self.current_hostname && !Self::is_process_alive(lock.process_id) {
            return true;
        }

        false
    }

    /// Check if a process is still running
    #[cfg(unix)]
    fn is_process_alive(pid: i32) -> bool {
        // On Unix, we can use kill with signal 0 to check if process exists
        // This doesn't actually send a signal, just checks if process exists
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(windows)]
    fn is_process_alive(pid: i32) -> bool {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };

        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if handle == 0 {
                return false;
            }

            let mut exit_code: u32 = 0;
            let result = GetExitCodeProcess(handle, &mut exit_code);
            CloseHandle(handle);

            result != 0 && exit_code == STILL_ACTIVE
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn is_process_alive(_pid: i32) -> bool {
        // On other platforms, assume process is alive to be safe
        true
    }

    /// Get lock statistics
    pub async fn stats(&self) -> Result<LockStats> {
        let locks = self.list_locks().await?;

        let mut file_read_locks = 0;
        let mut file_write_locks = 0;
        let mut build_locks = 0;
        let mut test_locks = 0;
        let mut stale_locks = 0;

        for lock in &locks {
            match lock.lock_type.as_str() {
                "file_read" => file_read_locks += 1,
                "file_write" => file_write_locks += 1,
                "build" => build_locks += 1,
                "test" => test_locks += 1,
                "build_test" => {
                    build_locks += 1;
                    test_locks += 1;
                }
                _ => {}
            }

            if self.is_lock_stale(lock) {
                stale_locks += 1;
            }
        }

        Ok(LockStats {
            total_locks: locks.len(),
            file_read_locks,
            file_write_locks,
            build_locks,
            test_locks,
            stale_locks,
        })
    }
}

/// Statistics about current locks
#[derive(Debug, Clone)]
pub struct LockStats {
    /// Total number of locks
    pub total_locks: usize,
    /// Number of file read locks
    pub file_read_locks: usize,
    /// Number of file write locks
    pub file_write_locks: usize,
    /// Number of build locks
    pub build_locks: usize,
    /// Number of test locks
    pub test_locks: usize,
    /// Number of stale locks (expired or from dead processes)
    pub stale_locks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> (LockStore, TempDir) {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test_locks.db");
        let store = LockStore::new_with_path(&db_path).await.unwrap();
        (store, temp)
    }

    #[tokio::test]
    async fn test_acquire_and_release_lock() {
        let (store, _temp) = create_test_store().await;

        // Acquire lock
        let acquired = store
            .try_acquire("file_write", "/test/file.txt", "agent-1", None)
            .await
            .unwrap();
        assert!(acquired);

        // Verify lock exists
        let lock = store
            .is_locked("file_write", "/test/file.txt")
            .await
            .unwrap();
        assert!(lock.is_some());
        assert_eq!(lock.unwrap().agent_id, "agent-1");

        // Release lock
        let released = store
            .release("file_write", "/test/file.txt", "agent-1")
            .await
            .unwrap();
        assert!(released);

        // Verify lock is gone
        let lock = store
            .is_locked("file_write", "/test/file.txt")
            .await
            .unwrap();
        assert!(lock.is_none());
    }

    #[tokio::test]
    async fn test_idempotent_acquire() {
        let (store, _temp) = create_test_store().await;

        // Acquire lock twice - should succeed both times
        let acquired1 = store
            .try_acquire("file_write", "/test/file.txt", "agent-1", None)
            .await
            .unwrap();
        let acquired2 = store
            .try_acquire("file_write", "/test/file.txt", "agent-1", None)
            .await
            .unwrap();

        assert!(acquired1);
        assert!(acquired2);
    }

    #[tokio::test]
    async fn test_lock_conflict() {
        let (store, _temp) = create_test_store().await;

        // Acquire lock as agent-1
        let acquired1 = store
            .try_acquire("file_write", "/test/file.txt", "agent-1", None)
            .await
            .unwrap();
        assert!(acquired1);

        // Try to acquire as agent-2 - should fail (same process, so not stale)
        // Note: In the same process, locks from different agents will conflict
        // because they have the same PID
        let acquired2 = store
            .try_acquire("file_write", "/test/file.txt", "agent-2", None)
            .await
            .unwrap();
        // In same process, different agent, same PID - this will fail
        assert!(!acquired2);
    }

    #[tokio::test]
    async fn test_release_all_for_agent() {
        let (store, _temp) = create_test_store().await;

        // Acquire multiple locks
        store
            .try_acquire("file_write", "/test/file1.txt", "agent-1", None)
            .await
            .unwrap();
        store
            .try_acquire("file_read", "/test/file2.txt", "agent-1", None)
            .await
            .unwrap();
        store
            .try_acquire("build", "/test/project", "agent-1", None)
            .await
            .unwrap();

        // Release all for agent-1
        let released = store.release_all_for_agent("agent-1").await.unwrap();
        assert_eq!(released, 3);

        // Verify all locks are gone
        let locks = store.list_locks().await.unwrap();
        assert!(locks.is_empty());
    }

    #[tokio::test]
    async fn test_expired_lock_cleanup() {
        let (store, _temp) = create_test_store().await;

        // Acquire lock with very short timeout (already expired)
        store
            .try_acquire(
                "file_write",
                "/test/file.txt",
                "agent-1",
                Some(Duration::from_millis(1)),
            )
            .await
            .unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cleanup should remove expired lock
        let cleaned = store.cleanup_stale().await.unwrap();
        assert_eq!(cleaned, 1);

        // Lock should be gone
        let lock = store
            .is_locked("file_write", "/test/file.txt")
            .await
            .unwrap();
        assert!(lock.is_none());
    }

    #[tokio::test]
    async fn test_list_locks() {
        let (store, _temp) = create_test_store().await;

        store
            .try_acquire("file_write", "/test/file1.txt", "agent-1", None)
            .await
            .unwrap();
        store
            .try_acquire("file_read", "/test/file2.txt", "agent-1", None)
            .await
            .unwrap();

        let locks = store.list_locks().await.unwrap();
        assert_eq!(locks.len(), 2);
    }

    #[tokio::test]
    async fn test_stats() {
        let (store, _temp) = create_test_store().await;

        store
            .try_acquire("file_write", "/test/file1.txt", "agent-1", None)
            .await
            .unwrap();
        store
            .try_acquire("file_read", "/test/file2.txt", "agent-1", None)
            .await
            .unwrap();
        store
            .try_acquire("build", "/test/project", "agent-1", None)
            .await
            .unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_locks, 3);
        assert_eq!(stats.file_write_locks, 1);
        assert_eq!(stats.file_read_locks, 1);
        assert_eq!(stats.build_locks, 1);
    }

    #[test]
    fn test_is_process_alive() {
        // Current process should be alive
        let current_pid = std::process::id() as i32;
        assert!(LockStore::is_process_alive(current_pid));

        // PID 0 (init/kernel) should exist on Unix
        #[cfg(unix)]
        {
            // Note: PID 1 (init) should always exist, but we might not have permission
            // to signal it. PID of current process is a safer test.
        }
    }
}
