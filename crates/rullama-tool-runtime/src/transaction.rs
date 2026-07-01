//! Two-phase commit transaction manager for file write operations.
//!
//! [`TransactionManager`] implements [`StagingBackend`] from `rullama-core`.
//!
//! ## Protocol
//!
//! 1. **Stage** — calls to [`TransactionManager::stage`] write content to a
//!    temporary directory with a key-addressed filename.  The target path is
//!    **not** touched.
//!
//! 2. **Commit** — [`TransactionManager::commit`] atomically renames each staged
//!    file to its target path.  On cross-filesystem moves a copy+delete fallback
//!    is used.  Parent directories are created as needed.
//!
//! 3. **Rollback** — [`TransactionManager::rollback`] deletes all staged files
//!    from the temp dir without touching any target path.
//!
//! A `TransactionManager` is single-use per transaction: after `commit()` or
//! `rollback()` the queue is empty and new stages can be accepted.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use rullama_tool_runtime::transaction::TransactionManager;
//! use rullama_core::{StagedWrite, ToolContext};
//!
//! let mgr = Arc::new(TransactionManager::new().unwrap());
//! let ctx = ToolContext::default().with_staging_backend(mgr.clone());
//!
//! // … execute file tools against `ctx` — writes are staged, not applied …
//!
//! mgr.commit().unwrap(); // atomically write all staged files
//! // or mgr.rollback();  // discard everything
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use rullama_core::{CommitResult, StagedWrite, StagingBackend};

// ── Internal types ────────────────────────────────────────────────────────────

/// One pending staged write entry.
#[derive(Debug)]
struct StagedEntry {
    staged_path: PathBuf,
    target_path: PathBuf,
    content: String,
}

#[derive(Debug)]
struct Inner {
    staging_dir: PathBuf,
    /// key → entry
    staged: HashMap<String, StagedEntry>,
}

// ── TransactionManager ────────────────────────────────────────────────────────

/// Filesystem-backed two-phase commit transaction manager.
///
/// Wrap in [`Arc`] and attach to [`ToolContext`][rullama_core::ToolContext]
/// via [`with_staging_backend`][rullama_core::ToolContext::with_staging_backend].
#[derive(Debug, Clone)]
pub struct TransactionManager {
    inner: Arc<Mutex<Inner>>,
}

impl TransactionManager {
    /// Create a new manager using the system temp directory.
    ///
    /// The staging directory is `<tmpdir>/rullama-txn-<uuid>` and is created
    /// on construction.
    pub fn new() -> Result<Self> {
        let staging_dir =
            std::env::temp_dir().join(format!("rullama-txn-{}", uuid::Uuid::new_v4()));
        Self::with_dir(staging_dir)
    }

    /// Create a new manager using a specific staging directory.
    ///
    /// The directory is created if it does not already exist.
    pub fn with_dir(staging_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&staging_dir)
            .with_context(|| format!("Failed to create staging dir: {}", staging_dir.display()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                staging_dir,
                staged: HashMap::new(),
            })),
        })
    }
}

impl StagingBackend for TransactionManager {
    fn stage(&self, write: StagedWrite) -> bool {
        let mut inner = self.inner.lock().expect("transaction log lock poisoned");

        // Idempotent: same key staged twice is a no-op (first write wins)
        if inner.staged.contains_key(&write.key) {
            return false;
        }

        // Write content to the staging directory under a key-addressed name
        let safe_name = format!("{}.staged", write.key);
        let staged_path = inner.staging_dir.join(&safe_name);

        if let Err(e) = fs::write(&staged_path, &write.content) {
            tracing::error!(
                key = %write.key,
                path = %staged_path.display(),
                error = %e,
                "TransactionManager: failed to stage write"
            );
            return false;
        }

        tracing::debug!(
            key = %write.key,
            target = %write.target_path.display(),
            "TransactionManager: staged write"
        );

        inner.staged.insert(
            write.key,
            StagedEntry {
                staged_path,
                target_path: write.target_path,
                content: write.content,
            },
        );

        true
    }

    fn commit(&self) -> Result<CommitResult> {
        let mut inner = self.inner.lock().expect("transaction log lock poisoned");
        let mut committed = 0;
        let mut paths = Vec::new();

        for entry in inner.staged.values() {
            // Ensure target parent directory exists
            if let Some(parent) = entry.target_path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create parent dir: {}", parent.display())
                })?;
            }

            // Attempt atomic rename (same filesystem); fall back to copy+delete
            if fs::rename(&entry.staged_path, &entry.target_path).is_err() {
                fs::write(&entry.target_path, &entry.content).with_context(|| {
                    format!(
                        "Failed to commit staged write to {}",
                        entry.target_path.display()
                    )
                })?;
                let _ = fs::remove_file(&entry.staged_path);
            }

            tracing::debug!(target = %entry.target_path.display(), "TransactionManager: committed");
            committed += 1;
            paths.push(entry.target_path.clone());
        }

        inner.staged.clear();
        Ok(CommitResult { committed, paths })
    }

    fn rollback(&self) {
        let mut inner = self.inner.lock().expect("transaction log lock poisoned");
        for entry in inner.staged.values() {
            let _ = fs::remove_file(&entry.staged_path);
        }
        inner.staged.clear();
        tracing::debug!("TransactionManager: rolled back");
    }

    fn pending_count(&self) -> usize {
        self.inner
            .lock()
            .expect("transaction log lock poisoned")
            .staged
            .len()
    }
}

impl Drop for TransactionManager {
    fn drop(&mut self) {
        // If we hold the last Arc reference, auto-rollback any remaining staged files
        if Arc::strong_count(&self.inner) == 1 {
            self.rollback();
            // Best-effort: remove the (now-empty) staging directory
            let inner = self.inner.lock().expect("transaction log lock poisoned");
            let _ = fs::remove_dir(&inner.staging_dir);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rullama_core::{StagedWrite, StagingBackend};
    use std::path::Path;
    use tempfile::TempDir;

    fn make_write(key: &str, path: &Path, content: &str) -> StagedWrite {
        StagedWrite {
            key: key.to_string(),
            target_path: path.to_path_buf(),
            content: content.to_string(),
        }
    }

    #[test]
    fn test_stage_and_commit() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("output.txt");
        let mgr = TransactionManager::new().unwrap();

        let staged = mgr.stage(make_write("k1", &target, "hello world"));
        assert!(staged);
        assert_eq!(mgr.pending_count(), 1);
        assert!(!target.exists(), "Target must not exist before commit");

        let result = mgr.commit().unwrap();
        assert_eq!(result.committed, 1);
        assert!(target.exists());
        assert_eq!(fs::read_to_string(&target).unwrap(), "hello world");
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_rollback_discards_staged_writes() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("discard.txt");
        let mgr = TransactionManager::new().unwrap();

        mgr.stage(make_write("k1", &target, "data"));
        assert_eq!(mgr.pending_count(), 1);

        mgr.rollback();
        assert_eq!(mgr.pending_count(), 0);
        assert!(!target.exists(), "Target must not exist after rollback");
    }

    #[test]
    fn test_duplicate_key_is_idempotent() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("idem.txt");
        let mgr = TransactionManager::new().unwrap();

        let first = mgr.stage(make_write("same-key", &target, "v1"));
        assert!(first);
        let second = mgr.stage(make_write("same-key", &target, "v2"));
        assert!(!second, "Same key must not be staged twice");
        assert_eq!(mgr.pending_count(), 1);

        mgr.commit().unwrap();
        // Only the first content should have been committed
        assert_eq!(fs::read_to_string(&target).unwrap(), "v1");
    }

    #[test]
    fn test_commit_multiple_files() {
        let temp = TempDir::new().unwrap();
        let mgr = TransactionManager::new().unwrap();

        let f1 = temp.path().join("a.txt");
        let f2 = temp.path().join("b.txt");
        mgr.stage(make_write("k-a", &f1, "alpha"));
        mgr.stage(make_write("k-b", &f2, "beta"));
        assert_eq!(mgr.pending_count(), 2);

        let result = mgr.commit().unwrap();
        assert_eq!(result.committed, 2);
        assert_eq!(fs::read_to_string(&f1).unwrap(), "alpha");
        assert_eq!(fs::read_to_string(&f2).unwrap(), "beta");
    }

    #[test]
    fn test_empty_commit_succeeds() {
        let mgr = TransactionManager::new().unwrap();
        let result = mgr.commit().unwrap();
        assert_eq!(result.committed, 0);
        assert!(result.paths.is_empty());
    }

    #[test]
    fn test_commit_creates_parent_directories() {
        let temp = TempDir::new().unwrap();
        let nested = temp.path().join("nested/deep/file.txt");
        let mgr = TransactionManager::new().unwrap();

        mgr.stage(make_write("k-nested", &nested, "content"));
        mgr.commit().unwrap();

        assert!(nested.exists());
        assert_eq!(fs::read_to_string(&nested).unwrap(), "content");
    }

    #[test]
    fn test_commit_clears_queue() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("f.txt");
        let mgr = TransactionManager::new().unwrap();

        mgr.stage(make_write("k", &target, "x"));
        mgr.commit().unwrap();
        assert_eq!(mgr.pending_count(), 0);

        // After commit, new stages are accepted
        mgr.stage(make_write("k2", &temp.path().join("g.txt"), "y"));
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_rollback_clears_queue() {
        let temp = TempDir::new().unwrap();
        let mgr = TransactionManager::new().unwrap();

        mgr.stage(make_write("k", &temp.path().join("f.txt"), "x"));
        mgr.rollback();
        assert_eq!(mgr.pending_count(), 0);
    }
}
