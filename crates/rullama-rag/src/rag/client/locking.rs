//! Index locking methods for [`RagClient`].
//!
//! Provides the two-layer locking strategy (filesystem + in-process broadcast)
//! used to coordinate concurrent indexing operations.

use super::{FsLockGuard, IndexLockGuard, IndexLockResult, IndexingOperation, RagClient};
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::broadcast;

impl RagClient {
    /// Try to acquire an indexing lock for a given path
    ///
    /// This uses a two-layer locking strategy:
    /// 1. Filesystem lock (flock) for cross-process coordination
    /// 2. In-memory lock for broadcasting results to waiters in the same process
    ///
    /// Returns either:
    /// - `IndexLockResult::Acquired(guard)` if we should perform the indexing
    /// - `IndexLockResult::WaitForResult(receiver)` if another task in THIS process is indexing
    /// - `IndexLockResult::WaitForFilesystemLock(path)` if ANOTHER PROCESS is indexing
    ///
    /// The lock is automatically released when the returned guard is dropped.
    pub(crate) async fn try_acquire_index_lock(&self, path: &str) -> Result<IndexLockResult> {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        // Normalize the path to ensure consistent locking across different path formats
        let normalized_path = Self::normalize_path(path)?;

        // STEP 1: Try to acquire filesystem lock first (cross-process coordination)
        // This must happen BEFORE checking in-memory state to prevent race conditions
        let fs_lock = {
            let path_clone = normalized_path.clone();
            tokio::task::spawn_blocking(move || FsLockGuard::try_acquire(&path_clone))
                .await
                .context("Filesystem lock task panicked")??
        };

        // If we couldn't get the filesystem lock, another PROCESS is indexing
        let fs_lock = match fs_lock {
            Some(lock) => lock,
            None => {
                tracing::info!(
                    "Another process is indexing {} - returning WaitForFilesystemLock",
                    normalized_path
                );
                return Ok(IndexLockResult::WaitForFilesystemLock(normalized_path));
            }
        };

        // STEP 2: We have the filesystem lock, now check in-memory state
        // This handles the case where another task in THIS process is indexing

        // Acquire write lock on the ops map
        let mut ops = self.indexing_ops.write().await;

        // Check if an operation is already in progress for this path (in this process)
        if let Some(existing_op) = ops.get(&normalized_path) {
            // Check if the operation is stale (timed out or crashed)
            if existing_op.is_stale() {
                tracing::warn!(
                    "Removing stale indexing lock for {} (operation timed out after {:?})",
                    normalized_path,
                    existing_op.started_at.elapsed()
                );
                ops.remove(&normalized_path);
            } else if existing_op.active.load(Ordering::Acquire) {
                // Operation is still active and not stale, subscribe to receive the result
                // Note: We drop the filesystem lock here since we won't be indexing
                drop(fs_lock);
                let receiver = existing_op.result_tx.subscribe();
                tracing::info!(
                    "Indexing already in progress in this process for {} (started {:?} ago), waiting for result",
                    normalized_path,
                    existing_op.started_at.elapsed()
                );
                return Ok(IndexLockResult::WaitForResult(receiver));
            } else {
                // Operation completed but cleanup hasn't happened yet
                tracing::debug!(
                    "Removing completed indexing lock for {} (cleanup pending)",
                    normalized_path
                );
                ops.remove(&normalized_path);
            }
        }

        // STEP 3: We have both locks, register the operation

        // Create a new broadcast channel for this operation
        // Capacity of 1 is enough since we only send one result
        let (result_tx, _) = broadcast::channel(1);

        // Create the active flag - starts as true (active)
        let active_flag = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // Register this operation with timestamp
        ops.insert(
            normalized_path.clone(),
            IndexingOperation {
                result_tx: result_tx.clone(),
                active: active_flag.clone(),
                started_at: Instant::now(),
            },
        );

        // Drop the write lock on the map
        drop(ops);

        Ok(IndexLockResult::Acquired(IndexLockGuard::new(
            normalized_path,
            self.indexing_ops.clone(),
            result_tx,
            active_flag,
            fs_lock,
        )))
    }
}
