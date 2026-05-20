//! Integration tests for FileLockManager multi-agent coordination.
//!
//! Tests cross-agent lock interactions, wait-based acquisition, deadlock
//! detection, and lock expiry cleanup across concurrent agents.

use brainwires_agent::file_locks::{FileLockManager, LockType};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Multi-agent read/write lock interactions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multiple_readers_then_exclusive_writer() {
    let mgr = Arc::new(FileLockManager::new());

    // Three agents acquire read locks
    let _r1 = mgr
        .acquire_lock("reader-1", "/src/lib.rs", LockType::Read)
        .await
        .unwrap();
    let _r2 = mgr
        .acquire_lock("reader-2", "/src/lib.rs", LockType::Read)
        .await
        .unwrap();
    let _r3 = mgr
        .acquire_lock("reader-3", "/src/lib.rs", LockType::Read)
        .await
        .unwrap();

    // Writer should be blocked while readers hold locks
    let write_result = mgr
        .acquire_lock("writer-1", "/src/lib.rs", LockType::Write)
        .await;
    assert!(write_result.is_err());

    // Drop all readers
    drop(_r1);
    drop(_r2);
    drop(_r3);

    // Give the async drop tasks time to execute
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Now writer should succeed
    let _w = mgr
        .acquire_lock("writer-1", "/src/lib.rs", LockType::Write)
        .await
        .unwrap();

    assert!(mgr.is_locked_by("/src/lib.rs", "writer-1").await);
}

#[tokio::test]
async fn agents_lock_different_files_independently() {
    let mgr = Arc::new(FileLockManager::new());

    let _w1 = mgr
        .acquire_lock("agent-1", "/src/main.rs", LockType::Write)
        .await
        .unwrap();
    let _w2 = mgr
        .acquire_lock("agent-2", "/src/utils.rs", LockType::Write)
        .await
        .unwrap();

    // Both should succeed -- different files
    assert!(mgr.is_locked_by("/src/main.rs", "agent-1").await);
    assert!(mgr.is_locked_by("/src/utils.rs", "agent-2").await);

    // Agent 1 cannot access agent 2's file
    assert!(
        !mgr.can_acquire("/src/utils.rs", "agent-1", LockType::Write)
            .await
    );
    assert!(
        !mgr.can_acquire("/src/main.rs", "agent-2", LockType::Write)
            .await
    );
}

// ---------------------------------------------------------------------------
// Wait-based acquisition
// ---------------------------------------------------------------------------

#[tokio::test]
async fn acquire_with_wait_succeeds_after_lock_released() {
    let mgr = Arc::new(FileLockManager::new());

    // Agent-1 holds a write lock
    let guard = mgr
        .acquire_lock("agent-1", "/shared.rs", LockType::Write)
        .await
        .unwrap();

    let mgr_clone = Arc::clone(&mgr);

    // Agent-1 releases after a delay
    let releaser = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        drop(guard);
        // Allow async drop to process
        tokio::time::sleep(Duration::from_millis(20)).await;
        mgr_clone.cleanup_expired().await;
    });

    // Agent-2 waits for the lock
    let result = mgr
        .acquire_with_wait(
            "agent-2",
            "/shared.rs",
            LockType::Write,
            Duration::from_millis(500),
        )
        .await;

    releaser.await.unwrap();
    assert!(
        result.is_ok(),
        "Agent-2 should acquire after agent-1 releases"
    );
}

#[tokio::test]
async fn acquire_with_wait_times_out() {
    let mgr = Arc::new(FileLockManager::new());

    // Agent-1 holds a write lock indefinitely
    let _guard = mgr
        .acquire_lock("agent-1", "/busy.rs", LockType::Write)
        .await
        .unwrap();

    // Agent-2 tries to wait but times out
    let result = mgr
        .acquire_with_wait(
            "agent-2",
            "/busy.rs",
            LockType::Write,
            Duration::from_millis(150),
        )
        .await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("timeout"),
        "Error should mention timeout: {}",
        err_msg
    );
}

// ---------------------------------------------------------------------------
// Deadlock prevention via timeout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_agents_contending_for_each_others_files() {
    // Two agents each hold a lock the other wants.
    // Without deadlock detection, both would block forever.
    // With acquire_with_wait timeout, at least one side will time out.

    let mgr = Arc::new(FileLockManager::new());

    // Agent-1 holds file1
    let _g1 = mgr
        .acquire_lock("agent-1", "/file1.rs", LockType::Write)
        .await
        .unwrap();
    // Agent-2 holds file2
    let _g2 = mgr
        .acquire_lock("agent-2", "/file2.rs", LockType::Write)
        .await
        .unwrap();

    let mgr_clone = Arc::clone(&mgr);

    // Agent-1 tries to get file2 (held by agent-2), with a timeout
    let handle_a = tokio::spawn(async move {
        mgr_clone
            .acquire_with_wait(
                "agent-1",
                "/file2.rs",
                LockType::Write,
                Duration::from_millis(200),
            )
            .await
    });

    // Agent-2 tries to get file1 (held by agent-1), with a timeout
    let handle_b = {
        let mgr_clone2 = Arc::clone(&mgr);
        tokio::spawn(async move {
            mgr_clone2
                .acquire_with_wait(
                    "agent-2",
                    "/file1.rs",
                    LockType::Write,
                    Duration::from_millis(200),
                )
                .await
        })
    };

    let (res_a, res_b) = tokio::join!(handle_a, handle_b);

    // At least one (likely both) should fail due to timeout or deadlock detection
    let a_failed = res_a.unwrap().is_err();
    let b_failed = res_b.unwrap().is_err();
    assert!(
        a_failed || b_failed,
        "At least one agent should fail when both contend for each other's locks"
    );
}

// ---------------------------------------------------------------------------
// Lock cleanup and release_all
// ---------------------------------------------------------------------------

#[tokio::test]
async fn release_all_clears_multiple_lock_types() {
    let mgr = Arc::new(FileLockManager::new());

    // Agent holds various locks
    let g1 = mgr
        .acquire_lock("busy-agent", "/a.rs", LockType::Write)
        .await
        .unwrap();
    let g2 = mgr
        .acquire_lock("busy-agent", "/b.rs", LockType::Read)
        .await
        .unwrap();
    let g3 = mgr
        .acquire_lock("busy-agent", "/c.rs", LockType::Write)
        .await
        .unwrap();

    // Forget guards so release_all does the work
    std::mem::forget(g1);
    std::mem::forget(g2);
    std::mem::forget(g3);

    let released = mgr.release_all_locks("busy-agent").await;
    assert_eq!(released, 3);

    // All files should now be acquirable
    assert!(mgr.can_acquire("/a.rs", "other", LockType::Write).await);
    assert!(mgr.can_acquire("/b.rs", "other", LockType::Write).await);
    assert!(mgr.can_acquire("/c.rs", "other", LockType::Write).await);
}

#[tokio::test]
async fn expired_locks_cleaned_up_automatically() {
    let mgr = Arc::new(FileLockManager::new());

    // Acquire with very short timeout
    let guard = mgr
        .acquire_lock_with_timeout(
            "temp-agent",
            "/temp.rs",
            LockType::Write,
            Some(Duration::from_millis(5)),
        )
        .await
        .unwrap();
    std::mem::forget(guard);

    // Wait for expiry
    tokio::time::sleep(Duration::from_millis(20)).await;

    let cleaned = mgr.cleanup_expired().await;
    assert_eq!(cleaned, 1);

    // Another agent can now acquire
    let result = mgr
        .acquire_lock("new-agent", "/temp.rs", LockType::Write)
        .await;
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// Stats and introspection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stats_reflect_current_lock_state() {
    let mgr = Arc::new(FileLockManager::new());

    let _w1 = mgr
        .acquire_lock("a1", "/x.rs", LockType::Write)
        .await
        .unwrap();
    let _r1 = mgr
        .acquire_lock("a2", "/y.rs", LockType::Read)
        .await
        .unwrap();
    let _r2 = mgr
        .acquire_lock("a3", "/y.rs", LockType::Read)
        .await
        .unwrap();

    let stats = mgr.stats().await;
    assert_eq!(stats.total_files, 2);
    assert_eq!(stats.total_write_locks, 1);
    assert_eq!(stats.total_read_locks, 2);
}

#[tokio::test]
async fn locks_for_agent_returns_only_that_agents_locks() {
    let mgr = Arc::new(FileLockManager::new());

    let _w = mgr
        .acquire_lock("agent-x", "/foo.rs", LockType::Write)
        .await
        .unwrap();
    let _r = mgr
        .acquire_lock("agent-x", "/bar.rs", LockType::Read)
        .await
        .unwrap();
    let _other = mgr
        .acquire_lock("agent-y", "/baz.rs", LockType::Write)
        .await
        .unwrap();

    let x_locks = mgr.locks_for_agent("agent-x").await;
    assert_eq!(x_locks.len(), 2);

    let y_locks = mgr.locks_for_agent("agent-y").await;
    assert_eq!(y_locks.len(), 1);
}
