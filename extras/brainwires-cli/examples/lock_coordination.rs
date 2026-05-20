//! Example: LockStore — cross-process lock coordination
//!
//! Demonstrates using `LockStore` for SQLite-backed distributed locking:
//! acquiring file and build locks, checking lock status, handling conflicts,
//! inspecting statistics, and cleaning up stale/expired locks.
//!
//! Run: cargo run -p brainwires-cli --example lock_coordination

use std::time::Duration;

use anyhow::Result;
use brainwires_stores::LockStore;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create a lock store in a temporary directory
    let tmp_dir = tempfile::tempdir()?;
    let db_path = tmp_dir.path().join("locks.db");
    let store = LockStore::new_with_path(&db_path).await?;
    println!("LockStore created at {:?}\n", db_path);

    // 2. Acquire a file-write lock
    let acquired = store
        .try_acquire("file_write", "/src/main.rs", "agent-alpha", None)
        .await?;
    println!(
        "Acquire file_write on /src/main.rs (agent-alpha): {}",
        if acquired { "OK" } else { "BLOCKED" },
    );

    // 3. Check lock status
    if let Some(lock) = store.is_locked("file_write", "/src/main.rs").await? {
        println!(
            "  Held by: {} (pid={}, host={})",
            lock.agent_id, lock.process_id, lock.hostname,
        );
    }
    println!();

    // 4. Demonstrate idempotent re-acquisition (same agent, same process)
    let reacquired = store
        .try_acquire("file_write", "/src/main.rs", "agent-alpha", None)
        .await?;
    println!(
        "Re-acquire same lock (idempotent): {}",
        if reacquired { "OK" } else { "BLOCKED" },
    );

    // 5. Demonstrate conflict — a different agent cannot take the same lock
    let conflict = store
        .try_acquire("file_write", "/src/main.rs", "agent-beta", None)
        .await?;
    println!(
        "Acquire same resource as agent-beta: {}",
        if conflict { "OK" } else { "BLOCKED (expected)" },
    );
    println!();

    // 6. Acquire additional locks for different resources
    store
        .try_acquire("file_read", "/src/lib.rs", "agent-alpha", None)
        .await?;
    store
        .try_acquire("build", "/project/root", "agent-alpha", None)
        .await?;
    println!("Acquired file_read on /src/lib.rs and build on /project/root");

    // 7. List all active locks
    let locks = store.list_locks().await?;
    println!("\nActive locks ({}):", locks.len());
    for lock in &locks {
        println!(
            "  [{}] {} -> {} (agent={})",
            lock.lock_type, lock.resource_path, lock.lock_id, lock.agent_id,
        );
    }

    // 8. View lock statistics
    let stats = store.stats().await?;
    println!("\nLock statistics:");
    println!("  Total:      {}", stats.total_locks);
    println!("  File read:  {}", stats.file_read_locks);
    println!("  File write: {}", stats.file_write_locks);
    println!("  Build:      {}", stats.build_locks);
    println!("  Stale:      {}", stats.stale_locks);
    println!();

    // 9. Acquire a lock with a short timeout (will expire quickly)
    store
        .try_acquire(
            "test",
            "/project/root",
            "agent-alpha",
            Some(Duration::from_millis(50)),
        )
        .await?;
    println!("Acquired test lock with 50ms timeout");

    // Wait for it to expire
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 10. Clean up expired and stale locks
    let cleaned = store.cleanup_stale().await?;
    println!("Cleaned up {} stale/expired lock(s)", cleaned);

    // 11. Release a specific lock
    let released = store
        .release("file_write", "/src/main.rs", "agent-alpha")
        .await?;
    println!(
        "\nReleased file_write on /src/main.rs: {}",
        if released { "OK" } else { "NOT FOUND" },
    );

    // 12. Release all locks for an agent
    let count = store.release_all_for_agent("agent-alpha").await?;
    println!("Released {} remaining lock(s) for agent-alpha", count);

    // Verify everything is cleaned up
    let remaining = store.list_locks().await?;
    println!("Remaining locks: {}", remaining.len());

    println!("\nDone.");
    Ok(())
}
