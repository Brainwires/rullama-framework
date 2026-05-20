//! Example: Optimistic concurrency with conflict detection and resolution
//!
//! Demonstrates how `OptimisticController` allows agents to proceed without
//! upfront locks, detecting conflicts at commit time and resolving them
//! via configurable strategies (FirstWriterWins, LastWriterWins, Retry).
//!
//! Run: cargo run -p brainwires-agent --example optimistic_concurrency

use anyhow::Result;

use brainwires_agent::optimistic::{CommitResult, OptimisticController, ResolutionStrategy};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Optimistic Concurrency Demo ===\n");

    // ── 1. Create controller (default: FirstWriterWins) ────────────────────

    let controller = OptimisticController::new();

    // ── 2. Agent-A begins an optimistic operation ──────────────────────────

    let token_a = controller.begin_optimistic("agent-a", "src/main.rs").await;
    println!(
        "Agent-A began optimistic operation on src/main.rs (base_version={})",
        token_a.base_version
    );

    // ── 3. Agent-B begins on the same resource ─────────────────────────────

    let token_b = controller.begin_optimistic("agent-b", "src/main.rs").await;
    println!(
        "Agent-B began optimistic operation on src/main.rs (base_version={})",
        token_b.base_version
    );

    // ── 4. Agent-A commits successfully ────────────────────────────────────

    let version_a = controller
        .commit_optimistic(token_a, "hash-after-a")
        .await
        .expect("Agent-A should commit without conflict");
    println!(
        "\nAgent-A committed successfully (new version={})",
        version_a
    );

    // ── 5. Agent-B tries to commit — conflict detected ─────────────────────

    let conflict = controller
        .commit_optimistic(token_b, "hash-after-b")
        .await
        .expect_err("Agent-B should hit a conflict");

    println!("\nAgent-B conflict detected:");
    println!("  Resource:         {}", conflict.resource_id);
    println!("  Conflicting agent: {}", conflict.conflicting_agent);
    println!("  Expected version: {}", conflict.expected_version);
    println!("  Actual version:   {}", conflict.actual_version);
    println!("  Holder agent:     {}", conflict.holder_agent);

    // ── 6. Show conflict details ───────────────────────────────────────────

    println!("  Version diff:     {}", conflict.version_diff());

    // ── 7. Demonstrate commit_or_resolve with different strategies ──────────

    println!("\n--- LastWriterWins strategy ---");

    let lww_controller =
        OptimisticController::with_default_strategy(ResolutionStrategy::LastWriterWins);

    let tok1 = lww_controller
        .begin_optimistic("agent-x", "config.json")
        .await;
    let tok2 = lww_controller
        .begin_optimistic("agent-y", "config.json")
        .await;

    // Agent-X commits first
    lww_controller
        .commit_optimistic(tok1, "hash-x")
        .await
        .expect("Agent-X commits first");

    // Agent-Y uses commit_or_resolve — LastWriterWins lets it succeed
    let result = lww_controller
        .commit_or_resolve(tok2, "hash-y", None)
        .await
        .expect("commit_or_resolve should not error");

    match &result {
        CommitResult::Committed { version } => {
            println!("Agent-Y committed (version={}) — last writer won", version);
        }
        other => {
            println!("Unexpected result: {:?}", other);
        }
    }
    println!("  is_success: {}", result.is_success());

    // ── Retry strategy ─────────────────────────────────────────────────────

    println!("\n--- Retry strategy ---");

    let retry_controller =
        OptimisticController::with_default_strategy(ResolutionStrategy::Retry { max_attempts: 3 });

    let tok_r1 = retry_controller
        .begin_optimistic("agent-r1", "data.db")
        .await;
    let tok_r2 = retry_controller
        .begin_optimistic("agent-r2", "data.db")
        .await;

    retry_controller
        .commit_optimistic(tok_r1, "hash-r1")
        .await
        .expect("Agent-R1 commits");

    let retry_result = retry_controller
        .commit_or_resolve(tok_r2, "hash-r2", None)
        .await
        .expect("commit_or_resolve should not error");

    match &retry_result {
        CommitResult::RetryNeeded { current_version } => {
            println!(
                "Agent-R2 told to retry (current_version={})",
                current_version
            );
        }
        other => {
            println!("Result: {:?}", other);
        }
    }

    // ── Stats ──────────────────────────────────────────────────────────────

    println!("\n--- Controller stats ---");

    let stats = lww_controller.get_stats().await;
    println!("  Total resources tracked: {}", stats.total_resources);
    println!("  Total conflicts:         {}", stats.total_conflicts);
    println!("  Resolved by retry:       {}", stats.resolved_by_retry);
    println!("  Escalated:               {}", stats.escalated);

    println!("\nOptimistic concurrency demo complete.");
    Ok(())
}
