//! Example: Crash Recovery — persistent state, diagnostics, and fix strategies.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example crash_recovery --features self-improve
//! ```

use chrono::Utc;

use brainwires_autonomy::config::CrashRecoveryConfig;
use brainwires_autonomy::self_improve::{
    CrashContext, CycleCheckpoint, FixStrategy, GitState, RecoveryState,
    recovery_state::RecoveryPlanState,
};

fn main() {
    println!("=== Crash Recovery Example ===\n");

    // 1. Show default configuration
    let config = CrashRecoveryConfig::default();
    println!("CrashRecoveryConfig:");
    println!("  max_fix_attempts = {}", config.max_fix_attempts);
    println!("  state_file       = {}", config.state_file);
    println!("  enabled          = {}", config.enabled);
    println!();

    // 2. Create a CycleCheckpoint (saved before each improvement cycle)
    println!("--- Cycle Checkpoint ---");
    let checkpoint = CycleCheckpoint {
        cycle_index: 3,
        total_cycles: 10,
        task_id: Some("task-clippy-42".to_string()),
        strategy: Some("clippy".to_string()),
        git_state: GitState {
            branch: "self-improve/clippy-batch-1".to_string(),
            last_commit: "abc123def456".to_string(),
            dirty_files: vec!["src/lib.rs".to_string()],
            has_uncommitted_changes: true,
        },
        timestamp: Utc::now(),
    };
    println!(
        "  Cycle {}/{} — strategy: {:?}",
        checkpoint.cycle_index, checkpoint.total_cycles, checkpoint.strategy
    );
    println!("  Branch: {}", checkpoint.git_state.branch);
    println!("  Dirty files: {:?}", checkpoint.git_state.dirty_files);
    println!();

    // 3. Simulate a crash context
    println!("--- Crash Context ---");
    let crash = CrashContext {
        crash_time: Utc::now(),
        exit_code: Some(101),
        signal: None,
        stderr_tail: "thread 'main' panicked at 'index out of bounds: the len is 0 but the index is 5'\nnote: run with `RUST_BACKTRACE=1` for a backtrace".to_string(),
        last_cycle_index: 3,
        last_task_id: Some("task-clippy-42".to_string()),
        last_strategy: Some("clippy".to_string()),
        working_directory: "/home/user/project".to_string(),
        git_state: checkpoint.git_state.clone(),
    };
    println!("  Exit code: {:?}", crash.exit_code);
    println!("  Cycle: {}", crash.last_cycle_index);
    println!("  Strategy: {:?}", crash.last_strategy);
    println!(
        "  Stderr: {}",
        crash.stderr_tail.lines().next().unwrap_or("")
    );
    println!();

    // 4. Create a RecoveryState (persisted to disk)
    println!("--- Recovery State ---");
    let state = RecoveryState {
        version: 1,
        crash_id: "crash-20260315-001".to_string(),
        crash_context: crash,
        fix_attempts: 0,
        max_fix_attempts: 3,
        recovery_plan: Some(RecoveryPlanState {
            root_cause: "Index out of bounds in clippy fix application".to_string(),
            fix_strategy: "skip_task".to_string(),
            files_to_fix: vec!["src/lib.rs".to_string()],
            rollback_needed: true,
            resume_from_cycle: 4,
        }),
    };

    println!("  Crash ID: {}", state.crash_id);
    println!(
        "  Fix attempts: {}/{}",
        state.fix_attempts, state.max_fix_attempts
    );
    println!("  Is meta-crash: {}", state.is_meta_crash());

    if let Some(plan) = &state.recovery_plan {
        println!("  Root cause: {}", plan.root_cause);
        println!("  Fix strategy: {}", plan.fix_strategy);
        println!("  Rollback needed: {}", plan.rollback_needed);
        println!("  Resume from cycle: {}", plan.resume_from_cycle);
    }
    println!();

    // 5. Demonstrate FixStrategy parsing
    println!("--- Fix Strategies ---");
    let strategies = [
        "revert_last_commit",
        "skip_task",
        "rollback_to_checkpoint",
        "apply_patch:some_data",
        "unknown_thing",
    ];
    for label in &strategies {
        let strategy = FixStrategy::from_label(label);
        println!(
            "  \"{label}\" -> {:?} (label: \"{}\")",
            strategy,
            strategy.label()
        );
    }
    println!();

    // 6. JSON serialization roundtrip
    println!("--- Serialization ---");
    let json = serde_json::to_string_pretty(&state).unwrap();
    println!("  JSON size: {} bytes", json.len());
    let deserialized: RecoveryState = serde_json::from_str(&json).unwrap();
    println!("  Roundtrip OK: crash_id={}", deserialized.crash_id);

    println!("\nDone.");
}
