//! Example: Three-State Model for comprehensive state tracking
//!
//! Demonstrates how `ThreeStateModel` separates application, operation, and
//! dependency state to enable conflict detection, operation validation, and
//! state snapshots for rollback support.
//!
//! Run: cargo run -p brainwires-agent --example three_state_model

use std::path::PathBuf;

use anyhow::Result;

use brainwires_agent::state_model::{
    ApplicationChange, OperationLog, StateChange, StateModelProposedOperation, ThreeStateModel,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Three-State Model Demo ===\n");

    // ── 1. Create model ────────────────────────────────────────────────────

    let model = ThreeStateModel::new();

    // ── 2. Register resources in application state ─────────────────────────

    println!("--- Updating application state ---");

    model
        .application_state
        .update_file(PathBuf::from("src/main.rs"), "abc123".to_string())
        .await;
    model
        .application_state
        .update_file(PathBuf::from("src/lib.rs"), "def456".to_string())
        .await;
    model
        .application_state
        .mark_resource_exists("build-cache")
        .await;

    let files = model.application_state.get_all_files().await;
    println!("Tracked files: {}", files.len());
    for (path, status) in &files {
        println!(
            "  {} (hash={}, dirty={})",
            path.display(),
            status.content_hash,
            status.dirty,
        );
    }

    // ── 3. Start and complete an operation ──────────────────────────────────

    println!("\n--- Operation lifecycle ---");

    let op_id = model.operation_state.generate_id().await;
    let log = OperationLog::new(
        op_id.clone(),
        "agent-1".to_string(),
        "build".to_string(),
        serde_json::json!({ "target": "release" }),
    )
    .with_resources(
        vec!["src/main.rs".to_string()],
        vec!["target/release/app".to_string()],
    );

    model.operation_state.start_operation(log).await;
    println!("Started operation: {op_id}");

    let active = model.operation_state.get_active_operations().await;
    println!("Active operations: {}", active.len());

    model
        .operation_state
        .complete_operation(&op_id, true, None, None)
        .await;

    let completed = model.operation_state.get_operation(&op_id).await.unwrap();
    println!("Operation {} status: {:?}", op_id, completed.status);

    // ── 4. Validate a proposed operation — should pass ──────────────────────

    println!("\n--- Operation validation (no conflict) ---");

    let proposed = StateModelProposedOperation {
        agent_id: "agent-2".to_string(),
        operation_type: "test".to_string(),
        resources_needed: vec!["src/lib.rs".to_string()],
        resources_produced: vec!["test-report.xml".to_string()],
    };

    let result = model.validate_operation(&proposed).await;
    println!(
        "Proposed test on src/lib.rs: valid={}, errors={}, warnings={}",
        result.valid,
        result.errors.len(),
        result.warnings.len(),
    );

    // ── 5. Show a conflicting scenario ──────────────────────────────────────

    println!("\n--- Operation validation (conflict) ---");

    // Start a long-running operation that holds src/main.rs
    let conflict_op_id = model.operation_state.generate_id().await;
    let conflict_log = OperationLog::new(
        conflict_op_id.clone(),
        "agent-3".to_string(),
        "refactor".to_string(),
        serde_json::json!({}),
    )
    .with_resources(vec!["src/main.rs".to_string()], vec![]);
    model.operation_state.start_operation(conflict_log).await;

    // Another agent tries to use the same resource
    let conflicting = StateModelProposedOperation {
        agent_id: "agent-4".to_string(),
        operation_type: "format".to_string(),
        resources_needed: vec!["src/main.rs".to_string()],
        resources_produced: vec![],
    };

    let conflict_result = model.validate_operation(&conflicting).await;
    println!(
        "Proposed format on src/main.rs while refactor is running: valid={}",
        conflict_result.valid
    );
    for err in &conflict_result.errors {
        println!("  Error: {err}");
    }

    // Clean up the running operation
    model
        .operation_state
        .complete_operation(&conflict_op_id, true, None, None)
        .await;

    // ── 6. Record a state change and take snapshot ──────────────────────────

    println!("\n--- State change + snapshot ---");

    let change = StateChange {
        operation_id: "op-change-1".to_string(),
        application_changes: vec![
            ApplicationChange::FileModified {
                path: PathBuf::from("src/main.rs"),
                new_hash: "updated-hash-789".to_string(),
            },
            ApplicationChange::ResourceCreated {
                resource_id: "deploy-artifact".to_string(),
            },
        ],
        new_dependencies: vec![],
    };
    model.record_state_change(change).await;

    let snapshot = model.snapshot().await;
    println!("Snapshot summary:");
    println!("  Files tracked:      {}", snapshot.files.len());
    println!("  Resource locks:     {}", snapshot.locks.len());
    println!("  Active operations:  {}", snapshot.active_operations.len());
    println!(
        "  Git branch:         {}",
        if snapshot.git_state.current_branch.is_empty() {
            "(none)"
        } else {
            &snapshot.git_state.current_branch
        }
    );

    // Verify the updated hash
    let main_rs = snapshot
        .files
        .get(&PathBuf::from("src/main.rs"))
        .expect("src/main.rs should be tracked");
    println!(
        "  src/main.rs hash:   {} (dirty={})",
        main_rs.content_hash, main_rs.dirty
    );

    println!("\nThree-state model demo complete.");
    Ok(())
}
