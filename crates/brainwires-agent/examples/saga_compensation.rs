//! Example: Saga compensating transactions with rollback on failure
//!
//! Shows how to use `SagaExecutor` with custom `CompensableOperation` impls to
//! execute a sequence of operations where a mid-sequence failure triggers
//! automatic compensation (rollback) of all previously completed steps.
//!
//! Run: cargo run -p brainwires-agent --example saga_compensation

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use brainwires_agent::saga::{
    CompensableOperation, OperationResult, SagaExecutor, SagaOperationType, SagaStatus,
};

// ── Mock Operations ──────────────────────────────────────────────────────

/// A file-write operation that always succeeds.
struct MockFileWrite;

#[async_trait]
impl CompensableOperation for MockFileWrite {
    async fn execute(&self) -> Result<OperationResult> {
        println!("    [execute] Writing config.toml ...");
        Ok(
            OperationResult::success("mock-file-write")
                .with_output("wrote 42 bytes to config.toml"),
        )
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        println!("    [compensate] Restoring original config.toml");
        Ok(())
    }

    fn description(&self) -> String {
        "Write file: config.toml".to_string()
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::FileWrite
    }
}

/// A git-stage operation that always succeeds.
struct MockGitStage;

#[async_trait]
impl CompensableOperation for MockGitStage {
    async fn execute(&self) -> Result<OperationResult> {
        println!("    [execute] Staging config.toml ...");
        Ok(OperationResult::success("mock-git-stage").with_output("staged 1 file"))
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        println!("    [compensate] Unstaging config.toml (git reset HEAD)");
        Ok(())
    }

    fn description(&self) -> String {
        "Git stage: config.toml".to_string()
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::GitStage
    }
}

/// A build operation that always fails.
struct MockBuild;

#[async_trait]
impl CompensableOperation for MockBuild {
    async fn execute(&self) -> Result<OperationResult> {
        println!("    [execute] Running cargo build ... FAILED");
        Ok(OperationResult::failure("mock-build"))
    }

    async fn compensate(&self, _result: &OperationResult) -> Result<()> {
        // Build is non-compensable, but we implement it anyway (will be skipped).
        Ok(())
    }

    fn description(&self) -> String {
        "Build project".to_string()
    }

    fn operation_type(&self) -> SagaOperationType {
        SagaOperationType::Build
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Saga Compensating Transactions ===\n");

    // 1. Create a saga executor
    let saga = SagaExecutor::new("agent-1", "edit-stage-build pipeline");
    println!("Saga created: {}", saga.saga_id);
    println!("  Agent:       {}", saga.agent_id);
    println!("  Description: {}", saga.description);
    println!("  Status:      {:?}\n", saga.status().await);

    // 2. Execute step 1: file write (succeeds)
    println!("Step 1: File write");
    let result1 = saga.execute_step(Arc::new(MockFileWrite)).await?;
    println!("  Success: {}", result1.success);
    println!("  Operations so far: {}\n", saga.operation_count().await);

    // 3. Execute step 2: git stage (succeeds)
    println!("Step 2: Git stage");
    let result2 = saga.execute_step(Arc::new(MockGitStage)).await?;
    println!("  Success: {}", result2.success);
    println!("  Operations so far: {}\n", saga.operation_count().await);

    // 4. Execute step 3: build (fails)
    println!("Step 3: Build");
    let result3 = saga.execute_step(Arc::new(MockBuild)).await?;
    println!("  Success: {}", result3.success);
    println!("  Status:  {:?}\n", saga.status().await);

    // 5. The build returned a failure result — trigger compensation
    if !result3.success {
        println!("Build failed! Compensating all completed operations...\n");
        let report = saga.compensate_all().await?;

        // 6. Print the compensation report
        println!("\n--- Compensation Report ---");
        println!("  Saga: {}", report.saga_id);
        for entry in &report.operations {
            let icon = match entry.status {
                brainwires_agent::saga::CompensationOutcome::Success => "OK",
                brainwires_agent::saga::CompensationOutcome::Failed => "FAIL",
                brainwires_agent::saga::CompensationOutcome::Skipped => "SKIP",
            };
            println!("  [{icon}] {}", entry.description);
            if let Some(err) = &entry.error {
                println!("        reason: {err}");
            }
        }
        println!("\n  Summary:        {}", report.summary());
        println!("  All successful: {}", report.all_successful());
        println!("  Final status:   {:?}", saga.status().await);
    }

    assert_eq!(saga.status().await, SagaStatus::Compensated);
    println!("\nSaga compensation demo complete.");
    Ok(())
}
