//! Integration tests for coordination patterns.
//!
//! Tests the ContractNet, Saga, and OptimisticConcurrency modules working
//! together and in realistic multi-step scenarios.

use brainwires_agent::contract_net::{
    BidEvaluationStrategy, ContractNetManager, ContractParticipant, TaskAnnouncement, TaskBid,
    TaskRequirements, TaskStatus,
};
use brainwires_agent::optimistic::{CommitResult, OptimisticController, ResolutionStrategy};
use brainwires_agent::saga::{NoOpCompensation, SagaExecutor, SagaOperationType, SagaStatus};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ===========================================================================
// Contract-Net: full auction lifecycle with multiple participants
// ===========================================================================

#[tokio::test]
async fn contract_net_full_auction_lifecycle() {
    let manager = ContractNetManager::new();

    // Create participants with different capabilities
    let mut participant_rust =
        ContractParticipant::new("rust-agent", vec!["rust".to_string(), "cargo".to_string()])
            .with_max_concurrent(3);
    let mut participant_ts = ContractParticipant::new(
        "ts-agent",
        vec!["typescript".to_string(), "npm".to_string()],
    )
    .with_max_concurrent(2);
    let mut participant_full = ContractParticipant::new(
        "fullstack-agent",
        vec![
            "rust".to_string(),
            "typescript".to_string(),
            "cargo".to_string(),
        ],
    )
    .with_max_concurrent(2);

    // Connect participants
    participant_rust.connect(&manager);
    participant_ts.connect(&manager);
    participant_full.connect(&manager);

    // Announce a Rust task
    let announcement = TaskAnnouncement::new(
        "build-cache",
        "Implement LRU cache in Rust",
        "orchestrator",
        Instant::now() + Duration::from_secs(30),
    )
    .with_requirements(
        TaskRequirements::new()
            .with_capabilities(vec!["rust".to_string(), "cargo".to_string()])
            .with_complexity(7),
    );

    let task_id = manager.announce_task(announcement.clone()).await;

    // Check which participants should bid
    assert!(participant_rust.should_bid(&announcement).await);
    assert!(!participant_ts.should_bid(&announcement).await); // lacks rust
    assert!(participant_full.should_bid(&announcement).await);

    // Generate and submit bids
    let bid_rust = participant_rust.generate_bid(&announcement).await;
    let bid_full = participant_full.generate_bid(&announcement).await;

    assert_eq!(bid_rust.capability_score, 1.0); // has all required
    assert!(bid_full.capability_score > 0.0); // partial match (2/2 matched)

    manager.receive_bid(bid_rust).await.unwrap();
    manager.receive_bid(bid_full).await.unwrap();

    // Award
    let winner = manager.award_task(&task_id).await.unwrap();
    // Both have score 1.0 for capabilities, but load differs; just check we got a winner
    assert!(winner == "rust-agent" || winner == "fullstack-agent");

    // Accept
    manager.accept_award(&task_id, &winner).await.unwrap();
    assert_eq!(
        manager.get_task_status(&task_id).await,
        Some(TaskStatus::InProgress)
    );

    // Complete
    manager
        .complete_task(&task_id, &winner, true, Some("Cache implemented".into()))
        .await
        .unwrap();
    assert_eq!(
        manager.get_task_status(&task_id).await,
        Some(TaskStatus::Completed)
    );
}

#[tokio::test]
async fn contract_net_no_bids_results_in_no_award() {
    let manager = ContractNetManager::new();

    let announcement = TaskAnnouncement::new(
        "impossible-task",
        "Requires quantum computing",
        "orchestrator",
        Instant::now() + Duration::from_secs(30),
    );
    manager.announce_task(announcement).await;

    // No bids submitted
    let winner = manager.award_task("impossible-task").await;
    assert!(winner.is_none());
}

#[tokio::test]
async fn contract_net_load_balancing_strategy() {
    let manager = ContractNetManager::with_strategy(BidEvaluationStrategy::LoadBalancing);

    let announcement = TaskAnnouncement::new(
        "balanced-task",
        "Some work",
        "orchestrator",
        Instant::now() + Duration::from_secs(30),
    );
    manager.announce_task(announcement).await;

    // Agent with high load
    manager
        .receive_bid(
            TaskBid::new("busy-agent", "balanced-task")
                .with_capability_score(0.9)
                .with_load(0.9),
        )
        .await
        .unwrap();

    // Agent with low load
    manager
        .receive_bid(
            TaskBid::new("idle-agent", "balanced-task")
                .with_capability_score(0.5)
                .with_load(0.1),
        )
        .await
        .unwrap();

    let winner = manager.award_task("balanced-task").await;
    assert_eq!(winner, Some("idle-agent".to_string()));
}

#[tokio::test]
async fn contract_net_decline_and_reannounce() {
    let manager = ContractNetManager::new();

    let announcement = TaskAnnouncement::new(
        "flakey-task",
        "Might be declined",
        "orchestrator",
        Instant::now() + Duration::from_secs(30),
    );
    manager.announce_task(announcement).await;

    manager
        .receive_bid(TaskBid::new("agent-1", "flakey-task").with_capability_score(0.9))
        .await
        .unwrap();

    manager.award_task("flakey-task").await;
    manager
        .decline_award("flakey-task", "agent-1", "State changed since bid")
        .await
        .unwrap();

    // Task should no longer have an award
    // Re-announce would be a new announcement
    assert!(
        manager.get_task_status("flakey-task").await.is_none()
            || manager.get_task_status("flakey-task").await == Some(TaskStatus::OpenForBids)
    );
}

// ===========================================================================
// Saga: multi-step execution with compensation
// ===========================================================================

#[tokio::test]
async fn saga_multi_step_success_then_complete() {
    let saga = SagaExecutor::new("agent-1", "deploy pipeline");

    // Step 1: Generic operation
    let op1 = Arc::new(NoOpCompensation {
        description: "Compile code".into(),
        op_type: SagaOperationType::Build,
    });
    let r1 = saga.execute_step(op1).await.unwrap();
    assert!(r1.success);

    // Step 2: File write (compensable)
    let op2 = Arc::new(NoOpCompensation {
        description: "Write config".into(),
        op_type: SagaOperationType::FileWrite,
    });
    let r2 = saga.execute_step(op2).await.unwrap();
    assert!(r2.success);

    // Step 3: Git stage (compensable)
    let op3 = Arc::new(NoOpCompensation {
        description: "Stage changes".into(),
        op_type: SagaOperationType::GitStage,
    });
    let r3 = saga.execute_step(op3).await.unwrap();
    assert!(r3.success);

    assert_eq!(saga.operation_count().await, 3);

    saga.complete().await;
    assert_eq!(saga.status().await, SagaStatus::Completed);
}

#[tokio::test]
async fn saga_compensation_reverses_completed_operations() {
    let saga = SagaExecutor::new("agent-1", "risky deploy");

    // Execute two compensable steps
    saga.execute_step(Arc::new(NoOpCompensation {
        description: "Write file A".into(),
        op_type: SagaOperationType::FileWrite,
    }))
    .await
    .unwrap();

    saga.execute_step(Arc::new(NoOpCompensation {
        description: "Stage file A".into(),
        op_type: SagaOperationType::GitStage,
    }))
    .await
    .unwrap();

    // Something went wrong -- trigger compensation
    saga.fail().await;
    assert_eq!(saga.status().await, SagaStatus::Failed);

    let report = saga.compensate_all().await.unwrap();
    assert_eq!(saga.status().await, SagaStatus::Compensated);

    // Both operations should have been compensated (in reverse order)
    assert_eq!(report.operations.len(), 2);
    assert!(report.all_successful());
}

#[tokio::test]
async fn saga_skips_non_compensable_operations() {
    let saga = SagaExecutor::new("agent-1", "mixed ops");

    // Non-compensable (Build)
    saga.execute_step(Arc::new(NoOpCompensation {
        description: "Build project".into(),
        op_type: SagaOperationType::Build,
    }))
    .await
    .unwrap();

    // Compensable (FileWrite)
    saga.execute_step(Arc::new(NoOpCompensation {
        description: "Write output".into(),
        op_type: SagaOperationType::FileWrite,
    }))
    .await
    .unwrap();

    saga.fail().await;
    let report = saga.compensate_all().await.unwrap();

    // Should have 2 entries: 1 compensated, 1 skipped
    assert_eq!(report.operations.len(), 2);
    assert!(report.all_successful()); // skipped counts as successful
    assert!(report.summary().contains("1 skipped"));
    assert!(report.summary().contains("1 successful"));
}

#[tokio::test]
async fn saga_cannot_execute_after_failure() {
    let saga = SagaExecutor::new("agent-1", "stopped saga");

    saga.fail().await;

    let result = saga
        .execute_step(Arc::new(NoOpCompensation {
            description: "Should not run".into(),
            op_type: SagaOperationType::Generic,
        }))
        .await;

    assert!(result.is_err());
}

// ===========================================================================
// Optimistic concurrency: multi-agent conflict scenarios
// ===========================================================================

#[tokio::test]
async fn optimistic_sequential_commits_succeed() {
    let controller = OptimisticController::new();

    // Agent 1 edits file
    let token1 = controller.begin_optimistic("agent-1", "config.toml").await;
    let v1 = controller
        .commit_optimistic(token1, "hash-v1")
        .await
        .unwrap();
    assert_eq!(v1, 1);

    // Agent 2 edits same file after agent 1
    let token2 = controller.begin_optimistic("agent-2", "config.toml").await;
    assert_eq!(token2.base_version, 1); // sees agent-1's commit
    let v2 = controller
        .commit_optimistic(token2, "hash-v2")
        .await
        .unwrap();
    assert_eq!(v2, 2);
}

#[tokio::test]
async fn optimistic_concurrent_conflict_detected() {
    let controller = OptimisticController::new();

    // Both agents read version 0
    let token_a = controller.begin_optimistic("agent-a", "shared.rs").await;
    let token_b = controller.begin_optimistic("agent-b", "shared.rs").await;

    // Agent A commits first
    controller
        .commit_optimistic(token_a, "hash-a")
        .await
        .unwrap();

    // Agent B's commit should conflict
    let conflict = controller
        .commit_optimistic(token_b, "hash-b")
        .await
        .unwrap_err();

    assert_eq!(conflict.expected_version, 0);
    assert_eq!(conflict.actual_version, 1);
    assert_eq!(conflict.holder_agent, "agent-a");
    assert_eq!(conflict.conflicting_agent, "agent-b");
}

#[tokio::test]
async fn optimistic_last_writer_wins_resolves_conflict() {
    let controller =
        OptimisticController::with_default_strategy(ResolutionStrategy::LastWriterWins);

    let token_a = controller.begin_optimistic("agent-a", "data.json").await;
    let token_b = controller.begin_optimistic("agent-b", "data.json").await;

    controller
        .commit_optimistic(token_a, "hash-a")
        .await
        .unwrap();

    // Agent B commits with auto-resolution
    let result = controller
        .commit_or_resolve(token_b, "hash-b", None)
        .await
        .unwrap();

    assert!(result.is_success());
    let version = controller.get_version("data.json").await.unwrap();
    assert_eq!(version.last_modifier, "agent-b"); // last writer won
}

#[tokio::test]
async fn optimistic_first_writer_wins_rejects_late_commit() {
    let controller =
        OptimisticController::with_default_strategy(ResolutionStrategy::FirstWriterWins);

    let token_a = controller.begin_optimistic("agent-a", "stable.rs").await;
    let token_b = controller.begin_optimistic("agent-b", "stable.rs").await;

    controller
        .commit_optimistic(token_a, "hash-a")
        .await
        .unwrap();

    let result = controller
        .commit_or_resolve(token_b, "hash-b", None)
        .await
        .unwrap();

    match result {
        CommitResult::Rejected { reason } => {
            assert!(reason.contains("agent-a"));
        }
        _ => panic!("Expected Rejected, got {:?}", result),
    }
}

#[tokio::test]
async fn optimistic_per_resource_strategy_override() {
    let controller = OptimisticController::new(); // default = FirstWriterWins

    // Override for a specific resource
    controller
        .register_strategy("mutable.json", ResolutionStrategy::LastWriterWins)
        .await;

    let token_a = controller.begin_optimistic("agent-a", "mutable.json").await;
    let token_b = controller.begin_optimistic("agent-b", "mutable.json").await;

    controller.commit_optimistic(token_a, "ha").await.unwrap();

    // Should use LastWriterWins for this resource
    let result = controller
        .commit_or_resolve(token_b, "hb", None)
        .await
        .unwrap();
    assert!(result.is_success());
}

#[tokio::test]
async fn optimistic_conflict_history_tracked() {
    let controller = OptimisticController::new();

    // Create two conflicts
    for resource in ["file1.rs", "file2.rs"] {
        let t1 = controller.begin_optimistic("a1", resource).await;
        let t2 = controller.begin_optimistic("a2", resource).await;
        controller.commit_optimistic(t1, "h1").await.unwrap();
        let _ = controller.commit_or_resolve(t2, "h2", None).await;
    }

    let stats = controller.get_stats().await;
    assert_eq!(stats.total_conflicts, 2);
    assert_eq!(stats.total_resources, 2);

    let history = controller.get_conflict_history().await;
    assert_eq!(history.len(), 2);

    controller.clear_history().await;
    assert!(controller.get_conflict_history().await.is_empty());
}

// ===========================================================================
// Cross-pattern: Saga + Optimistic concurrency
// ===========================================================================

#[tokio::test]
async fn saga_with_optimistic_version_check() {
    // Scenario: A saga performs operations, each guarded by optimistic versioning.
    // If a version conflict occurs mid-saga, the saga compensates.

    let controller = OptimisticController::new();
    let saga = SagaExecutor::new("agent-1", "versioned deploy");

    // Step 1: Read current version and write
    let token = controller.begin_optimistic("agent-1", "app.rs").await;
    let commit_result = controller.commit_optimistic(token, "v1-hash").await;
    assert!(commit_result.is_ok());

    saga.execute_step(Arc::new(NoOpCompensation {
        description: "Write app.rs v1".into(),
        op_type: SagaOperationType::FileWrite,
    }))
    .await
    .unwrap();

    // Step 2: Another agent modifies the same resource concurrently
    let other_token = controller.begin_optimistic("agent-2", "app.rs").await;
    assert_eq!(other_token.base_version, 1);
    controller
        .commit_optimistic(other_token, "v2-hash-by-agent2")
        .await
        .unwrap();

    // Step 3: Our saga tries to commit again -- conflict!
    let stale_token = controller.begin_optimistic("agent-1", "app.rs").await;
    // Simulate our agent having an old token (version 1, but current is 2)
    let stale = brainwires_agent::optimistic::OptimisticToken {
        resource_id: "app.rs".into(),
        base_version: 1,
        base_hash: "v1-hash".into(),
        agent_id: "agent-1".into(),
        created_at: stale_token.created_at,
    };
    let conflict = controller.commit_optimistic(stale, "v3-hash").await;
    assert!(conflict.is_err());

    // Conflict detected -- compensate the saga
    saga.fail().await;
    let report = saga.compensate_all().await.unwrap();
    assert!(report.all_successful());
    assert_eq!(saga.status().await, SagaStatus::Compensated);
}
