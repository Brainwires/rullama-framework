//! Tests for TaskManager

use super::*;
use brainwires_core::{TaskPriority, TaskStatus};

#[tokio::test]
async fn test_create_task() {
    let manager = TaskManager::new();
    let id = manager
        .create_task("Test task".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    let task = manager.get_task(&id).await.unwrap();
    assert_eq!(task.description, "Test task");
    assert_eq!(task.status, TaskStatus::Pending);
}

#[tokio::test]
async fn test_create_subtask() {
    let manager = TaskManager::new();
    let parent_id = manager
        .create_task("Parent task".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    let child_id = manager
        .add_subtask(parent_id.clone(), "Child task".to_string())
        .await
        .unwrap();

    let parent = manager.get_task(&parent_id).await.unwrap();
    let child = manager.get_task(&child_id).await.unwrap();

    assert!(parent.children.contains(&child_id));
    assert_eq!(child.parent_id, Some(parent_id));
}

#[tokio::test]
async fn test_task_lifecycle() {
    let manager = TaskManager::new();
    let id = manager
        .create_task("Lifecycle test".to_string(), None, TaskPriority::High)
        .await
        .unwrap();

    // Start
    manager.start_task(&id).await.unwrap();
    let task = manager.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::InProgress);

    // Complete
    manager
        .complete_task(&id, "Done!".to_string())
        .await
        .unwrap();
    let task = manager.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.summary, Some("Done!".to_string()));
}

#[tokio::test]
async fn test_dependencies() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    let task_b = manager
        .create_task(
            "Task B (depends on A)".to_string(),
            None,
            TaskPriority::Normal,
        )
        .await
        .unwrap();

    manager.add_dependency(&task_b, &task_a).await.unwrap();

    // Task B should be blocked
    let b = manager.get_task(&task_b).await.unwrap();
    assert_eq!(b.status, TaskStatus::Blocked);

    // Only Task A should be ready
    let ready = manager.get_ready_tasks().await;
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].id, task_a);
}

#[tokio::test]
async fn test_get_stats() {
    let manager = TaskManager::new();

    let id1 = manager
        .create_task("Task 1".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let id2 = manager
        .create_task("Task 2".to_string(), None, TaskPriority::High)
        .await
        .unwrap();
    let _id3 = manager
        .create_task("Task 3".to_string(), None, TaskPriority::Low)
        .await
        .unwrap();

    manager.start_task(&id1).await.unwrap();
    manager
        .complete_task(&id2, "Done".to_string())
        .await
        .unwrap();

    let stats = manager.get_stats().await;
    assert_eq!(stats.total, 3);
    assert_eq!(stats.pending, 1);
    assert_eq!(stats.in_progress, 1);
    assert_eq!(stats.completed, 1);
}

#[tokio::test]
async fn test_skip_task() {
    let manager = TaskManager::new();

    let id = manager
        .create_task("Skip me".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    manager
        .skip_task(&id, Some("Not needed".to_string()))
        .await
        .unwrap();

    let task = manager.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::Skipped);
    assert_eq!(task.summary, Some("Not needed".to_string()));
    assert!(task.completed_at.is_some());
}

#[tokio::test]
async fn test_skip_unblocks_dependents() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let task_b = manager
        .create_task("Task B".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    manager.add_dependency(&task_b, &task_a).await.unwrap();

    // Task B should be blocked
    let b = manager.get_task(&task_b).await.unwrap();
    assert_eq!(b.status, TaskStatus::Blocked);

    // Skip task A - this should unblock task B
    manager.skip_task(&task_a, None).await.unwrap();

    let b = manager.get_task(&task_b).await.unwrap();
    assert_eq!(b.status, TaskStatus::Pending);
}

#[tokio::test]
async fn test_circular_dependency_detection() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let task_b = manager
        .create_task("Task B".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    // A depends on B
    manager.add_dependency(&task_a, &task_b).await.unwrap();

    // B depends on A should fail (circular)
    let result = manager.add_dependency(&task_b, &task_a).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("circular"));
}

#[tokio::test]
async fn test_self_dependency_detection() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    // Self-dependency should fail
    let result = manager.add_dependency(&task_a, &task_a).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("circular"));
}

#[tokio::test]
async fn test_can_start() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let task_b = manager
        .create_task("Task B".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    manager.add_dependency(&task_b, &task_a).await.unwrap();

    // Task A can start (no dependencies)
    assert!(manager.can_start(&task_a).await.is_ok());
    assert_eq!(manager.can_start(&task_a).await, Ok(true));

    // Task B cannot start (blocked by A)
    let result = manager.can_start(&task_b).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), vec![task_a.clone()]);

    // Complete A, then B can start
    manager
        .complete_task(&task_a, "Done".to_string())
        .await
        .unwrap();
    assert_eq!(manager.can_start(&task_b).await, Ok(true));
}

#[tokio::test]
async fn test_remove_dependency() {
    let manager = TaskManager::new();

    let task_a = manager
        .create_task("Task A".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let task_b = manager
        .create_task("Task B".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    manager.add_dependency(&task_b, &task_a).await.unwrap();

    // B is blocked
    let b = manager.get_task(&task_b).await.unwrap();
    assert_eq!(b.status, TaskStatus::Blocked);

    // Remove the dependency
    manager.remove_dependency(&task_b, &task_a).await.unwrap();

    // B should now be pending (unblocked)
    let b = manager.get_task(&task_b).await.unwrap();
    assert_eq!(b.status, TaskStatus::Pending);
}

#[tokio::test]
async fn test_block_task() {
    let manager = TaskManager::new();

    let id = manager
        .create_task("Block me".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    manager
        .block_task(&id, Some("Waiting on external".to_string()))
        .await
        .unwrap();

    let task = manager.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::Blocked);
    assert_eq!(task.summary, Some("Waiting on external".to_string()));
}

#[tokio::test]
async fn test_format_duration() {
    use super::time_tracking::format_duration_secs;

    assert_eq!(format_duration_secs(30), "30s");
    assert_eq!(format_duration_secs(90), "1m 30s");
    assert_eq!(format_duration_secs(3600), "1h 0m");
    assert_eq!(format_duration_secs(3665), "1h 1m");
    assert_eq!(format_duration_secs(-1), "-");
}

#[tokio::test]
async fn test_skipped_in_stats() {
    let manager = TaskManager::new();

    let id1 = manager
        .create_task("Task 1".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let id2 = manager
        .create_task("Task 2".to_string(), None, TaskPriority::Normal)
        .await
        .unwrap();

    manager.skip_task(&id1, None).await.unwrap();
    manager
        .complete_task(&id2, "Done".to_string())
        .await
        .unwrap();

    let stats = manager.get_stats().await;
    assert_eq!(stats.total, 2);
    assert_eq!(stats.skipped, 1);
    assert_eq!(stats.completed, 1);
}
