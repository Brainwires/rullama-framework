//! Integration tests for task lifecycle management.
//!
//! Tests cross-module flows: TaskManager + TaskQueue + dependency resolution.

use brainwires_agent::task_manager::TaskManager;
use brainwires_agent::task_queue::TaskQueue;
use brainwires_core::{Task, TaskPriority, TaskStatus};

// ---------------------------------------------------------------------------
// TaskManager: create -> assign -> status transitions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn task_create_assign_start_complete() {
    let mgr = TaskManager::new();

    let id = mgr
        .create_task("Build the widget".into(), None, TaskPriority::Normal)
        .await
        .unwrap();

    // Task starts in Pending
    let task = mgr.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::Pending);
    assert!(task.assigned_to.is_none());

    // Assign to an agent
    mgr.assign_task(&id, "agent-alpha").await.unwrap();
    let task = mgr.get_task(&id).await.unwrap();
    assert_eq!(task.assigned_to.as_deref(), Some("agent-alpha"));

    // Start it
    mgr.start_task(&id).await.unwrap();
    let task = mgr.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::InProgress);

    // Complete it
    mgr.complete_task(&id, "Done!".into()).await.unwrap();
    let task = mgr.get_task(&id).await.unwrap();
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.summary.as_deref(), Some("Done!"));
}

#[tokio::test]
async fn task_fail_and_skip_transitions() {
    let mgr = TaskManager::new();

    let id_fail = mgr
        .create_task("Will fail".into(), None, TaskPriority::High)
        .await
        .unwrap();
    let id_skip = mgr
        .create_task("Will skip".into(), None, TaskPriority::Low)
        .await
        .unwrap();

    mgr.start_task(&id_fail).await.unwrap();
    mgr.fail_task(&id_fail, "Oops".into()).await.unwrap();
    let task = mgr.get_task(&id_fail).await.unwrap();
    assert_eq!(task.status, TaskStatus::Failed);

    mgr.skip_task(&id_skip, Some("Not needed".into()))
        .await
        .unwrap();
    let task = mgr.get_task(&id_skip).await.unwrap();
    assert_eq!(task.status, TaskStatus::Skipped);
}

// ---------------------------------------------------------------------------
// Parent-child: auto-completion when all children complete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parent_auto_completes_when_all_children_done() {
    let mgr = TaskManager::new();

    let parent_id = mgr
        .create_task("Parent task".into(), None, TaskPriority::Normal)
        .await
        .unwrap();

    let child_a = mgr
        .add_subtask(parent_id.clone(), "Child A".into())
        .await
        .unwrap();
    let child_b = mgr
        .add_subtask(parent_id.clone(), "Child B".into())
        .await
        .unwrap();

    // Start parent
    mgr.start_task(&parent_id).await.unwrap();

    // Complete child A -- parent still in progress
    mgr.complete_task(&child_a, "A done".into()).await.unwrap();
    let parent = mgr.get_task(&parent_id).await.unwrap();
    assert_eq!(parent.status, TaskStatus::InProgress);

    // Complete child B -- parent should auto-complete
    mgr.complete_task(&child_b, "B done".into()).await.unwrap();
    let parent = mgr.get_task(&parent_id).await.unwrap();
    assert_eq!(parent.status, TaskStatus::Completed);
}

// ---------------------------------------------------------------------------
// Dependencies: blocking and unblocking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dependency_blocks_then_unblocks_on_completion() {
    let mgr = TaskManager::new();

    let prereq = mgr
        .create_task("Prerequisite".into(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let dependent = mgr
        .create_task("Dependent task".into(), None, TaskPriority::Normal)
        .await
        .unwrap();

    mgr.add_dependency(&dependent, &prereq).await.unwrap();

    // Dependent should be blocked because prereq is not done
    let task = mgr.get_task(&dependent).await.unwrap();
    assert_eq!(task.status, TaskStatus::Blocked);

    // Cannot start the dependent
    let can = mgr.can_start(&dependent).await;
    assert!(can.is_err()); // returns Err with blocking IDs

    // Complete the prerequisite
    mgr.start_task(&prereq).await.unwrap();
    mgr.complete_task(&prereq, "prereq done".into())
        .await
        .unwrap();

    // Dependent should now be unblocked (Pending)
    let task = mgr.get_task(&dependent).await.unwrap();
    assert_eq!(task.status, TaskStatus::Pending);

    let can = mgr.can_start(&dependent).await;
    assert_eq!(can, Ok(true));
}

#[tokio::test]
async fn circular_dependency_rejected() {
    let mgr = TaskManager::new();

    let a = mgr
        .create_task("A".into(), None, TaskPriority::Normal)
        .await
        .unwrap();
    let b = mgr
        .create_task("B".into(), None, TaskPriority::Normal)
        .await
        .unwrap();

    mgr.add_dependency(&b, &a).await.unwrap(); // B depends on A
    let result = mgr.add_dependency(&a, &b).await; // A depends on B -> cycle
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// TaskManager + TaskQueue interplay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enqueue_dequeue_respects_priority_order() {
    let queue = TaskQueue::new(100);

    let low_task = Task::new("low-1", "Low priority task");
    let high_task = Task::new("high-1", "High priority task");
    let urgent_task = Task::new("urgent-1", "Urgent priority task");

    queue.enqueue(low_task, TaskPriority::Low).await.unwrap();
    queue.enqueue(high_task, TaskPriority::High).await.unwrap();
    queue
        .enqueue(urgent_task, TaskPriority::Urgent)
        .await
        .unwrap();

    // Should dequeue in priority order: urgent, high, low
    let first = queue.dequeue().await.unwrap();
    assert_eq!(first.task.id, "urgent-1");

    let second = queue.dequeue().await.unwrap();
    assert_eq!(second.task.id, "high-1");

    let third = queue.dequeue().await.unwrap();
    assert_eq!(third.task.id, "low-1");

    // Queue should now be empty
    assert!(queue.dequeue().await.is_none());
}

#[tokio::test]
async fn queue_rejects_when_full() {
    let queue = TaskQueue::new(2);

    queue
        .enqueue(Task::new("t1", "task 1"), TaskPriority::Normal)
        .await
        .unwrap();
    queue
        .enqueue(Task::new("t2", "task 2"), TaskPriority::Normal)
        .await
        .unwrap();

    let result = queue
        .enqueue(Task::new("t3", "task 3"), TaskPriority::Normal)
        .await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// End-to-end: TaskManager creates tasks -> TaskQueue schedules them
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manager_creates_tasks_and_queue_schedules_them() {
    let mgr = TaskManager::new();
    let queue = TaskQueue::new(100);

    // Create tasks via manager
    let id1 = mgr
        .create_task("First task".into(), None, TaskPriority::High)
        .await
        .unwrap();
    let id2 = mgr
        .create_task("Second task".into(), None, TaskPriority::Normal)
        .await
        .unwrap();

    // Retrieve and enqueue them
    let task1 = mgr.get_task(&id1).await.unwrap();
    let task2 = mgr.get_task(&id2).await.unwrap();

    queue.enqueue(task1, TaskPriority::High).await.unwrap();
    queue.enqueue(task2, TaskPriority::Normal).await.unwrap();

    // Dequeue and process
    let queued = queue.dequeue().await.unwrap();
    assert_eq!(queued.task.id, id1); // High comes first

    // Simulate agent processing
    mgr.assign_task(&queued.task.id, "worker-1").await.unwrap();
    mgr.start_task(&queued.task.id).await.unwrap();
    mgr.complete_task(&queued.task.id, "Processed by worker-1".into())
        .await
        .unwrap();

    let completed = mgr.get_task(&queued.task.id).await.unwrap();
    assert_eq!(completed.status, TaskStatus::Completed);
    assert_eq!(completed.assigned_to.as_deref(), Some("worker-1"));
}

// ---------------------------------------------------------------------------
// Load / export round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn export_and_reload_tasks() {
    let mgr = TaskManager::new();

    let id = mgr
        .create_task("Persistent task".into(), None, TaskPriority::High)
        .await
        .unwrap();
    mgr.start_task(&id).await.unwrap();

    let exported = mgr.export_tasks().await;
    assert_eq!(exported.len(), 1);

    // Load into a fresh manager
    let mgr2 = TaskManager::new();
    mgr2.load_tasks(exported).await;

    let task = mgr2.get_task(&id).await.unwrap();
    assert_eq!(task.description, "Persistent task");
    assert_eq!(task.status, TaskStatus::InProgress);
}
