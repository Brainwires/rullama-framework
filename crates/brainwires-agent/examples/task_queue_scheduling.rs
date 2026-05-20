//! Example: Priority-based task scheduling with TaskQueue
//!
//! Shows how to create a bounded `TaskQueue`, enqueue tasks at different
//! priority levels, and dequeue them in strict priority order
//! (Urgent > High > Normal > Low).
//!
//! Run: cargo run -p brainwires-agent --example task_queue_scheduling

use anyhow::Result;

use brainwires_agent::TaskQueue;
use brainwires_agent::brainwires_core::{Task, TaskPriority};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Task Queue Scheduling ===\n");

    // 1. Create a queue with max capacity of 10
    let queue = TaskQueue::new(10);
    println!("Created TaskQueue (max_size=10)");
    println!("  Empty: {}\n", queue.is_empty().await);

    // 2. Enqueue tasks at different priorities (inserted out of priority order)
    let tasks: Vec<(&str, &str, TaskPriority)> = vec![
        ("task-low", "Update README", TaskPriority::Low),
        (
            "task-normal-1",
            "Add unit tests for parser",
            TaskPriority::Normal,
        ),
        ("task-high", "Fix authentication bug", TaskPriority::High),
        (
            "task-urgent",
            "Patch critical security vuln",
            TaskPriority::Urgent,
        ),
        (
            "task-normal-2",
            "Refactor error types",
            TaskPriority::Normal,
        ),
    ];

    println!("Enqueuing {} tasks ...", tasks.len());
    for (id, desc, priority) in &tasks {
        let task = Task::new(id.to_string(), desc.to_string());
        queue.enqueue(task, *priority).await?;
        println!("  [{:?}] {}: {}", priority, id, desc);
    }
    println!();

    // 3. Inspect queue sizes
    let total = queue.size().await;
    let (urgent, high, normal, low) = queue.size_by_priority().await;
    println!("Queue stats:");
    println!("  Total:  {total}");
    println!("  Urgent: {urgent}");
    println!("  High:   {high}");
    println!("  Normal: {normal}");
    println!("  Low:    {low}");
    println!("  Full:   {}\n", queue.is_full().await);

    // 4. Peek at the next task (does not remove it)
    if let Some(next) = queue.peek().await {
        println!("Peek => {} ({:?})", next.task.id, next.priority);
        println!("  Queue size still: {}\n", queue.size().await);
    }

    // 5. Dequeue all tasks — they come out in strict priority order
    println!("Dequeuing in priority order:");
    let mut order = 1;
    while let Some(qt) = queue.dequeue().await {
        println!(
            "  {}. [{:?}] {} - \"{}\"",
            order, qt.priority, qt.task.id, qt.task.description
        );
        order += 1;
    }
    println!();

    // 6. Verify queue is empty
    println!("Queue empty after drain: {}", queue.is_empty().await);

    // 7. Demonstrate dequeue_and_assign
    println!("\n--- Worker Assignment ---");
    queue
        .enqueue(
            Task::new("assigned-1".to_string(), "Deploy to staging".to_string()),
            TaskPriority::High,
        )
        .await?;

    if let Some(qt) = queue.dequeue_and_assign("worker-A".to_string()).await {
        println!(
            "  Task \"{}\" assigned to: {}",
            qt.task.id,
            qt.assigned_to.as_deref().unwrap_or("none"),
        );
    }

    println!("\nTask queue scheduling demo complete.");
    Ok(())
}
