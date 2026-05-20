//! Priority-based task queue for agent scheduling
//!
//! Provides a thread-safe queue with priority levels (Urgent, High, Normal, Low)
//! for scheduling tasks across worker agents.

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

use brainwires_core::Task;
// Re-export from core to maintain public API compatibility
pub use brainwires_core::TaskPriority;

/// A queued task with priority and metadata
#[derive(Debug, Clone)]
pub struct QueuedTask {
    /// The underlying task.
    pub task: Task,
    /// Priority level.
    pub priority: TaskPriority,
    /// When the task was queued.
    pub queued_at: std::time::SystemTime,
    /// Worker ID if assigned.
    pub assigned_to: Option<String>,
}

impl QueuedTask {
    /// Create a new queued task
    pub fn new(task: Task, priority: TaskPriority) -> Self {
        Self {
            task,
            priority,
            queued_at: std::time::SystemTime::now(),
            assigned_to: None,
        }
    }

    /// Assign this task to a worker
    pub fn assign_to(&mut self, worker_id: String) {
        self.assigned_to = Some(worker_id);
    }

    /// Check if task is assigned
    pub fn is_assigned(&self) -> bool {
        self.assigned_to.is_some()
    }
}

/// Thread-safe task queue with priority support
pub struct TaskQueue {
    queues: Arc<Mutex<PriorityQueues>>,
    max_size: usize,
}

struct PriorityQueues {
    urgent: VecDeque<QueuedTask>,
    high: VecDeque<QueuedTask>,
    normal: VecDeque<QueuedTask>,
    low: VecDeque<QueuedTask>,
}

impl TaskQueue {
    /// Create a new task queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queues: Arc::new(Mutex::new(PriorityQueues {
                urgent: VecDeque::new(),
                high: VecDeque::new(),
                normal: VecDeque::new(),
                low: VecDeque::new(),
            })),
            max_size,
        }
    }

    /// Add a task to the queue
    pub async fn enqueue(&self, task: Task, priority: TaskPriority) -> Result<()> {
        let mut queues = self.queues.lock().await;

        // Check if queue is full
        if self.total_size(&queues) >= self.max_size {
            anyhow::bail!("Task queue is full (max: {})", self.max_size);
        }

        let queued_task = QueuedTask::new(task, priority);

        match priority {
            TaskPriority::Urgent => queues.urgent.push_back(queued_task),
            TaskPriority::High => queues.high.push_back(queued_task),
            TaskPriority::Normal => queues.normal.push_back(queued_task),
            TaskPriority::Low => queues.low.push_back(queued_task),
        }

        Ok(())
    }

    /// Dequeue the highest priority task
    pub async fn dequeue(&self) -> Option<QueuedTask> {
        let mut queues = self.queues.lock().await;

        // Try to dequeue from highest priority first
        queues
            .urgent
            .pop_front()
            .or_else(|| queues.high.pop_front())
            .or_else(|| queues.normal.pop_front())
            .or_else(|| queues.low.pop_front())
    }

    /// Dequeue a task and assign it to a worker
    pub async fn dequeue_and_assign(&self, worker_id: String) -> Option<QueuedTask> {
        let mut queues = self.queues.lock().await;

        // Try to dequeue from highest priority first
        let mut task = queues
            .urgent
            .pop_front()
            .or_else(|| queues.high.pop_front())
            .or_else(|| queues.normal.pop_front())
            .or_else(|| queues.low.pop_front());

        if let Some(ref mut t) = task {
            t.assign_to(worker_id);
        }

        task
    }

    /// Peek at the next task without removing it
    pub async fn peek(&self) -> Option<QueuedTask> {
        let queues = self.queues.lock().await;

        queues
            .urgent
            .front()
            .or_else(|| queues.high.front())
            .or_else(|| queues.normal.front())
            .or_else(|| queues.low.front())
            .cloned()
    }

    /// Get the total number of tasks in the queue
    pub async fn size(&self) -> usize {
        let queues = self.queues.lock().await;
        self.total_size(&queues)
    }

    /// Get the number of tasks at each priority level
    pub async fn size_by_priority(&self) -> (usize, usize, usize, usize) {
        let queues = self.queues.lock().await;
        (
            queues.urgent.len(),
            queues.high.len(),
            queues.normal.len(),
            queues.low.len(),
        )
    }

    /// Check if the queue is empty
    pub async fn is_empty(&self) -> bool {
        self.size().await == 0
    }

    /// Check if the queue is full
    pub async fn is_full(&self) -> bool {
        self.size().await >= self.max_size
    }

    /// Clear all tasks from the queue
    pub async fn clear(&self) {
        let mut queues = self.queues.lock().await;
        queues.urgent.clear();
        queues.high.clear();
        queues.normal.clear();
        queues.low.clear();
    }

    /// Get all tasks (for inspection/debugging)
    pub async fn all_tasks(&self) -> Vec<QueuedTask> {
        let queues = self.queues.lock().await;
        let mut tasks = Vec::new();

        tasks.extend(queues.urgent.iter().cloned());
        tasks.extend(queues.high.iter().cloned());
        tasks.extend(queues.normal.iter().cloned());
        tasks.extend(queues.low.iter().cloned());

        tasks
    }

    /// Find tasks by status
    pub async fn find_by_status(&self, status: brainwires_core::TaskStatus) -> Vec<QueuedTask> {
        let all_tasks = self.all_tasks().await;
        all_tasks
            .into_iter()
            .filter(|qt| qt.task.status == status)
            .collect()
    }

    /// Remove a specific task by ID
    pub async fn remove_by_id(&self, task_id: &str) -> Option<QueuedTask> {
        let mut queues = self.queues.lock().await;

        // Try to find and remove from each queue
        if let Some(pos) = queues.urgent.iter().position(|t| t.task.id == task_id) {
            return queues.urgent.remove(pos);
        }
        if let Some(pos) = queues.high.iter().position(|t| t.task.id == task_id) {
            return queues.high.remove(pos);
        }
        if let Some(pos) = queues.normal.iter().position(|t| t.task.id == task_id) {
            return queues.normal.remove(pos);
        }
        if let Some(pos) = queues.low.iter().position(|t| t.task.id == task_id) {
            return queues.low.remove(pos);
        }

        None
    }

    /// Helper to calculate total size
    fn total_size(&self, queues: &PriorityQueues) -> usize {
        queues.urgent.len() + queues.high.len() + queues.normal.len() + queues.low.len()
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new(100) // Default max size of 100 tasks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_queue_enqueue_dequeue() {
        let queue = TaskQueue::new(10);
        let task = Task::new("test-1".to_string(), "Test task".to_string());

        queue
            .enqueue(task.clone(), TaskPriority::Normal)
            .await
            .unwrap();
        assert_eq!(queue.size().await, 1);

        let dequeued = queue.dequeue().await;
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().task.id, "test-1");
        assert_eq!(queue.size().await, 0);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = TaskQueue::new(10);

        let low = Task::new("low-1".to_string(), "Low priority".to_string());
        let normal = Task::new("normal-1".to_string(), "Normal priority".to_string());
        let high = Task::new("high-1".to_string(), "High priority".to_string());
        let urgent = Task::new("urgent-1".to_string(), "Urgent priority".to_string());

        // Enqueue in reverse priority order
        queue.enqueue(low, TaskPriority::Low).await.unwrap();
        queue.enqueue(normal, TaskPriority::Normal).await.unwrap();
        queue.enqueue(high, TaskPriority::High).await.unwrap();
        queue.enqueue(urgent, TaskPriority::Urgent).await.unwrap();

        // Should dequeue in priority order: Urgent, High, Normal, Low
        assert_eq!(queue.dequeue().await.unwrap().task.id, "urgent-1");
        assert_eq!(queue.dequeue().await.unwrap().task.id, "high-1");
        assert_eq!(queue.dequeue().await.unwrap().task.id, "normal-1");
        assert_eq!(queue.dequeue().await.unwrap().task.id, "low-1");
    }

    #[tokio::test]
    async fn test_max_size() {
        let queue = TaskQueue::new(2);

        let task1 = Task::new("1".to_string(), "Task 1".to_string());
        let task2 = Task::new("2".to_string(), "Task 2".to_string());
        let task3 = Task::new("3".to_string(), "Task 3".to_string());

        queue.enqueue(task1, TaskPriority::Normal).await.unwrap();
        queue.enqueue(task2, TaskPriority::Normal).await.unwrap();

        // Should fail - queue is full
        let result = queue.enqueue(task3, TaskPriority::Normal).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remove_by_id() {
        let queue = TaskQueue::new(10);

        let task1 = Task::new("1".to_string(), "Task 1".to_string());
        let task2 = Task::new("2".to_string(), "Task 2".to_string());

        queue.enqueue(task1, TaskPriority::Normal).await.unwrap();
        queue.enqueue(task2, TaskPriority::High).await.unwrap();

        assert_eq!(queue.size().await, 2);

        let removed = queue.remove_by_id("1").await;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().task.id, "1");
        assert_eq!(queue.size().await, 1);
    }

    #[tokio::test]
    async fn test_assign_to_worker() {
        let queue = TaskQueue::new(10);
        let task = Task::new("test-1".to_string(), "Test task".to_string());

        queue.enqueue(task, TaskPriority::Normal).await.unwrap();

        let dequeued = queue.dequeue_and_assign("worker-1".to_string()).await;
        assert!(dequeued.is_some());

        let qt = dequeued.unwrap();
        assert!(qt.is_assigned());
        assert_eq!(qt.assigned_to.unwrap(), "worker-1");
    }

    #[tokio::test]
    async fn test_peek() {
        let queue = TaskQueue::new(10);
        let task = Task::new("test-1".to_string(), "Test task".to_string());

        queue
            .enqueue(task.clone(), TaskPriority::High)
            .await
            .unwrap();

        let peeked = queue.peek().await;
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap().task.id, "test-1");

        // Size should still be 1 after peek
        assert_eq!(queue.size().await, 1);
    }

    #[tokio::test]
    async fn test_is_empty_and_full() {
        let queue = TaskQueue::new(2);

        assert!(queue.is_empty().await);
        assert!(!queue.is_full().await);

        let task1 = Task::new("1".to_string(), "Task 1".to_string());
        let task2 = Task::new("2".to_string(), "Task 2".to_string());

        queue.enqueue(task1, TaskPriority::Normal).await.unwrap();
        assert!(!queue.is_empty().await);
        assert!(!queue.is_full().await);

        queue.enqueue(task2, TaskPriority::Normal).await.unwrap();
        assert!(!queue.is_empty().await);
        assert!(queue.is_full().await);
    }

    #[tokio::test]
    async fn test_clear() {
        let queue = TaskQueue::new(10);
        let task1 = Task::new("1".to_string(), "Task 1".to_string());
        let task2 = Task::new("2".to_string(), "Task 2".to_string());

        queue.enqueue(task1, TaskPriority::Normal).await.unwrap();
        queue.enqueue(task2, TaskPriority::High).await.unwrap();

        assert_eq!(queue.size().await, 2);

        queue.clear().await;

        assert_eq!(queue.size().await, 0);
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn test_size_by_priority() {
        let queue = TaskQueue::new(10);

        queue
            .enqueue(
                Task::new("1".to_string(), "T1".to_string()),
                TaskPriority::Urgent,
            )
            .await
            .unwrap();
        queue
            .enqueue(
                Task::new("2".to_string(), "T2".to_string()),
                TaskPriority::High,
            )
            .await
            .unwrap();
        queue
            .enqueue(
                Task::new("3".to_string(), "T3".to_string()),
                TaskPriority::High,
            )
            .await
            .unwrap();
        queue
            .enqueue(
                Task::new("4".to_string(), "T4".to_string()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();

        let (urgent, high, normal, low) = queue.size_by_priority().await;
        assert_eq!(urgent, 1);
        assert_eq!(high, 2);
        assert_eq!(normal, 1);
        assert_eq!(low, 0);
    }

    #[tokio::test]
    async fn test_default_queue() {
        let queue = TaskQueue::default();
        assert_eq!(queue.max_size, 100);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Urgent > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_queued_task_new() {
        let task = Task::new("test".to_string(), "Test task".to_string());
        let queued = QueuedTask::new(task.clone(), TaskPriority::High);

        assert_eq!(queued.task.id, task.id);
        assert_eq!(queued.priority, TaskPriority::High);
        assert!(!queued.is_assigned());
    }
}
