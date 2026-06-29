//! Priority Command Queue
//!
//! Implements a priority queue for remote commands with deadline tracking
//! and retry logic.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};
use tracing::debug;

use super::protocol::{BackendCommand, CommandPriority, PrioritizedCommand};

const DEFAULT_QUEUE_MAX_DEPTH: usize = 1000;

/// Entry in the priority queue
#[derive(Debug)]
pub struct QueueEntry {
    /// The prioritized command
    pub command: PrioritizedCommand,
    /// When the command was enqueued
    pub enqueued_at: Instant,
    /// Deadline instant (if set)
    pub deadline: Option<Instant>,
    /// Current retry attempt (0-based)
    pub retry_attempt: u32,
    /// Sequence number for FIFO within same priority
    pub sequence: u64,
}

impl QueueEntry {
    /// Create a new queue entry
    pub fn new(command: PrioritizedCommand, sequence: u64) -> Self {
        let now = Instant::now();
        let deadline = command
            .deadline_ms
            .map(|ms| now + Duration::from_millis(ms));

        Self {
            command,
            enqueued_at: now,
            deadline,
            retry_attempt: 0,
            sequence,
        }
    }

    /// Check if the command has expired
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Instant::now() > d).unwrap_or(false)
    }

    /// Get time until deadline (if set)
    pub fn time_until_deadline(&self) -> Option<Duration> {
        self.deadline.and_then(|d| {
            let now = Instant::now();
            if now < d { Some(d - now) } else { None }
        })
    }

    /// Calculate next retry delay
    pub fn next_retry_delay(&self) -> Option<Duration> {
        self.command.retry_policy.as_ref().and_then(|policy| {
            if self.retry_attempt >= policy.max_attempts {
                None
            } else {
                let delay_ms = policy.initial_delay_ms as f32
                    * policy.backoff_multiplier.powi(self.retry_attempt as i32);
                Some(Duration::from_millis(delay_ms as u64))
            }
        })
    }

    /// Increment retry attempt
    pub fn increment_retry(&mut self) {
        self.retry_attempt += 1;
    }

    /// Check if should retry
    pub fn should_retry(&self) -> bool {
        self.command
            .retry_policy
            .as_ref()
            .map(|p| self.retry_attempt < p.max_attempts)
            .unwrap_or(false)
    }
}

// Implement ordering for BinaryHeap (max-heap, so we reverse for min priority)
impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.command.priority == other.command.priority && self.sequence == other.sequence
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority value = higher priority (Critical=0 is highest)
        // Reverse ordering because BinaryHeap is a max-heap
        match other.command.priority.cmp(&self.command.priority) {
            Ordering::Equal => {
                // Within same priority, use sequence (FIFO)
                // Lower sequence = earlier = should be processed first
                other.sequence.cmp(&self.sequence)
            }
            ord => ord,
        }
    }
}

/// Priority command queue
pub struct CommandQueue {
    /// The priority queue
    queue: BinaryHeap<QueueEntry>,
    /// Sequence counter for FIFO ordering within priority
    sequence: u64,
    /// Maximum queue depth
    max_depth: usize,
}

impl CommandQueue {
    /// Create a new command queue
    pub fn new(max_depth: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            sequence: 0,
            max_depth,
        }
    }

    /// Enqueue a command with priority
    pub fn enqueue(&mut self, command: PrioritizedCommand) -> Result<(), QueueError> {
        // Check queue depth
        if self.queue.len() >= self.max_depth {
            // For critical commands, we allow exceeding the limit
            if command.priority != CommandPriority::Critical {
                return Err(QueueError::QueueFull);
            }
        }

        let entry = QueueEntry::new(command, self.sequence);
        self.sequence = self.sequence.wrapping_add(1);
        self.queue.push(entry);
        Ok(())
    }

    /// Enqueue a simple command (no priority metadata)
    pub fn enqueue_simple(&mut self, command: BackendCommand) -> Result<(), QueueError> {
        self.enqueue(PrioritizedCommand {
            command,
            priority: CommandPriority::Normal,
            deadline_ms: None,
            retry_policy: None,
        })
    }

    /// Dequeue the highest priority command
    pub fn dequeue(&mut self) -> Option<QueueEntry> {
        // Remove expired entries
        self.remove_expired();

        self.queue.pop()
    }

    /// Peek at the highest priority command without removing it
    pub fn peek(&self) -> Option<&QueueEntry> {
        self.queue.peek()
    }

    /// Get current queue depth
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Remove expired entries
    fn remove_expired(&mut self) {
        let mut temp = BinaryHeap::new();
        while let Some(entry) = self.queue.pop() {
            if !entry.is_expired() {
                temp.push(entry);
            } else {
                debug!(
                    "Removed expired command: {:?}",
                    std::mem::discriminant(&entry.command.command)
                );
            }
        }
        self.queue = temp;
    }

    /// Re-enqueue a command for retry
    pub fn requeue_for_retry(&mut self, mut entry: QueueEntry) -> Result<(), QueueError> {
        if !entry.should_retry() {
            return Err(QueueError::MaxRetriesExceeded);
        }

        entry.increment_retry();
        // Update sequence to maintain fairness
        entry.sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        self.queue.push(entry);
        Ok(())
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let mut critical = 0;
        let mut high = 0;
        let mut normal = 0;
        let mut low = 0;

        for entry in self.queue.iter() {
            match entry.command.priority {
                CommandPriority::Critical => critical += 1,
                CommandPriority::High => high += 1,
                CommandPriority::Normal => normal += 1,
                CommandPriority::Low => low += 1,
            }
        }

        QueueStats {
            total: self.queue.len(),
            critical,
            high,
            normal,
            low,
        }
    }
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new(DEFAULT_QUEUE_MAX_DEPTH)
    }
}

/// Queue statistics
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total number of commands in the queue.
    pub total: usize,
    /// Number of critical priority commands.
    pub critical: usize,
    /// Number of high priority commands.
    pub high: usize,
    /// Number of normal priority commands.
    pub normal: usize,
    /// Number of low priority commands.
    pub low: usize,
}

/// Queue errors
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    /// The queue has reached its maximum depth.
    #[error("Queue is full")]
    QueueFull,
    /// The command has exceeded its maximum retry attempts.
    #[error("Maximum retries exceeded")]
    MaxRetriesExceeded,
}

#[cfg(test)]
mod tests {
    use super::super::protocol::RetryPolicy;
    use super::*;

    fn make_command(priority: CommandPriority) -> PrioritizedCommand {
        PrioritizedCommand {
            command: BackendCommand::Ping { timestamp: 0 },
            priority,
            deadline_ms: None,
            retry_policy: None,
        }
    }

    #[test]
    fn test_priority_ordering() {
        let mut queue = CommandQueue::new(100);

        queue.enqueue(make_command(CommandPriority::Low)).unwrap();
        queue.enqueue(make_command(CommandPriority::High)).unwrap();
        queue
            .enqueue(make_command(CommandPriority::Normal))
            .unwrap();
        queue
            .enqueue(make_command(CommandPriority::Critical))
            .unwrap();

        assert_eq!(
            queue.dequeue().unwrap().command.priority,
            CommandPriority::Critical
        );
        assert_eq!(
            queue.dequeue().unwrap().command.priority,
            CommandPriority::High
        );
        assert_eq!(
            queue.dequeue().unwrap().command.priority,
            CommandPriority::Normal
        );
        assert_eq!(
            queue.dequeue().unwrap().command.priority,
            CommandPriority::Low
        );
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut queue = CommandQueue::new(100);

        // Enqueue multiple normal priority commands
        for i in 0..5 {
            queue
                .enqueue(PrioritizedCommand {
                    command: BackendCommand::Ping { timestamp: i },
                    priority: CommandPriority::Normal,
                    deadline_ms: None,
                    retry_policy: None,
                })
                .unwrap();
        }

        // Should come out in FIFO order
        for i in 0..5 {
            let entry = queue.dequeue().unwrap();
            if let BackendCommand::Ping { timestamp } = entry.command.command {
                assert_eq!(timestamp, i);
            } else {
                panic!("Expected Ping command");
            }
        }
    }

    #[test]
    fn test_queue_full() {
        let mut queue = CommandQueue::new(2);

        queue
            .enqueue(make_command(CommandPriority::Normal))
            .unwrap();
        queue
            .enqueue(make_command(CommandPriority::Normal))
            .unwrap();

        // Third normal should fail
        assert!(matches!(
            queue.enqueue(make_command(CommandPriority::Normal)),
            Err(QueueError::QueueFull)
        ));

        // But critical should succeed even when full
        assert!(
            queue
                .enqueue(make_command(CommandPriority::Critical))
                .is_ok()
        );
    }

    #[test]
    fn test_retry_logic() {
        let mut queue = CommandQueue::new(100);

        let cmd = PrioritizedCommand {
            command: BackendCommand::Ping { timestamp: 42 },
            priority: CommandPriority::Normal,
            deadline_ms: None,
            retry_policy: Some(RetryPolicy {
                max_attempts: 3,
                backoff_multiplier: 2.0,
                initial_delay_ms: 100,
            }),
        };

        queue.enqueue(cmd).unwrap();
        let mut entry = queue.dequeue().unwrap();

        // Should be able to retry 3 times
        assert!(entry.should_retry());
        queue.requeue_for_retry(entry).unwrap();

        entry = queue.dequeue().unwrap();
        assert_eq!(entry.retry_attempt, 1);
        assert!(entry.should_retry());
        queue.requeue_for_retry(entry).unwrap();

        entry = queue.dequeue().unwrap();
        assert_eq!(entry.retry_attempt, 2);
        assert!(entry.should_retry());
        queue.requeue_for_retry(entry).unwrap();

        entry = queue.dequeue().unwrap();
        assert_eq!(entry.retry_attempt, 3);
        assert!(!entry.should_retry()); // No more retries

        // Should fail to requeue
        assert!(matches!(
            queue.requeue_for_retry(entry),
            Err(QueueError::MaxRetriesExceeded)
        ));
    }
}
