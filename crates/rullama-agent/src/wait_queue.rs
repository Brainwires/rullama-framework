//! Wait queue implementation for resource coordination
//!
//! Manages agents waiting for locked resources with priority ordering
//! and notification when resources become available.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast, oneshot};

/// Priority-ordered wait queue for resource locks
pub struct WaitQueue {
    /// Queues indexed by resource key (e.g., "build:/path/to/project")
    queues: RwLock<HashMap<String, VecDeque<WaitEntry>>>,
    /// Historical wait times for estimation (resource_key -> durations)
    wait_history: RwLock<HashMap<String, Vec<Duration>>>,
    /// Notification broadcaster for queue events
    event_sender: broadcast::Sender<WaitQueueEvent>,
    /// Maximum history entries to keep per resource
    max_history_entries: usize,
}

/// Entry in the wait queue
#[derive(Debug)]
pub struct WaitEntry {
    /// Agent waiting for the resource
    pub agent_id: String,
    /// Priority (0 = highest, higher numbers = lower priority)
    pub priority: u8,
    /// When the agent registered in the queue
    pub registered_at: Instant,
    /// Whether to automatically acquire when reaching front
    pub auto_acquire: bool,
    /// Channel to notify when agent reaches front of queue
    notify_sender: Option<oneshot::Sender<()>>,
}

/// Handle returned when registering in wait queue
pub struct WaitQueueHandle {
    /// Receiver that fires when agent reaches front of queue
    pub ready: oneshot::Receiver<()>,
    /// Initial position in queue (0 = front)
    pub initial_position: usize,
    /// Resource being waited for
    pub resource_key: String,
    /// Agent ID
    pub agent_id: String,
    /// Reference to wait queue for cancellation
    wait_queue: Arc<WaitQueue>,
}

impl WaitQueueHandle {
    /// Cancel waiting and remove from queue
    pub async fn cancel(self) -> bool {
        self.wait_queue
            .cancel(&self.resource_key, &self.agent_id)
            .await
    }
}

/// Events emitted by the wait queue
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WaitQueueEvent {
    /// Agent registered in queue.
    Registered {
        /// Agent that registered.
        agent_id: String,
        /// Resource being waited for.
        resource_key: String,
        /// Initial position in the queue.
        position: usize,
        /// Agent's priority level.
        priority: u8,
    },
    /// Agent's position changed (due to higher priority agent joining).
    PositionChanged {
        /// Affected agent identifier.
        agent_id: String,
        /// Resource being waited for.
        resource_key: String,
        /// Previous position in queue.
        old_position: usize,
        /// New position in queue.
        new_position: usize,
    },
    /// Agent reached front of queue and can acquire.
    Ready {
        /// Agent that is ready.
        agent_id: String,
        /// Resource now available.
        resource_key: String,
        /// Time spent waiting in milliseconds.
        wait_duration_ms: u64,
    },
    /// Agent was removed from queue (cancelled or resource acquired).
    Removed {
        /// Agent that was removed.
        agent_id: String,
        /// Resource that was being waited for.
        resource_key: String,
        /// Why the agent was removed.
        reason: RemovalReason,
    },
    /// Queue became empty for a resource.
    QueueEmpty {
        /// Resource whose queue is now empty.
        resource_key: String,
    },
}

/// Reason for removal from queue
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemovalReason {
    /// Agent cancelled the wait
    Cancelled,
    /// Agent acquired the resource
    Acquired,
    /// Agent timed out (if timeout was set)
    Timeout,
    /// Resource became unavailable
    ResourceUnavailable,
}

/// Status of a wait queue for a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    /// Resource identifier.
    pub resource_key: String,
    /// Number of agents waiting.
    pub queue_length: usize,
    /// Details about each waiter.
    pub waiters: Vec<WaiterInfo>,
    /// Estimated wait time in milliseconds.
    pub estimated_wait_ms: Option<u64>,
}

/// Information about a waiter in the queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaiterInfo {
    /// Agent identifier.
    pub agent_id: String,
    /// Current position in queue (0 = front).
    pub position: usize,
    /// Priority level (lower = higher priority).
    pub priority: u8,
    /// Seconds since the agent started waiting.
    pub waiting_since_secs: u64,
    /// Whether to auto-acquire when reaching front.
    pub auto_acquire: bool,
}

impl WaitQueue {
    /// Create a new wait queue
    pub fn new() -> Arc<Self> {
        Self::with_max_history(100)
    }

    /// Create a new wait queue with custom max history entries
    pub fn with_max_history(max_history_entries: usize) -> Arc<Self> {
        let (event_sender, _) = broadcast::channel(256);
        Arc::new(Self {
            queues: RwLock::new(HashMap::new()),
            wait_history: RwLock::new(HashMap::new()),
            event_sender,
            max_history_entries,
        })
    }

    /// Subscribe to queue events
    pub fn subscribe(&self) -> broadcast::Receiver<WaitQueueEvent> {
        self.event_sender.subscribe()
    }

    /// Register interest in a resource
    ///
    /// Returns a handle with a receiver that fires when the agent reaches
    /// the front of the queue.
    pub async fn register(
        self: &Arc<Self>,
        resource_key: &str,
        agent_id: &str,
        priority: u8,
        auto_acquire: bool,
    ) -> WaitQueueHandle {
        let (notify_sender, notify_receiver) = oneshot::channel();

        let entry = WaitEntry {
            agent_id: agent_id.to_string(),
            priority,
            registered_at: Instant::now(),
            auto_acquire,
            notify_sender: Some(notify_sender),
        };

        let position = {
            let mut queues = self.queues.write().await;
            let queue = queues.entry(resource_key.to_string()).or_default();

            // Find insertion position based on priority (lower number = higher priority)
            let insert_pos = queue
                .iter()
                .position(|e| e.priority > priority)
                .unwrap_or(queue.len());

            queue.insert(insert_pos, entry);

            // Notify agents whose position changed
            for (i, e) in queue.iter().enumerate().skip(insert_pos + 1) {
                let _ = self.event_sender.send(WaitQueueEvent::PositionChanged {
                    agent_id: e.agent_id.clone(),
                    resource_key: resource_key.to_string(),
                    old_position: i - 1,
                    new_position: i,
                });
            }

            insert_pos
        };

        let _ = self.event_sender.send(WaitQueueEvent::Registered {
            agent_id: agent_id.to_string(),
            resource_key: resource_key.to_string(),
            position,
            priority,
        });

        WaitQueueHandle {
            ready: notify_receiver,
            initial_position: position,
            resource_key: resource_key.to_string(),
            agent_id: agent_id.to_string(),
            wait_queue: Arc::clone(self),
        }
    }

    /// Remove an agent from the queue
    pub async fn cancel(&self, resource_key: &str, agent_id: &str) -> bool {
        let mut queues = self.queues.write().await;

        if let Some(queue) = queues.get_mut(resource_key)
            && let Some(pos) = queue.iter().position(|e| e.agent_id == agent_id)
        {
            queue.remove(pos);

            // Notify agents whose position changed
            for (i, e) in queue.iter().enumerate().skip(pos) {
                let _ = self.event_sender.send(WaitQueueEvent::PositionChanged {
                    agent_id: e.agent_id.clone(),
                    resource_key: resource_key.to_string(),
                    old_position: i + 1,
                    new_position: i,
                });
            }

            let _ = self.event_sender.send(WaitQueueEvent::Removed {
                agent_id: agent_id.to_string(),
                resource_key: resource_key.to_string(),
                reason: RemovalReason::Cancelled,
            });

            if queue.is_empty() {
                queues.remove(resource_key);
                let _ = self.event_sender.send(WaitQueueEvent::QueueEmpty {
                    resource_key: resource_key.to_string(),
                });
            }

            return true;
        }
        false
    }

    /// Notify that a resource was released
    ///
    /// Returns the agent_id of the next waiter (if any) who should acquire.
    pub async fn notify_released(&self, resource_key: &str) -> Option<String> {
        let mut queues = self.queues.write().await;

        if let Some(queue) = queues.get_mut(resource_key)
            && let Some(mut entry) = queue.pop_front()
        {
            let wait_duration = entry.registered_at.elapsed();

            // Record wait time for estimation
            {
                let mut history = self.wait_history.write().await;
                let times = history.entry(resource_key.to_string()).or_default();
                times.push(wait_duration);
                if times.len() > self.max_history_entries {
                    times.remove(0);
                }
            }

            // Notify the waiter
            if let Some(sender) = entry.notify_sender.take() {
                let _ = sender.send(());
            }

            let agent_id = entry.agent_id.clone();

            let _ = self.event_sender.send(WaitQueueEvent::Ready {
                agent_id: agent_id.clone(),
                resource_key: resource_key.to_string(),
                wait_duration_ms: wait_duration.as_millis() as u64,
            });

            // Update positions for remaining waiters
            for (i, e) in queue.iter().enumerate() {
                let _ = self.event_sender.send(WaitQueueEvent::PositionChanged {
                    agent_id: e.agent_id.clone(),
                    resource_key: resource_key.to_string(),
                    old_position: i + 1,
                    new_position: i,
                });
            }

            if queue.is_empty() {
                queues.remove(resource_key);
                let _ = self.event_sender.send(WaitQueueEvent::QueueEmpty {
                    resource_key: resource_key.to_string(),
                });
            }

            return Some(agent_id);
        }
        None
    }

    /// Get queue length for a resource
    pub async fn queue_length(&self, resource_key: &str) -> usize {
        let queues = self.queues.read().await;
        queues.get(resource_key).map_or(0, |q| q.len())
    }

    /// Get position of agent in queue (0 = front)
    pub async fn position(&self, resource_key: &str, agent_id: &str) -> Option<usize> {
        let queues = self.queues.read().await;
        queues
            .get(resource_key)
            .and_then(|q| q.iter().position(|e| e.agent_id == agent_id))
    }

    /// Estimate wait time based on historical data
    pub async fn estimate_wait(&self, resource_key: &str) -> Option<Duration> {
        let history = self.wait_history.read().await;
        if let Some(times) = history.get(resource_key) {
            if times.is_empty() {
                return None;
            }
            // Return average wait time
            let total: Duration = times.iter().sum();
            Some(total / times.len() as u32)
        } else {
            None
        }
    }

    /// Estimate wait time for a specific position
    pub async fn estimate_wait_at_position(
        &self,
        resource_key: &str,
        position: usize,
    ) -> Option<Duration> {
        let base_estimate = self.estimate_wait(resource_key).await?;
        Some(base_estimate * (position as u32 + 1))
    }

    /// Get detailed status of a queue
    pub async fn get_queue_status(&self, resource_key: &str) -> Option<QueueStatus> {
        let queues = self.queues.read().await;
        let queue = queues.get(resource_key)?;

        let waiters: Vec<WaiterInfo> = queue
            .iter()
            .enumerate()
            .map(|(i, e)| WaiterInfo {
                agent_id: e.agent_id.clone(),
                position: i,
                priority: e.priority,
                waiting_since_secs: e.registered_at.elapsed().as_secs(),
                auto_acquire: e.auto_acquire,
            })
            .collect();

        let estimated_wait_ms = self
            .estimate_wait(resource_key)
            .await
            .map(|d| d.as_millis() as u64);

        Some(QueueStatus {
            resource_key: resource_key.to_string(),
            queue_length: queue.len(),
            waiters,
            estimated_wait_ms,
        })
    }

    /// Get all active queues
    pub async fn list_queues(&self) -> Vec<String> {
        let queues = self.queues.read().await;
        queues.keys().cloned().collect()
    }

    /// Check if an agent is waiting for any resource
    pub async fn is_waiting(&self, agent_id: &str) -> bool {
        let queues = self.queues.read().await;
        queues
            .values()
            .any(|q| q.iter().any(|e| e.agent_id == agent_id))
    }

    /// Get all resources an agent is waiting for
    pub async fn waiting_for(&self, agent_id: &str) -> Vec<String> {
        let queues = self.queues.read().await;
        queues
            .iter()
            .filter(|(_, q)| q.iter().any(|e| e.agent_id == agent_id))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Record a completed wait time (for external tracking)
    pub async fn record_wait_time(&self, resource_key: &str, duration: Duration) {
        let mut history = self.wait_history.write().await;
        let times = history.entry(resource_key.to_string()).or_default();
        times.push(duration);
        if times.len() > self.max_history_entries {
            times.remove(0);
        }
    }

    /// Get the next waiter without removing them (peek)
    pub async fn peek_next(&self, resource_key: &str) -> Option<WaiterInfo> {
        let queues = self.queues.read().await;
        queues.get(resource_key).and_then(|q| {
            q.front().map(|e| WaiterInfo {
                agent_id: e.agent_id.clone(),
                position: 0,
                priority: e.priority,
                waiting_since_secs: e.registered_at.elapsed().as_secs(),
                auto_acquire: e.auto_acquire,
            })
        })
    }

    /// Check if agent should auto-acquire (is at front and has auto_acquire set)
    pub async fn should_auto_acquire(&self, resource_key: &str, agent_id: &str) -> bool {
        let queues = self.queues.read().await;
        if let Some(queue) = queues.get(resource_key)
            && let Some(front) = queue.front()
        {
            return front.agent_id == agent_id && front.auto_acquire;
        }
        false
    }
}

impl Default for WaitQueue {
    fn default() -> Self {
        let (event_sender, _) = broadcast::channel(256);
        Self {
            queues: RwLock::new(HashMap::new()),
            wait_history: RwLock::new(HashMap::new()),
            event_sender,
            max_history_entries: 100,
        }
    }
}

/// Generate a resource key for a given operation type and scope
pub fn resource_key(operation_type: &str, scope: &str) -> String {
    format!("{}:{}", operation_type, scope)
}

/// Generate a resource key for a file
pub fn file_resource_key(path: &std::path::Path) -> String {
    format!("file:{}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_position() {
        let queue = WaitQueue::new();

        let handle1 = queue.register("build:/project", "agent-1", 5, false).await;
        let handle2 = queue.register("build:/project", "agent-2", 5, false).await;

        assert_eq!(handle1.initial_position, 0);
        assert_eq!(handle2.initial_position, 1);

        assert_eq!(queue.position("build:/project", "agent-1").await, Some(0));
        assert_eq!(queue.position("build:/project", "agent-2").await, Some(1));
        assert_eq!(queue.queue_length("build:/project").await, 2);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = WaitQueue::new();

        // Register with different priorities
        let _handle1 = queue.register("build:/project", "agent-1", 5, false).await;
        let handle2 = queue.register("build:/project", "agent-2", 1, false).await; // Higher priority
        let _handle3 = queue.register("build:/project", "agent-3", 10, false).await; // Lower priority

        // agent-2 should be at front (priority 1)
        assert_eq!(handle2.initial_position, 0);
        assert_eq!(queue.position("build:/project", "agent-2").await, Some(0));
        assert_eq!(queue.position("build:/project", "agent-1").await, Some(1));
        assert_eq!(queue.position("build:/project", "agent-3").await, Some(2));
    }

    #[tokio::test]
    async fn test_cancel() {
        let queue = WaitQueue::new();

        let _handle1 = queue.register("build:/project", "agent-1", 5, false).await;
        let _handle2 = queue.register("build:/project", "agent-2", 5, false).await;

        assert!(queue.cancel("build:/project", "agent-1").await);
        assert_eq!(queue.position("build:/project", "agent-1").await, None);
        assert_eq!(queue.position("build:/project", "agent-2").await, Some(0));
        assert_eq!(queue.queue_length("build:/project").await, 1);
    }

    #[tokio::test]
    async fn test_notify_released() {
        let queue = WaitQueue::new();

        let _handle1 = queue.register("build:/project", "agent-1", 5, false).await;
        let _handle2 = queue.register("build:/project", "agent-2", 5, false).await;

        // Notify release - should return agent-1
        let next = queue.notify_released("build:/project").await;
        assert_eq!(next, Some("agent-1".to_string()));

        // handle1.ready should now be signaled
        // (Can't easily test this without spawning tasks)

        // agent-2 should now be at front
        assert_eq!(queue.position("build:/project", "agent-2").await, Some(0));
        assert_eq!(queue.queue_length("build:/project").await, 1);
    }

    #[tokio::test]
    async fn test_empty_queue_cleanup() {
        let queue = WaitQueue::new();

        let _handle = queue.register("build:/project", "agent-1", 5, false).await;
        assert!(queue.cancel("build:/project", "agent-1").await);

        // Queue should be removed when empty
        assert_eq!(queue.queue_length("build:/project").await, 0);
        assert!(queue.list_queues().await.is_empty());
    }

    #[tokio::test]
    async fn test_wait_time_estimation() {
        let queue = WaitQueue::new();

        // Record some wait times
        queue
            .record_wait_time("build:/project", Duration::from_secs(10))
            .await;
        queue
            .record_wait_time("build:/project", Duration::from_secs(20))
            .await;
        queue
            .record_wait_time("build:/project", Duration::from_secs(30))
            .await;

        let estimate = queue.estimate_wait("build:/project").await.unwrap();
        assert_eq!(estimate, Duration::from_secs(20)); // Average of 10, 20, 30
    }

    #[tokio::test]
    async fn test_is_waiting() {
        let queue = WaitQueue::new();

        let _handle = queue.register("build:/project", "agent-1", 5, false).await;

        assert!(queue.is_waiting("agent-1").await);
        assert!(!queue.is_waiting("agent-2").await);
    }

    #[tokio::test]
    async fn test_waiting_for() {
        let queue = WaitQueue::new();

        let _handle1 = queue.register("build:/project1", "agent-1", 5, false).await;
        let _handle2 = queue.register("build:/project2", "agent-1", 5, false).await;

        let waiting = queue.waiting_for("agent-1").await;
        assert_eq!(waiting.len(), 2);
        assert!(waiting.contains(&"build:/project1".to_string()));
        assert!(waiting.contains(&"build:/project2".to_string()));
    }

    #[tokio::test]
    async fn test_peek_next() {
        let queue = WaitQueue::new();

        let _handle = queue.register("build:/project", "agent-1", 5, true).await;

        let next = queue.peek_next("build:/project").await.unwrap();
        assert_eq!(next.agent_id, "agent-1");
        assert_eq!(next.priority, 5);
        assert!(next.auto_acquire);

        // Queue should still have the entry
        assert_eq!(queue.queue_length("build:/project").await, 1);
    }

    #[tokio::test]
    async fn test_should_auto_acquire() {
        let queue = WaitQueue::new();

        let _handle1 = queue.register("build:/project", "agent-1", 5, true).await;
        let _handle2 = queue.register("build:/project", "agent-2", 5, false).await;

        assert!(queue.should_auto_acquire("build:/project", "agent-1").await);
        assert!(!queue.should_auto_acquire("build:/project", "agent-2").await);
    }

    #[tokio::test]
    async fn test_queue_status() {
        let queue = WaitQueue::new();

        let _handle1 = queue.register("build:/project", "agent-1", 5, false).await;
        let _handle2 = queue.register("build:/project", "agent-2", 3, true).await;

        let status = queue.get_queue_status("build:/project").await.unwrap();
        assert_eq!(status.queue_length, 2);
        assert_eq!(status.waiters.len(), 2);

        // agent-2 should be first (priority 3 < 5)
        assert_eq!(status.waiters[0].agent_id, "agent-2");
        assert_eq!(status.waiters[1].agent_id, "agent-1");
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let queue = WaitQueue::new();
        let mut receiver = queue.subscribe();

        let _handle = queue.register("build:/project", "agent-1", 5, false).await;

        // Should receive registered event
        let event = receiver.try_recv().unwrap();
        match event {
            WaitQueueEvent::Registered { agent_id, .. } => {
                assert_eq!(agent_id, "agent-1");
            }
            _ => panic!("Expected Registered event"),
        }
    }
}
