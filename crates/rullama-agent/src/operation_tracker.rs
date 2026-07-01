//! Operation tracker with heartbeat-based liveness checking
//!
//! Replaces fixed timeouts with active liveness monitoring. Operations remain
//! valid as long as:
//! 1. The holding agent is still running (heartbeats received)
//! 2. Any attached external process (e.g., cargo build) is still alive
//!
//! This allows long-running operations (30+ minute builds) to hold locks
//! without arbitrary timeout expiration.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast, oneshot};
use tokio::task::JoinHandle;

use crate::resource_locks::{ResourceScope, ResourceType};

/// Default heartbeat interval (5 seconds)
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Default max missed heartbeats before considered dead (3 = 15 seconds)
pub const DEFAULT_MAX_MISSED_HEARTBEATS: u32 = 3;

/// Maximum number of output lines to keep per operation
const MAX_OUTPUT_LINES: usize = 100;

/// Tracks active operations with heartbeat-based liveness checking
pub struct OperationTracker {
    /// Active operations indexed by operation_id
    operations: RwLock<HashMap<String, ActiveOperation>>,
    /// Heartbeat interval
    heartbeat_interval: Duration,
    /// Max missed heartbeats before considered dead
    max_missed_heartbeats: u32,
    /// Notification channel for operation events
    event_sender: broadcast::Sender<OperationEvent>,
    /// Counter for generating unique operation IDs
    next_id: RwLock<u64>,
}

/// Information about an active operation
#[derive(Debug, Clone)]
pub struct ActiveOperation {
    /// Unique operation ID
    pub operation_id: String,
    /// ID of the agent performing the operation
    pub agent_id: String,
    /// Type of resource being used
    pub resource_type: ResourceType,
    /// Scope of the operation
    pub scope: ResourceScope,
    /// When the operation started
    pub started_at: Instant,
    /// Last heartbeat received
    pub last_heartbeat: Instant,
    /// Process ID if running an external command (e.g., cargo build)
    pub process_id: Option<u32>,
    /// Current status message
    pub status: String,
    /// Recent output lines from the operation
    pub output_lines: VecDeque<String>,
    /// Description of what the operation is doing
    pub description: String,
    /// Whether the operation has been explicitly completed
    pub completed: bool,
}

impl ActiveOperation {
    /// Check if the operation is still alive based on heartbeat
    pub fn is_heartbeat_alive(&self, heartbeat_interval: Duration, max_missed: u32) -> bool {
        if self.completed {
            return false;
        }
        let max_silence = heartbeat_interval * max_missed;
        self.last_heartbeat.elapsed() < max_silence
    }

    /// Get elapsed time since operation started
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get time since last heartbeat
    pub fn time_since_heartbeat(&self) -> Duration {
        self.last_heartbeat.elapsed()
    }
}

/// Events emitted by the operation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OperationEvent {
    /// Operation started.
    Started {
        /// Unique operation identifier.
        operation_id: String,
        /// Agent performing the operation.
        agent_id: String,
        /// Type of resource being used.
        resource_type: String,
        /// Scope of the operation.
        scope: String,
        /// Human-readable description.
        description: String,
    },
    /// Heartbeat received with status update.
    Heartbeat {
        /// Unique operation identifier.
        operation_id: String,
        /// Agent performing the operation.
        agent_id: String,
        /// Current status message.
        status: String,
        /// Seconds elapsed since operation started.
        elapsed_secs: u64,
    },
    /// Operation completed.
    Completed {
        /// Unique operation identifier.
        operation_id: String,
        /// Agent that performed the operation.
        agent_id: String,
        /// Type of resource that was used.
        resource_type: String,
        /// Scope of the operation.
        scope: String,
        /// Total duration in seconds.
        duration_secs: u64,
        /// Whether the operation succeeded.
        success: bool,
        /// Summary of the outcome.
        summary: String,
    },
    /// Operation detected as stale (no heartbeats).
    Stale {
        /// Unique operation identifier.
        operation_id: String,
        /// Agent that was performing the operation.
        agent_id: String,
        /// Type of resource that was held.
        resource_type: String,
        /// Scope of the operation.
        scope: String,
        /// Seconds since last heartbeat.
        last_heartbeat_secs_ago: u64,
    },
    /// Process attached to operation terminated.
    ProcessTerminated {
        /// Unique operation identifier.
        operation_id: String,
        /// Agent that owned the process.
        agent_id: String,
        /// OS process identifier.
        process_id: u32,
        /// Exit status code if available.
        exit_status: Option<i32>,
    },
}

/// Status of an operation for querying
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStatus {
    /// Unique operation identifier.
    pub operation_id: String,
    /// Agent performing the operation.
    pub agent_id: String,
    /// Type of resource being used.
    pub resource_type: String,
    /// Scope of the operation.
    pub scope: String,
    /// Seconds since the operation started.
    pub started_at_secs_ago: u64,
    /// Seconds since the last heartbeat.
    pub last_heartbeat_secs_ago: u64,
    /// Whether the operation is still alive.
    pub is_alive: bool,
    /// Current status message.
    pub status: String,
    /// Human-readable description.
    pub description: String,
    /// Attached OS process identifier, if any.
    pub process_id: Option<u32>,
    /// Recent output lines from the operation.
    pub recent_output: Vec<String>,
}

/// Handle returned when starting an operation
///
/// The handle spawns a background heartbeat task and provides methods
/// to update status and attach processes. When dropped, it signals
/// completion of the operation.
pub struct OperationHandle {
    tracker: Arc<OperationTracker>,
    operation_id: String,
    agent_id: String,
    _resource_type: ResourceType,
    _scope: ResourceScope,
    heartbeat_task: Option<JoinHandle<()>>,
    completion_sender: Option<oneshot::Sender<OperationCompletion>>,
}

/// Completion status sent when handle is dropped
#[derive(Debug)]
struct OperationCompletion {
    success: bool,
    summary: String,
}

impl OperationTracker {
    /// Create a new operation tracker with default settings
    pub fn new() -> Arc<Self> {
        Self::with_config(DEFAULT_HEARTBEAT_INTERVAL, DEFAULT_MAX_MISSED_HEARTBEATS)
    }

    /// Create a new operation tracker with custom settings
    pub fn with_config(heartbeat_interval: Duration, max_missed_heartbeats: u32) -> Arc<Self> {
        let (event_sender, _) = broadcast::channel(256);
        Arc::new(Self {
            operations: RwLock::new(HashMap::new()),
            heartbeat_interval,
            max_missed_heartbeats,
            event_sender,
            next_id: RwLock::new(1),
        })
    }

    /// Subscribe to operation events
    pub fn subscribe(&self) -> broadcast::Receiver<OperationEvent> {
        self.event_sender.subscribe()
    }

    /// Generate a unique operation ID
    async fn generate_id(&self) -> String {
        let mut id = self.next_id.write().await;
        let operation_id = format!("op-{}", *id);
        *id += 1;
        operation_id
    }

    /// Start tracking a new operation
    ///
    /// Returns an OperationHandle that:
    /// - Automatically sends heartbeats
    /// - Can have a process attached for liveness monitoring
    /// - Signals completion when dropped
    pub async fn start_operation(
        self: &Arc<Self>,
        agent_id: &str,
        resource_type: ResourceType,
        scope: ResourceScope,
        description: &str,
    ) -> Result<OperationHandle> {
        let operation_id = self.generate_id().await;
        let now = Instant::now();

        let operation = ActiveOperation {
            operation_id: operation_id.clone(),
            agent_id: agent_id.to_string(),
            resource_type,
            scope: scope.clone(),
            started_at: now,
            last_heartbeat: now,
            process_id: None,
            status: "Starting".to_string(),
            output_lines: VecDeque::new(),
            description: description.to_string(),
            completed: false,
        };

        // Store the operation
        {
            let mut ops = self.operations.write().await;
            ops.insert(operation_id.clone(), operation);
        }

        // Emit started event
        let _ = self.event_sender.send(OperationEvent::Started {
            operation_id: operation_id.clone(),
            agent_id: agent_id.to_string(),
            resource_type: format!("{:?}", resource_type),
            scope: format!("{:?}", scope),
            description: description.to_string(),
        });

        // Create completion channel
        let (completion_sender, completion_receiver) = oneshot::channel::<OperationCompletion>();

        // Spawn heartbeat task
        let tracker = Arc::clone(self);
        let op_id = operation_id.clone();
        let agent = agent_id.to_string();
        let heartbeat_interval = self.heartbeat_interval;

        let heartbeat_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat_interval);

            loop {
                interval.tick().await;

                // Check if operation still exists and check process liveness
                let ops = tracker.operations.read().await;
                if let Some(op) = ops.get(&op_id) {
                    if op.completed {
                        break;
                    }

                    // Check process liveness if attached
                    if let Some(pid) = op.process_id
                        && !is_process_alive(pid)
                    {
                        drop(ops);
                        // Process died - mark operation as potentially stale
                        let _ = tracker
                            .event_sender
                            .send(OperationEvent::ProcessTerminated {
                                operation_id: op_id.clone(),
                                agent_id: agent.clone(),
                                process_id: pid,
                                exit_status: None, // Could try to get this
                            });
                    }
                } else {
                    // Operation was removed
                    break;
                }
            }
        });

        // Spawn a task to wait for completion signal
        let tracker_for_completion = Arc::clone(self);
        let op_id_for_completion = operation_id.clone();
        let agent_for_completion = agent_id.to_string();
        let resource_type_for_completion = resource_type;
        let scope_for_completion = scope.clone();

        tokio::spawn(async move {
            if let Ok(completion) = completion_receiver.await {
                tracker_for_completion
                    .complete_operation_internal(
                        &op_id_for_completion,
                        &agent_for_completion,
                        resource_type_for_completion,
                        &scope_for_completion,
                        completion.success,
                        &completion.summary,
                    )
                    .await;
            }
        });

        Ok(OperationHandle {
            tracker: Arc::clone(self),
            operation_id,
            agent_id: agent_id.to_string(),
            _resource_type: resource_type,
            _scope: scope,
            heartbeat_task: Some(heartbeat_task),
            completion_sender: Some(completion_sender),
        })
    }

    /// Update heartbeat and status for an operation
    pub async fn heartbeat(&self, operation_id: &str, status: &str) -> Result<()> {
        let mut ops = self.operations.write().await;
        let op = ops
            .get_mut(operation_id)
            .ok_or_else(|| anyhow!("Operation {} not found", operation_id))?;

        op.last_heartbeat = Instant::now();
        op.status = status.to_string();

        let _ = self.event_sender.send(OperationEvent::Heartbeat {
            operation_id: operation_id.to_string(),
            agent_id: op.agent_id.clone(),
            status: status.to_string(),
            elapsed_secs: op.elapsed().as_secs(),
        });

        Ok(())
    }

    /// Add output line to an operation
    pub async fn add_output(&self, operation_id: &str, line: &str) -> Result<()> {
        let mut ops = self.operations.write().await;
        let op = ops
            .get_mut(operation_id)
            .ok_or_else(|| anyhow!("Operation {} not found", operation_id))?;

        op.output_lines.push_back(line.to_string());
        if op.output_lines.len() > MAX_OUTPUT_LINES {
            op.output_lines.pop_front();
        }

        Ok(())
    }

    /// Attach a process ID to an operation for liveness monitoring
    pub async fn attach_process(&self, operation_id: &str, process_id: u32) -> Result<()> {
        let mut ops = self.operations.write().await;
        let op = ops
            .get_mut(operation_id)
            .ok_or_else(|| anyhow!("Operation {} not found", operation_id))?;

        op.process_id = Some(process_id);
        Ok(())
    }

    /// Check if an operation is still alive
    pub async fn is_alive(&self, operation_id: &str) -> bool {
        let ops = self.operations.read().await;
        if let Some(op) = ops.get(operation_id) {
            if op.completed {
                return false;
            }

            // Check heartbeat
            if !op.is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats) {
                return false;
            }

            // Check process if attached
            if let Some(pid) = op.process_id
                && !is_process_alive(pid)
            {
                return false;
            }

            true
        } else {
            false
        }
    }

    /// Get status of an operation
    pub async fn get_status(&self, operation_id: &str) -> Option<OperationStatus> {
        let ops = self.operations.read().await;
        ops.get(operation_id).map(|op| OperationStatus {
            operation_id: op.operation_id.clone(),
            agent_id: op.agent_id.clone(),
            resource_type: format!("{:?}", op.resource_type),
            scope: format!("{:?}", op.scope),
            started_at_secs_ago: op.elapsed().as_secs(),
            last_heartbeat_secs_ago: op.time_since_heartbeat().as_secs(),
            is_alive: op.is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats)
                && op.process_id.is_none_or(is_process_alive),
            status: op.status.clone(),
            description: op.description.clone(),
            process_id: op.process_id,
            recent_output: op.output_lines.iter().cloned().collect(),
        })
    }

    /// Get all active operations
    pub async fn list_operations(&self) -> Vec<OperationStatus> {
        let ops = self.operations.read().await;
        ops.values()
            .filter(|op| !op.completed)
            .map(|op| OperationStatus {
                operation_id: op.operation_id.clone(),
                agent_id: op.agent_id.clone(),
                resource_type: format!("{:?}", op.resource_type),
                scope: format!("{:?}", op.scope),
                started_at_secs_ago: op.elapsed().as_secs(),
                last_heartbeat_secs_ago: op.time_since_heartbeat().as_secs(),
                is_alive: op
                    .is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats)
                    && op.process_id.is_none_or(is_process_alive),
                status: op.status.clone(),
                description: op.description.clone(),
                process_id: op.process_id,
                recent_output: op.output_lines.iter().cloned().collect(),
            })
            .collect()
    }

    /// Get operations for a specific agent
    pub async fn operations_for_agent(&self, agent_id: &str) -> Vec<OperationStatus> {
        let ops = self.operations.read().await;
        ops.values()
            .filter(|op| op.agent_id == agent_id && !op.completed)
            .map(|op| OperationStatus {
                operation_id: op.operation_id.clone(),
                agent_id: op.agent_id.clone(),
                resource_type: format!("{:?}", op.resource_type),
                scope: format!("{:?}", op.scope),
                started_at_secs_ago: op.elapsed().as_secs(),
                last_heartbeat_secs_ago: op.time_since_heartbeat().as_secs(),
                is_alive: op
                    .is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats)
                    && op.process_id.is_none_or(is_process_alive),
                status: op.status.clone(),
                description: op.description.clone(),
                process_id: op.process_id,
                recent_output: op.output_lines.iter().cloned().collect(),
            })
            .collect()
    }

    /// Find operations by resource type and scope
    pub async fn find_operation(
        &self,
        resource_type: ResourceType,
        scope: &ResourceScope,
    ) -> Option<OperationStatus> {
        let ops = self.operations.read().await;
        ops.values()
            .find(|op| op.resource_type == resource_type && &op.scope == scope && !op.completed)
            .map(|op| OperationStatus {
                operation_id: op.operation_id.clone(),
                agent_id: op.agent_id.clone(),
                resource_type: format!("{:?}", op.resource_type),
                scope: format!("{:?}", op.scope),
                started_at_secs_ago: op.elapsed().as_secs(),
                last_heartbeat_secs_ago: op.time_since_heartbeat().as_secs(),
                is_alive: op
                    .is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats)
                    && op.process_id.is_none_or(is_process_alive),
                status: op.status.clone(),
                description: op.description.clone(),
                process_id: op.process_id,
                recent_output: op.output_lines.iter().cloned().collect(),
            })
    }

    /// Clean up stale operations (those with expired heartbeats)
    pub async fn cleanup_stale(&self) -> Vec<String> {
        let mut ops = self.operations.write().await;
        let mut stale = Vec::new();

        ops.retain(|id, op| {
            if op.completed {
                return false; // Remove completed operations
            }

            let is_alive = op
                .is_heartbeat_alive(self.heartbeat_interval, self.max_missed_heartbeats)
                && op.process_id.is_none_or(is_process_alive);

            if !is_alive {
                stale.push(id.clone());
                let _ = self.event_sender.send(OperationEvent::Stale {
                    operation_id: id.clone(),
                    agent_id: op.agent_id.clone(),
                    resource_type: format!("{:?}", op.resource_type),
                    scope: format!("{:?}", op.scope),
                    last_heartbeat_secs_ago: op.time_since_heartbeat().as_secs(),
                });
                false
            } else {
                true
            }
        });

        stale
    }

    /// Internal method to complete an operation
    async fn complete_operation_internal(
        &self,
        operation_id: &str,
        agent_id: &str,
        resource_type: ResourceType,
        scope: &ResourceScope,
        success: bool,
        summary: &str,
    ) {
        let duration_secs = {
            let mut ops = self.operations.write().await;
            if let Some(op) = ops.get_mut(operation_id) {
                op.completed = true;
                op.status = if success { "Completed" } else { "Failed" }.to_string();
                op.elapsed().as_secs()
            } else {
                0
            }
        };

        let _ = self.event_sender.send(OperationEvent::Completed {
            operation_id: operation_id.to_string(),
            agent_id: agent_id.to_string(),
            resource_type: format!("{:?}", resource_type),
            scope: format!("{:?}", scope),
            duration_secs,
            success,
            summary: summary.to_string(),
        });

        // Remove the operation after a short delay to allow event processing
        let tracker = Arc::clone(&Arc::new(self.clone_inner().await));
        let op_id = operation_id.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let mut ops = tracker.operations.write().await;
            ops.remove(&op_id);
        });
    }

    /// Clone inner state for spawned tasks
    async fn clone_inner(&self) -> OperationTrackerInner {
        OperationTrackerInner {
            operations: Arc::new(RwLock::new(self.operations.read().await.clone())),
        }
    }
}

/// Inner state for cloning
struct OperationTrackerInner {
    operations: Arc<RwLock<HashMap<String, ActiveOperation>>>,
}

impl Default for OperationTracker {
    fn default() -> Self {
        let (event_sender, _) = broadcast::channel(256);
        Self {
            operations: RwLock::new(HashMap::new()),
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            max_missed_heartbeats: DEFAULT_MAX_MISSED_HEARTBEATS,
            event_sender,
            next_id: RwLock::new(1),
        }
    }
}

impl OperationHandle {
    /// Get the operation ID
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    /// Get the agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Update status (sends heartbeat)
    pub async fn update_status(&self, status: &str) -> Result<()> {
        self.tracker.heartbeat(&self.operation_id, status).await
    }

    /// Add output line
    pub async fn add_output(&self, line: &str) -> Result<()> {
        self.tracker.add_output(&self.operation_id, line).await
    }

    /// Attach a process for liveness monitoring
    pub async fn attach_process(&self, process_id: u32) -> Result<()> {
        self.tracker
            .attach_process(&self.operation_id, process_id)
            .await
    }

    /// Mark operation as completed successfully
    pub fn complete(mut self, summary: &str) {
        if let Some(sender) = self.completion_sender.take() {
            let _ = sender.send(OperationCompletion {
                success: true,
                summary: summary.to_string(),
            });
        }
    }

    /// Mark operation as failed
    pub fn fail(mut self, error: &str) {
        if let Some(sender) = self.completion_sender.take() {
            let _ = sender.send(OperationCompletion {
                success: false,
                summary: error.to_string(),
            });
        }
    }
}

impl Drop for OperationHandle {
    fn drop(&mut self) {
        // Cancel heartbeat task
        if let Some(task) = self.heartbeat_task.take() {
            task.abort();
        }

        // Send completion if not already sent
        if let Some(sender) = self.completion_sender.take() {
            let _ = sender.send(OperationCompletion {
                success: true,
                summary: "Operation handle dropped".to_string(),
            });
        }
    }
}

/// Check if a process is still alive
#[cfg(all(unix, feature = "native"))]
fn is_process_alive(pid: u32) -> bool {
    // Send signal 0 to check if process exists
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    use std::process::Command;
    // On Windows, use tasklist to check if process exists
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {}", pid), "/NH"])
        .output()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains(&pid.to_string())
        })
        .unwrap_or(false)
}

#[cfg(all(not(windows), not(all(unix, feature = "native"))))]
fn is_process_alive(_pid: u32) -> bool {
    // Process liveness checking not available in WASM/non-native
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_start_operation() {
        let tracker = OperationTracker::new();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        assert!(tracker.is_alive(handle.operation_id()).await);

        let status = tracker.get_status(handle.operation_id()).await.unwrap();
        assert_eq!(status.agent_id, "agent-1");
        assert!(status.is_alive);
    }

    #[tokio::test]
    async fn test_heartbeat_updates_status() {
        let tracker = OperationTracker::new();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        handle.update_status("Compiling crate...").await.unwrap();

        let status = tracker.get_status(handle.operation_id()).await.unwrap();
        assert_eq!(status.status, "Compiling crate...");
    }

    #[tokio::test]
    async fn test_operation_completion() {
        let tracker = OperationTracker::new();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        let op_id = handle.operation_id().to_string();
        handle.complete("Build succeeded");

        // Give time for completion to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Operation should be marked as not alive after completion
        assert!(!tracker.is_alive(&op_id).await);
    }

    #[tokio::test]
    async fn test_stale_detection() {
        let tracker = OperationTracker::with_config(
            Duration::from_millis(10), // 10ms heartbeat
            2,                         // 2 missed = 20ms timeout
        );
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        let op_id = handle.operation_id().to_string();

        // Drop handle without completing - simulates crash
        std::mem::forget(handle);

        // Wait for heartbeat timeout
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should now be stale
        assert!(!tracker.is_alive(&op_id).await);
    }

    #[tokio::test]
    async fn test_list_operations() {
        let tracker = OperationTracker::new();
        let scope1 = ResourceScope::Project(PathBuf::from("/test/project1"));
        let scope2 = ResourceScope::Project(PathBuf::from("/test/project2"));

        let _handle1 = tracker
            .start_operation("agent-1", ResourceType::Build, scope1, "build 1")
            .await
            .unwrap();

        let _handle2 = tracker
            .start_operation("agent-2", ResourceType::Test, scope2, "test 2")
            .await
            .unwrap();

        let ops = tracker.list_operations().await;
        assert_eq!(ops.len(), 2);
    }

    #[tokio::test]
    async fn test_find_operation() {
        let tracker = OperationTracker::new();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        let found = tracker.find_operation(ResourceType::Build, &scope).await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().agent_id, "agent-1");

        // Should not find non-existent operation
        let not_found = tracker.find_operation(ResourceType::Test, &scope).await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_output_lines() {
        let tracker = OperationTracker::new();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        handle.add_output("Compiling foo v1.0.0").await.unwrap();
        handle.add_output("Compiling bar v2.0.0").await.unwrap();

        let status = tracker.get_status(handle.operation_id()).await.unwrap();
        assert_eq!(status.recent_output.len(), 2);
        assert_eq!(status.recent_output[0], "Compiling foo v1.0.0");
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let tracker = OperationTracker::new();
        let mut receiver = tracker.subscribe();
        let scope = ResourceScope::Project(PathBuf::from("/test/project"));

        let _handle = tracker
            .start_operation("agent-1", ResourceType::Build, scope.clone(), "cargo build")
            .await
            .unwrap();

        // Should receive started event
        let event = receiver.try_recv().unwrap();
        match event {
            OperationEvent::Started { agent_id, .. } => {
                assert_eq!(agent_id, "agent-1");
            }
            _ => panic!("Expected Started event"),
        }
    }
}
