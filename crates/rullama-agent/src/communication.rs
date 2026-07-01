//! Inter-agent communication hub and message types
//!
//! Provides a broadcast-based messaging system for agent coordination,
//! including status updates, help requests, task results, and conflict notifications.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc};

/// Types of messages agents can send to each other
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessage {
    /// Request to execute a task
    TaskRequest {
        /// Unique task identifier.
        task_id: String,
        /// Task description.
        description: String,
        /// Task priority (lower = higher priority).
        priority: u8,
    },
    /// Result of task execution
    TaskResult {
        /// Unique task identifier.
        task_id: String,
        /// Whether the task succeeded.
        success: bool,
        /// Result summary.
        result: String,
    },
    /// Status update
    StatusUpdate {
        /// Agent reporting the status.
        agent_id: String,
        /// Current status string.
        status: String,
        /// Optional additional details.
        details: Option<String>,
    },
    /// Request for help/collaboration
    HelpRequest {
        /// Unique request identifier.
        request_id: String,
        /// Help topic.
        topic: String,
        /// Detailed description of what help is needed.
        details: String,
    },
    /// Response to help request
    HelpResponse {
        /// ID of the original help request.
        request_id: String,
        /// Help response content.
        response: String,
    },
    /// Broadcast message to all agents
    Broadcast {
        /// ID of the sending agent.
        sender: String,
        /// Broadcast message content.
        message: String,
    },
    /// Custom message with arbitrary data
    Custom {
        /// Custom message type identifier.
        message_type: String,
        /// Arbitrary JSON data payload.
        data: serde_json::Value,
    },
    /// Notification that an agent was spawned
    AgentSpawned {
        /// ID of the spawned agent.
        agent_id: String,
        /// ID of the task assigned to the agent.
        task_id: String,
    },
    /// Progress update from an agent
    AgentProgress {
        /// ID of the reporting agent.
        agent_id: String,
        /// Completion percentage (0-100).
        progress_percent: u8,
        /// Progress description.
        message: String,
    },
    /// Notification that an agent completed
    AgentCompleted {
        /// ID of the completed agent.
        agent_id: String,
        /// ID of the completed task.
        task_id: String,
        /// Completion summary.
        summary: String,
    },
    /// Notification about lock contention
    LockContention {
        /// ID of the waiting agent.
        agent_id: String,
        /// Path being contended.
        path: String,
        /// ID of the agent holding the lock.
        waiting_for: String,
    },
    /// Request for approval (dangerous operation)
    ApprovalRequest {
        /// Unique request identifier.
        request_id: String,
        /// ID of the requesting agent.
        agent_id: String,
        /// Operation requiring approval.
        operation: String,
        /// Operation details.
        details: String,
    },
    /// Response to approval request
    ApprovalResponse {
        /// ID of the original approval request.
        request_id: String,
        /// Whether the operation was approved.
        approved: bool,
        /// Reason for approval or rejection.
        reason: Option<String>,
    },

    // === New messages for agent coordination ===
    /// Notification that an exclusive operation has started
    OperationStarted {
        /// ID of the agent performing the operation.
        agent_id: String,
        /// Type of operation.
        operation_type: OperationType,
        /// Scope of the operation (e.g., project path).
        scope: String,
        /// Estimated duration in milliseconds.
        estimated_duration_ms: Option<u64>,
        /// Human-readable description.
        description: String,
    },
    /// Notification that an exclusive operation has completed
    OperationCompleted {
        /// ID of the agent that completed the operation.
        agent_id: String,
        /// Type of operation.
        operation_type: OperationType,
        /// Scope of the operation.
        scope: String,
        /// Whether the operation succeeded.
        success: bool,
        /// Actual duration in milliseconds.
        duration_ms: u64,
        /// Completion summary.
        summary: String,
    },
    /// Notification that a lock has become available
    LockAvailable {
        /// Type of operation the lock was for.
        operation_type: OperationType,
        /// Scope of the released lock.
        scope: String,
        /// ID of the agent that released the lock.
        released_by: String,
    },
    /// Update on wait queue position
    WaitQueuePosition {
        /// ID of the waiting agent.
        agent_id: String,
        /// Type of operation being waited on.
        operation_type: OperationType,
        /// Scope of the operation.
        scope: String,
        /// Current position in the wait queue.
        position: usize,
        /// Estimated wait time in milliseconds.
        estimated_wait_ms: Option<u64>,
    },
    /// Git operation started
    GitOperationStarted {
        /// ID of the agent performing the git operation.
        agent_id: String,
        /// Type of git operation.
        git_op: GitOperationType,
        /// Target branch (if applicable).
        branch: Option<String>,
        /// Human-readable description.
        description: String,
    },
    /// Git operation completed
    GitOperationCompleted {
        /// ID of the agent that completed the operation.
        agent_id: String,
        /// Type of git operation.
        git_op: GitOperationType,
        /// Whether the operation succeeded.
        success: bool,
        /// Completion summary.
        summary: String,
    },
    /// Build blocked due to conflicts
    BuildBlocked {
        /// ID of the blocked agent.
        agent_id: String,
        /// Reason the build is blocked.
        reason: String,
        /// List of conflicts causing the block.
        conflicts: Vec<ConflictInfo>,
        /// Estimated wait time in milliseconds.
        estimated_wait_ms: Option<u64>,
    },
    /// File write blocked due to conflicts
    FileWriteBlocked {
        /// ID of the blocked agent.
        agent_id: String,
        /// Path that cannot be written.
        path: String,
        /// Reason the write is blocked.
        reason: String,
        /// List of conflicts causing the block.
        conflicts: Vec<ConflictInfo>,
    },
    /// Resource conflict resolved - agent can proceed
    ConflictResolved {
        /// ID of the agent that can now proceed.
        agent_id: String,
        /// Type of operation that was unblocked.
        operation_type: OperationType,
        /// Scope of the resolved conflict.
        scope: String,
    },

    // === Saga Protocol Messages ===
    /// A saga (multi-step transaction) has started
    SagaStarted {
        /// Unique saga identifier.
        saga_id: String,
        /// ID of the agent executing the saga.
        agent_id: String,
        /// Saga description.
        description: String,
        /// Total number of steps in the saga.
        total_steps: usize,
    },
    /// A saga step has completed
    SagaStepCompleted {
        /// Unique saga identifier.
        saga_id: String,
        /// ID of the executing agent.
        agent_id: String,
        /// Index of the completed step.
        step_index: usize,
        /// Name of the completed step.
        step_name: String,
        /// Whether the step succeeded.
        success: bool,
    },
    /// A saga has completed (successfully or with compensation)
    SagaCompleted {
        /// Unique saga identifier.
        saga_id: String,
        /// ID of the executing agent.
        agent_id: String,
        /// Whether the saga completed successfully.
        success: bool,
        /// Whether compensation was applied.
        compensated: bool,
        /// Completion summary.
        summary: String,
    },
    /// A saga is being compensated (rolling back)
    SagaCompensating {
        /// Unique saga identifier.
        saga_id: String,
        /// ID of the executing agent.
        agent_id: String,
        /// Reason for compensation.
        reason: String,
        /// Number of steps to compensate.
        steps_to_compensate: usize,
    },

    // === Contract-Net Protocol Messages ===
    /// A task has been announced for bidding
    TaskAnnounced {
        /// Unique task identifier.
        task_id: String,
        /// ID of the announcing agent.
        announcer: String,
        /// Task description.
        description: String,
        /// Deadline for bids in milliseconds.
        bid_deadline_ms: u64,
    },
    /// An agent has submitted a bid
    BidSubmitted {
        /// Task being bid on.
        task_id: String,
        /// ID of the bidding agent.
        agent_id: String,
        /// Self-assessed capability score (0.0-1.0).
        capability_score: f32,
        /// Current workload (0.0-1.0).
        current_load: f32,
    },
    /// A task has been awarded to an agent
    TaskAwarded {
        /// Task that was awarded.
        task_id: String,
        /// ID of the winning agent.
        winner: String,
        /// ID of the announcing agent.
        announcer: String,
    },
    /// An agent has accepted an awarded task
    TaskAccepted {
        /// Task that was accepted.
        task_id: String,
        /// ID of the accepting agent.
        agent_id: String,
    },
    /// An agent has declined an awarded task
    TaskDeclined {
        /// Task that was declined.
        task_id: String,
        /// ID of the declining agent.
        agent_id: String,
        /// Reason for declining.
        reason: String,
    },

    // === Market Allocation Messages ===
    /// A resource is available for bidding
    ResourceAvailable {
        /// Unique resource identifier.
        resource_id: String,
        /// Type of resource.
        resource_type: String,
    },
    /// A resource bid has been submitted
    ResourceBidSubmitted {
        /// Resource being bid on.
        resource_id: String,
        /// ID of the bidding agent.
        agent_id: String,
        /// Bid priority.
        priority: u8,
        /// Bid urgency (0.0-1.0).
        urgency: f32,
    },
    /// A resource has been allocated to an agent
    ResourceAllocated {
        /// Allocated resource identifier.
        resource_id: String,
        /// ID of the agent receiving the resource.
        agent_id: String,
        /// Allocation price in credits.
        price: u32,
    },
    /// A resource has been released
    ResourceReleased {
        /// Released resource identifier.
        resource_id: String,
        /// ID of the releasing agent.
        agent_id: String,
    },

    // === Worktree Messages ===
    /// A worktree has been created for an agent
    WorktreeCreated {
        /// ID of the agent that owns the worktree.
        agent_id: String,
        /// Filesystem path to the worktree.
        worktree_path: String,
        /// Git branch for the worktree.
        branch: String,
    },
    /// A worktree has been removed
    WorktreeRemoved {
        /// ID of the agent whose worktree was removed.
        agent_id: String,
        /// Path of the removed worktree.
        worktree_path: String,
    },
    /// An agent is switching worktrees
    WorktreeSwitched {
        /// ID of the switching agent.
        agent_id: String,
        /// Previous worktree path (if any).
        from_path: Option<String>,
        /// New worktree path.
        to_path: String,
    },

    // === Validation Messages ===
    /// A validation check has failed
    ValidationFailed {
        /// ID of the agent that failed validation.
        agent_id: String,
        /// Operation that was being validated.
        operation: String,
        /// Name of the failed validation rule.
        rule_name: String,
        /// Failure description.
        message: String,
    },
    /// A validation warning was raised
    ValidationWarning {
        /// ID of the agent that triggered the warning.
        agent_id: String,
        /// Operation that was being validated.
        operation: String,
        /// Name of the warning rule.
        rule_name: String,
        /// Warning description.
        message: String,
    },

    // === Cycle Orchestration Messages ===
    /// A Plan→Work→Judge cycle has started
    CycleStarted {
        /// Current cycle number (0-indexed).
        cycle_number: u32,
        /// The high-level goal being pursued.
        goal: String,
    },
    /// A Plan→Work→Judge cycle has completed
    CycleCompleted {
        /// Completed cycle number.
        cycle_number: u32,
        /// Type of verdict reached (complete, continue, fresh_restart, abort).
        verdict_type: String,
    },
    /// A planner has produced a task plan
    PlanCreated {
        /// Cycle number the plan belongs to.
        cycle_number: u32,
        /// Number of tasks in the plan.
        task_count: usize,
        /// Planner's rationale summary.
        rationale: String,
    },
    /// A worker's branch has been merged
    WorkerBranchMerged {
        /// ID of the worker agent.
        agent_id: String,
        /// Name of the merged branch.
        branch: String,
        /// Merge status description.
        status: String,
    },

    // === Optimistic Concurrency Messages ===
    /// A version conflict was detected
    VersionConflict {
        /// Resource with the version conflict.
        resource_id: String,
        /// ID of the agent that encountered the conflict.
        agent_id: String,
        /// Version the agent expected.
        expected_version: u64,
        /// Actual current version.
        actual_version: u64,
    },
    /// A conflict has been resolved
    ConflictResolutionApplied {
        /// Resource where the conflict was resolved.
        resource_id: String,
        /// Type of resolution applied.
        resolution_type: String,
        /// Agent whose changes won (if applicable).
        winning_agent: Option<String>,
    },
}

/// Types of operations that require coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationType {
    /// Build operation (cargo build, npm build, etc.)
    Build,
    /// Test operation (cargo test, npm test, etc.)
    Test,
    /// Combined build and test
    BuildTest,
    /// Git index/staging operations
    GitIndex,
    /// Git commit operations
    GitCommit,
    /// Git push operations
    GitPush,
    /// Git pull operations
    GitPull,
    /// Git branch operations
    GitBranch,
    /// File write operation
    FileWrite,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Build => write!(f, "Build"),
            OperationType::Test => write!(f, "Test"),
            OperationType::BuildTest => write!(f, "BuildTest"),
            OperationType::GitIndex => write!(f, "GitIndex"),
            OperationType::GitCommit => write!(f, "GitCommit"),
            OperationType::GitPush => write!(f, "GitPush"),
            OperationType::GitPull => write!(f, "GitPull"),
            OperationType::GitBranch => write!(f, "GitBranch"),
            OperationType::FileWrite => write!(f, "FileWrite"),
        }
    }
}

/// Git-specific operation types for finer-grained control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitOperationType {
    /// Read-only operations (status, diff, log, fetch)
    ReadOnly,
    /// Staging operations (stage, unstage)
    Staging,
    /// Commit operations
    Commit,
    /// Remote write operations (push)
    RemoteWrite,
    /// Remote read/merge operations (pull)
    RemoteMerge,
    /// Branch operations (create, switch, delete)
    Branch,
    /// Destructive operations (discard)
    Destructive,
}

impl std::fmt::Display for GitOperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitOperationType::ReadOnly => write!(f, "ReadOnly"),
            GitOperationType::Staging => write!(f, "Staging"),
            GitOperationType::Commit => write!(f, "Commit"),
            GitOperationType::RemoteWrite => write!(f, "RemoteWrite"),
            GitOperationType::RemoteMerge => write!(f, "RemoteMerge"),
            GitOperationType::Branch => write!(f, "Branch"),
            GitOperationType::Destructive => write!(f, "Destructive"),
        }
    }
}

/// Information about a conflict blocking an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Agent holding the conflicting resource
    pub holder_agent: String,
    /// Resource identifier (path or scope)
    pub resource: String,
    /// How long the conflict has been active (seconds)
    pub duration_secs: u64,
    /// Current status of the blocking operation
    pub status: String,
}

/// Types of conflicts that can block operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConflictType {
    /// File write lock blocks build
    FileWriteBlocksBuild {
        /// Path of the file causing the block.
        path: PathBuf,
    },
    /// Build in progress blocks file write
    BuildBlocksFileWrite,
    /// Test in progress blocks file write
    TestBlocksFileWrite,
    /// Git operation blocks file write
    GitBlocksFileWrite,
    /// File write blocks git operation
    FileWriteBlocksGit {
        /// Path of the file causing the block.
        path: PathBuf,
    },
    /// Build blocks git operation
    BuildBlocksGit,
}

/// Envelope containing message metadata
#[derive(Debug, Clone)]
pub struct MessageEnvelope {
    /// Sender agent ID.
    pub from: String,
    /// Recipient agent ID.
    pub to: String,
    /// The message payload.
    pub message: AgentMessage,
    /// When the message was created.
    pub timestamp: std::time::SystemTime,
}

impl MessageEnvelope {
    /// Create a new message envelope with the current timestamp.
    pub fn new(from: String, to: String, message: AgentMessage) -> Self {
        Self {
            from,
            to,
            message,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

/// Agent communication channel
pub struct AgentChannel {
    sender: mpsc::UnboundedSender<MessageEnvelope>,
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<MessageEnvelope>>>,
}

impl AgentChannel {
    /// Create a new agent channel
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    /// Send a message on this channel
    pub fn send(&self, envelope: MessageEnvelope) -> Result<()> {
        self.sender
            .send(envelope)
            .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))
    }

    /// Receive a message from this channel (async, blocking)
    pub async fn receive(&self) -> Option<MessageEnvelope> {
        self.receiver.lock().await.recv().await
    }

    /// Try to receive a message without blocking
    pub async fn try_receive(&self) -> Option<MessageEnvelope> {
        self.receiver.lock().await.try_recv().ok()
    }
}

impl Default for AgentChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Communication hub for managing multiple agent channels
pub struct CommunicationHub {
    channels: Arc<RwLock<HashMap<String, AgentChannel>>>,
    _broadcast_channel: AgentChannel,
}

impl CommunicationHub {
    /// Create a new communication hub
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
            _broadcast_channel: AgentChannel::new(),
        }
    }

    /// Register an agent with the hub
    #[tracing::instrument(name = "agent.register", skip(self))]
    pub async fn register_agent(&self, agent_id: String) -> Result<()> {
        let mut channels = self.channels.write().await;
        if channels.contains_key(&agent_id) {
            anyhow::bail!("Agent {} is already registered", agent_id);
        }
        channels.insert(agent_id.clone(), AgentChannel::new());
        Ok(())
    }

    /// Unregister an agent from the hub
    #[tracing::instrument(name = "agent.unregister", skip(self))]
    pub async fn unregister_agent(&self, agent_id: &str) -> Result<()> {
        let mut channels = self.channels.write().await;
        if channels.remove(agent_id).is_none() {
            anyhow::bail!("Agent {} is not registered", agent_id);
        }
        Ok(())
    }

    /// Send a message from one agent to another
    #[tracing::instrument(name = "agent.send_message", skip(self, message))]
    pub async fn send_message(
        &self,
        from: String,
        to: String,
        message: AgentMessage,
    ) -> Result<()> {
        let channels = self.channels.read().await;
        let channel = channels
            .get(&to)
            .ok_or_else(|| anyhow::anyhow!("Agent {} is not registered", to))?;

        let envelope = MessageEnvelope::new(from, to, message);
        channel.send(envelope)
    }

    /// Broadcast a message to all agents
    #[tracing::instrument(name = "agent.broadcast", skip(self, message))]
    pub async fn broadcast(&self, from: String, message: AgentMessage) -> Result<()> {
        let channels = self.channels.read().await;
        for (agent_id, channel) in channels.iter() {
            let envelope = MessageEnvelope::new(from.clone(), agent_id.clone(), message.clone());
            channel.send(envelope)?;
        }
        Ok(())
    }

    /// Receive a message for a specific agent
    pub async fn receive_message(&self, agent_id: &str) -> Option<MessageEnvelope> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(agent_id) {
            channel.receive().await
        } else {
            None
        }
    }

    /// Try to receive a message without blocking
    pub async fn try_receive_message(&self, agent_id: &str) -> Option<MessageEnvelope> {
        let channels = self.channels.read().await;
        if let Some(channel) = channels.get(agent_id) {
            channel.try_receive().await
        } else {
            None
        }
    }

    /// Get the number of registered agents
    pub async fn agent_count(&self) -> usize {
        self.channels.read().await.len()
    }

    /// Get list of registered agent IDs
    pub async fn list_agents(&self) -> Vec<String> {
        self.channels.read().await.keys().cloned().collect()
    }

    /// Check if an agent is registered
    pub async fn is_registered(&self, agent_id: &str) -> bool {
        self.channels.read().await.contains_key(agent_id)
    }
}

impl Default for CommunicationHub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_channel() {
        let channel = AgentChannel::new();
        let envelope = MessageEnvelope::new(
            "agent-1".to_string(),
            "agent-2".to_string(),
            AgentMessage::StatusUpdate {
                agent_id: "agent-1".to_string(),
                status: "working".to_string(),
                details: None,
            },
        );

        channel.send(envelope.clone()).unwrap();
        let received = channel.receive().await;
        assert!(received.is_some());
        assert_eq!(received.unwrap().from, "agent-1");
    }

    #[tokio::test]
    async fn test_communication_hub_register() {
        let hub = CommunicationHub::new();

        hub.register_agent("agent-1".to_string()).await.unwrap();
        assert_eq!(hub.agent_count().await, 1);
        assert!(hub.is_registered("agent-1").await);

        // Try to register again - should fail
        let result = hub.register_agent("agent-1".to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_receive_message() {
        let hub = CommunicationHub::new();

        hub.register_agent("agent-1".to_string()).await.unwrap();
        hub.register_agent("agent-2".to_string()).await.unwrap();

        let message = AgentMessage::TaskRequest {
            task_id: "task-1".to_string(),
            description: "Do something".to_string(),
            priority: 5,
        };

        hub.send_message("agent-1".to_string(), "agent-2".to_string(), message)
            .await
            .unwrap();

        let received = hub.receive_message("agent-2").await;
        assert!(received.is_some());

        let envelope = received.unwrap();
        assert_eq!(envelope.from, "agent-1");
        assert_eq!(envelope.to, "agent-2");
    }

    #[tokio::test]
    async fn test_broadcast() {
        let hub = CommunicationHub::new();

        hub.register_agent("agent-1".to_string()).await.unwrap();
        hub.register_agent("agent-2".to_string()).await.unwrap();
        hub.register_agent("agent-3".to_string()).await.unwrap();

        let message = AgentMessage::Broadcast {
            sender: "orchestrator".to_string(),
            message: "Hello all!".to_string(),
        };

        hub.broadcast("orchestrator".to_string(), message)
            .await
            .unwrap();

        // All agents should receive the message
        assert!(hub.try_receive_message("agent-1").await.is_some());
        assert!(hub.try_receive_message("agent-2").await.is_some());
        assert!(hub.try_receive_message("agent-3").await.is_some());
    }

    #[tokio::test]
    async fn test_unregister() {
        let hub = CommunicationHub::new();

        hub.register_agent("agent-1".to_string()).await.unwrap();
        assert_eq!(hub.agent_count().await, 1);

        hub.unregister_agent("agent-1").await.unwrap();
        assert_eq!(hub.agent_count().await, 0);
        assert!(!hub.is_registered("agent-1").await);
    }
}
