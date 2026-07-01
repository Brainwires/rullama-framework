//! Contract-Net Protocol for Task Allocation
//!
//! Based on Multi-Agent Coordination Survey research, this module implements
//! the Contract-Net Protocol where:
//! 1. Manager broadcasts task announcements
//! 2. Agents submit bids based on capability and availability
//! 3. Manager awards contract to best bidder
//! 4. Winner executes, others continue with other work
//!
//! # Key Concepts
//!
//! - **TaskAnnouncement**: Broadcast to all agents describing a task
//! - **TaskBid**: Agent's response with capability and availability
//! - **ContractMessage**: Protocol messages (Announce, Bid, Award, etc.)
//! - **ContractNetManager**: Manages the bidding process
//! - **ContractParticipant**: Agent-side contract handling

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};

/// Contract-Net Protocol manager
pub struct ContractNetManager {
    /// Active task announcements
    announcements: RwLock<HashMap<String, TaskAnnouncement>>,
    /// Received bids by task_id
    bids: RwLock<HashMap<String, Vec<TaskBid>>>,
    /// Awarded contracts
    awarded: RwLock<HashMap<String, AwardedContract>>,
    /// Communication channel for protocol messages
    broadcast_tx: broadcast::Sender<ContractMessage>,
    /// Bid evaluation strategy
    evaluation_strategy: BidEvaluationStrategy,
    /// Task ID counter
    next_task_id: RwLock<u64>,
}

/// Task announcement broadcast to all agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnnouncement {
    /// Unique task identifier
    pub task_id: String,
    /// Human-readable description
    pub description: String,
    /// Task requirements
    pub requirements: TaskRequirements,
    /// When the task should be completed by (if any)
    #[serde(skip, default)]
    pub deadline: Option<Instant>,
    /// When bidding closes
    #[serde(skip, default = "Instant::now")]
    pub bid_deadline: Instant,
    /// Who announced the task
    pub announcer: String,
    /// When announced
    #[serde(skip, default = "Instant::now")]
    pub announced_at: Instant,
}

impl TaskAnnouncement {
    /// Create a new task announcement
    pub fn new(
        task_id: impl Into<String>,
        description: impl Into<String>,
        announcer: impl Into<String>,
        bid_deadline: Instant,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            description: description.into(),
            requirements: TaskRequirements::default(),
            deadline: None,
            bid_deadline,
            announcer: announcer.into(),
            announced_at: Instant::now(),
        }
    }

    /// Set task requirements
    pub fn with_requirements(mut self, requirements: TaskRequirements) -> Self {
        self.requirements = requirements;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if bidding is still open
    pub fn is_bidding_open(&self) -> bool {
        Instant::now() < self.bid_deadline
    }

    /// Time remaining to bid
    pub fn time_remaining(&self) -> Duration {
        self.bid_deadline.saturating_duration_since(Instant::now())
    }
}

/// Requirements for a task
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Required capabilities (e.g., "rust", "git", "testing")
    pub capabilities: Vec<String>,
    /// Resources needed
    pub resources_needed: Vec<String>,
    /// Estimated complexity (1-10)
    pub complexity: u8,
    /// Priority level (higher = more important)
    pub priority: u8,
    /// Minimum capability score required (0.0 - 1.0)
    pub min_capability_score: f32,
}

impl TaskRequirements {
    /// Create empty task requirements with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set required capabilities.
    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set estimated complexity (clamped to 1-10).
    pub fn with_complexity(mut self, complexity: u8) -> Self {
        self.complexity = complexity.min(10);
        self
    }

    /// Set priority level.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Bid submitted by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBid {
    /// Agent submitting the bid
    pub agent_id: String,
    /// Task being bid on
    pub task_id: String,
    /// Agent's capability match score (0.0 - 1.0)
    pub capability_score: f32,
    /// Agent's current load (0.0 = idle, 1.0 = fully busy)
    pub current_load: f32,
    /// Estimated completion time
    #[serde(skip, default = "default_duration")]
    pub estimated_duration: Duration,
    /// Any constraints or conditions
    pub conditions: Vec<String>,
    /// When the bid was submitted
    #[serde(skip, default = "Instant::now")]
    pub submitted_at: Instant,
}

/// Default duration for serde deserialization
fn default_duration() -> Duration {
    Duration::from_secs(60)
}

impl TaskBid {
    /// Create a new bid
    pub fn new(agent_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            task_id: task_id.into(),
            capability_score: 0.0,
            current_load: 0.0,
            estimated_duration: Duration::from_secs(60),
            conditions: Vec::new(),
            submitted_at: Instant::now(),
        }
    }

    /// Set capability score
    pub fn with_capability_score(mut self, score: f32) -> Self {
        self.capability_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set current load
    pub fn with_load(mut self, load: f32) -> Self {
        self.current_load = load.clamp(0.0, 1.0);
        self
    }

    /// Set estimated duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.estimated_duration = duration;
        self
    }

    /// Add a condition
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Calculate overall bid score (higher is better)
    pub fn score(&self) -> f32 {
        // Weighted combination:
        // - 40% capability match
        // - 30% availability (inverse of load)
        // - 30% speed (inverse of duration)
        let availability = 1.0 - self.current_load;
        let speed = 1.0 / (1.0 + self.estimated_duration.as_secs_f32() / 60.0);

        0.4 * self.capability_score + 0.3 * availability + 0.3 * speed
    }

    /// Calculate score with custom weights
    pub fn score_weighted(
        &self,
        capability_weight: f32,
        availability_weight: f32,
        speed_weight: f32,
    ) -> f32 {
        let total_weight = capability_weight + availability_weight + speed_weight;
        if total_weight == 0.0 {
            return 0.0;
        }

        let availability = 1.0 - self.current_load;
        let speed = 1.0 / (1.0 + self.estimated_duration.as_secs_f32() / 60.0);

        (capability_weight * self.capability_score
            + availability_weight * availability
            + speed_weight * speed)
            / total_weight
    }
}

/// Strategy for evaluating bids
#[derive(Debug, Clone, Default)]
pub enum BidEvaluationStrategy {
    /// Highest overall score wins
    #[default]
    HighestScore,
    /// Lowest estimated duration wins
    FastestCompletion,
    /// Least loaded agent wins
    LoadBalancing,
    /// Highest capability score wins
    BestCapability,
    /// Custom weights for scoring.
    CustomWeights {
        /// Weight for capability score.
        capability: f32,
        /// Weight for availability score.
        availability: f32,
        /// Weight for speed score.
        speed: f32,
    },
}

/// Protocol messages
#[derive(Debug, Clone)]
pub enum ContractMessage {
    /// Broadcast task announcement
    Announce(TaskAnnouncement),
    /// Agent submits bid
    Bid(TaskBid),
    /// Task awarded to agent.
    Award {
        /// Task identifier.
        task_id: String,
        /// Winning agent identifier.
        winner: String,
        /// Winning bid score.
        score: f32,
    },
    /// Task bidding closed with no winner.
    NoAward {
        /// Task identifier.
        task_id: String,
        /// Reason no award was made.
        reason: String,
    },
    /// Winner confirms acceptance.
    Accept {
        /// Task identifier.
        task_id: String,
        /// Accepting agent identifier.
        agent_id: String,
    },
    /// Winner declines (e.g., state changed since bid).
    Decline {
        /// Task identifier.
        task_id: String,
        /// Declining agent identifier.
        agent_id: String,
        /// Reason for declining.
        reason: String,
    },
    /// Task completed notification.
    Complete {
        /// Task identifier.
        task_id: String,
        /// Agent that completed the task.
        agent_id: String,
        /// Whether the task succeeded.
        success: bool,
        /// Optional result output.
        result: Option<String>,
    },
    /// Task cancelled.
    Cancel {
        /// Task identifier.
        task_id: String,
        /// Reason for cancellation.
        reason: String,
    },
}

/// Information about an awarded contract
#[derive(Debug, Clone)]
pub struct AwardedContract {
    /// Task identifier.
    pub task_id: String,
    /// Winning agent identifier.
    pub winner: String,
    /// The winning bid details.
    pub winning_bid: TaskBid,
    /// When the contract was awarded.
    pub awarded_at: Instant,
    /// Whether the winner accepted the contract.
    pub accepted: bool,
    /// Completion status (None if still in progress).
    pub completed: Option<bool>,
}

impl ContractNetManager {
    /// Create a new contract-net manager
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1024);
        Self {
            announcements: RwLock::new(HashMap::new()),
            bids: RwLock::new(HashMap::new()),
            awarded: RwLock::new(HashMap::new()),
            broadcast_tx,
            evaluation_strategy: BidEvaluationStrategy::HighestScore,
            next_task_id: RwLock::new(1),
        }
    }

    /// Create with a specific evaluation strategy
    pub fn with_strategy(strategy: BidEvaluationStrategy) -> Self {
        let mut manager = Self::new();
        manager.evaluation_strategy = strategy;
        manager
    }

    /// Get a receiver for protocol messages
    pub fn subscribe(&self) -> broadcast::Receiver<ContractMessage> {
        self.broadcast_tx.subscribe()
    }

    /// Generate a unique task ID
    pub async fn generate_task_id(&self) -> String {
        let mut id = self.next_task_id.write().await;
        let task_id = format!("task-{}", *id);
        *id += 1;
        task_id
    }

    /// Announce a task for bidding
    pub async fn announce_task(&self, mut announcement: TaskAnnouncement) -> String {
        let task_id = if announcement.task_id.is_empty() {
            self.generate_task_id().await
        } else {
            announcement.task_id.clone()
        };
        announcement.task_id = task_id.clone();

        // Store the announcement
        self.announcements
            .write()
            .await
            .insert(task_id.clone(), announcement.clone());

        // Initialize bids collection
        self.bids.write().await.insert(task_id.clone(), Vec::new());

        // Broadcast the announcement
        let _ = self
            .broadcast_tx
            .send(ContractMessage::Announce(announcement));

        task_id
    }

    /// Process a bid from an agent
    pub async fn receive_bid(&self, bid: TaskBid) -> Result<(), String> {
        let announcements = self.announcements.read().await;

        // Check if task exists and bidding is open
        let announcement = announcements
            .get(&bid.task_id)
            .ok_or_else(|| format!("Unknown task: {}", bid.task_id))?;

        if !announcement.is_bidding_open() {
            return Err("Bid deadline passed".to_string());
        }

        // Check minimum capability score if required
        if bid.capability_score < announcement.requirements.min_capability_score {
            return Err(format!(
                "Capability score {} below minimum {}",
                bid.capability_score, announcement.requirements.min_capability_score
            ));
        }

        // Store the bid
        let mut bids = self.bids.write().await;
        if let Some(task_bids) = bids.get_mut(&bid.task_id) {
            // Remove any existing bid from the same agent
            task_bids.retain(|b| b.agent_id != bid.agent_id);
            task_bids.push(bid.clone());
        }

        // Broadcast the bid
        let bid_task_id = bid.task_id.clone();
        if let Err(e) = self.broadcast_tx.send(ContractMessage::Bid(bid)) {
            tracing::warn!("Failed to broadcast bid for task {}: {}", bid_task_id, e);
        }

        Ok(())
    }

    /// Evaluate bids and award task to winner
    pub async fn award_task(&self, task_id: &str) -> Option<String> {
        let bids = self.bids.read().await;
        let task_bids = bids.get(task_id)?;

        if task_bids.is_empty() {
            if let Err(e) = self.broadcast_tx.send(ContractMessage::NoAward {
                task_id: task_id.to_string(),
                reason: "No bids received".to_string(),
            }) {
                tracing::warn!("Failed to broadcast no-award for task {}: {}", task_id, e);
            }
            return None;
        }

        // Find the winner based on strategy
        let (winner, winning_bid) = self.evaluate_bids(task_bids)?;
        let score = winning_bid.score();

        // Record the award
        self.awarded.write().await.insert(
            task_id.to_string(),
            AwardedContract {
                task_id: task_id.to_string(),
                winner: winner.clone(),
                winning_bid: winning_bid.clone(),
                awarded_at: Instant::now(),
                accepted: false,
                completed: None,
            },
        );

        // Broadcast the award
        let _ = self.broadcast_tx.send(ContractMessage::Award {
            task_id: task_id.to_string(),
            winner: winner.clone(),
            score,
        });

        Some(winner)
    }

    /// Evaluate bids according to strategy
    fn evaluate_bids(&self, bids: &[TaskBid]) -> Option<(String, TaskBid)> {
        if bids.is_empty() {
            return None;
        }

        // Helper to safely compare f32 values, treating NaN as less than all other values
        fn safe_f32_cmp(a: f32, b: f32) -> std::cmp::Ordering {
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Less)
        }

        let winning_bid = match &self.evaluation_strategy {
            BidEvaluationStrategy::HighestScore => bids
                .iter()
                .max_by(|a, b| safe_f32_cmp(a.score(), b.score())),
            BidEvaluationStrategy::FastestCompletion => {
                bids.iter().min_by_key(|b| b.estimated_duration)
            }
            BidEvaluationStrategy::LoadBalancing => bids
                .iter()
                .min_by(|a, b| safe_f32_cmp(a.current_load, b.current_load)),
            BidEvaluationStrategy::BestCapability => bids
                .iter()
                .max_by(|a, b| safe_f32_cmp(a.capability_score, b.capability_score)),
            BidEvaluationStrategy::CustomWeights {
                capability,
                availability,
                speed,
            } => bids.iter().max_by(|a, b| {
                safe_f32_cmp(
                    a.score_weighted(*capability, *availability, *speed),
                    b.score_weighted(*capability, *availability, *speed),
                )
            }),
        }?;

        Some((winning_bid.agent_id.clone(), winning_bid.clone()))
    }

    /// Record acceptance of an award
    pub async fn accept_award(&self, task_id: &str, agent_id: &str) -> Result<(), String> {
        let mut awarded = self.awarded.write().await;
        let contract = awarded
            .get_mut(task_id)
            .ok_or_else(|| format!("No award found for task: {}", task_id))?;

        if contract.winner != agent_id {
            return Err(format!(
                "Agent {} is not the winner of task {}",
                agent_id, task_id
            ));
        }

        contract.accepted = true;

        let _ = self.broadcast_tx.send(ContractMessage::Accept {
            task_id: task_id.to_string(),
            agent_id: agent_id.to_string(),
        });

        Ok(())
    }

    /// Record decline of an award
    pub async fn decline_award(
        &self,
        task_id: &str,
        agent_id: &str,
        reason: &str,
    ) -> Result<(), String> {
        let mut awarded = self.awarded.write().await;
        awarded.remove(task_id);

        let _ = self.broadcast_tx.send(ContractMessage::Decline {
            task_id: task_id.to_string(),
            agent_id: agent_id.to_string(),
            reason: reason.to_string(),
        });

        Ok(())
    }

    /// Record task completion
    pub async fn complete_task(
        &self,
        task_id: &str,
        agent_id: &str,
        success: bool,
        result: Option<String>,
    ) -> Result<(), String> {
        let mut awarded = self.awarded.write().await;
        let contract = awarded
            .get_mut(task_id)
            .ok_or_else(|| format!("No contract found for task: {}", task_id))?;

        if contract.winner != agent_id {
            return Err(format!(
                "Agent {} is not the contractor for task {}",
                agent_id, task_id
            ));
        }

        contract.completed = Some(success);

        let _ = self.broadcast_tx.send(ContractMessage::Complete {
            task_id: task_id.to_string(),
            agent_id: agent_id.to_string(),
            success,
            result,
        });

        // Clean up
        self.announcements.write().await.remove(task_id);
        self.bids.write().await.remove(task_id);

        Ok(())
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        if let Some(contract) = self.awarded.read().await.get(task_id) {
            return Some(if contract.completed.is_some() {
                TaskStatus::Completed
            } else if contract.accepted {
                TaskStatus::InProgress
            } else {
                TaskStatus::Awarded
            });
        }

        if self.announcements.read().await.contains_key(task_id) {
            return Some(TaskStatus::OpenForBids);
        }

        None
    }

    /// Get all pending tasks
    pub async fn get_pending_tasks(&self) -> Vec<TaskAnnouncement> {
        self.announcements.read().await.values().cloned().collect()
    }

    /// Get bids for a task
    pub async fn get_bids(&self, task_id: &str) -> Vec<TaskBid> {
        self.bids
            .read()
            .await
            .get(task_id)
            .cloned()
            .unwrap_or_default()
    }
}

impl Default for ContractNetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Open for bids
    OpenForBids,
    /// Awarded but not yet accepted
    Awarded,
    /// In progress
    InProgress,
    /// Completed
    Completed,
}

/// Agent-side contract participant
pub struct ContractParticipant {
    agent_id: String,
    capabilities: Vec<String>,
    current_tasks: RwLock<Vec<String>>,
    max_concurrent: usize,
    /// Channel for receiving announcements
    message_rx: Option<broadcast::Receiver<ContractMessage>>,
}

impl ContractParticipant {
    /// Create a new contract participant
    pub fn new(agent_id: impl Into<String>, capabilities: Vec<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            capabilities,
            current_tasks: RwLock::new(Vec::new()),
            max_concurrent: 3,
            message_rx: None,
        }
    }

    /// Set maximum concurrent tasks
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Connect to a contract-net manager
    pub fn connect(&mut self, manager: &ContractNetManager) {
        self.message_rx = Some(manager.subscribe());
    }

    /// Check if agent should bid on a task
    pub async fn should_bid(&self, announcement: &TaskAnnouncement) -> bool {
        // Check capability match
        let has_capabilities = announcement
            .requirements
            .capabilities
            .iter()
            .all(|req| self.capabilities.contains(req));

        if !has_capabilities {
            return false;
        }

        // Check capacity
        let current = self.current_tasks.read().await.len();
        if current >= self.max_concurrent {
            return false;
        }

        // Check bidding deadline
        announcement.is_bidding_open()
    }

    /// Generate a bid for a task
    pub async fn generate_bid(&self, announcement: &TaskAnnouncement) -> TaskBid {
        let current_tasks = self.current_tasks.read().await.len();

        TaskBid::new(&self.agent_id, &announcement.task_id)
            .with_capability_score(self.calculate_capability_score(&announcement.requirements))
            .with_load(current_tasks as f32 / self.max_concurrent as f32)
            .with_duration(self.estimate_duration(&announcement.requirements))
    }

    /// Calculate capability score for requirements
    fn calculate_capability_score(&self, requirements: &TaskRequirements) -> f32 {
        if requirements.capabilities.is_empty() {
            return 1.0; // No specific requirements
        }

        let matched = requirements
            .capabilities
            .iter()
            .filter(|c| self.capabilities.contains(c))
            .count();

        matched as f32 / requirements.capabilities.len() as f32
    }

    /// Estimate duration for requirements
    fn estimate_duration(&self, requirements: &TaskRequirements) -> Duration {
        // Base estimate on complexity
        let base_seconds = (requirements.complexity as u64 + 1) * 60;
        Duration::from_secs(base_seconds)
    }

    /// Accept a task (add to current tasks)
    pub async fn accept_task(&self, task_id: &str) {
        self.current_tasks.write().await.push(task_id.to_string());
    }

    /// Complete a task (remove from current tasks)
    pub async fn complete_task(&self, task_id: &str) {
        self.current_tasks.write().await.retain(|t| t != task_id);
    }

    /// Get current task count
    pub async fn current_task_count(&self) -> usize {
        self.current_tasks.read().await.len()
    }

    /// Get available capacity
    pub async fn available_capacity(&self) -> usize {
        self.max_concurrent - self.current_tasks.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_announcement() {
        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        )
        .with_requirements(TaskRequirements::new().with_complexity(5));

        assert!(announcement.is_bidding_open());
        assert!(announcement.time_remaining() <= Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_task_bid_scoring() {
        let bid = TaskBid::new("agent-1", "task-1")
            .with_capability_score(0.8)
            .with_load(0.2)
            .with_duration(Duration::from_secs(120));

        let score = bid.score();
        assert!(score > 0.0 && score <= 1.0);

        // Higher capability should give higher score
        let high_cap_bid = TaskBid::new("agent-2", "task-1")
            .with_capability_score(1.0)
            .with_load(0.2)
            .with_duration(Duration::from_secs(120));

        assert!(high_cap_bid.score() > bid.score());
    }

    #[tokio::test]
    async fn test_announce_and_bid() {
        let manager = ContractNetManager::new();

        // Subscribe before announcing
        let _rx = manager.subscribe();

        // Announce task
        let announcement = TaskAnnouncement::new(
            "",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        );
        let task_id = manager.announce_task(announcement).await;
        assert!(!task_id.is_empty());

        // Submit bid
        let bid = TaskBid::new("agent-1", &task_id)
            .with_capability_score(0.9)
            .with_load(0.1);

        let result = manager.receive_bid(bid).await;
        assert!(result.is_ok());

        // Check bids
        let bids = manager.get_bids(&task_id).await;
        assert_eq!(bids.len(), 1);
        assert_eq!(bids[0].agent_id, "agent-1");
    }

    #[tokio::test]
    async fn test_award_task() {
        let manager = ContractNetManager::new();

        // Announce task
        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        );
        manager.announce_task(announcement).await;

        // Submit bids
        manager
            .receive_bid(TaskBid::new("agent-1", "task-1").with_capability_score(0.7))
            .await
            .unwrap();

        manager
            .receive_bid(TaskBid::new("agent-2", "task-1").with_capability_score(0.9))
            .await
            .unwrap();

        // Award task
        let winner = manager.award_task("task-1").await;
        assert_eq!(winner, Some("agent-2".to_string())); // Higher capability wins
    }

    #[tokio::test]
    async fn test_evaluation_strategies() {
        // Test load balancing strategy
        let manager = ContractNetManager::with_strategy(BidEvaluationStrategy::LoadBalancing);

        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        );
        manager.announce_task(announcement).await;

        manager
            .receive_bid(TaskBid::new("agent-1", "task-1").with_load(0.8))
            .await
            .unwrap();

        manager
            .receive_bid(TaskBid::new("agent-2", "task-1").with_load(0.2))
            .await
            .unwrap();

        let winner = manager.award_task("task-1").await;
        assert_eq!(winner, Some("agent-2".to_string())); // Lower load wins
    }

    #[tokio::test]
    async fn test_bid_rejection_after_deadline() {
        let manager = ContractNetManager::new();

        // Announce with past deadline
        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() - Duration::from_secs(1), // Already past
        );
        manager.announce_task(announcement).await;

        // Try to bid - should fail
        let result = manager.receive_bid(TaskBid::new("agent-1", "task-1")).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("deadline"));
    }

    #[tokio::test]
    async fn test_task_lifecycle() {
        let manager = ContractNetManager::new();

        // Announce
        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        );
        manager.announce_task(announcement).await;
        assert_eq!(
            manager.get_task_status("task-1").await,
            Some(TaskStatus::OpenForBids)
        );

        // Bid
        manager
            .receive_bid(TaskBid::new("agent-1", "task-1").with_capability_score(0.9))
            .await
            .unwrap();

        // Award
        manager.award_task("task-1").await;
        assert_eq!(
            manager.get_task_status("task-1").await,
            Some(TaskStatus::Awarded)
        );

        // Accept
        manager.accept_award("task-1", "agent-1").await.unwrap();
        assert_eq!(
            manager.get_task_status("task-1").await,
            Some(TaskStatus::InProgress)
        );

        // Complete
        manager
            .complete_task("task-1", "agent-1", true, Some("Done".to_string()))
            .await
            .unwrap();
        assert_eq!(
            manager.get_task_status("task-1").await,
            Some(TaskStatus::Completed)
        );
    }

    #[tokio::test]
    async fn test_contract_participant() {
        let participant =
            ContractParticipant::new("agent-1", vec!["rust".to_string(), "git".to_string()])
                .with_max_concurrent(2);

        let announcement = TaskAnnouncement::new(
            "task-1",
            "Test task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        )
        .with_requirements(
            TaskRequirements::new()
                .with_capabilities(vec!["rust".to_string()])
                .with_complexity(5),
        );

        // Should bid - has capability and capacity
        assert!(participant.should_bid(&announcement).await);

        // Generate bid
        let bid = participant.generate_bid(&announcement).await;
        assert_eq!(bid.agent_id, "agent-1");
        assert_eq!(bid.capability_score, 1.0); // Has all required capabilities

        // Accept task
        participant.accept_task("task-1").await;
        assert_eq!(participant.current_task_count().await, 1);

        // Complete task
        participant.complete_task("task-1").await;
        assert_eq!(participant.current_task_count().await, 0);
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let participant =
            ContractParticipant::new("agent-1", vec!["rust".to_string()]).with_max_concurrent(1);

        // Accept one task
        participant.accept_task("task-1").await;

        let announcement = TaskAnnouncement::new(
            "task-2",
            "Another task",
            "manager",
            Instant::now() + Duration::from_secs(60),
        );

        // Should not bid - at capacity
        assert!(!participant.should_bid(&announcement).await);
    }
}
