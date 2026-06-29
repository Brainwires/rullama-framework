//! Market-Based Resource Allocation with Priority Bidding
//!
//! Based on Multi-Agent Coordination Survey and Hierarchical Multi-Agent Systems
//! research, this module implements market-based allocation where agents bid for
//! resources with dynamic urgency scores. Higher urgency = higher chance of getting
//! the resource.
//!
//! # Key Concepts
//!
//! - **ResourceBid**: Agent's bid with base priority and urgency multiplier
//! - **AgentBudget**: Budget management for fair allocation
//! - **MarketAllocator**: Manages auctions and allocations
//! - **PricingStrategy**: How prices are calculated (first-price, second-price, etc.)
//! - **UrgencyCalculator**: Dynamic priority based on context

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Market-based resource allocator
pub struct MarketAllocator {
    /// Resource auctions
    auctions: RwLock<HashMap<String, ResourceAuction>>,
    /// Agent budgets (for fair allocation)
    budgets: RwLock<HashMap<String, AgentBudget>>,
    /// Pricing strategy
    pricing: PricingStrategy,
    /// Allocation history for analysis
    allocation_history: RwLock<Vec<AllocationRecord>>,
    /// Maximum history entries
    max_history: usize,
}

/// A resource auction
pub struct ResourceAuction {
    /// Resource being auctioned
    pub resource_id: String,
    /// Current bids
    pub bids: Vec<ResourceBid>,
    /// Current holder (if any)
    pub current_holder: Option<CurrentHolder>,
    /// When the auction started
    pub auction_start: Instant,
}

/// Information about current resource holder
#[derive(Debug, Clone)]
pub struct CurrentHolder {
    /// Agent holding the resource
    pub agent_id: String,
    /// When they acquired it
    pub acquired_at: Instant,
    /// Expected release time (if known)
    pub expected_release: Option<Instant>,
}

/// Bid submitted by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBid {
    /// Agent submitting the bid
    pub agent_id: String,
    /// Resource being bid on
    pub resource_id: String,
    /// Base priority (0-10, static)
    pub base_priority: u8,
    /// Urgency multiplier (1.0 = normal, 2.0 = double urgency)
    pub urgency_multiplier: f32,
    /// Maximum bid amount from budget
    pub max_bid: u32,
    /// Reason for urgency (for logging/debugging)
    pub urgency_reason: String,
    /// Estimated hold duration
    #[serde(skip, default = "default_duration")]
    pub estimated_duration: Duration,
    /// When the bid was submitted
    #[serde(skip, default = "Instant::now")]
    pub submitted_at: Instant,
}

/// Default duration for serde deserialization
fn default_duration() -> Duration {
    Duration::from_secs(60)
}

impl ResourceBid {
    /// Create a new bid
    pub fn new(agent_id: impl Into<String>, resource_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            resource_id: resource_id.into(),
            base_priority: 5,
            urgency_multiplier: 1.0,
            max_bid: 10,
            urgency_reason: String::new(),
            estimated_duration: Duration::from_secs(60),
            submitted_at: Instant::now(),
        }
    }

    /// Set base priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.base_priority = priority.min(10);
        self
    }

    /// Set urgency multiplier
    pub fn with_urgency(mut self, multiplier: f32, reason: impl Into<String>) -> Self {
        self.urgency_multiplier = multiplier.clamp(0.1, 10.0);
        self.urgency_reason = reason.into();
        self
    }

    /// Set max bid
    pub fn with_max_bid(mut self, max_bid: u32) -> Self {
        self.max_bid = max_bid;
        self
    }

    /// Set estimated duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.estimated_duration = duration;
        self
    }

    /// Calculate effective priority
    pub fn effective_priority(&self) -> f32 {
        self.base_priority as f32 * self.urgency_multiplier
    }

    /// Calculate bid score (for ranking)
    pub fn score(&self) -> f32 {
        // Combine priority, urgency, and bid amount
        let priority_factor = self.effective_priority() / 10.0;
        let bid_factor = (self.max_bid as f32 / 100.0).min(1.0);

        0.7 * priority_factor + 0.3 * bid_factor
    }
}

/// Agent's budget for bidding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBudget {
    /// Agent identifier
    pub agent_id: String,
    /// Total budget points
    pub total_budget: u32,
    /// Currently available points
    pub available: u32,
    /// Budget replenishment rate (points per second)
    pub replenish_rate: f32,
    /// Last replenishment time
    #[serde(skip, default = "Instant::now")]
    pub last_replenish: Instant,
}

impl AgentBudget {
    /// Create a new budget
    pub fn new(agent_id: impl Into<String>, total_budget: u32) -> Self {
        Self {
            agent_id: agent_id.into(),
            total_budget,
            available: total_budget,
            replenish_rate: 1.0, // 1 point per second by default
            last_replenish: Instant::now(),
        }
    }

    /// Set replenishment rate
    pub fn with_replenish_rate(mut self, rate: f32) -> Self {
        self.replenish_rate = rate.max(0.0);
        self
    }

    /// Replenish budget based on elapsed time
    pub fn replenish(&mut self) {
        let elapsed = self.last_replenish.elapsed().as_secs_f32();
        let replenished = (elapsed * self.replenish_rate) as u32;
        self.available = (self.available + replenished).min(self.total_budget);
        self.last_replenish = Instant::now();
    }

    /// Check if agent can afford a bid
    pub fn can_afford(&self, amount: u32) -> bool {
        self.available >= amount
    }

    /// Spend budget points
    pub fn spend(&mut self, amount: u32) -> bool {
        if self.available >= amount {
            self.available -= amount;
            true
        } else {
            false
        }
    }

    /// Refund budget points (e.g., if allocation failed)
    pub fn refund(&mut self, amount: u32) {
        self.available = (self.available + amount).min(self.total_budget);
    }

    /// Get current availability as percentage
    pub fn availability_percent(&self) -> f32 {
        self.available as f32 / self.total_budget as f32 * 100.0
    }
}

/// Strategy for calculating prices
#[derive(Debug, Clone, Default)]
pub enum PricingStrategy {
    /// Winner pays their bid
    FirstPrice,
    /// Winner pays second-highest bid + 1
    #[default]
    SecondPrice,
    /// Fixed price based on resource type
    FixedPrice(HashMap<String, u32>),
    /// Dynamic based on demand
    Dynamic {
        /// Base price before demand adjustment.
        base_price: u32,
        /// Multiplier applied per competing bid.
        demand_multiplier: f32,
    },
    /// Free (no budget consumption)
    Free,
}

/// Result of an allocation attempt
#[derive(Debug, Clone)]
pub enum AllocationResult {
    /// Resource allocated to agent
    Allocated {
        /// Winning agent identifier.
        agent_id: String,
        /// Price paid for the allocation.
        price: u32,
        /// Position in bid ranking.
        position: usize,
    },
    /// No valid bids
    NoBids,
    /// Resource still held by current owner
    StillHeld {
        /// Agent currently holding the resource.
        holder: String,
        /// Estimated time until release.
        remaining: Option<Duration>,
    },
    /// Agent doesn't have enough budget
    InsufficientBudget {
        /// Agent that could not afford.
        agent_id: String,
        /// Budget required.
        required: u32,
        /// Budget available.
        available: u32,
    },
    /// Bid was outbid by another agent
    Outbid {
        /// Agent that was outbid.
        agent_id: String,
        /// Agent that won.
        winning_agent: String,
        /// Winning bid score.
        winning_score: f32,
    },
}

impl AllocationResult {
    /// Check if allocation was successful
    pub fn is_success(&self) -> bool {
        matches!(self, AllocationResult::Allocated { .. })
    }

    /// Get the winning agent if allocation succeeded
    pub fn winning_agent(&self) -> Option<&str> {
        match self {
            AllocationResult::Allocated { agent_id, .. } => Some(agent_id),
            _ => None,
        }
    }
}

/// Record of an allocation for history
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Resource that was allocated
    pub resource_id: String,
    /// Winning agent
    pub winner: String,
    /// Price paid
    pub price: u32,
    /// Number of competing bids
    pub competing_bids: usize,
    /// When allocated
    pub allocated_at: Instant,
}

impl MarketAllocator {
    /// Create a new market allocator with default settings
    pub fn new() -> Self {
        Self {
            auctions: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            pricing: PricingStrategy::SecondPrice,
            allocation_history: RwLock::new(Vec::new()),
            max_history: 1000,
        }
    }

    /// Create with a specific pricing strategy
    pub fn with_pricing(pricing: PricingStrategy) -> Self {
        Self {
            auctions: RwLock::new(HashMap::new()),
            budgets: RwLock::new(HashMap::new()),
            pricing,
            allocation_history: RwLock::new(Vec::new()),
            max_history: 1000,
        }
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Initialize budget for an agent
    pub async fn register_agent(&self, agent_id: &str, total_budget: u32, replenish_rate: f32) {
        self.budgets.write().await.insert(
            agent_id.to_string(),
            AgentBudget::new(agent_id, total_budget).with_replenish_rate(replenish_rate),
        );
    }

    /// Get an agent's current budget
    pub async fn get_budget(&self, agent_id: &str) -> Option<AgentBudget> {
        let mut budgets = self.budgets.write().await;
        if let Some(budget) = budgets.get_mut(agent_id) {
            budget.replenish();
            Some(budget.clone())
        } else {
            None
        }
    }

    /// Submit a bid for a resource
    pub async fn submit_bid(&self, bid: ResourceBid) -> Result<(), String> {
        // Check budget
        let mut budgets = self.budgets.write().await;
        let budget = budgets
            .get_mut(&bid.agent_id)
            .ok_or_else(|| "Agent not registered".to_string())?;

        budget.replenish();
        if !budget.can_afford(bid.max_bid) {
            return Err(format!(
                "Insufficient budget: have {}, need {}",
                budget.available, bid.max_bid
            ));
        }

        // Add bid to auction
        let mut auctions = self.auctions.write().await;
        let auction = auctions
            .entry(bid.resource_id.clone())
            .or_insert_with(|| ResourceAuction {
                resource_id: bid.resource_id.clone(),
                bids: Vec::new(),
                current_holder: None,
                auction_start: Instant::now(),
            });

        // Remove existing bid from same agent
        auction.bids.retain(|b| b.agent_id != bid.agent_id);
        auction.bids.push(bid);

        Ok(())
    }

    /// Cancel a bid
    pub async fn cancel_bid(&self, agent_id: &str, resource_id: &str) -> bool {
        let mut auctions = self.auctions.write().await;
        if let Some(auction) = auctions.get_mut(resource_id) {
            let len_before = auction.bids.len();
            auction.bids.retain(|b| b.agent_id != agent_id);
            return auction.bids.len() < len_before;
        }
        false
    }

    /// Allocate resource to highest bidder
    pub async fn allocate(&self, resource_id: &str) -> AllocationResult {
        let mut auctions = self.auctions.write().await;
        let auction = match auctions.get_mut(resource_id) {
            Some(a) => a,
            None => return AllocationResult::NoBids,
        };

        // Check if currently held
        if let Some(ref holder) = auction.current_holder {
            let remaining = holder
                .expected_release
                .map(|r| r.saturating_duration_since(Instant::now()));
            return AllocationResult::StillHeld {
                holder: holder.agent_id.clone(),
                remaining,
            };
        }

        if auction.bids.is_empty() {
            return AllocationResult::NoBids;
        }

        // Sort by score (highest first)
        auction.bids.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Calculate price
        let price = self.calculate_price(&auction.bids);

        // Try to charge the winner
        let mut budgets = self.budgets.write().await;
        for (position, bid) in auction.bids.iter().enumerate() {
            if let Some(budget) = budgets.get_mut(&bid.agent_id) {
                budget.replenish();
                if budget.spend(price) {
                    let winner_id = bid.agent_id.clone();
                    let expected_release = Some(Instant::now() + bid.estimated_duration);

                    // Record holder
                    auction.current_holder = Some(CurrentHolder {
                        agent_id: winner_id.clone(),
                        acquired_at: Instant::now(),
                        expected_release,
                    });

                    // Record allocation
                    drop(budgets);
                    let competing_bids = auction.bids.len();
                    auction.bids.clear();
                    drop(auctions);

                    self.record_allocation(resource_id, &winner_id, price, competing_bids)
                        .await;

                    return AllocationResult::Allocated {
                        agent_id: winner_id,
                        price,
                        position,
                    };
                } else {
                    // Winner can't afford, will try next bidder
                    continue;
                }
            }
        }

        // No one could afford the price
        let first_bid = &auction.bids[0];
        AllocationResult::InsufficientBudget {
            agent_id: first_bid.agent_id.clone(),
            required: price,
            available: budgets
                .get(&first_bid.agent_id)
                .map(|b| b.available)
                .unwrap_or(0),
        }
    }

    /// Calculate price based on strategy
    fn calculate_price(&self, bids: &[ResourceBid]) -> u32 {
        match &self.pricing {
            PricingStrategy::FirstPrice => bids.first().map(|b| b.max_bid).unwrap_or(0),
            PricingStrategy::SecondPrice => {
                if bids.len() >= 2 {
                    bids[1].max_bid.min(bids[0].max_bid)
                } else {
                    1 // Minimum price
                }
            }
            PricingStrategy::FixedPrice(prices) => bids
                .first()
                .and_then(|b| prices.get(&b.resource_id))
                .copied()
                .unwrap_or(1),
            PricingStrategy::Dynamic {
                base_price,
                demand_multiplier,
            } => {
                let demand = bids.len() as f32;
                (*base_price as f32 * (1.0 + demand * demand_multiplier)) as u32
            }
            PricingStrategy::Free => 0,
        }
    }

    /// Release a resource (current holder done)
    pub async fn release(&self, resource_id: &str, agent_id: &str) -> bool {
        let mut auctions = self.auctions.write().await;
        if let Some(auction) = auctions.get_mut(resource_id)
            && let Some(ref holder) = auction.current_holder
            && holder.agent_id == agent_id
        {
            auction.current_holder = None;
            return true;
        }
        false
    }

    /// Get current market state for a resource
    pub async fn market_status(&self, resource_id: &str) -> Option<MarketStatus> {
        let auctions = self.auctions.read().await;
        auctions.get(resource_id).map(|a| MarketStatus {
            resource_id: resource_id.to_string(),
            current_holder: a.current_holder.as_ref().map(|h| h.agent_id.clone()),
            pending_bids: a.bids.len(),
            highest_score: a.bids.first().map(|b| b.score()),
            auction_age: a.auction_start.elapsed(),
        })
    }

    /// Get all active auctions
    pub async fn list_auctions(&self) -> Vec<MarketStatus> {
        let auctions = self.auctions.read().await;
        auctions
            .iter()
            .map(|(resource_id, a)| MarketStatus {
                resource_id: resource_id.clone(),
                current_holder: a.current_holder.as_ref().map(|h| h.agent_id.clone()),
                pending_bids: a.bids.len(),
                highest_score: a.bids.first().map(|b| b.score()),
                auction_age: a.auction_start.elapsed(),
            })
            .collect()
    }

    /// Record an allocation in history
    async fn record_allocation(
        &self,
        resource_id: &str,
        winner: &str,
        price: u32,
        competing_bids: usize,
    ) {
        let mut history = self.allocation_history.write().await;
        history.push(AllocationRecord {
            resource_id: resource_id.to_string(),
            winner: winner.to_string(),
            price,
            competing_bids,
            allocated_at: Instant::now(),
        });

        // Trim history
        while history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Get allocation history
    pub async fn get_history(&self) -> Vec<AllocationRecord> {
        self.allocation_history.read().await.clone()
    }

    /// Get market statistics
    pub async fn get_stats(&self) -> MarketStats {
        let history = self.allocation_history.read().await;
        let auctions = self.auctions.read().await;
        let budgets = self.budgets.read().await;

        let total_allocations = history.len();
        let total_revenue: u32 = history.iter().map(|r| r.price).sum();
        let avg_price = if total_allocations > 0 {
            total_revenue as f32 / total_allocations as f32
        } else {
            0.0
        };
        let avg_competition = if total_allocations > 0 {
            history.iter().map(|r| r.competing_bids).sum::<usize>() as f32
                / total_allocations as f32
        } else {
            0.0
        };

        MarketStats {
            active_auctions: auctions.len(),
            total_pending_bids: auctions.values().map(|a| a.bids.len()).sum(),
            registered_agents: budgets.len(),
            total_allocations,
            total_revenue,
            avg_price,
            avg_competition,
        }
    }
}

impl Default for MarketAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Status of a specific resource's market
#[derive(Debug, Clone)]
pub struct MarketStatus {
    /// Resource identifier
    pub resource_id: String,
    /// Current holder (if any)
    pub current_holder: Option<String>,
    /// Number of pending bids
    pub pending_bids: usize,
    /// Highest bid score (if any bids)
    pub highest_score: Option<f32>,
    /// How long the auction has been running
    pub auction_age: Duration,
}

/// Overall market statistics
#[derive(Debug, Clone)]
pub struct MarketStats {
    /// Number of active auctions
    pub active_auctions: usize,
    /// Total pending bids across all auctions
    pub total_pending_bids: usize,
    /// Number of registered agents
    pub registered_agents: usize,
    /// Total allocations made
    pub total_allocations: usize,
    /// Total revenue (budget points collected)
    pub total_revenue: u32,
    /// Average price per allocation
    pub avg_price: f32,
    /// Average number of competing bids
    pub avg_competition: f32,
}

/// Urgency factors for dynamic priority calculation
pub struct UrgencyCalculator;

impl UrgencyCalculator {
    /// Calculate urgency multiplier based on context
    pub fn calculate(context: &UrgencyContext) -> f32 {
        let mut multiplier = 1.0;

        // User is actively waiting
        if context.user_waiting {
            multiplier *= 2.0;
        }

        // Deadline approaching
        if let Some(deadline) = context.deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining < Duration::from_secs(60) {
                multiplier *= 3.0;
            } else if remaining < Duration::from_secs(300) {
                multiplier *= 2.0;
            } else if remaining < Duration::from_secs(600) {
                multiplier *= 1.5;
            }
        }

        // Part of critical path
        if context.critical_path {
            multiplier *= 1.5;
        }

        // Holding other resources (avoid starvation)
        multiplier *= 1.0 + (context.resources_held as f32 * 0.2);

        // Wait time factor (longer wait = higher urgency)
        if let Some(wait_time) = context.wait_time {
            let wait_secs = wait_time.as_secs();
            if wait_secs > 60 {
                multiplier *= 1.0 + (wait_secs as f32 / 120.0).min(2.0);
            }
        }

        multiplier.min(10.0) // Cap at 10x
    }

    /// Create an urgency context builder
    pub fn builder() -> UrgencyContextBuilder {
        UrgencyContextBuilder::new()
    }
}

/// Context for calculating urgency
#[derive(Debug, Clone, Default)]
pub struct UrgencyContext {
    /// User is actively waiting for the result
    pub user_waiting: bool,
    /// Deadline for the operation (if any)
    pub deadline: Option<Instant>,
    /// Operation is on the critical path
    pub critical_path: bool,
    /// Number of other resources currently held
    pub resources_held: usize,
    /// How long the agent has been waiting for this resource
    pub wait_time: Option<Duration>,
}

/// Builder for UrgencyContext
pub struct UrgencyContextBuilder {
    context: UrgencyContext,
}

impl UrgencyContextBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            context: UrgencyContext::default(),
        }
    }

    /// Set user waiting flag
    pub fn user_waiting(mut self, waiting: bool) -> Self {
        self.context.user_waiting = waiting;
        self
    }

    /// Set deadline
    pub fn deadline(mut self, deadline: Instant) -> Self {
        self.context.deadline = Some(deadline);
        self
    }

    /// Set deadline from duration
    pub fn deadline_in(mut self, duration: Duration) -> Self {
        self.context.deadline = Some(Instant::now() + duration);
        self
    }

    /// Set critical path flag
    pub fn critical_path(mut self, critical: bool) -> Self {
        self.context.critical_path = critical;
        self
    }

    /// Set resources held count
    pub fn resources_held(mut self, count: usize) -> Self {
        self.context.resources_held = count;
        self
    }

    /// Set wait time
    pub fn wait_time(mut self, duration: Duration) -> Self {
        self.context.wait_time = Some(duration);
        self
    }

    /// Build the context
    pub fn build(self) -> UrgencyContext {
        self.context
    }

    /// Calculate urgency from the built context
    pub fn calculate(self) -> f32 {
        UrgencyCalculator::calculate(&self.context)
    }
}

impl Default for UrgencyContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_registration() {
        let allocator = MarketAllocator::new();

        allocator.register_agent("agent-1", 100, 1.0).await;

        let budget = allocator.get_budget("agent-1").await.unwrap();
        assert_eq!(budget.total_budget, 100);
        assert_eq!(budget.available, 100);
    }

    #[tokio::test]
    async fn test_submit_bid() {
        let allocator = MarketAllocator::new();

        allocator.register_agent("agent-1", 100, 1.0).await;

        let bid = ResourceBid::new("agent-1", "resource-a")
            .with_priority(8)
            .with_urgency(1.5, "user waiting")
            .with_max_bid(20);

        let result = allocator.submit_bid(bid).await;
        assert!(result.is_ok());

        let status = allocator.market_status("resource-a").await.unwrap();
        assert_eq!(status.pending_bids, 1);
    }

    #[tokio::test]
    async fn test_allocation() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::Free);

        allocator.register_agent("agent-1", 100, 1.0).await;
        allocator.register_agent("agent-2", 100, 1.0).await;

        // Agent 1 bids with lower priority
        let bid1 = ResourceBid::new("agent-1", "resource-a").with_priority(5);
        allocator.submit_bid(bid1).await.unwrap();

        // Agent 2 bids with higher priority
        let bid2 = ResourceBid::new("agent-2", "resource-a").with_priority(8);
        allocator.submit_bid(bid2).await.unwrap();

        // Allocate - agent-2 should win
        let result = allocator.allocate("resource-a").await;
        match result {
            AllocationResult::Allocated { agent_id, .. } => {
                assert_eq!(agent_id, "agent-2");
            }
            _ => panic!("Expected allocation"),
        }
    }

    #[tokio::test]
    async fn test_urgency_affects_allocation() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::Free);

        allocator.register_agent("agent-1", 100, 1.0).await;
        allocator.register_agent("agent-2", 100, 1.0).await;

        // Agent 1 has higher base priority but lower urgency
        let bid1 = ResourceBid::new("agent-1", "resource-a")
            .with_priority(8)
            .with_urgency(1.0, "normal");
        allocator.submit_bid(bid1).await.unwrap();

        // Agent 2 has lower base priority but higher urgency
        let bid2 = ResourceBid::new("agent-2", "resource-a")
            .with_priority(5)
            .with_urgency(2.5, "deadline approaching");
        allocator.submit_bid(bid2).await.unwrap();

        // Agent 1 effective: 8 * 1.0 = 8
        // Agent 2 effective: 5 * 2.5 = 12.5
        // Agent 2 should win

        let result = allocator.allocate("resource-a").await;
        match result {
            AllocationResult::Allocated { agent_id, .. } => {
                assert_eq!(agent_id, "agent-2");
            }
            _ => panic!("Expected allocation"),
        }
    }

    #[tokio::test]
    async fn test_second_price_auction() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::SecondPrice);

        allocator.register_agent("agent-1", 100, 1.0).await;
        allocator.register_agent("agent-2", 100, 1.0).await;

        // Agent 1 bids 30
        let bid1 = ResourceBid::new("agent-1", "resource-a")
            .with_priority(8)
            .with_max_bid(30);
        allocator.submit_bid(bid1).await.unwrap();

        // Agent 2 bids 20
        let bid2 = ResourceBid::new("agent-2", "resource-a")
            .with_priority(5)
            .with_max_bid(20);
        allocator.submit_bid(bid2).await.unwrap();

        // Agent 1 should win but pay agent-2's bid (20)
        let result = allocator.allocate("resource-a").await;
        match result {
            AllocationResult::Allocated { price, .. } => {
                assert_eq!(price, 20);
            }
            _ => panic!("Expected allocation"),
        }
    }

    #[tokio::test]
    async fn test_insufficient_budget() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::FirstPrice);

        allocator.register_agent("agent-1", 10, 1.0).await;

        // Try to bid more than budget allows
        let bid = ResourceBid::new("agent-1", "resource-a").with_max_bid(20);
        let result = allocator.submit_bid(bid).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient budget"));
    }

    #[tokio::test]
    async fn test_release_resource() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::Free);

        allocator.register_agent("agent-1", 100, 1.0).await;

        let bid = ResourceBid::new("agent-1", "resource-a");
        allocator.submit_bid(bid).await.unwrap();
        allocator.allocate("resource-a").await;

        // Resource is held
        let status = allocator.market_status("resource-a").await.unwrap();
        assert!(status.current_holder.is_some());

        // Release
        let released = allocator.release("resource-a", "agent-1").await;
        assert!(released);

        // Resource is free
        let status = allocator.market_status("resource-a").await.unwrap();
        assert!(status.current_holder.is_none());
    }

    #[tokio::test]
    async fn test_cannot_allocate_held_resource() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::Free);

        allocator.register_agent("agent-1", 100, 1.0).await;
        allocator.register_agent("agent-2", 100, 1.0).await;

        // Agent 1 gets the resource
        let bid1 = ResourceBid::new("agent-1", "resource-a");
        allocator.submit_bid(bid1).await.unwrap();
        allocator.allocate("resource-a").await;

        // Agent 2 tries to bid and allocate
        let bid2 = ResourceBid::new("agent-2", "resource-a");
        allocator.submit_bid(bid2).await.unwrap();

        let result = allocator.allocate("resource-a").await;
        match result {
            AllocationResult::StillHeld { holder, .. } => {
                assert_eq!(holder, "agent-1");
            }
            _ => panic!("Expected StillHeld result"),
        }
    }

    #[test]
    fn test_urgency_calculator() {
        // Normal context
        let context = UrgencyContext::default();
        let urgency = UrgencyCalculator::calculate(&context);
        assert!((urgency - 1.0).abs() < 0.01);

        // User waiting
        let context = UrgencyContext {
            user_waiting: true,
            ..Default::default()
        };
        let urgency = UrgencyCalculator::calculate(&context);
        assert!((urgency - 2.0).abs() < 0.01);

        // Critical path
        let context = UrgencyContext {
            critical_path: true,
            ..Default::default()
        };
        let urgency = UrgencyCalculator::calculate(&context);
        assert!((urgency - 1.5).abs() < 0.01);

        // Both
        let context = UrgencyContext {
            user_waiting: true,
            critical_path: true,
            ..Default::default()
        };
        let urgency = UrgencyCalculator::calculate(&context);
        assert!((urgency - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_urgency_builder() {
        let urgency = UrgencyCalculator::builder()
            .user_waiting(true)
            .critical_path(true)
            .resources_held(2)
            .calculate();

        // 1.0 * 2.0 (user waiting) * 1.5 (critical) * 1.4 (2 resources held)
        // = 4.2
        assert!(urgency > 4.0 && urgency < 4.5);
    }

    #[tokio::test]
    async fn test_budget_replenishment() {
        let mut budget = AgentBudget::new("agent-1", 100).with_replenish_rate(10.0);

        // Spend some
        budget.spend(50);
        assert_eq!(budget.available, 50);

        // Simulate time passing (we can't actually wait in tests)
        // But we can verify the replenish logic
        budget.last_replenish = Instant::now() - Duration::from_secs(5);
        budget.replenish();

        // Should have replenished 50 points (5 seconds * 10 per second)
        assert_eq!(budget.available, 100); // Capped at total
    }

    #[tokio::test]
    async fn test_market_stats() {
        let allocator = MarketAllocator::with_pricing(PricingStrategy::Free);

        allocator.register_agent("agent-1", 100, 1.0).await;
        allocator.register_agent("agent-2", 100, 1.0).await;

        // Make some allocations
        for i in 0..5 {
            let bid = ResourceBid::new("agent-1", format!("resource-{}", i));
            allocator.submit_bid(bid).await.unwrap();
            allocator.allocate(&format!("resource-{}", i)).await;
        }

        let stats = allocator.get_stats().await;
        assert_eq!(stats.registered_agents, 2);
        assert_eq!(stats.total_allocations, 5);
    }

    #[test]
    fn test_bid_scoring() {
        // Higher priority = higher score
        let bid1 = ResourceBid::new("agent-1", "resource").with_priority(8);
        let bid2 = ResourceBid::new("agent-2", "resource").with_priority(5);
        assert!(bid1.score() > bid2.score());

        // Higher urgency = higher effective priority
        let bid3 = ResourceBid::new("agent-3", "resource")
            .with_priority(5)
            .with_urgency(2.0, "urgent");
        assert!(bid3.effective_priority() > bid2.effective_priority());
    }

    #[tokio::test]
    async fn test_cancel_bid() {
        let allocator = MarketAllocator::new();

        allocator.register_agent("agent-1", 100, 1.0).await;

        let bid = ResourceBid::new("agent-1", "resource-a");
        allocator.submit_bid(bid).await.unwrap();

        let status = allocator.market_status("resource-a").await.unwrap();
        assert_eq!(status.pending_bids, 1);

        let cancelled = allocator.cancel_bid("agent-1", "resource-a").await;
        assert!(cancelled);

        let status = allocator.market_status("resource-a").await.unwrap();
        assert_eq!(status.pending_bids, 0);
    }
}
