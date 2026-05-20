//! Example: Contract-Net bidding protocol for task allocation
//!
//! Shows how to create task announcements with requirements, simulate agents
//! submitting bids with different profiles, score bids, and evaluate winners
//! under different `BidEvaluationStrategy` variants.
//!
//! Run: cargo run -p brainwires-agent --example contract_net

use std::time::{Duration, Instant};

use anyhow::Result;

use brainwires_agent::contract_net::{
    BidEvaluationStrategy, ContractNetManager, TaskAnnouncement, TaskBid, TaskRequirements,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Contract-Net Bidding Protocol ===\n");

    // 1. Create a task announcement with requirements
    let requirements = TaskRequirements::new()
        .with_capabilities(vec!["rust".to_string(), "testing".to_string()])
        .with_complexity(5)
        .with_priority(3);

    let announcement = TaskAnnouncement::new(
        "task-refactor",
        "Refactor error handling module with full test coverage",
        "orchestrator",
        Instant::now() + Duration::from_secs(30),
    )
    .with_requirements(requirements);

    println!("Task announced: {}", announcement.description);
    println!(
        "  Requirements: {:?}, complexity={}",
        announcement.requirements.capabilities, announcement.requirements.complexity
    );
    println!("  Bidding open: {}\n", announcement.is_bidding_open());

    // 2. Simulate three agents submitting bids
    let bid_alpha = TaskBid::new("agent-alpha", "task-refactor")
        .with_capability_score(0.95) // Rust expert, great at testing
        .with_load(0.6) // Moderately busy
        .with_duration(Duration::from_secs(300)); // 5 minutes

    let bid_beta = TaskBid::new("agent-beta", "task-refactor")
        .with_capability_score(0.70) // Decent Rust, less testing experience
        .with_load(0.1) // Nearly idle
        .with_duration(Duration::from_secs(180)); // 3 minutes

    let bid_gamma = TaskBid::new("agent-gamma", "task-refactor")
        .with_capability_score(0.85) // Strong Rust + testing
        .with_load(0.3) // Light load
        .with_duration(Duration::from_secs(240)); // 4 minutes

    let bids = [&bid_alpha, &bid_beta, &bid_gamma];

    // 3. Show bid scores (default weighted: 40% capability, 30% availability, 30% speed)
    println!("--- Bid Scores (default weights) ---");
    for bid in &bids {
        println!(
            "  {}: score={:.3}  (capability={:.2}, load={:.2}, duration={}s)",
            bid.agent_id,
            bid.score(),
            bid.capability_score,
            bid.current_load,
            bid.estimated_duration.as_secs(),
        );
    }
    println!();

    // 4. Evaluate winners under different strategies
    let strategies: Vec<(&str, BidEvaluationStrategy)> = vec![
        ("HighestScore", BidEvaluationStrategy::HighestScore),
        (
            "FastestCompletion",
            BidEvaluationStrategy::FastestCompletion,
        ),
        ("LoadBalancing", BidEvaluationStrategy::LoadBalancing),
        ("BestCapability", BidEvaluationStrategy::BestCapability),
    ];

    println!("--- Winners by Strategy ---");
    for (label, strategy) in strategies {
        let manager = ContractNetManager::with_strategy(strategy);

        // Announce the task
        let task_id = manager
            .announce_task(TaskAnnouncement::new(
                "",
                "Refactor error handling",
                "orchestrator",
                Instant::now() + Duration::from_secs(30),
            ))
            .await;

        // Submit all bids (re-create with correct task_id)
        for bid in &bids {
            let new_bid = TaskBid::new(&bid.agent_id, &task_id)
                .with_capability_score(bid.capability_score)
                .with_load(bid.current_load)
                .with_duration(bid.estimated_duration);
            manager
                .receive_bid(new_bid)
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        // Award the task
        let winner = manager.award_task(&task_id).await;
        println!(
            "  {:<20} => winner: {}",
            label,
            winner.unwrap_or_else(|| "none".into())
        );
    }
    println!();

    // 5. Full lifecycle with the default manager
    println!("--- Full Lifecycle Demo ---");
    let manager = ContractNetManager::new();

    let task_id = manager
        .announce_task(TaskAnnouncement::new(
            "lifecycle-task",
            "Build auth module",
            "orchestrator",
            Instant::now() + Duration::from_secs(30),
        ))
        .await;

    manager
        .receive_bid(
            TaskBid::new("agent-alpha", &task_id)
                .with_capability_score(0.9)
                .with_load(0.2),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    let winner = manager.award_task(&task_id).await.unwrap();
    println!("  Awarded to: {winner}");

    manager
        .accept_award(&task_id, &winner)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;
    let status = manager.get_task_status(&task_id).await;
    println!("  Status after accept: {:?}", status);

    manager
        .complete_task(&task_id, &winner, true, Some("Auth module ready".into()))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;
    let status = manager.get_task_status(&task_id).await;
    println!("  Status after complete: {:?}", status);

    println!("\nContract-Net demo complete.");
    Ok(())
}
