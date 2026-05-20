//! SafetyGuard circuit breakers and budget tracking.
//!
//! ```bash
//! cargo run --example safety_guard
//! ```

use std::sync::Arc;

use brainwires_autonomy::{
    SessionMetrics,
    config::SafetyConfig,
    safety::{AlwaysApprove, ApprovalPolicy, AutonomousOperation, SafetyGuard, SafetyStop},
};

// ── Custom approval policy ──────────────────────────────────────────────────

/// Policy that rejects operations estimated to cost more than $2.
struct CostCapPolicy {
    max_cost: f64,
}

#[async_trait::async_trait]
impl ApprovalPolicy for CostCapPolicy {
    async fn check(&self, op: &AutonomousOperation) -> Result<(), SafetyStop> {
        if let AutonomousOperation::StartImprovement { estimated_cost, .. } = op
            && *estimated_cost > self.max_cost
        {
            return Err(SafetyStop::OperationRejected(format!(
                "Estimated cost ${estimated_cost:.2} exceeds cap ${:.2}",
                self.max_cost
            )));
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Setup — build SafetyConfig with tight limits for the demo
    println!("=== SafetyGuard Example ===\n");

    let config = SafetyConfig {
        max_total_cost: 10.0,
        max_per_operation_cost: 3.0,
        max_daily_operations: 50,
        circuit_breaker_threshold: 3,
        circuit_breaker_cooldown_secs: 60,
        max_diff_per_task: 100,
        max_total_diff: 500,
        max_concurrent_agents: 4,
        heartbeat_timeout_secs: 1800,
        allowed_paths: vec!["src/**".to_string()],
        forbidden_paths: vec!["src/secrets/**".to_string()],
    };

    println!("SafetyConfig created:");
    println!("  max_total_cost     = ${:.2}", config.max_total_cost);
    println!(
        "  circuit_breaker    = {} failures",
        config.circuit_breaker_threshold
    );
    println!("  max_daily_ops      = {}", config.max_daily_operations);
    println!();

    // 2. Create SafetyGuard from config (max 5 cycles)
    let max_cycles = 5;
    let guard = SafetyGuard::from_config(&config, max_cycles);

    // Attach a custom approval policy
    let policy = Arc::new(CostCapPolicy { max_cost: 2.0 });
    let mut guard = guard.with_approval_policy(policy);

    println!("SafetyGuard created with max_cycles={max_cycles}\n");

    // 3. Approval policy checks
    println!("--- Approval Policy ---");

    let cheap_op = AutonomousOperation::StartImprovement {
        strategy: "clippy-fixes".to_string(),
        estimated_cost: 1.50,
    };
    match guard.check_approval(&cheap_op).await {
        Ok(()) => println!("  Approved: {cheap_op}"),
        Err(e) => println!("  Rejected: {e}"),
    }

    let expensive_op = AutonomousOperation::StartImprovement {
        strategy: "full-refactor".to_string(),
        estimated_cost: 5.00,
    };
    match guard.check_approval(&expensive_op).await {
        Ok(()) => println!("  Approved: {expensive_op}"),
        Err(e) => println!("  Rejected: {e}"),
    }
    println!();

    // 4. Record successes and check can_continue
    println!("--- Cycle Tracking ---");

    for i in 1..=max_cycles {
        match guard.check_can_continue() {
            Ok(()) => {
                guard.heartbeat();
                guard.record_success(20); // 20 diff lines per cycle
                guard.record_cost(1.25);
                println!("  Cycle {i}: success (diff=20, cost=$1.25)");
            }
            Err(stop) => {
                println!("  Cycle {i}: stopped — {stop}");
                break;
            }
        }
    }

    // One more attempt should fail — cycle limit reached
    match guard.check_can_continue() {
        Ok(()) => println!("  Extra cycle: allowed (unexpected)"),
        Err(stop) => println!("  Extra cycle: stopped — {stop}"),
    }

    println!("\n  Cycles completed : {}", guard.cycles_completed());
    println!("  Total cost       : ${:.2}", guard.total_cost());
    println!("  Total diff lines : {}", guard.total_diff_lines());
    println!();

    // 5. Circuit breaker demo (fresh guard)
    println!("--- Circuit Breaker ---");

    let mut guard2 =
        SafetyGuard::from_config(&config, 20).with_approval_policy(Arc::new(AlwaysApprove));

    println!("  State: {:?}", guard2.circuit_breaker_state());

    for i in 1..=4 {
        guard2.record_failure();
        println!("  After failure {i}: {:?}", guard2.circuit_breaker_state());
    }

    match guard2.check_can_continue() {
        Ok(()) => println!("  check_can_continue: OK"),
        Err(stop) => println!("  check_can_continue: {stop}"),
    }
    println!();

    // 6. SessionMetrics summary
    println!("--- Session Metrics ---");

    let mut metrics = SessionMetrics::new();

    metrics.record_attempt("clippy");
    metrics.record_attempt("clippy");
    metrics.record_attempt("dead_code");
    metrics.record_attempt("dead_code");
    metrics.record_attempt("dead_code");

    metrics.record_success("clippy", 5);
    metrics.record_success("clippy", 8);
    metrics.record_success("dead_code", 12);

    metrics.record_failure("dead_code");
    metrics.record_failure("dead_code");

    metrics.record_commit("abc1234".to_string());
    metrics.record_commit("def5678".to_string());

    println!("  Tasks attempted  : {}", metrics.tasks_attempted);
    println!("  Tasks succeeded  : {}", metrics.tasks_succeeded);
    println!("  Tasks failed     : {}", metrics.tasks_failed);
    println!("  Total iterations : {}", metrics.total_iterations);
    println!(
        "  Success rate     : {:.0}%",
        metrics.success_rate() * 100.0
    );
    println!("  Commits          : {:?}", metrics.commits);

    println!("\n  Per-strategy breakdown:");
    for (name, stats) in &metrics.per_strategy {
        println!(
            "    {name}: attempted={}, succeeded={}",
            stats.tasks_attempted, stats.tasks_succeeded
        );
    }

    println!("\nDone.");
    Ok(())
}
