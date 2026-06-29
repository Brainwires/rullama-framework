//! Example: Health Monitor — agent health tracking and degradation detection.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example health_monitor
//! ```

use std::time::Instant;

use brainwires_autonomy::agent_ops::{
    HealthMonitor, PerformanceMetrics, health::HealthMonitorConfig,
};

fn main() {
    println!("=== Health Monitor Example ===\n");

    // 1. Create a health monitor with custom thresholds
    let config = HealthMonitorConfig {
        slow_iteration_threshold_ms: 5_000,
        error_rate_threshold: 0.25,
        heartbeat_timeout_secs: 120,
        stall_timeout_secs: 300,
    };
    let mut monitor = HealthMonitor::new(config);

    println!("HealthMonitor created with:");
    println!("  error_rate_threshold = 25%");
    println!("  heartbeat_timeout    = 120s");
    println!();

    // 2. Register agents
    monitor.register("agent-alpha");
    monitor.register("agent-beta");
    monitor.register("agent-gamma");

    println!("Registered 3 agents:");
    for id in &["agent-alpha", "agent-beta", "agent-gamma"] {
        println!("  {id}: {:?}", monitor.status(id));
    }
    println!();

    // 3. Update metrics for each agent
    println!("--- Updating Metrics ---");

    // Agent Alpha: healthy, good performance
    let alpha_metrics = PerformanceMetrics {
        iterations: 10,
        total_tokens: 5_000,
        total_cost: 0.50,
        errors: 1,
        tool_calls: 40,
        files_modified: 3,
        last_activity: Some(Instant::now()),
    };
    monitor.update_metrics("agent-alpha", alpha_metrics.clone());
    monitor.heartbeat("agent-alpha");
    println!(
        "  agent-alpha: {} iterations, error_rate={:.1}%, tokens/iter={}",
        alpha_metrics.iterations,
        alpha_metrics.error_rate() * 100.0,
        alpha_metrics.avg_tokens_per_iteration(),
    );

    // Agent Beta: degraded, high error rate
    let beta_metrics = PerformanceMetrics {
        iterations: 8,
        total_tokens: 12_000,
        total_cost: 1.20,
        errors: 5,
        tool_calls: 15,
        files_modified: 1,
        last_activity: Some(Instant::now()),
    };
    monitor.update_metrics("agent-beta", beta_metrics.clone());
    monitor.heartbeat("agent-beta");
    println!(
        "  agent-beta:  {} iterations, error_rate={:.1}%, tokens/iter={}",
        beta_metrics.iterations,
        beta_metrics.error_rate() * 100.0,
        beta_metrics.avg_tokens_per_iteration(),
    );

    // Agent Gamma: no heartbeat, no activity update (will be Unknown)
    println!("  agent-gamma: no heartbeat sent (simulating unresponsive agent)");
    println!();

    // 4. Evaluate all agents
    println!("--- Evaluation Results ---");
    let degraded = monitor.evaluate_all();

    if degraded.is_empty() {
        println!("  All agents are healthy.");
    } else {
        for (id, status, signals) in &degraded {
            println!("  {id}: {:?}", status);
            for signal in signals {
                println!("    Signal: {signal:?}");
            }
        }
    }
    println!();

    // 5. Final status check
    println!("--- Final Status ---");
    for id in &["agent-alpha", "agent-beta", "agent-gamma"] {
        println!("  {id}: {:?}", monitor.status(id));
    }

    // 6. Unregister completed agent
    monitor.unregister("agent-alpha");
    println!("\n  Unregistered agent-alpha");
    println!("  agent-alpha status: {:?}", monitor.status("agent-alpha"));

    println!("\nDone.");
}
