//! Example: Self-Improvement Strategies — listing and configuring strategies.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example self_improve_strategies --features self-improve
//! ```

use brainwires_autonomy::config::SelfImprovementConfig;
use brainwires_autonomy::self_improve::TaskGenerator;
use brainwires_autonomy::self_improve::strategies::all_strategies;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Self-Improvement Strategies Example ===\n");

    // 1. List all built-in strategies
    println!("--- Built-in Strategies ---");
    let strategies = all_strategies();
    for (i, s) in strategies.iter().enumerate() {
        println!("  {}. {} (category: {})", i + 1, s.name(), s.category());
    }
    println!("  Total: {} strategies\n", strategies.len());

    // 2. Show default configuration
    let config = SelfImprovementConfig::default();
    println!("--- Default SelfImprovementConfig ---");
    println!("  max_cycles           = {}", config.max_cycles);
    println!("  max_budget           = ${:.2}", config.max_budget);
    println!("  agent_iterations     = {}", config.agent_iterations);
    println!(
        "  max_diff_per_task    = {} lines",
        config.max_diff_per_task
    );
    println!("  max_total_diff       = {} lines", config.max_total_diff);
    println!(
        "  circuit_breaker      = {} failures",
        config.circuit_breaker_threshold
    );
    println!("  branch_prefix        = \"{}\"", config.branch_prefix);
    println!("  dry_run              = {}", config.dry_run);
    println!("  create_prs           = {}", config.create_prs);
    println!();

    // 3. Strategy filtering
    println!("--- Strategy Filtering ---");
    let enabled_only = SelfImprovementConfig {
        strategies: vec!["clippy".to_string(), "dead_code".to_string()],
        ..Default::default()
    };
    for name in &["clippy", "dead_code", "todo_scanner", "doc_gaps"] {
        println!(
            "  is_strategy_enabled(\"{name}\"): {}",
            enabled_only.is_strategy_enabled(name)
        );
    }
    println!();

    // 4. TaskGenerator with all strategies
    println!("--- Task Generator ---");
    let generator = TaskGenerator::from_config(&config);
    println!("  Active strategies: {:?}", generator.strategy_names());
    println!();

    // 5. TaskGenerator with filtered strategies
    let filtered_config = SelfImprovementConfig {
        strategies: vec!["clippy".to_string(), "todo_scanner".to_string()],
        ..Default::default()
    };
    let filtered_gen = TaskGenerator::from_config(&filtered_config);
    println!("  Filtered strategies: {:?}", filtered_gen.strategy_names());

    println!("\nDone.");
    Ok(())
}
