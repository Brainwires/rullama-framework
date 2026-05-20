//! Task Decomposition Example
//!
//! Demonstrates task decomposition strategies and MDAP cost estimation
//! using the scaling laws from the MAKER paper.
//!
//! Run with: `cargo run -p brainwires-agent --features mdap --example task_decomposition`

use brainwires_mdap::decomposition::{DecomposeContext, SequentialDecomposer, TaskDecomposer};
use brainwires_mdap::scaling::{ModelCosts, estimate_mdap};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Task Decomposition & MDAP Cost Estimation ===\n");

    // 1. Decompose a multi-step task using SequentialDecomposer
    println!("--- 1. Sequential Task Decomposition ---\n");

    let decomposer = SequentialDecomposer::new(5);
    let context = DecomposeContext::new("/home/user/project")
        .with_tools(vec![
            "read_file".to_string(),
            "write_file".to_string(),
            "bash".to_string(),
        ])
        .with_context("Rust web server project");

    let task = "\
1. Read the current handler code in src/handlers.rs
2. Add a new GET /health endpoint that returns 200 OK
3. Register the route in src/router.rs
4. Write a test for the health endpoint in tests/api.rs
5. Run cargo test to verify everything compiles";

    let result = decomposer.decompose(task, &context).await?;

    println!("  Task decomposed into {} subtasks:", result.subtasks.len());
    println!("  Is minimal: {}", result.is_minimal);
    println!("  Total complexity: {:.2}", result.total_complexity);
    println!(
        "  Composition: {}",
        result.composition_function.description()
    );
    println!();

    for subtask in &result.subtasks {
        let deps = if subtask.depends_on.is_empty() {
            "none".to_string()
        } else {
            subtask.depends_on.join(", ")
        };
        println!(
            "  [{}] {} (complexity: {:.2}, depends on: {})",
            subtask.id, subtask.description, subtask.complexity_estimate, deps,
        );
    }

    // 2. Check if simple tasks are considered minimal
    println!("\n--- 2. Minimality Check ---\n");

    let simple_task = "Return the sum of two numbers";
    let complex_task = "1. Parse the input\n2. Validate\n3. Compute\n4. Format output";

    println!(
        "  '{}' is minimal: {}",
        simple_task,
        decomposer.is_minimal(simple_task)
    );
    println!(
        "  Multi-line task is minimal: {}",
        decomposer.is_minimal(complex_task),
    );

    // 3. MDAP cost estimation with estimate_mdap
    println!("\n--- 3. MDAP Cost Estimation ---\n");

    let scenarios = vec![
        ("Simple (5 steps, p=0.95)", 5, 0.95, 0.90, 0.003, 0.95),
        ("Moderate (10 steps, p=0.85)", 10, 0.85, 0.85, 0.003, 0.95),
        ("Complex (20 steps, p=0.75)", 20, 0.75, 0.80, 0.003, 0.95),
        (
            "High-reliability (10 steps, p=0.90, t=0.99)",
            10,
            0.90,
            0.90,
            0.003,
            0.99,
        ),
    ];

    println!(
        "  {:<45} {:>5} {:>8} {:>10} {:>8}",
        "Scenario", "k", "Calls", "Cost ($)", "P(success)",
    );
    println!("  {}", "-".repeat(80));

    for (name, steps, p, v, cost, target) in &scenarios {
        let estimate = estimate_mdap(*steps, *p, *v, *cost, *target)?;
        println!(
            "  {:<45} {:>5} {:>8} {:>10.4} {:>7.1}%",
            name,
            estimate.recommended_k,
            estimate.expected_api_calls,
            estimate.expected_cost_usd,
            estimate.success_probability * 100.0,
        );
    }

    // 4. Compare model costs
    println!("\n--- 4. Model Cost Comparison ---\n");

    let models: Vec<(&str, ModelCosts)> = vec![
        ("Claude Sonnet", ModelCosts::claude_sonnet()),
        ("Claude Haiku", ModelCosts::claude_haiku()),
        ("GPT-4o", ModelCosts::gpt4o()),
        ("GPT-4o Mini", ModelCosts::gpt4o_mini()),
    ];

    let input_tokens = 500;
    let output_tokens = 200;

    println!(
        "  {:<16} {:>12} {:>12} {:>14}",
        "Model", "Input/1K", "Output/1K", "Per Call Cost",
    );
    println!("  {}", "-".repeat(58));

    for (name, costs) in &models {
        let call_cost = costs.estimate_call_cost(input_tokens, output_tokens);
        println!(
            "  {:<16} {:>11.5}$ {:>11.5}$ {:>13.6}$",
            name, costs.input_per_1k, costs.output_per_1k, call_cost,
        );
    }

    // 5. Full cost projection: model costs * MDAP scaling
    println!("\n--- 5. Full MDAP Cost Projection (10 steps, p=0.85, target=0.95) ---\n");

    let steps = 10_u64;
    let p = 0.85;
    let v = 0.90;
    let target = 0.95;

    println!(
        "  {:<16} {:>14} {:>10} {:>12}",
        "Model", "Per Call ($)", "Est. Calls", "Total ($)",
    );
    println!("  {}", "-".repeat(56));

    for (name, costs) in &models {
        let call_cost = costs.estimate_call_cost(input_tokens, output_tokens);
        let estimate = estimate_mdap(steps, p, v, call_cost, target)?;
        println!(
            "  {:<16} {:>14.6} {:>10} {:>12.4}",
            name, call_cost, estimate.expected_api_calls, estimate.expected_cost_usd,
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
