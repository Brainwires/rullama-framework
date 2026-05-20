use anyhow::Result;
use std::sync::Arc;

use brainwires_eval::EvaluationCase;

use crate::self_improve::{
    AutonomousFeedbackLoop, FeedbackLoopConfig, SelfImprovementConfig, SelfImprovementController,
};

#[allow(clippy::too_many_arguments)]
pub async fn handle_eval_improve(
    baselines_path: String,
    max_rounds: u32,
    n_trials: usize,
    improvement_threshold: f64,
    auto_update_baselines: bool,
    commit_baselines: bool,
    dry_run: bool,
    max_budget: f64,
    no_bridge: bool,
    no_direct: bool,
) -> Result<()> {
    // Use the standard long-horizon stability suite as default eval cases.
    // Contributors can extend this by passing a custom case set programmatically.
    let cases: Vec<Arc<dyn EvaluationCase>> = brainwires_eval::long_horizon_stability_suite();

    let self_improve_config = SelfImprovementConfig {
        max_cycles: 10,
        max_budget,
        dry_run,
        strategies: Vec::new(),
        agent_iterations: 25,
        max_diff_per_task: 200,
        max_total_diff: 2000,
        create_prs: false,
        branch_prefix: "eval-improve/".to_string(),
        no_bridge,
        no_direct,
        model: None,
        provider: None,
        circuit_breaker_threshold: 3,
    };

    if dry_run {
        // Dry-run: just print the detected faults and exit.
        use brainwires_eval::fault_report::analyze_suite_for_faults;
        use brainwires_eval::{EvaluationSuite, RegressionSuite, SuiteConfig};

        println!("\n=== Eval-Improve Dry Run ===\n");
        println!(
            "Running {} eval cases ({} trials each)…\n",
            cases.len(),
            n_trials
        );

        let suite = EvaluationSuite::with_config(SuiteConfig {
            n_trials,
            ..SuiteConfig::default()
        });
        let result = suite.run_suite(&cases).await;

        let regression_suite = std::fs::read_to_string(&baselines_path)
            .ok()
            .and_then(|json| RegressionSuite::load_baselines_from_json(&json).ok());

        let faults = analyze_suite_for_faults(&result, regression_suite.as_ref(), 0.2, 0.25);

        if faults.is_empty() {
            println!("✅ No faults detected — nothing to improve.");
        } else {
            println!("Detected {} fault(s):\n", faults.len());
            for (i, fault) in faults.iter().enumerate() {
                println!(
                    "  {}. [P{}] [{}] {}",
                    i + 1,
                    fault.priority(),
                    fault.fault_kind.label(),
                    fault
                        .suggested_task_description
                        .chars()
                        .take(100)
                        .collect::<String>()
                );
            }
        }

        println!("\nBaselines path: {baselines_path}");
        return Ok(());
    }

    let config = FeedbackLoopConfig {
        self_improve: self_improve_config,
        baselines_path,
        auto_update_baselines,
        improvement_threshold,
        max_feedback_rounds: max_rounds,
        n_eval_trials: n_trials,
        commit_baselines,
        consistent_failure_threshold: 0.2,
        flaky_ci_threshold: 0.25,
    };

    let lp = AutonomousFeedbackLoop::new(config, cases);
    let report = lp.run().await?;

    println!("{}", report.to_markdown());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_self_improve(
    max_cycles: u32,
    max_budget: f64,
    dry_run: bool,
    strategies: Option<String>,
    agent_iterations: u32,
    max_diff: u32,
    create_prs: bool,
    branch_prefix: String,
    no_bridge: bool,
    no_direct: bool,
    model: Option<String>,
    provider: Option<String>,
) -> Result<()> {
    let strategies_list = strategies
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let config = SelfImprovementConfig {
        max_cycles,
        max_budget,
        dry_run,
        strategies: strategies_list,
        agent_iterations,
        max_diff_per_task: max_diff,
        max_total_diff: max_diff * max_cycles,
        create_prs,
        branch_prefix,
        no_bridge,
        no_direct,
        model,
        provider,
        circuit_breaker_threshold: 3,
    };

    let mut controller = SelfImprovementController::new(config);
    let report = controller.run().await?;

    // Print summary
    println!("{}", report.to_markdown());

    Ok(())
}
