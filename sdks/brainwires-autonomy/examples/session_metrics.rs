//! Example: Session Metrics — tracking and reporting improvement sessions.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example session_metrics
//! ```

use std::time::Duration;

use brainwires_autonomy::metrics::{ComparisonResult, SessionMetrics, SessionReport};

fn main() {
    println!("=== Session Metrics & Reporting Example ===\n");

    // 1. Build up session metrics
    let mut metrics = SessionMetrics::new();

    // Simulate a session with multiple strategies
    let strategies = [
        ("clippy", 3, 2, vec![5, 8]),
        ("dead_code", 2, 1, vec![12]),
        ("doc_gaps", 4, 3, vec![3, 6, 4]),
        ("refactoring", 1, 0, vec![]),
    ];

    println!("--- Recording Tasks ---");
    for (name, generated, succeeded, iters) in &strategies {
        metrics.record_generated(name, *generated);
        for _ in 0..*generated {
            metrics.record_attempt(name);
        }
        for &iter_count in iters.iter() {
            metrics.record_success(name, iter_count);
        }
        let failed = *generated - *succeeded;
        for _ in 0..failed {
            metrics.record_failure(name);
        }
        println!(
            "  {name}: generated={generated}, succeeded={succeeded}, failed={}",
            generated - succeeded
        );
    }

    // Record commits
    metrics.record_commit("a1b2c3d".to_string());
    metrics.record_commit("e4f5g6h".to_string());
    metrics.record_commit("i7j8k9l".to_string());

    // Record a dual-path comparison
    metrics.record_comparison(ComparisonResult {
        both_succeeded: true,
        both_failed: false,
        diffs_match: true,
        iteration_delta: -3,
        bridge_specific_errors: vec![],
    });
    metrics.record_comparison(ComparisonResult {
        both_succeeded: false,
        both_failed: false,
        diffs_match: false,
        iteration_delta: 5,
        bridge_specific_errors: vec!["timeout on bridge path".to_string()],
    });

    println!();

    // 2. Print summary
    println!("--- Summary ---");
    println!("  Tasks attempted  : {}", metrics.tasks_attempted);
    println!("  Tasks succeeded  : {}", metrics.tasks_succeeded);
    println!("  Tasks failed     : {}", metrics.tasks_failed);
    println!("  Total iterations : {}", metrics.total_iterations);
    println!(
        "  Success rate     : {:.1}%",
        metrics.success_rate() * 100.0
    );
    println!("  Commits          : {}", metrics.commits.len());
    println!("  Comparisons      : {}", metrics.comparisons.len());
    println!();

    // 3. Generate a session report
    let report = SessionReport::new(
        metrics,
        Duration::from_secs(185),
        None, // no safety stop
    );

    // 4. JSON output
    println!("--- JSON Report (truncated) ---");
    let json = report.to_json().unwrap();
    // Print first 500 chars
    let preview: String = json.chars().take(500).collect();
    println!("{preview}...\n");

    // 5. Markdown output
    println!("--- Markdown Report ---");
    let md = report.to_markdown();
    println!("{md}");

    println!("Done.");
}
