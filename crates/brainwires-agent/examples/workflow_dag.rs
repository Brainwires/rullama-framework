//! Example: DAG workflow pipelines
//!
//! Demonstrates the `WorkflowBuilder` API for constructing and executing
//! directed acyclic graph (DAG) workflows with parallel nodes and shared
//! context. No mocks or AI providers needed — workflow nodes are plain
//! async closures.
//!
//! Run: cargo run -p brainwires-agent --example workflow_dag

use brainwires_agent::{WorkflowBuilder, WorkflowContext};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 1. Build a diamond-shaped workflow (fork-join parallelism) ──────
    //
    //       fetch
    //      /     \
    //   lint    review    (run in parallel)
    //      \     /
    //     summarize
    //

    let workflow = WorkflowBuilder::new("code-review-pipeline")
        // Entry node: fetch the code
        .node("fetch", |ctx| {
            Box::pin(async move {
                println!("[fetch] Fetching source code...");
                ctx.set(
                    "code",
                    serde_json::json!("fn main() { println!(\"hello\"); }"),
                )
                .await;
                Ok(serde_json::json!({"status": "fetched", "lines": 1}))
            })
        })
        // Branch A: lint the code
        .node("lint", |ctx| {
            Box::pin(async move {
                let code = ctx.get("code").await.unwrap_or_default();
                println!(
                    "[lint] Linting code: {}...",
                    &code.to_string()[..30.min(code.to_string().len())]
                );
                let warnings = if code.to_string().contains("unwrap") {
                    1
                } else {
                    0
                };
                Ok(serde_json::json!({"warnings": warnings, "passed": true}))
            })
        })
        // Branch B: review the code
        .node("review", |ctx| {
            Box::pin(async move {
                let code = ctx.get("code").await.unwrap_or_default();
                println!("[review] Reviewing code quality...");
                let has_docs = code.to_string().contains("///");
                Ok(serde_json::json!({"quality": "good", "has_docs": has_docs}))
            })
        })
        // Join node: summarize results from both branches
        .node("summarize", |ctx| {
            Box::pin(async move {
                let lint = ctx.node_result("lint").await.unwrap_or_default();
                let review = ctx.node_result("review").await.unwrap_or_default();
                println!("[summarize] Combining lint and review results...");

                let summary = serde_json::json!({
                    "lint_warnings": lint.get("warnings"),
                    "lint_passed": lint.get("passed"),
                    "review_quality": review.get("quality"),
                    "review_has_docs": review.get("has_docs"),
                    "overall": "approved"
                });
                ctx.set("final_report", summary.clone()).await;
                Ok(summary)
            })
        })
        // Wire up the diamond edges
        .edge("fetch", "lint")
        .edge("fetch", "review")
        .edge("lint", "summarize")
        .edge("review", "summarize")
        .build()?;

    println!("=== Diamond Workflow ===");
    println!("Name: {}", workflow.name());
    println!("Entry nodes: {:?}", workflow.entry_nodes());
    println!("All nodes: {:?}\n", workflow.node_names());

    let result = workflow.run().await?;

    println!("\n--- Results ---");
    println!("Success: {}", result.success);
    println!("Nodes executed: {}", result.node_results.len());
    for (name, value) in &result.node_results {
        println!("  {name}: {value}");
    }
    if !result.skipped_nodes.is_empty() {
        println!("Skipped: {:?}", result.skipped_nodes);
    }
    if !result.failed_nodes.is_empty() {
        println!("Failed: {:?}", result.failed_nodes);
    }

    // ── 2. Workflow with pre-populated context ─────────────────────────

    println!("\n=== Pre-populated Context ===");

    let ctx = WorkflowContext::new();
    ctx.set("threshold", serde_json::json!(80)).await;

    let scoring = WorkflowBuilder::new("scoring-pipeline")
        .node("score", |ctx| {
            Box::pin(async move {
                let threshold = ctx.get("threshold").await.unwrap();
                let score = 92;
                println!("[score] Score={score}, threshold={threshold}");
                ctx.set("score", serde_json::json!(score)).await;
                Ok(serde_json::json!({"score": score, "threshold": threshold}))
            })
        })
        .node("decide", |ctx| {
            Box::pin(async move {
                let score = ctx.get("score").await.and_then(|v| v.as_i64()).unwrap_or(0);
                let threshold = ctx
                    .get("threshold")
                    .await
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let passed = score >= threshold;
                println!("[decide] {score} >= {threshold} -> {passed}");
                Ok(serde_json::json!({"passed": passed}))
            })
        })
        .edge("score", "decide")
        .build()?;

    let result = scoring.run_with_context(ctx).await?;
    println!("Decision: {}", result.node_results["decide"]);

    // ── 3. Conditional workflow ────────────────────────────────────────

    println!("\n=== Conditional Workflow ===");

    let conditional = WorkflowBuilder::new("conditional-pipeline")
        .node("check", |_| {
            Box::pin(async {
                println!("[check] Evaluating route...");
                Ok(serde_json::json!({"route": "fast"}))
            })
        })
        .node("fast_path", |_| {
            Box::pin(async {
                println!("[fast_path] Taking the fast path!");
                Ok(serde_json::json!("fast_done"))
            })
        })
        .node("slow_path", |_| {
            Box::pin(async {
                println!("[slow_path] Taking the slow path!");
                Ok(serde_json::json!("slow_done"))
            })
        })
        .edge("check", "fast_path")
        .edge("check", "slow_path")
        .conditional("check", |result| {
            let route = result
                .get("route")
                .and_then(|v| v.as_str())
                .unwrap_or("fast");
            if route == "fast" {
                vec!["fast_path".to_string()]
            } else {
                vec!["slow_path".to_string()]
            }
        })
        .build()?;

    let result = conditional.run().await?;
    println!(
        "Executed: {:?}",
        result.node_results.keys().collect::<Vec<_>>()
    );
    println!("Skipped: {:?}", result.skipped_nodes);

    println!("\nAll workflow demonstrations complete.");
    Ok(())
}
