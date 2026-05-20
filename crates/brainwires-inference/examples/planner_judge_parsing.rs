//! Example: Planner & Judge parsing API
//!
//! Demonstrates the static parsing methods on `PlannerAgent` and `JudgeAgent`
//! without any async runtime, mock providers, or network calls. This is the
//! simplest possible entry point for understanding the structured output
//! formats used in the Plan->Work->Judge cycle.
//!
//! Run: cargo run -p brainwires-inference --example planner_judge_parsing

use brainwires_inference::{JudgeAgent, JudgeVerdict, PlannerAgent, PlannerAgentConfig};

fn main() {
    // ── 1. Parse a planner output from a fenced JSON block ─────────────

    let planner_text = r#"
I've analyzed the codebase. Here is the plan:

```json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Add error handling to the parser module",
      "files_involved": ["src/parser.rs", "src/error.rs"],
      "depends_on": [],
      "priority": "high",
      "estimated_iterations": 10
    },
    {
      "id": "task-2",
      "description": "Write unit tests for the parser",
      "files_involved": ["tests/parser_test.rs"],
      "depends_on": ["task-1"],
      "priority": "normal",
      "estimated_iterations": 5
    },
    {
      "id": "task-3",
      "description": "Update documentation with new error types",
      "files_involved": ["docs/errors.md"],
      "depends_on": ["task-1"],
      "priority": "low"
    }
  ],
  "sub_planners": [
    {
      "focus_area": "Integration tests",
      "context": "Need end-to-end tests for the parser pipeline",
      "max_depth": 1
    }
  ],
  "rationale": "Parser needs robust error handling before tests can be meaningful"
}
```

This plan prioritizes correctness before documentation.
"#;

    let config = PlannerAgentConfig::default();
    let output = PlannerAgent::parse_output(planner_text, &config).unwrap();

    println!("=== Planner Output ===");
    println!("Rationale: {}", output.rationale);
    println!("Tasks ({}):", output.tasks.len());
    for task in &output.tasks {
        println!(
            "  [{:?}] {} - {} (depends on: {:?})",
            task.priority, task.id, task.description, task.depends_on
        );
        if !task.files_involved.is_empty() {
            println!("         files: {:?}", task.files_involved);
        }
    }
    println!("Sub-planners ({}):", output.sub_planners.len());
    for sp in &output.sub_planners {
        println!("  focus: {}, depth: {}", sp.focus_area, sp.max_depth);
    }

    // ── 2. Demonstrate task limit enforcement ──────────────────────────

    let strict_config = PlannerAgentConfig {
        max_tasks: 2,
        max_sub_planners: 0,
        ..Default::default()
    };
    let limited = PlannerAgent::parse_output(planner_text, &strict_config).unwrap();
    println!("\n=== With limits (max_tasks=2, max_sub_planners=0) ===");
    println!("Tasks: {} (truncated from 3)", limited.tasks.len());
    println!("Sub-planners: {}", limited.sub_planners.len());

    // ── 3. Demonstrate cycle detection ─────────────────────────────────

    let cyclic_plan = r#"```json
{
  "tasks": [
    {"id": "a", "description": "Step A", "depends_on": ["b"]},
    {"id": "b", "description": "Step B", "depends_on": ["a"]}
  ],
  "rationale": "This plan has a circular dependency"
}
```"#;

    match PlannerAgent::parse_output(cyclic_plan, &config) {
        Ok(_) => println!("\nUnexpected: cyclic plan was accepted"),
        Err(e) => println!("\n=== Cycle Detection ===\nCorrectly rejected: {}", e),
    }

    // ── 4. Parse all four judge verdict types ──────────────────────────

    println!("\n=== Judge Verdicts ===");

    // Complete
    let complete_text = r#"```json
{"verdict": "complete", "summary": "All tasks finished, tests pass, code reviewed"}
```"#;
    let verdict = JudgeAgent::parse_verdict(complete_text).unwrap();
    print_verdict(&verdict);

    // Continue
    let continue_text = r#"```json
{
  "verdict": "continue",
  "summary": "Two of three tasks done, error handling still missing",
  "additional_tasks": [
    {"id": "fix-1", "description": "Add missing error variants", "priority": "high"}
  ],
  "retry_tasks": ["task-2"],
  "hints": ["Focus on the From<io::Error> impl", "Check edge cases in parse_header"]
}
```"#;
    let verdict = JudgeAgent::parse_verdict(continue_text).unwrap();
    print_verdict(&verdict);

    // FreshRestart
    let restart_text = r#"```json
{
  "verdict": "fresh_restart",
  "reason": "Workers modified the wrong module entirely",
  "hints": ["Target src/parser.rs not src/lexer.rs"],
  "summary": "Completely off track, need to start over"
}
```"#;
    let verdict = JudgeAgent::parse_verdict(restart_text).unwrap();
    print_verdict(&verdict);

    // Abort
    let abort_text = r#"```json
{
  "verdict": "abort",
  "reason": "Goal requires access to a private API we cannot reach",
  "summary": "Impossible to proceed without API credentials"
}
```"#;
    let verdict = JudgeAgent::parse_verdict(abort_text).unwrap();
    print_verdict(&verdict);

    println!("\nAll parsing demonstrations complete.");
}

fn print_verdict(verdict: &JudgeVerdict) {
    println!("\n  Verdict type: {}", verdict.verdict_type());
    match verdict {
        JudgeVerdict::Complete { summary } => {
            println!("  Summary: {summary}");
        }
        JudgeVerdict::Continue {
            summary,
            additional_tasks,
            retry_tasks,
            hints,
        } => {
            println!("  Summary: {summary}");
            println!("  Additional tasks: {}", additional_tasks.len());
            println!("  Retry tasks: {retry_tasks:?}");
            println!("  Hints: {hints:?}");
        }
        JudgeVerdict::FreshRestart {
            reason,
            hints,
            summary,
        } => {
            println!("  Reason: {reason}");
            println!("  Summary: {summary}");
            println!("  Hints: {hints:?}");
        }
        JudgeVerdict::Abort { reason, summary } => {
            println!("  Reason: {reason}");
            println!("  Summary: {summary}");
        }
    }
}
