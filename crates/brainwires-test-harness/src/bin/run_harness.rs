//! `run-harness` binary — execute Tier A/B/C cases through
//! [`brainwires_eval::EvaluationSuite`] and print a structured report.
//!
//! Called by `cargo xtask test-harness run`. Usage:
//!
//! ```text
//! cargo run -p brainwires-test-harness --bin run-harness -- --tier=b
//! cargo run -p brainwires-test-harness --bin run-harness -- --tier=all --json
//! ```

use std::process::ExitCode;

use brainwires_eval::EvaluationSuite;
use brainwires_test_harness::{all_cases, tier_a_suite, tier_b_suite, tier_c_suite};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    A,
    B,
    C,
    All,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut tier = Tier::All;
    let mut trials: usize = 1;
    let mut json = false;
    let mut filter: Option<String> = None;

    for arg in &args {
        if let Some(v) = arg.strip_prefix("--tier=") {
            tier = match v {
                "a" | "A" => Tier::A,
                "b" | "B" => Tier::B,
                "c" | "C" => Tier::C,
                "all" => Tier::All,
                other => {
                    eprintln!("unknown tier: {other} (expected a, b, c, all)");
                    return ExitCode::FAILURE;
                }
            };
        } else if let Some(v) = arg.strip_prefix("--trials=") {
            trials = v.parse().unwrap_or(1).max(1);
        } else if let Some(v) = arg.strip_prefix("--filter=") {
            filter = Some(v.to_string());
        } else if arg == "--json" {
            json = true;
        } else if arg == "--help" || arg == "-h" {
            print_help();
            return ExitCode::SUCCESS;
        } else {
            eprintln!("unknown argument: {arg}");
            return ExitCode::FAILURE;
        }
    }

    let mut cases = match tier {
        Tier::A => tier_a_suite(),
        Tier::B => tier_b_suite(),
        Tier::C => tier_c_suite(),
        Tier::All => all_cases(),
    };

    if let Some(f) = &filter {
        cases.retain(|c| c.name().contains(f.as_str()));
    }

    if cases.is_empty() {
        eprintln!("no cases matched (tier={tier:?}, filter={filter:?})");
        return ExitCode::FAILURE;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let suite = EvaluationSuite::new(trials);
    let result = runtime.block_on(suite.run_suite(&cases));

    if json {
        print_json(&result);
    } else {
        print_human(&result);
    }

    let any_failed = result
        .stats
        .values()
        .any(|s| s.success_rate < 1.0);
    if any_failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn print_help() {
    println!("Usage: run-harness [--tier=a|b|c|all] [--trials=N] [--filter=substring] [--json]");
    println!();
    println!("Run brainwires test-harness cases and report results.");
    println!();
    println!("Flags:");
    println!("  --tier=<x>     Restrict to one tier (default: all)");
    println!("  --trials=<N>   Trials per case (default: 1)");
    println!("  --filter=<s>   Only cases whose name contains <s>");
    println!("  --json         Emit a single-line JSON report instead of human output");
}

fn print_human(result: &brainwires_eval::SuiteResult) {
    println!("=== brainwires test-harness ===");
    println!(
        "{} case(s), overall success_rate={:.3}",
        result.stats.len(),
        result.overall_success_rate()
    );
    let mut names: Vec<&String> = result.stats.keys().collect();
    names.sort();
    for name in names {
        let s = &result.stats[name];
        let mark = if s.success_rate >= 1.0 { "PASS" } else { "FAIL" };
        println!(
            "  {mark}  {name}  ({:.0}%  n={}  CI=[{:.3}, {:.3}])",
            s.success_rate * 100.0,
            s.n_trials,
            s.confidence_interval_95.lower,
            s.confidence_interval_95.upper,
        );
        if let Some(trials) = result.case_results.get(name) {
            for trial in trials {
                if !trial.success {
                    println!(
                        "      └─ trial {} failed: {}",
                        trial.trial_id,
                        trial.error.as_deref().unwrap_or("(no error message)")
                    );
                }
            }
        }
    }
}

fn print_json(result: &brainwires_eval::SuiteResult) {
    // Minimal hand-rolled JSON to avoid pulling another dep. Each case
    // becomes one object in an array.
    print!("{{\"cases\":[");
    let mut first = true;
    let mut names: Vec<&String> = result.stats.keys().collect();
    names.sort();
    for name in names {
        if !first {
            print!(",");
        }
        first = false;
        let s = &result.stats[name];
        print!(
            "{{\"name\":\"{}\",\"success_rate\":{:.6},\"n\":{},\"ci_lo\":{:.6},\"ci_hi\":{:.6}}}",
            name.replace('"', "\\\""),
            s.success_rate,
            s.n_trials,
            s.confidence_interval_95.lower,
            s.confidence_interval_95.upper,
        );
    }
    println!("]}}");
}
