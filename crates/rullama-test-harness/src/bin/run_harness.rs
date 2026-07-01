//! `run-harness` binary — execute Tier A/B/C cases through
//! [`rullama_eval::EvaluationSuite`] and print a structured report.
//!
//! Called by `cargo xtask test-harness run`. Usage:
//!
//! ```text
//! cargo run -p rullama-test-harness --bin run-harness -- --tier=b
//! cargo run -p rullama-test-harness --bin run-harness -- --tier=all --json
//! ```

use std::process::ExitCode;

use rullama_eval::EvaluationSuite;
use rullama_test_harness::{all_cases, tier_a_suite, tier_b_suite, tier_c_suite, tier_d_suite};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    A,
    B,
    C,
    D,
    All,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut tier = Tier::All;
    let mut trials: usize = 1;
    let mut json = false;
    let mut filter: Option<String> = None;
    let mut record_baselines: Option<String> = None;

    for arg in &args {
        if let Some(v) = arg.strip_prefix("--tier=") {
            tier = match v {
                "a" | "A" => Tier::A,
                "b" | "B" => Tier::B,
                "c" | "C" => Tier::C,
                "d" | "D" => Tier::D,
                "all" => Tier::All,
                other => {
                    eprintln!("unknown tier: {other} (expected a, b, c, d, all)");
                    return ExitCode::FAILURE;
                }
            };
        } else if let Some(v) = arg.strip_prefix("--trials=") {
            trials = v.parse().unwrap_or(1).max(1);
        } else if let Some(v) = arg.strip_prefix("--filter=") {
            filter = Some(v.to_string());
        } else if arg == "--json" {
            json = true;
        } else if let Some(v) = arg.strip_prefix("--record-baselines=") {
            record_baselines = Some(v.to_string());
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
        Tier::D => tier_d_suite(),
        Tier::All => all_cases(),
    };

    if let Some(f) = &filter {
        cases.retain(|c| c.name().contains(f.as_str()));
    }

    if cases.is_empty() {
        eprintln!("no cases matched (tier={tier:?}, filter={filter:?})");
        return ExitCode::FAILURE;
    }

    // Multi-threaded so cases that depend on `tokio::task::block_in_place`
    // (e.g. the SQLite analytics sink) work. Current-thread runtime would
    // panic on those even when only a single test invokes them.
    let runtime = tokio::runtime::Builder::new_multi_thread()
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

    if let Some(path) = &record_baselines {
        let mut suite = rullama_eval::RegressionSuite::new();
        suite.record_baselines(&result);
        match suite.baselines_to_json() {
            Ok(s) => {
                if let Err(e) = std::fs::write(path, s) {
                    eprintln!("failed to write baselines to {path}: {e}");
                    return ExitCode::FAILURE;
                }
                eprintln!("baselines written to {path}");
            }
            Err(e) => {
                eprintln!("failed to serialise baselines: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let any_failed = result.stats.values().any(|s| s.success_rate < 1.0);
    if any_failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn print_help() {
    println!("Usage: run-harness [--tier=a|b|c|d|all] [--trials=N] [--filter=substring] [--json]");
    println!();
    println!("Run rullama test-harness cases and report results.");
    println!();
    println!("Flags:");
    println!("  --tier=<x>     Restrict to one tier (default: all = A+B+C; D is opt-in)");
    println!("  --trials=<N>   Trials per case (default: 1)");
    println!("  --filter=<s>   Only cases whose name contains <s>");
    println!("  --json         Emit a single-line JSON report instead of human output");
    println!("  --record-baselines=PATH  Write a rullama-eval RegressionSuite baseline file");
}

fn print_human(result: &rullama_eval::SuiteResult) {
    println!("=== rullama test-harness ===");
    println!(
        "{} case(s), overall success_rate={:.3}",
        result.stats.len(),
        result.overall_success_rate()
    );
    let mut names: Vec<&String> = result.stats.keys().collect();
    names.sort();
    for name in names {
        let s = &result.stats[name];
        let trials = result.case_results.get(name);
        let all_skipped = trials
            .map(|ts| !ts.is_empty() && ts.iter().all(|t| t.skipped))
            .unwrap_or(false);
        let mark = if all_skipped {
            "SKIP"
        } else if s.success_rate >= 1.0 {
            "PASS"
        } else {
            "FAIL"
        };
        if all_skipped {
            let reason = trials
                .and_then(|ts| ts.first())
                .and_then(|t| t.error.as_deref())
                .unwrap_or("no live env vars");
            println!("  {mark}  {name}  ({reason})");
        } else {
            println!(
                "  {mark}  {name}  ({:.0}%  n={}  CI=[{:.3}, {:.3}])",
                s.success_rate * 100.0,
                s.n_trials,
                s.confidence_interval_95.lower,
                s.confidence_interval_95.upper,
            );
        }
        if let Some(trials) = trials {
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

fn print_json(result: &rullama_eval::SuiteResult) {
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
