use std::env;
use std::process::{Command, ExitCode};

mod lint_deps;
mod package_count;
mod stubs;
mod version;

struct Step {
    key: &'static str,
    name: &'static str,
    cmd: &'static [&'static str],
}

struct CiOptions {
    fix: bool,
    max_turns: u32,
}

enum FixOutcome {
    AutoFixed,
    ClaudeFixed,
    ClaudeFailed(String),
    ClaudeUnavailable,
}

const STEPS: &[Step] = &[
    Step {
        key: "fmt",
        name: "Format",
        cmd: &["cargo", "fmt", "--all", "--check"],
    },
    Step {
        key: "check",
        name: "Check",
        cmd: &["cargo", "check", "--workspace"],
    },
    Step {
        key: "clippy",
        name: "Clippy",
        cmd: &["cargo", "clippy", "--workspace", "--", "-D", "warnings"],
    },
    Step {
        key: "test",
        name: "Test",
        cmd: &["cargo", "test", "--workspace"],
    },
    Step {
        key: "doc",
        name: "Doc",
        cmd: &["cargo", "doc", "--workspace", "--no-deps"],
    },
];

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();

    // Dispatch subcommands
    match args.first().map(|s| s.as_str()) {
        Some("bump-version") => return version::bump_version(&args[1..]),
        Some("check-stubs") => return stubs::check_stubs(&args[1..]),
        Some("lint-deps") => return lint_deps::lint_deps(&args[1..]),
        Some("package-count") => return package_count::update_package_count(&args[1..]),
        Some("--help" | "-h") => {
            print_help();
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    // Default: CI mode
    run_ci(&args)
}

fn print_help() {
    println!("Usage: cargo xtask <command>");
    println!();
    println!("Commands:");
    println!(
        "  bump-version <VERSION> [--crates a,b]  Bump versions (patch=selective, minor/major=all)"
    );
    println!("  check-stubs             Scan for unfinished code (todo!(), FIXME, etc.)");
    println!("  lint-deps               Enforce framework/extras boundary (ADR-0004)");
    println!("  package-count [--dry-run]  Update crate/extras count references in .md files");
    println!("  [step ...]              Run CI steps: fmt, check, clippy, test, doc");
    println!();
    println!("Flags (CI mode):");
    println!("  --fix                   Auto-fix failures (cargo fmt + Claude Code CLI)");
    println!("  --max-turns <N>         Max conversation turns per fix attempt (default: 30)");
    println!();
    println!("Run with no arguments to execute all CI steps.");
}

fn parse_ci_options(args: &[String]) -> (CiOptions, Vec<String>) {
    let mut fix = false;
    let mut max_turns = 30u32;
    let mut step_args: Vec<String> = Vec::new();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--fix" => fix = true,
            "--max-turns" => {
                if let Some(val) = iter.next() {
                    max_turns = val.parse().unwrap_or(30);
                }
            }
            _ => step_args.push(arg.clone()),
        }
    }

    (CiOptions { fix, max_turns }, step_args)
}

fn claude_available() -> bool {
    Command::new("claude")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn truncate_output(output: &str, max_bytes: usize) -> &str {
    if output.len() <= max_bytes {
        return output;
    }
    // Take the tail — errors are at the end
    let start = output.len() - max_bytes;
    // Find the next newline to avoid cutting mid-line
    match output[start..].find('\n') {
        Some(pos) => &output[start + pos + 1..],
        None => &output[start..],
    }
}

fn build_fix_prompt(step: &Step, error_output: &str) -> String {
    let cwd = env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let cmd_str = step.cmd.join(" ");
    let truncated = truncate_output(error_output, 8000);

    format!(
        r#"You are fixing a CI failure in a Rust workspace.

Step: {name}
Command: {cmd}
Working directory: {cwd}

The command failed with the following output:

```
{output}
```

Fix the underlying source code so that the command succeeds.

Rules:
- Do NOT suppress warnings with #[allow(...)]; fix the root cause.
- For clippy lints, fix the code to satisfy the lint.
- For test failures, investigate actual vs expected values and fix the logic.
- For compilation errors, fix type errors, missing imports, etc.
- Do NOT modify test expectations unless the test itself is wrong.
- Do NOT add, remove, or rename public API items unless necessary to fix the error."#,
        name = step.name,
        cmd = cmd_str,
        cwd = cwd,
        output = truncated,
    )
}

fn try_fix(step: &Step, error_output: &str, options: &CiOptions) -> FixOutcome {
    // Special case: fmt just needs cargo fmt --all
    if step.key == "fmt" {
        println!("  Running cargo fmt --all ...");
        let fmt_status = Command::new("cargo").args(["fmt", "--all"]).status();
        if fmt_status.is_ok_and(|s| s.success()) {
            // Re-verify
            let verify = Command::new(step.cmd[0])
                .args(&step.cmd[1..])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            if verify.is_ok_and(|s| s.success()) {
                return FixOutcome::AutoFixed;
            }
        }
        return FixOutcome::ClaudeFailed("cargo fmt did not resolve all issues".into());
    }

    // Check if claude is available
    if !claude_available() {
        return FixOutcome::ClaudeUnavailable;
    }

    let prompt = build_fix_prompt(step, error_output);

    println!("  Invoking Claude Code to fix {} ...", step.name);

    let claude_result = Command::new("claude")
        .arg("-p")
        .arg(&prompt)
        .arg("--output-format")
        .arg("json")
        .arg("--max-turns")
        .arg(options.max_turns.to_string())
        .arg("--allowedTools")
        .arg("Read,Edit,Glob,Grep,Bash(cargo *)")
        .output();

    match claude_result {
        Ok(output) if output.status.success() => {
            // Re-verify the step
            println!("  Re-verifying {} ...", step.name);
            let verify = Command::new(step.cmd[0]).args(&step.cmd[1..]).status();
            if verify.is_ok_and(|s| s.success()) {
                FixOutcome::ClaudeFixed
            } else {
                FixOutcome::ClaudeFailed("Claude's fix did not resolve the issue".into())
            }
        }
        Ok(_) => FixOutcome::ClaudeFailed("Claude exited with non-zero status".into()),
        Err(e) => FixOutcome::ClaudeFailed(format!("Failed to invoke Claude: {e}")),
    }
}

fn run_ci(args: &[String]) -> ExitCode {
    let (options, step_args) = parse_ci_options(args);
    let filter: Vec<&str> = step_args.iter().map(|s| s.as_str()).collect();

    let steps: Vec<&Step> = if filter.is_empty() {
        STEPS.iter().collect()
    } else {
        let mut selected = Vec::new();
        for name in &filter {
            match STEPS.iter().find(|s| s.key.eq_ignore_ascii_case(name)) {
                Some(s) => selected.push(s),
                None => {
                    eprintln!("Unknown step: {name}");
                    eprintln!("Valid steps: fmt, check, clippy, test, doc");
                    return ExitCode::FAILURE;
                }
            }
        }
        selected
    };

    let total = steps.len();

    // In fix mode, disable color so captured output is clean
    // SAFETY: single-threaded at this point, before spawning any child processes.
    if options.fix {
        unsafe { env::set_var("CARGO_TERM_COLOR", "never") };
    } else {
        unsafe { env::set_var("CARGO_TERM_COLOR", "always") };
    }

    println!("Brainwires Framework — Local CI");
    println!(
        "Steps: {}",
        steps.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
    );
    if options.fix {
        println!("Mode: autofix (max turns: {})", options.max_turns);
    }
    println!("============================================");

    // Phase 1: Run all steps
    struct StepResult<'a> {
        step: &'a Step,
        passed: bool,
        output: String,
    }

    let mut results: Vec<StepResult> = Vec::new();

    for (i, step) in steps.iter().enumerate() {
        println!("\n[{}/{}] {}", i + 1, total, step.name);

        if options.fix {
            // Capture output for potential fix
            let output = Command::new(step.cmd[0]).args(&step.cmd[1..]).output();
            match output {
                Ok(o) => {
                    let passed = o.status.success();
                    let combined = format!(
                        "{}{}",
                        String::from_utf8_lossy(&o.stdout),
                        String::from_utf8_lossy(&o.stderr),
                    );
                    if passed {
                        println!("PASS {}", step.name);
                    } else {
                        println!("FAIL {}", step.name);
                    }
                    results.push(StepResult {
                        step,
                        passed,
                        output: combined,
                    });
                }
                Err(e) => {
                    println!("FAIL {} (could not execute: {e})", step.name);
                    results.push(StepResult {
                        step,
                        passed: false,
                        output: format!("Failed to execute command: {e}"),
                    });
                }
            }
        } else {
            // Stream output directly to terminal
            let status = Command::new(step.cmd[0]).args(&step.cmd[1..]).status();
            let passed = status.is_ok_and(|s| s.success());
            if passed {
                println!("PASS {}", step.name);
            } else {
                println!("FAIL {}", step.name);
            }
            results.push(StepResult {
                step,
                passed,
                output: String::new(),
            });
        }
    }

    // Phase 2: Fix failures (if --fix)
    let mut fix_outcomes: Vec<(&str, &str)> = Vec::new(); // (name, status_label)
    let mut final_pass_count = 0usize;
    let mut final_fail_names: Vec<&str> = Vec::new();

    if options.fix {
        let has_failures = results.iter().any(|r| !r.passed);
        if has_failures {
            // Re-enable color for the fix/verify phase terminal output
            unsafe { env::set_var("CARGO_TERM_COLOR", "always") };
            println!("\n============================================");
            println!("Attempting fixes...\n");
        }
    }

    for result in &results {
        if result.passed {
            fix_outcomes.push((result.step.name, "PASS"));
            final_pass_count += 1;
            continue;
        }

        if !options.fix {
            fix_outcomes.push((result.step.name, "FAIL"));
            final_fail_names.push(result.step.name);
            continue;
        }

        // Attempt fix
        match try_fix(result.step, &result.output, &options) {
            FixOutcome::AutoFixed => {
                println!("  {} -> AUTO-FIXED\n", result.step.name);
                fix_outcomes.push((result.step.name, "AUTO-FIXED"));
                final_pass_count += 1;
            }
            FixOutcome::ClaudeFixed => {
                println!("  {} -> CLAUDE-FIXED\n", result.step.name);
                fix_outcomes.push((result.step.name, "CLAUDE-FIXED"));
                final_pass_count += 1;
            }
            FixOutcome::ClaudeFailed(reason) => {
                println!("  {} -> CLAUDE-FAILED ({})\n", result.step.name, reason);
                fix_outcomes.push((result.step.name, "CLAUDE-FAILED"));
                final_fail_names.push(result.step.name);
            }
            FixOutcome::ClaudeUnavailable => {
                println!(
                    "  {} -> SKIPPED (claude not found on PATH)\n",
                    result.step.name
                );
                fix_outcomes.push((result.step.name, "SKIPPED"));
                final_fail_names.push(result.step.name);
            }
        }
    }

    // Final report
    println!("\n============================================");

    if options.fix && results.iter().any(|r| !r.passed) {
        println!("Results:");
        for (name, label) in &fix_outcomes {
            println!("  {name:<12} {label}");
        }
        println!();
    }

    if final_fail_names.is_empty() {
        if options.fix && results.iter().any(|r| !r.passed) {
            println!("All {final_pass_count} steps passed after fixes.");
        } else {
            println!("All {final_pass_count} steps passed.");
        }
        ExitCode::SUCCESS
    } else {
        println!(
            "{}/{total} steps failed: {}",
            final_fail_names.len(),
            final_fail_names.join(", ")
        );
        ExitCode::FAILURE
    }
}
