//! Example: ValidationLoop quality gates before agent completion
//!
//! Demonstrates how `run_validation` enforces checks (file existence,
//! duplicate detection, syntax validity) on an agent's working set.
//! A file with intentional issues is created, validated (fails), fixed,
//! then re-validated (passes).
//!
//! Run: cargo run -p brainwires-agent --example validation_loop

use std::fs;

use anyhow::Result;

use brainwires_inference::validation_loop::{
    ValidationCheck, ValidationConfig, ValidationSeverity, format_validation_feedback,
    run_validation,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Validation Loop Demo ===\n");

    // ── 1. Create a temp directory with a file that has issues ──────────────

    let tmp_dir = tempfile::tempdir()?;
    let tmp_path = tmp_dir.path().to_string_lossy().to_string();

    let bad_file = tmp_dir.path().join("utils.ts");
    let bad_content = r#"export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export function greet(name: string): string {
    return `Hi, ${name}!`;
}
"#;
    fs::write(&bad_file, bad_content)?;
    println!(
        "Created file with duplicate function: {}",
        bad_file.display()
    );

    // ── 2. Configure validation ────────────────────────────────────────────

    let config = ValidationConfig {
        checks: vec![ValidationCheck::NoDuplicates, ValidationCheck::SyntaxValid],
        working_directory: tmp_path.clone(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec!["utils.ts".to_string()],
        intended_writes: None,
    };

    println!("Validation config:");
    println!("  Checks:            {:?}", config.checks);
    println!("  Working directory:  {}", config.working_directory);
    println!("  Max retries:       {}", config.max_retries);
    println!("  Working set files: {:?}", config.working_set_files);

    // ── 3. Run validation — expect failure ─────────────────────────────────

    println!("\n--- First validation run (expecting issues) ---");

    let result = run_validation(&config).await?;
    println!("Passed: {}", result.passed);
    println!("Issues found: {}", result.issues.len());

    for issue in &result.issues {
        let severity = match issue.severity {
            ValidationSeverity::Error => "ERROR",
            ValidationSeverity::Warning => "WARN",
            ValidationSeverity::Info => "INFO",
        };
        println!("  [{}] {}: {}", severity, issue.check, issue.message);
        if let Some(file) = &issue.file {
            print!("         File: {file}");
            if let Some(line) = issue.line {
                print!(" (line {line})");
            }
            println!();
        }
    }

    // Show the formatted feedback an agent would receive
    let feedback = format_validation_feedback(&result);
    println!("\nAgent feedback:\n{feedback}");

    // ── 4. Fix the file ────────────────────────────────────────────────────

    println!("--- Fixing the file ---");

    let good_content = r#"export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export function farewell(name: string): string {
    return `Goodbye, ${name}!`;
}
"#;
    fs::write(&bad_file, good_content)?;
    println!("Replaced duplicate function with unique 'farewell'");

    // ── 5. Re-run validation — expect pass ─────────────────────────────────

    println!("\n--- Second validation run (expecting pass) ---");

    let result2 = run_validation(&config).await?;
    println!("Passed: {}", result2.passed);
    println!("Issues found: {}", result2.issues.len());

    let feedback2 = format_validation_feedback(&result2);
    println!("Agent feedback: {feedback2}");

    // ── 6. Demonstrate file-existence check ────────────────────────────────

    println!("--- File existence check ---");

    let missing_config = ValidationConfig {
        checks: vec![],
        working_directory: tmp_path.clone(),
        max_retries: 1,
        enabled: true,
        working_set_files: vec!["nonexistent.rs".to_string()],
        intended_writes: None,
    };

    let missing_result = run_validation(&missing_config).await?;
    println!("Passed (with missing file): {}", missing_result.passed);
    for issue in &missing_result.issues {
        println!("  [{}] {}", issue.check, issue.message);
    }

    // ── 7. Summary ─────────────────────────────────────────────────────────

    println!("\n--- Summary ---");
    println!("The validation loop prevents agents from reporting success when:");
    println!("  - Files in the working set do not exist on disk (Bug #5)");
    println!("  - Duplicate exports/functions/types are present");
    println!("  - Basic syntax errors are detected");
    println!("  - Build commands fail (when BuildSuccess check is enabled)");

    println!("\nValidation loop demo complete.");
    Ok(())
}
