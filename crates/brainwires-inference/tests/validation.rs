//! Integration tests for the validation loop.
//!
//! Tests ValidationConfig construction, disabled validation, file-existence
//! checks, and feedback formatting -- all through the public API.

use brainwires_inference::validation_loop::{
    ValidationCheck, ValidationConfig, ValidationIssue, ValidationResult, ValidationSeverity,
    format_validation_feedback, run_validation,
};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// ValidationConfig builder
// ---------------------------------------------------------------------------

#[test]
fn config_default_has_duplicates_and_syntax_checks() {
    let config = ValidationConfig::default();
    assert!(config.enabled);
    assert_eq!(config.checks.len(), 2);
    assert!(config.working_set_files.is_empty());
}

#[test]
fn config_with_build_appends_build_check() {
    let config = ValidationConfig::default().with_build("typescript");
    assert_eq!(config.checks.len(), 3);
    match &config.checks[2] {
        ValidationCheck::BuildSuccess { build_type } => {
            assert_eq!(build_type, "typescript");
        }
        _ => panic!("Expected BuildSuccess check"),
    }
}

#[test]
fn config_disabled_creates_disabled_config() {
    let config = ValidationConfig::disabled();
    assert!(!config.enabled);
}

#[test]
fn config_with_working_set_files() {
    let config = ValidationConfig::default()
        .with_working_set_files(vec!["src/main.rs".into(), "lib.rs".into()]);
    assert_eq!(config.working_set_files.len(), 2);
}

// ---------------------------------------------------------------------------
// Disabled validation always passes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disabled_validation_passes_immediately() {
    let config = ValidationConfig::disabled();
    let result = run_validation(&config).await.unwrap();
    assert!(result.passed);
    assert!(result.issues.is_empty());
}

// ---------------------------------------------------------------------------
// File existence check (Bug #5 prevention)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn validation_catches_missing_working_set_files() {
    let dir = tempdir().unwrap();

    // Working set claims a file exists that does NOT
    let config = ValidationConfig {
        checks: vec![], // No duplicates/syntax checks -- just file existence
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec!["nonexistent.rs".into()],
        intended_writes: None,
    };

    let result = run_validation(&config).await.unwrap();
    assert!(!result.passed);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].check, "file_existence");
    assert_eq!(result.issues[0].severity, ValidationSeverity::Error);
    assert!(result.issues[0].message.contains("does not exist"));
}

#[tokio::test]
async fn validation_passes_when_working_set_files_exist() {
    let dir = tempdir().unwrap();

    // Create the file
    let file_path = dir.path().join("exists.txt");
    std::fs::write(&file_path, "content").unwrap();

    let config = ValidationConfig {
        checks: vec![], // No other checks
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec!["exists.txt".into()],
        intended_writes: None,
    };

    let result = run_validation(&config).await.unwrap();
    assert!(result.passed);
    assert!(result.issues.is_empty());
}

#[tokio::test]
async fn validation_mixed_existing_and_missing_files() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("real.rs"), "fn main() {}").unwrap();

    let config = ValidationConfig {
        checks: vec![],
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec!["real.rs".into(), "ghost.rs".into()],
        intended_writes: None,
    };

    let result = run_validation(&config).await.unwrap();
    assert!(!result.passed);
    assert_eq!(result.issues.len(), 1);
    assert!(result.issues[0].file.as_deref() == Some("ghost.rs"));
}

// ---------------------------------------------------------------------------
// Feedback formatting
// ---------------------------------------------------------------------------

#[test]
fn format_feedback_for_passed_validation() {
    let result = ValidationResult {
        passed: true,
        issues: vec![],
    };
    let feedback = format_validation_feedback(&result);
    assert!(feedback.contains("passed"));
}

#[test]
fn format_feedback_includes_all_issues() {
    let result = ValidationResult {
        passed: false,
        issues: vec![
            ValidationIssue {
                check: "duplicate_check".into(),
                severity: ValidationSeverity::Error,
                message: "Duplicate export 'Foo'".into(),
                file: Some("src/lib.rs".into()),
                line: Some(42),
            },
            ValidationIssue {
                check: "file_existence".into(),
                severity: ValidationSeverity::Error,
                message: "File does not exist".into(),
                file: Some("missing.rs".into()),
                line: None,
            },
        ],
    };

    let feedback = format_validation_feedback(&result);
    assert!(feedback.contains("VALIDATION FAILED"));
    assert!(feedback.contains("src/lib.rs:42:"));
    assert!(feedback.contains("Duplicate export 'Foo'"));
    assert!(feedback.contains("missing.rs:"));
    assert!(feedback.contains("File does not exist"));
    assert!(feedback.contains("MUST fix ALL"));
}

// ---------------------------------------------------------------------------
// Empty working set with no checks passes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_working_set_no_checks_passes() {
    let dir = tempdir().unwrap();

    let config = ValidationConfig {
        checks: vec![],
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 3,
        enabled: true,
        working_set_files: vec![],
        intended_writes: None,
    };

    let result = run_validation(&config).await.unwrap();
    assert!(result.passed);
}

// ---------------------------------------------------------------------------
// content_persisted — catches post-validation clobber by concurrent writer
// ---------------------------------------------------------------------------

/// End-to-end proof of the R1 fix:
///
/// An agent records the SHA-256 of content it wrote.  Before the agent
/// finalises `Success: true`, a *different* writer (simulated here by a
/// direct `fs::write`) clobbers the file with different bytes.  The
/// validation loop must emit a `content_persisted` error so the agent's
/// retry machinery runs instead of silently claiming success.
///
/// Without this check, the losing agent would report `Success: true`
/// while its content is gone from disk — the failure mode that prompted
/// this R1 fix.
#[tokio::test]
async fn validation_catches_post_validation_clobber() {
    use brainwires_core::IntendedWrites;
    use sha2::{Digest, Sha256};

    let dir = tempdir().unwrap();
    let target = dir.path().join("contested.txt");

    // ── Step 1: simulate a successful write_file ─────────────────────────
    // Tool writes "agent-A content" and records the hash.  (The readback
    // check inside write_file would have passed at this instant.)
    let agent_a_content = b"agent-A content";
    std::fs::write(&target, agent_a_content).unwrap();
    let agent_a_hash: [u8; 32] = Sha256::digest(agent_a_content).into();

    let intended = IntendedWrites::new();
    intended.record(target.clone(), agent_a_hash);

    // Sanity: validation should currently pass (content matches hash).
    let config_ok = ValidationConfig {
        checks: vec![],
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 1,
        enabled: true,
        working_set_files: vec![],
        intended_writes: Some(intended.clone()),
    };
    let ok = run_validation(&config_ok).await.unwrap();
    assert!(
        ok.passed,
        "expected validation to pass immediately after write; got issues: {:?}",
        ok.issues
    );

    // ── Step 2: a concurrent writer clobbers our bytes ───────────────────
    // Agent A has NOT yet finalised success.  Agent B silently overwrites.
    std::fs::write(&target, b"agent-B clobbered this").unwrap();

    // ── Step 3: agent A runs validation again before finalising ──────────
    let config_clobbered = ValidationConfig {
        checks: vec![],
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 1,
        enabled: true,
        working_set_files: vec![],
        intended_writes: Some(intended),
    };
    let result = run_validation(&config_clobbered).await.unwrap();

    assert!(
        !result.passed,
        "validation MUST fail when a concurrent writer has clobbered \
         this agent's content; otherwise two agents can both report Success: true"
    );

    let content_persisted_issues: Vec<_> = result
        .issues
        .iter()
        .filter(|i| i.check == "content_persisted")
        .collect();
    assert_eq!(
        content_persisted_issues.len(),
        1,
        "expected exactly one content_persisted issue, got {:?}",
        result.issues
    );

    let issue = content_persisted_issues[0];
    assert_eq!(issue.severity, ValidationSeverity::Error);
    assert!(
        issue.message.contains("overwritten by a concurrent writer"),
        "message should explain the clobber clearly, got: {}",
        issue.message
    );
    assert_eq!(
        issue.file.as_deref(),
        Some(target.display().to_string()).as_deref()
    );
}

/// If a concurrently running process deletes the file between our write
/// and finalisation, validation must still catch it (same class of bug).
#[tokio::test]
async fn validation_catches_deleted_written_file() {
    use brainwires_core::IntendedWrites;
    use sha2::{Digest, Sha256};

    let dir = tempdir().unwrap();
    let target = dir.path().join("ephemeral.txt");
    let content = b"will be deleted";
    std::fs::write(&target, content).unwrap();
    let hash: [u8; 32] = Sha256::digest(content).into();

    let intended = IntendedWrites::new();
    intended.record(target.clone(), hash);

    std::fs::remove_file(&target).unwrap();

    let config = ValidationConfig {
        checks: vec![],
        working_directory: dir.path().to_str().unwrap().to_string(),
        max_retries: 1,
        enabled: true,
        working_set_files: vec![],
        intended_writes: Some(intended),
    };

    let result = run_validation(&config).await.unwrap();
    assert!(!result.passed);
    let has_persisted_issue = result.issues.iter().any(|i| i.check == "content_persisted");
    assert!(
        has_persisted_issue,
        "expected content_persisted error when written file disappeared; got {:?}",
        result.issues
    );
}
