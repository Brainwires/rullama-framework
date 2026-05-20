//! Validation Loop - Enforces quality checks before agent completion
//!
//! Wraps task agent execution to automatically validate work before allowing completion.
//! If validation fails, forces the agent to fix issues before succeeding.
//!
//! When the `tools` feature is enabled, uses brainwires-tools validation functions
//! (check_duplicates, verify_build, check_syntax). Without it, those checks are skipped.

use anyhow::Result;
use brainwires_core::IntendedWrites;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::process::Command;

const DEFAULT_VALIDATION_MAX_RETRIES: usize = 3;

/// Validation checks to enforce
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCheck {
    /// Check for duplicate exports/constants
    NoDuplicates,
    /// Verify build succeeds
    BuildSuccess {
        /// Build system type (e.g. "typescript", "rust").
        build_type: String,
    },
    /// Check syntax validity
    SyntaxValid,
    /// Custom validation command
    CustomCommand {
        /// Command to run.
        command: String,
        /// Arguments for the command.
        args: Vec<String>,
    },
}

/// Result of validation checks
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// Issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

/// A single issue found during validation
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Name of the check that found this issue.
    pub check: String,
    /// Severity of the issue.
    pub severity: ValidationSeverity,
    /// Human-readable description.
    pub message: String,
    /// File where the issue was found.
    pub file: Option<String>,
    /// Line number of the issue.
    pub line: Option<usize>,
}

/// Severity level for a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Blocks completion.
    Error,
    /// Non-blocking but notable.
    Warning,
    /// Informational only (does not block completion)
    Info,
}

/// Configuration for validation loop
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Checks to run
    pub checks: Vec<ValidationCheck>,
    /// Working directory for validation
    pub working_directory: String,
    /// Maximum validation retry attempts
    pub max_retries: usize,
    /// Whether to run validation (can disable for testing)
    pub enabled: bool,
    /// Specific files to validate (from working set). If empty, falls back to git diff.
    pub working_set_files: Vec<String>,
    /// Shared registry of `(path -> SHA-256 of most recent intended write)`.
    ///
    /// When present, the validation loop re-reads each tracked path and
    /// compares its on-disk SHA-256 against the recorded hash.  A mismatch
    /// means a concurrent writer overwrote our content after our own
    /// read-back succeeded — the agent must NOT report `Success: true`.
    pub intended_writes: Option<IntendedWrites>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            checks: vec![ValidationCheck::NoDuplicates, ValidationCheck::SyntaxValid],
            working_directory: ".".to_string(),
            max_retries: DEFAULT_VALIDATION_MAX_RETRIES,
            enabled: true,
            working_set_files: Vec::new(),
            intended_writes: None,
        }
    }
}

impl ValidationConfig {
    /// Create config with build validation
    pub fn with_build(mut self, build_type: impl Into<String>) -> Self {
        self.checks.push(ValidationCheck::BuildSuccess {
            build_type: build_type.into(),
        });
        self
    }

    /// Disable validation (for testing)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the working set files to validate (from agent's working set)
    pub fn with_working_set_files(mut self, files: Vec<String>) -> Self {
        self.working_set_files = files;
        self
    }

    /// Attach the shared intended-writes registry so the validation loop
    /// can detect post-validation clobber by a concurrent writer.
    pub fn with_intended_writes(mut self, registry: IntendedWrites) -> Self {
        self.intended_writes = Some(registry);
        self
    }
}

/// Run validation checks on changed files
#[tracing::instrument(name = "agent.validate", skip(config), fields(working_dir = %config.working_directory))]
pub async fn run_validation(config: &ValidationConfig) -> Result<ValidationResult> {
    if !config.enabled {
        return Ok(ValidationResult {
            passed: true,
            issues: vec![],
        });
    }

    let mut issues = Vec::new();

    // Get list of modified files - prefer working set, fallback to git
    let changed_files = if !config.working_set_files.is_empty() {
        tracing::debug!(
            "Using working set files for validation: {:?}",
            config.working_set_files
        );
        config.working_set_files.clone()
    } else {
        tracing::debug!("No working set provided, falling back to git diff");
        get_modified_files(&config.working_directory)?
    };
    tracing::debug!("Validating {} changed files", changed_files.len());

    // CRITICAL: Verify that all files in the working set actually exist on disk
    // This catches Bug #5 where agents report success without creating files
    for file in &changed_files {
        let file_path = PathBuf::from(&config.working_directory).join(file);
        if !file_path.exists() {
            issues.push(ValidationIssue {
                check: "file_existence".to_string(),
                severity: ValidationSeverity::Error,
                message: format!(
                    "File '{}' is in working set but does not exist on disk. Agent must create file before completing.",
                    file
                ),
                file: Some(file.clone()),
                line: None,
            });
            tracing::error!(
                "Validation failed: File {} does not exist but is in working set",
                file
            );
        }
    }

    // CRITICAL: Content-persistence check — catches post-validation clobber.
    //
    // The tool-level read-back in write_file catches interleaved concurrent
    // writes within a single call.  It does NOT catch: agent A writes at T1,
    // A's read-back passes, A's validation passes at T2, agent B writes at
    // T3, A finalises `Success: true` at T4 claiming content that's no
    // longer on disk.
    //
    // Here we re-read each file that THIS agent recorded an intended SHA-256
    // for, and compare.  A mismatch means a concurrent writer overwrote our
    // content after our own read-back — emit a retryable error so the
    // agent's retry machinery runs.  If retries are exhausted, `success`
    // correctly propagates up as `false` and at most one of two racing
    // agents can legitimately report success.
    if let Some(ref intended) = config.intended_writes {
        use sha2::{Digest, Sha256};
        for (path, expected_hash) in intended.snapshot() {
            match std::fs::read(&path) {
                Ok(current_bytes) => {
                    let current_hash: [u8; 32] = Sha256::digest(&current_bytes).into();
                    if current_hash != expected_hash {
                        let display_path = path.display().to_string();
                        tracing::error!(
                            path = %display_path,
                            "Validation failed: content_persisted — file clobbered by concurrent writer after write_file's own read-back passed"
                        );
                        issues.push(ValidationIssue {
                            check: "content_persisted".to_string(),
                            severity: ValidationSeverity::Error,
                            message: format!(
                                "File '{}' was written by this agent but its current on-disk \
                                 contents do not match the expected SHA-256 — it was likely \
                                 overwritten by a concurrent writer. Re-write or fail.",
                                display_path
                            ),
                            file: Some(display_path),
                            line: None,
                        });
                    }
                }
                Err(e) => {
                    // File disappeared between our write and validation — same
                    // category of problem.  Surface as a content_persisted error.
                    let display_path = path.display().to_string();
                    tracing::error!(
                        path = %display_path,
                        error = %e,
                        "Validation failed: content_persisted — file unreadable after write"
                    );
                    issues.push(ValidationIssue {
                        check: "content_persisted".to_string(),
                        severity: ValidationSeverity::Error,
                        message: format!(
                            "File '{}' was written by this agent but can no longer be read: {}. \
                             Likely removed or replaced by a concurrent writer.",
                            display_path, e
                        ),
                        file: Some(display_path),
                        line: None,
                    });
                }
            }
        }
    }

    for check in &config.checks {
        match check {
            ValidationCheck::NoDuplicates => {
                #[cfg(feature = "native")]
                run_duplicates_check(&changed_files, &mut issues).await;
                #[cfg(not(feature = "native"))]
                {
                    let _ = &changed_files;
                }
            }

            ValidationCheck::SyntaxValid => {
                #[cfg(feature = "native")]
                run_syntax_check(&changed_files, &mut issues).await;
                #[cfg(not(feature = "native"))]
                {
                    let _ = &changed_files;
                }
            }

            ValidationCheck::BuildSuccess { build_type } => {
                #[cfg(feature = "native")]
                run_build_check(&config.working_directory, build_type, &mut issues).await;
                #[cfg(not(feature = "native"))]
                {
                    let _ = (&config.working_directory, build_type);
                }
            }

            ValidationCheck::CustomCommand { command, args } => {
                #[cfg(feature = "native")]
                {
                    match Command::new(command)
                        .args(args)
                        .current_dir(&config.working_directory)
                        .output()
                    {
                        Ok(output) => {
                            if !output.status.success() {
                                let stderr = String::from_utf8_lossy(&output.stderr);
                                issues.push(ValidationIssue {
                                    check: "custom_command".to_string(),
                                    severity: ValidationSeverity::Error,
                                    message: format!("Command '{}' failed: {}", command, stderr),
                                    file: None,
                                    line: None,
                                });
                            }
                        }
                        Err(e) => {
                            issues.push(ValidationIssue {
                                check: "custom_command".to_string(),
                                severity: ValidationSeverity::Error,
                                message: format!("Failed to run command '{}': {}", command, e),
                                file: None,
                                line: None,
                            });
                        }
                    }
                }
                #[cfg(not(feature = "native"))]
                {
                    let _ = (command, args);
                    issues.push(ValidationIssue {
                        check: "custom_command".to_string(),
                        severity: ValidationSeverity::Warning,
                        message: "Custom command validation not available in WASM".to_string(),
                        file: None,
                        line: None,
                    });
                }
            }
        }
    }

    Ok(ValidationResult {
        passed: issues.is_empty(),
        issues,
    })
}

// ── Validation tool dispatch (feature-gated) ─────────────────────────────────

#[cfg(feature = "native")]
async fn run_duplicates_check(changed_files: &[String], issues: &mut Vec<ValidationIssue>) {
    use brainwires_tool_runtime::validation::check_duplicates;

    for file in changed_files {
        if !is_source_file(file) {
            continue;
        }

        match check_duplicates(file).await {
            Ok(result) => {
                if let Ok(result_value) = serde_json::from_str::<serde_json::Value>(&result.content)
                    && result_value["has_duplicates"].as_bool().unwrap_or(false)
                    && let Some(duplicates) = result_value["duplicates"].as_array()
                {
                    for dup in duplicates {
                        issues.push(ValidationIssue {
                            check: "duplicate_check".to_string(),
                            severity: ValidationSeverity::Error,
                            message: format!(
                                "Duplicate export '{}' found at lines {} and {}",
                                dup["name"].as_str().unwrap_or("unknown"),
                                dup["first_line"].as_u64().unwrap_or(0),
                                dup["duplicate_line"].as_u64().unwrap_or(0)
                            ),
                            file: Some(file.clone()),
                            line: dup["duplicate_line"].as_u64().map(|n| n as usize),
                        });
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to check duplicates in {}: {}", file, e);
            }
        }
    }
}

#[cfg(feature = "native")]
async fn run_syntax_check(changed_files: &[String], issues: &mut Vec<ValidationIssue>) {
    use brainwires_tool_runtime::validation::check_syntax;

    for file in changed_files {
        if !is_source_file(file) {
            continue;
        }

        match check_syntax(file).await {
            Ok(result) => {
                if let Ok(result_value) = serde_json::from_str::<serde_json::Value>(&result.content)
                    && !result_value["valid_syntax"].as_bool().unwrap_or(true)
                    && let Some(errors) = result_value["errors"].as_array()
                {
                    for error in errors {
                        issues.push(ValidationIssue {
                            check: "syntax_check".to_string(),
                            severity: ValidationSeverity::Error,
                            message: error["message"]
                                .as_str()
                                .unwrap_or("Unknown syntax error")
                                .to_string(),
                            file: Some(file.clone()),
                            line: None,
                        });
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to check syntax in {}: {}", file, e);
            }
        }
    }
}

#[cfg(feature = "native")]
async fn run_build_check(
    working_directory: &str,
    build_type: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    use brainwires_tool_runtime::validation::verify_build;

    match verify_build(working_directory, build_type).await {
        Ok(result) => {
            if let Ok(result_value) = serde_json::from_str::<serde_json::Value>(&result.content)
                && !result_value["success"].as_bool().unwrap_or(false)
            {
                let error_count = result_value["error_count"].as_u64().unwrap_or(0);

                if let Some(errors) = result_value["errors"].as_array() {
                    for error in errors.iter().take(5) {
                        issues.push(ValidationIssue {
                            check: "build_check".to_string(),
                            severity: ValidationSeverity::Error,
                            message: error["message"]
                                .as_str()
                                .or_else(|| error["line"].as_str())
                                .unwrap_or("Build error")
                                .to_string(),
                            file: error["location"].as_str().map(|s| s.to_string()),
                            line: None,
                        });
                    }
                }

                if error_count > 5 {
                    issues.push(ValidationIssue {
                        check: "build_check".to_string(),
                        severity: ValidationSeverity::Error,
                        message: format!("... and {} more build errors", error_count - 5),
                        file: None,
                        line: None,
                    });
                }
            }
        }
        Err(e) => {
            issues.push(ValidationIssue {
                check: "build_check".to_string(),
                severity: ValidationSeverity::Error,
                message: format!("Build validation failed: {}", e),
                file: None,
                line: None,
            });
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format validation result as feedback for agent
pub fn format_validation_feedback(result: &ValidationResult) -> String {
    if result.passed {
        return "All validation checks passed!".to_string();
    }

    let mut feedback = String::from("VALIDATION FAILED - You must fix these issues:\n\n");

    for (idx, issue) in result.issues.iter().enumerate() {
        feedback.push_str(&format!("{}. [{}] ", idx + 1, issue.check));

        if let Some(file) = &issue.file {
            feedback.push_str(&format!("{}:", file));
            if let Some(line) = issue.line {
                feedback.push_str(&format!("{}:", line));
            }
            feedback.push(' ');
        }

        feedback.push_str(&issue.message);
        feedback.push('\n');
    }

    feedback.push('\n');
    feedback
        .push_str("IMPORTANT: You MUST fix ALL of these issues before the task can complete.\n");
    feedback.push_str("After fixing, verify your changes by reading the files back.\n");

    feedback
}

/// Get list of files modified in working directory (git-aware)
#[cfg(feature = "native")]
fn get_modified_files(working_directory: &str) -> Result<Vec<String>> {
    if let Ok(output) = Command::new("git")
        .args(["diff", "--name-only", "HEAD"])
        .current_dir(working_directory)
        .output()
        && output.status.success()
    {
        let files: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if !files.is_empty() {
            return Ok(files);
        }
    }

    // Fallback: check for recently modified files
    let path = PathBuf::from(working_directory);
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&path) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata()
                && metadata.is_file()
                && let Some(file_name) = entry.file_name().to_str()
            {
                files.push(file_name.to_string());
            }
        }
    }

    Ok(files)
}

/// Get list of files modified in working directory (WASM fallback)
#[cfg(not(feature = "native"))]
fn get_modified_files(_working_directory: &str) -> Result<Vec<String>> {
    Ok(Vec::new())
}

/// Check if file is a source code file worth validating
#[allow(dead_code)]
fn is_source_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();

    path_lower.ends_with(".rs")
        || path_lower.ends_with(".ts")
        || path_lower.ends_with(".tsx")
        || path_lower.ends_with(".js")
        || path_lower.ends_with(".jsx")
        || path_lower.ends_with(".py")
        || path_lower.ends_with(".java")
        || path_lower.ends_with(".cpp")
        || path_lower.ends_with(".c")
        || path_lower.ends_with(".go")
        || path_lower.ends_with(".rb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_source_file() {
        assert!(is_source_file("src/main.rs"));
        assert!(is_source_file("app.ts"));
        assert!(is_source_file("Component.tsx"));
        assert!(!is_source_file("README.md"));
        assert!(!is_source_file("package.json"));
    }

    #[test]
    fn test_format_validation_feedback() {
        let result = ValidationResult {
            passed: false,
            issues: vec![ValidationIssue {
                check: "duplicate_check".to_string(),
                severity: ValidationSeverity::Error,
                message: "Duplicate export 'FOO'".to_string(),
                file: Some("src/test.ts".to_string()),
                line: Some(42),
            }],
        };

        let feedback = format_validation_feedback(&result);
        assert!(feedback.contains("VALIDATION FAILED"));
        assert!(feedback.contains("src/test.ts:42"));
        assert!(feedback.contains("Duplicate export 'FOO'"));
    }
}
