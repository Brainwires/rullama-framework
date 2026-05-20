// Integration tests for git tool
mod common;

use brainwires_cli::tools::GitTool;
use brainwires_cli::types::tool::ToolContext;
use serde_json::json;
use std::env;

#[tokio::test]
async fn test_git_status_in_git_repo() {
    // Use current project directory which is a git repo
    let current_dir = env::current_dir().unwrap();
    let context = ToolContext {
        working_directory: current_dir.to_str().unwrap().to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    let status_input = json!({});
    let result = GitTool::execute("test-id-1", "git_status", &status_input, &context);

    // Should succeed in a git repo
    assert!(!result.is_error, "Git status should succeed in git repo");
    assert!(
        !result.content.is_empty(),
        "Git status should return output"
    );
}

#[tokio::test]
async fn test_git_log_in_git_repo() {
    let current_dir = env::current_dir().unwrap();
    let context = ToolContext {
        working_directory: current_dir.to_str().unwrap().to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    let log_input = json!({
        "max_count": 5
    });

    let result = GitTool::execute("test-id-2", "git_log", &log_input, &context);

    // Should succeed in a git repo
    assert!(!result.is_error, "Git log should succeed in git repo");
    assert!(!result.content.is_empty(), "Git log should return output");
}

#[tokio::test]
async fn test_git_diff_in_git_repo() {
    let current_dir = env::current_dir().unwrap();
    let context = ToolContext {
        working_directory: current_dir.to_str().unwrap().to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    let diff_input = json!({});
    let result = GitTool::execute("test-id-3", "git_diff", &diff_input, &context);

    // Should succeed (might be empty if no changes)
    assert!(!result.is_error, "Git diff should succeed in git repo");
}
