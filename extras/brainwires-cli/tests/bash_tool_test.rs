// Integration tests for bash tool
mod common;

use brainwires_cli::tools::BashTool;
use brainwires_cli::types::tool::ToolContext;
use common::create_test_dir;
use serde_json::json;

#[test]
fn test_bash_command_execution() {
    let temp_dir = create_test_dir();
    let temp_path = temp_dir.path().to_str().unwrap();

    let context = ToolContext {
        working_directory: temp_path.to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    // Test simple echo command
    let echo_input = json!({
        "command": "echo 'Hello from bash'"
    });

    let result = BashTool::execute("test-id-1", "execute_command", &echo_input, &context);
    assert!(!result.is_error, "Echo command should succeed");
    assert!(result.content.contains("Hello from bash"));
}

#[test]
fn test_bash_working_directory() {
    let temp_dir = create_test_dir();
    let temp_path = temp_dir.path().to_str().unwrap();

    let context = ToolContext {
        working_directory: temp_path.to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    // Test pwd command
    let pwd_input = json!({
        "command": "pwd"
    });

    let result = BashTool::execute("test-id-2", "execute_command", &pwd_input, &context);
    assert!(!result.is_error, "PWD command should succeed");
    assert!(
        result.content.contains(temp_path),
        "Should execute in working directory"
    );
}

#[test]
fn test_bash_dangerous_commands_blocked() {
    let temp_dir = create_test_dir();
    let temp_path = temp_dir.path().to_str().unwrap();

    let context = ToolContext {
        working_directory: temp_path.to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    // Test dangerous command is blocked
    let dangerous_input = json!({
        "command": "rm -rf /"
    });

    let result = BashTool::execute("test-id-3", "execute_command", &dangerous_input, &context);
    assert!(result.is_error, "Dangerous command should be blocked");
    assert!(
        result.content.to_lowercase().contains("dangerous")
            || result.content.to_lowercase().contains("blocked")
            || result.content.to_lowercase().contains("not allowed")
    );
}

#[test]
fn test_bash_command_with_pipe() {
    let temp_dir = create_test_dir();
    let temp_path = temp_dir.path().to_str().unwrap();

    let context = ToolContext {
        working_directory: temp_path.to_string(),
        user_id: None,
        metadata: std::collections::HashMap::new(),
        capabilities: None,
        idempotency_registry: None,
        staging_backend: None,
        intended_writes: None,
    };

    // Test command with pipe
    let pipe_input = json!({
        "command": "echo 'apple\nbanana\ncherry' | grep banana"
    });

    let result = BashTool::execute("test-id-4", "execute_command", &pipe_input, &context);
    assert!(!result.is_error, "Pipe command should succeed");
    assert!(result.content.contains("banana"));
}
