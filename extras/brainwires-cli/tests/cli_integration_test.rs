//! Integration tests for CLI commands
//! These test actual CLI command execution end-to-end

use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use tempfile::TempDir;

/// Helper to create a test command with clean environment
fn brainwires_cmd() -> Command {
    let mut cmd = Command::cargo_bin("brainwires").expect("Failed to find brainwires binary");

    // Use a temporary config directory to avoid interfering with real config
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    cmd.env("HOME", temp_dir.path());
    cmd.env("XDG_CONFIG_HOME", temp_dir.path().join(".config"));
    cmd.env("XDG_DATA_HOME", temp_dir.path().join(".local/share"));

    cmd
}

/// Helper for commands that need authentication
fn authenticated_cmd() -> Command {
    let mut cmd = brainwires_cmd();

    // Set test API key if provided via env
    if let Ok(api_key) = env::var("TEST_API_KEY") {
        cmd.env("BRAINWIRES_API_KEY", api_key);
    }

    cmd
}

// ============================================================================
// Config Command Tests
// ============================================================================

#[test]
fn test_config_help() {
    brainwires_cmd()
        .arg("config")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Manage configuration"))
        .stdout(predicate::str::contains("--list"))
        .stdout(predicate::str::contains("--get"))
        .stdout(predicate::str::contains("--set"));
}

#[test]
fn test_config_list_empty() {
    brainwires_cmd()
        .arg("config")
        .arg("--list")
        .assert()
        .success();
    // Config might show defaults even when empty
}

#[test]
fn test_config_set_and_get() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Set a config value
    let mut cmd = Command::cargo_bin("brainwires").unwrap();
    cmd.env("HOME", temp_dir.path());
    cmd.arg("config")
        .arg("--set")
        .arg("model=test-model")
        .assert()
        .success();

    // Get the config value
    let mut cmd = Command::cargo_bin("brainwires").unwrap();
    cmd.env("HOME", temp_dir.path());
    cmd.arg("config")
        .arg("--get")
        .arg("model")
        .assert()
        .success()
        .stdout(predicate::str::contains("test-model"));
}

#[test]
fn test_config_set_multiple_formats() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let home_path = temp_dir.path();

    // Format 1: key=value (set backend_url)
    let mut cmd = Command::cargo_bin("brainwires").unwrap();
    cmd.env("HOME", home_path);
    cmd.env("XDG_CONFIG_HOME", home_path.join(".config"));
    cmd.arg("config")
        .arg("--set")
        .arg("backend_url=https://test.example.com")
        .assert()
        .success();

    // Format 2: key value (space-separated) - set temperature
    let mut cmd = Command::cargo_bin("brainwires").unwrap();
    cmd.env("HOME", home_path);
    cmd.env("XDG_CONFIG_HOME", home_path.join(".config"));
    cmd.arg("config")
        .arg("--set")
        .arg("temperature")
        .arg("0.9")
        .assert()
        .success();

    // Verify both were set
    let mut cmd = Command::cargo_bin("brainwires").unwrap();
    cmd.env("HOME", home_path);
    cmd.env("XDG_CONFIG_HOME", home_path.join(".config"));
    cmd.arg("config")
        .arg("--list")
        .assert()
        .success()
        .stdout(predicate::str::contains("test.example.com"))
        .stdout(predicate::str::contains("0.9"));
}

// ============================================================================
// Models Command Tests
// ============================================================================

#[test]
fn test_models_help() {
    brainwires_cmd()
        .arg("models")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("List available AI models"))
        .stdout(predicate::str::contains("list"));
}

#[test]
#[ignore] // Requires network/API access
fn test_models_list_all() {
    authenticated_cmd()
        .arg("models")
        .assert()
        .success()
        .stdout(predicate::str::contains("Available Models").or(predicate::str::contains("model")));
}

#[test]
#[ignore] // Requires network/API access
fn test_models_list_by_provider() {
    authenticated_cmd()
        .arg("models")
        .arg("--provider")
        .arg("anthropic")
        .assert()
        .success();
}

// ============================================================================
// History Command Tests
// ============================================================================

#[test]
fn test_history_help() {
    brainwires_cmd()
        .arg("history")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Manage conversation history"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("search"))
        .stdout(predicate::str::contains("show"))
        .stdout(predicate::str::contains("delete"));
}

#[test]
fn test_history_list_empty() {
    brainwires_cmd()
        .arg("history")
        .arg("list")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("No conversations found")
                .or(predicate::str::contains("Saved Conversations")),
        );
}

#[test]
fn test_history_list_with_limit() {
    brainwires_cmd()
        .arg("history")
        .arg("list")
        .arg("--limit")
        .arg("5")
        .assert()
        .success();
}

#[test]
fn test_history_search_empty() {
    brainwires_cmd()
        .arg("history")
        .arg("search")
        .arg("test query")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Searching for").or(predicate::str::contains("No matching")),
        );
}

#[test]
fn test_history_search_with_options() {
    brainwires_cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .arg("--limit")
        .arg("10")
        .arg("--min-score")
        .arg("0.7")
        .assert()
        .success();
}

#[test]
fn test_history_show_nonexistent() {
    brainwires_cmd()
        .arg("history")
        .arg("show")
        .arg("nonexistent-id-12345")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("error")));
}

#[test]
fn test_history_delete_without_confirm() {
    brainwires_cmd()
        .arg("history")
        .arg("delete")
        .arg("00000000-0000-0000-0000-000000000000")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Use --confirm").or(predicate::str::contains("not found")),
        );
}

#[test]
fn test_history_delete_nonexistent_with_confirm() {
    brainwires_cmd()
        .arg("history")
        .arg("delete")
        .arg("nonexistent-id")
        .arg("--confirm")
        .assert()
        .success()
        .stdout(predicate::str::contains("not found").or(predicate::str::contains("Deleted")));
}

// ============================================================================
// Main Command Tests
// ============================================================================

#[test]
fn test_no_command_shows_help() {
    brainwires_cmd()
        .assert()
        .success()
        .stdout(predicate::str::contains("AI-powered agentic CLI"))
        .stdout(predicate::str::contains("Usage:").or(predicate::str::contains("USAGE:")))
        .stdout(predicate::str::contains("Commands:").or(predicate::str::contains("COMMANDS:")));
}

#[test]
fn test_help_flag() {
    brainwires_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("auth"))
        .stdout(predicate::str::contains("chat"))
        .stdout(predicate::str::contains("config"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("history"));
}

#[test]
fn test_version_flag() {
    brainwires_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::is_match(r"\d+\.\d+\.\d+").unwrap());
}

// ============================================================================
// Chat Command Tests (Non-interactive)
// ============================================================================

#[test]
fn test_chat_help() {
    brainwires_cmd()
        .arg("chat")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Start an interactive chat"))
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--provider"))
        .stdout(predicate::str::contains("--system"));
}

// Note: Interactive chat testing requires special handling (stdin simulation)
// and is tested separately in chat_interactive_test.rs

// ============================================================================
// Cost Command Tests
// ============================================================================

#[test]
fn test_cost_help() {
    brainwires_cmd()
        .arg("cost")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("View API usage and costs"))
        .stdout(predicate::str::contains("--period"))
        .stdout(predicate::str::contains("--reset"));
}

#[test]
fn test_cost_default() {
    brainwires_cmd().arg("cost").assert().success();
}

#[test]
fn test_cost_with_period() {
    brainwires_cmd()
        .arg("cost")
        .arg("--period")
        .arg("week")
        .assert()
        .success();
}

// ============================================================================
// MCP Command Tests
// ============================================================================

#[test]
fn test_mcp_help() {
    brainwires_cmd()
        .arg("mcp")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Manage MCP servers"));
}

#[test]
fn test_mcp_list_empty() {
    brainwires_cmd().arg("mcp").arg("list").assert().success();
}

// ============================================================================
// Plan Command Tests
// ============================================================================

#[test]
fn test_plan_help() {
    brainwires_cmd()
        .arg("plan")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Manage execution plans"))
        .stdout(predicate::str::contains("create"));
}

// ============================================================================
// Init Command Tests
// ============================================================================

#[test]
fn test_init() {
    // `brainwires init` is documented as not yet implemented and now exits
    // non-zero so scripts can detect the no-op. Assert the current, correct
    // behavior: failure + "not yet implemented" message.
    brainwires_cmd().arg("init").assert().failure().stderr(
        predicate::str::contains("Init not yet implemented")
            .or(predicate::str::contains("not yet implemented")),
    );
}
