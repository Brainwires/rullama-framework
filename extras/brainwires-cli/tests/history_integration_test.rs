//! Deep integration tests for the history command
//! Tests actual conversation storage, search, and retrieval using LanceDB

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Setup a test environment with temp directories
struct TestEnv {
    temp_dir: TempDir,
}

impl TestEnv {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create necessary subdirectories
        let data_dir = temp_dir.path().join(".local/share/brainwires");
        fs::create_dir_all(&data_dir).expect("Failed to create data dir");

        Self { temp_dir }
    }

    fn db_path(&self) -> PathBuf {
        self.temp_dir
            .path()
            .join(".local/share/brainwires/conversations.lance")
    }

    fn cmd(&self) -> Command {
        let mut cmd = Command::cargo_bin("brainwires").expect("Failed to find brainwires binary");

        cmd.env("HOME", self.temp_dir.path());
        cmd.env("XDG_DATA_HOME", self.temp_dir.path().join(".local/share"));

        cmd
    }
}

// ============================================================================
// History List Tests
// ============================================================================

#[test]
fn test_history_list_no_conversations() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("No conversations found"));
}

// Now that FastEmbed loads lazily, `CachedEmbeddingProvider::new()` no longer
// touches the network, so list paths that construct one but never embed
// can run offline in CI.
#[test]
fn test_history_list_with_zero_limit() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("list")
        .arg("--limit")
        .arg("0")
        .assert()
        .success();
}

#[test]
fn test_history_list_with_large_limit() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("list")
        .arg("--limit")
        .arg("1000")
        .assert()
        .success();
}

// ============================================================================
// History Search Tests
// ============================================================================

#[test]
fn test_history_search_empty_database() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test query")
        .assert()
        .success()
        .stdout(predicate::str::contains("Searching for"))
        .stdout(predicate::str::contains("No matching conversations found"));
}

#[test]
fn test_history_search_with_quotes() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("\"multi word query\"")
        .assert()
        .success()
        .stdout(predicate::str::contains("Searching for"));
}

#[test]
fn test_history_search_special_characters() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("code: fn main() { }")
        .assert()
        .success();
}

#[test]
fn test_history_search_limit_parameter() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .arg("--limit")
        .arg("5")
        .assert()
        .success();
}

#[test]
fn test_history_search_min_score_parameter() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .arg("--min-score")
        .arg("0.8")
        .assert()
        .success();
}

// Actual embedding call happens here (search) — gated behind the
// TEST_EMBED_NETWORK env var for CI environments without a cached model.
// Set the var to 1 to enable; default is offline-skip (success).
#[test]
fn test_history_search_combined_parameters() {
    if std::env::var("TEST_EMBED_NETWORK").ok().as_deref() != Some("1") {
        eprintln!("skipping: set TEST_EMBED_NETWORK=1 to run (needs FastEmbed model)");
        return;
    }
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("rust code")
        .arg("--limit")
        .arg("10")
        .arg("--min-score")
        .arg("0.6")
        .assert()
        .success();
}

#[test]
fn test_history_search_invalid_min_score() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .arg("--min-score")
        .arg("invalid")
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid").or(predicate::str::contains("error")));
}

// ============================================================================
// History Show Tests
// ============================================================================

#[test]
fn test_history_show_invalid_id() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("show")
        .arg("invalid-uuid-12345")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("error")));
}

#[test]
fn test_history_show_without_messages_flag() {
    let env = TestEnv::new();

    // Try to show a non-existent conversation
    env.cmd()
        .arg("history")
        .arg("show")
        .arg("00000000-0000-0000-0000-000000000000")
        .assert()
        .failure(); // Should fail because conversation doesn't exist
}

#[test]
fn test_history_show_with_messages_flag() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("show")
        .arg("00000000-0000-0000-0000-000000000000")
        .arg("--messages")
        .assert()
        .failure(); // Should fail because conversation doesn't exist
}

// ============================================================================
// History Delete Tests
// ============================================================================

#[test]
fn test_history_delete_without_confirm() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("delete")
        .arg("00000000-0000-0000-0000-000000000000")
        .assert()
        .success()
        // Either shows "Use --confirm" or "not found" depending on if conversation exists
        .stdout(
            predicate::str::contains("Use --confirm").or(predicate::str::contains("not found")),
        );
}

#[test]
fn test_history_delete_nonexistent_with_confirm() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("delete")
        .arg("00000000-0000-0000-0000-000000000000")
        .arg("--confirm")
        .assert()
        .success()
        .stdout(predicate::str::contains("not found"));
}

#[test]
fn test_history_delete_invalid_id_format() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("delete")
        .arg("not-a-uuid")
        .arg("--confirm")
        .assert()
        .success(); // Should handle gracefully, just show "not found"
}

// ============================================================================
// Database Initialization Tests
// ============================================================================

#[test]
fn test_history_creates_database_dir() {
    let env = TestEnv::new();

    // Run any history command
    env.cmd().arg("history").arg("list").assert().success();

    // Check that lance directory structure was created
    let db_path = env.db_path();
    let db_parent = db_path.parent().unwrap();
    assert!(
        db_parent.exists(),
        "Database directory should be created: {:?}",
        db_parent
    );
}

#[test]
fn test_history_commands_are_idempotent() {
    let env = TestEnv::new();

    // Run list twice
    env.cmd().arg("history").arg("list").assert().success();
    env.cmd().arg("history").arg("list").assert().success();

    // Run search twice
    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .assert()
        .success();
    env.cmd()
        .arg("history")
        .arg("search")
        .arg("test")
        .assert()
        .success();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_history_list_handles_corrupted_db_gracefully() {
    let env = TestEnv::new();

    // Create a file where the database directory should be to simulate corruption
    let db_path = env.db_path();
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).ok();
        fs::write(&db_path, b"corrupted").ok();
    }

    // Should handle error gracefully
    let _ = env.cmd().arg("history").arg("list").assert(); // May fail or succeed depending on error handling
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_history_search_empty_query() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("")
        .assert()
        .success(); // Should handle empty query
}

#[test]
fn test_history_search_very_long_query() {
    let env = TestEnv::new();

    let long_query = "a".repeat(10000);
    env.cmd()
        .arg("history")
        .arg("search")
        .arg(&long_query)
        .assert()
        .success(); // Should handle long queries
}

#[test]
fn test_history_show_empty_conversation_id() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("show")
        .arg("")
        .assert()
        .failure(); // Should fail with empty ID
}

// ============================================================================
// Help Text Tests
// ============================================================================

#[test]
fn test_history_list_help() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("list")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("List all saved conversations"))
        .stdout(predicate::str::contains("--limit"));
}

#[test]
fn test_history_search_help() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("search")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Search conversations by semantic similarity",
        ))
        .stdout(predicate::str::contains("--limit"))
        .stdout(predicate::str::contains("--min-score"));
}

#[test]
fn test_history_show_help() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("show")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Show conversation details"))
        .stdout(predicate::str::contains("--messages"));
}

#[test]
fn test_history_delete_help() {
    let env = TestEnv::new();

    env.cmd()
        .arg("history")
        .arg("delete")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Delete a conversation"))
        .stdout(predicate::str::contains("--confirm"));
}
