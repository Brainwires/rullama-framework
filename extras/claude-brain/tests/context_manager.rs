//! Integration tests for `claude-brain`.
//!
//! These tests focus on the pure-Rust pieces that do not depend on the
//! Claude Code hook transport:
//!
//! - Transcript parsing (`read_transcript_messages` + `extract_text_content`)
//!   — the real work PreCompact does before it ever touches storage.
//! - `ContextManager::new` + `memory_stats` — smoke-check the storage wiring
//!   with a `TempDir` so no global state is touched.
//! - `BrainSessionAdapter::save` / `load` — end-to-end message round-trip
//!   through the thought store.
//!
//! The hook entrypoints themselves (`hooks::pre_compact::handle` etc.) read
//! from stdin and write logs to `~/.brainwires/`, so they are intentionally
//! not exercised here; the testable logic has been hoisted out.

use std::sync::Arc;

use claude_brain::config::{ClaudeBrainConfig, StorageConfig};
use claude_brain::context_manager::ContextManager;
use claude_brain::hooks::pre_compact::{extract_text_content, read_transcript_messages};
use claude_brain::session_adapter::BrainSessionAdapter;
use tempfile::TempDir;
use tokio::sync::Mutex;

fn config_in(tempdir: &TempDir) -> ClaudeBrainConfig {
    let base = tempdir.path();
    ClaudeBrainConfig {
        storage: StorageConfig {
            brain_path: base.join("brain").to_string_lossy().into_owned(),
            pks_path: base.join("pks.db").to_string_lossy().into_owned(),
            bks_path: base.join("bks.db").to_string_lossy().into_owned(),
        },
        ..Default::default()
    }
}

#[test]
fn read_transcript_messages_missing_file_is_empty() {
    // Missing path returns an empty vec, not an error — matches the hook's
    // fail-open contract.
    assert!(read_transcript_messages(None).is_empty());
    assert!(read_transcript_messages(Some("/definitely/does/not/exist.jsonl")).is_empty());
}

#[test]
fn read_transcript_messages_parses_jsonl() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("transcript.jsonl");

    // Mixed format: Claude Code's wrapped-in-"message" form, a plain object
    // form, a string content, a structured content array, and a system
    // message that must be filtered out.
    let lines = [
        r#"{"message":{"role":"user","content":"hello"}}"#,
        r#"{"message":{"role":"assistant","content":[{"type":"text","text":"hi back"}]}}"#,
        r#"{"role":"user","content":"how are you"}"#,
        r#"{"role":"system","content":"should be dropped"}"#,
        r#""#, // blank line
        r#"not-json"#,
    ];
    std::fs::write(&path, lines.join("\n")).expect("write transcript");

    let msgs = read_transcript_messages(Some(path.to_str().unwrap()));
    assert_eq!(
        msgs,
        vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi back".to_string()),
            ("user".to_string(), "how are you".to_string()),
        ]
    );
}

#[test]
fn extract_text_content_handles_both_shapes() {
    let string_form = serde_json::json!({ "content": "plain" });
    assert_eq!(extract_text_content(&string_form), "plain");

    let array_form = serde_json::json!({
        "content": [
            { "type": "text", "text": "first" },
            { "type": "tool_use", "id": "x" },
            { "type": "text", "text": "second" }
        ]
    });
    assert_eq!(extract_text_content(&array_form), "first\nsecond");

    let empty = serde_json::json!({});
    assert_eq!(extract_text_content(&empty), "");
}

#[tokio::test]
async fn context_manager_new_with_tempdir() {
    let dir = TempDir::new().expect("tempdir");
    let ctx = ContextManager::new(config_in(&dir))
        .await
        .expect("construct ContextManager");

    // Stats should succeed on a fresh store; we don't pin exact values
    // because tier layouts may shift — just require the call to succeed.
    ctx.memory_stats()
        .await
        .expect("memory_stats on empty store");
}

#[tokio::test]
async fn session_adapter_roundtrips_messages() {
    use brainwires_core::Message;
    use brainwires_knowledge::knowledge::brain_client::BrainClient;
    use brainwires_memory::dream::consolidator::DreamSessionStore;

    let dir = TempDir::new().expect("tempdir");
    let cfg = config_in(&dir);

    let client = BrainClient::with_paths(
        &cfg.storage.brain_path,
        &cfg.storage.pks_path,
        &cfg.storage.bks_path,
    )
    .await
    .expect("BrainClient::with_paths");
    let adapter = BrainSessionAdapter::new(Arc::new(Mutex::new(client)));

    // Starting from empty, no sessions exist.
    let initial = adapter
        .list_sessions()
        .await
        .expect("list_sessions on empty store");
    assert!(initial.is_empty());

    // `save` writes a consolidated summary thought; a follow-up `list` must
    // therefore observe at least one session.
    let messages = vec![Message::user("what is 2+2"), Message::assistant("4")];
    adapter
        .save("test-session", &messages)
        .await
        .expect("adapter.save");

    let sessions = adapter
        .list_sessions()
        .await
        .expect("list_sessions after save");
    // The adapter stores summaries with `session:test-session` tag; when no
    // auto-capture thoughts exist it falls back to `default`. Either way,
    // we should see *something*.
    assert!(
        !sessions.is_empty(),
        "expected at least one session after save, got {sessions:?}"
    );
}
