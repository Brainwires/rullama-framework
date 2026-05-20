//! Generates canonical JSON fixture files for cross-language serialization
//! verification between Rust and Deno.
//!
//! Run explicitly with:
//!   cargo test -p brainwires-core -- --ignored generate_json_fixtures

use brainwires_core::message::{ContentBlock, Message, Role, Usage};
use brainwires_core::permission::PermissionMode;
use brainwires_core::plan::PlanStatus;
use brainwires_core::provider::ChatOptions;
use brainwires_core::search::SearchResult;
use brainwires_core::task::{Task, TaskPriority, TaskStatus};
use brainwires_core::tool::{Tool, ToolInputSchema, ToolResult};
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Write a value as pretty-printed JSON to the fixtures directory.
fn write_fixture<T: Serialize>(dir: &Path, name: &str, value: &T) {
    let json = serde_json::to_string_pretty(value).expect("failed to serialize fixture");
    let path = dir.join(format!("{}.json", name));
    fs::write(&path, json).unwrap_or_else(|e| panic!("failed to write {}: {}", path.display(), e));
    eprintln!("  wrote {}", path.display());
}

#[test]
#[ignore]
fn generate_json_fixtures() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deno/fixtures");
    fs::create_dir_all(&fixtures_dir).expect("failed to create fixtures directory");

    eprintln!("Generating fixtures in {}", fixtures_dir.display());

    // ── 1. Role ────────────────────────────────────────────────────────
    write_fixture(&fixtures_dir, "role_user", &Role::User);
    write_fixture(&fixtures_dir, "role_assistant", &Role::Assistant);
    write_fixture(&fixtures_dir, "role_system", &Role::System);
    write_fixture(&fixtures_dir, "role_tool", &Role::Tool);

    // ── 2. Message ─────────────────────────────────────────────────────
    write_fixture(
        &fixtures_dir,
        "message_user",
        &Message::user("Hello, how are you?"),
    );
    write_fixture(
        &fixtures_dir,
        "message_assistant",
        &Message::assistant("I'm doing well, thank you!"),
    );
    write_fixture(
        &fixtures_dir,
        "message_tool_result",
        &Message::tool_result("tool-call-001", "File contents: fn main() {}"),
    );

    // ── 3. ContentBlock ────────────────────────────────────────────────
    write_fixture(
        &fixtures_dir,
        "content_block_text",
        &ContentBlock::Text {
            text: "Here is the analysis result.".to_string(),
        },
    );
    write_fixture(
        &fixtures_dir,
        "content_block_tool_use",
        &ContentBlock::ToolUse {
            id: "tool-call-001".to_string(),
            name: "read_file".to_string(),
            input: json!({"path": "/src/main.rs"}),
        },
    );
    write_fixture(
        &fixtures_dir,
        "content_block_tool_result",
        &ContentBlock::ToolResult {
            tool_use_id: "tool-call-001".to_string(),
            content: "fn main() { println!(\"hello\"); }".to_string(),
            is_error: Some(false),
        },
    );

    // ── 4. Tool ────────────────────────────────────────────────────────
    let mut props = HashMap::new();
    props.insert(
        "path".to_string(),
        json!({"type": "string", "description": "The file path to read"}),
    );
    let tool = Tool {
        name: "read_file".to_string(),
        description: "Read the contents of a file at the given path".to_string(),
        input_schema: ToolInputSchema::object(props, vec!["path".to_string()]),
        requires_approval: false,
        defer_loading: false,
        allowed_callers: vec![],
        input_examples: vec![],
        serialize: false,
    };
    write_fixture(&fixtures_dir, "tool_sample", &tool);

    // ── 5. ToolResult ──────────────────────────────────────────────────
    write_fixture(
        &fixtures_dir,
        "tool_result_success",
        &ToolResult::success("tool-call-001", "Operation completed successfully"),
    );
    write_fixture(
        &fixtures_dir,
        "tool_result_error",
        &ToolResult::error("tool-call-002", "File not found: /nonexistent.txt"),
    );

    // ── 6. ChatOptions ─────────────────────────────────────────────────
    write_fixture(
        &fixtures_dir,
        "chat_options_default",
        &ChatOptions::default(),
    );
    let custom_opts = ChatOptions {
        temperature: Some(0.0),
        max_tokens: Some(100),
        top_p: Some(0.9),
        stop: Some(vec!["\n".to_string(), "END".to_string()]),
        system: Some("You are a helpful assistant.".to_string()),
        model: None,
        cache_strategy: Default::default(),
    };
    write_fixture(&fixtures_dir, "chat_options_custom", &custom_opts);

    // ── 7. TaskStatus ──────────────────────────────────────────────────
    write_fixture(&fixtures_dir, "task_status_pending", &TaskStatus::Pending);
    write_fixture(
        &fixtures_dir,
        "task_status_inprogress",
        &TaskStatus::InProgress,
    );
    write_fixture(
        &fixtures_dir,
        "task_status_completed",
        &TaskStatus::Completed,
    );
    write_fixture(&fixtures_dir, "task_status_failed", &TaskStatus::Failed);
    write_fixture(&fixtures_dir, "task_status_blocked", &TaskStatus::Blocked);
    write_fixture(&fixtures_dir, "task_status_skipped", &TaskStatus::Skipped);

    // ── 8. TaskPriority ────────────────────────────────────────────────
    write_fixture(&fixtures_dir, "task_priority_low", &TaskPriority::Low);
    write_fixture(&fixtures_dir, "task_priority_normal", &TaskPriority::Normal);
    write_fixture(&fixtures_dir, "task_priority_high", &TaskPriority::High);
    write_fixture(&fixtures_dir, "task_priority_urgent", &TaskPriority::Urgent);

    // ── 9. Task ────────────────────────────────────────────────────────
    let task = Task {
        id: "task-001".to_string(),
        description: "Implement user authentication module".to_string(),
        status: TaskStatus::Pending,
        plan_id: None,
        parent_id: None,
        children: vec![],
        depends_on: vec![],
        priority: TaskPriority::Normal,
        assigned_to: None,
        iterations: 0,
        summary: None,
        created_at: 1_700_000_000,
        updated_at: 1_700_000_000,
        started_at: None,
        completed_at: None,
    };
    write_fixture(&fixtures_dir, "task_sample", &task);

    // ── 10. PlanStatus ─────────────────────────────────────────────────
    write_fixture(&fixtures_dir, "plan_status_draft", &PlanStatus::Draft);
    write_fixture(&fixtures_dir, "plan_status_active", &PlanStatus::Active);
    write_fixture(&fixtures_dir, "plan_status_paused", &PlanStatus::Paused);
    write_fixture(
        &fixtures_dir,
        "plan_status_completed",
        &PlanStatus::Completed,
    );
    write_fixture(
        &fixtures_dir,
        "plan_status_abandoned",
        &PlanStatus::Abandoned,
    );

    // ── 11. Usage ──────────────────────────────────────────────────────
    write_fixture(&fixtures_dir, "usage_sample", &Usage::new(150, 50));

    // ── 12. SearchResult ───────────────────────────────────────────────
    let search_result = SearchResult {
        file_path: "src/main.rs".to_string(),
        root_path: Some("/home/user/project".to_string()),
        content: "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
        score: 0.95,
        vector_score: 0.92,
        keyword_score: Some(0.88),
        start_line: 1,
        end_line: 3,
        language: "rust".to_string(),
        project: Some("my-project".to_string()),
        indexed_at: 1_700_000_000,
    };
    write_fixture(&fixtures_dir, "search_result_sample", &search_result);

    // ── 13. PermissionMode ─────────────────────────────────────────────
    write_fixture(
        &fixtures_dir,
        "permission_mode_read_only",
        &PermissionMode::ReadOnly,
    );
    write_fixture(&fixtures_dir, "permission_mode_auto", &PermissionMode::Auto);
    write_fixture(&fixtures_dir, "permission_mode_full", &PermissionMode::Full);

    eprintln!("All fixtures generated successfully.");
}
