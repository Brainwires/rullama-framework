// Integration tests for file operations
mod common;

use brainwires_cli::tools::FileOpsTool;
use brainwires_cli::types::tool::ToolContext;
use common::create_test_dir;
use serde_json::json;
use std::fs;

#[test]
fn test_file_operations_integration() {
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

    // Test 1: Write a file
    let write_input = json!({
        "path": "test_file.txt",
        "content": "Hello, integration test!"
    });

    let result = FileOpsTool::execute("test-id-1", "write_file", &write_input, &context);
    assert!(!result.is_error, "Write file should succeed");

    // Verify file was created
    let file_path = temp_dir.path().join("test_file.txt");
    assert!(file_path.exists(), "File should exist");

    // Test 2: Read the file
    let read_input = json!({
        "path": "test_file.txt"
    });

    let result = FileOpsTool::execute("test-id-2", "read_file", &read_input, &context);
    assert!(!result.is_error, "Read file should succeed");
    assert!(result.content.contains("Hello, integration test!"));

    // Test 3: Edit the file
    let edit_input = json!({
        "path": "test_file.txt",
        "old_text": "Hello",
        "new_text": "Goodbye"
    });

    let result = FileOpsTool::execute("test-id-3", "edit_file", &edit_input, &context);
    assert!(!result.is_error, "Edit file should succeed");

    // Verify edit worked
    let contents = fs::read_to_string(&file_path).unwrap();
    assert!(
        contents.contains("Goodbye"),
        "File should contain edited content"
    );

    // Test 4: List directory
    let list_input = json!({
        "path": "."
    });

    let result = FileOpsTool::execute("test-id-4", "list_directory", &list_input, &context);
    assert!(!result.is_error, "List directory should succeed");
    assert!(result.content.contains("test_file.txt"));

    // Test 5: Delete the file
    let delete_input = json!({
        "path": "test_file.txt"
    });

    let result = FileOpsTool::execute("test-id-5", "delete_file", &delete_input, &context);
    assert!(!result.is_error, "Delete file should succeed");
    assert!(!file_path.exists(), "File should be deleted");
}

#[test]
fn test_file_operations_error_cases() {
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

    // Test reading nonexistent file
    let read_input = json!({
        "path": "nonexistent.txt"
    });

    let result = FileOpsTool::execute("test-id-6", "read_file", &read_input, &context);
    assert!(result.is_error, "Reading nonexistent file should error");

    // Test deleting nonexistent file
    let delete_input = json!({
        "path": "nonexistent.txt"
    });

    let result = FileOpsTool::execute("test-id-7", "delete_file", &delete_input, &context);
    assert!(result.is_error, "Deleting nonexistent file should error");
}

#[test]
fn test_create_and_list_directory() {
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

    // Create a directory
    let create_input = json!({
        "path": "test_subdir"
    });

    let result = FileOpsTool::execute("test-id-8", "create_directory", &create_input, &context);
    assert!(!result.is_error, "Create directory should succeed");

    let dir_path = temp_dir.path().join("test_subdir");
    assert!(dir_path.exists(), "Directory should exist");
    assert!(dir_path.is_dir(), "Path should be a directory");
}
