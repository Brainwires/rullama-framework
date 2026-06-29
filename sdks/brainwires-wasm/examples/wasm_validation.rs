//! Example: Message and tool validation (native binary)
//!
//! Demonstrates the same validation logic that the WASM `validate_message`,
//! `validate_tool`, and `serialize_history` functions use — parsing JSON into
//! the core Brainwires types and re-serializing to normalize them.
//!
//! Because the WASM entry points are behind `#[wasm_bindgen]`, we call the
//! underlying `serde_json` round-trip directly using `brainwires_core` types
//! that the WASM crate re-exports.
//!
//! Run: cargo run -p brainwires-wasm --example wasm_validation

use brainwires_wasm::brainwires_core::{Message, Tool, serialize_messages_to_stateless_history};

// ── Helpers that mirror the WASM bindings ───────────────────────────────

/// Validate and normalize a JSON-encoded message (mirrors `validate_message`).
fn validate_message(json: &str) -> Result<String, String> {
    let msg: Message =
        serde_json::from_str(json).map_err(|e| format!("Invalid message JSON: {e}"))?;
    serde_json::to_string_pretty(&msg).map_err(|e| format!("Serialization error: {e}"))
}

/// Validate and normalize a JSON-encoded tool definition (mirrors `validate_tool`).
fn validate_tool(json: &str) -> Result<String, String> {
    let tool: Tool = serde_json::from_str(json).map_err(|e| format!("Invalid tool JSON: {e}"))?;
    serde_json::to_string_pretty(&tool).map_err(|e| format!("Serialization error: {e}"))
}

/// Serialize a conversation to stateless history format (mirrors `serialize_history`).
fn serialize_history(messages_json: &str) -> Result<String, String> {
    let messages: Vec<Message> =
        serde_json::from_str(messages_json).map_err(|e| format!("Invalid messages JSON: {e}"))?;
    let history = serialize_messages_to_stateless_history(&messages);
    serde_json::to_string_pretty(&history).map_err(|e| format!("Serialization error: {e}"))
}

fn main() {
    // ── 1. Validate a well-formed message ───────────────────────────────
    println!("=== Valid message ===");
    let valid_msg = r#"{"role": "user", "content": "Hello, world!"}"#;
    match validate_message(valid_msg) {
        Ok(normalized) => println!("  Normalized:\n  {normalized}"),
        Err(e) => println!("  ERROR: {e}"),
    }

    // ── 2. Validate a message with extra fields (stripped on round-trip) ─
    println!("\n=== Message with extra fields ===");
    let extra_fields = r#"{
        "role": "assistant",
        "content": "I can help with that.",
        "unknown_field": 42
    }"#;
    match validate_message(extra_fields) {
        Ok(normalized) => println!("  Normalized (extra fields stripped):\n  {normalized}"),
        Err(e) => println!("  ERROR: {e}"),
    }

    // ── 3. Invalid message (missing required field) ─────────────────────
    println!("\n=== Invalid message (missing role) ===");
    let invalid_msg = r#"{"content": "no role!"}"#;
    match validate_message(invalid_msg) {
        Ok(_) => println!("  Unexpectedly succeeded"),
        Err(e) => println!("  Expected error: {e}"),
    }

    // ── 4. Invalid message (bad JSON syntax) ────────────────────────────
    println!("\n=== Malformed JSON ===");
    let bad_json = r#"{"role": "user", content: }"#;
    match validate_message(bad_json) {
        Ok(_) => println!("  Unexpectedly succeeded"),
        Err(e) => println!("  Expected error: {e}"),
    }

    // ── 5. Validate a well-formed tool definition ───────────────────────
    println!("\n=== Valid tool ===");
    let valid_tool = r#"{
        "name": "calculator",
        "description": "Performs basic arithmetic",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "Math expression" }
            },
            "required": ["expression"]
        }
    }"#;
    match validate_tool(valid_tool) {
        Ok(normalized) => println!("  Normalized:\n  {normalized}"),
        Err(e) => println!("  ERROR: {e}"),
    }

    // ── 6. Invalid tool (not an object) ─────────────────────────────────
    println!("\n=== Invalid tool (array instead of object) ===");
    let invalid_tool = r#"[1, 2, 3]"#;
    match validate_tool(invalid_tool) {
        Ok(_) => println!("  Unexpectedly succeeded"),
        Err(e) => println!("  Expected error: {e}"),
    }

    // ── 7. Serialize conversation history ───────────────────────────────
    println!("\n=== Serialize history ===");
    let conversation = r#"[
        {"role": "user", "content": "What is Rust?"},
        {"role": "assistant", "content": "Rust is a systems programming language focused on safety and performance."},
        {"role": "user", "content": "How do I install it?"}
    ]"#;
    match serialize_history(conversation) {
        Ok(history) => println!("  Stateless history:\n  {history}"),
        Err(e) => println!("  ERROR: {e}"),
    }

    // ── 8. Invalid history (not an array) ───────────────────────────────
    println!("\n=== Invalid history (not an array) ===");
    let not_array = r#"{"role": "user", "content": "single message, not array"}"#;
    match serialize_history(not_array) {
        Ok(_) => println!("  Unexpectedly succeeded"),
        Err(e) => println!("  Expected error: {e}"),
    }

    println!("\nAll validation demos complete.");
}
