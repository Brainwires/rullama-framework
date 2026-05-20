//! PreCompact hook — export conversation from transcript file to Brainwires before compaction.

use anyhow::Result;
use std::collections::HashSet;
use std::io::BufRead;

use brainwires_storage::{FieldValue, Filter};

use crate::config::ClaudeBrainConfig;
use crate::context_manager::ContextManager;
use crate::hook_protocol::{self, PreCompactPayload};
use crate::{sanitize_tag_value, truncate_utf8};

/// Max chars to store per individual thought (truncate beyond this).
const MAX_THOUGHT_CHARS: usize = 2_000;
/// Prefix length used for deduplication matching.
const DEDUP_PREFIX_LEN: usize = 150;
/// Minimum message length worth capturing.
const MIN_MESSAGE_LEN: usize = 20;
/// Max char budget for the session digest content.
const DIGEST_MAX_CHARS: usize = 1_500;
/// Max preview length per message in the digest.
const DIGEST_PREVIEW_LEN: usize = 100;
/// Max existing thoughts to query for dedup check.
const DEDUP_QUERY_LIMIT: usize = 500;

/// Handle the PreCompact hook event.
///
/// Claude Code sends `transcript_path` pointing to the JSONL conversation file.
/// We read it, extract user/assistant messages, and store them in Brainwires
/// before compaction destroys the full context.
pub async fn handle() -> Result<()> {
    let payload: PreCompactPayload = hook_protocol::read_payload().await?;
    let config = ClaudeBrainConfig::load()?;
    let ctx = ContextManager::new(config).await?;

    let session_tag = payload
        .session_id
        .as_deref()
        .map(|id| format!("session:{}", sanitize_tag_value(id)))
        .unwrap_or_else(|| "session:default".to_string());

    // Read messages from transcript file
    let messages = read_transcript_messages(payload.transcript_path.as_deref());
    let msg_count = messages.len();

    // Log
    let log_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".brainwires")
        .join("claude-brain-hooks.log");
    let _ = std::fs::create_dir_all(log_path.parent().unwrap_or(std::path::Path::new("/tmp")));
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let log_line = format!(
        "[{timestamp}] PRE-COMPACT fired — {msg_count} messages from transcript (trigger={})\n",
        payload.trigger.as_deref().unwrap_or("?")
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));

    // Query existing thoughts for this session to avoid duplicates from Stop hook
    let existing_prefixes: HashSet<String> = {
        let arc = ctx.client();
        let client = arc.lock().await;
        let filter = Filter::And(vec![
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
            Filter::Raw(format!(
                "tags LIKE '%auto-capture%' AND tags LIKE '%{}%'",
                session_tag
            )),
        ]);
        let contents = client
            .query_thought_contents(&filter, DEDUP_QUERY_LIMIT)
            .await
            .unwrap_or_default();
        contents
            .into_iter()
            .map(|c| truncate_utf8(&c, DEDUP_PREFIX_LEN).to_string())
            .collect()
    };

    // Build batch of capture requests, skipping already-captured messages
    let requests: Vec<brainwires_knowledge::knowledge::types::CaptureThoughtRequest> = messages
        .iter()
        .filter(|(_, content)| content.len() >= MIN_MESSAGE_LEN)
        .filter(|(role, content)| {
            let tagged = format!("[{role}] {content}");
            !existing_prefixes.contains(truncate_utf8(&tagged, DEDUP_PREFIX_LEN))
        })
        .map(|(role, content)| {
            let tagged_content = format!("[{role}] {content}");
            let store_content = if tagged_content.len() > MAX_THOUGHT_CHARS {
                format!(
                    "{}...[truncated]",
                    truncate_utf8(&tagged_content, MAX_THOUGHT_CHARS)
                )
            } else {
                tagged_content
            };
            brainwires_knowledge::knowledge::types::CaptureThoughtRequest {
                content: store_content,
                category: None,
                tags: Some(vec![
                    "claude-code".to_string(),
                    "auto-capture".to_string(),
                    "pre-compact".to_string(),
                    session_tag.clone(),
                ]),
                importance: None,
                source: Some("pre-compact-export".to_string()),
                owner_id: None,
            }
        })
        .collect();

    // Single lock, single embed, single insert
    let batch_count = requests.len();
    let mut client = ctx.client().lock_owned().await;
    let stored = client.capture_thoughts_batch(requests).await.unwrap_or(0);

    // Create a session digest for PostCompact to find
    if !messages.is_empty() {
        let mut digest_parts: Vec<String> = Vec::new();
        let mut total_len = 0;
        for (role, content) in &messages {
            if total_len >= DIGEST_MAX_CHARS {
                break;
            }
            let preview = truncate_utf8(content, DIGEST_PREVIEW_LEN);
            let part = format!("[{role}] {preview}");
            total_len += part.len();
            digest_parts.push(part);
        }
        let digest_content = digest_parts.join("\n");
        let _ = client
            .capture_thought(
                brainwires_knowledge::knowledge::types::CaptureThoughtRequest {
                    content: digest_content,
                    category: Some("insight".to_string()),
                    tags: Some(vec![
                        "session-digest".to_string(),
                        session_tag.clone(),
                        "claude-code".to_string(),
                    ]),
                    importance: Some(0.9),
                    source: Some("pre-compact-digest".to_string()),
                    owner_id: None,
                },
            )
            .await;
    }

    drop(client);

    tracing::info!(
        "Pre-compact: batch-stored {stored}/{batch_count} messages (from {msg_count} total)"
    );

    Ok(())
}

/// Read the JSONL transcript file and extract (role, content) pairs.
///
/// Each line is a JSON object. We look for objects with `role` and `content` fields
/// (the standard Claude API message format). Content can be a string or an array
/// of content blocks — we extract text from both.
///
/// Exposed for integration testing; the hook handler is still the sole
/// production caller.
pub fn read_transcript_messages(path: Option<&str>) -> Vec<(String, String)> {
    let Some(path) = path else {
        return Vec::new();
    };

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = std::io::BufReader::new(file);
    let mut messages = Vec::new();

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }

        let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };

        // Claude Code transcript format: messages nested under "message" key
        let msg_obj = if let Some(msg) = obj.get("message") {
            msg
        } else {
            &obj
        };

        let Some(role) = msg_obj.get("role").and_then(|v| v.as_str()) else {
            continue;
        };

        // Only capture user and assistant messages
        if role != "user" && role != "assistant" {
            continue;
        }

        let content = extract_text_content(msg_obj);
        if !content.is_empty() {
            messages.push((role.to_string(), content));
        }
    }

    messages
}

/// Extract text content from a message object.
/// Handles both `"content": "string"` and `"content": [{"type":"text","text":"..."}]`.
///
/// Exposed for integration testing.
pub fn extract_text_content(msg: &serde_json::Value) -> String {
    let Some(content) = msg.get("content") else {
        return String::new();
    };

    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => {
            let mut texts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    texts.push(text);
                }
            }
            texts.join("\n")
        }
        _ => String::new(),
    }
}
