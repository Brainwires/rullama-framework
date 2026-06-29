//! Stop hook — capture assistant turn into hot-tier storage.

use anyhow::Result;

use crate::config::ClaudeBrainConfig;
use crate::context_manager::ContextManager;
use crate::hook_protocol::{self, StopPayload};
use crate::sanitize_tag_value;

/// Handle the Stop hook event.
///
/// Captures the assistant's response (and optionally the user's prompt)
/// into Brainwires' hot-tier storage for future recall.
pub async fn handle() -> Result<()> {
    let payload: StopPayload = hook_protocol::read_payload().await?;
    let config = ClaudeBrainConfig::load()?;
    let ctx = ContextManager::new(config).await?;

    // Log to file
    let log_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".brainwires")
        .join("claude-brain-hooks.log");
    let _ = std::fs::create_dir_all(log_path.parent().unwrap_or(std::path::Path::new("/tmp")));
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let msg_len = payload
        .assistant_message
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(0);
    let log_line = format!("[{timestamp}] STOP fired — assistant_message {msg_len} chars\n");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));

    let session_tag = payload
        .session_id
        .as_deref()
        .map(|id| format!("session:{}", sanitize_tag_value(id)))
        .unwrap_or_else(|| "session:default".to_string());

    // Derive project tag from cwd
    let project_tag = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .map(|name| format!("project:{}", sanitize_tag_value(&name)));

    // Capture assistant message
    if let Some(ref msg) = payload.assistant_message {
        // Skip very short messages (likely just tool calls)
        if msg.len() > 50 {
            let content = format!("[assistant] {msg}");
            let mut tags = vec![
                "claude-code".to_string(),
                "auto-capture".to_string(),
                session_tag.clone(),
            ];
            if let Some(ref pt) = project_tag {
                tags.push(pt.clone());
            }

            if let Some(ref reason) = payload.stop_reason {
                tags.push(format!("stop:{reason}"));
            }

            let mut client = ctx.client().lock_owned().await;
            client
                .capture_thought(
                    brainwires_knowledge::knowledge::types::CaptureThoughtRequest {
                        content,
                        category: None,
                        tags: Some(tags),
                        importance: None,
                        source: Some("claude-code-turn".to_string()),
                        owner_id: None,
                    },
                )
                .await?;
        }
    }

    // Capture user message if present
    if let Some(ref msg) = payload.user_message
        && msg.len() > 20
    {
        let content = format!("[user] {msg}");
        let mut tags = vec![
            "claude-code".to_string(),
            "auto-capture".to_string(),
            session_tag,
        ];
        if let Some(ref pt) = project_tag {
            tags.push(pt.clone());
        }

        let mut client = ctx.client().lock_owned().await;
        client
            .capture_thought(
                brainwires_knowledge::knowledge::types::CaptureThoughtRequest {
                    content,
                    category: None,
                    tags: Some(tags),
                    importance: None,
                    source: Some("claude-code-turn".to_string()),
                    owner_id: None,
                },
            )
            .await?;
    }

    Ok(())
}
