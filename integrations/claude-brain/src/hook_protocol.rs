//! Claude Code hook JSON protocol — stdin/stdout communication.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

/// Payload sent by Claude Code on SessionStart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartPayload {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Path to conversation transcript (JSONL).
    #[serde(default)]
    pub transcript_path: Option<String>,
    #[serde(default)]
    pub hook_event_name: Option<String>,
    /// "compact" when fired after compaction, absent on normal session start.
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// Payload sent by Claude Code on Stop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopPayload {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// The assistant's message for this turn.
    #[serde(default, alias = "last_assistant_message")]
    pub assistant_message: Option<String>,
    /// The user's message that triggered this turn.
    #[serde(default, alias = "last_user_message")]
    pub user_message: Option<String>,
}

/// Payload sent by Claude Code on PreCompact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreCompactPayload {
    #[serde(default)]
    pub session_id: Option<String>,
    /// Path to the full conversation transcript (JSONL file).
    #[serde(default)]
    pub transcript_path: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub hook_event_name: Option<String>,
    /// "auto" or "manual"
    #[serde(default)]
    pub trigger: Option<String>,
}

/// Payload sent by Claude Code on PostCompact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostCompactPayload {
    #[serde(default)]
    pub session_id: Option<String>,
    /// Path to the full conversation transcript (JSONL file).
    #[serde(default)]
    pub transcript_path: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub hook_event_name: Option<String>,
    /// "auto" or "manual"
    #[serde(default)]
    pub trigger: Option<String>,
    /// The compaction summary produced by Claude.
    #[serde(default)]
    pub compact_summary: Option<String>,
}

/// Read raw JSON from stdin (Claude Code hook protocol).
pub async fn read_stdin_json() -> Result<serde_json::Value> {
    let mut buf = String::new();
    tokio::io::stdin().read_to_string(&mut buf).await?;

    if buf.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str(&buf).context("Failed to parse hook JSON from stdin")
}

/// Read stdin and deserialize to a typed payload.
pub async fn read_payload<T: serde::de::DeserializeOwned>() -> Result<T> {
    let value = read_stdin_json().await?;
    serde_json::from_value(value).context("Failed to deserialize hook payload")
}

/// Write hook output to stdout.
pub fn write_output(output: &str) {
    print!("{output}");
}
