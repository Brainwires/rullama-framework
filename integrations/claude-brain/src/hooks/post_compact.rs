//! PostCompact hook — logging only.
//!
//! PostCompact stdout is NOT injected into Claude's context (only SessionStart
//! and UserPromptSubmit stdout reaches context). All context restoration is
//! handled by SessionStart when `source = "compact"`.

use anyhow::Result;

use crate::hook_protocol::{self, PostCompactPayload};

/// Handle the PostCompact hook event.
///
/// Logs the event for diagnostics. Does NOT write to stdout — Claude Code
/// ignores PostCompact stdout (debug log only).
pub async fn handle() -> Result<()> {
    let payload: PostCompactPayload = hook_protocol::read_payload().await?;

    let log_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".brainwires")
        .join("claude-brain-hooks.log");
    let _ = std::fs::create_dir_all(log_path.parent().unwrap_or(std::path::Path::new("/tmp")));
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let summary_len = payload
        .compact_summary
        .as_ref()
        .map(|s| s.len())
        .unwrap_or(0);
    let log_line = format!(
        "[{timestamp}] POST-COMPACT fired — summary {summary_len} chars, trigger={} (stdout ignored, context restored via SessionStart)\n",
        payload.trigger.as_deref().unwrap_or("?")
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));

    Ok(())
}
