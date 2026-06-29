//! SessionStart hook — load relevant context from all memory tiers.
//!
//! Routes by `source` field in the payload:
//! - "startup" / None → fresh session, load general context
//! - "compact"        → post-compaction restart, restore from Brainwires
//! - "resume"         → resumed session, load general context
//! - "clear"          → user cleared intentionally, emit nothing
//!
//! **Loop detection:** If the same session has fired `source=compact` more than
//! `MAX_COMPACTIONS_IN_WINDOW` times in `LOOP_WINDOW_SECS`, we're in a
//! compaction loop.  Emit nothing to break the cycle.

use anyhow::Result;

use crate::config::ClaudeBrainConfig;
use crate::context_manager::ContextManager;
use crate::hook_protocol::{self, SessionStartPayload};

/// Max compaction-source SessionStart events for one session within the window
/// before we consider it a loop and suppress output.
const MAX_COMPACTIONS_IN_WINDOW: usize = 2;
/// Time window (seconds) for loop detection.
const LOOP_WINDOW_SECS: u64 = 300; // 5 minutes

/// Check the hook log for recent `source=compact` entries for this session.
/// Returns the count found within `LOOP_WINDOW_SECS` of now.
fn count_recent_compactions(log_path: &std::path::Path, session_id: &str) -> usize {
    let content = match std::fs::read_to_string(log_path) {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let now = chrono::Utc::now();
    let cutoff = now - chrono::Duration::seconds(LOOP_WINDOW_SECS as i64);

    content
        .lines()
        .filter(|line| {
            line.contains("SESSION-START fired")
                && line.contains("source=compact")
                && line.contains(session_id)
        })
        .filter(|line| {
            // Parse timestamp from "[YYYY-MM-DD HH:MM:SS UTC]"
            line.get(1..24)
                .and_then(|ts| {
                    chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S UTC").ok()
                })
                .map(|ndt| ndt.and_utc() >= cutoff)
                .unwrap_or(false)
        })
        .count()
}

/// Handle the SessionStart hook event.
pub async fn handle() -> Result<()> {
    let payload: SessionStartPayload = hook_protocol::read_payload().await?;

    let log_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".brainwires")
        .join("claude-brain-hooks.log");
    let _ = std::fs::create_dir_all(log_path.parent().unwrap_or(std::path::Path::new("/tmp")));
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let source = payload.source.as_deref().unwrap_or("startup");
    let session = payload.session_id.as_deref().unwrap_or("?");

    // Route by source
    let (output, output_len) = match source {
        "clear" => {
            // User cleared intentionally — emit nothing
            let log_line = format!(
                "[{timestamp}] SESSION-START fired — source={source} cwd={} session={session} output=0\n",
                payload.cwd.as_deref().unwrap_or("?"),
            );
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));
            return Ok(());
        }
        "compact" => {
            // Loop detection: check how many compactions this session has had recently
            let recent_count = count_recent_compactions(&log_path, session);
            if recent_count >= MAX_COMPACTIONS_IN_WINDOW {
                let log_line = format!(
                    "[{timestamp}] SESSION-START fired — source={source} cwd={} session={session} output=0 (LOOP DETECTED: {recent_count} compactions in {}s, suppressing)\n",
                    payload.cwd.as_deref().unwrap_or("?"),
                    LOOP_WINDOW_SECS,
                );
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));
                return Ok(());
            }

            // Post-compaction restart — restore context from Brainwires
            let config = ClaudeBrainConfig::load()?;
            let ctx = ContextManager::new(config).await?;
            let context = ctx
                .load_post_compact_context(payload.cwd.as_deref(), payload.session_id.as_deref())
                .await?;
            let len = context.len();
            if !context.is_empty() {
                hook_protocol::write_output(&context);
            }
            (Some(context), len)
        }
        // "startup" | "resume" | anything else → fresh/resumed session context
        _ => {
            let config = ClaudeBrainConfig::load()?;
            let ctx = ContextManager::new(config).await?;
            let context = ctx
                .load_session_context(payload.cwd.as_deref(), payload.session_id.as_deref())
                .await?;
            let len = context.len();
            if !context.is_empty() {
                hook_protocol::write_output(&context);
            }
            (Some(context), len)
        }
    };

    // Log with output size for diagnostics
    let budget = crate::compute_output_budget();
    let log_line = format!(
        "[{timestamp}] SESSION-START fired — source={source} cwd={} session={session} output={output_len} budget={budget}\n",
        payload.cwd.as_deref().unwrap_or("?"),
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, log_line.as_bytes()));

    drop(output); // silence unused warning

    Ok(())
}
