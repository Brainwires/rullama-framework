pub mod config;
pub mod context_manager;
pub mod hook_protocol;
pub mod hooks;
pub mod mcp_server;
pub mod session_adapter;

// ── Budget constants ───────────────────────────────────────────────────
/// Target fill ratio — keep post-compaction total at this fraction of the
/// compaction threshold so messages can accumulate before the next trigger.
const POST_COMPACT_TARGET_RATIO: f64 = 0.70;
/// Fraction of the target budget allocated to hook output
/// (the rest goes to system prompt + compaction summary).
const HOOK_SHARE_RATIO: f64 = 0.25;
/// Approximate characters per token for budget conversion.
const CHARS_PER_TOKEN: f64 = 4.0;
/// Minimum hook output budget (chars).
const MIN_OUTPUT_BUDGET: usize = 2_000;
/// Maximum hook output budget (chars).
const MAX_OUTPUT_BUDGET: usize = 40_000;

/// Truncate a string to at most `max_bytes` without splitting a UTF-8 character.
pub fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Sanitize a value used inside `Filter::Raw` SQL-like expressions.
/// Strips everything except alphanumeric, hyphen, and underscore.
pub fn sanitize_tag_value(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

/// Read Claude Code's compaction settings directly from settings files.
/// Checks project-level settings.local.json first, then global ~/.claude/settings.json.
/// Returns (window_tokens, compact_pct as 0..1 fraction).
fn read_claude_compact_settings() -> (usize, f64) {
    // Project-level settings take priority
    let project_settings = std::env::current_dir()
        .ok()
        .map(|d| d.join(".claude").join("settings.local.json"));
    let global_settings = dirs::home_dir().map(|d| d.join(".claude").join("settings.json"));

    let mut window_tokens: Option<usize> = None;
    let mut compact_pct: Option<f64> = None;

    // Read global first, then project overrides
    for path in [global_settings, project_settings].into_iter().flatten() {
        if let Ok(content) = std::fs::read_to_string(&path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(env) = json.get("env")
        {
            if let Some(val) = env
                .get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<usize>().ok())
            {
                window_tokens = Some(val);
            }
            if let Some(val) = env
                .get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<f64>().ok())
            {
                compact_pct = Some(val);
            }
        }
    }

    // Also check env vars (Claude Code may set these for child processes)
    if window_tokens.is_none() {
        window_tokens = std::env::var("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok());
    }
    if compact_pct.is_none() {
        compact_pct = std::env::var("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
            .ok()
            .and_then(|v| v.parse().ok());
    }

    let window = window_tokens.unwrap_or(200_000);
    let pct = compact_pct.unwrap_or(50.0) / 100.0;
    (window, pct)
}

/// Compute a safe character budget for hook output based on Claude Code's
/// compaction window settings read from settings files.
///
///   threshold  = window_tokens × compact_pct
///   target     = threshold × POST_COMPACT_TARGET_RATIO
///   hook_share = target × HOOK_SHARE_RATIO
///   budget     = hook_share × CHARS_PER_TOKEN
pub fn compute_output_budget() -> usize {
    let (window_tokens, compact_pct) = read_claude_compact_settings();

    let threshold = window_tokens as f64 * compact_pct;
    let target = threshold * POST_COMPACT_TARGET_RATIO;
    let hook_share = target * HOOK_SHARE_RATIO;
    let budget = (hook_share * CHARS_PER_TOKEN) as usize;
    budget.clamp(MIN_OUTPUT_BUDGET, MAX_OUTPUT_BUDGET)
}
