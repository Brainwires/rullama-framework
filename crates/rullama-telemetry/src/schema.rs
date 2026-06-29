#[cfg(feature = "sqlite")]
use rusqlite::Connection;

/// SQLite DDL for the analytics database.
///
/// All tables use `IF NOT EXISTS` for idempotent initialization.
/// Call [`ensure_tables`] once after opening the connection.
#[cfg(feature = "sqlite")]
const SCHEMA_DDL: &str = r#"
-- Raw append-only event log. Every AnalyticsEvent lands here as JSON.
-- This is the source of truth; all other tables are derived from it.
CREATE TABLE IF NOT EXISTS analytics_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    session_id  TEXT,
    payload     TEXT    NOT NULL,
    recorded_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type
    ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_session
    ON analytics_events(session_id)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_recorded_at
    ON analytics_events(recorded_at);

-- Materialized: daily cost per provider/model.
-- Rebuilt by AnalyticsQuery::rebuild_summaries().
CREATE TABLE IF NOT EXISTS cost_by_model (
    date                    TEXT    NOT NULL,
    provider                TEXT    NOT NULL,
    model                   TEXT    NOT NULL,
    call_count              INTEGER NOT NULL DEFAULT 0,
    total_prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    total_completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost_usd          REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (date, provider, model)
);

-- Materialized: daily tool usage.
CREATE TABLE IF NOT EXISTS tool_usage (
    date        TEXT    NOT NULL,
    tool_name   TEXT    NOT NULL,
    call_count  INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, tool_name)
);

-- Materialized: daily agent run summaries.
CREATE TABLE IF NOT EXISTS agent_run_summaries (
    date            TEXT    NOT NULL PRIMARY KEY,
    total_runs      INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  REAL    NOT NULL DEFAULT 0.0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    avg_iterations  REAL    NOT NULL DEFAULT 0.0
);

-- Materialized: per-session aggregates.
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id      TEXT    NOT NULL PRIMARY KEY,
    first_event_at  INTEGER NOT NULL,
    last_event_at   INTEGER NOT NULL,
    provider_calls  INTEGER NOT NULL DEFAULT 0,
    agent_runs      INTEGER NOT NULL DEFAULT 0,
    tool_calls      INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  REAL    NOT NULL DEFAULT 0.0,
    total_tokens    INTEGER NOT NULL DEFAULT 0
);
"#;

/// Create all analytics tables if they don't already exist.
///
/// Safe to call multiple times — all statements use `IF NOT EXISTS`.
#[cfg(feature = "sqlite")]
pub fn ensure_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(SCHEMA_DDL)
        .map_err(|e| anyhow::anyhow!("Failed to create analytics schema: {e}"))?;
    Ok(())
}
