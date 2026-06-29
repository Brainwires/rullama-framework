//! SQLite schema for the billing ledger.

use anyhow::Context;
use rusqlite::Connection;

/// Create the billing tables if they don't already exist.
pub fn ensure_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS billing_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT    NOT NULL,
            kind        TEXT    NOT NULL,
            cost_usd    REAL    NOT NULL DEFAULT 0.0,
            payload     TEXT    NOT NULL,
            recorded_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_billing_agent  ON billing_events (agent_id);
        CREATE INDEX IF NOT EXISTS idx_billing_ts     ON billing_events (recorded_at);
        CREATE INDEX IF NOT EXISTS idx_billing_kind   ON billing_events (kind);",
    )
    .context("Failed to create billing_events table")
}
