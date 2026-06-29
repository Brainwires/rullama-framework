use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use async_trait::async_trait;
use rusqlite::{Connection, params};

use crate::{AnalyticsError, AnalyticsEvent, AnalyticsSink, schema};

/// Rusqlite-backed durable analytics sink.
///
/// Persists every [`AnalyticsEvent`] as JSON into the `analytics_events` table
/// at `~/.rullama/analytics/analytics.db` by default.
///
/// Follows the same patterns as `LockStore`:
/// - `Mutex<Connection>` — single-writer, drain task is the sole caller
/// - WAL mode for concurrent read access from [`crate::query::AnalyticsQuery`]
/// - Idempotent schema initialization via [`schema::ensure_tables`]
///
/// The `record()` method uses `tokio::task::block_in_place` so it is safe to
/// call from within an async context without blocking the executor thread pool.
pub struct SqliteAnalyticsSink {
    conn: Mutex<Connection>,
}

impl SqliteAnalyticsSink {
    /// Open (or create) the analytics database at the default path.
    ///
    /// Default: `~/.rullama/analytics/analytics.db`
    pub fn new_default() -> anyhow::Result<Self> {
        let db_path = Self::default_db_path()?;
        Self::new_with_path(&db_path)
    }

    /// Open (or create) the analytics database at a custom path.
    ///
    /// Creates parent directories as needed.
    pub fn new_with_path(db_path: &PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create analytics directory at {parent:?}"))?;
        }

        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open analytics DB at {db_path:?}"))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA busy_timeout=5000;
             PRAGMA synchronous=NORMAL;",
        )
        .context("Failed to configure SQLite pragmas")?;

        schema::ensure_tables(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn default_db_path() -> anyhow::Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home
            .join(".rullama")
            .join("analytics")
            .join("analytics.db"))
    }

    fn insert_event(conn: &Connection, event: &AnalyticsEvent) -> anyhow::Result<()> {
        let event_type = event.event_type();
        let session_id = event.session_id();
        let payload = serde_json::to_string(event).context("Failed to serialize AnalyticsEvent")?;
        let recorded_at = event.timestamp().timestamp_millis();

        conn.execute(
            "INSERT INTO analytics_events (event_type, session_id, payload, recorded_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![event_type, session_id, payload, recorded_at],
        )
        .context("Failed to insert analytics event")?;

        Ok(())
    }
}

#[async_trait]
impl AnalyticsSink for SqliteAnalyticsSink {
    async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
        // rusqlite is synchronous. block_in_place allows blocking within an
        // async context without starving the tokio thread pool.
        tokio::task::block_in_place(|| {
            let conn = self.conn.lock().expect("analytics DB lock poisoned");
            Self::insert_event(&conn, &event).map_err(AnalyticsError::Other)
        })
    }

    /// Checkpoint the WAL file so all written events are durable in the main DB file.
    ///
    /// Without this, recent events are only in the `-wal` sidecar and may be lost
    /// if the process is killed before SQLite performs an automatic checkpoint.
    async fn flush(&self) -> Result<(), AnalyticsError> {
        tokio::task::block_in_place(|| {
            let conn = self.conn.lock().expect("analytics DB lock poisoned");
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                .map_err(|e| AnalyticsError::Other(anyhow::anyhow!(e)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;

    fn make_sink() -> (SqliteAnalyticsSink, TempDir) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("analytics.db");
        let sink = SqliteAnalyticsSink::new_with_path(&path).unwrap();
        (sink, tmp)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_creates_schema() {
        let (_sink, tmp) = make_sink();
        let path = tmp.path().join("analytics.db");
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='analytics_events'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_records_event() {
        let (sink, tmp) = make_sink();
        sink.record(AnalyticsEvent::Custom {
            session_id: Some("s1".into()),
            name: "test".into(),
            payload: serde_json::Value::Null,
            timestamp: Utc::now(),
        })
        .await
        .unwrap();

        let path = tmp.path().join("analytics.db");
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM analytics_events WHERE event_type='custom'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
