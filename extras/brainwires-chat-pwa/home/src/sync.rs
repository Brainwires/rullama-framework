use std::path::Path;

use anyhow::{Context, Result};
use rsqlite::core::database::Database;
use rsqlite::core::types::Value;
use rsqlite::vfs::native::NativeVfs;

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS sync_entries (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id   TEXT NOT NULL,
        table_name  TEXT NOT NULL,
        row_key     TEXT NOT NULL,
        op          TEXT NOT NULL,
        ts          INTEGER NOT NULL,
        snapshot    TEXT,
        received_at INTEGER NOT NULL
    );",
    "CREATE INDEX IF NOT EXISTS idx_sync_device ON sync_entries(device_id, id);",
    "CREATE TABLE IF NOT EXISTS sync_cursors (
        device_id     TEXT PRIMARY KEY,
        acked_seq     INTEGER NOT NULL DEFAULT 0
    );",
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncEntry {
    #[serde(alias = "table_name")]
    pub table: String,
    pub row_key: String,
    pub op: String,
    pub ts: i64,
    #[serde(default)]
    pub device_id: String,
    pub snapshot: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoredSyncEntry {
    pub seq: i64,
    pub table: String,
    pub row_key: String,
    pub op: String,
    pub ts: i64,
    pub device_id: String,
    pub snapshot: Option<String>,
    pub received_at: i64,
}

#[derive(Debug)]
pub struct PullResult {
    pub entries: Vec<StoredSyncEntry>,
    pub has_more: bool,
    pub latest_seq: i64,
}

pub struct SyncStore {
    db_path: String,
}

// NativeVfs is Send and SyncStore owns only a String, so this is safe.
// Each method opens the DB fresh — rsqlite is lightweight and sync ops
// are infrequent.
unsafe impl Send for SyncStore {}
unsafe impl Sync for SyncStore {}

impl SyncStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let db_path = path
            .as_ref()
            .to_str()
            .context("sync db path must be valid UTF-8")?
            .to_string();

        let vfs = NativeVfs::new();
        let exists = Path::new(&db_path).exists();
        let mut db = if exists {
            Database::open(&vfs, &db_path).with_context(|| format!("open sync db at {db_path}"))?
        } else {
            Database::create(&vfs, &db_path)
                .with_context(|| format!("create sync db at {db_path}"))?
        };

        for ddl in SCHEMA {
            db.execute(ddl)
                .with_context(|| format!("sync schema init: {ddl}"))?;
        }

        tracing::debug!(path = %db_path, "sync store initialized");
        Ok(Self { db_path })
    }

    fn open_db(&self) -> Result<Database> {
        let vfs = NativeVfs::new();
        Database::open(&vfs, &self.db_path)
            .with_context(|| format!("reopen sync db at {}", self.db_path))
    }

    pub fn push(&self, device_id: &str, entries: &[SyncEntry]) -> Result<i64> {
        let mut db = self.open_db()?;
        let now = chrono::Utc::now().timestamp_millis();

        for e in entries {
            db.execute_with_params(
                "INSERT INTO sync_entries (device_id, table_name, row_key, op, ts, snapshot, received_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                vec![
                    Value::Text(device_id.to_string()),
                    Value::Text(e.table.clone()),
                    Value::Text(e.row_key.clone()),
                    Value::Text(e.op.clone()),
                    Value::Integer(e.ts),
                    match &e.snapshot {
                        Some(s) => Value::Text(s.clone()),
                        None => Value::Null,
                    },
                    Value::Integer(now),
                ],
            ).context("sync push: insert entry")?;
        }

        let result = db
            .query("SELECT MAX(id) FROM sync_entries;")
            .context("sync push: query max id")?;
        let latest_seq = result
            .rows
            .first()
            .and_then(|r| r.values.first())
            .and_then(|v| match v {
                Value::Integer(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);

        tracing::debug!(
            device_id,
            count = entries.len(),
            latest_seq,
            "sync push accepted",
        );
        Ok(latest_seq)
    }

    pub fn pull(&self, device_id: &str, since_seq: i64, limit: usize) -> Result<PullResult> {
        let mut db = self.open_db()?;

        let fetch_limit = limit + 1;
        let sql = format!(
            "SELECT id, device_id, table_name, row_key, op, ts, snapshot, received_at FROM sync_entries WHERE device_id != ? AND id > ? ORDER BY id LIMIT {fetch_limit};"
        );
        let result = db
            .query_with_params(
                &sql,
                vec![
                    Value::Text(device_id.to_string()),
                    Value::Integer(since_seq),
                ],
            )
            .context("sync pull: query entries")?;

        let mut entries: Vec<StoredSyncEntry> = result
            .rows
            .iter()
            .map(|row| StoredSyncEntry {
                seq: extract_i64(&row.values, 0),
                device_id: extract_text(&row.values, 1),
                table: extract_text(&row.values, 2),
                row_key: extract_text(&row.values, 3),
                op: extract_text(&row.values, 4),
                ts: extract_i64(&row.values, 5),
                snapshot: extract_opt_text(&row.values, 6),
                received_at: extract_i64(&row.values, 7),
            })
            .collect();

        let has_more = entries.len() > limit;
        if has_more {
            entries.truncate(limit);
        }

        let latest_seq = entries.last().map(|e| e.seq).unwrap_or(since_seq);

        tracing::debug!(
            device_id,
            since_seq,
            returned = entries.len(),
            has_more,
            "sync pull",
        );

        Ok(PullResult {
            entries,
            has_more,
            latest_seq,
        })
    }

    pub fn ack(&self, device_id: &str, acked_seq: i64) -> Result<()> {
        let mut db = self.open_db()?;
        upsert_cursor(&mut db, device_id, acked_seq)?;
        tracing::debug!(device_id, acked_seq, "sync ack");
        Ok(())
    }

    pub fn compact(&self) -> Result<usize> {
        let mut db = self.open_db()?;
        let result = db
            .query("SELECT MIN(acked_seq) FROM sync_cursors;")
            .context("sync compact: query min acked")?;
        let min_acked = result
            .rows
            .first()
            .and_then(|r| r.values.first())
            .and_then(|v| match v {
                Value::Integer(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);

        if min_acked <= 0 {
            return Ok(0);
        }

        let del = db
            .execute_with_params(
                "DELETE FROM sync_entries WHERE id <= ?;",
                vec![Value::Integer(min_acked)],
            )
            .context("sync compact: delete")?;

        let deleted = del.rows_affected as usize;
        if deleted > 0 {
            tracing::debug!(min_acked, deleted, "sync compact");
        }
        Ok(deleted)
    }
}

// DELETE + INSERT instead of INSERT OR REPLACE — rsqlite's or_replace
// only checks integer rowid, not TEXT PRIMARY KEY constraints.
fn upsert_cursor(db: &mut Database, device_id: &str, acked_seq: i64) -> Result<()> {
    db.execute_with_params(
        "DELETE FROM sync_cursors WHERE device_id = ?;",
        vec![Value::Text(device_id.to_string())],
    )
    .context("upsert_cursor: delete")?;
    db.execute_with_params(
        "INSERT INTO sync_cursors (device_id, acked_seq) VALUES (?, ?);",
        vec![
            Value::Text(device_id.to_string()),
            Value::Integer(acked_seq),
        ],
    )
    .context("upsert_cursor: insert")?;
    Ok(())
}

fn extract_i64(values: &[Value], idx: usize) -> i64 {
    match values.get(idx) {
        Some(Value::Integer(n)) => *n,
        _ => 0,
    }
}

fn extract_text(values: &[Value], idx: usize) -> String {
    match values.get(idx) {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    }
}

fn extract_opt_text(values: &[Value], idx: usize) -> Option<String> {
    match values.get(idx) {
        Some(Value::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, SyncStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync.db");
        let store = SyncStore::new(&path).unwrap();
        (dir, store)
    }

    #[test]
    fn push_and_pull_round_trip() {
        let (_dir, store) = temp_store();
        let entries = vec![SyncEntry {
            table: "conversations".into(),
            row_key: r#"{"id":"c1"}"#.into(),
            op: "I".into(),
            ts: 1000,
            device_id: "dev-a".into(),
            snapshot: Some(r#"{"id":"c1","title":"Hello"}"#.into()),
        }];
        let seq = store.push("dev-a", &entries).unwrap();
        assert!(seq > 0);

        let pulled = store.pull("dev-b", 0, 100).unwrap();
        assert_eq!(pulled.entries.len(), 1);
        assert_eq!(pulled.entries[0].table, "conversations");
        assert_eq!(pulled.entries[0].op, "I");
        assert!(!pulled.has_more);
    }

    #[test]
    fn pull_excludes_own_device() {
        let (_dir, store) = temp_store();
        store
            .push(
                "dev-a",
                &[SyncEntry {
                    table: "messages".into(),
                    row_key: r#"{"cid":"c1","mid":"m1"}"#.into(),
                    op: "I".into(),
                    ts: 2000,
                    device_id: "dev-a".into(),
                    snapshot: None,
                }],
            )
            .unwrap();

        let pulled = store.pull("dev-a", 0, 100).unwrap();
        assert_eq!(pulled.entries.len(), 0);
    }

    #[test]
    fn pull_pagination() {
        let (_dir, store) = temp_store();
        for i in 0..5 {
            store
                .push(
                    "dev-a",
                    &[SyncEntry {
                        table: "settings".into(),
                        row_key: format!(r#"{{"key":"k{i}"}}"#),
                        op: "I".into(),
                        ts: 3000 + i,
                        device_id: "dev-a".into(),
                        snapshot: None,
                    }],
                )
                .unwrap();
        }

        let page1 = store.pull("dev-b", 0, 3).unwrap();
        assert_eq!(page1.entries.len(), 3);
        assert!(page1.has_more);

        let page2 = store.pull("dev-b", page1.latest_seq, 3).unwrap();
        assert_eq!(page2.entries.len(), 2);
        assert!(!page2.has_more);
    }

    #[test]
    fn ack_and_compact() {
        let (_dir, store) = temp_store();
        store
            .push(
                "dev-a",
                &[SyncEntry {
                    table: "conversations".into(),
                    row_key: r#"{"id":"c1"}"#.into(),
                    op: "I".into(),
                    ts: 4000,
                    device_id: "dev-a".into(),
                    snapshot: None,
                }],
            )
            .unwrap();

        // dev-b pulls and acks
        let pulled = store.pull("dev-b", 0, 100).unwrap();
        assert_eq!(pulled.entries.len(), 1);
        store.ack("dev-b", pulled.latest_seq).unwrap();

        // Compact should delete the entry
        let deleted = store.compact().unwrap();
        assert_eq!(deleted, 1);

        // Second compact should be a no-op
        let deleted = store.compact().unwrap();
        assert_eq!(deleted, 0);
    }

    #[test]
    fn compact_respects_unacked_devices() {
        let (_dir, store) = temp_store();
        store
            .push(
                "dev-a",
                &[SyncEntry {
                    table: "conversations".into(),
                    row_key: r#"{"id":"c1"}"#.into(),
                    op: "I".into(),
                    ts: 5000,
                    device_id: "dev-a".into(),
                    snapshot: None,
                }],
            )
            .unwrap();

        // dev-b hasn't acked — register it with acked_seq=0
        store.ack("dev-b", 0).unwrap();

        let deleted = store.compact().unwrap();
        assert_eq!(deleted, 0, "should not compact when a device hasn't acked");
    }
}
