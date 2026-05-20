//! SQLite-backed [`SessionStore`] implementation.
//!
//! Messages are serialised to JSON and stored in a single table keyed by
//! [`SessionId`]. Schema is auto-migrated on first connect.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use tokio::sync::Mutex;

use brainwires_core::Message;

use super::{ListOptions, SessionId, SessionRecord, SessionStore};

/// Disk-backed session store. Access is serialised through a single
/// connection — adequate for single-node agent workloads.
pub struct SqliteSessionStore {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl SqliteSessionStore {
    /// Open (or create) the store at `path`, auto-migrating the schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)
            .with_context(|| format!("opening session store at {}", path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );",
        )?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path,
        })
    }

    /// Path this store writes to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn ts_to_utc(secs: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(secs, 0).single().unwrap_or_else(Utc::now)
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn load(&self, id: &SessionId) -> Result<Option<Vec<Message>>> {
        let conn = self.conn.lock().await;
        let payload: Option<String> = conn
            .query_row(
                "SELECT payload FROM sessions WHERE id = ?1",
                params![id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(match payload {
            Some(s) => Some(serde_json::from_str(&s)?),
            None => None,
        })
    }

    async fn save(&self, id: &SessionId, messages: &[Message]) -> Result<()> {
        let payload = serde_json::to_string(messages)?;
        let now = Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO sessions (id, payload, message_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(id) DO UPDATE SET
                payload = excluded.payload,
                message_count = excluded.message_count,
                updated_at = excluded.updated_at",
            params![id.as_str(), payload, messages.len() as i64, now],
        )?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionRecord>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, message_count, created_at, updated_at FROM sessions ORDER BY updated_at ASC",
        )?;
        let rows = stmt.query_map(params![], |row| {
            Ok(SessionRecord {
                id: SessionId::new(row.get::<_, String>(0)?),
                message_count: row.get::<_, i64>(1)? as usize,
                created_at: ts_to_utc(row.get::<_, i64>(2)?),
                updated_at: ts_to_utc(row.get::<_, i64>(3)?),
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM sessions WHERE id = ?1", params![id.as_str()])?;
        Ok(())
    }

    async fn list_paginated(&self, opts: ListOptions) -> Result<Vec<SessionRecord>> {
        let conn = self.conn.lock().await;
        // SQLite uses i64 for LIMIT/OFFSET. -1 means "no limit" — use it when
        // the caller passed `None`.
        let limit_sql: i64 = opts
            .limit
            .map(|l| l.try_into().unwrap_or(i64::MAX))
            .unwrap_or(-1);
        let offset_sql: i64 = opts.offset.try_into().unwrap_or(i64::MAX);
        let mut stmt = conn.prepare(
            "SELECT id, message_count, created_at, updated_at FROM sessions
             ORDER BY updated_at ASC LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map(params![limit_sql, offset_sql], |row| {
            Ok(SessionRecord {
                id: SessionId::new(row.get::<_, String>(0)?),
                message_count: row.get::<_, i64>(1)? as usize,
                created_at: ts_to_utc(row.get::<_, i64>(2)?),
                updated_at: ts_to_utc(row.get::<_, i64>(3)?),
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store() -> (SqliteSessionStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sessions.db");
        (SqliteSessionStore::open(&path).unwrap(), tmp)
    }

    #[tokio::test]
    async fn roundtrip() {
        let (store, _tmp) = tmp_store();
        let id = SessionId::new("u1");
        store.save(&id, &[Message::user("hi")]).await.unwrap();
        let loaded = store.load(&id).await.unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text(), Some("hi"));
    }

    #[tokio::test]
    async fn survives_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sessions.db");
        {
            let store = SqliteSessionStore::open(&path).unwrap();
            store
                .save(&SessionId::new("persist"), &[Message::user("keep me")])
                .await
                .unwrap();
        }
        let store = SqliteSessionStore::open(&path).unwrap();
        let loaded = store
            .load(&SessionId::new("persist"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[tokio::test]
    async fn list_and_delete() {
        let (store, _tmp) = tmp_store();
        store
            .save(&SessionId::new("a"), &[Message::user("x")])
            .await
            .unwrap();
        store
            .save(&SessionId::new("b"), &[Message::user("y")])
            .await
            .unwrap();
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 2);
        store.delete(&SessionId::new("a")).await.unwrap();
        assert_eq!(store.list().await.unwrap().len(), 1);
    }
}
