use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use brainwires_telemetry::UsageEvent;
use chrono::{DateTime, Utc};

use crate::BillingImplError;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Pluggable durable store for [`UsageEvent`]s.
#[async_trait]
pub trait BillingLedger: Send + Sync + 'static {
    async fn record(&self, event: UsageEvent) -> Result<(), BillingImplError>;
    async fn total_cost(&self, agent_id: &str) -> Result<f64, BillingImplError>;
    async fn events_for(
        &self,
        agent_id: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>, BillingImplError>;
}

// ── InMemoryLedger ─────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct InMemoryLedger {
    events: Arc<Mutex<Vec<UsageEvent>>>,
}

impl InMemoryLedger {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl BillingLedger for InMemoryLedger {
    async fn record(&self, event: UsageEvent) -> Result<(), BillingImplError> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }

    async fn total_cost(&self, agent_id: &str) -> Result<f64, BillingImplError> {
        let total = self
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.agent_id() == agent_id)
            .map(|e| e.cost_usd())
            .sum();
        Ok(total)
    }

    async fn events_for(
        &self,
        agent_id: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>, BillingImplError> {
        let events = self
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.agent_id() == agent_id && since.is_none_or(|t| e.timestamp() >= t))
            .cloned()
            .collect();
        Ok(events)
    }
}

// ── SqliteLedger ──────────────────────────────────────────────────────────────

#[cfg(feature = "sqlite")]
pub struct SqliteLedger {
    conn: Mutex<rusqlite::Connection>,
}

#[cfg(feature = "sqlite")]
impl SqliteLedger {
    pub fn new_default() -> anyhow::Result<Self> {
        let db_path = Self::default_db_path()?;
        Self::new_with_path(&db_path)
    }

    pub fn new_with_path(db_path: &std::path::PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create billing directory at {parent:?}"))?;
        }

        let conn = rusqlite::Connection::open(db_path)
            .with_context(|| format!("Failed to open billing DB at {db_path:?}"))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA busy_timeout=5000;
             PRAGMA synchronous=NORMAL;",
        )
        .context("Failed to configure SQLite pragmas")?;

        crate::schema::ensure_tables(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    #[cfg(feature = "native")]
    fn default_db_path() -> anyhow::Result<std::path::PathBuf> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join(".brainwires").join("billing").join("billing.db"))
    }
}

#[cfg(feature = "sqlite")]
#[async_trait]
impl BillingLedger for SqliteLedger {
    async fn record(&self, event: UsageEvent) -> Result<(), BillingImplError> {
        let agent_id = event.agent_id().to_string();
        let kind = event.kind().to_string();
        let cost_usd = event.cost_usd();
        let recorded_at = event.timestamp().timestamp_millis();
        let payload = serde_json::to_string(&event)?;

        tokio::task::block_in_place(|| {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO billing_events (agent_id, kind, cost_usd, payload, recorded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![agent_id, kind, cost_usd, payload, recorded_at],
            )
            .context("Failed to insert billing event")
            .map_err(BillingImplError::Ledger)?;
            Ok(())
        })
    }

    async fn total_cost(&self, agent_id: &str) -> Result<f64, BillingImplError> {
        let agent_id = agent_id.to_string();
        tokio::task::block_in_place(|| {
            let conn = self.conn.lock().unwrap();
            let total: f64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM billing_events WHERE agent_id = ?1",
                    rusqlite::params![agent_id],
                    |row| row.get(0),
                )
                .context("Failed to query total cost")
                .map_err(BillingImplError::Ledger)?;
            Ok(total)
        })
    }

    async fn events_for(
        &self,
        agent_id: &str,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<UsageEvent>, BillingImplError> {
        let agent_id = agent_id.to_string();
        let since_ms = since.map(|t| t.timestamp_millis()).unwrap_or(0);

        tokio::task::block_in_place(|| {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT payload FROM billing_events
                      WHERE agent_id = ?1 AND recorded_at >= ?2
                      ORDER BY recorded_at ASC",
                )
                .context("Failed to prepare events query")
                .map_err(BillingImplError::Ledger)?;

            let events = stmt
                .query_map(rusqlite::params![agent_id, since_ms], |row| {
                    row.get::<_, String>(0)
                })
                .context("Failed to execute events query")
                .map_err(BillingImplError::Ledger)?
                .filter_map(|r| r.ok())
                .filter_map(|payload| serde_json::from_str::<UsageEvent>(&payload).ok())
                .collect();

            Ok(events)
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_telemetry::UsageEvent;

    #[tokio::test]
    async fn in_memory_record_and_total() {
        let ledger = InMemoryLedger::new();
        ledger
            .record(UsageEvent::tokens("a1", "model", 100, 0.001))
            .await
            .unwrap();
        ledger
            .record(UsageEvent::tokens("a1", "model", 200, 0.002))
            .await
            .unwrap();
        ledger
            .record(UsageEvent::tool_call("a2", "bash"))
            .await
            .unwrap();

        let total = ledger.total_cost("a1").await.unwrap();
        assert!((total - 0.003).abs() < 1e-9);
        assert_eq!(ledger.total_cost("a2").await.unwrap(), 0.0);
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test(flavor = "multi_thread")]
    async fn sqlite_record_and_total() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("billing.db");
        let ledger = SqliteLedger::new_with_path(&db_path).unwrap();

        ledger
            .record(UsageEvent::tokens("x", "gpt-4o", 500, 0.005))
            .await
            .unwrap();
        ledger
            .record(UsageEvent::tool_call("x", "bash"))
            .await
            .unwrap();

        let total = ledger.total_cost("x").await.unwrap();
        assert!((total - 0.005).abs() < 1e-9);
    }
}
