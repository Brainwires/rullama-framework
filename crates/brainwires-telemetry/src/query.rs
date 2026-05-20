use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use rusqlite::{Connection, params};

use crate::schema;

/// Cost breakdown by provider and model for a given date range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostByModelRow {
    /// Date the costs were aggregated for, `YYYY-MM-DD`.
    pub date: String,
    /// Provider name (e.g. `"anthropic"`, `"openai"`).
    pub provider: String,
    /// Model identifier as reported by the provider.
    pub model: String,
    /// Number of provider calls that day for this `(provider, model)`.
    pub call_count: i64,
    /// Sum of prompt tokens across all calls.
    pub total_prompt_tokens: i64,
    /// Sum of completion tokens across all calls.
    pub total_completion_tokens: i64,
    /// Sum of USD charges across all calls.
    pub total_cost_usd: f64,
}

/// Tool call frequency for a given date range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolFrequencyRow {
    /// Date the counts were aggregated for, `YYYY-MM-DD`.
    pub date: String,
    /// Tool identifier as seen by the agent runtime.
    pub tool_name: String,
    /// Total calls that day.
    pub call_count: i64,
    /// Subset of `call_count` that returned an error.
    pub error_count: i64,
}

/// Per-day agent run summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DailySummaryRow {
    /// Date the summary covers, `YYYY-MM-DD`.
    pub date: String,
    /// Total agent runs on this date.
    pub total_runs: i64,
    /// Runs that reported success.
    pub success_count: i64,
    /// Runs that reported failure.
    pub failure_count: i64,
    /// Sum of USD spend across every run.
    pub total_cost_usd: f64,
    /// Sum of prompt + completion tokens across every run.
    pub total_tokens: i64,
    /// Mean `total_iterations` across runs this day.
    pub avg_iterations: f64,
}

/// Read-only query interface backed by the analytics SQLite database.
///
/// Designed for dashboard queries: cost per model, tool frequency, agent success
/// rates. All heavy aggregation is deferred to [`rebuild_summaries`], which
/// replays the raw event log into materialized tables.
///
/// [`rebuild_summaries`]: Self::rebuild_summaries
pub struct AnalyticsQuery {
    conn: Mutex<Connection>,
}

impl AnalyticsQuery {
    /// Open the query interface at the default analytics database path.
    pub fn new_default() -> anyhow::Result<Self> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        let path = home
            .join(".brainwires")
            .join("analytics")
            .join("analytics.db");
        Self::new_with_path(&path)
    }

    /// Open the query interface at a custom path (read-write for rebuilds).
    pub fn new_with_path(db_path: &PathBuf) -> anyhow::Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open analytics DB at {db_path:?}"))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA busy_timeout=5000;",
        )
        .context("Failed to configure SQLite pragmas")?;

        // Ensure schema exists (handles case where query is called before any sink)
        schema::ensure_tables(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Rebuild all materialized summary tables from the raw event log.
    ///
    /// Uses REPLACE INTO / upserts. Safe to call at any time; existing data is
    /// fully replaced with a fresh aggregation of `analytics_events`.
    pub fn rebuild_summaries(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().expect("analytics query lock poisoned");

        // --- cost_by_model from provider_call events ---
        conn.execute_batch(
            r#"
            DELETE FROM cost_by_model;
            INSERT INTO cost_by_model
                (date, provider, model, call_count,
                 total_prompt_tokens, total_completion_tokens, total_cost_usd)
            SELECT
                date(recorded_at / 1000, 'unixepoch') AS date,
                json_extract(payload, '$.provider')   AS provider,
                json_extract(payload, '$.model')      AS model,
                count(*)                              AS call_count,
                sum(COALESCE(json_extract(payload, '$.prompt_tokens'), 0))     AS total_prompt_tokens,
                sum(COALESCE(json_extract(payload, '$.completion_tokens'), 0)) AS total_completion_tokens,
                sum(COALESCE(json_extract(payload, '$.cost_usd'), 0.0))        AS total_cost_usd
            FROM analytics_events
            WHERE event_type = 'provider_call'
            GROUP BY date, provider, model;
            "#,
        )
        .context("Failed to rebuild cost_by_model")?;

        // --- tool_usage from tool_call events ---
        conn.execute_batch(
            r#"
            DELETE FROM tool_usage;
            INSERT INTO tool_usage (date, tool_name, call_count, error_count)
            SELECT
                date(recorded_at / 1000, 'unixepoch')       AS date,
                json_extract(payload, '$.tool_name')         AS tool_name,
                count(*)                                     AS call_count,
                sum(CASE WHEN json_extract(payload, '$.is_error') = 1 THEN 1 ELSE 0 END) AS error_count
            FROM analytics_events
            WHERE event_type = 'tool_call'
            GROUP BY date, tool_name;
            "#,
        )
        .context("Failed to rebuild tool_usage")?;

        // --- agent_run_summaries from agent_run events ---
        conn.execute_batch(
            r#"
            DELETE FROM agent_run_summaries;
            INSERT INTO agent_run_summaries
                (date, total_runs, success_count, failure_count,
                 total_cost_usd, total_tokens, avg_iterations)
            SELECT
                date(recorded_at / 1000, 'unixepoch') AS date,
                count(*)                               AS total_runs,
                sum(CASE WHEN json_extract(payload, '$.success') = 1 THEN 1 ELSE 0 END) AS success_count,
                sum(CASE WHEN json_extract(payload, '$.success') = 0 THEN 1 ELSE 0 END) AS failure_count,
                sum(COALESCE(json_extract(payload, '$.total_cost_usd'), 0.0))             AS total_cost_usd,
                sum(COALESCE(json_extract(payload, '$.total_prompt_tokens'), 0) +
                    COALESCE(json_extract(payload, '$.total_completion_tokens'), 0))      AS total_tokens,
                avg(COALESCE(json_extract(payload, '$.total_iterations'), 0))             AS avg_iterations
            FROM analytics_events
            WHERE event_type = 'agent_run'
            GROUP BY date;
            "#,
        )
        .context("Failed to rebuild agent_run_summaries")?;

        // --- session_summaries ---
        conn.execute_batch(
            r#"
            DELETE FROM session_summaries;
            INSERT INTO session_summaries
                (session_id, first_event_at, last_event_at,
                 provider_calls, agent_runs, tool_calls,
                 total_cost_usd, total_tokens)
            SELECT
                session_id,
                min(recorded_at)  AS first_event_at,
                max(recorded_at)  AS last_event_at,
                sum(CASE WHEN event_type = 'provider_call'  THEN 1 ELSE 0 END) AS provider_calls,
                sum(CASE WHEN event_type = 'agent_run'      THEN 1 ELSE 0 END) AS agent_runs,
                sum(CASE WHEN event_type = 'tool_call'      THEN 1 ELSE 0 END) AS tool_calls,
                sum(CASE
                    WHEN event_type = 'provider_call'
                    THEN COALESCE(json_extract(payload, '$.cost_usd'), 0.0)
                    WHEN event_type = 'agent_run'
                    THEN COALESCE(json_extract(payload, '$.total_cost_usd'), 0.0)
                    ELSE 0.0
                END) AS total_cost_usd,
                sum(CASE
                    WHEN event_type = 'provider_call'
                    THEN COALESCE(json_extract(payload, '$.prompt_tokens'), 0) +
                         COALESCE(json_extract(payload, '$.completion_tokens'), 0)
                    ELSE 0
                END) AS total_tokens
            FROM analytics_events
            WHERE session_id IS NOT NULL
            GROUP BY session_id;
            "#,
        )
        .context("Failed to rebuild session_summaries")?;

        Ok(())
    }

    // --- Public query methods ---

    /// Cost breakdown grouped by provider and model, optionally filtered by date range.
    ///
    /// `from` / `to` are inclusive date strings in `YYYY-MM-DD` format.
    pub fn cost_by_model(
        &self,
        from: Option<&str>,
        to: Option<&str>,
    ) -> anyhow::Result<Vec<CostByModelRow>> {
        let conn = self.conn.lock().expect("analytics query lock poisoned");
        let where_ = build_date_filter("date", from, to);
        let sql = format!(
            "SELECT date, provider, model, call_count,
                    total_prompt_tokens, total_completion_tokens, total_cost_usd
             FROM cost_by_model
             {where_}
             ORDER BY total_cost_usd DESC"
        );
        let mut stmt = conn
            .prepare(&sql)
            .context("Failed to prepare cost_by_model query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(CostByModelRow {
                    date: row.get(0)?,
                    provider: row.get(1)?,
                    model: row.get(2)?,
                    call_count: row.get(3)?,
                    total_prompt_tokens: row.get(4)?,
                    total_completion_tokens: row.get(5)?,
                    total_cost_usd: row.get(6)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Tool call frequency, optionally filtered by date range.
    pub fn tool_frequency(
        &self,
        from: Option<&str>,
        to: Option<&str>,
    ) -> anyhow::Result<Vec<ToolFrequencyRow>> {
        let conn = self.conn.lock().expect("analytics query lock poisoned");
        let where_ = build_date_filter("date", from, to);
        let sql = format!(
            "SELECT date, tool_name, call_count, error_count
             FROM tool_usage
             {where_}
             ORDER BY call_count DESC"
        );
        let mut stmt = conn
            .prepare(&sql)
            .context("Failed to prepare tool_frequency query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ToolFrequencyRow {
                    date: row.get(0)?,
                    tool_name: row.get(1)?,
                    call_count: row.get(2)?,
                    error_count: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Per-day agent run summaries (includes success rate, cost, token totals).
    pub fn daily_summaries(
        &self,
        from: Option<&str>,
        to: Option<&str>,
    ) -> anyhow::Result<Vec<DailySummaryRow>> {
        let conn = self.conn.lock().expect("analytics query lock poisoned");
        let where_ = build_date_filter("date", from, to);
        let sql = format!(
            "SELECT date, total_runs, success_count, failure_count,
                    total_cost_usd, total_tokens, avg_iterations
             FROM agent_run_summaries
             {where_}
             ORDER BY date DESC"
        );
        let mut stmt = conn
            .prepare(&sql)
            .context("Failed to prepare daily_summaries query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(DailySummaryRow {
                    date: row.get(0)?,
                    total_runs: row.get(1)?,
                    success_count: row.get(2)?,
                    failure_count: row.get(3)?,
                    total_cost_usd: row.get(4)?,
                    total_tokens: row.get(5)?,
                    avg_iterations: row.get(6)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Most recent `limit` raw events, optionally filtered by event type.
    pub fn recent_events(
        &self,
        limit: usize,
        event_type: Option<&str>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().expect("analytics query lock poisoned");
        let (sql, filter_val) = match event_type {
            Some(et) => (
                format!(
                    "SELECT payload FROM analytics_events \
                     WHERE event_type = ?1 ORDER BY recorded_at DESC LIMIT {limit}"
                ),
                Some(et.to_string()),
            ),
            None => (
                format!(
                    "SELECT payload FROM analytics_events \
                     ORDER BY recorded_at DESC LIMIT {limit}"
                ),
                None,
            ),
        };

        let mut stmt = conn
            .prepare(&sql)
            .context("Failed to prepare recent_events query")?;
        let rows: Vec<serde_json::Value> = if let Some(val) = filter_val {
            stmt.query_map(params![val], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .filter_map(|s| serde_json::from_str(&s).ok())
                .collect()
        } else {
            stmt.query_map([], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .filter_map(|s| serde_json::from_str(&s).ok())
                .collect()
        };
        Ok(rows)
    }

    /// Total cost in USD over the given period.
    pub fn total_cost_usd(&self, from: Option<&str>, to: Option<&str>) -> anyhow::Result<f64> {
        let rows = self.cost_by_model(from, to)?;
        Ok(rows.iter().map(|r| r.total_cost_usd).sum())
    }
}

/// Build a WHERE clause for date-range filtering on a TEXT date column.
fn build_date_filter(col: &str, from: Option<&str>, to: Option<&str>) -> String {
    match (from, to) {
        (Some(f), Some(t)) => format!("WHERE {col} >= '{f}' AND {col} <= '{t}'"),
        (Some(f), None) => format!("WHERE {col} >= '{f}'"),
        (None, Some(t)) => format!("WHERE {col} <= '{t}'"),
        (None, None) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinks::sqlite::SqliteAnalyticsSink;
    use crate::{AnalyticsEvent, AnalyticsSink};
    use chrono::Utc;
    use tempfile::TempDir;

    async fn setup() -> (AnalyticsQuery, TempDir) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("analytics.db");
        let q = AnalyticsQuery::new_with_path(&path).unwrap();
        (q, tmp)
    }

    async fn insert_event(tmp: &TempDir, event: AnalyticsEvent) {
        let path = tmp.path().join("analytics.db");
        let sink = SqliteAnalyticsSink::new_with_path(&path).unwrap();
        sink.record(event).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rebuild_and_cost_by_model() {
        let (q, tmp) = setup().await;

        insert_event(
            &tmp,
            AnalyticsEvent::ProviderCall {
                session_id: None,
                provider: "anthropic".into(),
                model: "claude-opus-4-6".into(),
                prompt_tokens: 1000,
                completion_tokens: 500,
                duration_ms: 300,
                cost_usd: 0.05,
                success: true,
                timestamp: Utc::now(),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                compliance: None,
            },
        )
        .await;

        q.rebuild_summaries().unwrap();

        let costs = q.cost_by_model(None, None).unwrap();
        assert_eq!(costs.len(), 1);
        assert_eq!(costs[0].provider, "anthropic");
        assert_eq!(costs[0].call_count, 1);
        assert!((costs[0].total_cost_usd - 0.05).abs() < 1e-9);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rebuild_agent_run_summaries() {
        let (q, tmp) = setup().await;

        insert_event(
            &tmp,
            AnalyticsEvent::AgentRun {
                session_id: None,
                agent_id: "a1".into(),
                task_id: "t1".into(),
                prompt_hash: "abc".into(),
                success: true,
                total_iterations: 5,
                total_tool_calls: 3,
                tool_error_count: 0,
                tools_used: vec!["bash".into()],
                total_prompt_tokens: 800,
                total_completion_tokens: 200,
                total_cost_usd: 0.02,
                duration_ms: 1200,
                failure_category: None,
                timestamp: Utc::now(),
                compliance: None,
            },
        )
        .await;

        q.rebuild_summaries().unwrap();

        let summaries = q.daily_summaries(None, None).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].total_runs, 1);
        assert_eq!(summaries[0].success_count, 1);
    }
}
