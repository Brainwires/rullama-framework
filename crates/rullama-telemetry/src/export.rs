//! Audit log export and retention management.
//!
//! Provides JSON and CSV export of analytics events filtered by time range,
//! and a retention policy helper that deletes events older than a configured
//! number of days from a [`MemoryAnalyticsSink`](crate::sinks::memory::MemoryAnalyticsSink).
//!
//! # Feature flag
//! CSV export is always available; the `csv` crate is a lightweight dependency.

use chrono::{DateTime, Utc};

use crate::AnalyticsEvent;
use crate::sinks::memory::MemoryAnalyticsSink;

/// Exports events from a [`MemoryAnalyticsSink`] within a time range.
pub struct AuditExporter<'a> {
    sink: &'a MemoryAnalyticsSink,
}

impl<'a> AuditExporter<'a> {
    /// Create a new exporter backed by the given sink.
    pub fn new(sink: &'a MemoryAnalyticsSink) -> Self {
        Self { sink }
    }

    /// Export matching events as a JSON array string.
    pub fn export_json(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<String, serde_json::Error> {
        let events = self.events_in_range(start, end);
        serde_json::to_string(&events)
    }

    /// Export matching events as CSV.
    ///
    /// Columns: `event_type`, `session_id`, `timestamp`, `payload_json`
    pub fn export_csv(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let events = self.events_in_range(start, end);
        let mut out = String::from("event_type,session_id,timestamp,payload_json\n");

        for event in &events {
            let event_type = event.event_type();
            let session_id = event.session_id().unwrap_or("").to_string();
            let timestamp = event.timestamp().to_rfc3339();
            let payload = serde_json::to_string(event)?;

            // Escape CSV fields containing commas or quotes
            let payload_escaped = csv_escape(&payload);
            let session_escaped = csv_escape(&session_id);

            out.push_str(&format!(
                "{},{},{},{}\n",
                event_type, session_escaped, timestamp, payload_escaped
            ));
        }

        Ok(out)
    }

    /// Delete events older than `retention_days` from the sink.
    ///
    /// Returns the number of events removed.
    pub fn apply_retention_policy(&self, retention_days: u32) -> usize {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
        self.sink.retain(|event| event.timestamp() >= cutoff)
    }

    fn events_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AnalyticsEvent> {
        self.sink
            .drain_matching(|e| e.timestamp() >= start && e.timestamp() <= end)
    }
}

/// Minimal CSV field escaping: wrap in double quotes if the value contains
/// commas, double quotes, or newlines; double any internal double quotes.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_CAPACITY;
    use chrono::Duration;

    fn make_event(offset_secs: i64) -> AnalyticsEvent {
        AnalyticsEvent::Custom {
            session_id: Some("sess-1".to_string()),
            name: "test".to_string(),
            payload: serde_json::json!({"k": "v"}),
            timestamp: Utc::now() + Duration::seconds(offset_secs),
        }
    }

    #[test]
    fn export_json_returns_events_in_range() {
        let sink = MemoryAnalyticsSink::new(DEFAULT_CAPACITY);
        let now = Utc::now();

        // Deposit 3 events: -10s, now, +10s
        sink.deposit(make_event(-10));
        sink.deposit(make_event(0));
        sink.deposit(make_event(10));

        let exporter = AuditExporter::new(&sink);
        let start = now - Duration::seconds(5);
        let end = now + Duration::seconds(5);

        let json = exporter.export_json(start, end).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn export_csv_has_header_and_rows() {
        let sink = MemoryAnalyticsSink::new(DEFAULT_CAPACITY);
        let now = Utc::now();
        sink.deposit(make_event(0));

        let exporter = AuditExporter::new(&sink);
        let csv = exporter
            .export_csv(now - Duration::seconds(1), now + Duration::seconds(1))
            .unwrap();

        assert!(csv.starts_with("event_type,session_id,timestamp,payload_json\n"));
        assert!(csv.contains("custom"));
    }

    #[test]
    fn retention_policy_removes_old_events() {
        let sink = MemoryAnalyticsSink::new(DEFAULT_CAPACITY);

        // Event from 10 days ago
        sink.deposit(AnalyticsEvent::Custom {
            session_id: None,
            name: "old".to_string(),
            payload: serde_json::Value::Null,
            timestamp: Utc::now() - Duration::days(10),
        });
        // Event from today
        sink.deposit(make_event(0));

        let exporter = AuditExporter::new(&sink);
        let removed = exporter.apply_retention_policy(7);
        assert_eq!(removed, 1);
        // The recent event is still present
        assert_eq!(sink.len(), 1);
    }
}
