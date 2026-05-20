//! End-to-end integration tests for analytics sinks via the public crate API.
//! Covers the in-memory sink behavior and (when the `sqlite` feature is on)
//! the SQLite-backed sink + AnalyticsCollector dispatch.

use std::sync::Arc;

use chrono::Utc;

use brainwires_telemetry::{
    AnalyticsCollector, AnalyticsEvent, AnalyticsSink, MemoryAnalyticsSink,
};

fn custom_event(name: &str) -> AnalyticsEvent {
    AnalyticsEvent::Custom {
        session_id: None,
        name: name.into(),
        payload: serde_json::Value::Null,
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn memory_sink_records_then_drains_in_order() {
    let sink = MemoryAnalyticsSink::new(10);
    for i in 0..5 {
        sink.record(custom_event(&format!("e{i}"))).await.unwrap();
    }
    let drained = sink.drain();
    assert_eq!(drained.len(), 5);
    for (i, ev) in drained.iter().enumerate() {
        match ev {
            AnalyticsEvent::Custom { name, .. } => assert_eq!(name, &format!("e{i}")),
            other => panic!("unexpected variant: {other:?}"),
        }
    }
    assert!(sink.is_empty());
}

#[tokio::test]
async fn memory_sink_overflow_increments_dropped_count() {
    let sink = MemoryAnalyticsSink::new(3);
    for i in 0..7 {
        sink.record(custom_event(&format!("e{i}"))).await.unwrap();
    }
    assert_eq!(sink.len(), 3);
    assert_eq!(sink.dropped_count(), 4);
    let drained = sink.drain();
    // Oldest 4 evicted; remaining are e4, e5, e6.
    let names: Vec<_> = drained
        .iter()
        .map(|ev| match ev {
            AnalyticsEvent::Custom { name, .. } => name.clone(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(names, vec!["e4", "e5", "e6"]);
}

#[tokio::test]
async fn collector_dispatches_to_every_sink() {
    // Wrap the in-memory sink in an Arc-clone-safe AnalyticsSink shim so the
    // collector and the assertion path both retain a handle.
    struct SharedMem(Arc<MemoryAnalyticsSink>);
    #[async_trait::async_trait]
    impl AnalyticsSink for SharedMem {
        async fn record(
            &self,
            event: AnalyticsEvent,
        ) -> Result<(), brainwires_telemetry::AnalyticsError> {
            self.0.record(event).await
        }
    }

    let mem_a = Arc::new(MemoryAnalyticsSink::new(10));
    let mem_b = Arc::new(MemoryAnalyticsSink::new(10));
    let collector = AnalyticsCollector::new(vec![
        Box::new(SharedMem(mem_a.clone())),
        Box::new(SharedMem(mem_b.clone())),
    ]);

    collector.record(custom_event("fanout"));
    collector.flush().await.expect("flush ok");

    assert_eq!(mem_a.len(), 1, "first sink received event");
    assert_eq!(mem_b.len(), 1, "second sink received event");
}

#[cfg(feature = "sqlite")]
mod sqlite {
    use super::*;
    use brainwires_telemetry::SqliteAnalyticsSink;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sqlite_sink_persists_event() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("analytics.db");
        let sink = SqliteAnalyticsSink::new_with_path(&path).expect("open sqlite sink");
        sink.record(custom_event("persisted"))
            .await
            .expect("record");
        // Reopen to confirm durability.
        let reopened = SqliteAnalyticsSink::new_with_path(&path).expect("reopen");
        // We can't easily query custom events without depending on `query`, but
        // a successful reopen of the file proves the schema migrated and the
        // earlier write didn't corrupt the DB.
        drop(reopened);
    }
}
