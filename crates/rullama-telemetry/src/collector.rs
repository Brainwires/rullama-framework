use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, watch};

use crate::{AnalyticsError, AnalyticsEvent, BoxedSink};

/// Internal channel capacity for the event queue.
const CHANNEL_CAPACITY: usize = 4096;

/// Central analytics collector.
///
/// Clone cheaply — all clones share the same background drain task and sinks.
///
/// # Usage
///
/// ```rust,ignore
/// let sink  = SqliteAnalyticsSink::new_default()?;
/// let collector = AnalyticsCollector::new(vec![Box::new(sink)]);
///
/// // Share across tasks/threads
/// let c2 = collector.clone();
/// tokio::spawn(async move { c2.record(event); });
///
/// // Graceful shutdown
/// collector.shutdown().await;
/// ```
pub struct AnalyticsCollector {
    inner: Arc<CollectorInner>,
}

impl Clone for AnalyticsCollector {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for AnalyticsCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyticsCollector").finish_non_exhaustive()
    }
}

struct CollectorInner {
    tx: mpsc::Sender<AnalyticsEvent>,
    /// Send a oneshot reply-channel to request a flush; drain loop responds after
    /// all pending events are processed and sinks are flushed.
    flush_tx: mpsc::Sender<oneshot::Sender<()>>,
    shutdown_tx: watch::Sender<bool>,
}

impl AnalyticsCollector {
    /// Create a new collector and start the background drain task.
    ///
    /// Must be called after the tokio runtime is running.
    pub fn new(sinks: Vec<BoxedSink>) -> Self {
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let (flush_tx, flush_rx) = mpsc::channel::<oneshot::Sender<()>>(8);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(drain_loop(rx, flush_rx, sinks, shutdown_rx));

        Self {
            inner: Arc::new(CollectorInner {
                tx,
                flush_tx,
                shutdown_tx,
            }),
        }
    }

    /// Emit an event. Returns immediately; delivery is async.
    ///
    /// Uses `try_send` — silently drops if the channel is full (fail-open).
    /// Analytics must never block or panic framework code.
    pub fn record(&self, event: AnalyticsEvent) {
        let _ = self.inner.tx.try_send(event);
    }

    /// Wait for all queued events to be delivered to sinks, then flush every sink.
    ///
    /// Uses a sentinel pattern: sends a oneshot reply-channel into the drain task's
    /// flush queue. The drain task drains all pending events first, calls
    /// `sink.flush()` on each sink, then replies. This guarantees durability
    /// (e.g. WAL checkpoint) before returning.
    pub async fn flush(&self) -> Result<(), AnalyticsError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        // If the flush channel is full or closed, fail-open.
        if self.inner.flush_tx.send(reply_tx).await.is_ok() {
            let _ = reply_rx.await;
        }
        Ok(())
    }

    /// Signal the background drain task to stop after emptying the queue.
    ///
    /// Drains pending events, flushes all sinks, then terminates the task.
    pub async fn shutdown(&self) {
        let _ = self.inner.shutdown_tx.send(true);
        // Give the drain task a moment to finish draining and flushing.
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

async fn drain_loop(
    mut rx: mpsc::Receiver<AnalyticsEvent>,
    mut flush_rx: mpsc::Receiver<oneshot::Sender<()>>,
    sinks: Vec<BoxedSink>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            biased;

            Some(event) = rx.recv() => {
                for sink in &sinks {
                    if let Err(e) = sink.record(event.clone()).await {
                        tracing::warn!(
                            error = %e,
                            event_type = event.event_type(),
                            "Analytics sink failed to record event"
                        );
                    }
                }
            }

            Some(reply_tx) = flush_rx.recv() => {
                // Drain all currently queued events before flushing sinks.
                while let Ok(event) = rx.try_recv() {
                    for sink in &sinks {
                        let _ = sink.record(event.clone()).await;
                    }
                }
                // Flush each sink to durable storage (e.g. WAL checkpoint).
                for sink in &sinks {
                    let _ = sink.flush().await;
                }
                // Signal caller that flush is complete.
                let _ = reply_tx.send(());
            }

            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    // Drain remaining events before stopping.
                    while let Ok(event) = rx.try_recv() {
                        for sink in &sinks {
                            let _ = sink.record(event.clone()).await;
                        }
                    }
                    // Flush all sinks.
                    for sink in &sinks {
                        let _ = sink.flush().await;
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinks::memory::MemoryAnalyticsSink;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_event() -> AnalyticsEvent {
        AnalyticsEvent::Custom {
            session_id: None,
            name: "test".into(),
            payload: serde_json::Value::Null,
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_fanout_to_sink() {
        let mem = Arc::new(MemoryAnalyticsSink::new(100));
        let mem2 = Arc::clone(&mem);

        // Wrap in a newtype so we can share the Arc with the sink trait
        struct SharedMemSink(Arc<MemoryAnalyticsSink>);
        #[async_trait::async_trait]
        impl crate::AnalyticsSink for SharedMemSink {
            async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
                self.0.record(event).await
            }
        }

        let collector = AnalyticsCollector::new(vec![Box::new(SharedMemSink(Arc::clone(&mem2)))]);

        for _ in 0..10 {
            collector.record(make_event());
        }

        // flush() now guarantees all events are delivered before returning.
        collector.flush().await.unwrap();

        assert_eq!(mem.len(), 10);
    }

    #[tokio::test]
    async fn test_record_does_not_block_when_full() {
        let collector = AnalyticsCollector::new(vec![]);
        // Fill past capacity — should not block or panic
        for _ in 0..(CHANNEL_CAPACITY + 100) {
            collector.record(make_event());
        }
    }

    #[tokio::test]
    async fn test_flush_ensures_delivery() {
        let mem = Arc::new(MemoryAnalyticsSink::new(1000));
        let mem2 = Arc::clone(&mem);

        struct SharedMemSink(Arc<MemoryAnalyticsSink>);
        #[async_trait::async_trait]
        impl crate::AnalyticsSink for SharedMemSink {
            async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
                self.0.record(event).await
            }
        }

        let collector = AnalyticsCollector::new(vec![Box::new(SharedMemSink(Arc::clone(&mem2)))]);
        for _ in 0..50 {
            collector.record(make_event());
        }
        collector.flush().await.unwrap();
        // After flush() returns, all 50 events must be in the sink.
        assert_eq!(mem.len(), 50);
    }
}
