use crate::{AnalyticsError, AnalyticsEvent};
use async_trait::async_trait;

/// Pluggable output for analytics events.
///
/// Implementations receive events from the [`AnalyticsCollector`](crate::collector::AnalyticsCollector)'s background
/// drain task. Each call to `record` should be non-blocking from the caller's
/// perspective; the drain task handles backpressure.
#[async_trait]
pub trait AnalyticsSink: Send + Sync + 'static {
    /// Persist or forward a single event.
    async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError>;

    /// Flush any buffered events to durable storage.
    ///
    /// Called by [`AnalyticsCollector::flush`](crate::collector::AnalyticsCollector::flush). Default is a no-op.
    async fn flush(&self) -> Result<(), AnalyticsError> {
        Ok(())
    }
}

/// A type-erased, heap-allocated [`AnalyticsSink`].
pub type BoxedSink = Box<dyn AnalyticsSink>;
