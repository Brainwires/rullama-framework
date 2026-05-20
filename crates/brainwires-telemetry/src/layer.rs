use std::collections::HashMap;
use std::time::Instant;

use tracing::{Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

use crate::{AnalyticsCollector, AnalyticsEvent};

/// `tracing_subscriber::Layer` that auto-captures known spans and emits analytics events.
///
/// Currently intercepts:
///
/// | Span name       | Fields captured        | Event emitted                        |
/// |-----------------|------------------------|--------------------------------------|
/// | `provider.chat` | `provider`, `model`    | [`AnalyticsEvent::ProviderCall`] (partial — tokens/cost come from explicit emission in brainwires-provider) |
///
/// All other spans pass through unmodified. This layer never alters span data.
pub struct AnalyticsLayer {
    collector: AnalyticsCollector,
}

impl AnalyticsLayer {
    /// Wrap a pre-built `AnalyticsCollector`. Install via
    /// `tracing_subscriber::registry().with(AnalyticsLayer::new(collector))`.
    pub fn new(collector: AnalyticsCollector) -> Self {
        Self { collector }
    }
}

// --- Span extension types stored per-span ---

/// Stores field key-value pairs captured at span creation.
struct FieldMap(HashMap<String, String>);

/// Stores the instant a span was entered (for duration computation).
struct EntryTime(Instant);

/// A `tracing::field::Visit` implementation that collects string representations
/// of span fields into a HashMap.
struct FieldCollector<'a>(&'a mut HashMap<String, String>);

impl Visit for FieldCollector<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S> Layer<S> for AnalyticsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };
        let mut map = HashMap::new();
        attrs.record(&mut FieldCollector(&mut map));
        span.extensions_mut().insert(FieldMap(map));
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };
        span.extensions_mut().replace(EntryTime(Instant::now()));
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let span = match ctx.span(&id) {
            Some(s) => s,
            None => return,
        };
        let meta = span.metadata();

        if meta.name() == "provider.chat" {
            let exts = span.extensions();
            let fields = exts.get::<FieldMap>().map(|m| &m.0);
            let provider = fields
                .and_then(|m| m.get("provider"))
                .cloned()
                .unwrap_or_default();
            let model = fields
                .and_then(|m| m.get("model"))
                .cloned()
                .unwrap_or_default();
            let duration_ms = exts
                .get::<EntryTime>()
                .map(|t| t.0.elapsed().as_millis() as u64)
                .unwrap_or(0);

            // Note: prompt_tokens, completion_tokens, and cost_usd are 0 here.
            // Complete ProviderCall events with token/cost data are emitted
            // explicitly by brainwires-provider after each chat() call (Phase 2).
            // This layer captures provider + model + duration from the span.
            self.collector.record(AnalyticsEvent::ProviderCall {
                session_id: None,
                provider,
                model,
                prompt_tokens: 0,
                completion_tokens: 0,
                duration_ms,
                cost_usd: 0.0,
                success: true,
                timestamp: chrono::Utc::now(),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                compliance: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinks::memory::MemoryAnalyticsSink;
    use std::sync::Arc;
    use tracing_subscriber::prelude::*;

    struct SharedMemSink(Arc<MemoryAnalyticsSink>);
    #[async_trait::async_trait]
    impl crate::AnalyticsSink for SharedMemSink {
        async fn record(&self, event: AnalyticsEvent) -> Result<(), crate::AnalyticsError> {
            self.0.record(event).await
        }
    }

    #[tokio::test]
    async fn test_intercepts_provider_chat_span() {
        let mem = Arc::new(MemoryAnalyticsSink::new(10));
        let mem2 = Arc::clone(&mem);
        let collector = AnalyticsCollector::new(vec![Box::new(SharedMemSink(mem2))]);
        let layer = AnalyticsLayer::new(collector.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        {
            let span = tracing::info_span!(
                "provider.chat",
                provider = "anthropic",
                model = "claude-opus-4-6"
            );
            let _enter = span.enter();
            // simulate some work
        }

        // Allow the drain task to process
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;

        let events = mem.drain();
        assert_eq!(events.len(), 1);
        match &events[0] {
            AnalyticsEvent::ProviderCall {
                provider, model, ..
            } => {
                assert_eq!(provider, "anthropic");
                assert_eq!(model, "claude-opus-4-6");
            }
            other => panic!("expected ProviderCall, got {other:?}"),
        }
    }
}
