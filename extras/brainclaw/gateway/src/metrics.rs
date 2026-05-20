//! In-memory metrics collection for the gateway.
//!
//! Provides atomic counters for messages, tool calls, errors, and per-channel
//! activity. No external dependencies — all in-memory.

use dashmap::DashMap;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Collects gateway metrics using atomic counters.
pub struct MetricsCollector {
    /// Total inbound messages processed.
    pub total_messages: AtomicU64,
    /// Total tool calls executed.
    pub total_tool_calls: AtomicU64,
    /// Total errors (failed agent processing, transport errors).
    pub total_errors: AtomicU64,
    /// Total rate-limited messages.
    pub total_rate_limited: AtomicU64,
    /// Total spoofing attempts blocked.
    pub total_spoofing_blocked: AtomicU64,
    /// Per-channel message counts: channel_type -> count.
    pub channel_message_counts: DashMap<String, u64>,
    /// Peak concurrent sessions observed.
    pub peak_sessions: AtomicU64,
    /// Cumulative prompt tokens across all agent completions.
    pub total_prompt_tokens: AtomicU64,
    /// Cumulative completion tokens across all agent completions.
    pub total_completion_tokens: AtomicU64,
    /// Analytics collector for ChannelMessage events.
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

/// Serializable metrics snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub total_messages: u64,
    pub total_tool_calls: u64,
    pub total_errors: u64,
    pub total_rate_limited: u64,
    pub total_spoofing_blocked: u64,
    pub peak_sessions: u64,
    pub channel_message_counts: std::collections::HashMap<String, u64>,
    /// Total prompt tokens consumed across all agent sessions.
    pub total_prompt_tokens: u64,
    /// Total completion tokens generated across all agent sessions.
    pub total_completion_tokens: u64,
    /// Total tokens (prompt + completion).
    pub total_tokens: u64,
}

impl MetricsCollector {
    /// Create a new metrics collector with all counters at zero.
    pub fn new() -> Self {
        Self {
            total_messages: AtomicU64::new(0),
            total_tool_calls: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_rate_limited: AtomicU64::new(0),
            total_spoofing_blocked: AtomicU64::new(0),
            channel_message_counts: DashMap::new(),
            peak_sessions: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Attach an analytics collector to record ChannelMessage events.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Record an inbound message from a channel type.
    pub fn record_message(&self, channel_type: &str) {
        self.total_messages.fetch_add(1, Ordering::Relaxed);
        self.channel_message_counts
            .entry(channel_type.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);

        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            collector.record(AnalyticsEvent::ChannelMessage {
                session_id: None,
                channel_type: channel_type.to_string(),
                direction: "inbound".to_string(),
                message_len: 0,
                timestamp: chrono::Utc::now(),
            });
        }
    }

    /// Record a tool call execution.
    pub fn record_tool_call(&self) {
        self.total_tool_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rate-limited message.
    pub fn record_rate_limited(&self) {
        self.total_rate_limited.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a spoofing attempt blocked.
    pub fn record_spoofing_blocked(&self) {
        self.total_spoofing_blocked.fetch_add(1, Ordering::Relaxed);
    }

    /// Update peak sessions watermark.
    pub fn update_peak_sessions(&self, current: u64) {
        self.peak_sessions.fetch_max(current, Ordering::Relaxed);
    }

    /// Record token usage from a completed agent turn.
    pub fn record_token_usage(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.total_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
    }

    /// Take a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let channel_counts: std::collections::HashMap<String, u64> = self
            .channel_message_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        let prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let completion = self.total_completion_tokens.load(Ordering::Relaxed);

        MetricsSnapshot {
            total_messages: self.total_messages.load(Ordering::Relaxed),
            total_tool_calls: self.total_tool_calls.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_rate_limited: self.total_rate_limited.load(Ordering::Relaxed),
            total_spoofing_blocked: self.total_spoofing_blocked.load(Ordering::Relaxed),
            peak_sessions: self.peak_sessions.load(Ordering::Relaxed),
            channel_message_counts: channel_counts,
            total_prompt_tokens: prompt,
            total_completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_counting() {
        let m = MetricsCollector::new();
        m.record_message("discord");
        m.record_message("discord");
        m.record_message("slack");
        m.record_tool_call();
        m.record_error();
        m.record_rate_limited();
        m.record_spoofing_blocked();
        m.update_peak_sessions(5);

        let snap = m.snapshot();
        assert_eq!(snap.total_messages, 3);
        assert_eq!(snap.total_tool_calls, 1);
        assert_eq!(snap.total_errors, 1);
        assert_eq!(snap.total_rate_limited, 1);
        assert_eq!(snap.total_spoofing_blocked, 1);
        assert_eq!(snap.peak_sessions, 5);
        assert_eq!(snap.channel_message_counts.get("discord"), Some(&2));
        assert_eq!(snap.channel_message_counts.get("slack"), Some(&1));
    }
}
