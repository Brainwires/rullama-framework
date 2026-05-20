//! Anomaly detection over observed audit-style events.
//!
//! [`AnomalyDetector`] tracks statistical baselines for tool-call frequency,
//! policy-violation rate, trust-level changes, and out-of-scope path access.
//! When observed values exceed configurable thresholds an [`AnomalyEvent`]
//! is queued; callers drain the queue via [`AnomalyDetector::drain_anomalies`].
//!
//! This module is consumer-agnostic: it operates on any type that implements
//! [`ObservedEvent`]. The canonical implementation lives in
//! `brainwires-permission::audit::AuditEvent`, but other crates can plug in
//! their own event types without dragging in the permissions crate.
//!
//! # Example (with a synthetic event)
//!
//! ```rust,ignore
//! use brainwires_telemetry::anomaly::{
//!     AnomalyConfig, AnomalyDetector, EventCategory, ObservedEvent,
//! };
//!
//! struct MyEvent { ts: i64, agent: String, cat: EventCategory }
//! impl ObservedEvent for MyEvent {
//!     fn timestamp_secs(&self) -> i64 { self.ts }
//!     fn agent_id(&self) -> Option<&str> { Some(&self.agent) }
//!     fn category(&self) -> EventCategory { self.cat }
//!     fn target(&self) -> Option<&str> { None }
//! }
//!
//! let detector = AnomalyDetector::new(AnomalyConfig::default());
//! detector.observe(&MyEvent { ts: 0, agent: "a1".into(), cat: EventCategory::PolicyViolation });
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// ── Public traits & types ─────────────────────────────────────────────────────

/// Category of event the anomaly detector branches on.
///
/// Concrete event-type enums (e.g., `AuditEventType` in `brainwires-permission`)
/// project onto this small set when implementing [`ObservedEvent::category`].
/// The `Other` variant captures anything the detector should ignore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventCategory {
    /// A policy violation — feeds the violation-window counter.
    PolicyViolation,
    /// A tool execution — feeds the tool-call rate counter and path-scope check.
    ToolExecution,
    /// A trust-level change — feeds the trust-change counter.
    TrustChange,
    /// Anything the detector should ignore.
    Other,
}

/// Minimum surface required by [`AnomalyDetector::observe`].
///
/// Implementations exist out-of-crate (e.g., for `AuditEvent` in
/// `brainwires-permission`); telemetry itself does not depend on permissions.
pub trait ObservedEvent {
    /// Unix timestamp in seconds when the event occurred.
    fn timestamp_secs(&self) -> i64;
    /// Agent that produced the event, if known.
    fn agent_id(&self) -> Option<&str>;
    /// Category of the event (drives which detector branch runs).
    fn category(&self) -> EventCategory;
    /// Target path / URL / resource of the event, if applicable.
    /// Used by the path-scope check on `ToolExecution` events.
    fn target(&self) -> Option<&str>;
}

// ── AnomalyKind / AnomalyEvent ────────────────────────────────────────────────

/// The kind of anomaly that was detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AnomalyKind {
    /// The same agent triggered policy violations more than `threshold` times
    /// within the sliding window.
    RepeatedPolicyViolation {
        /// Number of violations observed.
        count: u32,
        /// Sliding window duration in seconds.
        window_secs: u64,
    },
    /// An agent made tool calls at a rate exceeding `threshold` calls per window.
    HighFrequencyToolCalls {
        /// Number of tool calls observed.
        count: u32,
        /// Sliding window duration in seconds.
        window_secs: u64,
    },
    /// An agent accessed a path that lies outside all expected path prefixes.
    UnusualFileScopeRequest {
        /// The unexpected path accessed.
        path: String,
    },
    /// An agent's trust level changed more than `threshold` times within the
    /// sliding window.
    RapidTrustChange {
        /// Number of trust level changes.
        changes: u32,
        /// Sliding window duration in seconds.
        window_secs: u64,
    },
}

/// A single anomaly event produced by the detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// Unique identifier for this anomaly occurrence.
    pub id: String,
    /// When the anomaly was detected.
    pub detected_at: DateTime<Utc>,
    /// Agent involved (if known).
    pub agent_id: Option<String>,
    /// Structured kind with supporting metrics.
    pub kind: AnomalyKind,
    /// Human-readable description suitable for logging or alerting.
    pub description: String,
}

impl AnomalyEvent {
    fn new(agent_id: Option<String>, kind: AnomalyKind, description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            detected_at: Utc::now(),
            agent_id,
            kind,
            description: description.into(),
        }
    }
}

// ── AnomalyConfig ─────────────────────────────────────────────────────────────

/// Configuration for the anomaly detector.
///
/// All thresholds use a sliding-window model: if the count within the last
/// `*_window_secs` seconds exceeds `*_threshold`, an anomaly is emitted.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Sliding window duration for policy-violation counting (seconds).
    pub violation_window_secs: u64,
    /// Number of violations within the window that triggers an anomaly.
    pub violation_threshold: u32,
    /// Sliding window duration for tool-call rate counting (seconds).
    pub tool_call_window_secs: u64,
    /// Number of tool calls within the window that triggers an anomaly.
    pub tool_call_threshold: u32,
    /// Sliding window duration for trust-change counting (seconds).
    pub trust_change_window_secs: u64,
    /// Number of trust changes within the window that triggers an anomaly.
    pub trust_change_threshold: u32,
    /// Optional set of "expected" path prefixes (e.g. `/workspace/`).
    ///
    /// When non-empty, any `ToolExecution` event whose `target` does not
    /// start with one of these prefixes is flagged as
    /// [`AnomalyKind::UnusualFileScopeRequest`].
    pub expected_path_prefixes: Vec<String>,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            violation_window_secs: 60,
            violation_threshold: 3,
            tool_call_window_secs: 10,
            tool_call_threshold: 20,
            trust_change_window_secs: 60,
            trust_change_threshold: 3,
            expected_path_prefixes: Vec::new(),
        }
    }
}

// ── Sliding-window counter ────────────────────────────────────────────────────

#[derive(Debug)]
struct WindowCounter {
    timestamps: VecDeque<i64>,
    window_secs: u64,
}

impl WindowCounter {
    fn new(window_secs: u64) -> Self {
        Self {
            timestamps: VecDeque::new(),
            window_secs,
        }
    }

    /// Record `now_secs` and evict stale entries.  Returns the in-window count.
    fn record_and_count(&mut self, now_secs: i64) -> u32 {
        self.timestamps.push_back(now_secs);
        let cutoff = now_secs - self.window_secs as i64;
        while self.timestamps.front().is_some_and(|&t| t <= cutoff) {
            self.timestamps.pop_front();
        }
        self.timestamps.len() as u32
    }
}

// ── AnomalyDetector ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct AnomalyDetectorInner {
    config: AnomalyConfig,
    violation_windows: HashMap<String, WindowCounter>,
    tool_call_windows: HashMap<String, WindowCounter>,
    trust_change_windows: HashMap<String, WindowCounter>,
    pending: Vec<AnomalyEvent>,
}

/// Stateful, thread-safe anomaly detector.
///
/// Wrap in `Arc` if sharing across threads; the inner state is `Mutex`-protected
/// so a plain `AnomalyDetector` can be stored as-is in an audit logger.
#[derive(Clone, Debug)]
pub struct AnomalyDetector {
    inner: Arc<Mutex<AnomalyDetectorInner>>,
}

impl AnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AnomalyDetectorInner {
                config,
                violation_windows: HashMap::new(),
                tool_call_windows: HashMap::new(),
                trust_change_windows: HashMap::new(),
                pending: Vec::new(),
            })),
        }
    }

    /// Observe an event and emit anomaly events if thresholds are breached.
    ///
    /// Designed to be called from inside the audit-logging hot path before
    /// the event is moved into a buffer.
    pub fn observe(&self, event: &dyn ObservedEvent) {
        let mut inner = self
            .inner
            .lock()
            .expect("anomaly detector state lock poisoned");
        let now_secs = event.timestamp_secs();
        let agent_key = event
            .agent_id()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Snapshot config values before the mutable borrows on `inner.*_windows`.
        let violation_threshold = inner.config.violation_threshold;
        let violation_window_secs = inner.config.violation_window_secs;
        let tool_call_threshold = inner.config.tool_call_threshold;
        let tool_call_window_secs = inner.config.tool_call_window_secs;
        let trust_change_threshold = inner.config.trust_change_threshold;
        let trust_change_window_secs = inner.config.trust_change_window_secs;
        let expected_prefixes = inner.config.expected_path_prefixes.clone();

        let agent_id_owned = event.agent_id().map(|s| s.to_string());

        match event.category() {
            EventCategory::PolicyViolation => {
                let window = inner
                    .violation_windows
                    .entry(agent_key.clone())
                    .or_insert_with(|| WindowCounter::new(violation_window_secs));
                let count = window.record_and_count(now_secs);
                if count >= violation_threshold {
                    inner.pending.push(AnomalyEvent::new(
                        agent_id_owned.clone(),
                        AnomalyKind::RepeatedPolicyViolation {
                            count,
                            window_secs: violation_window_secs,
                        },
                        format!(
                            "Agent '{}' triggered {} policy violations in {}s",
                            agent_key, count, violation_window_secs
                        ),
                    ));
                }
            }

            EventCategory::ToolExecution => {
                // Rate check
                let window = inner
                    .tool_call_windows
                    .entry(agent_key.clone())
                    .or_insert_with(|| WindowCounter::new(tool_call_window_secs));
                let count = window.record_and_count(now_secs);
                if count >= tool_call_threshold {
                    inner.pending.push(AnomalyEvent::new(
                        agent_id_owned.clone(),
                        AnomalyKind::HighFrequencyToolCalls {
                            count,
                            window_secs: tool_call_window_secs,
                        },
                        format!(
                            "Agent '{}' made {} tool calls in {}s",
                            agent_key, count, tool_call_window_secs
                        ),
                    ));
                }

                // Path-scope check
                if !expected_prefixes.is_empty()
                    && let Some(target) = event.target()
                {
                    let is_expected = expected_prefixes
                        .iter()
                        .any(|prefix| target.starts_with(prefix.as_str()));
                    if !is_expected {
                        inner.pending.push(AnomalyEvent::new(
                            agent_id_owned.clone(),
                            AnomalyKind::UnusualFileScopeRequest {
                                path: target.to_string(),
                            },
                            format!(
                                "Agent '{}' requested path '{}' outside expected scope",
                                agent_key, target
                            ),
                        ));
                    }
                }
            }

            EventCategory::TrustChange => {
                let window = inner
                    .trust_change_windows
                    .entry(agent_key.clone())
                    .or_insert_with(|| WindowCounter::new(trust_change_window_secs));
                let count = window.record_and_count(now_secs);
                if count >= trust_change_threshold {
                    inner.pending.push(AnomalyEvent::new(
                        agent_id_owned.clone(),
                        AnomalyKind::RapidTrustChange {
                            changes: count,
                            window_secs: trust_change_window_secs,
                        },
                        format!(
                            "Agent '{}' had {} trust changes in {}s",
                            agent_key, count, trust_change_window_secs
                        ),
                    ));
                }
            }

            EventCategory::Other => {}
        }
    }

    /// Drain all pending anomaly events (clears the internal queue).
    pub fn drain_anomalies(&self) -> Vec<AnomalyEvent> {
        let mut inner = self
            .inner
            .lock()
            .expect("anomaly detector state lock poisoned");
        std::mem::take(&mut inner.pending)
    }

    /// Return the number of pending anomaly events without draining.
    pub fn pending_count(&self) -> usize {
        self.inner
            .lock()
            .expect("anomaly detector state lock poisoned")
            .pending
            .len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `ObservedEvent` impl for unit tests — no audit-event coupling.
    struct TestEvent {
        ts: i64,
        agent: Option<String>,
        cat: EventCategory,
        target: Option<String>,
    }

    impl ObservedEvent for TestEvent {
        fn timestamp_secs(&self) -> i64 {
            self.ts
        }
        fn agent_id(&self) -> Option<&str> {
            self.agent.as_deref()
        }
        fn category(&self) -> EventCategory {
            self.cat
        }
        fn target(&self) -> Option<&str> {
            self.target.as_deref()
        }
    }

    fn ev(cat: EventCategory, agent: &str) -> TestEvent {
        TestEvent {
            ts: 0,
            agent: Some(agent.to_string()),
            cat,
            target: None,
        }
    }

    #[test]
    fn test_no_anomaly_below_threshold() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            violation_threshold: 3,
            ..Default::default()
        });
        let e = ev(EventCategory::PolicyViolation, "agent-1");
        detector.observe(&e);
        detector.observe(&e);
        assert_eq!(detector.pending_count(), 0);
    }

    #[test]
    fn test_repeated_violations_trigger_anomaly() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            violation_threshold: 3,
            violation_window_secs: 60,
            ..Default::default()
        });
        let e = ev(EventCategory::PolicyViolation, "agent-1");
        detector.observe(&e);
        detector.observe(&e);
        detector.observe(&e);
        assert_eq!(detector.pending_count(), 1);
        let anomalies = detector.drain_anomalies();
        assert!(matches!(
            anomalies[0].kind,
            AnomalyKind::RepeatedPolicyViolation { count: 3, .. }
        ));
    }

    #[test]
    fn test_high_frequency_tool_calls() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            tool_call_threshold: 5,
            tool_call_window_secs: 60,
            ..Default::default()
        });
        let e = ev(EventCategory::ToolExecution, "agent-2");
        for _ in 0..5 {
            detector.observe(&e);
        }
        assert_eq!(detector.pending_count(), 1);
        let anomalies = detector.drain_anomalies();
        assert!(matches!(
            anomalies[0].kind,
            AnomalyKind::HighFrequencyToolCalls { count: 5, .. }
        ));
    }

    #[test]
    fn test_unusual_file_scope_request() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            expected_path_prefixes: vec!["/workspace/".to_string()],
            tool_call_threshold: 1_000,
            ..Default::default()
        });
        let e = TestEvent {
            ts: 0,
            agent: Some("agent-3".to_string()),
            cat: EventCategory::ToolExecution,
            target: Some("/etc/secrets".to_string()),
        };
        detector.observe(&e);
        let anomalies = detector.drain_anomalies();
        assert!(anomalies.iter().any(|a| matches!(
            &a.kind,
            AnomalyKind::UnusualFileScopeRequest { path } if path == "/etc/secrets"
        )));
    }

    #[test]
    fn test_within_scope_path_no_scope_anomaly() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            expected_path_prefixes: vec!["/workspace/".to_string()],
            tool_call_threshold: 1_000,
            ..Default::default()
        });
        let e = TestEvent {
            ts: 0,
            agent: Some("agent-3".to_string()),
            cat: EventCategory::ToolExecution,
            target: Some("/workspace/src/main.rs".to_string()),
        };
        detector.observe(&e);
        let anomalies = detector.drain_anomalies();
        assert!(
            !anomalies
                .iter()
                .any(|a| matches!(a.kind, AnomalyKind::UnusualFileScopeRequest { .. }))
        );
    }

    #[test]
    fn test_rapid_trust_change() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            trust_change_threshold: 3,
            trust_change_window_secs: 60,
            ..Default::default()
        });
        let e = ev(EventCategory::TrustChange, "agent-4");
        for _ in 0..3 {
            detector.observe(&e);
        }
        let anomalies = detector.drain_anomalies();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a.kind, AnomalyKind::RapidTrustChange { changes: 3, .. }))
        );
    }

    #[test]
    fn test_drain_clears_pending() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            violation_threshold: 1,
            ..Default::default()
        });
        let e = ev(EventCategory::PolicyViolation, "agent-5");
        detector.observe(&e);
        assert_eq!(detector.pending_count(), 1);
        detector.drain_anomalies();
        assert_eq!(detector.pending_count(), 0);
    }

    #[test]
    fn test_different_agents_tracked_separately() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            violation_threshold: 3,
            ..Default::default()
        });
        let e1 = ev(EventCategory::PolicyViolation, "agent-A");
        let e2 = ev(EventCategory::PolicyViolation, "agent-B");
        detector.observe(&e1);
        detector.observe(&e1);
        detector.observe(&e2);
        detector.observe(&e2);
        assert_eq!(detector.pending_count(), 0);
    }

    #[test]
    fn test_anomaly_event_has_agent_id() {
        let detector = AnomalyDetector::new(AnomalyConfig {
            violation_threshold: 1,
            ..Default::default()
        });
        let e = ev(EventCategory::PolicyViolation, "my-agent");
        detector.observe(&e);
        let anomalies = detector.drain_anomalies();
        assert_eq!(anomalies[0].agent_id.as_deref(), Some("my-agent"));
    }
}
