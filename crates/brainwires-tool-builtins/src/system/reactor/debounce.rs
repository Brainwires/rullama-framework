//! Event debouncing and rate limiting for the file system reactor.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Tracks event times per key and coalesces rapid-fire events.
///
/// Applies both per-key debouncing (suppresses repeated events for the same
/// file/rule combo) and global rate limiting (max events per minute).
pub struct EventDebouncer {
    /// Per-key last event time.
    last_events: HashMap<String, Instant>,
    /// Global debounce window.
    global_debounce: Duration,
    /// Rate limit: max events per minute.
    max_events_per_minute: u32,
    /// Event count in the current minute window.
    event_count: u32,
    /// Start of the current minute window.
    window_start: Instant,
}

impl EventDebouncer {
    /// Create a new debouncer with a global debounce window and rate limit.
    pub fn new(global_debounce_ms: u64, max_events_per_minute: u32) -> Self {
        Self {
            last_events: HashMap::new(),
            global_debounce: Duration::from_millis(global_debounce_ms),
            max_events_per_minute,
            event_count: 0,
            window_start: Instant::now(),
        }
    }

    /// Check if an event should be processed or debounced.
    ///
    /// Returns `true` if the event should be processed, `false` if it should
    /// be suppressed (debounced or rate-limited).
    pub fn should_process(&mut self, key: &str, per_rule_debounce_ms: u64) -> bool {
        let now = Instant::now();

        // Reset minute window if expired
        if now.duration_since(self.window_start) >= Duration::from_secs(60) {
            self.event_count = 0;
            self.window_start = now;
        }

        // Check rate limit
        if self.event_count >= self.max_events_per_minute {
            tracing::warn!(
                "Rate limit reached ({} events/minute), suppressing event for {key}",
                self.max_events_per_minute
            );
            return false;
        }

        // Check per-rule debounce
        let debounce = Duration::from_millis(per_rule_debounce_ms).max(self.global_debounce);
        if let Some(last) = self.last_events.get(key)
            && now.duration_since(*last) < debounce
        {
            return false;
        }

        // Allow the event
        self.last_events.insert(key.to_string(), now);
        self.event_count += 1;
        true
    }

    /// Get the current event count in this minute window.
    pub fn event_count(&self) -> u32 {
        self.event_count
    }

    /// Clean up stale entries older than the given duration.
    pub fn cleanup(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.last_events
            .retain(|_, last| now.duration_since(*last) < max_age);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_event_always_passes() {
        let mut d = EventDebouncer::new(1000, 60);
        assert!(d.should_process("file.txt", 1000));
    }

    #[test]
    fn rapid_events_are_debounced() {
        let mut d = EventDebouncer::new(5000, 60); // 5 second debounce
        assert!(d.should_process("file.txt", 5000));
        // Immediately after, should be debounced
        assert!(!d.should_process("file.txt", 5000));
    }

    #[test]
    fn different_keys_are_independent() {
        let mut d = EventDebouncer::new(5000, 60);
        assert!(d.should_process("a.txt", 5000));
        assert!(d.should_process("b.txt", 5000)); // different key, should pass
        assert!(!d.should_process("a.txt", 5000)); // same key, debounced
    }

    #[test]
    fn rate_limit_enforced() {
        let mut d = EventDebouncer::new(0, 2); // 0ms debounce, max 2/min
        assert!(d.should_process("a", 0));
        assert!(d.should_process("b", 0));
        assert!(!d.should_process("c", 0)); // rate limited
    }

    #[test]
    fn cleanup_removes_stale_entries() {
        let mut d = EventDebouncer::new(0, 60);
        d.should_process("old", 0);
        // Can't easily test time-based cleanup in unit tests without sleeping,
        // but we verify cleanup doesn't panic
        d.cleanup(Duration::from_secs(0));
        assert!(d.last_events.is_empty());
    }
}
