//! Per-user rate limiting middleware.
//!
//! Provides sliding-window rate limiting for both messages and tool calls,
//! keyed by `(platform, user_id)` to prevent abuse from any single user.

use dashmap::DashMap;
use std::time::Instant;

/// Per-user, per-platform rate limiter for messages and tool calls.
///
/// Uses a simple fixed-window approach: each `(platform, user_id)` pair
/// gets a counter that resets every 60 seconds. When a counter exceeds
/// the configured maximum, subsequent requests are rejected until the
/// window expires.
pub struct RateLimiter {
    message_limits: DashMap<(String, String), RateWindow>,
    tool_limits: DashMap<(String, String), RateWindow>,
    /// Maximum messages per minute per user.
    pub max_messages_per_minute: u32,
    /// Maximum tool calls per minute per user.
    pub max_tool_calls_per_minute: u32,
}

/// A fixed-window counter.
struct RateWindow {
    count: u32,
    window_start: Instant,
}

impl RateLimiter {
    /// Create a new `RateLimiter` with the given per-minute limits.
    pub fn new(max_messages_per_minute: u32, max_tool_calls_per_minute: u32) -> Self {
        Self {
            message_limits: DashMap::new(),
            tool_limits: DashMap::new(),
            max_messages_per_minute,
            max_tool_calls_per_minute,
        }
    }

    /// Check whether the user is under the message rate limit (does **not** increment).
    pub fn check_message_rate(&self, platform: &str, user_id: &str) -> bool {
        self.check_rate(
            &self.message_limits,
            platform,
            user_id,
            self.max_messages_per_minute,
        )
    }

    /// Check whether the user is under the tool-call rate limit (does **not** increment).
    pub fn check_tool_rate(&self, platform: &str, user_id: &str) -> bool {
        self.check_rate(
            &self.tool_limits,
            platform,
            user_id,
            self.max_tool_calls_per_minute,
        )
    }

    /// Record that a message was sent by this user.
    pub fn record_message(&self, platform: &str, user_id: &str) {
        self.record(&self.message_limits, platform, user_id);
    }

    /// Record that a tool call was made by this user.
    pub fn record_tool_call(&self, platform: &str, user_id: &str) {
        self.record(&self.tool_limits, platform, user_id);
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn check_rate(
        &self,
        map: &DashMap<(String, String), RateWindow>,
        platform: &str,
        user_id: &str,
        max: u32,
    ) -> bool {
        let key = (platform.to_string(), user_id.to_string());
        match map.get(&key) {
            Some(entry) => {
                if entry.window_start.elapsed().as_secs() >= 60 {
                    // Window expired — would be reset on next record
                    true
                } else {
                    entry.count < max
                }
            }
            None => true,
        }
    }

    fn record(&self, map: &DashMap<(String, String), RateWindow>, platform: &str, user_id: &str) {
        let key = (platform.to_string(), user_id.to_string());
        let mut entry = map.entry(key).or_insert_with(|| RateWindow {
            count: 0,
            window_start: Instant::now(),
        });

        // Reset window if expired
        if entry.window_start.elapsed().as_secs() >= 60 {
            entry.count = 0;
            entry.window_start = Instant::now();
        }

        entry.count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_limit_passes() {
        let limiter = RateLimiter::new(5, 10);
        assert!(limiter.check_message_rate("discord", "user1"));
        limiter.record_message("discord", "user1");
        assert!(limiter.check_message_rate("discord", "user1"));
    }

    #[test]
    fn over_message_limit_blocks() {
        let limiter = RateLimiter::new(3, 10);
        for _ in 0..3 {
            assert!(limiter.check_message_rate("discord", "user1"));
            limiter.record_message("discord", "user1");
        }
        // 4th should be blocked
        assert!(!limiter.check_message_rate("discord", "user1"));
    }

    #[test]
    fn over_tool_limit_blocks() {
        let limiter = RateLimiter::new(10, 2);
        for _ in 0..2 {
            assert!(limiter.check_tool_rate("slack", "user2"));
            limiter.record_tool_call("slack", "user2");
        }
        assert!(!limiter.check_tool_rate("slack", "user2"));
    }

    #[test]
    fn different_users_independent() {
        let limiter = RateLimiter::new(2, 10);
        limiter.record_message("discord", "user1");
        limiter.record_message("discord", "user1");
        assert!(!limiter.check_message_rate("discord", "user1"));
        assert!(limiter.check_message_rate("discord", "user2"));
    }

    #[test]
    fn different_platforms_independent() {
        let limiter = RateLimiter::new(2, 10);
        limiter.record_message("discord", "user1");
        limiter.record_message("discord", "user1");
        assert!(!limiter.check_message_rate("discord", "user1"));
        assert!(limiter.check_message_rate("slack", "user1"));
    }

    #[test]
    fn new_user_always_passes() {
        let limiter = RateLimiter::new(1, 1);
        assert!(limiter.check_message_rate("telegram", "new_user"));
        assert!(limiter.check_tool_rate("telegram", "new_user"));
    }
}
