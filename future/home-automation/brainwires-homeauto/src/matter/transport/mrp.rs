//! Matter Message Reliability Protocol (MRP) per Matter spec §4.12.
//!
//! MRP provides reliable delivery over UDP by retransmitting messages that have
//! not been acknowledged within a configurable window.  This module tracks the
//! state for a single exchange and provides helper methods for computing retry
//! delays and building standalone ACK payloads.

// ── MRP configuration ─────────────────────────────────────────────────────────

/// Tunable MRP parameters.
///
/// Default values match the Matter 1.3 specification:
/// - Idle retry interval  : 5 000 ms  (sleepy/idle-mode devices)
/// - Active retry interval:   300 ms  (active devices, e.g. mains-powered)
/// - Max retransmit count :     4
/// - ACK timeout          :   200 ms  (delay before sending a standalone ACK)
#[derive(Debug, Clone)]
pub struct MrpConfig {
    /// Retry interval (ms) used when the peer device is in idle mode.
    pub idle_retry_interval_ms: u64,
    /// Retry interval (ms) used when the peer device is in active mode.
    pub active_retry_interval_ms: u64,
    /// Maximum number of retransmit attempts before giving up.
    pub max_retransmit_count: u8,
    /// Maximum delay (ms) before sending a standalone ACK for a received message.
    pub ack_timeout_ms: u64,
}

impl Default for MrpConfig {
    fn default() -> Self {
        Self {
            idle_retry_interval_ms: 5_000,
            active_retry_interval_ms: 300,
            max_retransmit_count: 4,
            ack_timeout_ms: 200,
        }
    }
}

// ── Per-exchange MRP state ────────────────────────────────────────────────────

/// Tracks reliable-delivery state for a single Matter exchange.
///
/// An exchange is identified by its `exchange_id`.  Each unacknowledged message
/// has a `message_counter` that the remote peer must echo back in an ACK.
pub struct MrpExchange {
    /// Exchange identifier (16-bit, scoped to a session).
    pub exchange_id: u16,
    /// Counter of the last message sent on this exchange.
    pub message_counter: u32,
    /// Number of retransmit attempts already made (0 = first attempt).
    pub retransmit_count: u8,
    /// MRP configuration for this exchange.
    pub config: MrpConfig,
}

impl MrpExchange {
    /// Create a new exchange with the given ID and initial message counter.
    pub fn new(exchange_id: u16, message_counter: u32) -> Self {
        Self {
            exchange_id,
            message_counter,
            retransmit_count: 0,
            config: MrpConfig::default(),
        }
    }

    /// Create a new exchange with a custom MRP configuration.
    pub fn with_config(exchange_id: u16, message_counter: u32, config: MrpConfig) -> Self {
        Self {
            exchange_id,
            message_counter,
            retransmit_count: 0,
            config,
        }
    }

    /// Compute the retry delay in milliseconds for the *current* attempt.
    ///
    /// The first retransmit uses the active retry interval; subsequent attempts
    /// double the delay (binary exponential back-off capped at the idle
    /// interval), matching the spirit of the Matter spec §4.12.4.
    pub fn retry_delay_ms(&self) -> u64 {
        let base = self.config.active_retry_interval_ms;
        let delay = base.saturating_mul(1u64 << self.retransmit_count.min(10));
        delay.min(self.config.idle_retry_interval_ms)
    }

    /// Record a retransmit attempt.
    ///
    /// Returns `true` if more retransmits are allowed, or `false` if the
    /// maximum retransmit count has already been reached (giving up).
    pub fn record_retry(&mut self) -> bool {
        if self.retransmit_count >= self.config.max_retransmit_count {
            return false;
        }
        self.retransmit_count += 1;
        true
    }

    /// Build a standalone ACK payload acknowledging the given `ack_counter`.
    ///
    /// Format: a single context-tagged TLV field with tag 1 and a uint32 value:
    ///   `0x25 <tag=1> <counter: 4 LE bytes>`
    ///
    /// Tag byte breakdown: 0x25 = context tag (bits 7:5 = 0b001) | uint32 type
    /// (bits 4:0 = 0b00101).
    pub fn build_ack_payload(ack_counter: u32) -> Vec<u8> {
        // TLV control byte: context tag (0x20) | unsigned int 4-byte (0x05).
        const TLV_CONTEXT_UINT32: u8 = 0x25;
        const TAG_ACK_COUNTER: u8 = 0x01;

        let mut payload = Vec::with_capacity(6);
        payload.push(TLV_CONTEXT_UINT32);
        payload.push(TAG_ACK_COUNTER);
        payload.extend_from_slice(&ack_counter.to_le_bytes());
        payload
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrp_retry_delay_uses_active_interval() {
        let ex = MrpExchange::new(1, 100);
        // First attempt (retransmit_count == 0): delay = active_interval * 2^0 = 300 ms.
        assert_eq!(ex.retry_delay_ms(), 300);
    }

    #[test]
    fn mrp_max_retries_returns_false() {
        let mut ex = MrpExchange::new(2, 200);
        // Exhaust all 4 allowed retries.
        assert!(ex.record_retry()); // 1
        assert!(ex.record_retry()); // 2
        assert!(ex.record_retry()); // 3
        assert!(ex.record_retry()); // 4 — last allowed
        // 5th attempt must return false.
        assert!(!ex.record_retry());
    }

    #[test]
    fn mrp_ack_payload_contains_counter() {
        let counter: u32 = 0xDEAD_BEEF;
        let payload = MrpExchange::build_ack_payload(counter);
        // Length: 1 control + 1 tag + 4 value = 6 bytes.
        assert_eq!(payload.len(), 6);
        // Control byte: 0x25.
        assert_eq!(payload[0], 0x25);
        // Tag: 0x01.
        assert_eq!(payload[1], 0x01);
        // Value: counter in little-endian.
        let decoded = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]);
        assert_eq!(decoded, counter);
    }

    #[test]
    fn mrp_back_off_caps_at_idle_interval() {
        let config = MrpConfig {
            active_retry_interval_ms: 300,
            idle_retry_interval_ms: 5_000,
            max_retransmit_count: 10,
            ..Default::default()
        };

        let mut ex = MrpExchange::with_config(3, 0, config);
        // After several retries the delay must not exceed the idle interval.
        for _ in 0..8 {
            ex.record_retry();
        }
        assert!(ex.retry_delay_ms() <= 5_000);
    }
}
