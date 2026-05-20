//! Audit logging for security-relevant gateway events.
//!
//! Logs structured JSON events for auth failures, rate limiting, spoofing
//! attempts, tool executions, admin API calls, and session lifecycle.

use chrono::Utc;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Maximum number of audit entries kept in memory for the admin API.
const MAX_RING_ENTRIES: usize = 1000;

/// Audit event types.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event_type")]
pub enum AuditEvent {
    /// Failed authentication attempt.
    AuthFailure { source: String, reason: String },
    /// Message dropped due to rate limiting.
    RateLimited { platform: String, user_id: String },
    /// System-message spoofing detected and blocked.
    SpoofingDetected {
        platform: String,
        user_id: String,
        channel_id: String,
    },
    /// Tool executed by an agent on behalf of a user.
    ToolExecution {
        platform: String,
        user_id: String,
        tool_name: String,
    },
    /// Admin API endpoint accessed.
    AdminAccess {
        endpoint: String,
        authenticated: bool,
    },
    /// User session created.
    SessionCreated {
        platform: String,
        user_id: String,
        session_id: String,
    },
    /// User session destroyed.
    SessionDestroyed {
        platform: String,
        user_id: String,
        session_id: String,
    },
    /// Webhook signature verification failed.
    WebhookAuthFailure { reason: String },
}

/// Timestamped audit entry.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// The audit event.
    #[serde(flatten)]
    pub event: AuditEvent,
}

/// Audit logger that emits structured JSON via tracing and keeps a ring buffer
/// for the admin API.
pub struct AuditLogger {
    ring: Mutex<VecDeque<AuditEntry>>,
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new() -> Self {
        Self {
            ring: Mutex::new(VecDeque::with_capacity(MAX_RING_ENTRIES)),
        }
    }

    /// Log an audit event.
    pub fn log(&self, event: AuditEvent) {
        let entry = AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            event: event.clone(),
        };

        // Emit via tracing (target: "audit" for filtering)
        if let Ok(json) = serde_json::to_string(&entry) {
            tracing::info!(target: "audit", "{}", json);
        }

        // Store in ring buffer
        if let Ok(mut ring) = self.ring.lock() {
            if ring.len() >= MAX_RING_ENTRIES {
                ring.pop_front();
            }
            ring.push_back(entry);
        }
    }

    /// Get recent audit entries (most recent first), optionally filtered by event type.
    pub fn recent(&self, limit: usize, event_type: Option<&str>) -> Vec<AuditEntry> {
        let ring = match self.ring.lock() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        ring.iter()
            .rev()
            .filter(|e| {
                event_type.is_none_or(|t| {
                    let json = serde_json::to_value(&e.event).unwrap_or_default();
                    json.get("event_type").and_then(|v| v.as_str()) == Some(t)
                })
            })
            .take(limit)
            .cloned()
            .collect()
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_ring_buffer() {
        let logger = AuditLogger::new();
        logger.log(AuditEvent::AuthFailure {
            source: "admin".into(),
            reason: "bad token".into(),
        });
        logger.log(AuditEvent::RateLimited {
            platform: "discord".into(),
            user_id: "user1".into(),
        });

        let entries = logger.recent(10, None);
        assert_eq!(entries.len(), 2);
        // Most recent first
        assert!(matches!(entries[0].event, AuditEvent::RateLimited { .. }));
    }

    #[test]
    fn test_audit_filter_by_type() {
        let logger = AuditLogger::new();
        logger.log(AuditEvent::AuthFailure {
            source: "admin".into(),
            reason: "bad".into(),
        });
        logger.log(AuditEvent::RateLimited {
            platform: "slack".into(),
            user_id: "u1".into(),
        });

        let filtered = logger.recent(10, Some("RateLimited"));
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_audit_ring_eviction() {
        let logger = AuditLogger::new();
        for i in 0..1005 {
            logger.log(AuditEvent::AuthFailure {
                source: format!("src-{i}"),
                reason: "test".into(),
            });
        }
        let entries = logger.recent(2000, None);
        assert_eq!(entries.len(), 1000); // capped at MAX_RING_ENTRIES
    }
}
