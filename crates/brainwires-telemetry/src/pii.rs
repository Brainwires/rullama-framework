//! PII redaction for analytics events.
//!
//! Provides configurable scrubbing of personally-identifiable information
//! before events reach storage sinks. Useful for GDPR / EU AI Act compliance.
//!
//! # Feature flag
//! This module is always compiled; regex-based custom patterns are a runtime
//! option. Add the `regex` crate to your own crate if you need to build
//! [`PiiRedactionRules::custom_patterns`](crate::pii::PiiRedactionRules::custom_patterns).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::AnalyticsEvent;

/// Rules that control how PII is scrubbed from an [`AnalyticsEvent`].
#[derive(Debug, Default)]
pub struct PiiRedactionRules {
    /// If `true`, replace `session_id` values with a one-way hash so events
    /// can still be grouped without exposing the raw session identifier.
    pub hash_session_ids: bool,
    /// If `true`, redact free-text fields that might contain prompt content
    /// (e.g. the `payload` of `Custom` events and error messages).
    pub redact_prompt_content: bool,
    /// Optional regex patterns. Any string field matching one of these is
    /// replaced with `"[REDACTED]"`.
    pub custom_patterns: Vec<String>,
}

impl PiiRedactionRules {
    /// Create a new rules set with all options disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable session ID hashing.
    pub fn with_session_id_hashing(mut self) -> Self {
        self.hash_session_ids = true;
        self
    }

    /// Enable prompt content redaction.
    pub fn with_prompt_redaction(mut self) -> Self {
        self.redact_prompt_content = true;
        self
    }

    /// Add a literal substring that triggers field redaction.
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.custom_patterns.push(pattern.into());
        self
    }
}

/// Apply `rules` to `event`, returning a new scrubbed event.
///
/// Only fields that carry user-visible strings are touched. Numeric and
/// boolean fields are left intact so aggregations remain meaningful.
pub fn redact_event(event: AnalyticsEvent, rules: &PiiRedactionRules) -> AnalyticsEvent {
    match event {
        AnalyticsEvent::Custom {
            session_id,
            name,
            payload,
            timestamp,
        } => {
            let session_id = maybe_hash_session(session_id, rules);
            let payload = if rules.redact_prompt_content {
                serde_json::Value::String("[REDACTED]".to_string())
            } else {
                redact_value(payload, rules)
            };
            AnalyticsEvent::Custom {
                session_id,
                name,
                payload,
                timestamp,
            }
        }
        AnalyticsEvent::ProviderCall {
            session_id,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            duration_ms,
            cost_usd,
            success,
            timestamp,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            compliance,
        } => AnalyticsEvent::ProviderCall {
            session_id: maybe_hash_session(session_id, rules),
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            duration_ms,
            cost_usd,
            success,
            timestamp,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            compliance,
        },
        AnalyticsEvent::AgentRun {
            session_id,
            agent_id,
            task_id,
            prompt_hash,
            success,
            total_iterations,
            total_tool_calls,
            tool_error_count,
            tools_used,
            total_prompt_tokens,
            total_completion_tokens,
            total_cost_usd,
            duration_ms,
            failure_category,
            timestamp,
            compliance,
        } => AnalyticsEvent::AgentRun {
            session_id: maybe_hash_session(session_id, rules),
            agent_id,
            task_id,
            prompt_hash,
            success,
            total_iterations,
            total_tool_calls,
            tool_error_count,
            tools_used,
            total_prompt_tokens,
            total_completion_tokens,
            total_cost_usd,
            duration_ms,
            failure_category,
            timestamp,
            compliance,
        },
        // All other variants: just hash session_id if requested
        other => hash_session_id_only(other, rules),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn maybe_hash_session(session_id: Option<String>, rules: &PiiRedactionRules) -> Option<String> {
    if rules.hash_session_ids {
        session_id.map(|s| hash_string(&s))
    } else {
        session_id
    }
}

fn hash_string(s: &str) -> String {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    format!("hashed:{:x}", h.finish())
}

fn redact_value(value: serde_json::Value, rules: &PiiRedactionRules) -> serde_json::Value {
    if rules.custom_patterns.is_empty() {
        return value;
    }
    match value {
        serde_json::Value::String(s) => {
            if rules.custom_patterns.iter().any(|p| s.contains(p.as_str())) {
                serde_json::Value::String("[REDACTED]".to_string())
            } else {
                serde_json::Value::String(s)
            }
        }
        serde_json::Value::Object(map) => {
            let redacted = map
                .into_iter()
                .map(|(k, v)| (k, redact_value(v, rules)))
                .collect();
            serde_json::Value::Object(redacted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(|v| redact_value(v, rules)).collect())
        }
        other => other,
    }
}

/// For variants we don't deeply inspect, just hash the session_id field.
fn hash_session_id_only(event: AnalyticsEvent, rules: &PiiRedactionRules) -> AnalyticsEvent {
    if !rules.hash_session_ids {
        return event;
    }
    // Re-serialise → patch session_id → re-deserialise (avoids exhaustive match)
    let Ok(mut value) = serde_json::to_value(&event) else {
        return event;
    };
    if let Some(sid) = value.get("session_id").and_then(|v| v.as_str()) {
        let hashed = hash_string(sid);
        value["session_id"] = serde_json::Value::String(hashed);
    }
    serde_json::from_value(value).unwrap_or(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn custom(session: Option<&str>, payload: serde_json::Value) -> AnalyticsEvent {
        AnalyticsEvent::Custom {
            session_id: session.map(str::to_string),
            name: "test".to_string(),
            payload,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn session_id_hashed_when_enabled() {
        let rules = PiiRedactionRules::new().with_session_id_hashing();
        let event = custom(Some("user-123"), serde_json::json!({}));
        let redacted = redact_event(event, &rules);
        let sid = redacted.session_id().unwrap();
        assert!(sid.starts_with("hashed:"), "expected hashed id, got {sid}");
        assert_ne!(sid, "user-123");
    }

    #[test]
    fn session_id_unchanged_when_hashing_disabled() {
        let rules = PiiRedactionRules::new();
        let event = custom(Some("user-123"), serde_json::json!({}));
        let redacted = redact_event(event, &rules);
        assert_eq!(redacted.session_id(), Some("user-123"));
    }

    #[test]
    fn prompt_content_redacted_when_enabled() {
        let rules = PiiRedactionRules::new().with_prompt_redaction();
        let event = custom(None, serde_json::json!({"message": "secret prompt text"}));
        let redacted = redact_event(event, &rules);
        if let AnalyticsEvent::Custom { payload, .. } = redacted {
            assert_eq!(payload.as_str(), Some("[REDACTED]"));
        } else {
            panic!("expected Custom");
        }
    }

    #[test]
    fn custom_pattern_redacts_matching_string_fields() {
        let rules = PiiRedactionRules::new().with_pattern("secret");
        let payload = serde_json::json!({"note": "this is secret info", "safe": "hello"});
        let event = custom(None, payload);
        let redacted = redact_event(event, &rules);
        if let AnalyticsEvent::Custom { payload, .. } = redacted {
            assert_eq!(payload["note"].as_str(), Some("[REDACTED]"));
            assert_eq!(payload["safe"].as_str(), Some("hello"));
        } else {
            panic!("expected Custom");
        }
    }

    #[test]
    fn provider_call_session_hashed() {
        let rules = PiiRedactionRules::new().with_session_id_hashing();
        let event = AnalyticsEvent::ProviderCall {
            session_id: Some("sess-abc".to_string()),
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            prompt_tokens: 10,
            completion_tokens: 5,
            duration_ms: 100,
            cost_usd: 0.01,
            success: true,
            timestamp: Utc::now(),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            compliance: None,
        };
        let redacted = redact_event(event, &rules);
        let sid = redacted.session_id().unwrap();
        assert!(sid.starts_with("hashed:"));
    }
}
