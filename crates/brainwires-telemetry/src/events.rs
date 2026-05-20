use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// Compliance metadata attached to auditable events (EU AI Act, HIPAA, SOC2).
///
/// All fields are optional so existing serialized events deserialise correctly
/// when this field is absent (`#[serde(default)]`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceMetadata {
    /// ISO 3166-1 alpha-2 region where data was processed (e.g. `"EU"`, `"US"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_region: Option<String>,
    /// Whether the event payload may contain PII.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pii_present: Option<bool>,
    /// Number of days this record must be retained before it can be deleted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_days: Option<u32>,
    /// Applicable regulation identifier (e.g. `"GDPR"`, `"HIPAA"`, `"EU_AI_ACT"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regulation: Option<String>,
    /// Whether this event requires inclusion in a compliance audit trail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_required: Option<bool>,
}

/// A typed analytics event emitted anywhere in the framework.
///
/// All variants are self-contained (no imports from other brainwires crates)
/// and fully serializable. The `session_id` field, when present, groups related
/// events across multiple emitting components.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum AnalyticsEvent {
    /// A provider `chat()` call completed (success or failure).
    ProviderCall {
        /// Session that owned the call, if any.
        session_id: Option<String>,
        /// Provider name (e.g. `"anthropic"`, `"openai"`).
        provider: String,
        /// Model identifier as reported by the provider.
        model: String,
        /// Input tokens billed.
        prompt_tokens: u32,
        /// Output tokens billed.
        completion_tokens: u32,
        /// Wall-clock call duration in milliseconds.
        duration_ms: u64,
        /// Pre-computed USD charge for this call.
        cost_usd: f64,
        /// Whether the call returned a usable response.
        success: bool,
        /// When the call completed.
        timestamp: DateTime<Utc>,
        /// Tokens charged to populate the provider's prompt cache this call.
        /// Zero when the provider doesn't support caching or the cache wasn't
        /// used. Anthropic only, today.
        #[serde(default, skip_serializing_if = "is_zero_u32")]
        cache_creation_input_tokens: u32,
        /// Tokens served from the provider's prompt cache this call — the
        /// primary cost-savings signal.
        #[serde(default, skip_serializing_if = "is_zero_u32")]
        cache_read_input_tokens: u32,
        /// Optional compliance metadata for audit / data-residency tracking.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compliance: Option<ComplianceMetadata>,
    },

    /// A TaskAgent run completed.
    AgentRun {
        /// Session that owned the run, if any.
        session_id: Option<String>,
        /// Agent identifier as seen by the runtime.
        agent_id: String,
        /// Task identifier that initiated this run.
        task_id: String,
        /// SHA-256 hex of the initial prompt (for de-duplication / replay).
        prompt_hash: String,
        /// Whether the run completed without a terminal error.
        success: bool,
        /// Number of reason/act/observe loops the agent performed.
        total_iterations: u32,
        /// Count of tool invocations across all iterations.
        total_tool_calls: u32,
        /// Subset of `total_tool_calls` that returned an error.
        tool_error_count: u32,
        /// Distinct tool names touched (deduped, unordered).
        tools_used: Vec<String>,
        /// Aggregate prompt tokens across every provider call in the run.
        total_prompt_tokens: u32,
        /// Aggregate completion tokens across every provider call in the run.
        total_completion_tokens: u32,
        /// Sum of USD spend across every provider and tool cost.
        total_cost_usd: f64,
        /// Wall-clock run duration in milliseconds.
        duration_ms: u64,
        /// Coarse failure bucket when `success == false` (`"timeout"`, `"tool"`, `"provider"`, …).
        failure_category: Option<String>,
        /// When the run ended.
        timestamp: DateTime<Utc>,
        /// Optional compliance metadata for audit / data-residency tracking.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compliance: Option<ComplianceMetadata>,
    },

    /// A single tool call within an agent run.
    ToolCall {
        /// Session that owned the call, if any.
        session_id: Option<String>,
        /// Agent that invoked the tool, if identifiable.
        agent_id: Option<String>,
        /// Tool identifier as seen by the runtime.
        tool_name: String,
        /// Unique id assigned to this invocation (matches the provider's tool_use id).
        tool_use_id: String,
        /// Whether the tool returned an error.
        is_error: bool,
        /// Wall-clock duration; `None` when not measured.
        duration_ms: Option<u64>,
        /// When the tool call completed.
        timestamp: DateTime<Utc>,
    },

    /// An MCP server request was handled.
    McpRequest {
        /// Session that owned the request, if any.
        session_id: Option<String>,
        /// MCP server the request was routed to.
        server_name: String,
        /// Tool the MCP server exposed and that was invoked.
        tool_name: String,
        /// Whether the server returned a successful response.
        success: bool,
        /// Wall-clock request duration in milliseconds.
        duration_ms: u64,
        /// When the request completed.
        timestamp: DateTime<Utc>,
    },

    /// A channel message was sent or received (Discord, Telegram, Slack, etc.).
    ChannelMessage {
        /// Session that owned the channel interaction, if any.
        session_id: Option<String>,
        /// Channel platform identifier (e.g. `"discord"`, `"slack"`).
        channel_type: String,
        /// `"inbound"` or `"outbound"`.
        direction: String,
        /// Payload length in bytes (after PII scrubbing).
        message_len: usize,
        /// When the message traversed the boundary.
        timestamp: DateTime<Utc>,
    },

    /// A storage operation completed.
    StorageOp {
        /// Session that owned the operation, if any.
        session_id: Option<String>,
        /// Backing store identifier (e.g. `"sqlite"`, `"lance"`, `"qdrant"`).
        store_type: String,
        /// Operation name (`"read"`, `"write"`, `"vector_search"`, …).
        operation: String,
        /// Whether the operation completed without error.
        success: bool,
        /// Wall-clock operation duration in milliseconds.
        duration_ms: u64,
        /// When the operation completed.
        timestamp: DateTime<Utc>,
    },

    /// A network message was sent or received over the agent network.
    NetworkMessage {
        /// Session that owned the message, if any.
        session_id: Option<String>,
        /// Transport identifier (`"tcp"`, `"ipc"`, `"a2a"`, …).
        protocol: String,
        /// `"inbound"` or `"outbound"`.
        direction: String,
        /// Serialised payload size in bytes (over the wire).
        bytes: u64,
        /// Whether the peer acknowledged the message.
        success: bool,
        /// When the message crossed the boundary.
        timestamp: DateTime<Utc>,
    },

    /// A dream consolidation cycle completed.
    DreamCycle {
        /// Session that owned the cycle, if any.
        session_id: Option<String>,
        /// Sessions visited during consolidation.
        sessions_processed: usize,
        /// Messages collapsed into summaries.
        messages_summarized: usize,
        /// Facts extracted into the cold tier.
        facts_extracted: usize,
        /// Total tokens held by the hot tier prior to the pass.
        tokens_before: usize,
        /// Total tokens held by the hot tier after the pass.
        tokens_after: usize,
        /// Wall-clock cycle duration in milliseconds.
        duration_ms: u64,
        /// When the cycle ended.
        timestamp: DateTime<Utc>,
    },

    /// An autonomy session completed.
    AutonomySession {
        /// Session that owned the autonomy run, if any.
        session_id: Option<String>,
        /// Tasks the session tried to execute.
        tasks_attempted: u32,
        /// Subset that reported success.
        tasks_succeeded: u32,
        /// Subset that reported failure (sum with `tasks_succeeded` ≤ `tasks_attempted`).
        tasks_failed: u32,
        /// Cumulative USD spend for the session.
        total_cost_usd: f64,
        /// Wall-clock session duration in milliseconds.
        duration_ms: u64,
        /// When the session ended.
        timestamp: DateTime<Utc>,
    },

    /// Escape hatch for user-defined events.
    Custom {
        /// Session that owned the event, if any.
        session_id: Option<String>,
        /// Caller-defined event name (surfaced to queries).
        name: String,
        /// Arbitrary JSON payload — consumed by downstream sinks as-is.
        payload: serde_json::Value,
        /// When the event occurred.
        timestamp: DateTime<Utc>,
    },
}

impl AnalyticsEvent {
    /// Returns the event's timestamp regardless of variant.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::ProviderCall { timestamp, .. } => *timestamp,
            Self::AgentRun { timestamp, .. } => *timestamp,
            Self::ToolCall { timestamp, .. } => *timestamp,
            Self::McpRequest { timestamp, .. } => *timestamp,
            Self::ChannelMessage { timestamp, .. } => *timestamp,
            Self::StorageOp { timestamp, .. } => *timestamp,
            Self::NetworkMessage { timestamp, .. } => *timestamp,
            Self::DreamCycle { timestamp, .. } => *timestamp,
            Self::AutonomySession { timestamp, .. } => *timestamp,
            Self::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// Returns the session_id if present.
    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::ProviderCall { session_id, .. } => session_id.as_deref(),
            Self::AgentRun { session_id, .. } => session_id.as_deref(),
            Self::ToolCall { session_id, .. } => session_id.as_deref(),
            Self::McpRequest { session_id, .. } => session_id.as_deref(),
            Self::ChannelMessage { session_id, .. } => session_id.as_deref(),
            Self::StorageOp { session_id, .. } => session_id.as_deref(),
            Self::NetworkMessage { session_id, .. } => session_id.as_deref(),
            Self::DreamCycle { session_id, .. } => session_id.as_deref(),
            Self::AutonomySession { session_id, .. } => session_id.as_deref(),
            Self::Custom { session_id, .. } => session_id.as_deref(),
        }
    }

    /// Returns the serde discriminant tag for this event (matches the SQLite `event_type` column).
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ProviderCall { .. } => "provider_call",
            Self::AgentRun { .. } => "agent_run",
            Self::ToolCall { .. } => "tool_call",
            Self::McpRequest { .. } => "mcp_request",
            Self::ChannelMessage { .. } => "channel_message",
            Self::StorageOp { .. } => "storage_op",
            Self::NetworkMessage { .. } => "network_message",
            Self::DreamCycle { .. } => "dream_cycle",
            Self::AutonomySession { .. } => "autonomy_session",
            Self::Custom { .. } => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    fn custom_event(session: Option<&str>, name: &str) -> AnalyticsEvent {
        AnalyticsEvent::Custom {
            session_id: session.map(str::to_string),
            name: name.to_string(),
            payload: serde_json::json!({"k": "v"}),
            timestamp: now(),
        }
    }

    fn provider_call_event() -> AnalyticsEvent {
        AnalyticsEvent::ProviderCall {
            session_id: Some("sess-1".to_string()),
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            prompt_tokens: 100,
            completion_tokens: 200,
            duration_ms: 500,
            cost_usd: 0.01,
            success: true,
            timestamp: now(),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            compliance: None,
        }
    }

    // --- event_type() ---

    #[test]
    fn event_type_matches_serde_tag() {
        let ts = now();
        let cases: Vec<(&str, AnalyticsEvent)> = vec![
            ("provider_call", provider_call_event()),
            (
                "agent_run",
                AnalyticsEvent::AgentRun {
                    session_id: None,
                    agent_id: "a".into(),
                    task_id: "t".into(),
                    prompt_hash: "h".into(),
                    success: true,
                    total_iterations: 1,
                    total_tool_calls: 0,
                    tool_error_count: 0,
                    tools_used: vec![],
                    total_prompt_tokens: 0,
                    total_completion_tokens: 0,
                    total_cost_usd: 0.0,
                    duration_ms: 0,
                    failure_category: None,
                    timestamp: ts,
                    compliance: None,
                },
            ),
            (
                "tool_call",
                AnalyticsEvent::ToolCall {
                    session_id: None,
                    agent_id: None,
                    tool_name: "bash".into(),
                    tool_use_id: "u1".into(),
                    is_error: false,
                    duration_ms: None,
                    timestamp: ts,
                },
            ),
            (
                "mcp_request",
                AnalyticsEvent::McpRequest {
                    session_id: None,
                    server_name: "s".into(),
                    tool_name: "t".into(),
                    success: true,
                    duration_ms: 10,
                    timestamp: ts,
                },
            ),
            (
                "channel_message",
                AnalyticsEvent::ChannelMessage {
                    session_id: None,
                    channel_type: "discord".into(),
                    direction: "inbound".into(),
                    message_len: 42,
                    timestamp: ts,
                },
            ),
            (
                "storage_op",
                AnalyticsEvent::StorageOp {
                    session_id: None,
                    store_type: "sqlite".into(),
                    operation: "read".into(),
                    success: true,
                    duration_ms: 1,
                    timestamp: ts,
                },
            ),
            (
                "network_message",
                AnalyticsEvent::NetworkMessage {
                    session_id: None,
                    protocol: "tcp".into(),
                    direction: "out".into(),
                    bytes: 128,
                    success: true,
                    timestamp: ts,
                },
            ),
            (
                "dream_cycle",
                AnalyticsEvent::DreamCycle {
                    session_id: None,
                    sessions_processed: 5,
                    messages_summarized: 20,
                    facts_extracted: 10,
                    tokens_before: 1000,
                    tokens_after: 200,
                    duration_ms: 300,
                    timestamp: ts,
                },
            ),
            (
                "autonomy_session",
                AnalyticsEvent::AutonomySession {
                    session_id: None,
                    tasks_attempted: 3,
                    tasks_succeeded: 2,
                    tasks_failed: 1,
                    total_cost_usd: 0.5,
                    duration_ms: 1000,
                    timestamp: ts,
                },
            ),
            ("custom", custom_event(None, "my_event")),
        ];

        for (expected_type, event) in &cases {
            assert_eq!(
                event.event_type(),
                *expected_type,
                "event_type() mismatch for {expected_type}"
            );
        }
    }

    #[test]
    fn event_type_matches_serde_json_tag() {
        let event = provider_call_event();
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["event_type"], event.event_type());
    }

    // --- session_id() ---

    #[test]
    fn session_id_returns_value_when_set() {
        let event = custom_event(Some("session-abc"), "x");
        assert_eq!(event.session_id(), Some("session-abc"));
    }

    #[test]
    fn session_id_returns_none_when_absent() {
        let event = custom_event(None, "x");
        assert!(event.session_id().is_none());
    }

    #[test]
    fn provider_call_session_id() {
        let event = provider_call_event();
        assert_eq!(event.session_id(), Some("sess-1"));
    }

    // --- timestamp() ---

    #[test]
    fn timestamp_is_accessible_for_all_variants() {
        let ts = now();
        let event = AnalyticsEvent::Custom {
            session_id: None,
            name: "t".into(),
            payload: serde_json::Value::Null,
            timestamp: ts,
        };
        // Within 1-second tolerance of our `ts`
        let diff = (event.timestamp() - ts).num_milliseconds().abs();
        assert!(diff < 1000);
    }

    // --- Serialization roundtrips ---

    #[test]
    fn custom_event_roundtrip() {
        let event = custom_event(Some("s1"), "my_event");
        let json = serde_json::to_string(&event).unwrap();
        let back: AnalyticsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type(), "custom");
        assert_eq!(back.session_id(), Some("s1"));
    }

    #[test]
    fn provider_call_roundtrip() {
        let event = provider_call_event();
        let json = serde_json::to_string(&event).unwrap();
        let back: AnalyticsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type(), "provider_call");
    }

    #[test]
    fn tool_call_roundtrip() {
        let event = AnalyticsEvent::ToolCall {
            session_id: None,
            agent_id: Some("agent-1".to_string()),
            tool_name: "read_file".to_string(),
            tool_use_id: "use-xyz".to_string(),
            is_error: true,
            duration_ms: Some(250),
            timestamp: now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AnalyticsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type(), "tool_call");
    }

    #[test]
    fn agent_run_event_roundtrip() {
        let event = AnalyticsEvent::AgentRun {
            session_id: Some("s".to_string()),
            agent_id: "agent-1".to_string(),
            task_id: "task-1".to_string(),
            prompt_hash: "abc123".to_string(),
            success: true,
            total_iterations: 5,
            total_tool_calls: 10,
            tool_error_count: 1,
            tools_used: vec!["bash".to_string(), "read".to_string()],
            total_prompt_tokens: 500,
            total_completion_tokens: 300,
            total_cost_usd: 0.05,
            duration_ms: 2000,
            failure_category: None,
            timestamp: now(),
            compliance: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AnalyticsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type(), "agent_run");
        assert_eq!(back.session_id(), Some("s"));
    }
}
