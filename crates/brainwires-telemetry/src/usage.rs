use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single billable action emitted by an agent or framework component.
///
/// Every variant carries the `agent_id` of the agent that incurred the cost
/// and a `timestamp` for ledger ordering.  The `cost_usd` field is the
/// pre-computed USD charge for this event using the provider's published rates
/// (or a conservative estimate when exact pricing is unavailable).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageEvent {
    /// Tokens consumed by a provider `chat()` call.
    Tokens {
        /// Agent that incurred this cost.
        agent_id: String,
        /// Human-readable model identifier, e.g. `"anthropic/claude-sonnet-4-6"`.
        model: String,
        /// Total tokens (prompt + completion).
        total_tokens: u64,
        /// Pre-computed USD charge for this call.
        cost_usd: f64,
        /// When the call completed.
        timestamp: DateTime<Utc>,
    },

    /// A single tool invocation (bash, file read, MCP call, etc.).
    ToolCall {
        /// Agent that invoked the tool.
        agent_id: String,
        /// Tool identifier as seen by the agent runtime.
        tool_name: String,
        /// Pre-computed USD charge (zero for most built-in tools; non-zero for
        /// paid external APIs billed per-call).
        cost_usd: f64,
        /// When the tool call completed.
        timestamp: DateTime<Utc>,
    },

    /// Seconds of remote sandbox execution consumed (E2B, OpenSandbox, etc.).
    SandboxSeconds {
        /// Agent that ran the sandbox.
        agent_id: String,
        /// Which sandbox provider was used.
        provider: String,
        /// Wall-clock seconds billed.
        seconds: f64,
        /// Pre-computed USD charge.
        cost_usd: f64,
        /// When the sandbox run ended.
        timestamp: DateTime<Utc>,
    },

    /// A call to an external paid API (image generation, web search, etc.).
    ApiCall {
        /// Agent that initiated the call.
        agent_id: String,
        /// Service identifier, e.g. `"openai/dall-e-3"`.
        service: String,
        /// Pre-computed USD charge.
        cost_usd: f64,
        /// When the API call completed.
        timestamp: DateTime<Utc>,
    },

    /// Escape hatch for custom billable events.
    Custom {
        /// Agent that incurred this cost.
        agent_id: String,
        /// Caller-defined event name (surfaced to the billing hook).
        name: String,
        /// Pre-computed USD charge.
        cost_usd: f64,
        /// Free-form payload for the billing hook to interpret.
        metadata: serde_json::Value,
        /// When the event occurred.
        timestamp: DateTime<Utc>,
    },
}

impl UsageEvent {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a [`UsageEvent::Tokens`] event stamped with the current time.
    pub fn tokens(
        agent_id: impl Into<String>,
        model: impl Into<String>,
        total_tokens: u64,
        cost_usd: f64,
    ) -> Self {
        Self::Tokens {
            agent_id: agent_id.into(),
            model: model.into(),
            total_tokens,
            cost_usd,
            timestamp: Utc::now(),
        }
    }

    /// Create a [`UsageEvent::ToolCall`] event with zero cost (most built-ins).
    pub fn tool_call(agent_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self::tool_call_paid(agent_id, tool_name, 0.0)
    }

    /// Create a [`UsageEvent::ToolCall`] event with an explicit USD charge.
    pub fn tool_call_paid(
        agent_id: impl Into<String>,
        tool_name: impl Into<String>,
        cost_usd: f64,
    ) -> Self {
        Self::ToolCall {
            agent_id: agent_id.into(),
            tool_name: tool_name.into(),
            cost_usd,
            timestamp: Utc::now(),
        }
    }

    /// Create a [`UsageEvent::SandboxSeconds`] event.
    pub fn sandbox_seconds(
        agent_id: impl Into<String>,
        provider: impl Into<String>,
        seconds: f64,
        cost_usd: f64,
    ) -> Self {
        Self::SandboxSeconds {
            agent_id: agent_id.into(),
            provider: provider.into(),
            seconds,
            cost_usd,
            timestamp: Utc::now(),
        }
    }

    /// Create a [`UsageEvent::ApiCall`] event.
    pub fn api_call(
        agent_id: impl Into<String>,
        service: impl Into<String>,
        cost_usd: f64,
    ) -> Self {
        Self::ApiCall {
            agent_id: agent_id.into(),
            service: service.into(),
            cost_usd,
            timestamp: Utc::now(),
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// The agent that incurred this cost.
    pub fn agent_id(&self) -> &str {
        match self {
            Self::Tokens { agent_id, .. } => agent_id,
            Self::ToolCall { agent_id, .. } => agent_id,
            Self::SandboxSeconds { agent_id, .. } => agent_id,
            Self::ApiCall { agent_id, .. } => agent_id,
            Self::Custom { agent_id, .. } => agent_id,
        }
    }

    /// USD cost for this single event.
    pub fn cost_usd(&self) -> f64 {
        match self {
            Self::Tokens { cost_usd, .. } => *cost_usd,
            Self::ToolCall { cost_usd, .. } => *cost_usd,
            Self::SandboxSeconds { cost_usd, .. } => *cost_usd,
            Self::ApiCall { cost_usd, .. } => *cost_usd,
            Self::Custom { cost_usd, .. } => *cost_usd,
        }
    }

    /// Event timestamp.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::Tokens { timestamp, .. } => *timestamp,
            Self::ToolCall { timestamp, .. } => *timestamp,
            Self::SandboxSeconds { timestamp, .. } => *timestamp,
            Self::ApiCall { timestamp, .. } => *timestamp,
            Self::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// Serde discriminant tag (matches the SQLite `kind` column).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Tokens { .. } => "tokens",
            Self::ToolCall { .. } => "tool_call",
            Self::SandboxSeconds { .. } => "sandbox_seconds",
            Self::ApiCall { .. } => "api_call",
            Self::Custom { .. } => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_constructor_sets_fields() {
        let ev = UsageEvent::tokens("agent-1", "openai/gpt-4o", 500, 0.005);
        assert_eq!(ev.agent_id(), "agent-1");
        assert_eq!(ev.cost_usd(), 0.005);
        assert_eq!(ev.kind(), "tokens");
    }

    #[test]
    fn tool_call_has_zero_cost() {
        let ev = UsageEvent::tool_call("agent-1", "bash");
        assert_eq!(ev.cost_usd(), 0.0);
        assert_eq!(ev.kind(), "tool_call");
    }

    #[test]
    fn serde_roundtrip() {
        let ev = UsageEvent::tokens("a", "model", 100, 0.001);
        let json = serde_json::to_string(&ev).unwrap();
        let back: UsageEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "tokens");
        assert_eq!(back.agent_id(), "a");
    }

    #[test]
    fn kind_matches_serde_tag() {
        let events = [
            UsageEvent::tokens("a", "m", 1, 0.0),
            UsageEvent::tool_call("a", "bash"),
            UsageEvent::sandbox_seconds("a", "opensandbox", 1.5, 0.001),
            UsageEvent::api_call("a", "openai/dalle", 0.04),
        ];
        for ev in &events {
            let json = serde_json::to_string(ev).unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(v["kind"].as_str().unwrap(), ev.kind());
        }
    }
}
