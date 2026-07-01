//! Lifecycle Hooks
//!
//! Provides an extensible hook system for intercepting and reacting to
//! framework events such as agent lifecycle transitions, tool executions,
//! provider requests, and validation passes.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rullama_core::lifecycle::*;
//!
//! struct MetricsHook;
//!
//! #[async_trait::async_trait]
//! impl LifecycleHook for MetricsHook {
//!     fn name(&self) -> &str { "metrics" }
//!     fn priority(&self) -> i32 { 100 }
//!
//!     async fn on_event(&self, event: &LifecycleEvent) -> HookResult {
//!         // record metrics...
//!         HookResult::Continue
//!     }
//! }
//!
//! let mut registry = HookRegistry::new();
//! registry.register(MetricsHook);
//! ```

use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

/// Events emitted during framework operation.
#[derive(Debug, Clone)]
pub enum LifecycleEvent {
    /// An agent has been created and is about to start.
    AgentStarted {
        /// The agent's unique identifier.
        agent_id: String,
        /// Description of the task assigned to the agent.
        task_description: String,
    },
    /// An agent completed its task successfully.
    AgentCompleted {
        /// The agent's unique identifier.
        agent_id: String,
        /// Number of iterations the agent executed.
        iterations: u32,
        /// Summary of the completed work.
        summary: String,
    },
    /// An agent failed to complete its task.
    AgentFailed {
        /// The agent's unique identifier.
        agent_id: String,
        /// Error message describing the failure.
        error: String,
        /// Number of iterations before failure.
        iterations: u32,
    },
    /// A tool is about to be executed.
    ToolBeforeExecute {
        /// The agent invoking the tool, if any.
        agent_id: Option<String>,
        /// Name of the tool being invoked.
        tool_name: String,
        /// Arguments passed to the tool.
        args: Value,
    },
    /// A tool has finished executing.
    ToolAfterExecute {
        /// The agent that invoked the tool, if any.
        agent_id: Option<String>,
        /// Name of the tool that was executed.
        tool_name: String,
        /// Whether the tool execution succeeded.
        success: bool,
        /// Duration of the tool execution in milliseconds.
        duration_ms: u64,
    },
    /// A request is about to be sent to an AI provider.
    ProviderRequest {
        /// The agent making the request, if any.
        agent_id: Option<String>,
        /// Provider name (e.g. "anthropic", "openai").
        provider: String,
        /// Model identifier.
        model: String,
    },
    /// A response was received from an AI provider.
    ProviderResponse {
        /// The agent that made the request, if any.
        agent_id: Option<String>,
        /// Provider name.
        provider: String,
        /// Model identifier.
        model: String,
        /// Number of input tokens consumed.
        input_tokens: u64,
        /// Number of output tokens generated.
        output_tokens: u64,
        /// Duration of the request in milliseconds.
        duration_ms: u64,
    },
    /// Validation has started for an agent's work.
    ValidationStarted {
        /// The agent whose work is being validated.
        agent_id: String,
        /// Names of the validation checks being run.
        checks: Vec<String>,
    },
    /// Validation completed for an agent's work.
    ValidationCompleted {
        /// The agent whose work was validated.
        agent_id: String,
        /// Whether all validation checks passed.
        passed: bool,
        /// List of validation issues found.
        issues: Vec<String>,
    },
}

impl LifecycleEvent {
    /// Returns the event type name for filtering.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::AgentStarted { .. } => "agent_started",
            Self::AgentCompleted { .. } => "agent_completed",
            Self::AgentFailed { .. } => "agent_failed",
            Self::ToolBeforeExecute { .. } => "tool_before_execute",
            Self::ToolAfterExecute { .. } => "tool_after_execute",
            Self::ProviderRequest { .. } => "provider_request",
            Self::ProviderResponse { .. } => "provider_response",
            Self::ValidationStarted { .. } => "validation_started",
            Self::ValidationCompleted { .. } => "validation_completed",
        }
    }

    /// Returns the agent ID associated with this event, if any.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            Self::AgentStarted { agent_id, .. }
            | Self::AgentCompleted { agent_id, .. }
            | Self::AgentFailed { agent_id, .. }
            | Self::ValidationStarted { agent_id, .. }
            | Self::ValidationCompleted { agent_id, .. } => Some(agent_id),
            Self::ToolBeforeExecute { agent_id, .. }
            | Self::ToolAfterExecute { agent_id, .. }
            | Self::ProviderRequest { agent_id, .. }
            | Self::ProviderResponse { agent_id, .. } => agent_id.as_deref(),
        }
    }

    /// Returns the tool name if this is a tool-related event.
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::ToolBeforeExecute { tool_name, .. }
            | Self::ToolAfterExecute { tool_name, .. } => Some(tool_name),
            _ => None,
        }
    }
}

/// Result of a hook invocation.
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue processing normally.
    Continue,
    /// Cancel the operation with a reason.
    Cancel {
        /// Human-readable reason for cancellation.
        reason: String,
    },
    /// Continue but with modified data (e.g., modified tool args).
    Modified(Value),
}

/// Filter to control which events a hook receives.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Only receive events for these agent IDs (empty = all).
    pub agent_ids: HashSet<String>,
    /// Only receive these event types (empty = all).
    pub event_types: HashSet<String>,
    /// Only receive tool events for these tool names (empty = all).
    pub tool_names: HashSet<String>,
}

impl EventFilter {
    /// Returns true if this filter matches the given event.
    pub fn matches(&self, event: &LifecycleEvent) -> bool {
        if !self.event_types.is_empty() && !self.event_types.contains(event.event_type()) {
            return false;
        }
        if !self.agent_ids.is_empty() {
            if let Some(id) = event.agent_id() {
                if !self.agent_ids.contains(id) {
                    return false;
                }
            } else {
                return false;
            }
        }
        if !self.tool_names.is_empty()
            && let Some(name) = event.tool_name()
            && !self.tool_names.contains(name)
        {
            return false;
        }
        true
    }
}

/// Trait for lifecycle hooks that react to framework events.
#[async_trait::async_trait]
pub trait LifecycleHook: Send + Sync {
    /// Human-readable name for this hook.
    fn name(&self) -> &str;

    /// Priority for ordering (lower runs first). Default: 0.
    fn priority(&self) -> i32 {
        0
    }

    /// Optional filter. Default: receive all events.
    fn filter(&self) -> Option<EventFilter> {
        None
    }

    /// Called when a matching event occurs.
    async fn on_event(&self, event: &LifecycleEvent) -> HookResult;
}

/// Registry that manages and dispatches lifecycle hooks.
pub struct HookRegistry {
    hooks: Vec<Arc<dyn LifecycleHook>>,
}

impl HookRegistry {
    /// Create an empty hook registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a new hook. Hooks are sorted by priority after insertion.
    pub fn register(&mut self, hook: impl LifecycleHook + 'static) {
        self.hooks.push(Arc::new(hook));
        self.hooks.sort_by_key(|h| h.priority());
    }

    /// Register a pre-built Arc hook.
    pub fn register_arc(&mut self, hook: Arc<dyn LifecycleHook>) {
        self.hooks.push(hook);
        self.hooks.sort_by_key(|h| h.priority());
    }

    /// Dispatch an event to all matching hooks.
    ///
    /// Returns `HookResult::Cancel` if any hook cancels, otherwise `Continue`.
    /// `Modified` results from earlier hooks are passed through.
    pub async fn dispatch(&self, event: &LifecycleEvent) -> HookResult {
        for hook in &self.hooks {
            let matches = hook.filter().map(|f| f.matches(event)).unwrap_or(true);

            if !matches {
                continue;
            }

            match hook.on_event(event).await {
                HookResult::Continue => {}
                result @ HookResult::Cancel { .. } => return result,
                result @ HookResult::Modified(_) => return result,
            }
        }
        HookResult::Continue
    }

    /// Number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Whether the registry has no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingHook {
        name: String,
    }

    #[async_trait::async_trait]
    impl LifecycleHook for CountingHook {
        fn name(&self) -> &str {
            &self.name
        }
        async fn on_event(&self, _event: &LifecycleEvent) -> HookResult {
            HookResult::Continue
        }
    }

    #[test]
    fn test_registry_register() {
        let mut registry = HookRegistry::new();
        assert!(registry.is_empty());
        registry.register(CountingHook {
            name: "test".to_string(),
        });
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_event_filter_matches_all() {
        let filter = EventFilter::default();
        let event = LifecycleEvent::AgentStarted {
            agent_id: "a1".to_string(),
            task_description: "test".to_string(),
        };
        assert!(filter.matches(&event));
    }

    #[test]
    fn test_event_filter_by_type() {
        let filter = EventFilter {
            event_types: HashSet::from(["agent_started".to_string()]),
            ..Default::default()
        };
        let started = LifecycleEvent::AgentStarted {
            agent_id: "a1".to_string(),
            task_description: "test".to_string(),
        };
        let completed = LifecycleEvent::AgentCompleted {
            agent_id: "a1".to_string(),
            iterations: 5,
            summary: "done".to_string(),
        };
        assert!(filter.matches(&started));
        assert!(!filter.matches(&completed));
    }

    #[test]
    fn test_event_type_names() {
        let event = LifecycleEvent::ToolBeforeExecute {
            agent_id: Some("a1".to_string()),
            tool_name: "read_file".to_string(),
            args: serde_json::json!({}),
        };
        assert_eq!(event.event_type(), "tool_before_execute");
        assert_eq!(event.agent_id(), Some("a1"));
        assert_eq!(event.tool_name(), Some("read_file"));
    }
}
