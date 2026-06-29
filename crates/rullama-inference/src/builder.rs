//! Quickstart builder for `ChatAgent`.
//!
//! Replaces the hand-wiring callers used to have to do:
//! ```ignore
//! let agent = ChatAgent::new(provider, executor, ChatOptions::default())
//!     .with_system_prompt("...")
//!     .with_max_tool_rounds(20)
//!     .with_tool_concurrency(4)
//!     .with_budget(guard);
//! ```
//! with the more discoverable:
//! ```ignore
//! let agent = AgentBuilder::new()
//!     .provider(provider)
//!     .tools(executor)
//!     .system("you are helpful")
//!     .max_iterations(20)
//!     .build_chat_agent()?;
//! ```
//!
//! Returns `Result` so the build step can fail with a clear message when
//! a required field is missing (provider, tools) instead of panicking
//! the way a `ChatAgent::new(...)` with `unwrap()`s would.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use rullama_call_policy::BudgetGuard;
use rullama_core::{ChatOptions, Provider};
use rullama_tool_runtime::ToolExecutor;

use crate::chat_agent::ChatAgent;

/// Fluent builder for [`ChatAgent`]. Most fields are optional; only
/// `provider` and `tools` are required to `build_chat_agent()`.
pub struct AgentBuilder {
    provider: Option<Arc<dyn Provider>>,
    executor: Option<Arc<dyn ToolExecutor>>,
    options: ChatOptions,
    system_prompt: Option<String>,
    max_iterations: Option<usize>,
    tool_concurrency: Option<usize>,
    summarization_keep_tail: Option<usize>,
    budget: Option<BudgetGuard>,
}

impl AgentBuilder {
    /// Start a fresh builder.
    pub fn new() -> Self {
        Self {
            provider: None,
            executor: None,
            options: ChatOptions::default(),
            system_prompt: None,
            max_iterations: None,
            tool_concurrency: None,
            summarization_keep_tail: None,
            budget: None,
        }
    }

    /// Set the LLM provider (required).
    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Set the tool executor (required).
    pub fn tools(mut self, e: Arc<dyn ToolExecutor>) -> Self {
        self.executor = Some(e);
        self
    }

    /// Override the default [`ChatOptions`] (temperature, model, etc.).
    pub fn options(mut self, o: ChatOptions) -> Self {
        self.options = o;
        self
    }

    /// Add a system prompt as the first message in the conversation.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Maximum number of tool-call rounds before the agent stops.
    /// Default: 10 (whatever [`ChatAgent::new`] defaults to).
    pub fn max_iterations(mut self, rounds: usize) -> Self {
        self.max_iterations = Some(rounds);
        self
    }

    /// Maximum parallel tool dispatches per round. Default: 4.
    pub fn tool_concurrency(mut self, n: usize) -> Self {
        self.tool_concurrency = Some(n);
        self
    }

    /// Number of trailing messages preserved verbatim during history
    /// summarisation. Default: 6.
    pub fn summarization_keep_tail(mut self, keep: usize) -> Self {
        self.summarization_keep_tail = Some(keep);
        self
    }

    /// Attach a shared [`BudgetGuard`] to enforce per-session caps.
    pub fn budget(mut self, guard: BudgetGuard) -> Self {
        self.budget = Some(guard);
        self
    }

    /// Build a [`ChatAgent`] with the accumulated configuration.
    ///
    /// Returns `Err` if `provider` or `tools` weren't set.
    pub fn build_chat_agent(self) -> Result<ChatAgent> {
        let provider = self.provider.ok_or_else(|| {
            anyhow!("AgentBuilder: `provider` is required; call .provider(...) before building")
        })?;
        let executor = self.executor.ok_or_else(|| {
            anyhow!("AgentBuilder: `tools` is required; call .tools(...) before building")
        })?;

        let mut agent = ChatAgent::new(provider, executor, self.options);
        if let Some(sp) = self.system_prompt {
            agent = agent.with_system_prompt(&sp);
        }
        if let Some(n) = self.max_iterations {
            agent = agent.with_max_tool_rounds(n);
        }
        if let Some(n) = self.tool_concurrency {
            agent = agent.with_tool_concurrency(n);
        }
        if let Some(n) = self.summarization_keep_tail {
            agent = agent.with_summarization_keep_tail(n);
        }
        if let Some(guard) = self.budget {
            agent = agent.with_budget(guard);
        }
        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rullama_tool_runtime::ToolRegistry;

    fn fake_executor() -> Arc<dyn ToolExecutor> {
        use rullama_core::ToolContext;
        use rullama_tool_builtins::BuiltinToolExecutor;
        Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        ))
    }

    #[test]
    fn missing_provider_errors() {
        let result = AgentBuilder::new().tools(fake_executor()).build_chat_agent();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("`provider` is required"), "got: {err}");
    }

    #[test]
    fn missing_tools_errors() {
        let provider = Arc::new(rullama_test_fixtures::ScriptedProvider::always_text(
            "test", "hi",
        )) as Arc<dyn Provider>;
        let result = AgentBuilder::new().provider(provider).build_chat_agent();
        let err = result.unwrap_err().to_string();
        assert!(err.contains("`tools` is required"), "got: {err}");
    }

    #[test]
    fn happy_path_builds() {
        let provider = Arc::new(rullama_test_fixtures::ScriptedProvider::always_text(
            "test", "hi",
        )) as Arc<dyn Provider>;
        let _agent = AgentBuilder::new()
            .provider(provider)
            .tools(fake_executor())
            .system("you are helpful")
            .max_iterations(20)
            .tool_concurrency(2)
            .build_chat_agent()
            .expect("builder should succeed with provider + tools");
    }
}
