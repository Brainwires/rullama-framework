//! `SessionController` implementation that wraps a live `ChatAgent` session.
//!
//! The gateway holds per-user `Arc<Mutex<ChatAgent>>` instances; this controller
//! acquires the lock lazily inside each method so the caller keeps the lock
//! scope minimal.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use brainwires_agent::ChatAgent;
use brainwires_core::{Message, Role};

use super::commands::{CompactResult, SessionController, StatusReport, ThinkLevel, UsageReport};

/// Static capabilities the gateway knows about the active session. Thinking
/// support is encoded here because `ChatOptions` in v0.11 does not expose a
/// thinking-budget field — providers that support it (e.g. Anthropic with
/// extended thinking) thread it through their own config.
pub struct AgentSessionHandle {
    pub agent: Arc<Mutex<ChatAgent>>,
    pub provider_name: String,
    pub model: String,
    pub session_id: String,
    pub channels_connected: Vec<String>,
    /// Whether the provider supports extended thinking at all.
    pub thinking_supported: bool,
    /// Callback invoked on `/restart` to rebuild the underlying `ChatAgent`.
    /// Receives the current agent Arc so the caller can swap it in place.
    #[allow(clippy::type_complexity)]
    pub rebuild: Option<Arc<dyn Fn() -> ChatAgent + Send + Sync>>,
}

#[async_trait]
impl SessionController for AgentSessionHandle {
    async fn reset_session(&mut self) -> anyhow::Result<()> {
        let mut agent = self.agent.lock().await;
        let system_prompt = agent
            .messages()
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()));
        agent.clear_history();
        if let Some(prompt) = system_prompt {
            agent.restore_messages(vec![Message::system(&prompt)]);
        }
        agent.reset_usage();
        Ok(())
    }

    async fn compact_session(&mut self) -> anyhow::Result<CompactResult> {
        let mut agent = self.agent.lock().await;
        let before = agent.message_count();
        let summary = summarise(agent.messages());
        let system_prompt = agent
            .messages()
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()));
        agent.clear_history();
        let mut restored = Vec::new();
        if let Some(p) = system_prompt {
            restored.push(Message::system(&p));
        }
        restored.push(Message::system(format!(
            "[conversation summary]\n{summary}",
        )));
        agent.restore_messages(restored);
        let after = agent.message_count();
        Ok(CompactResult {
            messages_before: before,
            messages_after: after,
        })
    }

    async fn usage_report(&self) -> anyhow::Result<UsageReport> {
        let agent = self.agent.lock().await;
        let usage = agent.cumulative_usage();
        Ok(UsageReport {
            input_tokens: usage.prompt_tokens as u64,
            output_tokens: usage.completion_tokens as u64,
            cache_read: 0,
            cache_write: 0,
            cost_usd: None,
        })
    }

    async fn status_report(&self) -> StatusReport {
        let message_count = {
            let agent = self.agent.lock().await;
            agent.message_count()
        };
        StatusReport {
            provider: self.provider_name.clone(),
            model: self.model.clone(),
            session_id: self.session_id.clone(),
            message_count,
            think_level: ThinkLevel::Off,
            trace_remaining: 0,
            channels_connected: self.channels_connected.clone(),
        }
    }

    async fn restart_session(&mut self) -> anyhow::Result<()> {
        let Some(rebuild) = self.rebuild.clone() else {
            // No rebuild callback: fall back to a full reset so the user still
            // gets a clean session.
            return self.reset_session().await;
        };
        let new_agent = (rebuild)();
        let mut guard = self.agent.lock().await;
        *guard = new_agent;
        Ok(())
    }

    fn set_think_level(&mut self, _level: ThinkLevel) -> bool {
        // v0.11: no provider surfaces a thinking knob through `ChatOptions`.
        // Gateway caches the request in `SessionSlashState`; providers that
        // grow a thinking field can read it from there in a follow-up.
        self.thinking_supported
    }
}

/// Build a best-effort plaintext summary of the current conversation suitable
/// for seeding a fresh session. We keep this LLM-free so `/compact` works even
/// when the provider is unreachable.
fn summarise(messages: &[Message]) -> String {
    let mut bullets: Vec<String> = Vec::new();
    for msg in messages {
        if msg.role == Role::System {
            continue;
        }
        let Some(text) = msg.text() else { continue };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::System => continue,
        };
        let snippet: String = trimmed.chars().take(200).collect();
        bullets.push(format!("- {role}: {snippet}"));
        if bullets.len() >= 10 {
            break;
        }
    }
    if bullets.is_empty() {
        "(empty conversation)".to_string()
    } else {
        bullets.join("\n")
    }
}
