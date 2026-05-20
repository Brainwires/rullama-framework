//! Concrete [`SessionSpawnFactory`] used by the BrainClaw daemon.
//!
//! When the agent calls the `sessions_spawn` tool, the gateway's
//! [`GatewaySessionBroker`] invokes this factory to actually materialise a
//! fresh [`ChatAgent`]. The factory captures the daemon's shared provider,
//! executor, and default chat options at construction time, then clones them
//! per-spawn.
//!
//! # Limitations
//!
//! * `SpawnRequest::model` is *not* currently honoured — the daemon has a
//!   single provider instance baked at startup, and swapping models at
//!   runtime per-spawn would require rebuilding the whole provider stack. If
//!   `req.model` is set the spawn returns an error result so the agent sees
//!   a clear failure rather than a silently-ignored override.
//! * `SpawnRequest::tools` is *not* honoured — the daemon's [`ToolExecutor`]
//!   is composed at startup (sandbox, sessions wrapper, etc.) and filtering
//!   its available toolset post-hoc would require rebuilding that stack.
//!   Requesting `tools` returns an error result.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};

use brainwires_agent::ChatAgent;
use brainwires_core::{ChatOptions, Provider};
use brainwires_gateway::sessions_broker::SessionSpawnFactory;
use brainwires_tools::{SessionId, SessionMessage, SpawnRequest, ToolExecutor};

/// Factory that builds a fresh [`ChatAgent`] per [`SessionSpawnFactory::spawn`]
/// call, seeds it with the caller's prompt, and drives a single
/// user -> assistant turn in the background — plus a long-running drainer
/// for follow-up `sessions_send` messages.
pub struct BrainClawSpawnFactory {
    provider: Arc<dyn Provider>,
    /// Shared executor. Cloning the `Arc` is cheap; the spawned ChatAgent
    /// inherits the parent's toolset (sandbox wrapping, sessions tools, etc.).
    executor: Arc<dyn ToolExecutor>,
    /// Default chat options including the baseline system prompt. Per-spawn
    /// overrides (`req.system`) replace the `system` field in a clone.
    default_options: ChatOptions,
    /// Max tool rounds for spawned agents.
    max_tool_rounds: usize,
}

impl BrainClawSpawnFactory {
    /// Construct a new factory. Provider, executor, and options are shared
    /// with the rest of the daemon.
    pub fn new(
        provider: Arc<dyn Provider>,
        executor: Arc<dyn ToolExecutor>,
        default_options: ChatOptions,
    ) -> Self {
        Self {
            provider,
            executor,
            default_options,
            max_tool_rounds: 10,
        }
    }

    /// Override the default tool-round cap (default: 10).
    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.max_tool_rounds = rounds;
        self
    }
}

#[async_trait]
impl SessionSpawnFactory for BrainClawSpawnFactory {
    async fn spawn(
        &self,
        new_session_id: &SessionId,
        _parent: &SessionId,
        req: &SpawnRequest,
        on_assistant_reply: Arc<dyn Fn(SessionMessage) + Send + Sync>,
        mut inbound_rx: mpsc::UnboundedReceiver<String>,
    ) -> anyhow::Result<Option<Arc<Mutex<ChatAgent>>>> {
        // Reject unsupported overrides with a clear error — do NOT silently
        // spawn a session that ignores the caller's request.
        if req.model.is_some() {
            anyhow::bail!(
                "sessions_spawn: `model` override is not supported by this daemon \
                 (provider is fixed at startup). Omit `model` to inherit the parent's provider."
            );
        }
        if req.tools.is_some() {
            anyhow::bail!(
                "sessions_spawn: `tools` override is not supported by this daemon \
                 (tool executor is fixed at startup). Omit `tools` to inherit the parent's toolset."
            );
        }

        // Apply per-spawn system-prompt override, if any.
        let mut options = self.default_options.clone();
        if let Some(ref sys) = req.system {
            options.system = Some(sys.clone());
        }

        // Build the ChatAgent. We deliberately *do not* attach pre-tool
        // hooks (approval / shell) to spawned sub-sessions: the parent has
        // already been gated (sessions_spawn requires approval at the
        // registry level), and cascading hooks into sub-sessions would
        // deadlock on the approval channel (no human peer to answer).
        let agent = ChatAgent::new(
            Arc::clone(&self.provider),
            Arc::clone(&self.executor),
            options,
        )
        .with_max_tool_rounds(self.max_tool_rounds);

        let agent_arc = Arc::new(Mutex::new(agent));

        // Drive the first turn + subsequent sessions_send follow-ups in a
        // background task so `sessions_spawn` returns promptly when the
        // caller did not ask for `wait_for_first_reply`.
        let agent_for_task = Arc::clone(&agent_arc);
        let prompt = req.prompt.clone();
        let new_sid_for_task = new_session_id.clone();
        let on_reply = Arc::clone(&on_assistant_reply);

        tokio::spawn(async move {
            // 1. First user -> assistant turn with the seed prompt.
            let first_reply_text = {
                let mut guard = agent_for_task.lock().await;
                match guard.process_message(&prompt).await {
                    Ok(t) => Some(t),
                    Err(e) => {
                        tracing::warn!(
                            session_id = %new_sid_for_task,
                            error = %e,
                            "spawn factory: first-turn process_message failed",
                        );
                        None
                    }
                }
            };

            if let Some(text) = first_reply_text.as_ref() {
                let msg = SessionMessage {
                    role: "assistant".to_string(),
                    content: text.clone(),
                    timestamp: chrono::Utc::now(),
                };
                (on_reply)(msg);
            }

            // 2. Drain follow-up `sessions_send` messages and run each
            //    through the ChatAgent. Replies are discarded (the parent
            //    can read them back via `sessions_history`), matching the
            //    fire-and-forget contract of `sessions_send`.
            while let Some(text) = inbound_rx.recv().await {
                let mut guard = agent_for_task.lock().await;
                if let Err(e) = guard.process_message(&text).await {
                    tracing::warn!(
                        session_id = %new_sid_for_task,
                        error = %e,
                        "spawn factory: follow-up process_message failed",
                    );
                }
            }
        });

        Ok(Some(agent_arc))
    }
}
