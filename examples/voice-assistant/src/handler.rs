//! Voice handler wired to the Brainwires harness.
//!
//! Replaces the legacy direct-OpenAI loop with the full `ChatAgent` pipeline:
//!
//! - `brainwires_core::Provider` for LLM I/O (OpenAI-compatible).
//! - `brainwires_call_policy::BudgetGuard` for hard token/cost caps.
//! - `brainwires_stores::SessionStore` for persistence across restarts.
//! - `brainwires_agent::personas::PersonaProvider` for prompt assembly.
//! - `CacheStrategy::SystemAndTools` so the static persona is cached on
//!   every turn instead of rebuilt from scratch.

use std::sync::Arc;

use async_trait::async_trait;
use brainwires_agent::personas::{PersonaContext, PersonaProvider, blocks_to_system_text};
use brainwires_call_policy::BudgetGuard;
use brainwires_core::ToolContext;
use brainwires_core::{CacheStrategy, ChatOptions, Provider};
use brainwires_hardware::audio::{
    assistant::VoiceAssistantHandler, error::AudioError, types::Transcript,
};
use brainwires_inference::ChatAgent;
use brainwires_stores::{ArcSessionStore, SessionId};
use brainwires_tool_builtins::BuiltinToolExecutor;
use brainwires_tool_runtime::{ToolExecutor, ToolRegistry};
use tokio::sync::Mutex;
use tracing::{info, warn};

#[cfg(any(feature = "wake-word", feature = "wake-word-dtw"))]
use brainwires_hardware::audio::wake_word::WakeWordDetection;

/// Voice handler that routes transcripts through a [`ChatAgent`].
pub struct LlmHandler {
    agent: Mutex<ChatAgent>,
    /// Session persistence — `Some` means we load on startup and save each
    /// turn; `None` keeps history in memory only.
    session: Option<SessionPersistence>,
}

struct SessionPersistence {
    store: ArcSessionStore,
    id: SessionId,
}

impl LlmHandler {
    /// Build a handler from a provider and persona text. Optional session
    /// persistence is attached via [`Self::with_session_store`].
    pub async fn new(
        provider: Arc<dyn Provider>,
        persona: Arc<dyn PersonaProvider>,
        budget: Option<BudgetGuard>,
    ) -> anyhow::Result<Self> {
        // Assemble the system prompt once per session — voice interactions
        // are a long-lived conversation with a stable persona, so there's
        // no reason to rebuild it on every turn like the legacy handler did.
        let blocks = persona.build(&PersonaContext::new()).await?;
        let system = blocks_to_system_text(&blocks);

        let options = ChatOptions::default().cache_strategy(CacheStrategy::SystemAndTools);

        let executor: Arc<dyn ToolExecutor> = Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        ));

        let mut agent = ChatAgent::new(provider, executor, options).with_system_prompt(&system);
        if let Some(guard) = budget {
            agent = agent.with_budget(guard);
        }
        Ok(Self {
            agent: Mutex::new(agent),
            session: None,
        })
    }

    /// Attach a session store. The handler will load any existing transcript
    /// for `id` on startup and overwrite it after every turn.
    pub async fn with_session_store(
        mut self,
        store: ArcSessionStore,
        id: SessionId,
    ) -> anyhow::Result<Self> {
        if let Some(msgs) = store.load(&id).await? {
            info!(
                "restored {} messages from session store for '{id}'",
                msgs.len()
            );
            self.agent.lock().await.restore_messages(msgs);
        }
        self.session = Some(SessionPersistence { store, id });
        Ok(self)
    }

    /// Clear conversation history (does NOT drop the system prompt).
    #[allow(dead_code)]
    pub async fn clear_history(&self) {
        self.agent.lock().await.clear_history();
    }

    async fn persist(&self) {
        if let Some(ref s) = self.session {
            let agent = self.agent.lock().await;
            if let Err(e) = s.store.save(&s.id, agent.messages()).await {
                warn!("failed to persist session '{}': {e}", s.id);
            }
        }
    }
}

#[async_trait]
impl VoiceAssistantHandler for LlmHandler {
    #[cfg(any(feature = "wake-word", feature = "wake-word-dtw"))]
    async fn on_wake_word(&self, detection: &WakeWordDetection) {
        info!(
            keyword = %detection.keyword,
            score = detection.score,
            "Wake word detected — listening…"
        );
    }

    async fn on_speech(&self, transcript: &Transcript) -> Option<String> {
        let text = transcript.text.trim();
        if text.is_empty() {
            return None;
        }
        info!("You: {text}");

        let reply = {
            let mut agent = self.agent.lock().await;
            match agent.process_message(text).await {
                Ok(r) => r,
                Err(e) => {
                    warn!("LLM error: {e:#}");
                    // Speak a short, user-visible reason instead of
                    // silently dropping like the legacy handler did.
                    return Some(short_error_for_tts(&e));
                }
            }
        };

        self.persist().await;

        if reply.trim().is_empty() {
            None
        } else {
            info!("Assistant: {reply}");
            Some(reply)
        }
    }

    async fn on_error(&self, error: &AudioError) {
        warn!("Pipeline error: {error}");
    }
}

/// Produce a short TTS-friendly message for a harness error.
fn short_error_for_tts(e: &anyhow::Error) -> String {
    use brainwires_call_policy::ResilienceError;
    if let Some(re) = e.downcast_ref::<ResilienceError>() {
        return match re {
            ResilienceError::BudgetExceeded { kind, .. } => {
                format!("Sorry, I've hit the {kind} budget for this session.")
            }
            ResilienceError::CircuitOpen { .. } => {
                "Sorry, the model is temporarily unavailable.".into()
            }
            ResilienceError::RetriesExhausted { .. } => {
                "Sorry, I couldn't reach the model after several tries.".into()
            }
            ResilienceError::DeadlineExceeded { .. } => {
                "Sorry, the model took too long to respond.".into()
            }
        };
    }
    "Sorry, I hit an error.".into()
}
