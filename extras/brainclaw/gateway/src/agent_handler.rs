//! Agent-backed inbound handler that bridges gateway events to [`ChatAgent`].
//!
//! [`AgentInboundHandler`] is the ready-to-use [`InboundHandler`] implementation
//! that actually invokes agents when messages arrive from channel adapters. It
//! manages per-user agent sessions and routes responses back to the originating
//! channel.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use brainwires_agent::ChatAgent;
use brainwires_core::lifecycle::{LifecycleEvent, LifecycleHook};
use brainwires_core::{ChatOptions, Provider, ToolContext, ToolUse};
use brainwires_network::channels::events::ChannelEvent;
use brainwires_network::channels::identity::ConversationId;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};
use brainwires_tools::{PreHookDecision, SessionBroker, SessionId, ToolExecutor, ToolPreHook};

use crate::approval::{ApprovalRegistry, ChatApprovalHook};
use crate::channel_registry::ChannelRegistry;
use crate::identity::UserIdentityStore;
use crate::media::MediaProcessor;
use crate::metrics::MetricsCollector;
use crate::middleware::rate_limit::RateLimiter;
use crate::middleware::sanitizer::MessageSanitizer;
use crate::pairing::{PairingHandler, PairingOutcome};
use crate::router::InboundHandler;
use crate::session::SessionManager;
use crate::session_persistence::{SessionStore, session_key};
use crate::sessions_broker::{SessionRegistry, SessionsExecutor};

/// Whether a provider (identified by its `Provider::name()`) is known to
/// support extended-thinking budgets. Used by `/think` to decide between
/// applying the budget and replying "not supported".
fn provider_supports_thinking(name: &str) -> bool {
    matches!(name, "anthropic" | "claude" | "bedrock-anthropic")
}

/// Runs a sequence of [`ToolPreHook`]s in order; the first rejection wins.
struct CompositePreToolHook {
    hooks: Vec<Arc<dyn ToolPreHook>>,
}

#[async_trait]
impl ToolPreHook for CompositePreToolHook {
    async fn before_execute(
        &self,
        tool_use: &ToolUse,
        context: &ToolContext,
    ) -> anyhow::Result<PreHookDecision> {
        for hook in &self.hooks {
            match hook.before_execute(tool_use, context).await? {
                PreHookDecision::Allow => {}
                reject => return Ok(reject),
            }
        }
        Ok(PreHookDecision::Allow)
    }
}

/// A synchronous function that can transform an inbound message text before it
/// is sent to the agent.
///
/// Return `Some(new_text)` to replace the original text, or `None` to leave it
/// unchanged.  Used by the BrainClaw daemon to wire in skill dispatch without
/// creating a circular crate dependency.
pub type TextPreprocessor = dyn Fn(&str) -> Option<String> + Send + Sync;

/// Parsed result of a `/model` command.
enum ModelCommand {
    /// Show the current model.
    Show,
    /// Set a new model for this session.
    Set(String),
    /// Reset to the provider's default model.
    Reset,
}

/// State cells shared between an `AgentInboundHandler` session and its
/// `ChatApprovalHook`: `(current_conversation, current_sender, current_channel_id)`.
type ApprovalContextCell = (
    Arc<RwLock<Option<ConversationId>>>,
    Arc<RwLock<Option<tokio::sync::mpsc::Sender<String>>>>,
    Arc<RwLock<Option<Uuid>>>,
);

/// An [`InboundHandler`] that dispatches incoming messages to per-user
/// [`ChatAgent`] instances and sends responses back through the channel.
pub struct AgentInboundHandler {
    /// Session manager for user-to-agent mapping.
    sessions: Arc<SessionManager>,
    /// Channel registry for sending responses back.
    channels: Arc<ChannelRegistry>,
    /// Per-user agent sessions: (platform, platform_user_id) -> ChatAgent.
    agent_sessions: DashMap<(String, String), Arc<Mutex<ChatAgent>>>,
    /// Shared provider instance.
    provider: Arc<dyn Provider>,
    /// Shared tool executor.
    executor: Arc<dyn ToolExecutor>,
    /// Default chat options (system prompt, temperature, etc.).
    default_options: ChatOptions,
    /// Max tool rounds per message.
    max_tool_rounds: usize,
    /// Optional session persistence backend.
    persistence: Option<Arc<dyn SessionStore>>,
    /// Optional media processor for handling attachments.
    media: Option<Arc<MediaProcessor>>,
    /// Optional message sanitizer for inbound spoofing detection and outbound redaction.
    sanitizer: Option<Arc<MessageSanitizer>>,
    /// Optional per-user rate limiter.
    rate_limiter: Option<Arc<RateLimiter>>,
    /// Optional text preprocessor (e.g. skill dispatch).
    text_preprocessor: Option<Arc<TextPreprocessor>>,
    /// When `Some`, tool calls matching `approval_tools` require user approval.
    approval_registry: Option<Arc<ApprovalRegistry>>,
    /// Set of tool names that require approval.  Empty = all tools.
    approval_tools: HashSet<String>,
    /// Per-user state cells shared with `ChatApprovalHook`:
    /// (platform, user_id) -> (current_conversation, current_sender, current_channel_id)
    approval_contexts: DashMap<(String, String), ApprovalContextCell>,
    /// Optional shell-script pre-tool hook (blocks tool calls on non-zero exit).
    shell_pre_tool_hook: Option<Arc<dyn ToolPreHook>>,
    /// Optional lifecycle hook for session start/end and post-tool events.
    session_hook: Option<Arc<dyn LifecycleHook>>,
    /// Optional TTS processor for synthesizing agent responses to audio.
    #[cfg(feature = "voice")]
    tts: Option<Arc<crate::tts::TtsProcessor>>,
    /// Optional shared metrics collector for token usage tracking.
    metrics: Option<Arc<MetricsCollector>>,
    /// Sessions with Talk Mode enabled: (platform, user_id) set.
    ///
    /// When a session is in Talk Mode, all agent responses are synthesised
    /// to audio via TTS (if configured) regardless of whether the input was
    /// voice or text. Toggle via `/talk on` / `/talk off` commands.
    talk_mode_sessions: DashSet<(String, String)>,
    /// Optional cross-channel user identity store.
    ///
    /// When set, `(platform, user_id)` pairs are resolved to a canonical
    /// UUID before looking up agent sessions, so the same person on Discord
    /// and Telegram shares one agent session and conversation history.
    identity_store: Option<Arc<UserIdentityStore>>,
    /// Per-session model overrides: session key -> model name.
    ///
    /// Set via `/model <name>` command. When present, the agent for that
    /// session uses `ChatOptions::model` override instead of the provider's
    /// default configured model.
    model_overrides: DashMap<(String, String), String>,
    /// Per-session slash-command state (thinking level, trace counter).
    slash_state: DashMap<(String, String), Arc<Mutex<crate::slash::SessionSlashState>>>,
    /// Optional session broker. When set, each `ChatAgent` constructed by
    /// this handler is given a [`SessionsExecutor`] that layers the four
    /// `sessions_*` tools on top of the shared executor — so the agent can
    /// introspect and orchestrate other sessions in the gateway.
    session_broker: Option<Arc<dyn SessionBroker>>,
    /// Shared [`SessionRegistry`]. Must point at the same registry the
    /// [`session_broker`] is backed by so the four `sessions_*` tools see
    /// handler-managed per-user sessions (not just spawned children).
    session_registry: Option<Arc<SessionRegistry>>,
    /// Optional pairing handler — intercepts inbound messages from
    /// unapproved peers before they reach the agent.
    pairing: Option<Arc<PairingHandler>>,
}

impl AgentInboundHandler {
    /// Create a new `AgentInboundHandler`.
    ///
    /// Defaults `max_tool_rounds` to 10.
    pub fn new(
        sessions: Arc<SessionManager>,
        channels: Arc<ChannelRegistry>,
        provider: Arc<dyn Provider>,
        executor: Arc<dyn ToolExecutor>,
        default_options: ChatOptions,
    ) -> Self {
        Self {
            sessions,
            channels,
            agent_sessions: DashMap::new(),
            provider,
            executor,
            default_options,
            max_tool_rounds: 10,
            persistence: None,
            media: None,
            sanitizer: None,
            rate_limiter: None,
            text_preprocessor: None,
            approval_registry: None,
            approval_tools: HashSet::new(),
            approval_contexts: DashMap::new(),
            shell_pre_tool_hook: None,
            session_hook: None,
            #[cfg(feature = "voice")]
            tts: None,
            metrics: None,
            talk_mode_sessions: DashSet::new(),
            identity_store: None,
            model_overrides: DashMap::new(),
            slash_state: DashMap::new(),
            session_broker: None,
            session_registry: None,
            pairing: None,
        }
    }

    /// Attach a [`PairingHandler`] so inbound DMs from unapproved peers
    /// are intercepted before they reach the agent.
    ///
    /// When unset, every inbound message is treated as allowed — callers
    /// that want the secure default must explicitly install a handler.
    pub fn with_pairing(mut self, pairing: Arc<PairingHandler>) -> Self {
        self.pairing = Some(pairing);
        self
    }

    /// Attach a session broker so this handler exposes the `sessions_list`,
    /// `sessions_history`, `sessions_send`, and `sessions_spawn` tools to
    /// every `ChatAgent` it creates.
    ///
    /// The broker is typically a `GatewaySessionBroker` wired over the
    /// gateway's own session registry — see `crate::sessions_broker`.
    pub fn with_session_broker(mut self, broker: Arc<dyn SessionBroker>) -> Self {
        self.session_broker = Some(broker);
        self
    }

    /// Attach the shared [`SessionRegistry`] so every ChatAgent this handler
    /// creates is also registered there. Must be the same Arc passed to the
    /// [`crate::sessions_broker::GatewaySessionBroker`] backing
    /// [`with_session_broker`] — otherwise `sessions_list` / `sessions_history`
    /// / `sessions_send` would not see handler-managed sessions.
    pub fn with_session_registry(mut self, registry: Arc<SessionRegistry>) -> Self {
        self.session_registry = Some(registry);
        self
    }

    /// Enable interactive tool approval via chat.
    ///
    /// When enabled, tool calls whose names are in `tool_names` (or all tools
    /// if `tool_names` is empty) are intercepted: the user receives an approval
    /// prompt in their channel and must reply **yes** or **no** within 60 s.
    pub fn with_tool_approval(mut self, tool_names: HashSet<String>) -> Self {
        self.approval_registry = Some(Arc::new(ApprovalRegistry::new()));
        self.approval_tools = tool_names;
        self
    }

    /// Attach a shell-script pre-tool hook.
    ///
    /// The hook script receives tool call details as JSON on stdin.
    /// A non-zero exit code blocks the tool call; the first line of stdout
    /// is used as the rejection reason.
    ///
    /// If `with_tool_approval` is also active, the shell hook runs first.
    pub fn with_shell_pre_tool_hook(mut self, hook: Arc<dyn ToolPreHook>) -> Self {
        self.shell_pre_tool_hook = Some(hook);
        self
    }

    /// Attach a lifecycle hook for session and post-tool events.
    ///
    /// Events fired: `AgentStarted` (before processing), `AgentCompleted`/
    /// `AgentFailed` (after processing), `ToolAfterExecute` (after each tool).
    pub fn with_session_hook(mut self, hook: Arc<dyn LifecycleHook>) -> Self {
        self.session_hook = Some(hook);
        self
    }

    /// Attach a TTS processor for synthesizing agent responses to audio.
    ///
    /// When set, the agent's text response is also synthesised to an audio file.
    /// Channels that support `MEDIA_UPLOAD` will receive an audio attachment URL.
    #[cfg(feature = "voice")]
    pub fn with_tts(mut self, tts: Arc<crate::tts::TtsProcessor>) -> Self {
        self.tts = Some(tts);
        self
    }

    /// Attach a media processor for handling message attachments.
    ///
    /// When set, attachments on inbound messages are downloaded, validated,
    /// and converted to text descriptions that are appended to the user's
    /// message before it is sent to the agent.
    pub fn with_media(mut self, processor: Arc<MediaProcessor>) -> Self {
        self.media = Some(processor);
        self
    }

    /// Attach a session persistence backend.
    ///
    /// When set, conversation history is loaded from the store when a new agent
    /// session is created and saved after each message is processed.
    pub fn with_persistence(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.persistence = Some(store);
        self
    }

    /// Attach a message sanitizer for inbound spoofing detection and outbound redaction.
    pub fn with_sanitizer(mut self, sanitizer: Arc<MessageSanitizer>) -> Self {
        self.sanitizer = Some(sanitizer);
        self
    }

    /// Attach a per-user rate limiter.
    pub fn with_rate_limiter(mut self, rate_limiter: Arc<RateLimiter>) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }

    /// Attach a text preprocessor that runs before every message is sent to the agent.
    ///
    /// The preprocessor receives the inbound text and may return a replacement string.
    /// If it returns `None` the original text is used unchanged.  This is the
    /// extension point for skill dispatch: the BrainClaw daemon installs a closure
    /// here that detects `/command` syntax and injects skill instructions.
    pub fn with_text_preprocessor(mut self, pp: Arc<TextPreprocessor>) -> Self {
        self.text_preprocessor = Some(pp);
        self
    }

    /// Set the maximum number of tool-call rounds per message.
    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.max_tool_rounds = rounds;
        self
    }

    /// Attach a cross-channel identity store.
    ///
    /// When set, `(platform, user_id)` pairs are resolved to a canonical
    /// UUID before looking up agent sessions.  Identities linked via the
    /// admin API or `/link` skill share one `ChatAgent` and full history.
    pub fn with_identity_store(mut self, store: Arc<UserIdentityStore>) -> Self {
        self.identity_store = Some(store);
        self
    }

    /// Attach a shared metrics collector for token usage tracking.
    ///
    /// When set, `record_token_usage()` is called after every agent turn so
    /// token counts accumulate in the shared `AppState` metrics.
    pub fn with_metrics(mut self, metrics: Arc<MetricsCollector>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Dispatch a synthetic message directly to the agent handler.
    ///
    /// Used by the cron runner to inject scheduled prompts without going through
    /// a real WebSocket channel. Bypasses rate limiting and sanitization.
    ///
    /// `channel_id` should be the UUID of a connected channel adapter that the
    /// response will be sent back through. Pass `Uuid::nil()` if there is no
    /// active channel (the response will be silently dropped with a warning).
    pub async fn dispatch_message(&self, channel_id: Uuid, msg: ChannelMessage) -> Result<()> {
        self.handle_message(channel_id, &msg).await
    }

    /// Return the number of active agent sessions.
    pub fn session_count(&self) -> usize {
        self.agent_sessions.len()
    }

    /// Remove the agent session for a specific platform user.
    ///
    /// Note: if identity linking is active this removes by the raw platform
    /// key, which may not find the session if it's stored under a canonical
    /// UUID. Use `remove_session_by_identity()` in that case.
    pub fn remove_session(&self, platform: &str, user_id: &str) {
        // Try both the raw key and the canonical-identity key format.
        let raw_key = (platform.to_string(), user_id.to_string());
        self.agent_sessions.remove(&raw_key);
    }

    /// Remove idle agent sessions whose last message is older than `timeout`.
    ///
    /// This inspects each agent's message history and removes sessions that
    /// have no messages or whose conversation has been idle.
    pub fn cleanup_idle(&self, _timeout: Duration) {
        // Currently ChatAgent does not expose timestamps on individual messages,
        // so we rely on the gateway SessionManager's `cleanup_expired` for
        // time-based cleanup. Here we remove agent sessions that have an empty
        // history (i.e., have been cleared or never used).
        self.agent_sessions.retain(|_key, agent| {
            // Try-lock to avoid blocking; keep the session if we can't
            // inspect it right now.
            match agent.try_lock() {
                Ok(guard) => !guard.messages().is_empty(),
                Err(_) => true, // busy — keep it
            }
        });
    }

    /// Handle an inbound message by routing it through the appropriate agent.
    async fn handle_message(&self, channel_id: Uuid, msg: &ChannelMessage) -> Result<()> {
        // 0a. Pairing gate — if a pairing handler is configured, ask it
        //     whether this peer is allowed through. This must run before
        //     approval checks, slash commands, and the agent call so
        //     unapproved peers never reach any of those paths.
        if let Some(ref pairing) = self.pairing {
            let channel = &msg.conversation.platform;
            let user_id = &msg.author;
            let peer_display = user_id;
            let incoming = Self::extract_text(&msg.content);
            match pairing
                .check(channel, user_id, peer_display, &incoming)
                .await?
            {
                PairingOutcome::Allow => {}
                PairingOutcome::Reject(reply) => {
                    tracing::info!(
                        channel = %channel,
                        user_id = %user_id,
                        "pairing: rejected peer"
                    );
                    self.send_response(channel_id, msg, &reply).await?;
                    return Ok(());
                }
                PairingOutcome::PendingCodeIssued { code, reply } => {
                    tracing::info!(
                        channel = %channel,
                        user_id = %user_id,
                        %code,
                        "pairing: issued code to unknown peer"
                    );
                    self.send_response(channel_id, msg, &reply).await?;
                    return Ok(());
                }
            }
        }

        // 0. If tool approval is enabled, check whether this message is a
        //    yes/no response to a pending approval request.
        if let Some(ref registry) = self.approval_registry {
            let platform = &msg.conversation.platform;
            let user_id = &msg.author;
            if registry.is_pending(platform, user_id) {
                let text_lower = Self::extract_text(&msg.content).to_lowercase();
                let approved = text_lower.starts_with("yes")
                    || text_lower == "y"
                    || text_lower == "ok"
                    || text_lower == "allow";
                let rejected = text_lower.starts_with("no")
                    || text_lower == "n"
                    || text_lower == "cancel"
                    || text_lower == "deny";
                if approved || rejected {
                    registry.resolve(platform, user_id, approved);
                    return Ok(());
                }
                // Unrecognised text while approval is pending — send a reminder.
                let reminder = "Please reply **yes** or **no** to the pending tool approval.";
                self.send_response(channel_id, msg, reminder).await?;
                return Ok(());
            }
        }

        // 1. Extract text from the message content
        let mut text = Self::extract_text(&msg.content);

        // 1b. Process attachments if a media processor is configured
        if let Some(ref media) = self.media
            && !msg.attachments.is_empty()
        {
            let descriptions = media.process_attachments(&msg.attachments).await;
            if !descriptions.is_empty() {
                let attachment_text = descriptions.join("\n");
                if text.is_empty() {
                    text = attachment_text;
                } else {
                    text = format!("{}\n\n{}", text, attachment_text);
                }
            }
        }

        if text.is_empty() {
            return Ok(());
        }

        // 1c. Handle /talk on|off commands before the text preprocessor.
        //     These toggle "Talk Mode" for the session: when active all agent
        //     responses are synthesised to audio via TTS (voice feature).
        let talk_cmd = Self::parse_talk_command(&text);
        if let Some(enable) = talk_cmd {
            let msg_platform = msg.conversation.platform.clone();
            let msg_author = msg.author.clone();
            let key = (msg_platform.clone(), msg_author.clone());
            let reply = if enable {
                self.talk_mode_sessions.insert(key);
                tracing::info!(platform = %msg_platform, user_id = %msg_author, "Talk Mode enabled");
                "Talk Mode enabled — I'll reply with voice from now on. Say `/talk off` to stop."
            } else {
                self.talk_mode_sessions.remove(&key);
                tracing::info!(platform = %msg_platform, user_id = %msg_author, "Talk Mode disabled");
                "Talk Mode disabled — switching back to text replies."
            };
            self.send_response(channel_id, msg, reply).await?;
            return Ok(());
        }

        // 1d. Handle `/model [name]` command — per-session model override.
        if let Some(model_cmd) = Self::parse_model_command(&text) {
            let msg_platform = msg.conversation.platform.clone();
            let msg_author = msg.author.clone();
            let key = (msg_platform.clone(), msg_author.clone());
            let reply = match model_cmd {
                ModelCommand::Show => {
                    let current = self
                        .model_overrides
                        .get(&key)
                        .map(|m| m.clone())
                        .unwrap_or_else(|| "(provider default)".to_string());
                    format!(
                        "Current model: **{}**\n\nTo switch: `/model claude-opus-4-6`\n\
                         To reset to default: `/model default`",
                        current
                    )
                }
                ModelCommand::Reset => {
                    self.model_overrides.remove(&key);
                    // Drop the existing agent so it is recreated with no override.
                    let session_key = self.resolve_session_key(&msg_platform, &msg_author).await;
                    self.agent_sessions.remove(&session_key);
                    self.unregister_session(&session_key.0, &session_key.1)
                        .await;
                    "Model reset to provider default. Starting a new session.".to_string()
                }
                ModelCommand::Set(model_name) => {
                    self.model_overrides.insert(key.clone(), model_name.clone());
                    // Drop the existing agent so it is recreated with the override.
                    let session_key = self.resolve_session_key(&msg_platform, &msg_author).await;
                    self.agent_sessions.remove(&session_key);
                    self.unregister_session(&session_key.0, &session_key.1)
                        .await;
                    tracing::info!(
                        platform = %msg_platform,
                        user_id = %msg_author,
                        model = %model_name,
                        "Per-session model override set"
                    );
                    format!(
                        "Switched to **{}** for this session. \
                         Previous conversation history has been cleared. \
                         Use `/model default` to revert.",
                        model_name
                    )
                }
            };
            self.send_response(channel_id, msg, &reply).await?;
            return Ok(());
        }

        // 1d2. Slash-command interception (P1.2). Runs after legacy `/talk`
        //       and `/model` so those remain handled by their dedicated paths.
        match crate::slash::parse(&text) {
            crate::slash::ParseResult::Forward(t) => {
                text = t;
            }
            crate::slash::ParseResult::Command(cmd) => {
                let reply = self
                    .handle_slash(&msg.conversation.platform, &msg.author, cmd)
                    .await;
                self.send_response(channel_id, msg, &reply).await?;
                return Ok(());
            }
        }

        // 1e. Apply text preprocessor (e.g. skill dispatch)
        if let Some(ref pp) = self.text_preprocessor
            && let Some(transformed) = pp(&text)
        {
            text = transformed;
        }

        // 1e. Sanitize inbound: detect and strip system-message spoofing
        if let Some(ref sanitizer) = self.sanitizer
            && sanitizer.strip_system_spoofing
            && MessageSanitizer::is_system_spoofing(&text)
        {
            tracing::warn!(
                author = %msg.author,
                conversation = %msg.conversation.channel_id,
                "System-message spoofing detected; rejecting message"
            );
            return Ok(());
        }

        // 2. Build a ChannelUser from the message and touch the gateway session
        let platform = msg.conversation.platform.clone();
        let user_id = msg.author.clone();

        // 2a. Rate-limit check
        if let Some(ref rate_limiter) = self.rate_limiter {
            if !rate_limiter.check_message_rate(&platform, &user_id) {
                tracing::warn!(
                    platform = %platform,
                    user_id = %user_id,
                    "Rate limit exceeded; dropping message"
                );
                return Ok(());
            }
            rate_limiter.record_message(&platform, &user_id);
        }

        let user = brainwires_network::channels::ChannelUser {
            platform: platform.clone(),
            platform_user_id: user_id.clone(),
            display_name: user_id.clone(),
            username: None,
            avatar_url: None,
        };
        let session = self.sessions.get_or_create_session(&user);

        tracing::info!(
            channel_id = %channel_id,
            session_id = %session.id,
            platform = %platform,
            author = %user_id,
            "Processing inbound message via agent",
        );

        // 3. Resolve canonical identity (cross-channel linking) and get agent.
        let agent = self.get_or_create_agent(&platform, &user_id).await;

        // 3b. Update the approval context so the hook knows the current
        //     channel and conversation for this turn.
        if self.approval_registry.is_some() {
            let key = (platform.clone(), user_id.clone());
            let ctx = self.approval_contexts.entry(key).or_insert_with(|| {
                (
                    Arc::new(RwLock::new(None)),
                    Arc::new(RwLock::new(None)),
                    Arc::new(RwLock::new(None)),
                )
            });
            *ctx.0.write().await = Some(msg.conversation.clone());
            *ctx.2.write().await = Some(channel_id);
            if let Some(tx) = self.channels.get_sender(&channel_id) {
                *ctx.1.write().await = Some(tx);
            }
        }

        // 4. Lock agent and process the message
        if let Some(ref hook) = self.session_hook {
            hook.on_event(&LifecycleEvent::AgentStarted {
                agent_id: format!("{}:{}", platform, user_id),
                task_description: text.chars().take(120).collect(),
            })
            .await;
        }

        // Touch the session in the shared registry (if wired) before and
        // after the agent turn so `sessions_list` surfaces accurate
        // `last_active` timestamps.
        if let Some(ref registry) = self.session_registry {
            let sid = Self::session_id_for(&platform, &user_id);
            registry.touch(&sid).await;
        }

        let mut agent = agent.lock().await;
        let usage_before = agent.cumulative_usage().clone();
        let process_result = agent.process_message(&text).await;

        // Record per-message token delta to the shared metrics collector.
        if let Some(ref metrics) = self.metrics {
            let usage_after = agent.cumulative_usage();
            let prompt_delta = usage_after
                .prompt_tokens
                .saturating_sub(usage_before.prompt_tokens);
            let completion_delta = usage_after
                .completion_tokens
                .saturating_sub(usage_before.completion_tokens);
            if prompt_delta > 0 || completion_delta > 0 {
                metrics.record_token_usage(prompt_delta as u64, completion_delta as u64);
            }
        }

        // Fire session end event regardless of success/failure
        if let Some(ref hook) = self.session_hook {
            match &process_result {
                Ok(resp) => {
                    hook.on_event(&LifecycleEvent::AgentCompleted {
                        agent_id: format!("{}:{}", platform, user_id),
                        iterations: 1,
                        summary: resp.chars().take(120).collect(),
                    })
                    .await;
                }
                Err(e) => {
                    hook.on_event(&LifecycleEvent::AgentFailed {
                        agent_id: format!("{}:{}", platform, user_id),
                        error: e.to_string(),
                        iterations: 1,
                    })
                    .await;
                }
            }
        }

        let raw_response = process_result?;

        // Decrement trace counter after a successful agent turn so `/trace on`
        // auto-disables after N turns.
        if let Some(state_arc) = self.slash_state.get(&(platform.clone(), user_id.clone())) {
            let mut s = state_arc.lock().await;
            if s.trace_remaining > 0 {
                s.trace_remaining -= 1;
            }
        }

        // 4b. Sanitize outbound response (redact secrets)
        let response = match &self.sanitizer {
            Some(sanitizer) => sanitizer.sanitize_outbound(&raw_response),
            None => raw_response,
        };

        // 5. Persist updated conversation history (use canonical identity key if available).
        if let Some(ref store) = self.persistence {
            let (sess_platform, sess_user) = if self.identity_store.is_some() {
                let id = {
                    let store = self.identity_store.as_ref().unwrap();
                    store.get_identity_id(&platform, &user_id).await
                };
                ("__identity__".to_string(), id.to_string())
            } else {
                (platform.clone(), user_id.clone())
            };
            let key = session_key(&sess_platform, &sess_user);
            if let Err(e) = store.save(&key, agent.messages()).await {
                tracing::warn!(error = %e, "Failed to persist session");
            }
        }

        // 6. Send the response back to the channel
        self.send_response(channel_id, msg, &response).await?;

        Ok(())
    }

    /// Resolve the canonical session key for a platform user.
    ///
    /// If an identity store is configured and the user is linked to a
    /// canonical UUID, returns `("__identity__", uuid_string)`.
    /// Otherwise returns `(platform, user_id)`.
    async fn resolve_session_key(&self, platform: &str, user_id: &str) -> (String, String) {
        if let Some(ref store) = self.identity_store {
            let id = store.get_identity_id(platform, user_id).await;
            ("__identity__".to_string(), id.to_string())
        } else {
            (platform.to_string(), user_id.to_string())
        }
    }

    /// Get (or lazily create) a [`ChatAgent`] for the given platform user.
    ///
    /// When a persistence backend is configured, newly created agents will have
    /// their conversation history restored from the store.
    async fn get_or_create_agent(&self, platform: &str, user_id: &str) -> Arc<Mutex<ChatAgent>> {
        let key = self.resolve_session_key(platform, user_id).await;

        // Fast path: agent already exists.
        if let Some(existing) = self.agent_sessions.get(&key) {
            return existing.clone();
        }

        // Slow path: create a new agent, optionally restoring persisted history.
        // Apply any per-session model override stored via `/model <name>`.
        let mut session_options = self.default_options.clone();
        let raw_key = (platform.to_string(), user_id.to_string());
        if let Some(model_override) = self.model_overrides.get(&raw_key) {
            session_options.model = Some(model_override.clone());
        }
        // If a session broker is configured, wrap the shared executor with a
        // `SessionsExecutor` bound to *this* session's id so the agent can
        // introspect/orchestrate sibling sessions. Otherwise fall through to
        // the raw executor unchanged.
        let executor_for_agent: Arc<dyn ToolExecutor> =
            if let Some(ref broker) = self.session_broker {
                let session_id_str = format!("{}:{}", key.0, key.1);
                Arc::new(SessionsExecutor::new(
                    self.executor.clone(),
                    broker.clone(),
                    Some(SessionId::new(session_id_str)),
                ))
            } else {
                self.executor.clone()
            };

        let mut agent = ChatAgent::new(self.provider.clone(), executor_for_agent, session_options)
            .with_max_tool_rounds(self.max_tool_rounds);

        // Build pre-tool hook(s). Shell hook runs first; approval hook second.
        let mut pre_hooks: Vec<Arc<dyn ToolPreHook>> = Vec::new();

        if let Some(ref shell_hook) = self.shell_pre_tool_hook {
            pre_hooks.push(Arc::clone(shell_hook));
        }

        if let Some(ref registry) = self.approval_registry {
            let ctx_key = (platform.to_string(), user_id.to_string());
            let (conv, sender, chan_id) = self
                .approval_contexts
                .entry(ctx_key)
                .or_insert_with(|| {
                    (
                        Arc::new(RwLock::new(None)),
                        Arc::new(RwLock::new(None)),
                        Arc::new(RwLock::new(None)),
                    )
                })
                .clone();

            let hook = ChatApprovalHook::new(
                platform.to_string(),
                user_id.to_string(),
                self.approval_tools.clone(),
                conv,
                sender,
                chan_id,
                Arc::clone(registry),
            );
            pre_hooks.push(Arc::new(hook));
        }

        if !pre_hooks.is_empty() {
            let composite = if pre_hooks.len() == 1 {
                pre_hooks.into_iter().next().unwrap()
            } else {
                Arc::new(CompositePreToolHook { hooks: pre_hooks })
            };
            agent = agent.with_pre_execute_hook(composite);
        }

        if let Some(ref store) = self.persistence {
            let skey = session_key(platform, user_id);
            match store.load(&skey).await {
                Ok(Some(messages)) => {
                    tracing::info!(
                        session_key = %skey,
                        message_count = messages.len(),
                        "Restored persisted conversation history",
                    );
                    agent.restore_messages(messages);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load persisted session");
                }
            }
        }

        let arc = Arc::new(Mutex::new(agent));
        self.agent_sessions
            .entry(key.clone())
            .or_insert(arc.clone());

        // Register with the shared session registry (if wired) so the four
        // `sessions_*` tools see this session in `sessions_list`,
        // `sessions_history`, and `sessions_send`. The session id format must
        // match what `get_or_create_agent`'s caller used when wrapping the
        // executor with `SessionsExecutor` (i.e. `"platform:user_id"`).
        if let Some(ref registry) = self.session_registry {
            let session_id = SessionId::new(format!("{}:{}", key.0, key.1));
            let channel = platform.to_string();
            let peer = user_id.to_string();
            let registry_clone = Arc::clone(registry);
            let agent_for_handle = Arc::clone(&arc);
            let sid_clone = session_id.clone();
            // `register()` returns a receiver that must be drained or
            // `sessions_send` will pile up messages. We spawn a simple
            // drainer that, for each inbound text, calls `process_message`
            // on the agent and discards the reply (fire-and-forget per the
            // sessions_send contract). Any errors are logged.
            let drainer_agent = Arc::clone(&arc);
            let drainer_sid = sid_clone.clone();
            let drainer_registry = Arc::clone(registry);
            tokio::spawn(async move {
                let mut rx = registry_clone
                    .register(
                        sid_clone.clone(),
                        channel,
                        peer,
                        None,
                        Some(agent_for_handle),
                    )
                    .await;
                while let Some(text) = rx.recv().await {
                    drainer_registry.touch(&drainer_sid).await;
                    let mut guard = drainer_agent.lock().await;
                    if let Err(e) = guard.process_message(&text).await {
                        tracing::warn!(
                            session_id = %drainer_sid,
                            error = %e,
                            "session inbound drainer: process_message failed",
                        );
                    }
                }
            });
        }
        arc
    }

    /// Session id string used when this handler registers a user session in
    /// the shared [`SessionRegistry`]. Format: `"platform:user_id"` so the
    /// id matches the one baked into each user's [`SessionsExecutor`].
    fn session_id_for(platform: &str, user_id: &str) -> SessionId {
        SessionId::new(format!("{platform}:{user_id}"))
    }

    /// Unregister a session from the shared registry, if wired. Called from
    /// the `/restart` and `/model` paths (which tear down the cached agent),
    /// and from [`remove_session_by_key`]. The session will be re-registered
    /// lazily on the next inbound message.
    async fn unregister_session(&self, platform: &str, user_id: &str) {
        if let Some(ref registry) = self.session_registry {
            let session_id = Self::session_id_for(platform, user_id);
            registry.unregister(&session_id).await;
        }
    }

    /// Handle a parsed slash command for a specific user and return the reply
    /// text to send back. Lazily materialises the per-user slash state and the
    /// `SessionController` bridge over the current `ChatAgent`.
    async fn handle_slash(
        &self,
        platform: &str,
        user_id: &str,
        cmd: crate::slash::SlashCommand,
    ) -> String {
        use crate::slash::{AgentSessionHandle, SessionSlashState, SlashOutcome};

        let key = (platform.to_string(), user_id.to_string());
        let state_arc = self
            .slash_state
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(SessionSlashState::default())))
            .clone();

        // Resolve (or lazily create) the agent so `/status`, `/usage`, etc.
        // report against the same session the user's messages hit.
        let agent = self.get_or_create_agent(platform, user_id).await;

        let session_key = self.resolve_session_key(platform, user_id).await;
        let model = self
            .model_overrides
            .get(&key)
            .map(|m| m.clone())
            .or_else(|| self.default_options.model.clone())
            .unwrap_or_else(|| "(provider default)".to_string());
        let channels: Vec<String> = self
            .channels
            .list()
            .into_iter()
            .map(|c| c.channel_type)
            .collect();
        let provider_name = self.provider.name().to_string();
        let thinking_supported = provider_supports_thinking(&provider_name);

        // Rebuild callback for `/restart`: reconstructs a fresh ChatAgent with
        // the same provider/executor/options. Session-specific model override
        // is re-applied. Pre-tool hooks are not rebuilt here — `/restart`
        // discards approval/shell hook state by design.
        let provider = self.provider.clone();
        let base_executor = self.executor.clone();
        let max_tool_rounds = self.max_tool_rounds;
        let mut session_options = self.default_options.clone();
        if let Some(model_override) = self.model_overrides.get(&key) {
            session_options.model = Some(model_override.clone());
        }
        // Preserve session-tool wiring across /restart.
        let broker_for_rebuild = self.session_broker.clone();
        let session_id_for_rebuild = format!("{}:{}", key.0, key.1);
        let rebuild: Arc<dyn Fn() -> ChatAgent + Send + Sync> = Arc::new(move || {
            let exec: Arc<dyn ToolExecutor> = if let Some(ref broker) = broker_for_rebuild {
                Arc::new(SessionsExecutor::new(
                    base_executor.clone(),
                    broker.clone(),
                    Some(SessionId::new(session_id_for_rebuild.clone())),
                ))
            } else {
                base_executor.clone()
            };
            ChatAgent::new(provider.clone(), exec, session_options.clone())
                .with_max_tool_rounds(max_tool_rounds)
        });

        let mut controller = AgentSessionHandle {
            agent,
            provider_name,
            model,
            session_id: format!("{}:{}", session_key.0, session_key.1),
            channels_connected: channels,
            thinking_supported,
            rebuild: Some(rebuild),
        };

        let mut state = state_arc.lock().await;
        match crate::slash::handle(cmd, &mut state, &mut controller).await {
            SlashOutcome::Reply(r) => r,
            SlashOutcome::Forward(t) => t,
        }
    }

    /// Parse `/model [name|default]` commands.
    ///
    /// Returns `None` if the text is not a model command.
    fn parse_model_command(text: &str) -> Option<ModelCommand> {
        let t = text.trim();
        if !t.starts_with("/model") {
            return None;
        }
        let rest = t["/model".len()..].trim();
        if rest.is_empty() || rest == "list" {
            Some(ModelCommand::Show)
        } else if rest == "default" || rest == "reset" {
            Some(ModelCommand::Reset)
        } else {
            Some(ModelCommand::Set(rest.to_string()))
        }
    }

    /// Parse `/talk [on|off|start|stop]` commands.
    ///
    /// Returns `Some(true)` to enable Talk Mode, `Some(false)` to disable,
    /// or `None` if the text is not a talk command.
    fn parse_talk_command(text: &str) -> Option<bool> {
        let t = text.trim().to_lowercase();
        if t == "/talk" || t == "/talk on" || t == "/talk start" {
            return Some(true);
        }
        if t == "/talk off" || t == "/talk stop" || t == "/talk end" {
            return Some(false);
        }
        None
    }

    /// Extract a text string from [`MessageContent`], returning an empty string
    /// for non-text variants (media, embeds).
    fn extract_text(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::RichText { markdown, .. } => markdown.clone(),
            MessageContent::Mixed(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    MessageContent::Text(t) => Some(t.as_str()),
                    MessageContent::RichText { markdown, .. } => Some(markdown.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    /// Send a response back to the channel that originated the message.
    ///
    /// Builds a `ChannelEvent::MessageReceived` with the assistant's reply,
    /// serializes it to JSON, and pushes it through the channel's sender.
    async fn send_response(
        &self,
        channel_id: Uuid,
        original_msg: &ChannelMessage,
        response_text: &str,
    ) -> Result<()> {
        // Synthesize audio when:
        //   (a) TTS is configured AND (b) the original input was a voice/audio
        //       message OR the session has Talk Mode enabled.
        #[cfg(feature = "voice")]
        let attachments: Vec<brainwires_network::channels::message::Attachment> = {
            let is_audio_input = matches!(
                &original_msg.content,
                MessageContent::Media(p) if matches!(p.media_type, brainwires_network::channels::message::MediaType::Audio)
            );
            let talk_mode_active = self.talk_mode_sessions.contains(&(
                original_msg.conversation.platform.clone(),
                original_msg.author.clone(),
            ));

            if let Some(ref tts) = self.tts {
                if is_audio_input || talk_mode_active {
                    if let Some(audio_url) = tts.synthesize_to_url(response_text).await {
                        vec![brainwires_network::channels::message::Attachment {
                            url: audio_url,
                            content_type: "audio/mpeg".to_string(),
                            filename: String::new(),
                            size_bytes: None,
                        }]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        };

        #[cfg(not(feature = "voice"))]
        let attachments: Vec<brainwires_network::channels::message::Attachment> = vec![];

        let response_event = ChannelEvent::MessageReceived(ChannelMessage {
            id: MessageId::new(Uuid::new_v4().to_string()),
            conversation: ConversationId {
                platform: original_msg.conversation.platform.clone(),
                channel_id: original_msg.conversation.channel_id.clone(),
                server_id: original_msg.conversation.server_id.clone(),
            },
            author: "assistant".to_string(),
            content: MessageContent::Text(response_text.to_string()),
            thread_id: original_msg.thread_id.clone(),
            reply_to: Some(original_msg.id.clone()),
            timestamp: chrono::Utc::now(),
            attachments,
            metadata: std::collections::HashMap::new(),
        });

        let json = serde_json::to_string(&response_event)?;

        if let Some(tx) = self.channels.get_sender(&channel_id) {
            tx.send(json).await.map_err(|e| {
                anyhow::anyhow!("Failed to send response to channel {channel_id}: {e}")
            })?;
            tracing::info!(
                channel_id = %channel_id,
                "Agent response sent to channel",
            );
        } else {
            tracing::warn!(
                channel_id = %channel_id,
                "No sender found for channel; response dropped",
            );
        }

        Ok(())
    }
}

#[async_trait]
impl InboundHandler for AgentInboundHandler {
    async fn handle_inbound(&self, channel_id: Uuid, event: &ChannelEvent) -> Result<()> {
        // Update heartbeat for the channel
        self.channels.touch_heartbeat(&channel_id);

        match event {
            ChannelEvent::MessageReceived(msg) => self.handle_message(channel_id, msg).await,
            _ => {
                tracing::debug!("Received non-message event: {:?}", event);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::{
        ChatOptions, ChatResponse, Message, StreamChunk, Tool, ToolContext, Usage,
    };
    use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};
    use brainwires_tools::{BuiltinToolExecutor, ToolRegistry};
    use futures::stream;
    use std::collections::HashMap;

    /// A mock provider that returns a fixed text response.
    struct MockProvider {
        response_text: String,
    }

    impl MockProvider {
        fn new(text: &str) -> Self {
            Self {
                response_text: text.to_string(),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn chat(
            &self,
            _messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(&self.response_text),
                usage: Usage::new(10, 20),
                finish_reason: Some("stop".to_string()),
            })
        }

        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> futures::stream::BoxStream<'a, Result<StreamChunk>> {
            let text = self.response_text.clone();
            Box::pin(stream::iter(vec![
                Ok(StreamChunk::Text(text)),
                Ok(StreamChunk::Done),
            ]))
        }
    }

    fn make_executor() -> Arc<dyn ToolExecutor> {
        let registry = ToolRegistry::new();
        let context = ToolContext::default();
        Arc::new(BuiltinToolExecutor::new(registry, context))
    }

    fn make_handler() -> AgentInboundHandler {
        let sessions = Arc::new(SessionManager::new());
        let channels = Arc::new(ChannelRegistry::new());
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::new("Hello from agent!"));
        let executor = make_executor();
        let options = ChatOptions::default();

        AgentInboundHandler::new(sessions, channels, provider, executor, options)
    }

    fn make_message(platform: &str, author: &str, text: &str) -> ChannelMessage {
        ChannelMessage {
            id: MessageId::new(Uuid::new_v4().to_string()),
            conversation: ConversationId {
                platform: platform.to_string(),
                channel_id: "general".to_string(),
                server_id: None,
            },
            author: author.to_string(),
            content: MessageContent::Text(text.to_string()),
            thread_id: None,
            reply_to: None,
            timestamp: chrono::Utc::now(),
            attachments: vec![],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_new_creates_successfully() {
        let handler = make_handler();
        assert_eq!(handler.session_count(), 0);
        assert_eq!(handler.max_tool_rounds, 10);
    }

    #[test]
    fn test_with_max_tool_rounds() {
        let handler = make_handler().with_max_tool_rounds(5);
        assert_eq!(handler.max_tool_rounds, 5);
    }

    #[test]
    fn test_extract_text_plain() {
        let content = MessageContent::Text("hello world".to_string());
        assert_eq!(AgentInboundHandler::extract_text(&content), "hello world");
    }

    #[test]
    fn test_extract_text_rich() {
        let content = MessageContent::RichText {
            markdown: "**bold text**".to_string(),
            fallback_plain: "bold text".to_string(),
        };
        assert_eq!(AgentInboundHandler::extract_text(&content), "**bold text**");
    }

    #[test]
    fn test_extract_text_mixed() {
        let content = MessageContent::Mixed(vec![
            MessageContent::Text("first line".to_string()),
            MessageContent::RichText {
                markdown: "second line".to_string(),
                fallback_plain: "second".to_string(),
            },
            MessageContent::Media(brainwires_network::channels::message::MediaPayload {
                media_type: brainwires_network::channels::message::MediaType::Image,
                url: "https://example.com/img.png".to_string(),
                caption: None,
                thumbnail_url: None,
            }),
        ]);
        assert_eq!(
            AgentInboundHandler::extract_text(&content),
            "first line\nsecond line"
        );
    }

    #[test]
    fn test_extract_text_media_returns_empty() {
        let content = MessageContent::Media(brainwires_network::channels::message::MediaPayload {
            media_type: brainwires_network::channels::message::MediaType::Image,
            url: "https://example.com/img.png".to_string(),
            caption: None,
            thumbnail_url: None,
        });
        assert_eq!(AgentInboundHandler::extract_text(&content), "");
    }

    #[test]
    fn test_extract_text_embed_returns_empty() {
        let content = MessageContent::Embed(brainwires_network::channels::message::EmbedPayload {
            title: Some("Title".to_string()),
            description: Some("Desc".to_string()),
            url: None,
            color: None,
            fields: vec![],
            thumbnail: None,
            footer: None,
        });
        assert_eq!(AgentInboundHandler::extract_text(&content), "");
    }

    #[tokio::test]
    async fn test_get_or_create_agent_returns_same_for_same_user() {
        let handler = make_handler();
        let agent1 = handler.get_or_create_agent("discord", "user-1").await;
        let agent2 = handler.get_or_create_agent("discord", "user-1").await;
        assert!(Arc::ptr_eq(&agent1, &agent2));
        assert_eq!(handler.session_count(), 1);
    }

    #[tokio::test]
    async fn test_get_or_create_agent_returns_different_for_different_users() {
        let handler = make_handler();
        let agent1 = handler.get_or_create_agent("discord", "user-1").await;
        let agent2 = handler.get_or_create_agent("discord", "user-2").await;
        assert!(!Arc::ptr_eq(&agent1, &agent2));
        assert_eq!(handler.session_count(), 2);
    }

    #[tokio::test]
    async fn test_get_or_create_agent_different_platforms() {
        let handler = make_handler();
        let agent1 = handler.get_or_create_agent("discord", "user-1").await;
        let agent2 = handler.get_or_create_agent("telegram", "user-1").await;
        assert!(!Arc::ptr_eq(&agent1, &agent2));
        assert_eq!(handler.session_count(), 2);
    }

    #[tokio::test]
    async fn test_session_count_tracks_correctly() {
        let handler = make_handler();
        assert_eq!(handler.session_count(), 0);

        handler.get_or_create_agent("discord", "user-1").await;
        assert_eq!(handler.session_count(), 1);

        handler.get_or_create_agent("telegram", "user-2").await;
        assert_eq!(handler.session_count(), 2);

        // Same user again — no new session
        handler.get_or_create_agent("discord", "user-1").await;
        assert_eq!(handler.session_count(), 2);
    }

    #[tokio::test]
    async fn test_remove_session_works() {
        let handler = make_handler();
        handler.get_or_create_agent("discord", "user-1").await;
        handler.get_or_create_agent("telegram", "user-2").await;
        assert_eq!(handler.session_count(), 2);

        handler.remove_session("discord", "user-1");
        assert_eq!(handler.session_count(), 1);

        // Removing non-existent session is a no-op
        handler.remove_session("slack", "user-99");
        assert_eq!(handler.session_count(), 1);
    }

    #[tokio::test]
    async fn test_handle_inbound_non_message_event() {
        let handler = make_handler();
        let event = ChannelEvent::TypingStarted {
            conversation: ConversationId {
                platform: "discord".to_string(),
                channel_id: "general".to_string(),
                server_id: None,
            },
            user: brainwires_network::channels::ChannelUser {
                platform: "discord".to_string(),
                platform_user_id: "user-1".to_string(),
                display_name: "User 1".to_string(),
                username: None,
                avatar_url: None,
            },
        };

        let result = handler.handle_inbound(Uuid::new_v4(), &event).await;
        assert!(result.is_ok());
        // No agent session should have been created
        assert_eq!(handler.session_count(), 0);
    }

    #[tokio::test]
    async fn test_handle_message_creates_agent_session() {
        let handler = make_handler();
        let msg = make_message("discord", "user-1", "Hello agent!");
        let channel_id = Uuid::new_v4();

        // We don't register a channel so the response will be dropped (no sender),
        // but the agent session should still be created and the message processed.
        let result = handler.handle_message(channel_id, &msg).await;
        assert!(result.is_ok());
        assert_eq!(handler.session_count(), 1);
    }

    #[tokio::test]
    async fn test_handle_message_empty_text_is_noop() {
        let handler = make_handler();
        let mut msg = make_message("discord", "user-1", "");
        msg.content = MessageContent::Media(brainwires_network::channels::message::MediaPayload {
            media_type: brainwires_network::channels::message::MediaType::Image,
            url: "https://example.com/img.png".to_string(),
            caption: None,
            thumbnail_url: None,
        });

        let result = handler.handle_message(Uuid::new_v4(), &msg).await;
        assert!(result.is_ok());
        // No agent session should have been created for a media-only message
        assert_eq!(handler.session_count(), 0);
    }
}
