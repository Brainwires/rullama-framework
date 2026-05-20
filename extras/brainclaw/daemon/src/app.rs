//! BrainClaw application — wires everything together and runs the daemon.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};

use brainwires_core::{ChatOptions, ToolContext};
use brainwires_gateway::agent_handler::AgentInboundHandler;
use brainwires_gateway::channel_registry::ChannelRegistry;
use brainwires_gateway::identity::UserIdentityStore;
use brainwires_gateway::media::MediaProcessor;
use brainwires_gateway::metrics::MetricsCollector;
use brainwires_gateway::middleware::rate_limit::RateLimiter;
use brainwires_gateway::middleware::sanitizer::MessageSanitizer;
use brainwires_gateway::pairing::{PairingHandler, PairingStore, default_policy};
use brainwires_gateway::server::Gateway;
use brainwires_gateway::session::SessionManager;
use brainwires_gateway::session_persistence::{JsonFileStore, expand_tilde};
use brainwires_gateway::sessions_broker::{GatewaySessionBroker, SessionRegistry};
use brainwires_providers::{ChatProviderFactory, ProviderConfig, ProviderType};
use brainwires_tools::{BuiltinToolExecutor, SessionBroker, ToolExecutor};

use brainwires_gateway::cron::CronStore;

use crate::config::BrainClawConfig;
use crate::cron::CronRunner;
use crate::persona::Persona;
use crate::session_spawn::BrainClawSpawnFactory;
use crate::shell_hooks::{ShellHookRunner, ShellPreToolHook};
use crate::skill_handler::SkillHandler;
use crate::tools::build_tool_registry;

/// The BrainClaw daemon.
pub struct BrainClaw {
    config: BrainClawConfig,
}

impl BrainClaw {
    /// Create a new BrainClaw instance with the given configuration.
    pub fn new(config: BrainClawConfig) -> Self {
        Self { config }
    }

    /// Run the BrainClaw daemon.
    ///
    /// This method blocks until the server is shut down (via SIGINT/SIGTERM).
    pub async fn run(&self) -> Result<()> {
        tracing::info!(
            provider = %self.config.provider.default_provider,
            persona = %self.config.persona.name,
            "Starting BrainClaw daemon"
        );

        // 1. Resolve API key
        let api_key = self.resolve_api_key()?;

        // 2. Create provider
        let provider_type: ProviderType = self.config.provider.default_provider.parse()?;
        let model = self
            .config
            .provider
            .default_model
            .clone()
            .unwrap_or_else(|| provider_type.default_model().to_string());

        let mut prov_config = ProviderConfig::new(provider_type, model.clone());
        if let Some(key) = api_key {
            prov_config = prov_config.with_api_key(key);
        } else if provider_type.requires_api_key() {
            bail!(
                "No API key found for provider '{}'.\n\
                 Set it via:\n  \
                 - Config file: provider.api_key or provider.api_key_env\n  \
                 - Environment variable (e.g. ANTHROPIC_API_KEY)\n  \
                 - CLI flag: --api-key",
                self.config.provider.default_provider
            );
        }

        let provider = ChatProviderFactory::create(&prov_config)?;

        tracing::info!(
            provider = %provider_type,
            model = %model,
            "Provider initialized"
        );

        // 3. Build tool registry
        let registry = build_tool_registry(&self.config.tools);
        let tool_count = registry.len();

        // 3b. Build ToolContext with provider configs embedded as metadata
        let mut context = ToolContext::default();
        self.inject_tool_configs(&mut context);

        let builtin = BuiltinToolExecutor::new(registry, context);
        let executor: Arc<dyn ToolExecutor> = self.wrap_with_sandbox(builtin)?;

        tracing::info!(tools = tool_count, "Tool registry built");

        // 4. Build ChatOptions with system prompt from persona + context files
        let persona = Persona::from_config(&self.config.persona)?;
        let context_text = load_context_files(&self.config.persona.context_files);
        let system_prompt = if context_text.is_empty() {
            persona.system_prompt.clone()
        } else {
            format!("{}\n\n---\n\n{}", persona.system_prompt, context_text)
        };

        let options = ChatOptions {
            temperature: Some(self.config.provider.temperature),
            max_tokens: Some(self.config.provider.max_tokens),
            system: Some(system_prompt),
            ..Default::default()
        };

        tracing::info!(persona = %persona.name, "Persona loaded");

        // 5. Build GatewayConfig
        let gateway_config = self.config.to_gateway_config();

        // 6. Create session manager, channel registry, and shared metrics.
        //    These must be shared between the AgentInboundHandler (which uses
        //    them to send responses) and the Gateway AppState (which uses them
        //    to register WebSocket connections).  Without sharing, channel
        //    adapters register into a different ChannelRegistry than the one
        //    the handler queries, so responses would silently drop.
        //    Metrics are shared so that token usage recorded by the handler
        //    appears in the admin /metrics endpoint served by the gateway.
        let sessions = Arc::new(SessionManager::new());
        let channels = Arc::new(ChannelRegistry::new());
        let metrics = Arc::new(MetricsCollector::new());

        // 7. Create AgentInboundHandler
        let openai_provider = Arc::clone(&provider);
        let mut handler = AgentInboundHandler::new(
            Arc::clone(&sessions),
            Arc::clone(&channels),
            Arc::clone(&provider),
            Arc::clone(&executor),
            options.clone(),
        )
        .with_max_tool_rounds(self.config.agent.max_tool_rounds);

        // 7aa. Session-as-tools wiring. Share one `SessionRegistry` between
        //       the handler (which registers each per-user ChatAgent there
        //       so the `sessions_*` tools can see them) and the
        //       `GatewaySessionBroker` (which backs the four tools and
        //       delegates `sessions_spawn` to `BrainClawSpawnFactory`).
        let session_registry = Arc::new(SessionRegistry::new());
        let spawn_factory = Arc::new(BrainClawSpawnFactory::new(
            Arc::clone(&provider),
            Arc::clone(&executor),
            options,
        ));
        let session_broker: Arc<dyn SessionBroker> = Arc::new(GatewaySessionBroker::new(
            (*session_registry).clone(),
            spawn_factory,
        ));
        handler = handler
            .with_session_registry(Arc::clone(&session_registry))
            .with_session_broker(Arc::clone(&session_broker));

        // 7b. Attach session persistence if configured
        if self.config.memory.persist_conversations {
            let storage_path = expand_tilde(&self.config.memory.storage_dir);
            match JsonFileStore::new(&storage_path) {
                Ok(store) => {
                    tracing::info!(
                        path = %storage_path.display(),
                        "Session persistence enabled (JsonFileStore)"
                    );
                    handler = handler.with_persistence(Arc::new(store));
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %storage_path.display(),
                        "Failed to initialize session persistence; continuing without it"
                    );
                }
            }
        }

        // 7c. Attach message sanitizer from security config
        let sanitizer = Arc::new(MessageSanitizer::new(
            self.config.security.strip_system_spoofing,
            self.config.security.redact_secrets_in_output,
        ));
        handler = handler.with_sanitizer(sanitizer);

        // 7d. Attach rate limiter from security config
        let rate_limiter = Arc::new(RateLimiter::new(
            self.config.security.max_messages_per_minute,
            self.config.security.max_tool_calls_per_minute,
        ));
        handler = handler.with_rate_limiter(rate_limiter);

        // 7e. Attach skill handler if enabled
        if self.config.skills.enabled {
            let skill_dirs: Vec<PathBuf> = self
                .config
                .skills
                .directories
                .iter()
                .map(|d| PathBuf::from(expand_tilde_str(d)))
                .collect();

            let skill_handler_result = SkillHandler::new(&skill_dirs).map(|h| {
                if let Some(ref url) = self.config.skills.registry_url {
                    tracing::info!(url = %url, "Skill registry fallback enabled");
                    h.with_registry_url(url.clone())
                } else {
                    h
                }
            });
            match skill_handler_result {
                Ok(skill_handler) => {
                    let count = skill_handler.skill_count();
                    tracing::info!(skills = count, "Skill system enabled");
                    let sh = Arc::new(Mutex::new(skill_handler));
                    handler = handler.with_text_preprocessor(Arc::new(move |text: &str| {
                        if let Some((cmd, args)) = SkillHandler::parse_command(text) {
                            let guard = match sh.lock() {
                                Ok(g) => g,
                                Err(_) => return None,
                            };
                            match guard.resolve_command(cmd, args) {
                                Ok(Some(instructions)) => {
                                    // Prepend skill instructions; agent sees the full context
                                    Some(format!(
                                        "Execute the following skill instructions:\n\n\
                                         {instructions}\n\n\
                                         User input: {text}"
                                    ))
                                }
                                Ok(None) => {
                                    tracing::debug!(command = cmd, "No skill found for command");
                                    None
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, command = cmd, "Skill resolution error");
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    }));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize skill system; continuing without it");
                }
            }
        }

        // 7f. Attach media processor (+ optional STT for voice)
        #[allow(unused_mut)] // mut required when `voice` feature is enabled.
        let mut media = MediaProcessor::new(10); // 10 MB attachment limit

        #[cfg(feature = "voice")]
        if let Some(ref voice_cfg) = self.config.voice
            && let Some(stt) = build_stt_provider(voice_cfg)
        {
            tracing::info!(
                provider = %voice_cfg.stt_provider,
                "Speech-to-text enabled"
            );
            media = media.with_stt(stt);
        }

        handler = handler.with_media(Arc::new(media));

        // 7g. Enable interactive tool approval if configured.
        if self.config.security.require_tool_approval {
            let approval_tools: std::collections::HashSet<String> = self
                .config
                .security
                .approval_tools
                .iter()
                .cloned()
                .collect();
            handler = handler.with_tool_approval(approval_tools);
            tracing::info!("Interactive tool approval enabled");
        }

        // 7h. Wire shell hooks if any are configured.
        let shell_runner = ShellHookRunner::from_config(&self.config.hooks);
        if shell_runner.has_any() {
            if let Some(pre_script) = shell_runner.pre_tool_use_path() {
                let pre_hook = ShellPreToolHook::new(pre_script.to_string());
                handler = handler.with_shell_pre_tool_hook(std::sync::Arc::new(pre_hook));
                tracing::info!("Shell pre-tool hook enabled");
            }
            handler = handler.with_session_hook(std::sync::Arc::new(shell_runner));
            tracing::info!("Shell session hooks enabled");
        }

        // 7i. Attach shared metrics for token usage tracking.
        handler = handler.with_metrics(Arc::clone(&metrics));

        // 7k. Wire cross-channel identity store if enabled.
        let mut identity_store: Option<Arc<UserIdentityStore>> = None;
        if self.config.identity.enabled {
            let store_path = expand_tilde_str(&self.config.identity.store_path);
            match UserIdentityStore::new(&store_path) {
                Ok(store) => {
                    let store = Arc::new(store);
                    handler = handler.with_identity_store(Arc::clone(&store));
                    identity_store = Some(store);
                    tracing::info!(path = %store_path, "Cross-channel identity store enabled");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to open identity store; cross-channel identity disabled");
                }
            }
        }

        // 7l. Wire DM pairing policy. Secure default: if no config is
        //      provided, the gateway runs in Pairing mode and unknown peers
        //      are intercepted before their messages reach the agent.
        let pairing_store = self.build_pairing_store().await;
        if let Some(ref store) = pairing_store {
            let default_pol = self
                .config
                .pairing
                .default
                .clone()
                .unwrap_or_else(default_policy);
            let per_channel = self.config.pairing.channels.clone();
            let policy_fn = std::sync::Arc::new(move |channel: &str| {
                per_channel
                    .get(channel)
                    .cloned()
                    .unwrap_or_else(|| default_pol.clone())
            });
            let pairing_handler =
                std::sync::Arc::new(PairingHandler::new(Arc::clone(store), policy_fn));
            handler = handler.with_pairing(pairing_handler);
            tracing::info!(
                path = %store.path().display(),
                "DM pairing policy enabled"
            );
        }

        // 7j. Wire TTS if configured (voice feature only).
        #[allow(unused_mut)] // mut required when `voice` feature is enabled.
        let mut tts_audio_dir: Option<std::path::PathBuf> = None;
        #[cfg(feature = "voice")]
        if let Some(ref voice_cfg) = self.config.voice
            && let Some(ref tts_provider_name) = voice_cfg.tts_provider
            && let Some(tts_provider) = build_tts_provider(voice_cfg)
        {
            use brainwires_gateway::tts::TtsProcessor;
            use brainwires_hardware::{OutputFormat, TtsOptions, Voice};

            let format = match voice_cfg.tts_format.as_deref().unwrap_or("mp3") {
                "opus" => OutputFormat::Opus,
                "flac" => OutputFormat::Flac,
                "wav" => OutputFormat::Wav,
                _ => OutputFormat::Mp3,
            };
            let voice_id = voice_cfg
                .tts_voice
                .clone()
                .unwrap_or_else(|| "alloy".to_string());
            let options = TtsOptions {
                voice: Voice {
                    id: voice_id,
                    name: None,
                    language: voice_cfg.language.clone(),
                },
                output_format: format,
                speed: None,
            };
            let audio_dir = voice_cfg
                .tts_audio_dir
                .as_deref()
                .map(|p| std::path::PathBuf::from(expand_tilde_str(p)))
                .unwrap_or_else(|| std::env::temp_dir().join("brainclaw-audio"));
            let base_url = voice_cfg.tts_base_url.clone().unwrap_or_else(|| {
                format!(
                    "http://{}:{}/audio",
                    self.config.gateway.host, self.config.gateway.port
                )
            });

            let processor = Arc::new(TtsProcessor::new(
                tts_provider,
                options,
                audio_dir.clone(),
                base_url,
            ));
            handler = handler.with_tts(processor);
            tts_audio_dir = Some(audio_dir);
            tracing::info!(provider = %tts_provider_name, "TTS output enabled");
        }

        // 7n. Wire Gmail push ingestion.  Each configured account gets a
        //       long-lived `GmailPushHandler`; the in-memory registry owns
        //       the per-mailbox cursors and is attached to both the
        //       gateway (for the webhook) and a background task (for watch
        //       renewal every ~6 days).  Gmail push and IMAP polling are
        //       mutually exclusive per account — if both are configured,
        //       push wins and a warning is emitted on startup.
        #[cfg(feature = "email-push")]
        let gmail_push_registry = self.build_gmail_push_registry().await;

        // 8. Create Gateway with handler, sharing the same sessions/channels/metrics.
        let handler = Arc::new(handler);
        let mut gateway = Gateway::with_handler(gateway_config.clone(), Arc::clone(&handler) as _)
            .with_shared_state(Arc::clone(&sessions), Arc::clone(&channels))
            .with_metrics(Arc::clone(&metrics));

        #[cfg(feature = "email-push")]
        if let Some(ref registry) = gmail_push_registry {
            gateway = gateway.with_gmail_push(Arc::clone(registry));
            tracing::info!("Gmail push webhook mounted at /webhooks/gmail-push");
        }

        if let Some(audio_dir) = tts_audio_dir {
            gateway = gateway.with_audio_dir(audio_dir);
        }

        // 8a. Attach provider for OpenAI-compatible endpoint.
        gateway = gateway.with_openai_provider(openai_provider);

        // 8ab. Wire the webchat verifier + history store when the browser
        //      chat channel is enabled and we have (or can derive) a
        //      shared HS256 secret.
        if self.config.webchat.enabled {
            if let Some(secret) = self.config.resolve_webchat_secret() {
                let verifier = Arc::new(brainwires_gateway::webchat::Hs256Verifier::new(
                    secret.as_bytes().to_vec(),
                ));
                let history = Arc::new(brainwires_gateway::webchat::WebChatHistory::new());
                gateway = gateway.with_webchat_verifier(verifier, history);
                tracing::info!("WebChat JWT endpoint enabled at /webchat/ws");
            } else {
                tracing::warn!(
                    "WebChat is enabled but no JWT secret could be resolved \
                     (set [webchat] jwt_secret or [security] admin_token); \
                     /webchat/ws will reject all upgrade attempts"
                );
            }
        }

        // 8c. Attach identity store to gateway if enabled.
        if let Some(ref store) = identity_store {
            gateway = gateway.with_identity_store(Arc::clone(store));
        }

        // 8d. Attach pairing store to gateway so the admin pairing endpoints
        //     are wired up.
        if let Some(ref store) = pairing_store {
            gateway = gateway.with_pairing_store(Arc::clone(store));
        }
        tracing::info!("OpenAI-compatible API enabled at /v1/chat/completions");

        // 8b. Start cron runner if enabled. Share the store with the gateway
        //     so the admin cron API endpoints can manage jobs at runtime.
        if self.config.cron.enabled {
            let cron_dir = expand_tilde(&self.config.cron.storage_dir);
            match CronStore::new(&cron_dir) {
                Ok(store) => {
                    let store = Arc::new(store);
                    let runner = Arc::new(CronRunner::new(
                        Arc::clone(&store),
                        Arc::clone(&handler),
                        Arc::clone(&channels),
                    ));
                    runner.spawn();
                    gateway = gateway.with_cron_store(store);
                    tracing::info!(dir = %cron_dir.display(), "Cron runner started");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize cron store; cron disabled");
                }
            }
        }

        // 8e. Spawn the Gmail watch-renewal task(s). Gmail watches expire
        //       after 7 days; we re-register every 6 days with a small
        //       random jitter so concurrent daemons don't all hammer
        //       Google at the same second. The task is detached — if
        //       renewal starts failing the operator finds out via the
        //       doctor check and the log.
        #[cfg(feature = "email-push")]
        if let Some(ref registry) = gmail_push_registry {
            self.spawn_gmail_watch_renewals(Arc::clone(registry)).await;
        }

        tracing::info!(
            address = %gateway_config.bind_address(),
            "BrainClaw ready"
        );

        // 9. Run gateway (blocks until shutdown)
        gateway.run().await
    }

    /// Build the [`GmailPushRegistry`] from config, seeding each handler
    /// and kicking off an initial watch registration when the cursor
    /// store has no entry for that mailbox yet.
    #[cfg(feature = "email-push")]
    async fn build_gmail_push_registry(
        &self,
    ) -> Option<Arc<brainwires_gateway::gmail_push::GmailPushRegistry>> {
        use brainwires_gateway::gmail_push::{GmailCursorStore, GmailPushRegistry};
        use brainwires_tools::gmail_push::GmailPushHandler;

        if !self.config.gmail_push.enabled || self.config.gmail_push.accounts.is_empty() {
            return None;
        }

        // The email tool (IMAP/SMTP) only runs on demand — it has no
        // polling loop in this codebase — so there's no double-delivery
        // to guard against.  Still, warn the operator when the same
        // mailbox is wired into both: it usually means the config was
        // copy-pasted.
        if let Some(ref email) = self.config.email {
            for acct in &self.config.gmail_push.accounts {
                if email.username.eq_ignore_ascii_case(&acct.email_address) {
                    tracing::warn!(
                        email = %acct.email_address,
                        "Gmail push is configured for the same account that has IMAP/SMTP credentials. \
                         Push will deliver inbound messages; the IMAP tool remains available for on-demand search."
                    );
                }
            }
        }

        let cursor_path = match self.config.gmail_push.cursor_store.clone() {
            Some(p) => p,
            None => match dirs::home_dir() {
                Some(h) => h.join(".brainclaw").join("gmail_cursor.json"),
                None => {
                    tracing::warn!(
                        "Cannot resolve home directory for gmail cursor store; Gmail push disabled"
                    );
                    return None;
                }
            },
        };

        let cursors = match GmailCursorStore::load(&cursor_path).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %cursor_path.display(),
                    "Failed to open Gmail cursor store; Gmail push disabled"
                );
                return None;
            }
        };
        let registry = Arc::new(GmailPushRegistry::new(Arc::clone(&cursors)));

        for acct in &self.config.gmail_push.accounts {
            // Resolve token again — keeps secrets off our argument list.
            let token = match (&acct.oauth_token_env, &acct.oauth_token) {
                (Some(env), _) => match std::env::var(env) {
                    Ok(v) if !v.is_empty() => v,
                    _ => continue,
                },
                (None, Some(inline)) if !inline.is_empty() => inline.clone(),
                _ => continue,
            };
            let label_ids = if acct.watched_label_ids.is_empty() {
                vec!["INBOX".to_string()]
            } else {
                acct.watched_label_ids.clone()
            };
            let cfg = brainwires_tools::gmail_push::GmailPushConfig {
                project_id: acct.project_id.clone(),
                topic_name: acct.topic_name.clone(),
                push_audience: acct.push_audience.clone(),
                watched_label_ids: label_ids,
                oauth_token: token,
                gmail_base_url: None,
            };
            let handler = Arc::new(GmailPushHandler::new(cfg));
            registry
                .register(acct.email_address.clone(), Arc::clone(&handler))
                .await;
            tracing::info!(
                email = %acct.email_address,
                topic = %acct.topic_name,
                "Gmail push handler registered"
            );
        }

        Some(registry)
    }

    /// Spawn background watch-renewal tasks — one per registered mailbox.
    ///
    /// Watches expire at 7 days; we renew at 6 days with ±1 hour jitter.
    /// Failures are logged but do not crash the daemon — the operator can
    /// re-run `brainclaw gmail-watch register --account <email>` to
    /// recover.
    #[cfg(feature = "email-push")]
    async fn spawn_gmail_watch_renewals(
        &self,
        registry: Arc<brainwires_gateway::gmail_push::GmailPushRegistry>,
    ) {
        use std::time::Duration;
        for acct in &self.config.gmail_push.accounts {
            let email = acct.email_address.clone();
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                // Perform an initial registration immediately so Pub/Sub
                // has a live subscription before the first message lands.
                if let Some(handler) = registry.get(&email).await {
                    match brainwires_gateway::gmail_push::register_watch_and_seed(
                        &handler,
                        &email,
                        &registry.cursors(),
                    )
                    .await
                    {
                        Ok(resp) => {
                            tracing::info!(
                                email = %email,
                                expires = %resp.expiration,
                                history_id = resp.history_id,
                                "Gmail watch registered"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                email = %email,
                                error = %e,
                                "Initial Gmail watch registration failed"
                            );
                        }
                    }
                }
                loop {
                    // 6 days +/- 1 hour.
                    let base = Duration::from_secs(6 * 24 * 60 * 60);
                    // Cheap jitter without pulling rand into the daemon
                    // for this one call — 0..3600 via the nanosecond
                    // clock.
                    let jitter_secs = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.subsec_nanos() as u64 % 7200)
                        .unwrap_or(0))
                    .saturating_sub(3600);
                    let sleep_for = base + Duration::from_secs(jitter_secs);
                    tokio::time::sleep(sleep_for).await;

                    if let Some(handler) = registry.get(&email).await {
                        match handler.register_watch().await {
                            Ok(resp) => {
                                tracing::info!(
                                    email = %email,
                                    expires = %resp.expiration,
                                    history_id = resp.history_id,
                                    "Gmail watch renewed"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    email = %email,
                                    error = %e,
                                    "Gmail watch renewal failed; will retry on next cycle"
                                );
                            }
                        }
                    } else {
                        break;
                    }
                }
            });
        }
    }

    /// Wrap `builtin` with a `SandboxedToolExecutor` if the `sandbox` feature
    /// is enabled AND `config.sandbox.enabled` is true. Otherwise return the
    /// builtin executor untouched.
    ///
    /// On sandbox construction failure, this method obeys
    /// `config.sandbox.fallback_to_host_on_error`: when `true`, it logs the
    /// error and falls back to the unsandboxed builtin; when `false`, it
    /// returns the error so the daemon exits instead of silently downgrading
    /// isolation.
    #[cfg(feature = "sandbox")]
    fn wrap_with_sandbox(&self, builtin: BuiltinToolExecutor) -> Result<Arc<dyn ToolExecutor>> {
        use brainwires_sandbox::{Sandbox, SandboxRuntime};
        use brainwires_tools::SandboxedToolExecutor;
        use std::time::Duration;

        let sb = &self.config.sandbox;
        if !sb.enabled {
            tracing::info!("Sandbox disabled by config; tool calls run on the host");
            return Ok(Arc::new(builtin) as Arc<dyn ToolExecutor>);
        }

        let policy = match sb.to_policy() {
            Ok(p) => p,
            Err(e) => {
                if sb.fallback_to_host_on_error {
                    tracing::error!(
                        error = %e,
                        "Sandbox policy invalid; falling back to unsandboxed executor"
                    );
                    return Ok(Arc::new(builtin) as Arc<dyn ToolExecutor>);
                }
                return Err(e);
            }
        };

        let sandbox_result: Result<Arc<dyn Sandbox>> = match policy.runtime {
            SandboxRuntime::Docker | SandboxRuntime::Podman => {
                match brainwires_sandbox::DockerSandbox::connect(policy.clone()) {
                    Ok(s) => Ok(Arc::new(s) as Arc<dyn Sandbox>),
                    Err(e) => Err(anyhow::anyhow!(
                        "sandbox: failed to connect to {:?} daemon: {}",
                        policy.runtime,
                        e
                    )),
                }
            }
            SandboxRuntime::Host => {
                #[cfg(feature = "sandbox-unsafe-host")]
                {
                    tracing::warn!(
                        "Sandbox runtime = 'host' — dev/testing only, NO isolation is applied"
                    );
                    Ok(
                        Arc::new(brainwires_sandbox::HostSandbox::new(policy.clone()))
                            as Arc<dyn Sandbox>,
                    )
                }
                #[cfg(not(feature = "sandbox-unsafe-host"))]
                {
                    Err(anyhow::anyhow!(
                        "sandbox.runtime = 'host' requires the `sandbox-unsafe-host` build feature"
                    ))
                }
            }
        };

        let sandbox = match sandbox_result {
            Ok(s) => s,
            Err(e) => {
                if sb.fallback_to_host_on_error {
                    tracing::error!(
                        error = %e,
                        "Sandbox backend unavailable; falling back to unsandboxed executor \
                         (sandbox.fallback_to_host_on_error = true)"
                    );
                    return Ok(Arc::new(builtin) as Arc<dyn ToolExecutor>);
                }
                return Err(e);
            }
        };

        tracing::info!(
            runtime = ?policy.runtime,
            image = %policy.image,
            timeout_secs = sb.default_timeout_secs,
            "Sandbox enabled; dangerous tool calls will be isolated"
        );

        let wrapped = SandboxedToolExecutor::new(builtin, sandbox, policy)
            .with_timeout(Duration::from_secs(sb.default_timeout_secs));
        Ok(Arc::new(wrapped) as Arc<dyn ToolExecutor>)
    }

    /// No-op stub used when the daemon is built without the `sandbox` feature.
    #[cfg(not(feature = "sandbox"))]
    fn wrap_with_sandbox(&self, builtin: BuiltinToolExecutor) -> Result<Arc<dyn ToolExecutor>> {
        tracing::info!("Sandbox feature not compiled in; tool calls run directly on the host");
        Ok(Arc::new(builtin) as Arc<dyn ToolExecutor>)
    }

    /// Inject tool-specific configs into `ToolContext.metadata` as JSON strings.
    ///
    /// Tools read their config from metadata at call time; this avoids passing
    /// typed configs through the generic tool registry.
    fn inject_tool_configs(
        &self,
        #[cfg_attr(
            not(any(feature = "email", feature = "calendar")),
            allow(unused_variables)
        )]
        context: &mut ToolContext,
    ) {
        #[cfg(feature = "email")]
        if let Some(result) = self.config.to_email_config() {
            match result {
                Ok(cfg) => match serde_json::to_string(&cfg) {
                    Ok(json) => {
                        context.metadata.insert("email_config".to_string(), json);
                        tracing::debug!("Email tool config injected into ToolContext");
                    }
                    Err(e) => tracing::warn!(error = %e, "Failed to serialize email config"),
                },
                Err(e) => tracing::warn!(
                    error = %e,
                    "Email config error; email tools will fail at call time"
                ),
            }
        }

        #[cfg(feature = "calendar")]
        if let Some(result) = self.config.to_calendar_config() {
            match result {
                Ok(cfg) => match serde_json::to_string(&cfg) {
                    Ok(json) => {
                        context.metadata.insert("calendar_config".to_string(), json);
                        tracing::debug!("Calendar tool config injected into ToolContext");
                    }
                    Err(e) => tracing::warn!(error = %e, "Failed to serialize calendar config"),
                },
                Err(e) => tracing::warn!(
                    error = %e,
                    "Calendar config error; calendar tools will fail at call time"
                ),
            }
        }

        // Inject browser config so BrowserTool can read thalora_binary / session_timeout_secs
        #[cfg(feature = "browser")]
        if let Some(ref browser_cfg) = self.config.browser {
            match serde_json::to_string(browser_cfg) {
                Ok(json) => {
                    context.metadata.insert("browser_config".to_string(), json);
                    tracing::debug!("Browser tool config injected into ToolContext");
                }
                Err(e) => tracing::warn!(error = %e, "Failed to serialize browser config"),
            }
        }
        #[cfg(not(feature = "browser"))]
        let _ = &self.config.browser;
    }

    /// Build (and load) the pairing store from the `[pairing]` config section.
    ///
    /// Falls back to `~/.brainclaw/pairing.json` when `pairing.store_path`
    /// is unset. Returns `None` if the path cannot be resolved (e.g.
    /// missing `$HOME`).
    async fn build_pairing_store(&self) -> Option<Arc<PairingStore>> {
        let configured = self.config.pairing.store_path.clone();
        let path: std::path::PathBuf = match configured {
            Some(p) => std::path::PathBuf::from(expand_tilde_str(&p)),
            None => match dirs::home_dir() {
                Some(home) => home.join(".brainclaw").join("pairing.json"),
                None => {
                    tracing::warn!(
                        "Cannot resolve home directory for default pairing store path; \
                         pairing disabled"
                    );
                    return None;
                }
            },
        };

        match PairingStore::load(&path) {
            Ok(store) => {
                let store = Arc::new(store);
                store
                    .set_allowlist(self.config.pairing.allow_from.clone())
                    .await;
                Some(store)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "Failed to open pairing store; pairing disabled"
                );
                None
            }
        }
    }

    /// Resolve the API key from config, environment variable, or standard env vars.
    fn resolve_api_key(&self) -> Result<Option<String>> {
        // 1. Direct config value
        if let Some(ref key) = self.config.provider.api_key
            && !key.is_empty()
        {
            return Ok(Some(key.clone()));
        }

        // 2. Custom env var name from config
        if let Some(ref env_name) = self.config.provider.api_key_env
            && let Ok(key) = std::env::var(env_name)
            && !key.is_empty()
        {
            return Ok(Some(key));
        }

        // 3. Standard env vars based on provider
        let env_var = match self.config.provider.default_provider.as_str() {
            "anthropic" => "ANTHROPIC_API_KEY",
            "openai" | "openai-responses" | "openai_responses" => "OPENAI_API_KEY",
            "google" | "gemini" => "GOOGLE_API_KEY",
            "groq" => "GROQ_API_KEY",
            "together" => "TOGETHER_API_KEY",
            "fireworks" => "FIREWORKS_API_KEY",
            "anyscale" => "ANYSCALE_API_KEY",
            "brainwires" => "BRAINWIRES_API_KEY",
            "elevenlabs" => "ELEVENLABS_API_KEY",
            "deepgram" => "DEEPGRAM_API_KEY",
            _ => "",
        };

        if !env_var.is_empty()
            && let Ok(key) = std::env::var(env_var)
            && !key.is_empty()
        {
            return Ok(Some(key));
        }

        Ok(None)
    }
}

/// Build an STT provider from the voice configuration.
///
/// Returns `None` with a warning if the provider is unknown or the API key is missing
/// for a provider that requires one.
#[cfg(feature = "voice")]
fn build_stt_provider(
    cfg: &crate::config::VoiceSection,
) -> Option<std::sync::Arc<dyn brainwires_hardware::SpeechToText>> {
    use brainwires_hardware::{
        AzureStt, DeepgramStt, ElevenLabsStt, FishStt, OpenAiStt, SpeechToText,
    };

    /// Resolve an API key: first from `api_key_env`, then from a named env var.
    fn resolve_key(cfg: &crate::config::VoiceSection, default_var: &str) -> Option<String> {
        let var_name = cfg.api_key_env.as_deref().unwrap_or(default_var);
        std::env::var(var_name).ok().filter(|k| !k.is_empty())
    }

    match cfg.stt_provider.as_str() {
        "openai" | "openai-responses" => {
            let key = resolve_key(cfg, "OPENAI_API_KEY")?;
            Some(std::sync::Arc::new(OpenAiStt::new(key)) as Arc<dyn SpeechToText>)
        }
        "deepgram" => {
            let key = resolve_key(cfg, "DEEPGRAM_API_KEY")?;
            Some(std::sync::Arc::new(DeepgramStt::new(key)) as Arc<dyn SpeechToText>)
        }
        "elevenlabs" => {
            let key = resolve_key(cfg, "ELEVENLABS_API_KEY")?;
            Some(std::sync::Arc::new(ElevenLabsStt::new(key)) as Arc<dyn SpeechToText>)
        }
        "fish" => {
            let key = resolve_key(cfg, "FISH_API_KEY")?;
            Some(std::sync::Arc::new(FishStt::new(key)) as Arc<dyn SpeechToText>)
        }
        "azure" => {
            // Azure requires both subscription key and region.
            let key = resolve_key(cfg, "AZURE_SPEECH_KEY")?;
            let region = std::env::var("AZURE_SPEECH_REGION")
                .ok()
                .filter(|r| !r.is_empty())?;
            Some(std::sync::Arc::new(AzureStt::new(key, region)) as Arc<dyn SpeechToText>)
        }
        #[cfg(feature = "local-stt")]
        "whisper-local" | "whisper" => {
            // Local Whisper requires a GGML model file. Resolve from
            // `WHISPER_MODEL_PATH`; if unset, voice transcription stays
            // disabled rather than panicking at runtime.
            let model_path = std::env::var("WHISPER_MODEL_PATH")
                .ok()
                .filter(|p| !p.is_empty())?;
            Some(
                std::sync::Arc::new(brainwires_hardware::WhisperStt::new(model_path))
                    as Arc<dyn SpeechToText>,
            )
        }
        other => {
            tracing::warn!(provider = %other, "Unknown STT provider; voice transcription disabled");
            None
        }
    }
}

/// Build a TTS provider from the voice configuration.
///
/// Uses `tts_provider` to select the implementation.  Returns `None` if the
/// provider is unknown or the required API key is missing.
#[cfg(feature = "voice")]
fn build_tts_provider(
    cfg: &crate::config::VoiceSection,
) -> Option<std::sync::Arc<dyn brainwires_hardware::TextToSpeech>> {
    use brainwires_hardware::{
        CartesiaTts, DeepgramTts, ElevenLabsTts, GoogleTts, OpenAiTts, TextToSpeech,
    };

    fn resolve_key(cfg: &crate::config::VoiceSection, default_var: &str) -> Option<String> {
        let var_name = cfg.api_key_env.as_deref().unwrap_or(default_var);
        std::env::var(var_name).ok().filter(|k| !k.is_empty())
    }

    match cfg.tts_provider.as_deref().unwrap_or("") {
        "openai" | "openai-responses" => {
            let key = resolve_key(cfg, "OPENAI_API_KEY")?;
            Some(std::sync::Arc::new(OpenAiTts::new(key)) as Arc<dyn TextToSpeech>)
        }
        "elevenlabs" => {
            let key = resolve_key(cfg, "ELEVENLABS_API_KEY")?;
            Some(std::sync::Arc::new(ElevenLabsTts::new(key)) as Arc<dyn TextToSpeech>)
        }
        "deepgram" => {
            let key = resolve_key(cfg, "DEEPGRAM_API_KEY")?;
            Some(std::sync::Arc::new(DeepgramTts::new(key)) as Arc<dyn TextToSpeech>)
        }
        "google" => {
            let key = resolve_key(cfg, "GOOGLE_API_KEY")?;
            Some(std::sync::Arc::new(GoogleTts::new(key)) as Arc<dyn TextToSpeech>)
        }
        "cartesia" => {
            let key = resolve_key(cfg, "CARTESIA_API_KEY")?;
            Some(std::sync::Arc::new(CartesiaTts::new(key)) as Arc<dyn TextToSpeech>)
        }
        other => {
            tracing::warn!(provider = %other, "Unknown TTS provider; voice output disabled");
            None
        }
    }
}

/// Load context from the standard CONTEXT.md locations and any extra paths.
///
/// Checks in order:
/// 1. `~/.brainclaw/CONTEXT.md` (global user context)
/// 2. `.brainclaw/CONTEXT.md` (project-level context in the daemon's cwd)
/// 3. Any extra paths specified in `persona.context_files`
///
/// Returns all content concatenated with blank-line separators.
fn load_context_files(extra_paths: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();

    let mut candidates: Vec<PathBuf> = Vec::new();

    // Standard locations
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".brainclaw").join("CONTEXT.md"));
    }
    candidates.push(PathBuf::from(".brainclaw/CONTEXT.md"));

    // User-configured extra files
    for p in extra_paths {
        candidates.push(PathBuf::from(expand_tilde_str(p)));
    }

    for path in candidates {
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) if !content.trim().is_empty() => {
                    tracing::info!(path = %path.display(), "Loaded context file");
                    parts.push(content);
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to read context file");
                }
            }
        }
    }

    parts.join("\n\n")
}

/// Expand a leading `~` to the home directory.
fn expand_tilde_str(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}
