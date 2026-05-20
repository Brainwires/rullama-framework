//! BrainClaw configuration — TOML-based with sensible defaults.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use brainwires_gateway::config::GatewayConfig;
use brainwires_gateway::pairing::PairingPolicy;
use serde::{Deserialize, Serialize};

/// Top-level BrainClaw configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct BrainClawConfig {
    /// Gateway (WebSocket server) settings.
    pub gateway: GatewaySection,
    /// AI provider settings.
    pub provider: ProviderSection,
    /// Agent behaviour settings.
    pub agent: AgentSection,
    /// Tool availability settings.
    pub tools: ToolsSection,
    /// Persona / system prompt settings.
    pub persona: PersonaSection,
    /// Conversation memory settings.
    pub memory: MemorySection,
    /// Skill system settings.
    pub skills: SkillsSection,
    /// Security settings.
    pub security: SecuritySection,
    /// Cron / scheduled task settings.
    pub cron: CronSection,
    /// User-configurable shell hooks.
    pub hooks: HooksSection,
    /// Email tool settings (requires `email` feature; tool group `"email"` must be in `tools.enabled`).
    pub email: Option<EmailSection>,
    /// Gmail push ingestion settings (requires `email-push` feature).
    /// When enabled, inbound Gmail is delivered via Google Pub/Sub instead
    /// of (or in addition to) IMAP polling.
    #[serde(default)]
    pub gmail_push: GmailPushSection,
    /// Calendar tool settings (requires `calendar` feature; tool group `"calendar"` must be in `tools.enabled`).
    pub calendar: Option<CalendarSection>,
    /// Browser automation settings (requires `browser` feature; tool group `"browser"` must be in `tools.enabled`).
    pub browser: Option<BrowserSection>,
    /// Voice / speech-to-text settings (requires `voice` feature).
    pub voice: Option<VoiceSection>,
    /// Cross-channel user identity settings.
    pub identity: IdentitySection,
    /// Sandbox settings — container-based isolation for dangerous tool calls.
    pub sandbox: SandboxConfig,
    /// DM pairing policy — gates unknown peers behind an operator-approval flow.
    pub pairing: PairingSection,
    /// Browser-based WebChat channel settings.
    pub webchat: WebChatSection,
}

/// Browser-based WebChat channel settings.
///
/// The webchat channel is exposed at `/webchat/ws` on the gateway and
/// authenticates each browser connection with an HS256 JWT signed by
/// `jwt_secret`. When `jwt_secret` is `None` at startup, the daemon
/// derives a stable secret from `security.admin_token`; if that is also
/// unset, a fresh random secret is generated and logged on first boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebChatSection {
    /// Whether the `/webchat/ws` endpoint is enabled.
    pub enabled: bool,
    /// Explicit HS256 shared secret. When unset, derived from
    /// `security.admin_token` or randomly generated at startup.
    pub jwt_secret: Option<String>,
    /// Maximum history entries retained per webchat session.
    pub session_history_limit: usize,
    /// Maximum attachment size in bytes (reserved — attachments scoped
    /// out in the initial cut).
    pub attachment_max_bytes: u64,
    /// Optional per-channel origin allow-list.  When empty, inherits
    /// `security.allowed_origins`.
    pub allowed_origins: Vec<String>,
}

impl Default for WebChatSection {
    fn default() -> Self {
        Self {
            enabled: true,
            jwt_secret: None,
            session_history_limit: 50,
            attachment_max_bytes: 10 * 1024 * 1024,
            allowed_origins: Vec::new(),
        }
    }
}

/// Per-channel DM pairing policy configuration.
///
/// The gateway rejects direct messages from unknown peers unless they are
/// explicitly paired via the approval flow. `default` is applied to any
/// channel not listed in `channels`; `allow_from` pre-approves peers
/// without requiring a code.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PairingSection {
    /// Default policy applied to channels without their own override.
    /// `None` resolves to the library default (Pairing mode, 15-minute TTL).
    pub default: Option<PairingPolicy>,
    /// Per-channel overrides keyed by channel name (e.g. "discord", "telegram").
    pub channels: std::collections::HashMap<String, PairingPolicy>,
    /// Pre-approved peers keyed by `<channel>:<user_id>`.
    pub allow_from: Vec<String>,
    /// Path to the pairing store JSON file.
    pub store_path: Option<String>,
}

// ── Section structs ─────────────────────────────────────────────────────

/// Gateway (WebSocket server) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewaySection {
    /// Host address to bind to.
    pub host: String,
    /// Port to listen on.
    pub port: u16,
    /// Maximum number of concurrent channel connections.
    pub max_connections: usize,
    /// Session inactivity timeout in seconds.
    pub session_timeout_secs: u64,
    /// Allowed API tokens for channel connections (empty = open mode).
    pub auth_tokens: Vec<String>,
}

/// AI provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderSection {
    /// Default provider name (anthropic, openai, google, groq, ollama, etc.).
    pub default_provider: String,
    /// Default model name (None = use provider default).
    pub default_model: Option<String>,
    /// API key (if set directly in config).
    pub api_key: Option<String>,
    /// Environment variable name to read the API key from.
    pub api_key_env: Option<String>,
    /// Sampling temperature (0.0 - 1.0).
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
}

/// Agent behaviour configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSection {
    /// Maximum tool-call rounds per message.
    pub max_tool_rounds: usize,
    /// Maximum concurrent agent sessions.
    pub max_concurrent_sessions: usize,
    /// Session idle timeout in seconds.
    pub session_idle_timeout_secs: u64,
}

/// Tool availability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsSection {
    /// List of enabled tool groups.
    pub enabled: Vec<String>,
    /// List of explicitly disabled tool groups (overrides enabled).
    pub disabled: Vec<String>,
    /// Whether the bash/shell tool is allowed.
    pub bash_allowed: bool,
}

/// Persona / system prompt configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaSection {
    /// Name of the assistant persona.
    pub name: String,
    /// Inline system prompt.
    pub system_prompt: Option<String>,
    /// Path to a file containing the system prompt.
    pub system_prompt_file: Option<String>,
    /// Additional context files to load and prepend to the system prompt.
    ///
    /// BrainClaw also checks `~/.brainclaw/CONTEXT.md` and `.brainclaw/CONTEXT.md`
    /// automatically without listing them here.
    pub context_files: Vec<String>,
}

/// Conversation memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemorySection {
    /// Whether conversation memory is enabled.
    pub enabled: bool,
    /// Directory for memory storage.
    pub storage_dir: String,
    /// Maximum history messages to keep per session.
    pub max_history_messages: usize,
    /// Whether to persist conversations across restarts (JSON file store).
    pub persist_conversations: bool,
}

/// Skill system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SkillsSection {
    /// Whether the skill system is enabled.
    pub enabled: bool,
    /// Directories to scan for SKILL.md files.
    pub directories: Vec<String>,
    /// URL of a remote skill registry server (e.g. `http://localhost:8765`).
    ///
    /// When set, skills not found locally are looked up in the registry and
    /// downloaded on demand.  Leave unset to use filesystem-only dispatch.
    pub registry_url: Option<String>,
}

/// Cron / scheduled task configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CronSection {
    /// Whether the cron runner is enabled.
    pub enabled: bool,
    /// Directory where cron job JSON files are persisted.
    pub storage_dir: String,
}

/// User-configurable shell hooks configuration.
///
/// Each field is an optional path to a shell script.  BrainClaw invokes the
/// script with the event payload as JSON on stdin.  For `pre_tool_use`,
/// a non-zero exit code blocks the tool call; the first line of stdout is
/// used as the rejection reason sent back to the agent.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct HooksSection {
    /// Script to run before each tool execution.  Exit non-zero to block.
    pub pre_tool_use: Option<String>,
    /// Script to run after each tool execution (informational).
    pub post_tool_use: Option<String>,
    /// Script to run when an agent session starts.
    pub session_start: Option<String>,
    /// Script to run when an agent session ends (completed or failed).
    pub session_end: Option<String>,
}

/// Security configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecuritySection {
    /// Allowed WebSocket origins (empty = allow all).
    pub allowed_origins: Vec<String>,
    /// Strip system-message spoofing from inbound messages.
    pub strip_system_spoofing: bool,
    /// Redact secret patterns in outbound messages.
    pub redact_secrets_in_output: bool,
    /// Maximum messages per minute per user.
    pub max_messages_per_minute: u32,
    /// Maximum tool calls per minute per user.
    pub max_tool_calls_per_minute: u32,
    /// Require cryptographic signatures on skill packages.
    pub require_signed_skills: bool,
    /// Bearer token for admin API authentication (None = no auth).
    pub admin_token: Option<String>,
    /// HMAC secret for webhook signature verification (None = no verification).
    pub webhook_secret: Option<String>,
    /// Master switch — when `false`, all channel connections are refused.
    pub channels_enabled: bool,
    /// Allowed channel adapter types (e.g. `["discord", "telegram"]`). Empty = allow all.
    pub allowed_channel_types: Vec<String>,
    /// Allowed channel adapter IDs. Empty = allow all.
    pub allowed_channel_ids: Vec<String>,
    /// Require interactive user approval before executing tool calls via chat.
    /// When enabled, the agent sends "⚠️ Run `tool`? Reply yes/no" to the user.
    pub require_tool_approval: bool,
    /// Tools that require approval.  Empty = all tools require approval.
    /// Example: ["bash", "file_write", "http_request"]
    pub approval_tools: Vec<String>,
}

/// Email tool configuration (IMAP + SMTP).
///
/// Stored in the `[email]` section. The password is read from an environment
/// variable at runtime so it is never written to disk in plaintext.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSection {
    /// IMAP server hostname (e.g. `imap.gmail.com`).
    pub imap_host: String,
    /// IMAP port (default 993 for TLS).
    #[serde(default = "default_imap_port")]
    pub imap_port: u16,
    /// SMTP server hostname (e.g. `smtp.gmail.com`).
    pub smtp_host: String,
    /// SMTP port (default 587 for STARTTLS).
    #[serde(default = "default_smtp_port")]
    pub smtp_port: u16,
    /// Email account username / address.
    pub username: String,
    /// Name of the environment variable holding the email password.
    pub password_env: String,
    /// Whether to use TLS (default true).
    #[serde(default = "default_true")]
    pub tls: bool,
    /// Default "From" address used when sending email.
    pub from_address: String,
}

fn default_imap_port() -> u16 {
    993
}

/// Gmail push (Google Pub/Sub) ingestion configuration.
///
/// Each watched Gmail account needs its own entry in `accounts`. When
/// `enabled = false`, no watches are registered and the webhook is not
/// exposed — IMAP polling (if configured) remains the only inbound path.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GmailPushSection {
    /// Whether Gmail push is enabled.
    pub enabled: bool,
    /// Watched Gmail accounts.
    pub accounts: Vec<GmailAccountConfig>,
    /// Optional override for the history-cursor JSON file.
    /// Defaults to `~/.brainclaw/gmail_cursor.json`.
    pub cursor_store: Option<PathBuf>,
}

/// One Gmail mailbox pushing into this daemon.
///
/// The OAuth token is resolved from either `oauth_token_env` (preferred,
/// keeps secrets off disk) or `oauth_token` (inline fallback).  When
/// neither is set and the environment variable isn't present at startup,
/// the handler for that account is not constructed and a warning is
/// emitted — the daemon continues to serve other channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmailAccountConfig {
    /// The mailbox address being watched (used as the lookup key when
    /// Pub/Sub pushes arrive).
    pub email_address: String,
    /// GCP project id that owns the Pub/Sub topic.
    pub project_id: String,
    /// Fully-qualified topic name (`projects/<proj>/topics/<topic>`).
    pub topic_name: String,
    /// Expected `aud` claim on the Google-signed push JWT.
    pub push_audience: String,
    /// Gmail labels to watch — defaults to `["INBOX"]`.
    #[serde(default = "default_inbox_labels")]
    pub watched_label_ids: Vec<String>,
    /// Name of the environment variable that holds the user's Gmail
    /// OAuth 2.0 access token.  Mutually exclusive with `oauth_token`.
    #[serde(default)]
    pub oauth_token_env: Option<String>,
    /// Inline OAuth 2.0 access token (dev/testing only). Prefer
    /// `oauth_token_env` in production so secrets are never written to
    /// disk.
    #[serde(default)]
    pub oauth_token: Option<String>,
}

fn default_inbox_labels() -> Vec<String> {
    vec!["INBOX".to_string()]
}
fn default_smtp_port() -> u16 {
    587
}
fn default_true() -> bool {
    true
}

/// Calendar tool configuration.
///
/// Stored in the `[calendar]` section.  Supports Google Calendar (OAuth2) and
/// any CalDAV-compatible server.  Credentials are resolved from environment
/// variables at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum CalendarSection {
    /// Google Calendar via OAuth2.
    Google {
        /// OAuth2 client ID.
        client_id: String,
        /// OAuth2 client secret.
        client_secret: String,
        /// Name of the environment variable holding the OAuth2 refresh token.
        refresh_token_env: String,
        /// Calendar ID to operate on (default `"primary"`).
        #[serde(default = "default_calendar_id")]
        default_calendar_id: String,
    },
    /// CalDAV-compatible server (Nextcloud, Fastmail, etc.).
    Caldav {
        /// CalDAV server URL.
        url: String,
        /// Authentication username.
        username: String,
        /// Name of the environment variable holding the password.
        password_env: String,
        /// Calendar ID to operate on (default `"primary"`).
        #[serde(default = "default_calendar_id")]
        default_calendar_id: String,
    },
}

fn default_calendar_id() -> String {
    "primary".to_string()
}

/// Browser automation configuration (Thalora).
///
/// Stored in the `[browser]` section.  Thalora must be in `$PATH` (or the path
/// set via `thalora_binary`); tool group `"browser"` must be in `tools.enabled`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserSection {
    /// Path to the `thalora` binary (default `"thalora"`, resolved via `$PATH`).
    pub thalora_binary: String,
    /// Browser session timeout in seconds (default 300).
    pub session_timeout_secs: u64,
}

/// Voice / speech-to-text and text-to-speech configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSection {
    /// STT provider name.
    ///
    /// Supported values: `"openai"`, `"deepgram"`, `"azure"`, `"elevenlabs"`,
    /// `"fish"`, `"whisper-local"` (on-device, no API key required).
    pub stt_provider: String,
    /// Name of the environment variable holding the STT API key.
    ///
    /// If `None`, the provider's default environment variable is checked
    /// (e.g. `OPENAI_API_KEY` for `"openai"`).
    pub api_key_env: Option<String>,
    /// Default language hint passed to the STT provider (ISO-639-1, e.g. `"en"`).
    pub language: Option<String>,

    // ── TTS settings ──────────────────────────────────────────────────────
    /// TTS provider name (optional).
    ///
    /// When set, agent text responses are also synthesised to audio and
    /// attached to outbound messages.
    ///
    /// Supported: `"openai"`, `"elevenlabs"`, `"deepgram"`, `"google"`, `"cartesia"`.
    pub tts_provider: Option<String>,
    /// Voice ID to use for TTS synthesis (provider-specific).
    ///
    /// OpenAI: `"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, `"shimmer"`.
    /// ElevenLabs: voice ID from your account.
    pub tts_voice: Option<String>,
    /// Audio output format for TTS files.
    ///
    /// Supported: `"mp3"` (default), `"opus"`, `"flac"`, `"wav"`.
    pub tts_format: Option<String>,
    /// Directory where TTS audio files are written (default: `/tmp/brainclaw-audio`).
    pub tts_audio_dir: Option<String>,
    /// Public base URL for TTS audio files.
    ///
    /// Defaults to `http://{gateway.host}:{gateway.port}/audio`.
    pub tts_base_url: Option<String>,
}

/// Cross-channel user identity settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IdentitySection {
    /// Enable cross-channel identity mapping.
    ///
    /// When enabled, platform identities can be linked so the same person
    /// on different platforms shares one agent session and conversation history.
    pub enabled: bool,
    /// Path to the identity store JSON file.
    pub store_path: String,
}

impl Default for IdentitySection {
    fn default() -> Self {
        Self {
            enabled: false,
            store_path: "~/.brainclaw/identity.json".to_string(),
        }
    }
}

/// Sandbox configuration — how BrainClaw isolates dangerous tool calls.
///
/// When `enabled`, the built-in tool executor is wrapped in a
/// `SandboxedToolExecutor` that routes `bash` / `execute_command` and
/// `execute_code` / `code_exec` calls through the configured runtime
/// (Docker, Podman, or — dev only — the host).
///
/// See `[sandbox]` in `brainclaw.example.toml` for a fully-annotated sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SandboxConfig {
    /// Whether sandboxing is enabled. Defaults to true — tool calls are
    /// isolated by default.
    pub enabled: bool,
    /// Which runtime to use: `"docker"`, `"podman"`, or `"host"` (dev only,
    /// requires the `sandbox-unsafe-host` build feature).
    pub runtime: String,
    /// Container image to launch. Ignored for the `host` runtime.
    pub image: String,
    /// CPU core limit (e.g. `2.0` = two cores). `None` disables the limit.
    pub cpu_limit: Option<f64>,
    /// Memory cap in megabytes. `None` disables the limit.
    pub memory_limit_mb: Option<u64>,
    /// Max process count inside the sandbox. `None` disables the limit.
    pub pid_limit: Option<u64>,
    /// Network policy: `"none"` (default), `"full"`, or `"limited"`. When
    /// `"limited"`, only hosts in `allowed_hosts` are reachable via the
    /// egress proxy sidecar.
    pub network: String,
    /// Hostnames permitted when `network = "limited"`. Exact or `*.wildcard`.
    pub allowed_hosts: Vec<String>,
    /// Optional workspace directory mounted into the container. If set, the
    /// sandbox's workdir defaults to this path.
    pub workspace_mount: Option<PathBuf>,
    /// Additional host paths allowed as bind-mount sources. Every requested
    /// mount is validated against this list plus `workspace_mount`.
    pub allowed_mount_sources: Vec<PathBuf>,
    /// Container image for the egress proxy sidecar (used when
    /// `network = "limited"`).
    pub proxy_image: String,
    /// TCP port the proxy listens on inside the internal network.
    pub proxy_listen_port: u16,
    /// If set, reuse a named long-lived proxy container across spawns
    /// instead of creating an ephemeral one per sandbox.
    pub proxy_container_name: Option<String>,
    /// Wall-clock timeout applied to sandboxed tool calls.
    pub default_timeout_secs: u64,
    /// If `true`, fall back to the unsandboxed executor when the sandbox
    /// backend can't be constructed (e.g. Docker socket missing). Defaults
    /// to `false` — a broken sandbox is an error, not a silent downgrade.
    pub fallback_to_host_on_error: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            runtime: "docker".to_string(),
            image: "ghcr.io/brainwires/brainclaw-sandbox:latest".to_string(),
            cpu_limit: Some(2.0),
            memory_limit_mb: Some(1024),
            pid_limit: Some(256),
            network: "none".to_string(),
            allowed_hosts: Vec::new(),
            workspace_mount: None,
            allowed_mount_sources: Vec::new(),
            proxy_image: "ghcr.io/brainwires/brainwires-sandbox-proxy:latest".to_string(),
            proxy_listen_port: 3128,
            proxy_container_name: None,
            default_timeout_secs: 300,
            fallback_to_host_on_error: false,
        }
    }
}

#[cfg(feature = "sandbox")]
impl SandboxConfig {
    /// Translate this config into a [`brainwires_sandbox::SandboxPolicy`].
    ///
    /// Returns an error if `runtime` or `network` contains an unknown value.
    pub fn to_policy(&self) -> anyhow::Result<brainwires_sandbox::SandboxPolicy> {
        use brainwires_sandbox::{NetworkPolicy, SandboxPolicy, SandboxRuntime};

        let runtime = match self.runtime.to_lowercase().as_str() {
            "docker" => SandboxRuntime::Docker,
            "podman" => SandboxRuntime::Podman,
            "host" => SandboxRuntime::Host,
            other => {
                anyhow::bail!(
                    "sandbox.runtime '{}' is not recognised; use 'docker', 'podman', or 'host'",
                    other
                );
            }
        };

        let network = match self.network.to_lowercase().as_str() {
            "none" => NetworkPolicy::None,
            "full" => NetworkPolicy::Full,
            "limited" => NetworkPolicy::Limited(self.allowed_hosts.clone()),
            other => {
                anyhow::bail!(
                    "sandbox.network '{}' is not recognised; use 'none', 'full', or 'limited'",
                    other
                );
            }
        };

        Ok(SandboxPolicy {
            runtime,
            image: self.image.clone(),
            network,
            cpu_limit: self.cpu_limit,
            memory_limit_mb: self.memory_limit_mb,
            pid_limit: self.pid_limit,
            read_only_rootfs: true,
            workspace_mount: self.workspace_mount.clone(),
            allowed_mount_sources: self.allowed_mount_sources.clone(),
            proxy_image: self.proxy_image.clone(),
            proxy_listen_port: self.proxy_listen_port,
            proxy_container_name: self.proxy_container_name.clone(),
        })
    }
}

// ── Defaults ────────────────────────────────────────────────────────────

impl Default for GatewaySection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 18789,
            max_connections: 256,
            session_timeout_secs: 3600,
            auth_tokens: Vec::new(),
        }
    }
}

impl Default for ProviderSection {
    fn default() -> Self {
        Self {
            default_provider: "anthropic".to_string(),
            default_model: None,
            api_key: None,
            api_key_env: None,
            temperature: 0.7,
            max_tokens: 16384,
        }
    }
}

impl Default for AgentSection {
    fn default() -> Self {
        Self {
            max_tool_rounds: 10,
            max_concurrent_sessions: 50,
            session_idle_timeout_secs: 1800,
        }
    }
}

impl Default for ToolsSection {
    fn default() -> Self {
        Self {
            enabled: vec![
                "bash".to_string(),
                "files".to_string(),
                "git".to_string(),
                "search".to_string(),
                "web".to_string(),
                "validation".to_string(),
            ],
            disabled: Vec::new(),
            bash_allowed: true,
        }
    }
}

impl Default for PersonaSection {
    fn default() -> Self {
        Self {
            name: "BrainClaw".to_string(),
            system_prompt: None,
            system_prompt_file: None,
            context_files: Vec::new(),
        }
    }
}

impl Default for BrowserSection {
    fn default() -> Self {
        Self {
            thalora_binary: "thalora".to_string(),
            session_timeout_secs: 300,
        }
    }
}

impl Default for MemorySection {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_dir: "~/.brainclaw/memory".to_string(),
            max_history_messages: 100,
            persist_conversations: true,
        }
    }
}

impl Default for CronSection {
    fn default() -> Self {
        Self {
            enabled: false,
            storage_dir: "~/.brainclaw/cron".to_string(),
        }
    }
}

impl Default for SecuritySection {
    fn default() -> Self {
        Self {
            allowed_origins: Vec::new(),
            strip_system_spoofing: true,
            redact_secrets_in_output: true,
            max_messages_per_minute: 20,
            max_tool_calls_per_minute: 30,
            require_signed_skills: false,
            admin_token: None,
            webhook_secret: None,
            channels_enabled: true,
            allowed_channel_types: Vec::new(),
            allowed_channel_ids: Vec::new(),
            require_tool_approval: false,
            approval_tools: Vec::new(),
        }
    }
}

// ── Methods ─────────────────────────────────────────────────────────────

impl BrainClawConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        Self::from_toml_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).context("Failed to parse BrainClaw config")
    }

    /// Try to load configuration from default locations, falling back to defaults.
    ///
    /// Search order:
    /// 1. `~/.brainclaw/brainclaw.toml`
    /// 2. `./brainclaw.toml`
    /// 3. Built-in defaults
    pub fn load_or_default() -> Result<Self> {
        // Try ~/.brainclaw/brainclaw.toml
        if let Some(home_dir) = dirs::home_dir() {
            let home_config = home_dir.join(".brainclaw").join("brainclaw.toml");
            if home_config.exists() {
                tracing::info!(path = %home_config.display(), "Loading config from home directory");
                return Self::load(&home_config);
            }
        }

        // Try ./brainclaw.toml
        let local_config = PathBuf::from("brainclaw.toml");
        if local_config.exists() {
            tracing::info!("Loading config from ./brainclaw.toml");
            return Self::load(&local_config);
        }

        tracing::info!("No config file found, using defaults");
        Ok(Self::default())
    }

    /// Validate the configuration for internal consistency.
    pub fn validate(&self) -> Result<()> {
        // Validate provider name is recognized
        use brainwires_providers::ProviderType;
        let _provider_type: ProviderType = self
            .provider
            .default_provider
            .parse()
            .map_err(|_| anyhow::anyhow!(
                "Unknown provider: '{}'. Valid providers: anthropic, openai, google, groq, ollama, \
                 brainwires, together, fireworks, anyscale, bedrock, vertex-ai",
                self.provider.default_provider
            ))?;

        // Validate temperature range
        if !(0.0..=2.0).contains(&self.provider.temperature) {
            bail!(
                "Temperature must be between 0.0 and 2.0, got {}",
                self.provider.temperature
            );
        }

        // Validate max_tokens is reasonable
        if self.provider.max_tokens == 0 {
            bail!("max_tokens must be greater than 0");
        }

        // Validate port
        if self.gateway.port == 0 {
            bail!("Gateway port must be greater than 0");
        }

        // Validate max_tool_rounds
        if self.agent.max_tool_rounds == 0 {
            bail!("max_tool_rounds must be greater than 0");
        }

        Ok(())
    }

    /// Build an [`EmailConfig`] from the `[email]` section.
    ///
    /// Resolves the password from the environment variable named in `password_env`.
    /// Returns `None` if no `[email]` section is present; returns `Err` if the
    /// env var is missing.
    #[cfg(feature = "email")]
    pub fn to_email_config(&self) -> Option<anyhow::Result<brainwires_tools::EmailConfig>> {
        use brainwires_tools::{EmailConfig, EmailProvider};
        self.email.as_ref().map(|e| {
            let password = std::env::var(&e.password_env).map_err(|_| {
                anyhow::anyhow!("Email password env var '{}' is not set", e.password_env)
            })?;
            Ok(EmailConfig {
                provider: EmailProvider::ImapSmtp {
                    imap_host: e.imap_host.clone(),
                    imap_port: e.imap_port,
                    smtp_host: e.smtp_host.clone(),
                    smtp_port: e.smtp_port,
                    username: e.username.clone(),
                    password,
                    tls: e.tls,
                },
                from_address: e.from_address.clone(),
            })
        })
    }

    /// Build [`GmailPushConfig`]s from the `[gmail_push]` section.
    ///
    /// Returns `None` when the section is disabled or contains no
    /// accounts.  Individual account entries with missing OAuth tokens
    /// are skipped with a warning — the returned list only holds
    /// well-formed configs ready for [`brainwires_tools::gmail_push::GmailPushHandler::new`].
    #[cfg(feature = "email-push")]
    pub fn to_gmail_push_configs(
        &self,
    ) -> Option<Vec<brainwires_tools::gmail_push::GmailPushConfig>> {
        use brainwires_tools::gmail_push::GmailPushConfig;
        if !self.gmail_push.enabled || self.gmail_push.accounts.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(self.gmail_push.accounts.len());
        for acct in &self.gmail_push.accounts {
            let token = match (&acct.oauth_token_env, &acct.oauth_token) {
                (Some(env), _) => match std::env::var(env) {
                    Ok(v) if !v.is_empty() => v,
                    _ => {
                        tracing::warn!(
                            email = %acct.email_address,
                            env = %env,
                            "Gmail push: OAuth token env var is not set; skipping account"
                        );
                        continue;
                    }
                },
                (None, Some(inline)) if !inline.is_empty() => inline.clone(),
                _ => {
                    tracing::warn!(
                        email = %acct.email_address,
                        "Gmail push: account has no oauth_token_env or oauth_token; skipping"
                    );
                    continue;
                }
            };
            let label_ids = if acct.watched_label_ids.is_empty() {
                vec!["INBOX".to_string()]
            } else {
                acct.watched_label_ids.clone()
            };
            out.push(GmailPushConfig {
                project_id: acct.project_id.clone(),
                topic_name: acct.topic_name.clone(),
                push_audience: acct.push_audience.clone(),
                watched_label_ids: label_ids,
                oauth_token: token,
                gmail_base_url: None,
            });
        }
        if out.is_empty() { None } else { Some(out) }
    }

    /// Build a [`CalendarConfig`] from the `[calendar]` section.
    ///
    /// Credentials are resolved from environment variables at runtime.
    /// Returns `None` if no `[calendar]` section is present.
    #[cfg(feature = "calendar")]
    pub fn to_calendar_config(
        &self,
    ) -> Option<anyhow::Result<brainwires_tools::calendar::CalendarConfig>> {
        use brainwires_tools::calendar::{CalendarConfig, CalendarProvider};
        self.calendar.as_ref().map(|c| match c {
            CalendarSection::Google {
                client_id,
                client_secret,
                refresh_token_env,
                default_calendar_id,
            } => {
                let refresh_token = std::env::var(refresh_token_env).map_err(|_| {
                    anyhow::anyhow!(
                        "Google Calendar refresh token env var '{}' is not set",
                        refresh_token_env
                    )
                })?;
                Ok(CalendarConfig {
                    provider: CalendarProvider::GoogleCalendar {
                        client_id: client_id.clone(),
                        client_secret: client_secret.clone(),
                        refresh_token,
                    },
                    default_calendar_id: default_calendar_id.clone(),
                })
            }
            CalendarSection::Caldav {
                url,
                username,
                password_env,
                default_calendar_id,
            } => {
                let password = std::env::var(password_env).map_err(|_| {
                    anyhow::anyhow!("CalDAV password env var '{}' is not set", password_env)
                })?;
                Ok(CalendarConfig {
                    provider: CalendarProvider::CalDav {
                        url: url.clone(),
                        username: username.clone(),
                        password,
                    },
                    default_calendar_id: default_calendar_id.clone(),
                })
            }
        })
    }

    /// Resolve the webchat JWT secret. Precedence:
    ///
    /// 1. Explicit `[webchat] jwt_secret` from config.
    /// 2. A secret derived deterministically from `security.admin_token`
    ///    so that webchat tokens survive daemon restarts without extra
    ///    setup. The derivation is `sha256("brainclaw-webchat:" + admin)`
    ///    hex-encoded, keeping the admin token itself off disk and out
    ///    of log messages.
    /// 3. `None` — callers must then generate their own.
    pub fn resolve_webchat_secret(&self) -> Option<String> {
        if let Some(s) = &self.webchat.jwt_secret
            && !s.is_empty()
        {
            return Some(s.clone());
        }
        if let Some(admin) = &self.security.admin_token
            && !admin.is_empty()
        {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"brainclaw-webchat:");
            h.update(admin.as_bytes());
            return Some(hex::encode(h.finalize()));
        }
        None
    }

    /// Convert to a [`GatewayConfig`] for the gateway server.
    pub fn to_gateway_config(&self) -> GatewayConfig {
        GatewayConfig {
            host: self.gateway.host.clone(),
            port: self.gateway.port,
            max_connections: self.gateway.max_connections,
            session_timeout: Duration::from_secs(self.gateway.session_timeout_secs),
            auth_tokens: self.gateway.auth_tokens.clone(),
            webhook_enabled: true,
            webhook_path: "/webhook".to_string(),
            admin_enabled: true,
            admin_path: "/admin".to_string(),
            allowed_origins: self.security.allowed_origins.clone(),
            strip_system_spoofing: self.security.strip_system_spoofing,
            redact_secrets_in_output: self.security.redact_secrets_in_output,
            max_messages_per_minute: self.security.max_messages_per_minute,
            max_tool_calls_per_minute: self.security.max_tool_calls_per_minute,
            webchat_enabled: self.webchat.enabled,
            webchat_jwt_secret: self.resolve_webchat_secret(),
            webchat_session_history_limit: self.webchat.session_history_limit,
            max_attachment_size_mb: 10,
            admin_token: self.security.admin_token.clone(),
            webhook_secret: self.security.webhook_secret.clone(),
            channels_enabled: self.security.channels_enabled,
            allowed_channel_types: self.security.allowed_channel_types.clone(),
            allowed_channel_ids: self.security.allowed_channel_ids.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_expected_values() {
        let config = BrainClawConfig::default();
        assert_eq!(config.gateway.host, "127.0.0.1");
        assert_eq!(config.gateway.port, 18789);
        assert_eq!(config.gateway.max_connections, 256);
        assert_eq!(config.provider.default_provider, "anthropic");
        assert!(config.provider.default_model.is_none());
        assert!(config.provider.api_key.is_none());
        assert_eq!(config.provider.temperature, 0.7);
        assert_eq!(config.provider.max_tokens, 16384);
        assert_eq!(config.agent.max_tool_rounds, 10);
        assert_eq!(config.agent.max_concurrent_sessions, 50);
        assert_eq!(config.tools.enabled.len(), 6);
        assert!(config.tools.bash_allowed);
        assert_eq!(config.persona.name, "BrainClaw");
        assert!(config.persona.system_prompt.is_none());
        assert!(config.memory.enabled);
        assert_eq!(config.memory.max_history_messages, 100);
        assert!(!config.skills.enabled);
        assert!(config.security.strip_system_spoofing);
        assert!(config.security.redact_secrets_in_output);
        assert_eq!(config.security.max_messages_per_minute, 20);
        assert_eq!(config.security.max_tool_calls_per_minute, 30);
        assert!(!config.security.require_signed_skills);
    }

    #[test]
    fn test_load_from_toml_string() {
        let toml_str = r#"
[gateway]
host = "0.0.0.0"
port = 9090

[provider]
default_provider = "openai"
default_model = "gpt-4o"
temperature = 0.5
max_tokens = 8192

[agent]
max_tool_rounds = 5

[tools]
enabled = ["bash", "files"]
bash_allowed = false

[persona]
name = "TestBot"
system_prompt = "You are a test bot."

[memory]
enabled = false
max_history_messages = 50

[skills]
enabled = true
directories = ["/home/user/skills"]

[security]
allowed_origins = ["https://example.com"]
max_messages_per_minute = 10
require_signed_skills = true
"#;

        let config = BrainClawConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.gateway.host, "0.0.0.0");
        assert_eq!(config.gateway.port, 9090);
        assert_eq!(config.provider.default_provider, "openai");
        assert_eq!(config.provider.default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(config.provider.temperature, 0.5);
        assert_eq!(config.provider.max_tokens, 8192);
        assert_eq!(config.agent.max_tool_rounds, 5);
        assert_eq!(config.tools.enabled, vec!["bash", "files"]);
        assert!(!config.tools.bash_allowed);
        assert_eq!(config.persona.name, "TestBot");
        assert_eq!(
            config.persona.system_prompt.as_deref(),
            Some("You are a test bot.")
        );
        assert!(!config.memory.enabled);
        assert_eq!(config.memory.max_history_messages, 50);
        assert!(config.skills.enabled);
        assert_eq!(config.skills.directories, vec!["/home/user/skills"]);
        assert_eq!(config.security.allowed_origins, vec!["https://example.com"]);
        assert_eq!(config.security.max_messages_per_minute, 10);
        assert!(config.security.require_signed_skills);
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[provider]
default_provider = "groq"
"#;

        let config = BrainClawConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.provider.default_provider, "groq");
        // Everything else should be defaults
        assert_eq!(config.gateway.host, "127.0.0.1");
        assert_eq!(config.gateway.port, 18789);
        assert_eq!(config.agent.max_tool_rounds, 10);
        assert_eq!(config.persona.name, "BrainClaw");
    }

    #[test]
    fn test_validate_valid_config() {
        let config = BrainClawConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_unknown_provider() {
        let mut config = BrainClawConfig::default();
        config.provider.default_provider = "nonexistent".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown provider"));
    }

    #[test]
    fn test_validate_bad_temperature() {
        let mut config = BrainClawConfig::default();
        config.provider.temperature = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_max_tokens() {
        let mut config = BrainClawConfig::default();
        config.provider.max_tokens = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_port() {
        let mut config = BrainClawConfig::default();
        config.gateway.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_tool_rounds() {
        let mut config = BrainClawConfig::default();
        config.agent.max_tool_rounds = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_to_gateway_config() {
        let mut config = BrainClawConfig::default();
        config.gateway.host = "0.0.0.0".to_string();
        config.gateway.port = 9999;
        config.gateway.max_connections = 128;
        config.gateway.session_timeout_secs = 7200;
        config.security.allowed_origins = vec!["https://example.com".to_string()];
        config.security.max_messages_per_minute = 15;

        let gw = config.to_gateway_config();
        assert_eq!(gw.host, "0.0.0.0");
        assert_eq!(gw.port, 9999);
        assert_eq!(gw.max_connections, 128);
        assert_eq!(gw.session_timeout, Duration::from_secs(7200));
        assert_eq!(gw.allowed_origins, vec!["https://example.com"]);
        assert_eq!(gw.max_messages_per_minute, 15);
        assert!(gw.strip_system_spoofing);
        assert!(gw.redact_secrets_in_output);
    }

    #[test]
    fn test_empty_toml_uses_all_defaults() {
        let config = BrainClawConfig::from_toml_str("").unwrap();
        let default = BrainClawConfig::default();
        assert_eq!(config.gateway.host, default.gateway.host);
        assert_eq!(config.gateway.port, default.gateway.port);
        assert_eq!(
            config.provider.default_provider,
            default.provider.default_provider
        );
        assert_eq!(config.persona.name, default.persona.name);
    }
}
