use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::Zeroizing;

static STALE_MODEL_WARNED: AtomicBool = AtomicBool::new(false);

use super::paths::PlatformPaths;
use crate::types::agent::PermissionMode;
use crate::types::provider::ProviderType;
use brainwires::agent_network::auth::keyring::KeyringKeyStore;
use brainwires::agent_network::traits::KeyStore;
use brainwires::knowledge::bks_pks::KnowledgeSettings as KnowledgeSettingsCore;
use brainwires_seal::SealConfig;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Active provider type (default: Brainwires SaaS)
    /// Use `brainwires auth login --provider <name>` to change
    #[serde(default = "default_provider_type", alias = "provider")]
    pub provider_type: ProviderType,

    /// Model name to use
    #[serde(default = "default_model")]
    pub model: String,

    /// Permission mode for tool execution
    #[serde(default)]
    pub permission_mode: PermissionMode,

    /// Backend URL for Brainwires Studio
    #[serde(default = "default_backend_url")]
    pub backend_url: String,

    /// Base URL override for the active provider (e.g. Ollama endpoint, custom OpenAI-compat URL)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_base_url: Option<String>,

    /// Temperature for AI responses
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Additional provider-specific settings
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,

    /// SEAL (Self-Evolving Agentic Learning) configuration
    #[serde(default)]
    pub seal: SealSettings,

    /// SEAL + Knowledge System Integration configuration
    #[serde(default)]
    pub seal_knowledge: SealKnowledgeSettings,

    /// Behavioral Knowledge System configuration
    #[serde(default)]
    pub knowledge: KnowledgeSettings,

    /// Remote Control Bridge configuration
    #[serde(default)]
    pub remote: RemoteSettings,

    /// Local LLM configuration for CPU-based inference
    #[serde(default)]
    pub local_llm: LocalLlmSettings,

    /// Optional shell command whose stdout is appended to the TUI status
    /// line. Executed via `bash -c` with a short timeout; its output is
    /// cached between renders so it never slows the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_line_command: Option<String>,
}

/// SEAL (Self-Evolving Agentic Learning) settings
///
/// Controls enhanced context understanding features including:
/// - Coreference resolution ("it" → "main.rs")
/// - Query core extraction (structured query understanding)
/// - Self-evolving learning from successful interactions
/// - Reflection and error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealSettings {
    /// Enable SEAL processing (default: true)
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Enable coreference resolution - resolves "it", "the file", etc. (default: true)
    #[serde(default = "default_true")]
    pub enable_coreference: bool,

    /// Enable query core extraction for structured understanding (default: true)
    #[serde(default = "default_true")]
    pub enable_query_cores: bool,

    /// Enable self-evolving learning from successful interactions (default: true)
    #[serde(default = "default_true")]
    pub enable_learning: bool,

    /// Enable reflection module for error detection (default: false - adds latency)
    #[serde(default)]
    pub enable_reflection: bool,

    /// Minimum confidence for coreference resolution (0.0-1.0, default: 0.5)
    #[serde(default = "default_coreference_confidence")]
    pub min_coreference_confidence: f32,

    /// Minimum pattern reliability for learning (0.0-1.0, default: 0.7)
    #[serde(default = "default_pattern_reliability")]
    pub min_pattern_reliability: f32,

    /// Show SEAL processing status in UI (default: true)
    #[serde(default = "default_true")]
    pub show_status: bool,
}

fn default_true() -> bool {
    true
}

fn default_coreference_confidence() -> f32 {
    0.5
}

fn default_pattern_reliability() -> f32 {
    0.7
}

impl Default for SealSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_coreference: true,
            enable_query_cores: true,
            enable_learning: true,
            enable_reflection: false, // Off by default - adds latency
            min_coreference_confidence: 0.5,
            min_pattern_reliability: 0.7,
            show_status: true,
        }
    }
}

impl SealSettings {
    /// Convert to SealConfig for use with SealProcessor
    pub fn to_seal_config(&self) -> SealConfig {
        SealConfig {
            enable_coreference: self.enable_coreference,
            enable_query_cores: self.enable_query_cores,
            enable_learning: self.enable_learning,
            enable_reflection: self.enable_reflection,
            max_reflection_retries: 2,
            min_coreference_confidence: self.min_coreference_confidence,
            min_pattern_reliability: self.min_pattern_reliability,
        }
    }
}

/// Behavioral Knowledge System (BKS) settings
///
/// Controls the collective learning system that discovers and shares
/// universal behavioral truths across all clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSettings {
    /// Enable the knowledge system (default: true)
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Enable explicit learning via /learn command (default: true)
    #[serde(default = "default_true")]
    pub enable_explicit_learning: bool,

    /// Enable implicit learning from conversation corrections (default: true)
    #[serde(default = "default_true")]
    pub enable_implicit_learning: bool,

    /// Enable aggressive learning from success/failure patterns (default: true)
    #[serde(default = "default_true")]
    pub enable_aggressive_learning: bool,

    /// Minimum confidence to inject truths into prompt (default: 0.5)
    #[serde(default = "default_min_confidence_apply")]
    pub min_confidence_to_apply: f32,

    /// Minimum confidence to prompt about conflicts (default: 0.7)
    #[serde(default = "default_min_confidence_prompt")]
    pub min_confidence_to_prompt: f32,

    /// Number of failures before detecting a pattern (default: 3)
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,

    /// EMA decay factor for confidence updates (default: 0.1)
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: f32,

    /// Days of non-use before decay starts (default: 30)
    #[serde(default = "default_decay_days")]
    pub decay_days: u32,

    /// How often to sync with server in seconds (default: 300)
    #[serde(default = "default_sync_interval")]
    pub sync_interval_secs: u64,

    /// Maximum queued submissions for offline mode (default: 100)
    #[serde(default = "default_offline_queue_size")]
    pub offline_queue_size: usize,

    /// Show when truths are applied to prompts (default: true)
    #[serde(default = "default_true")]
    pub show_applied_truths: bool,

    /// Ask user about conflicts between truths and instructions (default: true)
    #[serde(default = "default_true")]
    pub show_conflict_prompts: bool,
}

fn default_min_confidence_apply() -> f32 {
    0.5
}

fn default_min_confidence_prompt() -> f32 {
    0.7
}

fn default_failure_threshold() -> u32 {
    3
}

fn default_ema_alpha() -> f32 {
    0.1
}

fn default_decay_days() -> u32 {
    30
}

fn default_sync_interval() -> u64 {
    300
}

fn default_offline_queue_size() -> usize {
    100
}

impl Default for KnowledgeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_explicit_learning: true,
            enable_implicit_learning: true,
            enable_aggressive_learning: true,
            min_confidence_to_apply: 0.5,
            min_confidence_to_prompt: 0.7,
            failure_threshold: 3,
            ema_alpha: 0.1,
            decay_days: 30,
            sync_interval_secs: 300,
            offline_queue_size: 100,
            show_applied_truths: true,
            show_conflict_prompts: true,
        }
    }
}

impl KnowledgeSettings {
    /// Convert to KnowledgeSettingsCore for use with the knowledge module
    pub fn to_core_settings(&self) -> KnowledgeSettingsCore {
        KnowledgeSettingsCore {
            enabled: self.enabled,
            enable_explicit_learning: self.enable_explicit_learning,
            enable_implicit_learning: self.enable_implicit_learning,
            enable_aggressive_learning: self.enable_aggressive_learning,
            min_confidence_to_apply: self.min_confidence_to_apply,
            min_confidence_to_prompt: self.min_confidence_to_prompt,
            failure_threshold: self.failure_threshold,
            ema_alpha: self.ema_alpha,
            decay_days: self.decay_days,
            sync_interval_secs: self.sync_interval_secs,
            offline_queue_size: self.offline_queue_size,
            show_applied_truths: self.show_applied_truths,
            show_conflict_prompts: self.show_conflict_prompts,
        }
    }
}

/// SEAL + Knowledge System Integration settings
///
/// Controls the bidirectional learning between SEAL (entity-centric) and
/// the Knowledge System (behavioral truths + personal facts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealKnowledgeSettings {
    /// Enable SEAL + Knowledge integration (default: true)
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Enable SEAL patterns → BKS promotion (default: true)
    #[serde(default = "default_true")]
    pub seal_to_knowledge: bool,

    /// Enable BKS truths → SEAL pattern loading (default: true)
    #[serde(default = "default_true")]
    pub knowledge_to_seal: bool,

    /// Minimum SEAL quality score for BKS boost (default: 0.7)
    #[serde(default = "default_seal_quality_bks")]
    pub min_seal_quality_for_bks_boost: f32,

    /// Minimum SEAL quality score for PKS boost (default: 0.5)
    #[serde(default = "default_seal_quality_pks")]
    pub min_seal_quality_for_pks_boost: f32,

    /// Pattern reliability threshold for BKS promotion (default: 0.8)
    #[serde(default = "default_promotion_threshold")]
    pub pattern_promotion_threshold: f32,

    /// Minimum pattern uses before promotion (default: 5)
    #[serde(default = "default_min_pattern_uses")]
    pub min_pattern_uses: u32,

    /// Cache BKS truths in SEAL's global memory (default: true)
    #[serde(default = "default_true")]
    pub cache_bks_in_seal: bool,

    /// Entity resolution strategy: "seal_first", "pks_first", or "hybrid" (default: "hybrid")
    #[serde(default = "default_resolution_strategy")]
    pub entity_resolution_strategy: String,

    /// SEAL weight in confidence harmonization (default: 0.5)
    #[serde(default = "default_seal_weight")]
    pub seal_weight: f32,

    /// BKS weight in confidence harmonization (default: 0.3)
    #[serde(default = "default_bks_weight")]
    pub bks_weight: f32,

    /// PKS weight in confidence harmonization (default: 0.2)
    #[serde(default = "default_pks_weight")]
    pub pks_weight: f32,
}

fn default_seal_quality_bks() -> f32 {
    0.7
}

fn default_seal_quality_pks() -> f32 {
    0.5
}

fn default_promotion_threshold() -> f32 {
    0.8
}

fn default_min_pattern_uses() -> u32 {
    5
}

fn default_resolution_strategy() -> String {
    "hybrid".to_string()
}

fn default_seal_weight() -> f32 {
    0.5
}

fn default_bks_weight() -> f32 {
    0.3
}

fn default_pks_weight() -> f32 {
    0.2
}

impl Default for SealKnowledgeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            seal_to_knowledge: true,
            knowledge_to_seal: true,
            min_seal_quality_for_bks_boost: 0.7,
            min_seal_quality_for_pks_boost: 0.5,
            pattern_promotion_threshold: 0.8,
            min_pattern_uses: 5,
            cache_bks_in_seal: true,
            entity_resolution_strategy: "hybrid".to_string(),
            seal_weight: 0.5,
            bks_weight: 0.3,
            pks_weight: 0.2,
        }
    }
}

impl SealKnowledgeSettings {
    /// Convert to IntegrationConfig for use with SealKnowledgeCoordinator
    pub fn to_integration_config(&self) -> brainwires_seal::IntegrationConfig {
        use brainwires_seal::EntityResolutionStrategy;

        let strategy = match self.entity_resolution_strategy.as_str() {
            "seal_first" => EntityResolutionStrategy::SealFirst,
            "pks_first" => EntityResolutionStrategy::PksContextFirst,
            _ => EntityResolutionStrategy::Hybrid {
                seal_weight: 0.6,
                pks_weight: 0.4,
            },
        };

        brainwires_seal::IntegrationConfig {
            enabled: self.enabled,
            seal_to_knowledge: self.seal_to_knowledge,
            knowledge_to_seal: self.knowledge_to_seal,
            min_seal_quality_for_bks_boost: self.min_seal_quality_for_bks_boost,
            min_seal_quality_for_pks_boost: self.min_seal_quality_for_pks_boost,
            pattern_promotion_threshold: self.pattern_promotion_threshold,
            min_pattern_uses: self.min_pattern_uses,
            cache_bks_in_seal: self.cache_bks_in_seal,
            entity_resolution_strategy: strategy,
            seal_weight: self.seal_weight,
            bks_weight: self.bks_weight,
            pks_weight: self.pks_weight,
        }
    }
}

/// Remote Control Bridge settings
///
/// Controls the HTTP polling connection to brainwires-studio
/// for remote monitoring and control of CLI agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSettings {
    /// Enable remote control bridge (default: false)
    #[serde(default)]
    pub enabled: bool,

    /// Backend base URL for remote connections
    /// Default: https://brainwires.studio
    #[serde(default = "default_remote_url")]
    pub backend_url: String,

    /// API key for remote authentication (optional, uses session API key if not set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Heartbeat interval in seconds (default: 30)
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u32,

    /// Reconnect delay in seconds after disconnect (default: 5)
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_secs: u32,

    /// Maximum reconnect attempts (0 = unlimited, default: 0)
    #[serde(default)]
    pub max_reconnect_attempts: u32,

    /// Auto-start bridge when agents are running (default: true when enabled)
    #[serde(default = "default_true")]
    pub auto_start: bool,

    /// Commands blocked from remote execution (default: ["exec"])
    /// Set to empty array to allow all commands remotely
    #[serde(default = "default_blocked_remote_commands")]
    pub blocked_remote_commands: Vec<String>,

    /// Commands that trigger warnings but are still allowed (default: ["exit"])
    #[serde(default = "default_warned_remote_commands")]
    pub warned_remote_commands: Vec<String>,
}

fn default_blocked_remote_commands() -> Vec<String> {
    vec!["exec".to_string()]
}

fn default_warned_remote_commands() -> Vec<String> {
    vec!["exit".to_string()]
}

fn default_remote_url() -> String {
    "https://brainwires.studio".to_string()
}

fn default_heartbeat_interval() -> u32 {
    30
}

fn default_reconnect_delay() -> u32 {
    5
}

impl Default for RemoteSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            backend_url: default_remote_url(),
            api_key: None,
            heartbeat_interval_secs: 30,
            reconnect_delay_secs: 5,
            max_reconnect_attempts: 0,
            auto_start: true,
            blocked_remote_commands: default_blocked_remote_commands(),
            warned_remote_commands: default_warned_remote_commands(),
        }
    }
}

impl RemoteSettings {
    /// Check if a command is blocked from remote execution
    pub fn is_command_blocked(&self, command: &str) -> bool {
        self.blocked_remote_commands
            .iter()
            .any(|c| c.eq_ignore_ascii_case(command))
    }

    /// Check if a command triggers a warning for remote execution
    pub fn is_command_warned(&self, command: &str) -> bool {
        self.warned_remote_commands
            .iter()
            .any(|c| c.eq_ignore_ascii_case(command))
    }
}

impl RemoteSettings {
    /// Auto-select the backend URL based on API key prefix.
    ///
    /// If no explicit backend_url is configured (i.e., using the default), this
    /// will select the appropriate backend based on the API key prefix:
    /// - `bw_dev_*` keys connect to dev.brainwires.net
    /// - `bw_prod_*` keys (or any other) connect to brainwires.studio
    pub fn auto_select_backend_url(&self, api_key: &str) -> String {
        if self.backend_url == default_remote_url() {
            get_remote_url_for_api_key(api_key)
        } else {
            self.backend_url.clone()
        }
    }
}

/// Local LLM settings for CPU-based inference
///
/// Controls local model inference for context processing, routing decisions,
/// and other tasks that don't require a foundation model API.
/// Optimized for high-throughput, low-latency CPU inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLlmSettings {
    /// Enable local LLM processing (default: false)
    #[serde(default)]
    pub enabled: bool,

    /// Default model ID to use (must be registered in local model registry)
    #[serde(default)]
    pub default_model: Option<String>,

    /// Directory to store/find local models
    #[serde(default = "default_local_models_dir")]
    pub models_dir: std::path::PathBuf,

    /// Number of CPU threads for inference (default: auto-detect)
    #[serde(default)]
    pub num_threads: Option<u32>,

    /// Context window size override (default: use model's setting)
    #[serde(default)]
    pub context_size: Option<u32>,

    /// Maximum tokens per response (default: use model's setting)
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Number of parallel model instances to run (for high throughput)
    /// Each instance uses separate memory. Default: 1
    #[serde(default = "default_one")]
    pub pool_size: u32,

    /// Use local LLM for query routing decisions (default: false)
    #[serde(default)]
    pub use_for_routing: bool,

    /// Use local LLM for context summarization (default: false)
    #[serde(default)]
    pub use_for_summarization: bool,

    /// Use local LLM for semantic analysis (default: false)
    #[serde(default)]
    pub use_for_analysis: bool,

    /// Use local LLM for response validation/red-flagging (default: false)
    #[serde(default)]
    pub use_for_validation: bool,

    /// Use local LLM for task complexity scoring in MDAP (default: false)
    #[serde(default)]
    pub use_for_complexity: bool,

    /// Model ID to use for routing (fast model preferred, default: lfm2-350m)
    #[serde(default)]
    pub routing_model: Option<String>,

    /// Model ID to use for validation (fast model preferred, default: lfm2-350m)
    #[serde(default)]
    pub validation_model: Option<String>,

    /// Model ID to use for complexity scoring (fast model preferred, default: lfm2-350m)
    #[serde(default)]
    pub complexity_model: Option<String>,

    /// Model ID to use for microagent execution (default: lfm2-1.2b)
    #[serde(default)]
    pub microagent_model: Option<String>,

    /// Preload model on startup (default: false - lazy load on first use)
    #[serde(default)]
    pub preload_on_startup: bool,

    /// Log all local inference calls for debugging (default: true)
    #[serde(default = "default_true")]
    pub log_inference: bool,
}

fn default_local_models_dir() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("brainwires")
        .join("models")
}

fn default_one() -> u32 {
    1
}

impl Default for LocalLlmSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            default_model: None,
            models_dir: default_local_models_dir(),
            num_threads: None,
            context_size: None,
            max_tokens: None,
            pool_size: 1,
            use_for_routing: false,
            use_for_summarization: false,
            use_for_analysis: false,
            use_for_validation: false,
            use_for_complexity: false,
            routing_model: None,
            validation_model: None,
            complexity_model: None,
            microagent_model: None,
            preload_on_startup: false,
            log_inference: true,
        }
    }
}

/// Get the appropriate remote backend URL based on API key prefix
fn get_remote_url_for_api_key(api_key: &str) -> String {
    if api_key.starts_with("bw_dev_") {
        "https://dev.brainwires.net".to_string()
    } else {
        default_remote_url()
    }
}

fn default_provider_type() -> ProviderType {
    ProviderType::Brainwires
}

pub(crate) fn default_model() -> String {
    // A first-run default must be a model the backend actually advertises in
    // `brainwires models list`. The previous default "gpt-5-mini" did not
    // appear in that list (the closest real model is "openai-gpt-5-mini"),
    // which produced silent request failures for fresh installs.
    //
    // Claude Haiku 4.5 is small, cheap, supported on the Brainwires SaaS
    // relay, and widely available to users who are just exploring the CLI.
    "claude-haiku-4-5-20251001".to_string()
}

pub(crate) fn default_backend_url() -> String {
    crate::config::constants::DEFAULT_BACKEND_URL.to_string()
}

pub(crate) fn default_temperature() -> f32 {
    0.7
}

pub(crate) fn default_max_tokens() -> u32 {
    4096
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider_type: default_provider_type(),
            model: default_model(),
            permission_mode: PermissionMode::default(),
            backend_url: default_backend_url(),
            provider_base_url: None,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            extra: std::collections::HashMap::new(),
            seal: SealSettings::default(),
            seal_knowledge: SealKnowledgeSettings::default(),
            knowledge: KnowledgeSettings::default(),
            remote: RemoteSettings::default(),
            local_llm: LocalLlmSettings::default(),
            status_line_command: None,
        }
    }
}

/// Configuration manager
pub struct ConfigManager {
    config: Config,
    config_path: PathBuf,
    /// `true` when the config file did not exist on load — indicates a fresh
    /// install that should trigger the first-run provider picker.
    is_new: bool,
}

impl ConfigManager {
    /// Create a new config manager
    pub fn new() -> Result<Self> {
        PlatformPaths::ensure_config_dir()?;
        let config_path = PlatformPaths::config_file()?;

        let existed = config_path.exists();
        let config = if existed {
            Self::load_from_file(&config_path)?
        } else {
            Config::default()
        };

        Ok(Self {
            config,
            config_path,
            is_new: !existed,
        })
    }

    /// Whether the config file did not exist when this manager loaded.
    ///
    /// Used by the CLI to decide whether to show the first-run provider
    /// picker. Once the user saves any config, this does not flip back —
    /// callers that care should check before calling `save()`.
    pub fn is_first_run(&self) -> bool {
        self.is_new
    }

    /// Load configuration from file
    pub(crate) fn load_from_file(path: &PathBuf) -> Result<Config> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let mut config: Config =
            serde_json::from_str(&contents).context("Failed to parse config file")?;

        // Migration: pre-v0.11 installs may still pin stale/phantom model names
        // as the value of `model`. Remap in-memory only — the user's config.json
        // stays untouched until they run `config --set`, which is the expected
        // place for persistent changes.
        const STALE_MODELS: &[&str] = &["openai-gpt-5.2", "gpt-5-mini"];
        if STALE_MODELS.contains(&config.model.as_str()) {
            let fresh = default_model();
            if !STALE_MODEL_WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "⚠ Config pins a stale model ('{}'). Using '{}' for this session. \
                     Run `brainwires config --set model=<name>` to persist a choice.",
                    config.model, fresh
                );
            }
            config.model = fresh;
        }

        Ok(config)
    }

    /// Get the current configuration
    pub fn get(&self) -> &Config {
        &self.config
    }

    /// Get a mutable reference to the configuration
    pub fn get_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Update configuration values
    pub fn update(&mut self, updates: ConfigUpdates) {
        if let Some(provider_type) = updates.provider_type {
            self.config.provider_type = provider_type;
        }
        if let Some(model) = updates.model {
            self.config.model = model;
        }
        if let Some(permission_mode) = updates.permission_mode {
            self.config.permission_mode = permission_mode;
        }
        if let Some(backend_url) = updates.backend_url {
            self.config.backend_url = backend_url;
        }
        if let Some(provider_base_url) = updates.provider_base_url {
            self.config.provider_base_url = provider_base_url;
        }
        if let Some(temperature) = updates.temperature {
            self.config.temperature = temperature;
        }
        if let Some(max_tokens) = updates.max_tokens {
            self.config.max_tokens = max_tokens;
        }
        if let Some(seal) = updates.seal {
            self.config.seal = seal;
        }
        if let Some(knowledge) = updates.knowledge {
            self.config.knowledge = knowledge;
        }
        if let Some(remote) = updates.remote {
            self.config.remote = remote;
        }
        if let Some(local_llm) = updates.local_llm {
            self.config.local_llm = local_llm;
        }
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        PlatformPaths::ensure_config_dir()?;

        let contents =
            serde_json::to_string_pretty(&self.config).context("Failed to serialize config")?;

        fs::write(&self.config_path, contents).with_context(|| {
            format!(
                "Failed to write config file: {}",
                self.config_path.display()
            )
        })?;

        Ok(())
    }

    /// Get API key for the current provider from the system keyring.
    ///
    /// - `Brainwires` → delegates to `SessionManager::get_api_key()`
    /// - `Ollama` → returns `Ok(None)` (no key needed)
    /// - Others → reads from keyring account `provider:{name}`
    pub fn get_provider_api_key(&self) -> Result<Option<Zeroizing<String>>> {
        match self.config.provider_type {
            ProviderType::Brainwires => {
                // Delegate to session manager
                crate::auth::SessionManager::get_api_key()
            }
            // These providers use their own credential chains, not API keys
            ProviderType::Ollama | ProviderType::Bedrock | ProviderType::VertexAI => Ok(None),
            _ => {
                let account = format!("provider:{}", self.config.provider_type.as_str());
                let store = KeyringKeyStore::new();
                store.get_key(&account)
            }
        }
    }

    /// Get API key for a specific provider from the system keyring.
    ///
    /// Unlike `get_provider_api_key()`, this takes an explicit provider type
    /// rather than using the active provider from config.
    pub fn get_provider_api_key_for(
        &self,
        provider: ProviderType,
    ) -> Result<Option<Zeroizing<String>>> {
        match provider {
            ProviderType::Brainwires => crate::auth::SessionManager::get_api_key(),
            ProviderType::Ollama | ProviderType::Bedrock | ProviderType::VertexAI => Ok(None),
            _ => {
                let account = format!("provider:{}", provider.as_str());
                let store = KeyringKeyStore::new();
                store.get_key(&account)
            }
        }
    }

    /// Store an API key for a provider in the system keyring.
    pub fn set_provider_api_key(&self, provider: ProviderType, key: &str) -> Result<()> {
        let account = format!("provider:{}", provider.as_str());
        let store = KeyringKeyStore::new();
        store.store_key(&account, key)
    }

    /// Delete the API key for a provider from the system keyring.
    pub fn delete_provider_api_key(&self, provider: ProviderType) -> Result<()> {
        let account = format!("provider:{}", provider.as_str());
        let store = KeyringKeyStore::new();
        store.delete_key(&account)
    }
}

/// Configuration updates
#[derive(Debug, Default)]
pub struct ConfigUpdates {
    pub provider_type: Option<ProviderType>,
    pub model: Option<String>,
    pub permission_mode: Option<PermissionMode>,
    pub backend_url: Option<String>,
    /// Double-option: `Some(None)` clears the field, `Some(Some(url))` sets it, `None` leaves it unchanged
    pub provider_base_url: Option<Option<String>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seal: Option<SealSettings>,
    pub knowledge: Option<KnowledgeSettings>,
    pub remote: Option<RemoteSettings>,
    pub local_llm: Option<LocalLlmSettings>,
}

#[cfg(test)]
mod tests;
