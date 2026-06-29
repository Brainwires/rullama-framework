use serde::{Deserialize, Serialize};

/// Voice assistant configuration file (TOML).
///
/// Default path: `~/.config/voice-assistant/config.toml`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VaConfig {
    /// OpenAI API key (or set via `OPENAI_API_KEY` env var).
    pub openai_api_key: Option<String>,

    /// STT model name. Default: "whisper-1".
    pub stt_model: String,

    /// TTS model name. Default: "tts-1".
    pub tts_model: String,

    /// TTS voice. Default: "alloy".
    pub tts_voice: String,

    /// Path to a rustpotter `.rpw` wake word model file.
    /// If not set, the assistant listens continuously.
    pub wake_word_model: Option<String>,

    /// Wake word detection threshold (0.0–1.0). Default: 0.5.
    pub wake_word_threshold: f32,

    /// Silence threshold in dBFS. Default: -40.0.
    pub silence_threshold_db: f32,

    /// Milliseconds of silence that end an utterance. Default: 800.
    pub silence_duration_ms: u32,

    /// Maximum recording duration (seconds). Default: 30.
    pub max_record_secs: f64,

    /// Name of the microphone device to use. `None` = system default.
    pub microphone: Option<String>,

    /// Name of the speaker device to use. `None` = system default.
    pub speaker: Option<String>,

    /// System prompt for the LLM. Default: a helpful assistant persona.
    pub system_prompt: String,

    /// LLM model. Default: "gpt-4o-mini".
    pub llm_model: String,

    /// Whether to speak responses via TTS. Default: true.
    pub tts_enabled: bool,

    /// Session id used as the persistence key. Default: "voice-assistant".
    pub session_id: String,

    /// Optional path to a SQLite file for persisted session history.
    /// If unset, conversation state lives only in memory for the process.
    pub session_db: Option<std::path::PathBuf>,

    /// Hard USD-cent budget for a single run. `None` = unbounded.
    /// Default: None.
    pub max_usd_cents: Option<u64>,

    /// Hard token budget for a single run. `None` = unbounded.
    pub max_tokens: Option<u64>,
}

impl Default for VaConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            stt_model: "whisper-1".into(),
            tts_model: "tts-1".into(),
            tts_voice: "alloy".into(),
            wake_word_model: None,
            wake_word_threshold: 0.5,
            silence_threshold_db: -40.0,
            silence_duration_ms: 800,
            max_record_secs: 30.0,
            microphone: None,
            speaker: None,
            system_prompt:
                "You are a helpful voice assistant. Keep responses concise and conversational."
                    .into(),
            llm_model: "gpt-4o-mini".into(),
            tts_enabled: true,
            session_id: "voice-assistant".into(),
            session_db: None,
            max_usd_cents: None,
            max_tokens: None,
        }
    }
}

impl VaConfig {
    /// Load from a TOML file.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config: VaConfig = toml::from_str(&text)?;
        Ok(config)
    }

    /// Load from the given path, or fall back to [`VaConfig::default`] if the
    /// file does not exist. This mirrors the behaviour the binary uses when
    /// `--config` is omitted: if no config has been written yet, running on
    /// defaults is a friendlier outcome than an error.
    pub fn load_from(path: &std::path::Path) -> anyhow::Result<Self> {
        if path.exists() {
            Self::from_file(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Resolve the OpenAI API key: config file first, then `OPENAI_API_KEY` env var.
    pub fn resolve_api_key(&self) -> anyhow::Result<String> {
        self.openai_api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No OpenAI API key found. Set `openai_api_key` in config or \
                 the `OPENAI_API_KEY` environment variable."
                )
            })
    }
}
