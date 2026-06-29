use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use brainwires_agent::personas::StaticPersonaProvider;
use brainwires_call_policy::{BudgetConfig, BudgetGuard};
use brainwires_hardware::audio::{
    api::{OpenAiStt, OpenAiTts},
    assistant::{VoiceAssistant, VoiceAssistantConfig},
    capture::AudioCapture,
    device::AudioDevice,
    hardware::{CpalCapture, CpalPlayback},
    playback::AudioPlayback,
    types::{TtsOptions, Voice},
};
use brainwires_provider::{OpenAiChatProvider, OpenAiClient};
use brainwires_stores::{ArcSessionStore, InMemorySessionStore, SessionId, SqliteSessionStore};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use voice_assistant::config::VaConfig;
use voice_assistant::handler::LlmHandler;

/// Brainwires Personal Voice Assistant
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to TOML configuration file.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// List available audio devices and exit.
    #[arg(long)]
    list_devices: bool,

    /// Override the wake word model path.
    #[arg(long, value_name = "FILE")]
    wake_word: Option<PathBuf>,

    /// Enable verbose logging.
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Logging
    let level = if cli.verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "voice_assistant={level},brainwires_hardware={level}"
        ))
    });
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // ── List devices mode ─────────────────────────────────────────────────────
    if cli.list_devices {
        let capture = CpalCapture;
        let playback = CpalPlayback;
        println!("Microphones:");
        for d in capture.list_devices().unwrap_or_default() {
            println!("  {} {}", if d.is_default { "*" } else { " " }, d.name);
        }
        println!("Speakers:");
        for d in playback.list_devices().unwrap_or_default() {
            println!("  {} {}", if d.is_default { "*" } else { " " }, d.name);
        }
        return Ok(());
    }

    // ── Load config ───────────────────────────────────────────────────────────
    let cfg = match cli.config.as_ref() {
        Some(path) => VaConfig::from_file(path)?,
        None => {
            let default_path = dirs_config();
            if default_path.exists() {
                VaConfig::from_file(&default_path)?
            } else {
                VaConfig::default()
            }
        }
    };

    let api_key = cfg.resolve_api_key()?;

    // ── Build components ──────────────────────────────────────────────────────
    // Chat provider wired through the harness. OpenAI-compatible
    // Chat Completions by default — swap for any `brainwires_core::Provider`
    // impl to talk to a different backend.
    let openai_client = Arc::new(OpenAiClient::new(api_key.clone(), cfg.llm_model.clone()));
    let llm_provider: Arc<dyn brainwires_core::Provider> = Arc::new(OpenAiChatProvider::new(
        openai_client,
        cfg.llm_model.clone(),
    ));

    let capture: Arc<CpalCapture> = Arc::new(CpalCapture);
    let playback: Arc<CpalPlayback> = Arc::new(CpalPlayback);
    let stt = Arc::new(OpenAiStt::new(&api_key).with_model(&cfg.stt_model));

    // ── Resolve audio devices ─────────────────────────────────────────────────
    let microphone = cfg
        .microphone
        .as_ref()
        .and_then(|name| find_input_device(&capture, name));
    let speaker = cfg
        .speaker
        .as_ref()
        .and_then(|name| find_output_device(&playback, name));

    // ── TTS ───────────────────────────────────────────────────────────────────
    let tts_options = if cfg.tts_enabled {
        Some(TtsOptions {
            voice: Voice {
                id: cfg.tts_voice.clone(),
                name: Some(cfg.tts_voice.clone()),
                language: None,
            },
            speed: None,
            output_format: brainwires_hardware::audio::types::OutputFormat::Pcm,
        })
    } else {
        None
    };

    // ── Assistant config ──────────────────────────────────────────────────────
    let assistant_config = VoiceAssistantConfig {
        silence_threshold_db: cfg.silence_threshold_db,
        silence_duration_ms: cfg.silence_duration_ms,
        max_record_secs: cfg.max_record_secs,
        tts_options,
        microphone,
        speaker,
        ..Default::default()
    };

    let mut builder = VoiceAssistant::builder(
        capture.clone() as Arc<dyn brainwires_hardware::audio::capture::AudioCapture>,
        stt,
    )
    .with_playback(playback.clone() as Arc<dyn AudioPlayback>)
    .with_config(assistant_config);

    if cfg.tts_enabled {
        let tts = Arc::new(OpenAiTts::new(&api_key).with_model(&cfg.tts_model));
        builder = builder.with_tts(tts);
    }

    // ── Wake word ─────────────────────────────────────────────────────────────
    let wake_word_path = cli
        .wake_word
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .or_else(|| cfg.wake_word_model.clone());

    if let Some(model_path) = wake_word_path {
        use brainwires_hardware::audio::wake_word::EnergyTriggerDetector;
        let _ = model_path; // model path unused for energy trigger
        info!("Wake trigger enabled (energy-based).");
        let detector = EnergyTriggerDetector::default();
        builder = builder.with_wake_word(Box::new(detector));
        info!("Say something loud to start listening.");
    }

    let mut assistant = builder.build();

    // ── Ctrl-C ────────────────────────────────────────────────────────────────
    let stop_flag = Arc::new(AtomicBool::new(false));
    let sf = Arc::clone(&stop_flag);
    ctrlc::set_handler(move || {
        sf.store(true, Ordering::Relaxed);
    })?;

    // ── Harness wiring ────────────────────────────────────────────────────────
    let persona = Arc::new(StaticPersonaProvider::new(cfg.system_prompt.clone()));

    let budget = if cfg.max_usd_cents.is_some() || cfg.max_tokens.is_some() {
        Some(BudgetGuard::new(BudgetConfig {
            max_tokens: cfg.max_tokens,
            max_usd_cents: cfg.max_usd_cents,
            max_rounds: None,
        }))
    } else {
        None
    };

    let session_store: ArcSessionStore = match cfg.session_db.as_ref() {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Arc::new(SqliteSessionStore::open(path)?)
        }
        None => Arc::new(InMemorySessionStore::new()),
    };

    let mut handler = LlmHandler::new(llm_provider, persona, budget).await?;
    handler = handler
        .with_session_store(session_store, SessionId::new(&cfg.session_id))
        .await?;

    // ── Run ───────────────────────────────────────────────────────────────────
    info!("Voice assistant ready. Press Ctrl-C to exit.");

    tokio::select! {
        result = assistant.run(&handler) => result?,
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down…");
        }
    }

    Ok(())
}

fn dirs_config() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("voice-assistant")
        .join("config.toml")
}

fn find_input_device(capture: &Arc<CpalCapture>, name: &str) -> Option<AudioDevice> {
    capture
        .list_devices()
        .unwrap_or_default()
        .into_iter()
        .find(|d| d.name.contains(name))
}

fn find_output_device(playback: &Arc<CpalPlayback>, name: &str) -> Option<AudioDevice> {
    playback
        .list_devices()
        .unwrap_or_default()
        .into_iter()
        .find(|d| d.name.contains(name))
}
