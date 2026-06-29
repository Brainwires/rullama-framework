//! Smoke tests for voice-assistant configuration loading.
//!
//! The handler itself is tightly coupled to real network providers
//! (`OpenAiChatProvider`, `ChatAgent`, `VoiceAssistantHandler` w/ CPAL), so
//! a pure prompt-builder unit test is not feasible at this layer — the
//! prompt assembly happens inside `brainwires-agent::ChatAgent` rather
//! than in any pure helper on `LlmHandler`. That behaviour is covered by
//! the agents crate's own test suite.

use std::path::PathBuf;

use tempfile::TempDir;
use voice_assistant::config::VaConfig;

#[test]
fn config_load_roundtrip() {
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("voice.toml");

    let sample = r#"
openai_api_key = "sk-test-abc"
stt_model = "whisper-large"
tts_model = "tts-hd"
tts_voice = "nova"
wake_word_model = "/tmp/wake.rpw"
wake_word_threshold = 0.75
silence_threshold_db = -35.5
silence_duration_ms = 1200
max_record_secs = 15.0
microphone = "USB Mic"
speaker = "Built-in"
system_prompt = "You are terse."
llm_model = "gpt-4o"
tts_enabled = false
session_id = "kitchen"
session_db = "/tmp/va.sqlite"
max_usd_cents = 500
max_tokens = 20000
"#;

    std::fs::write(&path, sample).expect("write config");

    let cfg = VaConfig::from_file(&path).expect("parse config");
    assert_eq!(cfg.openai_api_key.as_deref(), Some("sk-test-abc"));
    assert_eq!(cfg.stt_model, "whisper-large");
    assert_eq!(cfg.tts_model, "tts-hd");
    assert_eq!(cfg.tts_voice, "nova");
    assert_eq!(cfg.wake_word_model.as_deref(), Some("/tmp/wake.rpw"));
    assert!((cfg.wake_word_threshold - 0.75).abs() < f32::EPSILON);
    assert!((cfg.silence_threshold_db - -35.5).abs() < f32::EPSILON);
    assert_eq!(cfg.silence_duration_ms, 1200);
    assert!((cfg.max_record_secs - 15.0).abs() < f64::EPSILON);
    assert_eq!(cfg.microphone.as_deref(), Some("USB Mic"));
    assert_eq!(cfg.speaker.as_deref(), Some("Built-in"));
    assert_eq!(cfg.system_prompt, "You are terse.");
    assert_eq!(cfg.llm_model, "gpt-4o");
    assert!(!cfg.tts_enabled);
    assert_eq!(cfg.session_id, "kitchen");
    assert_eq!(cfg.session_db, Some(PathBuf::from("/tmp/va.sqlite")));
    assert_eq!(cfg.max_usd_cents, Some(500));
    assert_eq!(cfg.max_tokens, Some(20_000));
}

#[test]
fn config_missing_file_falls_back_to_defaults() {
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("does-not-exist.toml");

    let cfg = VaConfig::load_from(&path).expect("load defaults for missing file");
    let defaults = VaConfig::default();

    assert_eq!(cfg.stt_model, defaults.stt_model);
    assert_eq!(cfg.tts_model, defaults.tts_model);
    assert_eq!(cfg.tts_voice, defaults.tts_voice);
    assert_eq!(cfg.llm_model, defaults.llm_model);
    assert_eq!(cfg.session_id, defaults.session_id);
    assert_eq!(cfg.tts_enabled, defaults.tts_enabled);
    assert!(cfg.openai_api_key.is_none());
    assert!(cfg.wake_word_model.is_none());
    assert!(!path.exists(), "load_from must not create the file");
}

#[test]
fn config_partial_toml_fills_remaining_fields_with_defaults() {
    // `#[serde(default)]` on VaConfig means an otherwise-empty file should
    // still parse and match `VaConfig::default()` on untouched fields.
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("partial.toml");
    std::fs::write(&path, "llm_model = \"gpt-3.5-turbo\"\n").expect("write partial");

    let cfg = VaConfig::from_file(&path).expect("parse partial config");
    let defaults = VaConfig::default();
    assert_eq!(cfg.llm_model, "gpt-3.5-turbo");
    assert_eq!(cfg.stt_model, defaults.stt_model);
    assert_eq!(cfg.tts_voice, defaults.tts_voice);
    assert_eq!(cfg.wake_word_threshold, defaults.wake_word_threshold);
}

#[test]
fn config_resolve_api_key_prefers_config_value() {
    let cfg = VaConfig {
        openai_api_key: Some("sk-from-config".into()),
        ..VaConfig::default()
    };
    assert_eq!(cfg.resolve_api_key().unwrap(), "sk-from-config");
}

#[test]
fn config_resolve_api_key_errors_without_source() {
    let cfg = VaConfig::default();
    // Ensure env var is not present for this test; we tolerate the case
    // where the environment happens to be pre-populated by the user.
    if std::env::var("OPENAI_API_KEY").is_ok() {
        eprintln!("skipping: OPENAI_API_KEY already set in environment");
        return;
    }
    assert!(cfg.resolve_api_key().is_err());
}
