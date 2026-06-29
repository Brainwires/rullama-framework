//! Minimal voice assistant pipeline demo.
//!
//! Records speech from the microphone, transcribes it with OpenAI Whisper,
//! then prints the transcript. Optionally speaks a fixed reply via TTS.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... \
//!   cargo run -p rullama-hardware --example voice_assistant --features voice-assistant
//! ```
//!
//! Optional flags:
//!   --wake-word              — show the wake-word-demo pointer (see wake_word_demo example)
//!   --silence-db <dB>        — silence threshold, default -40
//!   --silence-ms <ms>        — silence duration to end utterance, default 800

use std::sync::Arc;

use async_trait::async_trait;
use rullama_hardware::audio::{
    api::OpenAiStt,
    assistant::{VoiceAssistant, VoiceAssistantConfig, VoiceAssistantHandler},
    capture::AudioCapture as _,
    error::AudioError,
    hardware::{CpalCapture, CpalPlayback},
    types::Transcript,
};

struct PrintHandler;

#[async_trait]
impl VoiceAssistantHandler for PrintHandler {
    async fn on_speech(&self, transcript: &Transcript) -> Option<String> {
        let text = transcript.text.trim();
        if text.is_empty() {
            return None;
        }
        println!("You said: {text}");
        // Echo the transcript back as the reply
        Some(format!("You said: {text}"))
    }

    async fn on_error(&self, error: &AudioError) {
        eprintln!("Assistant error: {error}");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable required");

    let capture = Arc::new(CpalCapture);
    let playback = Arc::new(CpalPlayback);
    let stt = Arc::new(OpenAiStt::new(api_key));

    // List and show available devices
    let devices = capture.list_devices().unwrap_or_default();
    if !devices.is_empty() {
        println!("Available microphones:");
        for d in &devices {
            println!("  {} {}", if d.is_default { "*" } else { " " }, d.name);
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let silence_threshold_db = find_arg_f32(&args, "--silence-db").unwrap_or(-40.0);
    let silence_duration_ms = find_arg_u32(&args, "--silence-ms").unwrap_or(800);

    let config = VoiceAssistantConfig {
        silence_threshold_db,
        silence_duration_ms,
        ..Default::default()
    };

    #[allow(unused_mut)]
    let mut builder = VoiceAssistant::builder(capture, stt)
        .with_playback(playback)
        .with_config(config);

    // Wake word support (in-house DTW + MFCC, speaker-dependent).
    // The DTW detector requires per-user enrollment recordings, which this
    // demo doesn't expose — use `cargo run --example wake_word_demo
    // --features wake-word-dtw` for the interactive enrollment ceremony.
    // Build with `--features wake-word-dtw` here to compile in the detector;
    // the enrollment-and-attach hook is intentionally stubbed out below.
    #[cfg(feature = "wake-word-dtw")]
    if find_arg_str(&args, "--wake-word").is_some() {
        eprintln!(
            "note: voice_assistant example does not enroll a wake word — see the \
             wake_word_demo example for the interactive enrollment flow."
        );
    }

    let mut assistant = builder.build();

    // Ctrl-C to stop
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sf = Arc::clone(&stop_flag);
    ctrlc::set_handler(move || sf.store(true, std::sync::atomic::Ordering::Relaxed))
        .unwrap_or_default();

    println!("Voice assistant started. Speak after the prompt. Ctrl-C to stop.");
    println!("---");

    let handler = PrintHandler;
    assistant.run(&handler).await?;

    Ok(())
}

fn find_arg_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn find_arg_f32(args: &[String], flag: &str) -> Option<f32> {
    find_arg_str(args, flag).and_then(|s| s.parse().ok())
}

fn find_arg_u32(args: &[String], flag: &str) -> Option<u32> {
    find_arg_str(args, flag).and_then(|s| s.parse().ok())
}
