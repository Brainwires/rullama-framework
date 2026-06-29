//! Speech-to-text demonstration using the `SpeechToText` trait.
//!
//! Shows how to configure `SttOptions`, handle `Transcript` / `TranscriptSegment`
//! results, and construct different cloud STT providers.
//!
//! Run:
//!   cargo run -p rullama-hardware --features native --example speech_to_text

use rullama_hardware::{
    AudioBuffer, AudioConfig, DeepgramStt, ElevenLabsStt, OpenAiStt, SpeechToText, SttOptions,
    Transcript,
};

/// Helper: pretty-print a transcript and its segments.
fn print_transcript(provider_name: &str, transcript: &Transcript) {
    println!("\n--- {provider_name} Transcript ---");
    println!("Text: {}", transcript.text);

    if let Some(lang) = &transcript.language {
        println!("Detected language: {lang}");
    }
    if let Some(dur) = transcript.duration_secs {
        println!("Duration: {dur:.2}s");
    }

    if !transcript.segments.is_empty() {
        println!("Segments:");
        for seg in &transcript.segments {
            println!("  [{:.2}s - {:.2}s] {}", seg.start, seg.end, seg.text);
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Speech-to-Text Example ===\n");

    // ── 1. Build a synthetic audio buffer (silence) for demonstration ────────
    // In a real application you would capture from a microphone or load a file.
    let config = AudioConfig::speech(); // 16 kHz, mono, i16
    let duration_secs = 2.0;
    let num_samples = (config.sample_rate as f64 * duration_secs) as usize;
    let silence = vec![0u8; num_samples * config.bytes_per_sample()];
    let audio = AudioBuffer::from_pcm(silence, config);

    println!(
        "Prepared demo audio buffer: {:.2}s, {} Hz, {} channel(s)",
        audio.duration_secs(),
        audio.config.sample_rate,
        audio.config.channels,
    );

    // ── 2. Configure transcription options ──────────────────────────────────
    // Basic options: just transcribe with defaults.
    let basic_opts = SttOptions::default();
    println!("\nBasic SttOptions: {basic_opts:?}");

    // Advanced options: language hint, timestamps, and a guiding prompt.
    let advanced_opts = SttOptions {
        language: Some("en".to_string()),
        timestamps: true,
        prompt: Some("Meeting notes about quarterly planning.".to_string()),
    };
    println!("Advanced SttOptions: {advanced_opts:?}");

    // ── 3. Create different cloud STT providers ─────────────────────────────
    // Each provider implements the same `SpeechToText` trait, so they are
    // interchangeable behind `Box<dyn SpeechToText>`.

    // Use a placeholder key; in production read from env or a secret store.
    let demo_key = "sk-demo-placeholder";

    let openai: Box<dyn SpeechToText> = Box::new(OpenAiStt::new(demo_key));
    let deepgram: Box<dyn SpeechToText> = Box::new(DeepgramStt::new(demo_key));
    let elevenlabs: Box<dyn SpeechToText> = Box::new(ElevenLabsStt::new(demo_key));

    let providers: Vec<(&str, &dyn SpeechToText)> = vec![
        ("OpenAI Whisper", openai.as_ref()),
        ("Deepgram", deepgram.as_ref()),
        ("ElevenLabs", elevenlabs.as_ref()),
    ];

    println!("\nRegistered providers:");
    for (label, provider) in &providers {
        println!("  - {label} (backend: {})", provider.name());
    }

    // ── 4. Demonstrate transcription (skipped without real API keys) ────────
    // Uncomment the block below when using real credentials:
    //
    // for (label, provider) in &providers {
    //     match provider.transcribe(&audio, &advanced_opts).await {
    //         Ok(transcript) => print_transcript(label, &transcript),
    //         Err(e) => println!("\n{label} error: {e}"),
    //     }
    // }

    // ── 5. Show how you would handle the result ─────────────────────────────
    // Build a mock transcript to demonstrate result handling.
    let mock = Transcript {
        text: "Hello world, this is a test transcript.".to_string(),
        language: Some("en".to_string()),
        duration_secs: Some(2.0),
        segments: vec![
            rullama_hardware::TranscriptSegment {
                text: "Hello world,".to_string(),
                start: 0.0,
                end: 0.8,
            },
            rullama_hardware::TranscriptSegment {
                text: "this is a test transcript.".to_string(),
                start: 0.8,
                end: 2.0,
            },
        ],
    };

    print_transcript("Mock", &mock);

    println!("\nDone.");
    Ok(())
}
