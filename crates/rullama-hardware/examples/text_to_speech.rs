//! Text-to-speech demonstration using the `TextToSpeech` trait.
//!
//! Shows how to configure `TtsOptions` with voice selection and `OutputFormat`,
//! and how to create different cloud TTS providers.
//!
//! Run:
//!   cargo run -p rullama-hardware --features native --example text_to_speech

use rullama_hardware::{
    CartesiaTts, DeepgramTts, ElevenLabsTts, GoogleTts, OpenAiTts, OutputFormat, TextToSpeech,
    TtsOptions, Voice,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Text-to-Speech Example ===\n");

    // ── 1. Build TTS options with voice selection ───────────────────────────
    // Default options use the "alloy" voice, WAV output, normal speed.
    let default_opts = TtsOptions::default();
    println!("Default TtsOptions: {default_opts:?}");

    // Custom options: pick a specific voice, speed, and output format.
    let custom_opts = TtsOptions {
        voice: Voice {
            id: "shimmer".to_string(),
            name: Some("Shimmer".to_string()),
            language: Some("en-US".to_string()),
        },
        speed: Some(1.25),
        output_format: OutputFormat::Mp3,
    };
    println!("Custom TtsOptions: {custom_opts:?}");

    // ── 2. Demonstrate all OutputFormat variants ────────────────────────────
    let formats = [
        OutputFormat::Wav,
        OutputFormat::Mp3,
        OutputFormat::Pcm,
        OutputFormat::Opus,
        OutputFormat::Flac,
    ];
    println!("\nAvailable output formats:");
    for fmt in &formats {
        println!("  - {fmt:?}");
    }

    // ── 3. Create different cloud TTS providers ─────────────────────────────
    // All providers implement `TextToSpeech`, so they can be used
    // interchangeably behind a trait object.

    let demo_key = "sk-demo-placeholder";

    let openai: Box<dyn TextToSpeech> = Box::new(OpenAiTts::new(demo_key));
    let deepgram: Box<dyn TextToSpeech> = Box::new(DeepgramTts::new(demo_key));
    let elevenlabs: Box<dyn TextToSpeech> = Box::new(ElevenLabsTts::new(demo_key));
    let cartesia: Box<dyn TextToSpeech> = Box::new(CartesiaTts::new(demo_key));
    let google: Box<dyn TextToSpeech> = Box::new(GoogleTts::new(demo_key));

    let providers: Vec<(&str, &dyn TextToSpeech)> = vec![
        ("OpenAI TTS", openai.as_ref()),
        ("Deepgram", deepgram.as_ref()),
        ("ElevenLabs", elevenlabs.as_ref()),
        ("Cartesia", cartesia.as_ref()),
        ("Google Cloud", google.as_ref()),
    ];

    println!("\nRegistered providers:");
    for (label, provider) in &providers {
        println!("  - {label} (backend: {})", provider.name());
    }

    // ── 4. Voice selection with the Voice struct ────────────────────────────
    let voices = vec![
        Voice::new("alloy"),
        Voice::new("echo"),
        Voice {
            id: "nova".to_string(),
            name: Some("Nova".to_string()),
            language: Some("en-US".to_string()),
        },
    ];

    println!("\nExample voices:");
    for v in &voices {
        let name = v.name.as_deref().unwrap_or("(unnamed)");
        let lang = v.language.as_deref().unwrap_or("(any)");
        println!("  - id={}, name={name}, lang={lang}", v.id);
    }

    // ── 5. Demonstrate synthesis (skipped without real API keys) ────────────
    // Uncomment the block below when using real credentials:
    //
    // let text = "Hello! This is a text-to-speech demonstration.";
    // let opts = TtsOptions {
    //     voice: Voice::new("alloy"),
    //     speed: None,
    //     output_format: OutputFormat::Wav,
    // };
    //
    // for (label, provider) in &providers {
    //     match provider.synthesize(text, &opts).await {
    //         Ok(buffer) => {
    //             println!(
    //                 "\n{label}: synthesized {:.2}s of audio ({} bytes PCM)",
    //                 buffer.duration_secs(),
    //                 buffer.data.len(),
    //             );
    //         }
    //         Err(e) => println!("\n{label} error: {e}"),
    //     }
    // }

    println!("\nDone.");
    Ok(())
}
