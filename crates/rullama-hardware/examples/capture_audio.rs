//! Record audio from the default microphone and save it as a WAV or FLAC file.
//!
//! Usage:
//!   cargo run --example capture_to_wav
//!   cargo run --example capture_to_wav -- --duration 10 --output recording.wav
//!   cargo run --example capture_to_wav -- --format flac

use rullama_hardware::{AudioCapture, AudioConfig, CpalCapture, encode_wav};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let duration = parse_flag(&args, "--duration").unwrap_or(5.0);
    let format = parse_flag_string(&args, "--format").unwrap_or_else(|| "wav".into());
    let default_output = match format.as_str() {
        "flac" => "capture.flac",
        _ => "capture.wav",
    };
    let output: PathBuf = parse_flag_string(&args, "--output")
        .unwrap_or_else(|| default_output.into())
        .into();

    let capture = CpalCapture::new();

    // Show available input devices
    let devices = capture.list_devices()?;
    println!("Input devices:");
    for dev in &devices {
        let marker = if dev.is_default { " (default)" } else { "" };
        println!("  - {}{marker}", dev.name);
    }

    // Record using speech-quality config (16 kHz mono i16 — ideal for STT)
    let config = AudioConfig::speech();
    println!(
        "\nRecording {duration}s of audio ({} Hz, {} ch, {:?})...",
        config.sample_rate, config.channels, config.sample_format,
    );

    let buffer = capture.record(None, &config, duration).await?;

    println!(
        "Captured {} frames ({:.2}s, {} bytes PCM)",
        buffer.num_frames(),
        buffer.duration_secs(),
        buffer.data.len(),
    );

    // Encode and write to disk
    let encoded = match format.as_str() {
        #[cfg(feature = "flac")]
        "flac" => {
            println!("Encoding as FLAC...");
            rullama_hardware::encode_flac(&buffer)?
        }
        #[cfg(not(feature = "flac"))]
        "flac" => anyhow::bail!("FLAC support requires the `flac` feature"),
        _ => {
            println!("Encoding as WAV...");
            encode_wav(&buffer)?
        }
    };

    std::fs::write(&output, &encoded)?;
    println!("Saved to {} ({} bytes)", output.display(), encoded.len());

    Ok(())
}

fn parse_flag(args: &[String], flag: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_flag_string(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
