//! Load a WAV or FLAC file and play it through the default speaker.
//!
//! Usage:
//!   cargo run --example play_audio -- recording.wav
//!   cargo run --example play_audio -- recording.flac
//!   cargo run --example play_audio -- --file recording.flac

use brainwires_hardware::{AudioBuffer, AudioPlayback, CpalPlayback, decode_wav};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = audio_path_from_args()?;
    let raw = std::fs::read(&path)?;

    let buffer: AudioBuffer = if path.ends_with(".flac") {
        decode_flac_or_bail(&raw)?
    } else {
        decode_wav(&raw)?
    };

    println!(
        "Loaded {} ({:.2}s, {} Hz, {} ch, {:?})",
        path,
        buffer.duration_secs(),
        buffer.config.sample_rate,
        buffer.config.channels,
        buffer.config.sample_format,
    );

    let playback = CpalPlayback::new();

    // Show available output devices
    let devices = playback.list_devices()?;
    println!("Output devices:");
    for dev in &devices {
        let marker = if dev.is_default { " (default)" } else { "" };
        println!("  - {}{marker}", dev.name);
    }

    println!("\nPlaying...");
    playback.play(None, &buffer).await?;
    println!("Done.");

    Ok(())
}

#[cfg(feature = "flac")]
fn decode_flac_or_bail(raw: &[u8]) -> anyhow::Result<AudioBuffer> {
    Ok(brainwires_hardware::decode_flac(raw)?)
}

#[cfg(not(feature = "flac"))]
fn decode_flac_or_bail(_raw: &[u8]) -> anyhow::Result<AudioBuffer> {
    anyhow::bail!("FLAC support requires the `flac` feature")
}

fn audio_path_from_args() -> anyhow::Result<String> {
    let args: Vec<String> = std::env::args().collect();

    if let Some(i) = args.iter().position(|a| a == "--file")
        && let Some(path) = args.get(i + 1)
    {
        return Ok(path.clone());
    }

    // Positional: first non-flag argument after the binary name
    if let Some(path) = args.get(1)
        && !path.starts_with('-')
    {
        return Ok(path.clone());
    }

    anyhow::bail!("Usage: play_audio <file.wav|file.flac>")
}
