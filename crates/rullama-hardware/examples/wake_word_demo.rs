//! Demonstrate the in-house DTW + MFCC wake-word detector.
//!
//! Speaker-dependent: you record the wake phrase 3 times (the "enrollment"
//! ceremony) and the detector then listens for an utterance that matches
//! any of those reference recordings via DTW.
//!
//! Run with:
//! ```bash
//! cargo run -p rullama-hardware --example wake_word_demo \
//!     --features wake-word-dtw
//! ```
//!
//! The demo:
//!   1. Records 3 × 1.2 s enrollment clips (you say the wake phrase each time)
//!   2. Switches into live-listen mode and prints `[score] wake!` on a match
//!
//! Press Ctrl-C to stop.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use rullama_hardware::audio::{
    capture::AudioCapture,
    hardware::cpal_capture::CpalCapture,
    types::AudioConfig,
    vad::pcm_to_i16_mono,
    wake_word::{DtwWakeWordDetector, WakeWordDetector},
};

const ENROLL_CLIP_MS: u64 = 1_200;
const ENROLLMENTS_NEEDED: usize = 3;

async fn record_clip_ms(
    capture: &CpalCapture,
    config: &AudioConfig,
    millis: u64,
) -> anyhow::Result<Vec<i16>> {
    let mut stream = capture.start_capture(None, config)?;
    let mut buf: Vec<i16> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(millis);

    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(Some(Ok(audio_buf))) => buf.extend_from_slice(&pcm_to_i16_mono(&audio_buf)),
            Ok(Some(Err(e))) => eprintln!("capture error: {e}"),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    Ok(buf)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let capture = CpalCapture;
    let config = AudioConfig::speech(); // 16 kHz mono i16

    let mut detector = DtwWakeWordDetector::new();
    println!(
        "Wake-word demo — DTW + MFCC, speaker-dependent. Each enrollment is {ENROLL_CLIP_MS} ms."
    );
    println!("You'll say the wake phrase {ENROLLMENTS_NEEDED} times to enroll, then it listens.\n");

    for i in 1..=ENROLLMENTS_NEEDED {
        println!("Enrollment {i}/{ENROLLMENTS_NEEDED}: say your wake phrase NOW...");
        let clip = record_clip_ms(&capture, &config, ENROLL_CLIP_MS).await?;
        detector.enroll_template(&clip)?;
        println!("  recorded {} samples — enrolled.", clip.len());
    }

    println!("\nListening for wake word... (Ctrl-C to stop)");
    let mut stream = capture.start_capture(None, &config)?;
    let running = Arc::new(AtomicBool::new(true));
    let r = Arc::clone(&running);
    ctrlc::set_handler(move || r.store(false, Ordering::Relaxed)).unwrap_or_default();

    let mut sample_buf: Vec<i16> = Vec::new();
    let frame_size = detector.frame_size();

    while running.load(Ordering::Relaxed) {
        match tokio::time::timeout(Duration::from_millis(100), stream.next()).await {
            Ok(Some(Ok(audio_buf))) => {
                let mono = pcm_to_i16_mono(&audio_buf);
                sample_buf.extend_from_slice(&mono);

                while sample_buf.len() >= frame_size {
                    let frame: Vec<i16> = sample_buf.drain(..frame_size).collect();
                    if let Some(det) = detector.process_frame(&frame) {
                        println!(
                            "[{:.1}s] wake! score={:.3} keyword=\"{}\"",
                            det.timestamp_ms as f64 / 1000.0,
                            det.score,
                            det.keyword,
                        );
                        detector.reset_window();
                    }
                }
            }
            Ok(Some(Err(e))) => eprintln!("capture error: {e}"),
            Ok(None) => break,
            Err(_) => {}
        }
    }

    println!("Done.");
    Ok(())
}
