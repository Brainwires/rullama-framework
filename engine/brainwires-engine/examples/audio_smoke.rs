//! End-to-end smoke test for the audio path.
//!
//! Synthesizes 1 second of pure-tone audio, runs it through the GPU audio
//! encoder (CPU SSCP prefix + 12 Conformer blocks on GPU + projector), and
//! prints summary stats of the resulting soft tokens. Validates that the
//! entire audio pipeline runs without panic and emits f32s within plausible
//! ranges.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example audio_smoke -- <gguf>

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use brainwires_engine::api::Model;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: audio_smoke <gguf>");
            return ExitCode::from(2);
        }
    };

    println!("loading {path} ...");
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());

    if !model.has_audio_native() {
        eprintln!("this checkpoint has no audio tower (no a.conv1d.0.weight)");
        return ExitCode::from(2);
    }

    // 1 second of A4 (440 Hz) sine at 16 kHz.
    let sr = 16_000usize;
    let n = sr;
    let omega = 2.0 * std::f32::consts::PI * 440.0 / sr as f32;
    let pcm: Vec<f32> = (0..n).map(|i| 0.3 * (omega * i as f32).sin()).collect();

    println!("encoding {} samples (~1 s @ 16 kHz) ...", pcm.len());
    let t0 = Instant::now();
    let soft = pollster::block_on(model.encode_audio_native(&pcm)).expect("encode_audio");
    let dt = t0.elapsed();

    let d_text = 1536usize;
    let n_soft = soft.len() / d_text;
    println!(
        "encoded in {:?} — {} audio soft tokens × {} dim = {} f32s",
        dt,
        n_soft,
        d_text,
        soft.len()
    );

    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut nan_count = 0usize;
    for &v in &soft {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let n_finite = (soft.len() - nan_count).max(1) as f64;
    let mean = sum / n_finite;
    let var = sum_sq / n_finite - mean * mean;
    println!(
        "stats: mean={:.4} stddev={:.4} min={:.4} max={:.4} nans={}",
        mean,
        var.sqrt(),
        min,
        max,
        nan_count
    );

    if nan_count > 0 {
        eprintln!("FAIL: NaNs in output");
        return ExitCode::from(1);
    }
    if !min.is_finite() || !max.is_finite() {
        eprintln!("FAIL: non-finite output");
        return ExitCode::from(1);
    }
    if soft.is_empty() {
        eprintln!("FAIL: empty output");
        return ExitCode::from(1);
    }
    println!("OK");
    ExitCode::SUCCESS
}
