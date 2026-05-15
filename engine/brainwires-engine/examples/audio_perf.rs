//! Audio encode timing — runs `encode_audio_native` three times on the same
//! synthetic clip and prints elapsed times. First call pays for weight uploads
//! (12 blocks × 21 weights = ~252 GPU uploads); subsequent calls hit the warm
//! `WeightCache` and show steady-state encode cost.
//!
//! Build:
//!   cargo run --release --example audio_perf -- <gguf>

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use rullama::api::Model;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let gguf = match args.next() {
        Some(p) => p,
        None => { eprintln!("usage: audio_perf <gguf>"); return ExitCode::from(2); }
    };

    println!("loading model ...");
    let t0 = Instant::now();
    let bytes = fs::read(&gguf).expect("read");
    let model = pollster::block_on(Model::load_native(bytes)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());

    if !model.has_audio_native() {
        eprintln!("FAIL: no audio tower");
        return ExitCode::from(2);
    }

    // 1 second of A4 (440 Hz) sine at 16 kHz, same shape as audio_smoke.
    let sr = 16_000usize;
    let mut pcm = vec![0f32; sr];
    let omega = 2.0 * std::f32::consts::PI * 440.0 / sr as f32;
    for i in 0..sr { pcm[i] = 0.3 * (omega * i as f32).sin(); }

    let baseline_bytes = model.cached_weight_bytes_native();
    println!("cached_weight_bytes before encode: {} MiB", baseline_bytes / (1024 * 1024));

    println!("\nFIRST encode (cold cache):");
    let t = Instant::now();
    let soft1 = pollster::block_on(model.encode_audio_native(&pcm)).expect("encode");
    let dt1 = t.elapsed();
    println!("  encoded {} f32 in {:?}", soft1.len(), dt1);
    let after_cold = model.cached_weight_bytes_native();
    println!("  cached_weight_bytes after cold: {} MiB (+{} MiB)",
        after_cold / (1024 * 1024), (after_cold - baseline_bytes) / (1024 * 1024));

    println!("\nSECOND encode (warm cache):");
    let t = Instant::now();
    let soft2 = pollster::block_on(model.encode_audio_native(&pcm)).expect("encode");
    let dt2 = t.elapsed();
    println!("  encoded {} f32 in {:?}", soft2.len(), dt2);

    println!("\nTHIRD encode (also warm):");
    let t = Instant::now();
    let _soft3 = pollster::block_on(model.encode_audio_native(&pcm)).expect("encode");
    let dt3 = t.elapsed();
    println!("  encoded in {:?}", dt3);

    let mut max_abs = 0f32;
    for i in 0..soft1.len() {
        let d = (soft1[i] - soft2[i]).abs();
        if d > max_abs { max_abs = d; }
    }
    println!("\nfirst vs second max_abs diff: {max_abs:e} (should be 0)");

    let freed = model.release_audio_weights_native();
    let after_release = model.cached_weight_bytes_native();
    println!("\nrelease_audio_weights freed {} entries; cache now {} MiB",
        freed, after_release / (1024 * 1024));

    println!("\nspeedup cold→warm: {:.1}×", dt1.as_secs_f64() / dt2.as_secs_f64());
    ExitCode::SUCCESS
}
