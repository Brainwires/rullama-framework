//! StyleTTS2 style-encoder CPU-oracle parity check.
//!
//! Loads the fixtures dumped by `scripts/styletts2_dump_style_fixtures.py`
//! (~/.cache/styletts2/fixtures/bin/), runs the Rust `StyleEncoder` oracle for both
//! the acoustic and prosodic encoders on the reference mel, and diffs the resulting
//! 128-d style vectors against the PyTorch reference. This is the cloning encoder's
//! GPU-vs-CPU-style parity gate (here CPU-oracle vs PyTorch).
//!
//!   cargo run -p rullama-engine --release --example styletts2_style_oracle

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rullama_engine::reference::kokoro::ops::max_abs_diff;
use rullama_engine::reference::styletts2::{MelFrontend, StyleEncoder};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/bin")
}

fn read_bin(p: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let dir = fixtures_dir();
    assert!(
        dir.is_dir(),
        "fixtures missing — run scripts/styletts2_dump_style_fixtures.py first ({dir:?})"
    );

    // load every <name>.bin into a weight/tensor map
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("bin") {
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            w.insert(name, read_bin(&path));
        }
    }

    let mel = w.get("mel").expect("mel fixture").clone();
    let n_mels = 80;
    let t = mel.len() / n_mels;
    println!("mel: [1, {n_mels}, {t}]  ({} values)", mel.len());

    let mut worst = 0f32;
    for (prefix, _half) in [("acoustic", 0), ("prosodic", 1)] {
        let enc = StyleEncoder::from_weights(&w, prefix);
        let got = enc.forward(&mel, n_mels, t);
        let want = w.get(&format!("{prefix}.style")).expect("style fixture");
        let d = max_abs_diff(&got, want);
        worst = worst.max(d);
        let norm = (got.iter().map(|v| v * v).sum::<f32>()).sqrt();
        println!("{prefix:9} style[128]  max_abs_diff = {d:.3e}   |v|={norm:.3}");
    }

    // also verify the 256-d concat ordering (acoustic ‖ prosodic)
    let a = StyleEncoder::from_weights(&w, "acoustic").forward(&mel, n_mels, t);
    let p = StyleEncoder::from_weights(&w, "prosodic").forward(&mel, n_mels, t);
    let concat: Vec<f32> = a.iter().chain(&p).copied().collect();
    let dc = max_abs_diff(&concat, w.get("concat256").expect("concat256"));
    println!("concat256        max_abs_diff = {dc:.3e}");
    worst = worst.max(dc);

    // ---- mel frontend parity: audio → log-mel vs torchaudio reference ----
    let audio = w.get("audio").expect("audio fixture");
    let fb = w.get("mel_filterbank").expect("mel_filterbank");
    let window = w.get("mel_window").expect("mel_window");
    let front = MelFrontend::new(window, fb);
    let (mel_got, n_frames) = front.compute(audio);
    let dmel = max_abs_diff(&mel_got, &mel);
    println!("\nmel frontend     max_abs_diff = {dmel:.3e}   ([80, {n_frames}] vs [80, {t}])");
    worst = worst.max(dmel);

    // ---- end-to-end: audio → our mel → our encoder → style vs reference ----
    let e2e = StyleEncoder::from_weights(&w, "acoustic").forward(&mel_got, n_mels, n_frames);
    let de2e = max_abs_diff(&e2e, w.get("acoustic.style").unwrap());
    println!("end-to-end       max_abs_diff = {de2e:.3e}   (audio→mel→encoder)");
    worst = worst.max(de2e);

    println!("\nworst max_abs_diff = {worst:.3e}");
    assert!(
        worst < 2e-3,
        "StyleTTS2 cloning-front parity FAILED (worst {worst:.3e})"
    );
    println!("✅ StyleTTS2 style encoder + mel frontend match PyTorch (end-to-end)");

    // ---- GPU StyleEncoder parity (both encoders) vs PyTorch ----
    use rullama_engine::backend::{Pipelines, WgpuCtx};
    use rullama_engine::reference::styletts2::gpu::StyleTtsGpu;
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let mut gwc = HashMap::new();
    // f32 oracle path: empty f16 weight map (the f16-resident variant isn't exercised here).
    let w16: HashMap<String, Vec<u16>> = HashMap::new();
    let gpu_vec = pollster::block_on(
        StyleTtsGpu::new(&w, &w16, &ctx, &pipes, &mut gwc).encode(&mel, n_mels, t),
    );
    let dgpu = max_abs_diff(&gpu_vec, w.get("concat256").unwrap());
    println!(
        "\nGPU encoder vs PyTorch  max_abs_diff = {dgpu:.3e}  (|v|={:.3})",
        (gpu_vec.iter().map(|v| v * v).sum::<f32>()).sqrt()
    );
    assert!(dgpu < 2e-3, "GPU StyleEncoder parity FAILED ({dgpu:.3e})");
    println!("✅ GPU StyleEncoder matches PyTorch");
}
