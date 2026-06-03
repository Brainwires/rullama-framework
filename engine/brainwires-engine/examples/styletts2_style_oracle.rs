//! StyleTTS2 style-encoder CPU-oracle parity check.
//!
//! Loads the fixtures dumped by `scripts/styletts2_dump_style_fixtures.py`
//! (~/.cache/styletts2/fixtures/bin/), runs the Rust `StyleEncoder` oracle for both
//! the acoustic and prosodic encoders on the reference mel, and diffs the resulting
//! 128-d style vectors against the PyTorch reference. This is the cloning encoder's
//! GPU-vs-CPU-style parity gate (here CPU-oracle vs PyTorch).
//!
//!   cargo run -p rullama --release --example styletts2_style_oracle

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rullama::reference::kokoro::ops::max_abs_diff;
use rullama::reference::styletts2::StyleEncoder;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/bin")
}

fn read_bin(p: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let dir = fixtures_dir();
    assert!(dir.is_dir(), "fixtures missing — run scripts/styletts2_dump_style_fixtures.py first ({dir:?})");

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

    println!("\nworst max_abs_diff = {worst:.3e}");
    assert!(worst < 2e-3, "style-encoder parity FAILED (worst {worst:.3e})");
    println!("✅ StyleTTS2 style-encoder CPU oracle matches PyTorch");
}
