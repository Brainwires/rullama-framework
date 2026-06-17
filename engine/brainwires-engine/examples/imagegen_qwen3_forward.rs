//! Run the Qwen3 text-encoder CPU forward (reference::qwen3) on the real
//! Z-Image weights and report the output hidden state. Validates the full
//! 36-layer encoder math end-to-end (embedding → GQA attention w/ QK-norm +
//! RoPE → SwiGLU → final norm) on ground-truth weights.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_qwen3_forward -- \
//!       weights/Z-Image-Turbo/text_encoder  1 2 3 4 5
//!
//! (Token ids after the dir are optional; default a short synthetic sequence.
//!  Real tokenization is a separate piece — this validates the forward math.)

use rullama::imagegen::{Qwen3Config, ShardedSafetensors};
use rullama::reference::qwen3::Qwen3Encoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/text_encoder".to_string());
    let tokens: Vec<u32> = {
        let v: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();
        if v.is_empty() {
            vec![151644, 9707, 11, 1879, 13, 151645]
        } else {
            v
        }
    };

    let cfg = Qwen3Config::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
        .expect("parse config");
    let st = ShardedSafetensors::open_dir(&dir, "model.safetensors.index.json").expect("open");

    println!(
        "encoding {} tokens through {} layers (hidden {})...",
        tokens.len(),
        cfg.num_hidden_layers,
        cfg.hidden_size
    );
    let t0 = std::time::Instant::now();
    let enc = Qwen3Encoder::new(&st, &cfg);
    let out = enc.forward(&tokens).expect("forward");
    let dt = t0.elapsed();

    let h = cfg.hidden_size as usize;
    assert_eq!(out.len(), tokens.len() * h, "output shape");
    let finite = out.iter().all(|v| v.is_finite());
    let l2 = |row: &[f32]| row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let first = &out[..h];
    let last = &out[(tokens.len() - 1) * h..];

    println!("done in {dt:.2?}");
    println!("output [{}, {h}], all finite = {finite}", tokens.len());
    println!(
        "  token0 hidden: L2={:.3}  [0..4]={:?}",
        l2(first),
        &first[..4]
    );
    println!(
        "  tokenN hidden: L2={:.3}  [0..4]={:?}",
        l2(last),
        &last[..4]
    );
    assert!(finite, "non-finite output");
    println!("\nOK — Qwen3 encoder forward ran clean on real weights.");
}
