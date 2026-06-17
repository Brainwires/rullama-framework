//! Run one S3-DiT denoise step (reference::dit) on the real Z-Image transformer
//! weights and report the predicted-velocity latent. Validates the full DiT
//! forward (patch/cap/t-embed → refiners → 30 adaLN layers w/ multi-axis RoPE →
//! final → unpatchify) end-to-end on ground-truth weights.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_dit_forward -- \
//!       weights/Z-Image-Turbo/transformer  4 4 3   # latent 4×4, cap_len 3
//!
//! Small latent + short caption keep the naive CPU forward tractable; this
//! validates the math, not speed.

use rullama::imagegen::{ShardedSafetensors, TransformerConfig};
use rullama::reference::dit::DitForward;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/transformer".to_string());
    let lh: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let lw: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let cap_len: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg =
        TransformerConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
            .expect("parse DiT config");
    let st = ShardedSafetensors::open_dir(&dir, "diffusion_pytorch_model.safetensors.index.json")
        .expect("open DiT");

    let cin = cfg.in_channels as usize;
    // Deterministic synthetic latent + caption features (no rng).
    let latent: Vec<f32> = (0..cin * lh * lw)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let cap: Vec<f32> = (0..cap_len * cfg.cap_feat_dim as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
        .collect();
    let t = 0.7f32; // a mid-schedule sigma

    println!(
        "DiT one-step: latent [{cin},{lh},{lw}] ({} img tokens), cap_len {cap_len}, t={t}",
        (lh / cfg.patch_size() as usize) * (lw / cfg.patch_size() as usize)
    );
    let t0 = std::time::Instant::now();
    let dit = DitForward::new(&st, &cfg);
    let vel = dit
        .forward(&latent, lh, lw, t, &cap, cap_len)
        .expect("forward");
    let dt = t0.elapsed();

    assert_eq!(vel.len(), cin * lh * lw, "output shape");
    let finite = vel.iter().all(|v| v.is_finite());
    let l2 = (vel.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let (mn, mx) = vel
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("done in {dt:.2?}");
    println!("velocity [{cin},{lh},{lw}]: finite={finite} L2={l2:.3} min={mn:.3} max={mx:.3}");
    println!("  [0..6]={:?}", &vel[..6.min(vel.len())]);
    assert!(finite, "non-finite DiT output");
    println!("\nOK — S3-DiT forward ran clean on real weights.");
}
