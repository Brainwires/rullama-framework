//! Run the VAE decoder CPU forward (reference::vae) on the real Z-Image VAE
//! weights, decoding a small synthetic latent → RGB. Validates the full decoder
//! math (conv_in → mid resnet/attn/resnet → up-blocks w/ upsample → groupnorm →
//! conv_out → [0,1]) on ground-truth weights.
//!
//! Usage:
//!   cargo run -p rullama-engine --release --example imagegen_vae_decode -- \
//!       weights/Z-Image-Turbo/vae  3   # latent 3×3 → 24×24 RGB
//!
//! A small latent keeps the naive CPU conv tractable; correctness, not speed.

use rullama_engine::imagegen::{ShardedSafetensors, VaeConfig};
use rullama_engine::reference::vae::VaeDecoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/vae".to_string());
    let lsz: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg = VaeConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
        .expect("parse VAE config");
    let st = ShardedSafetensors::open_single(format!("{dir}/diffusion_pytorch_model.safetensors"))
        .expect("open VAE");

    let lc = cfg.latent_channels as usize;
    // Deterministic synthetic latent (no rng): a smooth ramp per channel.
    let latent: Vec<f32> = (0..lc * lsz * lsz)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();

    println!(
        "decoding latent [{lc},{lsz},{lsz}] → RGB [{},{},{}] ({}× upscale)...",
        cfg.out_channels,
        lsz * cfg.downscale() as usize,
        lsz * cfg.downscale() as usize,
        cfg.downscale()
    );
    let t0 = std::time::Instant::now();
    let dec = VaeDecoder::new(&st, &cfg);
    let rgb = dec.decode(&latent, lsz, lsz).expect("decode");
    let dt = t0.elapsed();

    let px = lsz * cfg.downscale() as usize;
    assert_eq!(rgb.len(), 3 * px * px, "output shape");
    let finite = rgb.iter().all(|v| v.is_finite());
    let in_range = rgb.iter().all(|v| (0.0..=1.0).contains(v));
    let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
    let (mn, mx) = rgb
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));

    println!("done in {dt:.2?}");
    println!(
        "RGB [3,{px},{px}]: finite={finite} in[0,1]={in_range} mean={mean:.4} min={mn:.4} max={mx:.4}"
    );
    assert!(finite && in_range, "VAE output not a valid image");
    println!("\nOK — VAE decoder forward ran clean on real weights.");
}
