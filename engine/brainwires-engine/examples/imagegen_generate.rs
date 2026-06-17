//! End-to-end Z-Image generation (CPU oracle): caption tokens → image, wiring
//! the Qwen3 encoder + S3-DiT denoise loop + VAE decoder + flow-match scheduler.
//! Writes a PPM. Proves the whole pipeline runs on real weights.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_generate -- \
//!       weights/Z-Image-Turbo  8 8 2 0 /tmp/zimage.ppm
//!   args: <model_dir> <latent_h> <latent_w> <steps> <seed> <out.ppm>
//!
//! NOTE: the naive CPU DiT re-dequantizes 24GB/step, so keep latent + steps
//! tiny — this validates the pipeline, not speed. (GPU path is future work.)

use rullama::imagegen::{Qwen3Config, ShardedSafetensors, TransformerConfig, VaeConfig};
use rullama::reference::pipeline::{generate, Components};

fn main() {
    let mut a = std::env::args().skip(1);
    let root = a.next().unwrap_or_else(|| "weights/Z-Image-Turbo".to_string());
    let lh: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let lw: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let steps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = a.next().unwrap_or_else(|| "/tmp/zimage.ppm".to_string());

    let rd = |p: String| std::fs::read(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let enc_cfg = Qwen3Config::parse(&rd(format!("{root}/text_encoder/config.json"))).unwrap();
    let dit_cfg = TransformerConfig::parse(&rd(format!("{root}/transformer/config.json"))).unwrap();
    let vae_cfg = VaeConfig::parse(&rd(format!("{root}/vae/config.json"))).unwrap();

    println!("loading weights (text_encoder + transformer + vae)...");
    let enc_st = ShardedSafetensors::open_dir(format!("{root}/text_encoder"), "model.safetensors.index.json").unwrap();
    let dit_st = ShardedSafetensors::open_dir(format!("{root}/transformer"), "diffusion_pytorch_model.safetensors.index.json").unwrap();
    let vae_st = ShardedSafetensors::open_single(format!("{root}/vae/diffusion_pytorch_model.safetensors")).unwrap();

    let comps = Components {
        enc_st: &enc_st, enc_cfg: &enc_cfg,
        dit_st: &dit_st, dit_cfg: &dit_cfg,
        vae_st: &vae_st, vae_cfg: &vae_cfg,
    };

    // Synthetic caption tokens (Qwen2 tokenizer is a separate piece).
    let tokens: Vec<u32> = vec![151644, 9707, 11, 1879, 13, 151645];
    let down = vae_cfg.downscale() as usize;
    println!("generating {}×{} image, {steps} steps, seed {seed}...", lw * down, lh * down);

    let t0 = std::time::Instant::now();
    let prog = |stage: &str, i: usize, n: usize| println!("  [{stage}] {}/{n}", i + 1);
    let rgb = generate(&comps, &tokens, lh, lw, steps, seed, Some(&prog)).expect("generate");
    println!("done in {:.1?}", t0.elapsed());

    let (h, w) = (lh * down, lw * down);
    assert_eq!(rgb.len(), 3 * h * w);
    let finite = rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v));
    let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
    println!("image [3,{h},{w}]: valid={finite} mean={mean:.4}");

    // write PPM (P6): channel-first [3,H,W] → interleaved RGB u8
    let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                buf.push((rgb[ch * h * w + y * w + x] * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    std::fs::write(&out, &buf).expect("write ppm");
    println!("wrote {out} ({} bytes)", buf.len());
    assert!(finite, "invalid image");
    println!("\nOK — full Z-Image pipeline generated an image on real weights.");
}
