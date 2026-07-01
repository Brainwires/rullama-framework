//! VAE-in-isolation sanity check. Decodes a deliberately SMOOTH latent (low-
//! frequency per channel) through the FLUX VAE on the GPU. A correct decoder
//! turns smooth latents into smooth color regions; a broken one yields static.
//! This separates "VAE is wrong" from "the diffusion produces a noisy latent".
//!
//!   cargo run -p rullama-engine --release --example imagegen_vae_check -- \
//!       weights/Z-Image-Turbo/vae 32 /tmp/vae_check.ppm

use rullama_engine::backend::{Pipelines, WgpuCtx};
use rullama_engine::imagegen::{FileBlobSource, StreamingShards, VaeConfig};
use rullama_engine::reference::vae_gpu::VaeGpu;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/vae".to_string());
    let l: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let out = a.next().unwrap_or_else(|| "/tmp/vae_check.ppm".to_string());

    let cfg = VaeConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("cfg")).unwrap();
    let lc = cfg.latent_channels as usize;
    println!(
        "vae: latent_channels={lc} scaling={} shift={} downscale={}",
        cfg.scaling_factor,
        cfg.shift_factor,
        cfg.downscale()
    );

    // Smooth latent: each channel a low-frequency 2D sinusoid (period ~ whole
    // canvas), amplitude ~1. In-distribution-ish, definitely NOT noise.
    let mut latent = vec![0.0f32; lc * l * l];
    for c in 0..lc {
        let fx = 1.0 + (c % 3) as f32;
        let fy = 1.0 + (c % 2) as f32;
        let ph = (c as f32) * 0.5;
        for y in 0..l {
            for x in 0..l {
                let v = (std::f32::consts::TAU * fx * x as f32 / l as f32 + ph).sin()
                    * (std::f32::consts::TAU * fy * y as f32 / l as f32).cos();
                latent[c * l * l + y * l + x] = v;
            }
        }
    }

    let ss = pollster::block_on(StreamingShards::open_single(
        FileBlobSource::new(&dir),
        "diffusion_pytorch_model.safetensors",
    ))
    .expect("open vae");
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    println!(
        "decoding smooth {l}×{l} latent → {}×{} image...",
        l * 8,
        l * 8
    );
    let rgb = pollster::block_on(VaeGpu::new(&ctx, &pipes, &ss, &cfg).decode(&latent, l, l))
        .expect("decode");

    let (h, w) = (l * 8, l * 8);
    let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
    // Local smoothness metric: mean |Δ| between horizontally adjacent pixels
    // (per channel). Smooth image ⇒ small; static ⇒ large (~0.3+).
    let mut tv = 0.0f64;
    let mut cnt = 0u64;
    for ch in 0..3 {
        for y in 0..h {
            for x in 1..w {
                let a = rgb[ch * h * w + y * w + x];
                let b = rgb[ch * h * w + y * w + x - 1];
                tv += (a - b).abs() as f64;
                cnt += 1;
            }
        }
    }
    let tv = tv / cnt as f64;
    println!("image mean={mean:.4}  mean|Δx|={tv:.4}  (smooth≪0.1, static≳0.3)");

    let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                buf.push(
                    (rgb[ch * h * w + y * w + x] * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                );
            }
        }
    }
    std::fs::write(&out, &buf).expect("write");
    println!("wrote {out}");
}
