//! GPU-matmul DiT forward vs the CPU oracle on real weights: validates that
//! routing the DiT linears through the GPU bf16 kernel (everything else
//! unchanged) matches reference::dit, and reports the speedup.
//!
//! Usage:
//!   cargo run -p brainwires-engine --release --example imagegen_dit_gpu -- \
//!       weights/Z-Image-Turbo/transformer 4 4 3

use brainwires_engine::backend::{Pipelines, WgpuCtx};
use brainwires_engine::imagegen::{ShardedSafetensors, TransformerConfig};
use brainwires_engine::reference::dit::DitForward;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/transformer".to_string());
    let lh: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let lw: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let cap_len: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg =
        TransformerConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
            .unwrap();
    let st = ShardedSafetensors::open_dir(&dir, "diffusion_pytorch_model.safetensors.index.json")
        .unwrap();
    let cin = cfg.in_channels as usize;
    let latent: Vec<f32> = (0..cin * lh * lw)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let cap: Vec<f32> = (0..cap_len * cfg.cap_feat_dim as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
        .collect();
    let t = 0.7f32;

    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);

    println!("CPU DiT forward...");
    let t0 = std::time::Instant::now();
    let cpu = DitForward::new(&st, &cfg)
        .forward(&latent, lh, lw, t, &cap, cap_len)
        .expect("cpu");
    let cpu_dt = t0.elapsed();

    println!("GPU-matmul DiT forward...");
    let t1 = std::time::Instant::now();
    let gpu = DitForward::with_gpu(&st, &cfg, &ctx, &pipes)
        .forward(&latent, lh, lw, t, &cap, cap_len)
        .expect("gpu");
    let gpu_dt = t1.elapsed();

    let md = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let rel = md / cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    println!(
        "CPU {cpu_dt:.1?}  vs  GPU {gpu_dt:.1?}  ({:.1}×)",
        cpu_dt.as_secs_f64() / gpu_dt.as_secs_f64().max(1e-9)
    );
    println!(
        "max|GPU-CPU| = {md:.5} (rel {rel:.4}), GPU finite = {}",
        gpu.iter().all(|v| v.is_finite())
    );
    assert!(gpu.iter().all(|v| v.is_finite()), "non-finite GPU output");
    assert!(
        rel < 0.05,
        "GPU-vs-CPU DiT rel diff {rel} too high (bf16 matmul expected ~1e-2)"
    );
    println!("\nOK — GPU-matmul DiT matches the CPU oracle (within bf16 tolerance).");
}
