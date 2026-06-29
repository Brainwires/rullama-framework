//! GPU VAE decode vs the CPU oracle on real weights — validates the first full
//! GPU component forward (conv/groupnorm/silu/upsample/residual on GPU, mid-attn
//! via readback) against reference::vae, and reports the GPU speedup.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_vae_gpu -- \
//!       weights/Z-Image-Turbo/vae 4

use rullama::backend::{Pipelines, WgpuCtx};
use rullama::imagegen::{FileBlobSource, ShardedSafetensors, StreamingShards, VaeConfig};
use rullama::reference::vae::VaeDecoder;
use rullama::reference::vae_gpu::VaeGpu;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo/vae".to_string());
    let lsz: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    let cfg =
        VaeConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config")).unwrap();
    // CPU oracle reads via the sync sharded loader; the GPU path streams its
    // weights via StreamingShards<FileBlobSource> — the exact wasm code path,
    // exercised here natively against the oracle.
    let st = ShardedSafetensors::open_single(format!("{dir}/diffusion_pytorch_model.safetensors"))
        .unwrap();
    let ss = pollster::block_on(StreamingShards::open_single(
        FileBlobSource::new(&dir),
        "diffusion_pytorch_model.safetensors",
    ))
    .unwrap();
    let lc = cfg.latent_channels as usize;
    let latent: Vec<f32> = (0..lc * lsz * lsz)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();

    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);

    println!(
        "CPU decode [{lc},{lsz},{lsz}] → RGB [3,{},{}]...",
        lsz * 8,
        lsz * 8
    );
    let t0 = std::time::Instant::now();
    let cpu = VaeDecoder::new(&st, &cfg)
        .decode(&latent, lsz, lsz)
        .expect("cpu decode");
    let cpu_dt = t0.elapsed();

    println!("GPU decode (streaming)...");
    let t1 = std::time::Instant::now();
    let gpu = pollster::block_on(VaeGpu::new(&ctx, &pipes, &ss, &cfg).decode(&latent, lsz, lsz))
        .expect("gpu decode");
    let gpu_dt = t1.elapsed();

    assert_eq!(cpu.len(), gpu.len());
    let md = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let gpu_finite = gpu.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v));
    println!(
        "CPU {cpu_dt:.2?}  vs  GPU {gpu_dt:.2?}  ({:.1}× )",
        cpu_dt.as_secs_f64() / gpu_dt.as_secs_f64().max(1e-9)
    );
    println!("GPU image valid={gpu_finite}, max|GPU-CPU| = {md:.6}");
    assert!(gpu_finite, "GPU image invalid");
    assert!(md < 2e-3, "GPU-vs-CPU VAE max_diff = {md} (too high)");
    println!("\nOK — GPU VAE decoder matches the CPU oracle on real weights.");
}
