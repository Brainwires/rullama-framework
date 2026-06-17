//! Async-streaming GPU DiT forward (the wasm code path) vs the CPU oracle on
//! real weights. Drives `imagegen::DitGpu` over `StreamingShards<FileBlobSource>`
//! — the identical loader the browser uses (HttpRangeBlobSource) — and diffs
//! against `reference::dit::DitForward`.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_dit_stream -- \
//!       weights/Z-Image-Turbo/transformer 4 4 3

use rullama::backend::{Pipelines, WgpuCtx};
use rullama::imagegen::{
    DitGpu, FileBlobSource, ShardedSafetensors, StreamingShards, TransformerConfig,
};
use rullama::reference::dit::DitForward;

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
    // CPU oracle via the sync sharded loader; GPU path via the streaming loader.
    let st = ShardedSafetensors::open_dir(&dir, "diffusion_pytorch_model.safetensors.index.json")
        .unwrap();
    let ss = pollster::block_on(StreamingShards::open_index(
        FileBlobSource::new(&dir),
        st.index(),
    ))
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

    println!("Async-streaming GPU DiT forward...");
    let t1 = std::time::Instant::now();
    let gpu = pollster::block_on(
        DitGpu::new(&ctx, &pipes, &ss, &cfg).forward(&latent, lh, lw, t, &cap, cap_len),
    )
    .expect("gpu");
    let gpu_dt = t1.elapsed();

    let md = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let rel = md / cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    println!("CPU {cpu_dt:.1?}  vs  GPU(stream) {gpu_dt:.1?}",);
    println!(
        "max|GPU-CPU| = {md:.5} (rel {rel:.4}), GPU finite = {}",
        gpu.iter().all(|v| v.is_finite())
    );
    assert!(gpu.iter().all(|v| v.is_finite()), "non-finite GPU output");
    assert!(
        rel < 0.05,
        "GPU-vs-CPU DiT rel diff {rel} too high (bf16 matmul expected ~1e-2)"
    );
    println!("\nOK — async-streaming GPU DiT matches the CPU oracle (within bf16 tolerance).");
}
