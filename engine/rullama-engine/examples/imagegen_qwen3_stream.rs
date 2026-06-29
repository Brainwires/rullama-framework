//! Async-streaming GPU Qwen3 encoder (the wasm code path) vs the CPU oracle on
//! real weights. Drives `imagegen::Qwen3Gpu` over `StreamingShards<FileBlobSource>`
//! — the loader the browser uses — and diffs against `reference::qwen3`.
//!
//! Usage:
//!   cargo run -p rullama-engine --release --example imagegen_qwen3_stream -- \
//!       weights/Z-Image-Turbo/text_encoder  151644 9707 11 1879 13 151645

use rullama_engine::backend::{Pipelines, WgpuCtx};
use rullama_engine::imagegen::{
    FileBlobSource, Qwen3Config, Qwen3Gpu, ShardedSafetensors, StreamingShards,
};
use rullama_engine::reference::qwen3::Qwen3Encoder;

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
    let ss = pollster::block_on(StreamingShards::open_index(
        FileBlobSource::new(&dir),
        st.index(),
    ))
    .expect("stream open");

    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);

    println!("CPU encode {} tokens...", tokens.len());
    let t0 = std::time::Instant::now();
    let cpu = Qwen3Encoder::new(&st, &cfg).forward(&tokens).expect("cpu");
    let cpu_dt = t0.elapsed();

    println!("Async-streaming GPU encode...");
    let t1 = std::time::Instant::now();
    let gpu =
        pollster::block_on(Qwen3Gpu::new(&ctx, &pipes, &ss, &cfg).forward(&tokens, None))
            .expect("gpu");
    let gpu_dt = t1.elapsed();

    assert_eq!(cpu.len(), gpu.len());
    let md = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let rel = md / cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    println!("CPU {cpu_dt:.1?}  vs  GPU(stream) {gpu_dt:.1?}");
    println!(
        "max|GPU-CPU| = {md:.5} (rel {rel:.4}), GPU finite = {}",
        gpu.iter().all(|v| v.is_finite())
    );
    assert!(gpu.iter().all(|v| v.is_finite()), "non-finite GPU output");
    assert!(
        rel < 0.06,
        "GPU-vs-CPU Qwen3 rel diff {rel} too high (bf16 matmul expected ~1e-2)"
    );
    println!(
        "\nOK — async-streaming GPU Qwen3 encoder matches the CPU oracle (within bf16 tolerance)."
    );
}
