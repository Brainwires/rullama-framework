//! Real Metal/GPU execution test of the **scaled-fp8 DiT path** on the actual
//! lightx2v `z_image_turbo_scaled_fp8_e4m3fn` weights (single-file, no shard
//! index). Streams the fp8 file via `StreamingShards::open_single` — the exact
//! loader the browser uses — and runs one `DitGpu::forward` on the GPU,
//! exercising the new `F8_E4M3 × weight_scale[row]` reconstruction +
//! `matmul_bf16` kernels end to end. Asserts the output is finite and sane.
//!
//! This does NOT compare against the bf16 original (that's a 24 GB download and
//! fp8 is lossy by design); it proves the fp8 path *executes correctly on the
//! GPU and is numerically well-behaved*. The real quality check is the
//! in-browser generated image.
//!
//! Usage:
//!   cargo run -p brainwires-engine --release --example imagegen_fp8_smoke -- \
//!       weights/Z-Image-Turbo-fp8 4 4 3

use brainwires_engine::backend::{Pipelines, WgpuCtx};
use brainwires_engine::imagegen::{DitGpu, FileBlobSource, StreamingShards, TransformerConfig};

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo-fp8".to_string());
    let lh: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let lw: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let cap_len: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg =
        TransformerConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
            .unwrap();

    // Single-file fp8 (no shard index) — the open_single path the loader falls
    // back to in the browser.
    let ss = pollster::block_on(StreamingShards::open_single(
        FileBlobSource::new(&dir),
        "diffusion_pytorch_model.safetensors",
    ))
    .expect("open fp8 single-file");

    // Confirm we're really looking at the scaled-fp8 build: a representative
    // weight is F8_E4M3 and carries a companion per-row weight_scale.
    let probe = "layers.0.feed_forward.w2.weight";
    println!(
        "{probe}: dtype={:?}  has weight_scale={}",
        ss.dtype(probe),
        ss.has("layers.0.feed_forward.w2.weight_scale"),
    );

    let cin = cfg.in_channels as usize;
    let latent: Vec<f32> = (0..cin * lh * lw)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let cap: Vec<f32> = (0..cap_len * cfg.cap_feat_dim as usize)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
        .collect();
    let t = 0.7f32;

    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let info = ctx.adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.backend);

    println!("scaled-fp8 DiT forward on GPU (lh={lh} lw={lw} cap_len={cap_len})...");
    let t0 = std::time::Instant::now();
    let pipes = Pipelines::new(&ctx.device);
    let out = pollster::block_on(
        DitGpu::new(&ctx, &pipes, &ss, &cfg).forward(&latent, lh, lw, t, &cap, cap_len, None),
    )
    .expect("fp8 DiT forward");
    println!("done in {:.1?}", t0.elapsed());

    let n = out.len();
    let finite = out.iter().all(|v| v.is_finite());
    let (mn, mx) = out
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let mean = out.iter().sum::<f32>() / n as f32;
    let var = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    println!(
        "velocity [{n}]: finite={finite} min={mn:.4} max={mx:.4} mean={mean:.4} std={:.4}",
        var.sqrt()
    );
    println!("first 8: {:?}", &out[..8.min(n)]);

    assert!(finite, "fp8 DiT output has NaN/Inf");
    assert!(mx.abs() < 1e4 && mn.abs() < 1e4, "fp8 DiT output magnitude exploded");
    println!("\nOK — scaled-fp8 DiT executed on the GPU with finite, sane output.");
}
