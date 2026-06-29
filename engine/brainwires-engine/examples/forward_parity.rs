//! M3 closure smoke test: run a single forward step on CPU and GPU for token=<bos>
//! at pos=0, compare logits and top-1.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example forward_parity -- <gguf>

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use rullama::backend::{Pipelines, WeightCache, WgpuCtx};
use rullama::gguf::GgufReader;
use rullama::model::config::Gemma4Config;
use rullama::reference::{KvState, Weights, forward_token, forward_token_gpu};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: forward_parity <gguf>");
            return ExitCode::from(2);
        }
    };
    let bytes = fs::read(&path).expect("read");
    let reader = GgufReader::new(bytes).expect("parse");
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
    let r_arc = std::sync::Arc::new(reader);
    let weights = Weights::new(r_arc.clone());

    let bos = cfg.bos_id.unwrap_or(2);

    println!("loading wgpu + compiling pipelines...");
    let t0 = Instant::now();
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let wcache = WeightCache::new(
        r_arc.clone(),
        ctx.device.clone(),
        ctx.queue.clone(),
        std::sync::Arc::clone(&ctx.bind_cache),
    );
    println!("  done in {:?}", t0.elapsed());

    println!("CPU forward at pos=0, token={bos} ...");
    let mut kv_cpu = KvState::new(&cfg);
    let t0 = Instant::now();
    let logits_cpu = forward_token(&cfg, &weights, &mut kv_cpu, bos, 0).expect("cpu");
    let dt_cpu = t0.elapsed();

    // Run two GPU forwards back-to-back. The first uploads weights to GPU on demand
    // (cold cache); the second reuses them (hot cache) — measures the steady-state
    // per-token cost we'd see during multi-token decode.
    println!("GPU forward #1 (cold cache) at pos=0 ...");
    let mut kv_gpu = KvState::new(&cfg);
    let t0 = Instant::now();
    let logits_gpu = pollster::block_on(forward_token_gpu(
        &cfg,
        &weights,
        &wcache,
        &ctx,
        &pipes,
        &mut kv_gpu,
        bos,
        0,
    ))
    .expect("gpu");
    let dt_gpu_cold = t0.elapsed();

    println!("GPU forward #2 (hot cache, fresh KV) at pos=0 ...");
    let mut kv_gpu2 = KvState::new(&cfg);
    let t0 = Instant::now();
    let _logits_gpu2 = pollster::block_on(forward_token_gpu(
        &cfg,
        &weights,
        &wcache,
        &ctx,
        &pipes,
        &mut kv_gpu2,
        bos,
        0,
    ))
    .expect("gpu hot");
    let dt_gpu_hot = t0.elapsed();

    println!("CPU forward: {dt_cpu:?}");
    println!(
        "GPU forward (cold): {dt_gpu_cold:?}  → speedup vs CPU: {:.1}x",
        dt_cpu.as_secs_f64() / dt_gpu_cold.as_secs_f64()
    );
    println!(
        "GPU forward (hot):  {dt_gpu_hot:?}  → speedup vs CPU: {:.1}x",
        dt_cpu.as_secs_f64() / dt_gpu_hot.as_secs_f64()
    );
    println!(
        "WeightCache: {} tensors, {:.1} MB on GPU",
        wcache.cached_count(),
        wcache.cached_bytes() as f64 / 1e6
    );

    // Distribution diff
    let n = logits_cpu.len();
    assert_eq!(n, logits_gpu.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut nans = 0usize;
    for i in 0..n {
        let c = logits_cpu[i];
        let g = logits_gpu[i];
        if g.is_nan() {
            nans += 1;
            continue;
        }
        let abs = (g - c).abs();
        let rel = if c.abs() > 1e-3 { abs / c.abs() } else { 0.0 };
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    println!("logit diff: max_abs={max_abs:.5}, max_rel={max_rel:.5}, gpu_nans={nans}");

    let mut cpu_top: Vec<(usize, f32)> = logits_cpu.iter().copied().enumerate().collect();
    cpu_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut gpu_top: Vec<(usize, f32)> = logits_gpu.iter().copied().enumerate().collect();
    gpu_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "CPU top-5: {:?}",
        cpu_top.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "GPU top-5: {:?}",
        gpu_top.iter().take(5).collect::<Vec<_>>()
    );

    let cpu_argmax = cpu_top[0].0;
    let gpu_argmax = gpu_top[0].0;

    if cpu_argmax == gpu_argmax {
        println!("PASS: top-1 token matches: {cpu_argmax}");
        ExitCode::SUCCESS
    } else {
        println!("FAIL: top-1 mismatch: cpu={cpu_argmax} gpu={gpu_argmax}");
        ExitCode::from(1)
    }
}
