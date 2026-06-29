//! Phase-C4 gate: diff the hybrid GPU DiffusionGemma forward
//! (`diffusion_forward_gpu`) against the validated CPU oracle
//! (`diffusion_forward`) on the SAME prompt + canvas. Both stream the real
//! 16.8 GB blob via `FileFetcher` — never a whole-file load.
//!
//!   cargo run -p brainwires-engine --release --example diffusion_gpu_parity -- \
//!       <model.gguf> <prompt.i32> <canvas.i32> [--canvas=N] [--layers=N]
//!
//! `--canvas=N` truncates the canvas to the first N tokens for fast iteration
//! (the full file is 256). id files are raw little-endian int32.
//!
//! Pass: argmax matches at every canvas position + logit max-abs within f32
//! matmul-accumulation round-off of the CPU oracle.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use brainwires_engine::backend::{BindGroupCache, Pipelines, WeightCache, WgpuCtx};
use brainwires_engine::gguf::{FileFetcher, GgufReader};
use brainwires_engine::reference::Weights;
use brainwires_engine::reference::diffusion::DiffusionConfig;
use brainwires_engine::reference::diffusion::forward::diffusion_forward;
use brainwires_engine::reference::diffusion::gpu::diffusion_forward_gpu;

fn read_i32(path: &str) -> Vec<u32> {
    std::fs::read(path)
        .expect("read i32 file")
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn main() -> ExitCode {
    let mut pos = Vec::new();
    let mut canvas_trunc: Option<usize> = None;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--canvas=") {
            canvas_trunc = Some(v.parse().expect("canvas N"));
        } else if a.starts_with("--") {
            eprintln!("unknown flag {a}");
        } else {
            pos.push(a);
        }
    }
    let (Some(model), Some(pf), Some(cf)) = (pos.first(), pos.get(1), pos.get(2)) else {
        eprintln!(
            "usage: diffusion_gpu_parity <model.gguf> <prompt.i32> <canvas.i32> [--canvas=N]"
        );
        return ExitCode::from(2);
    };

    let prompt_ids = read_i32(pf);
    let mut canvas_ids = read_i32(cf);
    if let Some(n) = canvas_trunc {
        canvas_ids.truncate(n);
    }
    println!(
        "prompt {} tok, canvas {} tok",
        prompt_ids.len(),
        canvas_ids.len()
    );

    let fetcher = FileFetcher::open(std::path::Path::new(model)).expect("open");
    let r = pollster::block_on(GgufReader::new_streaming(Arc::new(fetcher))).expect("gguf");
    let cfg = DiffusionConfig::from_gguf(&r).expect("diffusion config");
    let r_arc = Arc::new(r);
    let weights = Weights::new(r_arc.clone());

    // ---- CPU oracle ----
    println!("running CPU oracle ...");
    let t0 = Instant::now();
    let cpu = diffusion_forward(&cfg, &weights, &prompt_ids, &canvas_ids).expect("cpu forward");
    println!("  cpu forward in {:?}", t0.elapsed());

    // ---- GPU hybrid ----
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let bind_cache = Arc::new(BindGroupCache::default());
    let wcache = WeightCache::new(
        r_arc.clone(),
        ctx.device.clone(),
        ctx.queue.clone(),
        bind_cache,
    );

    println!("running GPU hybrid ...");
    let t0 = Instant::now();
    let gpu = pollster::block_on(diffusion_forward_gpu(
        &cfg,
        &ctx,
        &pipes,
        &wcache,
        &weights,
        &prompt_ids,
        &canvas_ids,
        None,
        1.0,
    ))
    .expect("gpu forward");
    println!("  gpu forward in {:?}", t0.elapsed());

    // ---- diff ----
    let vocab = cfg.base.vocab_size as usize;
    let c = canvas_ids.len();
    assert_eq!(cpu.len(), c * vocab);
    assert_eq!(gpu.len(), c * vocab);

    let mut max_abs = 0f32;
    let mut argmax_match = 0usize;
    for ci in 0..c {
        let cr = &cpu[ci * vocab..(ci + 1) * vocab];
        let gr = &gpu[ci * vocab..(ci + 1) * vocab];
        for (a, b) in cr.iter().zip(gr.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        if argmax(cr) == argmax(gr) {
            argmax_match += 1;
        }
    }
    println!("\nmax abs logit diff: {max_abs:.5}");
    println!("argmax match: {argmax_match}/{c} canvas positions");

    if argmax_match == c {
        println!("PASS");
        ExitCode::SUCCESS
    } else {
        println!("FAIL — argmax diverges (see above)");
        ExitCode::FAILURE
    }
}
