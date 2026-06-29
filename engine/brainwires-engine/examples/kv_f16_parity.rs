//! Parity check: f16 KV cache vs f32 KV cache, both on the GPU forward.
//!
//! Halving the KV cache to packed f16 (attention_f16kv + pack_f16_row) must not
//! change which token the model picks. This drives the SAME token sequence
//! through an f32 `Forward` and an f16 `Forward` and compares per-position
//! logits: top-1 and top-5 must be identical every step; the raw logit delta is
//! reported (f16 rounding ~1e-3 accumulates over the history dot-products, so a
//! few e-3 of max-abs is expected — the load-bearing assertion is the argmax
//! match, not the delta).
//!
//! Build:
//!   cargo run -p brainwires-engine --release --example kv_f16_parity -- <gguf> [user_msg] [--max=N]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use brainwires_engine::api::{ChatMessage, ChatRole};
use brainwires_engine::backend::{Pipelines, WeightCache, WgpuCtx};
use brainwires_engine::gguf::GgufReader;
use brainwires_engine::model::config::Gemma4Config;
use brainwires_engine::reference::Weights;
use brainwires_engine::reference::forward_chained::Forward;
use brainwires_engine::sampling::{Sampler, SamplingOptions};
use brainwires_engine::template::gemma4_small;
use brainwires_engine::tokenizer::BpeTokenizer;

/// Indices of the `k` largest logits, descending.
fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

fn build_forward(
    reader: &Arc<GgufReader>,
    cfg: &Gemma4Config,
    kv_f16: bool,
    max_context: u32,
) -> Forward {
    let weights = Weights::new(reader.clone());
    let ctx = pollster::block_on(WgpuCtx::new()).expect("WgpuCtx");
    let pipes = Arc::new(Pipelines::new(&ctx.device));
    let wcache = Arc::new(WeightCache::new(
        reader.clone(),
        ctx.device.clone(),
        ctx.queue.clone(),
        Arc::clone(&ctx.bind_cache),
    ));
    pollster::block_on(Forward::new_with_max_context(
        cfg.clone(),
        ctx,
        pipes,
        weights,
        wcache,
        max_context,
        kv_f16,
    ))
    .expect("Forward::new_with_max_context")
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: kv_f16_parity <gguf> [user_msg] [--max=N]");
            return ExitCode::from(2);
        }
    };
    let mut user_msg = String::from("The capital of France is");
    let mut max_tokens: usize = 16;
    for a in args {
        if let Some(rest) = a.strip_prefix("--max=") {
            max_tokens = rest.parse().unwrap_or(16);
        } else {
            user_msg = a;
        }
    }

    println!("loading model ...");
    let bytes = fs::read(&path).expect("read");
    let reader = Arc::new(GgufReader::new(bytes).expect("parse"));
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
    let tok = BpeTokenizer::from_gguf(&reader).expect("tokenizer");

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: user_msg.clone(),
    }];
    let prompt = gemma4_small::render_for_completion(&messages, false);
    let prompt_ids = tok.encode(&prompt);
    println!(
        "prompt has {} tokens; generating up to {max_tokens}",
        prompt_ids.len()
    );

    // ---- Pass 1: f32 KV. Greedy-generate to FIX the token sequence + record logits.
    let mut f32_fwd = build_forward(&reader, &cfg, false, 4096);
    let mut sampler = Sampler::new(SamplingOptions::greedy());
    let mut inputs: Vec<u32> = Vec::new(); // every token fed, in order
    let mut logits_f32: Vec<Vec<f32>> = Vec::new();

    let mut next: u32 = 0;
    for &id in &prompt_ids {
        inputs.push(id);
        let logits = pollster::block_on(f32_fwd.step(id)).expect("f32 step");
        next = sampler.sample(&logits);
        logits_f32.push(logits);
    }
    for _ in 0..max_tokens {
        if cfg.eos_ids.contains(&next) {
            break;
        }
        inputs.push(next);
        let logits = pollster::block_on(f32_fwd.step(next)).expect("f32 gen");
        next = sampler.sample(&logits);
        logits_f32.push(logits);
    }
    drop(f32_fwd); // free the f32 device before building the f16 one
    println!("f32 pass: {} positions", inputs.len());

    // ---- Pass 2: f16 KV. Replay the EXACT same input sequence, record logits.
    let mut f16_fwd = build_forward(&reader, &cfg, true, 4096);
    let mut logits_f16: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    for &id in &inputs {
        let logits = pollster::block_on(f16_fwd.step(id)).expect("f16 step");
        logits_f16.push(logits);
    }
    drop(f16_fwd);
    println!("f16 pass: {} positions", logits_f16.len());

    // ---- Compare ----
    let mut top1_mismatches = 0usize;
    let mut top5_mismatches = 0usize;
    let mut max_abs = 0f32;
    let mut worst_pos = 0usize;
    for (p, (lf, lh)) in logits_f32.iter().zip(logits_f16.iter()).enumerate() {
        let a1 = top_k(lf, 1)[0];
        let b1 = top_k(lh, 1)[0];
        if a1 != b1 {
            top1_mismatches += 1;
            println!("  pos {p}: TOP-1 MISMATCH f32={a1} f16={b1}");
        }
        let a5 = top_k(lf, 5);
        let b5: std::collections::HashSet<u32> = top_k(lh, 5).into_iter().collect();
        if !a5.iter().all(|t| b5.contains(t)) {
            top5_mismatches += 1;
        }
        for (x, y) in lf.iter().zip(lh.iter()) {
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
                worst_pos = p;
            }
        }
    }

    let n = logits_f32.len();
    println!("\n=== kv_f16 parity over {n} positions ===");
    println!("top-1 mismatches: {top1_mismatches}");
    println!("top-5 mismatches: {top5_mismatches}");
    println!("logit max_abs:    {max_abs:.5} (worst pos {worst_pos})");

    // Correctness gate is top-1/top-5 identity — f16 KV must never change which
    // token the model picks. The raw logit delta is f16 rounding propagating
    // through attention → lm_head: empirically ~0.05 max-abs on logits of
    // magnitude O(10–30) (~0.3%), so it's reported for visibility but only an
    // EGREGIOUS delta (>0.25, i.e. a packing/offset corruption that didn't
    // happen to flip a top-1) fails the run.
    let pass = top1_mismatches == 0 && top5_mismatches == 0 && max_abs < 0.25;
    if pass {
        println!("PASS ✓ (top-1/top-5 identical; logit delta within f16 noise)");
        ExitCode::SUCCESS
    } else {
        println!("FAIL ✗");
        ExitCode::from(1)
    }
}
