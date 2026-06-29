//! Parity: the mobile streaming `token_embd` output projection (fetch tile → matmul → destroy)
//! must produce bit-identical logits to the cached `buffer_tiles_async` path. In the forward
//! path, `mobile_mode` gates ONLY the outproj streaming (the other mobile knobs are backward-only
//! or native no-ops), so running the same prompt with the flag off vs on isolates the change —
//! and exercises the destroy-after-submit path on native to rule out a GPU use-after-free.
//!
//!   cargo run --release -p rullama-engine --example outproj_stream_parity -- <gguf> ["prompt"]

use std::env;
use std::fs;
use std::sync::Arc;

use rullama_engine::backend::{Pipelines, WeightCache, WgpuCtx};
use rullama_engine::gguf::GgufReader;
use rullama_engine::model::config::Gemma4Config;
use rullama_engine::reference::Weights;
use rullama_engine::reference::forward_chained::Forward;
use rullama_engine::tokenizer::BpeTokenizer;

fn run_prompt(fwd: &mut Forward, ids: &[u32]) -> Vec<Vec<f32>> {
    fwd.reset();
    ids.iter()
        .map(|&id| pollster::block_on(fwd.step(id)).expect("step"))
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .expect("usage: outproj_stream_parity <gguf> [prompt]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "The quick brown fox jumps over".to_string());

    let bytes = fs::read(&path).expect("read gguf");
    let reader = GgufReader::new(bytes).expect("parse gguf");
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
    let tok = BpeTokenizer::from_gguf(&reader).expect("tokenizer");
    let r_arc = Arc::new(reader);
    let weights = Weights::new(r_arc.clone());

    let ctx = pollster::block_on(WgpuCtx::new()).expect("WgpuCtx");
    let pipes = Arc::new(Pipelines::new(&ctx.device));
    let wcache = Arc::new(WeightCache::new(
        r_arc,
        ctx.device.clone(),
        ctx.queue.clone(),
        Arc::clone(&ctx.bind_cache),
    ));
    let mut fwd =
        pollster::block_on(Forward::new(cfg, ctx, pipes, weights, wcache)).expect("Forward::new");

    let ids = tok.encode(&prompt);
    println!("prompt {prompt:?} -> {} tokens", ids.len());

    fwd.set_mobile_mode(false);
    let cached = run_prompt(&mut fwd, &ids);
    fwd.set_mobile_mode(true);
    let streamed = run_prompt(&mut fwd, &ids);

    let mut max_abs = 0f32;
    let mut argmax_mismatch = 0;
    for (a, b) in cached.iter().zip(streamed.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        if argmax(a) != argmax(b) {
            argmax_mismatch += 1;
        }
    }
    println!(
        "positions={} vocab={} max_abs_logit_diff={max_abs:e} argmax_mismatches={argmax_mismatch}",
        cached.len(),
        cached.first().map(|v| v.len()).unwrap_or(0),
    );
    assert_eq!(argmax_mismatch, 0, "streaming outproj changed an argmax!");
    assert!(
        max_abs == 0.0,
        "streaming outproj is not bit-identical: max_abs={max_abs}"
    );
    println!("PASS: streaming token_embd outproj is bit-identical to the cached path");
}
