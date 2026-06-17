//! Parity check for `Forward::truncate_kv` — the primitive behind
//! longest-common-prefix KV reuse (hot-starting a new chat off the cached
//! system prompt).
//!
//! Claim: feeding a continuation `C` after `truncate_kv(n)` must be
//! bit-identical to feeding `C` after a *fresh* prefill of just the first
//! `n` tokens. I.e. truncation only rewinds bookkeeping — the surviving
//! KV positions `[0, n)` are exactly what a from-scratch prefill of the
//! shared prefix would produce, so the continuation computes the same.
//!
//! Build:
//!   cargo run --release --example kv_truncate_parity -- <gguf> [--n=N]
//!
//! Exit code 0 on PASS (max-abs logit diff under tolerance), 1 on FAIL.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use rullama::backend::{Pipelines, WeightCache, WgpuCtx};
use rullama::gguf::GgufReader;
use rullama::model::config::Gemma4Config;
use rullama::reference::Weights;
use rullama::reference::forward_chained::Forward;
use rullama::tokenizer::BpeTokenizer;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: kv_truncate_parity <gguf> [--n=N]");
            return ExitCode::from(2);
        }
    };
    let mut keep_n: usize = 4;
    for a in args {
        if let Some(rest) = a.strip_prefix("--n=") {
            keep_n = rest.parse().unwrap_or(4);
        }
    }

    println!("loading model ...");
    let bytes = fs::read(&path).expect("read");
    let reader = GgufReader::new(bytes).expect("parse");
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

    // A "first conversation" prefix P, and a continuation C (a distinct
    // short token sequence fed at positions [n, n+|C|)). C reuses P's
    // leading ids purely so they're guaranteed-valid vocab tokens.
    let prefix: Vec<u32> = tok.encode("<bos>The quick brown fox jumps over the lazy dog.");
    assert!(prefix.len() > keep_n, "prefix too short for --n={keep_n}");
    let cont: Vec<u32> = prefix.iter().take(6).copied().collect();
    println!(
        "prefix={} tokens, keep_n={keep_n}, cont={} tokens",
        prefix.len(),
        cont.len()
    );

    // Path A: feed the FULL prefix, truncate back to n, then feed C.
    for &id in &prefix {
        let _ = pollster::block_on(fwd.step(id)).expect("step A.prefill");
    }
    fwd.truncate_kv(keep_n as u32);
    assert_eq!(fwd.pos(), keep_n as u32, "pos should rewind to n");
    let mut logits_a = Vec::new();
    for &id in &cont {
        logits_a = pollster::block_on(fwd.step(id)).expect("step A.cont");
    }

    // Path B: fresh — feed only prefix[0..n], then C.
    fwd.reset();
    for &id in &prefix[..keep_n] {
        let _ = pollster::block_on(fwd.step(id)).expect("step B.prefill");
    }
    let mut logits_b = Vec::new();
    for &id in &cont {
        logits_b = pollster::block_on(fwd.step(id)).expect("step B.cont");
    }

    let diff = max_abs_diff(&logits_a, &logits_b);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    let (am_a, am_b) = (argmax(&logits_a), argmax(&logits_b));
    println!("max-abs logit diff (truncate+cont vs fresh prefix+cont): {diff:.3e}");
    println!("argmax: truncate={am_a} fresh={am_b}");

    let tol = 1e-4;
    if diff <= tol && am_a == am_b {
        println!("PASS (<= {tol:.0e})");
        ExitCode::SUCCESS
    } else {
        println!("FAIL");
        ExitCode::from(1)
    }
}
