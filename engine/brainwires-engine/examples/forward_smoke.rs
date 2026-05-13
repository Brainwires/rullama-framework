//! M1 exit-criterion smoke test: load the local Gemma 4 GGUF, run a single forward
//! step with token = `<bos>`, print top-5 logits + timings.
//!
//! Build: cargo run --release --features cpu-reference --example forward_smoke -- <gguf>
//!
//! Sanity checks:
//!   - completes without panic
//!   - no NaN/Inf in logits
//!   - softcap clamps logits into [-30, 30]
//!   - top tokens are plausible (not always the same constant id)

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use rullama::gguf::GgufReader;
use rullama::model::config::Gemma4Config;
use rullama::reference::{KvState, Weights, forward_token};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => { eprintln!("usage: forward_smoke <path-to-gguf>"); return ExitCode::from(2); }
    };
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => { eprintln!("read error: {e}"); return ExitCode::from(1); }
    };
    let r = match GgufReader::new(bytes) {
        Ok(r) => r,
        Err(e) => { eprintln!("gguf parse error: {e}"); return ExitCode::from(1); }
    };
    let cfg = match Gemma4Config::from_gguf(&r) {
        Ok(c) => c,
        Err(e) => { eprintln!("config error: {e}"); return ExitCode::from(1); }
    };
    let r_arc = std::sync::Arc::new(r);
    let weights = Weights::new(r_arc.clone());
    let mut kv = KvState::new(&cfg);

    let bos = cfg.bos_id.unwrap_or(2);
    println!("forward step at pos=0 with token_id={bos} (bos)");

    let t0 = Instant::now();
    let logits = match forward_token(&cfg, &weights, &mut kv, bos, 0) {
        Ok(l) => l,
        Err(e) => { eprintln!("forward error: {e}"); return ExitCode::from(1); }
    };
    let dt = t0.elapsed();

    // sanity stats
    let mut nans = 0;
    let mut infs = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &logits {
        if v.is_nan() { nans += 1; }
        else if v.is_infinite() { infs += 1; }
        else { if v < min { min = v; } if v > max { max = v; } }
    }

    println!("forward took {dt:?}");
    println!("logit stats: min={min:.4} max={max:.4} nans={nans} infs={infs} (cap={})", cfg.final_logit_softcap);

    // top-K
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("top 5 logits (token_id, value):");
    for (idx, v) in indexed.iter().take(5) {
        println!("  {idx:>6}  {v:+.4}");
    }
    println!("bottom 3 logits (token_id, value):");
    for (idx, v) in indexed.iter().rev().take(3) {
        println!("  {idx:>6}  {v:+.4}");
    }

    if nans > 0 || infs > 0 {
        eprintln!("FAIL: non-finite logits");
        ExitCode::from(1)
    } else if max <= cfg.final_logit_softcap + 1e-3 && min >= -cfg.final_logit_softcap - 1e-3 {
        println!("PASS: forward completed, logits in softcap range");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL: logits exceed softcap range");
        ExitCode::from(1)
    }
}
