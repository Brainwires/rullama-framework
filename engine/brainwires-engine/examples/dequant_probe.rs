//! Native-only sanity tool: dequantize a couple of representative tensors from a real
//! Gemma 4 Q4_K_M GGUF and report distribution statistics. Used during M1 to confirm
//! Q4_K and Q6_K dequant produce reasonable weight magnitudes (not NaN, no extreme
//! outliers, RMS in the range typical of well-trained transformer weights).
//!
//! Usage:
//!   cargo run --release --example dequant_probe -- <path-to-model.gguf>

use std::env;
use std::fs;
use std::process::ExitCode;

use brainwires_engine::gguf::{GgufReader, dequant_tensor_to_f32};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: dequant_probe <path-to-model.gguf>");
            return ExitCode::from(2);
        }
    };

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read error: {e}");
            return ExitCode::from(1);
        }
    };
    let r = match GgufReader::new(bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse error: {e}");
            return ExitCode::from(1);
        }
    };

    // Representative tensors: one Q4_K, one Q6_K, one F32 norm.
    let probes = [
        "blk.0.attn_q.weight",      // Q4_K
        "blk.0.attn_v.weight",      // Q6_K
        "blk.0.ffn_down.weight",    // Q6_K (large)
        "blk.0.attn_norm.weight",   // F32
        "blk.0.attn_k_norm.weight", // F32
    ];

    for name in probes {
        match r.tensor(name) {
            Ok(desc) => {
                let elems = desc.elem_count() as usize;
                let started = std::time::Instant::now();
                let v = match dequant_tensor_to_f32(&r, name) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("  {name}: dequant error: {e}");
                        continue;
                    }
                };
                let dur = started.elapsed();
                let stats = stats_of(&v);
                println!(
                    "{:<28} {:>7?} dims={:?} elems={} dequant in {:?}\n  min={:.5} max={:.5} mean={:+.5} rms={:.5} nan={} inf={}",
                    name,
                    desc.dtype,
                    desc.dims,
                    elems,
                    dur,
                    stats.min,
                    stats.max,
                    stats.mean,
                    stats.rms,
                    stats.nans,
                    stats.infs,
                );
            }
            Err(e) => println!("{name}: {e}"),
        }
    }

    ExitCode::SUCCESS
}

struct Stats {
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
    nans: usize,
    infs: usize,
}

fn stats_of(v: &[f32]) -> Stats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f64;
    let mut sumsq = 0f64;
    let mut nans = 0;
    let mut infs = 0;
    for &x in v {
        if x.is_nan() {
            nans += 1;
            continue;
        }
        if x.is_infinite() {
            infs += 1;
            continue;
        }
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x as f64;
        sumsq += (x as f64) * (x as f64);
    }
    let n = (v.len() - nans - infs).max(1) as f64;
    Stats {
        min,
        max,
        mean: (sum / n) as f32,
        rms: ((sumsq / n).sqrt()) as f32,
        nans,
        infs,
    }
}
