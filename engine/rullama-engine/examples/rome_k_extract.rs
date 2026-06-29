//! ROME Phase 1.1 smoke test: extract k* from a subject prompt.
//!
//! Runs the prompt through `Model::extract_mlp_input_native` at the
//! given layer, prints summary statistics of the returned vector
//! (length, mean, std, min, max, first/last 4 elements) so a human
//! can sanity-check that the readback path works.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-engine --release --example rome_k_extract -- \
//!     ~/.ollama/models/blobs/sha256-<digest>     \
//!     5                                          \
//!     "What's the capital of France?"
//! ```
//!
//! Expected for a working pipeline: vector of length d_ffn (Gemma
//! 4 e2b: layer-specific, typically 3840-5120), finite values,
//! sparse — GEGLU is a gating activation so many entries are near
//! zero, but not all. Some negative entries are normal (GEGLU's
//! `gate * SiLU(up)` can be either sign).
//! All zeros = readback path or capture buffer wiring is broken.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama_engine::api::Model;

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "usage: rome_k_extract <gguf-path> <layer> <prompt>".into() })?
        .into();
    let layer: u32 = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <layer>".into() })?
        .parse()?;
    let prompt: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <prompt> (quoted string)".into() })?;

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    eprintln!("[encode] prompt = {prompt:?}");
    let tokens = model.encode_tokens(&prompt);
    eprintln!("[encode] {} tokens", tokens.len());
    if tokens.is_empty() {
        return Err("prompt tokenized to empty".into());
    }

    eprintln!("[rome] extracting k* at layer {layer} (last-token MLP-input)…");
    let k_star = model
        .extract_mlp_input_native(&tokens, layer)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    // Summary stats.
    let n = k_star.len();
    let sum: f64 = k_star.iter().map(|&x| x as f64).sum();
    let mean = sum / n as f64;
    let var: f64 = k_star
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt();
    let min = k_star.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = k_star.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan_count = k_star.iter().filter(|x| x.is_nan()).count();
    let zero_count = k_star.iter().filter(|&&x| x == 0.0).count();

    println!();
    println!("=== k* statistics (layer {layer}) ===");
    println!("  length:        {n}");
    println!("  mean:          {mean:.6e}");
    println!("  std:           {std:.6e}");
    println!("  min:           {min:.6e}");
    println!("  max:           {max:.6e}");
    println!("  nan count:     {nan_count}");
    println!(
        "  zero count:    {zero_count} ({:.1}%)",
        100.0 * zero_count as f64 / n as f64
    );
    println!();
    println!("  first 4:       {:?}", &k_star[..4.min(n)]);
    println!("  last  4:       {:?}", &k_star[n.saturating_sub(4)..]);

    // Sanity verdict.
    if nan_count > 0 {
        eprintln!("\n[FAIL] k* contains {nan_count} NaN values");
        std::process::exit(1);
    }
    if zero_count == n {
        eprintln!("\n[FAIL] k* is all zeros — capture buffer not written, readback path broken");
        std::process::exit(1);
    }
    if std < 1e-6 {
        eprintln!(
            "\n[WARN] k* std is suspiciously small ({std:.3e}) — RMSNorm output should be O(1)"
        );
    } else {
        eprintln!("\n[PASS] k* extracted: length={n}, std={std:.3e}, finite, non-zero");
    }
    Ok(())
}
