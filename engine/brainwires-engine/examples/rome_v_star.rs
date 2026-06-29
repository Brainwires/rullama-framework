//! ROME Phase 1.2 smoke test: compute the v* gradient direction.
//!
//! Calls `Model::compute_rome_gradient_native` to obtain
//! `∂loss/∂ffn_out[target_layer]` at the subject prompt's last token,
//! where `loss = -log P(target_token | prompt)`. Prints summary
//! statistics — finite values + non-zero std confirms the backward
//! path is propagating gradients correctly.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama --release --example rome_v_star -- \
//!     ~/.ollama/models/blobs/sha256-<digest>  \
//!     5                                       \
//!     "What's the capital of France?"         \
//!     "Brie"
//! ```
//!
//! Expected for a working pipeline: vector of length d_model
//! (Gemma 4 e2b = 1536), finite values, std on the order of
//! 1e-4 to 1e-1 (gradient magnitudes vary with target probability).
//! All zeros = backward path is broken; NaN = numerical overflow
//! in some intermediate.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::Model;

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: rome_v_star <gguf-path> <layer> <prompt> <target-text>".into()
        })?
        .into();
    let layer: u32 = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <layer>".into() })?
        .parse()?;
    let prompt: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <prompt> (quoted string)".into() })?;
    let target_text: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <target-text>".into() })?;

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    eprintln!("[encode] prompt = {prompt:?}");
    let prompt_tokens = model.encode_tokens(&prompt);
    eprintln!("[encode] {} prompt tokens", prompt_tokens.len());

    let target_tokens = model.encode_tokens(&target_text);
    if target_tokens.is_empty() {
        return Err("target_text tokenized to empty".into());
    }
    let target_token_id = target_tokens[0];
    let target_str = model.token_str_native(target_token_id).unwrap_or_default();
    eprintln!("[encode] target_token = {target_token_id} ({target_str:?})");

    eprintln!("[rome] computing v* gradient at layer {layer}…");
    let grad = model
        .compute_rome_gradient_native(&prompt_tokens, layer, target_token_id)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    // Summary stats.
    let n = grad.len();
    let sum: f64 = grad.iter().map(|&x| x as f64).sum();
    let mean = sum / n as f64;
    let var: f64 = grad
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt();
    let min = grad.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = grad.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan_count = grad.iter().filter(|x| x.is_nan()).count();
    let zero_count = grad.iter().filter(|&&x| x == 0.0).count();

    println!();
    println!("=== v* gradient (∂loss/∂ffn_out[layer {layer}]) ===");
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
    println!("  first 4:       {:?}", &grad[..4.min(n)]);
    println!("  last  4:       {:?}", &grad[n.saturating_sub(4)..]);

    if nan_count > 0 {
        eprintln!("\n[FAIL] gradient contains {nan_count} NaN values");
        std::process::exit(1);
    }
    if zero_count == n {
        eprintln!("\n[FAIL] gradient is all zeros — backward path not propagating");
        std::process::exit(1);
    }
    if std < 1e-10 {
        eprintln!(
            "\n[WARN] gradient std is suspiciously small ({std:.3e}) — model may already predict target with prob≈1"
        );
    } else {
        eprintln!("\n[PASS] v* gradient computed: length={n}, std={std:.3e}, finite, non-zero");
    }
    Ok(())
}
