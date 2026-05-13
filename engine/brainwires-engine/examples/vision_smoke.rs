//! End-to-end smoke test for `VisionForward`.
//!
//! Generates a synthetic 48×48 image (gradient pattern), runs it through
//! `Model::encode_image_native`, and prints summary statistics of the resulting
//! soft tokens. Validates that the entire vision pipeline (Conv2D → 16 ViT
//! blocks → AvgPool → projector → RMSNorm) runs without panic and emits f32s
//! within plausible ranges.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example vision_smoke -- <gguf>

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use rullama::api::Model;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: vision_smoke <gguf>");
            return ExitCode::from(2);
        }
    };

    println!("loading {path} ...");
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read");
    let model = pollster::block_on(Model::load_native(bytes)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());

    if !model.has_vision_native() {
        eprintln!("this checkpoint has no vision tower (no v.patch_embd.weight)");
        return ExitCode::from(2);
    }

    // Synthesize a 48×48 image: simple [R, G, B] gradient ramp normalized to [-1, 1].
    // 48 = patch_size(16) × n_merge(3), so this gives 1 pooled token (3×3 patches → 1).
    let h = 48usize;
    let w = 48usize;
    let n = h * w;
    let mut pixels = vec![0f32; 3 * n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // R = x-gradient, G = y-gradient, B = constant 0.5 → normalize to [-1, 1].
            pixels[i] = (x as f32 / w as f32) * 2.0 - 1.0;
            pixels[n + i] = (y as f32 / h as f32) * 2.0 - 1.0;
            pixels[2 * n + i] = 0.0;
        }
    }

    let expected = model.image_soft_token_count_native(h, w).unwrap_or(0);
    println!("encoding {h}×{w} (expected soft tokens: {expected}) ...");

    let t0 = Instant::now();
    let soft = pollster::block_on(model.encode_image_native(&pixels, h, w, None))
        .expect("encode_image");
    let dt = t0.elapsed();

    let d_text = soft.len() / expected.max(1);
    println!("encoded in {:?} — {} soft tokens × {} dim = {} f32s",
        dt, expected, d_text, soft.len());

    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut nan_count = 0usize;
    for &v in &soft {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
        if v < min { min = v; }
        if v > max { max = v; }
    }
    let n_finite = (soft.len() - nan_count) as f64;
    let mean = sum / n_finite;
    let var = sum_sq / n_finite - mean * mean;
    println!("stats: mean={:.4} stddev={:.4} min={:.4} max={:.4} nans={}",
        mean, var.sqrt(), min, max, nan_count);

    if nan_count > 0 {
        eprintln!("FAIL: NaNs in output");
        return ExitCode::from(1);
    }
    if !min.is_finite() || !max.is_finite() {
        eprintln!("FAIL: non-finite output");
        return ExitCode::from(1);
    }
    println!("OK");
    ExitCode::SUCCESS
}
