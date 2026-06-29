//! ROME Phase 2.1 — covariance calibration tool.
//!
//! Compute `C = E[k kᵀ]` over a generic text corpus for a chosen
//! transformer layer, where `k = ffn_act` (post-GEGLU; the input to
//! `ffn_down`). Used by full ROME's covariance-corrected scaling
//! `s = 1 / (k*ᵀ C⁻¹ k*)` to replace ROME-lite's spherical `||k*||²`.
//!
//! Writes a safetensors sidecar containing the Cholesky factor `L` of
//! `C + ridge·I`. At edit time, `RomeCovariance::cov_inv_k` performs
//! two triangular solves (forward + back-sub) to compute `C⁻¹ k*`
//! without ever materializing `C⁻¹` explicitly.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-engine --release --example compute_rome_covariance -- \
//!     ~/.ollama/models/blobs/sha256-<digest>             \
//!     <layer>                                            \
//!     <corpus.txt>                                       \
//!     <out.safetensors>
//! ```
//!
//! Env knobs:
//!   - `RULLAMA_COV_RIDGE` — ridge added to the diagonal before
//!     Cholesky (default 0.01, per Meng et al. 2022 Appendix A).
//!     Higher values trade conditioning for fidelity to true C.
//!   - `RULLAMA_COV_CHUNK_TOKENS` — corpus is processed in chunks
//!     (KV is reset between chunks). Default 256.
//!   - `RULLAMA_COV_MAX_TOKENS` — stop after this many tokens
//!     (useful for quick smoke tests; default = whole corpus).
//!
//! For Gemma 4 e2b layer 10: `d_ffn = 6144`, so:
//!   - Accumulator: 6144² × 4 = ~144 MB peak RAM
//!   - Cholesky time: ~30-90 s single-threaded
//!   - Forward time dominates for corpora > ~10k tokens
//!
//! Output safetensors layout:
//!   - key: `rome_cov_chol.blk.{layer}.ffn_down`
//!   - shape: `[d_ffn, d_ffn]` — lower triangle holds `L` (upper is 0)
//!   - dtype: `F32`
//!   - metadata: `format=rullama-rome-cov-v0`, `ridge=<value>`,
//!     `n_samples=<count>`, `layer=<layer>`

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rullama_engine::api::Model;

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: compute_rome_covariance <gguf-path> <layer> <corpus.txt> <out.safetensors>"
                .into()
        })?
        .into();
    let target_layer: u32 = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <layer>".into() })?
        .parse()?;
    let corpus_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <corpus.txt>".into() })?
        .into();
    let out_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <out.safetensors>".into() })?
        .into();

    let ridge: f32 = env::var("RULLAMA_COV_RIDGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    // Smaller chunk → less GPU memory pressure per RomeCapture and
    // more frequent progress logs. The forward-step rate on the iris
    // pro 555 with full capture is ~3-5 tok/s, so 64 = ~15-20 s per
    // chunk.
    let chunk_tokens: usize = env::var("RULLAMA_COV_CHUNK_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let max_tokens: Option<usize> = env::var("RULLAMA_COV_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok());

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    let n_layers = model.forward().cfg().n_layers;
    if target_layer >= n_layers {
        return Err(format!("target_layer {target_layer} out of range (have {n_layers})").into());
    }
    let d_ffn = model.forward().cfg().ffn(target_layer) as usize;
    eprintln!("[cfg]  layer={target_layer}, d_ffn={d_ffn}, ridge={ridge}");
    eprintln!(
        "[cfg]  accumulator size: {} MB",
        (d_ffn * d_ffn * 4) / (1024 * 1024)
    );

    eprintln!("[corpus] reading {} …", corpus_path.display());
    let corpus_text = fs::read_to_string(&corpus_path)?;
    let mut all_tokens = model.encode_tokens(&corpus_text);
    if let Some(limit) = max_tokens {
        all_tokens.truncate(limit);
    }
    eprintln!(
        "[corpus] {} tokens ({} bytes raw, chunk size = {})",
        all_tokens.len(),
        corpus_text.len(),
        chunk_tokens
    );
    if all_tokens.len() < d_ffn {
        eprintln!(
            "[warn] corpus has {} tokens but d_ffn = {}; covariance will be \
             rank-deficient (ridge will mask but invertibility relies on ridge \
             alone). Recommend >> {} tokens.",
            all_tokens.len(),
            d_ffn,
            d_ffn
        );
    }

    // Symmetric accumulator C = Σ k_i k_iᵀ. Stored row-major as [d_ffn × d_ffn]
    // f32. For d_ffn = 6144 this is 144 MB — a single big Vec allocation.
    let mut cov: Vec<f32> = vec![0.0f32; d_ffn * d_ffn];
    let mut n_samples: u64 = 0;

    let ctx_arc = std::sync::Arc::new(model.forward().ctx().clone());
    let cfg = model.forward().cfg().clone();

    let t_start = Instant::now();
    let mut tokens_seen = 0usize;
    for (chunk_idx, chunk) in all_tokens.chunks(chunk_tokens).enumerate() {
        let n_chunks = all_tokens.len().div_ceil(chunk_tokens);
        let seq_len = chunk.len() as u32;
        let t_chunk = Instant::now();
        eprintln!(
            "[chunk {}/{}] fwd {} tokens (seq_len={}, allocating capture buffers …)",
            chunk_idx + 1,
            n_chunks,
            chunk.len(),
            seq_len
        );
        let capture = rullama_engine::reference::rome::RomeCapture::new(&ctx_arc, &cfg, seq_len);
        let captures = capture.as_captures();
        model.forward_mut().reset();
        for &tok in chunk {
            let _ = model
                .forward_mut()
                .step_capture(tok, &captures, None, None)
                .await
                .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
        }
        drop(captures);
        let t_fwd = t_chunk.elapsed().as_secs_f64();

        // Read back ffn_act at every position in this chunk and rank-1
        // update the accumulator. Each position contributes
        // `k k^T` (symmetric); we update only the lower triangle and
        // mirror at Cholesky time.
        let t_acc_start = Instant::now();
        for pos in 0..chunk.len() {
            let k = capture
                .read_ffn_act(target_layer, pos as u32)
                .await
                .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
            if k.len() != d_ffn {
                return Err(format!("ffn_act len {} != d_ffn {}", k.len(), d_ffn).into());
            }
            // Outer-product accumulate. Hot loop — inner is one row's
            // worth of fma. ~38M ops per sample at d_ffn=6144.
            for i in 0..d_ffn {
                let ki = k[i];
                if ki == 0.0 {
                    continue;
                }
                let row = &mut cov[i * d_ffn..(i + 1) * d_ffn];
                for j in 0..d_ffn {
                    row[j] += ki * k[j];
                }
            }
            n_samples += 1;
        }
        tokens_seen += chunk.len();
        let t_acc = t_acc_start.elapsed().as_secs_f64();
        let elapsed = t_start.elapsed().as_secs_f64();
        let rate = (tokens_seen as f64) / elapsed.max(1e-6);
        eprintln!(
            "[chunk {}/{}] done fwd={:.1}s acc={:.1}s  total tok={}/{} ({:.1} tok/s, elapsed {:.1}s)",
            chunk_idx + 1,
            n_chunks,
            t_fwd,
            t_acc,
            tokens_seen,
            all_tokens.len(),
            rate,
            elapsed
        );
    }

    eprintln!("[stat] {} samples accumulated", n_samples);
    if n_samples == 0 {
        return Err("no samples accumulated — empty corpus?".into());
    }

    // Average: C = (Σ k k^T) / N
    let inv_n = 1.0_f32 / (n_samples as f32);
    for x in &mut cov {
        *x *= inv_n;
    }

    // Add ridge to diagonal — guarantees SPD for Cholesky.
    for i in 0..d_ffn {
        cov[i * d_ffn + i] += ridge;
    }

    eprintln!("[chol] Cholesky factor of (C + {ridge}·I), d = {d_ffn} …");
    let chol_start = Instant::now();
    cholesky_in_place(&mut cov, d_ffn).map_err(|e| -> BoxError { e.into() })?;
    eprintln!("[chol] done in {:.1}s", chol_start.elapsed().as_secs_f64());

    // Zero the upper triangle for cleanliness (Cholesky writes only the
    // lower; existing upper-tri entries are stale `C` values).
    for i in 0..d_ffn {
        for j in (i + 1)..d_ffn {
            cov[i * d_ffn + j] = 0.0;
        }
    }

    eprintln!("[save] writing {} …", out_path.display());
    use safetensors::tensor::{Dtype, TensorView};
    let tensor_name = format!("rome_cov_chol.blk.{}.ffn_down", target_layer);
    let chol_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&cov).to_vec();
    let view = TensorView::new(Dtype::F32, vec![d_ffn, d_ffn], &chol_bytes)
        .map_err(|e| -> BoxError { format!("safetensors view: {e}").into() })?;
    let mut views: std::collections::HashMap<&str, TensorView<'_>> =
        std::collections::HashMap::new();
    views.insert(tensor_name.as_str(), view);
    let metadata: std::collections::HashMap<String, String> = [
        ("format".to_string(), "rullama-rome-cov-v0".to_string()),
        ("layer".to_string(), target_layer.to_string()),
        ("d_ffn".to_string(), d_ffn.to_string()),
        ("ridge".to_string(), ridge.to_string()),
        ("n_samples".to_string(), n_samples.to_string()),
    ]
    .into_iter()
    .collect();
    let out_bytes = safetensors::serialize(&views, &Some(metadata))
        .map_err(|e| -> BoxError { format!("safetensors serialize: {e}").into() })?;
    fs::write(&out_path, &out_bytes)?;
    eprintln!("[save] wrote {} bytes", out_bytes.len());

    Ok(())
}

/// In-place Cholesky factorization. On entry `a` holds an SPD matrix
/// (lower triangle suffices; upper is read but not modified). On exit
/// `a` holds `L` in its lower triangle such that `L Lᵀ = A`.
///
/// Standard column-oriented algorithm; O(n³/3). Single-threaded for
/// simplicity — for n=6144 this is ~30-90s depending on CPU. The matrix
/// is row-major; column-oriented Cholesky is fine because each column
/// only reads/writes elements in one column's lower-triangular slice.
fn cholesky_in_place(a: &mut [f32], n: usize) -> Result<(), String> {
    for j in 0..n {
        // Diagonal: a[j,j] -= sum over k < j of a[j,k]^2
        let mut diag = a[j * n + j];
        for k in 0..j {
            let v = a[j * n + k];
            diag -= v * v;
        }
        if diag <= 0.0 || !diag.is_finite() {
            return Err(format!(
                "Cholesky failed at column {j}: diag = {diag:.3e} (not SPD; try larger ridge)"
            ));
        }
        let l_jj = diag.sqrt();
        a[j * n + j] = l_jj;
        let inv_l_jj = 1.0 / l_jj;

        // Below-diagonal entries of column j:
        //   a[i,j] = (a[i,j] - Σ_{k<j} a[i,k]*a[j,k]) / l_jj  for i > j
        for i in (j + 1)..n {
            let mut sum = a[i * n + j];
            // dot-product of row i and row j over columns 0..j
            let row_i = &a[i * n..i * n + j];
            let row_j = &a[j * n..j * n + j];
            for k in 0..j {
                sum -= row_i[k] * row_j[k];
            }
            a[i * n + j] = sum * inv_l_jj;
        }
    }
    Ok(())
}
