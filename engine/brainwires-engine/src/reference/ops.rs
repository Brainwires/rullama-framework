//! f32 tensor primitives. Plain `Vec<f32>` + explicit shape; no broadcasting magic.
//!
//! Every weight matrix here is laid out as it appears in GGUF: `[k, n]` in metadata
//! means dim[0]=k is the fastest-varying axis, so the memory order is
//! `data[j * k + i]` for element `(i, j)` and the natural matvec computes
//! `y[j] = Σ_i W[j*k + i] * x[i]`. (This matches `ggml_mul_mat(W, x)` in llama.cpp.)

/// Multiply matrix `w[n, k]` (stored row-major; row j is `w[j*k..j*k+k]`) by vector
/// `x[k]` and write result to `y[n]`.
pub fn matvec(w: &[f32], k: usize, n: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), n);
    for j in 0..n {
        let mut acc = 0f32;
        let row = &w[j * k..j * k + k];
        for i in 0..k {
            acc += row[i] * x[i];
        }
        y[j] = acc;
    }
}

/// RMSNorm with optional weight: `y = x / sqrt(mean(x²) + eps) * w` (or `* 1` if w is None).
pub fn rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(out.len(), n);
    let mut sumsq = 0f64;
    for &v in x {
        sumsq += (v as f64) * (v as f64);
    }
    let rms = ((sumsq / n as f64) as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    if let Some(w) = w {
        debug_assert_eq!(w.len(), n);
        for i in 0..n {
            out[i] = x[i] * inv * w[i];
        }
    } else {
        for i in 0..n {
            out[i] = x[i] * inv;
        }
    }
}

/// Apply NeoX-style RoPE in-place to `x` of shape `[head_dim, n_heads]`. Rotates the
/// first `rope_dims` of each head's vector. With `freq_factors` of length
/// `rope_dims/2`, divides each pair's theta by the factor (matches llama.cpp's
/// proportional RoPE for Gemma 4 global layers).
pub fn rope_neox(
    x: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    pos: usize,
    rope_dims: usize,
    base: f32,
    freq_factors: Option<&[f32]>,
) {
    debug_assert!(rope_dims <= head_dim);
    debug_assert!(rope_dims % 2 == 0);
    debug_assert_eq!(x.len(), head_dim * n_heads);

    // NeoX layout: pair (x[i], x[i + rope_dims/2]) for i in 0..rope_dims/2.
    let half = rope_dims / 2;
    for h in 0..n_heads {
        let base_off = h * head_dim;
        for i in 0..half {
            let theta = (pos as f32) * base.powf(-(2.0 * i as f32) / rope_dims as f32);
            let theta = if let Some(f) = freq_factors {
                theta / f[i]
            } else {
                theta
            };
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let a = x[base_off + i];
            let b = x[base_off + i + half];
            x[base_off + i]        = a * cos_t - b * sin_t;
            x[base_off + i + half] = a * sin_t + b * cos_t;
        }
        // Dimensions [rope_dims..head_dim] are untouched.
    }
}

/// In-place softmax over `x` of length `n`.
pub fn softmax(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max { max = v; }
    }
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `y = gelu(gate) * up` — Gemma 4's GeGLU MLP activation.
/// Uses the tanh-approximation GELU that ggml's `ggml_geglu_split` actually
/// computes (see `ggml-cpu/vec.h: ggml_gelu_f32`). Note: ggml ALSO has an
/// `ggml_geglu_erf_split` variant; Ollama's `Tensor.GELU` routes to the
/// tanh-approximation `ggml_geglu_split`, so we use that here for parity.
pub fn geglu_split(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), gate.len());
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const GELU_COEF_A: f32 = 0.044_715;
    for i in 0..gate.len() {
        let g = gate[i];
        let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
        let gelu = 0.5 * g * (1.0 + inner.tanh());
        out[i] = gelu * up[i];
    }
}

/// `y = cap * tanh(x / cap)` element-wise. Used for Gemma 4's final logit softcap.
pub fn softcap(x: &mut [f32], cap: f32) {
    if cap <= 0.0 { return; }
    let inv = 1.0 / cap;
    for v in x.iter_mut() {
        *v = cap * (*v * inv).tanh();
    }
}

/// Add `b` into `a` element-wise.
pub fn add_into(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Multiply `a` by scalar `s` in place.
pub fn scale(a: &mut [f32], s: f32) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

/// Sentinel target id used by the dataset loader to mask positions on which
/// no loss should fire (prompt tokens, the final position with no next
/// token). Matches `u32::MAX`.
pub const TARGET_MASK: u32 = u32::MAX;

/// RMSNorm backward w.r.t. the input.
///
/// Forward:  `y = (x / r) * w`            where `r = sqrt(mean(x²) + eps)`
/// Backward: `dx[j] = w[j]·dy[j]/r - x[j]·s/(n·r³)`,
///           `s = Σ_i dy[i]·w[i]·x[i]`
///
/// Weight is frozen (LoRA convention); no `dw` is produced.
pub fn rmsnorm_backward(
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    eps: f32,
    dx: &mut [f32],
) {
    let n = x.len();
    assert_eq!(dy.len(), n);
    assert_eq!(dx.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }

    let mut sumsq = 0f64;
    for &v in x {
        sumsq += (v as f64) * (v as f64);
    }
    let nf = n as f64;
    let inv_r = 1.0 / ((sumsq / nf + eps as f64).sqrt());

    let mut s = 0f64;
    for i in 0..n {
        let wi = w.map_or(1.0, |ws| ws[i] as f64);
        s += (dy[i] as f64) * wi * (x[i] as f64);
    }
    let coef = s * inv_r * inv_r * inv_r / nf;

    for j in 0..n {
        let wj = w.map_or(1.0, |ws| ws[j] as f64);
        dx[j] = (wj * dy[j] as f64 * inv_r - x[j] as f64 * coef) as f32;
    }
}

/// GeGLU backward — produces `d_gate` and `d_up` given `dy`, `gate`, `up`.
///
/// Matches the tanh-approximation GELU used by [`geglu_split`]:
///   `gelu(g) = 0.5 · g · (1 + tanh(φ))`, where
///   `φ(g)   = √(2/π) · g · (1 + α·g²)`, `α = 0.044715`.
/// So `gelu'(g) = 0.5·(1 + tanh φ) + 0.5·g·sech²(φ)·φ'(g)`,
/// with `φ'(g) = √(2/π) · (1 + 3·α·g²)`.
pub fn geglu_backward(
    gate: &[f32],
    up: &[f32],
    dy: &[f32],
    d_gate: &mut [f32],
    d_up: &mut [f32],
) {
    let n = gate.len();
    assert_eq!(up.len(), n);
    assert_eq!(dy.len(), n);
    assert_eq!(d_gate.len(), n);
    assert_eq!(d_up.len(), n);
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const GELU_COEF_A: f32 = 0.044_715;
    for i in 0..n {
        let g = gate[i];
        let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
        let t = inner.tanh();
        let dphi = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEF_A * g * g);
        let gelu = 0.5 * g * (1.0 + t);
        let gelu_prime = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * dphi;
        d_gate[i] = dy[i] * gelu_prime * up[i];
        d_up[i] = dy[i] * gelu;
    }
}

/// NeoX RoPE backward — inverse in-place rotation.
///
/// Pass the same `pos`, `base`, and `freq_factors` as the forward and the
/// rotation undoes itself. `dx` is rotated in place; on return it holds
/// `dx_pre_rope` from the upstream `dx_post_rope`.
pub fn rope_neox_backward(
    dx: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    pos: usize,
    rope_dims: usize,
    base: f32,
    freq_factors: Option<&[f32]>,
) {
    debug_assert!(rope_dims <= head_dim);
    debug_assert!(rope_dims % 2 == 0);
    debug_assert_eq!(dx.len(), head_dim * n_heads);
    let half = rope_dims / 2;
    for h in 0..n_heads {
        let base_off = h * head_dim;
        for i in 0..half {
            let theta = (pos as f32) * base.powf(-(2.0 * i as f32) / rope_dims as f32);
            let theta = if let Some(f) = freq_factors {
                theta / f[i]
            } else {
                theta
            };
            let c = theta.cos();
            let s = theta.sin();
            let a = dx[base_off + i];
            let b = dx[base_off + i + half];
            // Inverse rotation: cos symmetric, sin sign-flipped.
            dx[base_off + i]        =  a * c + b * s;
            dx[base_off + i + half] = -a * s + b * c;
        }
    }
}

/// AdamW step over a flat parameter slice — mirrors `adam_step.wgsl`.
///
/// `step` is 1-based. `param`, `m`, `v` are updated in place.
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    grad: &[f32],
    param: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) {
    let n = grad.len();
    assert_eq!(param.len(), n);
    assert_eq!(m.len(), n);
    assert_eq!(v.len(), n);
    let step_f = step as f32;
    let bc1 = 1.0 - beta1.powf(step_f);
    let bc2 = 1.0 - beta2.powf(step_f);
    for i in 0..n {
        let g = grad[i];
        let m_new = beta1 * m[i] + (1.0 - beta1) * g;
        let v_new = beta2 * v[i] + (1.0 - beta2) * g * g;
        m[i] = m_new;
        v[i] = v_new;
        let m_hat = m_new / bc1;
        let v_hat = v_new / bc2;
        let p = param[i];
        param[i] = p - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p);
    }
}

/// LoRA primitive: `y = scale · W @ x` (or `+=` when `accumulate`).
/// `W` is `[n, k]` row-major.
pub fn lora_matmul_row(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    k: usize,
    n: usize,
    scale: f32,
    accumulate: bool,
) {
    assert_eq!(w.len(), n * k);
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    for (j, slot) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for i in 0..k {
            acc += w[j * k + i] * x[i];
        }
        let v = scale * acc;
        *slot = if accumulate { *slot + v } else { v };
    }
}

/// LoRA primitive: `y = scale · Wᵀ @ x` (or `+=`).
/// Same physical `[outer, inner]` layout as `lora_matmul_row`'s W; iterates
/// by column to produce the transposed product.
pub fn lora_matmul_col(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    outer: usize,
    inner: usize,
    scale: f32,
    accumulate: bool,
) {
    assert_eq!(w.len(), outer * inner);
    assert_eq!(x.len(), outer);
    assert_eq!(y.len(), inner);
    for (p, slot) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..outer {
            acc += w[j * inner + p] * x[j];
        }
        let v = scale * acc;
        *slot = if accumulate { *slot + v } else { v };
    }
}

/// LoRA primitive: rank-1 outer-product accumulator.
/// `out[i, j] += scale · a[i] · b[j]` (or `=` when `!accumulate`).
/// `out` shape `[outer_a, outer_b]`.
pub fn lora_outer_add(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    scale: f32,
    accumulate: bool,
) {
    let outer_a = a.len();
    let outer_b = b.len();
    assert_eq!(out.len(), outer_a * outer_b);
    for i in 0..outer_a {
        for j in 0..outer_b {
            let v = scale * a[i] * b[j];
            let off = i * outer_b + j;
            out[off] = if accumulate { out[off] + v } else { v };
        }
    }
}

/// Forward softmax attention over a KV history — single-batch, GQA-aware.
///
/// Mirrors `kernels/wgsl/attention.wgsl` exactly, including the
/// **un-scaled** dot-product score (Gemma 4 absorbs the inverse-sqrt-d
/// factor into the q RMSNorm earlier in the layer). Returns the
/// post-softmax probabilities alongside the output so the backward
/// pass can consume them without redoing the forward.
pub fn attention_forward(
    q: &[f32],
    k_hist: &[f32],
    v_hist: &[f32],
    out: &mut [f32],
    probs: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    history_len: usize,
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_hist.len(), history_len * n_kv_heads * head_dim);
    assert_eq!(v_hist.len(), history_len * n_kv_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(probs.len(), n_heads * history_len);
    assert!(n_kv_heads > 0 && n_heads % n_kv_heads == 0);
    let heads_per_kv = n_heads / n_kv_heads;

    for h in 0..n_heads {
        let kv = h / heads_per_kv;
        let q_off = h * head_dim;

        // scores
        let mut scores = vec![0.0f32; history_len];
        let mut max_s = f32::NEG_INFINITY;
        for j in 0..history_len {
            let k_off = (j * n_kv_heads + kv) * head_dim;
            let mut s = 0f32;
            for d in 0..head_dim {
                s += q[q_off + d] * k_hist[k_off + d];
            }
            scores[j] = s;
            if s > max_s {
                max_s = s;
            }
        }
        // softmax(scores)
        let mut total = 0f64;
        for s in scores.iter_mut() {
            *s = ((*s - max_s) as f64).exp() as f32;
            total += *s as f64;
        }
        let inv = (1.0f64 / total) as f32;
        for j in 0..history_len {
            scores[j] *= inv;
            probs[h * history_len + j] = scores[j];
        }
        // out
        for d in 0..head_dim {
            let mut acc = 0f32;
            for j in 0..history_len {
                let v_off = (j * n_kv_heads + kv) * head_dim;
                acc += scores[j] * v_hist[v_off + d];
            }
            out[q_off + d] = acc;
        }
    }
}

/// Backward of `attention_forward` w.r.t. its three inputs.
///
/// Inputs:
/// - `q`, `k_hist`, `v_hist` — the same tensors fed to the forward.
/// - `probs` — the saved softmax probabilities from the forward (the
///   trainer captures these in `LayerActivations`).
/// - `d_out` — gradient flowing in from above.
///
/// Outputs:
/// - `d_q[h, :]`             — `Σ_j d_scores[h, j] · k_hist[j, kv, :]`
/// - `d_k_hist[j, kv, :]`    — `Σ_{h in kv} d_scores[h, j] · q[h, :]`
/// - `d_v_hist[j, kv, :]`    — `Σ_{h in kv} probs[h, j] · d_out[h, :]`
///
/// Where `d_probs[h, j] = d_out[h, :] · v_hist[j, kv, :]`,
/// `sum_pd[h] = Σ_j probs[h, j] · d_probs[h, j]`,
/// `d_scores[h, j] = probs[h, j] · (d_probs[h, j] - sum_pd[h])`.
///
/// All accumulator buffers are zeroed by the function before writing,
/// so callers can pass scratch buffers without pre-clearing.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    q: &[f32],
    k_hist: &[f32],
    v_hist: &[f32],
    probs: &[f32],
    d_out: &[f32],
    d_q: &mut [f32],
    d_k_hist: &mut [f32],
    d_v_hist: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    history_len: usize,
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k_hist.len(), history_len * n_kv_heads * head_dim);
    assert_eq!(v_hist.len(), history_len * n_kv_heads * head_dim);
    assert_eq!(probs.len(), n_heads * history_len);
    assert_eq!(d_out.len(), n_heads * head_dim);
    assert_eq!(d_q.len(), n_heads * head_dim);
    assert_eq!(d_k_hist.len(), history_len * n_kv_heads * head_dim);
    assert_eq!(d_v_hist.len(), history_len * n_kv_heads * head_dim);
    let heads_per_kv = n_heads / n_kv_heads;

    for x in d_q.iter_mut() { *x = 0.0; }
    for x in d_k_hist.iter_mut() { *x = 0.0; }
    for x in d_v_hist.iter_mut() { *x = 0.0; }

    for h in 0..n_heads {
        let kv = h / heads_per_kv;
        let q_off = h * head_dim;
        let p_off = h * history_len;

        // d_probs[j] = d_out[h, :] · v_hist[j, kv, :]
        let mut d_probs = vec![0f32; history_len];
        for j in 0..history_len {
            let v_off = (j * n_kv_heads + kv) * head_dim;
            let mut dp = 0f32;
            for d in 0..head_dim {
                dp += d_out[q_off + d] * v_hist[v_off + d];
            }
            d_probs[j] = dp;
        }

        // sum_pd = Σ_j probs[h, j] · d_probs[j]
        let mut sum_pd = 0f64;
        for j in 0..history_len {
            sum_pd += probs[p_off + j] as f64 * d_probs[j] as f64;
        }
        let sum_pd = sum_pd as f32;

        // d_scores[j] = probs[h, j] · (d_probs[j] - sum_pd)
        let mut d_scores = vec![0f32; history_len];
        for j in 0..history_len {
            d_scores[j] = probs[p_off + j] * (d_probs[j] - sum_pd);
        }

        // d_q[h, d] = Σ_j d_scores[j] · k_hist[j, kv, d]
        for d in 0..head_dim {
            let mut acc = 0f32;
            for j in 0..history_len {
                let k_off = (j * n_kv_heads + kv) * head_dim;
                acc += d_scores[j] * k_hist[k_off + d];
            }
            d_q[q_off + d] = acc;
        }

        // d_k_hist[j, kv, d] += d_scores[j] · q[h, d]
        // d_v_hist[j, kv, d] += probs[h, j] · d_out[h, d]
        for j in 0..history_len {
            let kv_off = (j * n_kv_heads + kv) * head_dim;
            let ds = d_scores[j];
            let pj = probs[p_off + j];
            for d in 0..head_dim {
                d_k_hist[kv_off + d] += ds * q[q_off + d];
                d_v_hist[kv_off + d] += pj * d_out[q_off + d];
            }
        }
    }
}

/// Backward of `y = matmul_q4_k(W, x)` w.r.t. the input vector `x`.
///
/// Computes `dx[i] = Σ_j dy[j] * dequant(W)[j, i]` where the dequanted
/// matrix has shape `[n, k]` (row-major). `w_bytes` is the raw Q4_K
/// byte stream (n × n_blocks × 144 bytes). The CPU path dequants the
/// matrix once and does a plain f32 transposed matvec.
///
/// The weight matrix is frozen (LoRA convention) — no weight gradient
/// is computed.
pub fn matmul_q4_k_backward_input(
    w_bytes: &[u8],
    dy: &[f32],
    k: usize,
    n: usize,
    dx: &mut [f32],
) {
    assert_eq!(dy.len(), n, "dy length mismatch");
    assert_eq!(dx.len(), k, "dx length mismatch");
    assert_eq!(k % 256, 0, "k must be divisible by 256 for Q4_K");

    let total = n * k;
    let mut w_f32 = vec![0.0f32; total];
    crate::gguf::quant::dequant_q4_k(w_bytes, &mut w_f32).expect("Q4_K dequant");

    // dx[i] = sum_j dy[j] * w_f32[j * k + i]
    for x in dx.iter_mut() {
        *x = 0.0;
    }
    for j in 0..n {
        let row = &w_f32[j * k..(j + 1) * k];
        let dyj = dy[j];
        for i in 0..k {
            dx[i] += dyj * row[i];
        }
    }
}

/// Backward of `y = matmul_q6_k(W, x)` w.r.t. `x`. Same shape conventions
/// as the Q4_K variant — `W` is `[n, k]` row-major, `dy` has length `n`,
/// `dx` has length `k`. Used by the training output-projection backward
/// for Gemma 4's tied Q6_K embedding.
pub fn matmul_q6_k_backward_input(
    w_bytes: &[u8],
    dy: &[f32],
    k: usize,
    n: usize,
    dx: &mut [f32],
) {
    assert_eq!(dy.len(), n, "dy length mismatch");
    assert_eq!(dx.len(), k, "dx length mismatch");
    assert_eq!(k % 256, 0, "k must be divisible by 256 for Q6_K");

    let total = n * k;
    let mut w_f32 = vec![0.0f32; total];
    crate::gguf::quant::dequant_q6_k(w_bytes, &mut w_f32).expect("Q6_K dequant");

    for x in dx.iter_mut() {
        *x = 0.0;
    }
    for j in 0..n {
        let row = &w_f32[j * k..(j + 1) * k];
        let dyj = dy[j];
        for i in 0..k {
            dx[i] += dyj * row[i];
        }
    }
}

/// Per-row RMSNorm backward — runs `rmsnorm_backward` independently per
/// row. `x` is `[n_rows × n]`, `w` is the shared per-element weight
/// `[n]` (`None` ⇒ unweighted), `dy` and `dx` are both `[n_rows × n]`.
/// Mirrors the WGSL `rmsnorm_per_row_backward` kernel.
pub fn rmsnorm_per_row_backward(
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    eps: f32,
    n_rows: usize,
    n: usize,
    dx: &mut [f32],
) {
    assert_eq!(x.len(), n_rows * n);
    assert_eq!(dy.len(), n_rows * n);
    assert_eq!(dx.len(), n_rows * n);
    if let Some(ww) = w {
        assert_eq!(ww.len(), n);
    }
    for r in 0..n_rows {
        let xs = &x[r * n..(r + 1) * n];
        let dys = &dy[r * n..(r + 1) * n];
        let dxs = &mut dx[r * n..(r + 1) * n];
        rmsnorm_backward(xs, w, dys, eps, dxs);
    }
}

/// Cross-entropy forward + backward over a single logit vector.
///
/// Writes `softmax(logits) - one_hot(target)` into `d_logits` and returns the
/// scalar loss `-log softmax(target)`. When `target == TARGET_MASK` or
/// `target >= logits.len()`, the gradient is zeroed and the loss is `0.0` —
/// matches the masking semantics of the WGSL kernel.
pub fn cross_entropy_backward(
    logits: &[f32],
    target: u32,
    d_logits: &mut [f32],
) -> f32 {
    debug_assert_eq!(logits.len(), d_logits.len());
    let n = logits.len();
    let masked = target == TARGET_MASK || (target as usize) >= n;
    if masked {
        for g in d_logits.iter_mut() {
            *g = 0.0;
        }
        return 0.0;
    }

    let mut max_v = f32::NEG_INFINITY;
    for &x in logits {
        if x > max_v {
            max_v = x;
        }
    }

    let mut sum_exp = 0.0f64;
    for &x in logits {
        sum_exp += ((x - max_v) as f64).exp();
    }
    let inv_sum = 1.0 / sum_exp;

    let target = target as usize;
    for (i, (&x, g)) in logits.iter().zip(d_logits.iter_mut()).enumerate() {
        let soft = (((x - max_v) as f64).exp() * inv_sum) as f32;
        *g = if i == target { soft - 1.0 } else { soft };
    }

    (-(logits[target] - max_v) as f64 + sum_exp.ln()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_3x2() {
        // w shape: [n=2, k=3] → row-major rows of length 3
        // w = [[1,2,3], [4,5,6]], x = [1,1,1] → y = [6, 15]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        matvec(&w, 3, 2, &x, &mut y);
        assert_eq!(y, [6.0, 15.0]);
    }

    #[test]
    fn rmsnorm_no_weight() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        rmsnorm(&x, None, 0.0, &mut y);
        // rms = sqrt((1+4+9)/3) = sqrt(14/3); each y_i = x_i / rms
        let rms = ((1.0_f32 + 4.0 + 9.0) / 3.0).sqrt();
        for i in 0..3 { assert!((y[i] - x[i] / rms).abs() < 1e-6); }
    }

    #[test]
    fn softmax_uniform() {
        let mut x = vec![0.0, 0.0, 0.0, 0.0];
        softmax(&mut x);
        for &v in &x { assert!((v - 0.25).abs() < 1e-6); }
    }

    #[test]
    fn softcap_caps() {
        // Use very large magnitudes so tanh saturates to ±1 within f32 precision.
        // tanh(1000/30) ≈ tanh(33.3) is 1.0 to f32; tanh(100/30) is only 0.9975.
        let mut x = vec![0.0, 1.0, 1000.0, -1000.0];
        softcap(&mut x, 30.0);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 30.0 * (1.0_f32 / 30.0).tanh()).abs() < 1e-6);
        assert!((x[2] - 30.0).abs() < 1e-4);
        assert!((x[3] + 30.0).abs() < 1e-4);
    }

    #[test]
    fn geglu_split_zero() {
        let gate = vec![0.0; 4];
        let up = vec![1.0; 4];
        let mut out = vec![999.0; 4];
        geglu_split(&gate, &up, &mut out);
        // gelu(0) = 0, so out = 0 * up = 0
        for &v in &out { assert!(v.abs() < 1e-6); }
    }

    #[test]
    fn rope_neox_zero_pos_is_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0]; // head_dim=4, n_heads=1
        let copy = x.clone();
        rope_neox(&mut x, 4, 1, 0, 4, 10000.0, None);
        for i in 0..4 { assert!((x[i] - copy[i]).abs() < 1e-6); }
    }

    #[test]
    fn cross_entropy_uniform_logits_match_log_n() {
        // Uniform logits -> softmax = 1/n, loss = ln(n) regardless of target.
        let n = 8;
        let logits = vec![0.5f32; n];
        let mut grad = vec![0.0f32; n];
        let loss = cross_entropy_backward(&logits, 3, &mut grad);
        let expected_loss = (n as f32).ln();
        assert!((loss - expected_loss).abs() < 1e-5, "loss {loss} != {expected_loss}");
        // dL/d_logits = softmax - one_hot; sums to 0.
        let s: f32 = grad.iter().sum();
        assert!(s.abs() < 1e-5, "sum of d_logits = {s}");
        // Target entry: softmax - 1 = 1/n - 1 ≈ -0.875.
        let expected_target = 1.0 / (n as f32) - 1.0;
        assert!((grad[3] - expected_target).abs() < 1e-5);
        // Non-target entry: softmax = 1/n.
        let expected_other = 1.0 / (n as f32);
        assert!((grad[0] - expected_other).abs() < 1e-5);
    }

    #[test]
    fn cross_entropy_masked_target_zero_grad_zero_loss() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut grad = vec![0.0; 4];
        let loss = cross_entropy_backward(&logits, TARGET_MASK, &mut grad);
        assert_eq!(loss, 0.0);
        for g in &grad { assert_eq!(*g, 0.0); }
    }

    #[test]
    fn cross_entropy_out_of_range_target_is_masked() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut grad = vec![0.0; 3];
        let loss = cross_entropy_backward(&logits, 99, &mut grad);
        assert_eq!(loss, 0.0);
        for g in &grad { assert_eq!(*g, 0.0); }
    }

    /// Compare analytical `rmsnorm_backward` against a finite-difference
    /// gradient of `L(x) = Σ dy·rmsnorm(x)`.
    #[test]
    fn rmsnorm_backward_matches_finite_difference() {
        let n = 12;
        let x: Vec<f32> = (1..=n).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (1..=n)
            .map(|i| (i as f32 * 0.7).sin() * 0.4 + 1.0)
            .collect();
        let dy: Vec<f32> = (1..=n)
            .map(|i| (i as f32 * 1.3).cos() * 0.5)
            .collect();
        let eps = 1e-6f32;

        let mut dx = vec![0.0; n];
        rmsnorm_backward(&x, Some(&w), &dy, eps, &mut dx);

        let mut y = vec![0.0; n];
        let h = 5e-4f32;
        for i in 0..n {
            let mut xp = x.clone();
            xp[i] += h;
            rmsnorm(&xp, Some(&w), eps, &mut y);
            let lp: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let mut xm = x.clone();
            xm[i] -= h;
            rmsnorm(&xm, Some(&w), eps, &mut y);
            let lm: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let num = (lp - lm) / (2.0 * h);
            let denom = dx[i].abs().max(1e-3);
            assert!(
                (dx[i] - num).abs() / denom < 5e-3,
                "i={i} analytic={a} numeric={num}",
                a = dx[i],
            );
        }
    }

    /// Same finite-difference check for RMSNorm with `w = None`.
    #[test]
    fn rmsnorm_backward_no_weight_matches_finite_difference() {
        let n = 8;
        let x: Vec<f32> = (1..=n).map(|i| (i as f32) * 0.2).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let eps = 1e-6f32;
        let mut dx = vec![0.0; n];
        rmsnorm_backward(&x, None, &dy, eps, &mut dx);
        let mut y = vec![0.0; n];
        let h = 5e-4f32;
        for i in 0..n {
            let mut xp = x.clone();
            xp[i] += h;
            rmsnorm(&xp, None, eps, &mut y);
            let lp: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let mut xm = x.clone();
            xm[i] -= h;
            rmsnorm(&xm, None, eps, &mut y);
            let lm: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let num = (lp - lm) / (2.0 * h);
            let denom = dx[i].abs().max(1e-3);
            assert!((dx[i] - num).abs() / denom < 5e-3);
        }
    }

    /// Per-row RMSNorm backward must equal calling the single-row
    /// `rmsnorm_backward` on each row independently. Verifies the loop
    /// version matches the well-tested single-row version row-for-row.
    #[test]
    fn rmsnorm_per_row_backward_matches_single_row_loop() {
        let n_rows = 3;
        let n = 8;
        let total = n_rows * n;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.13).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.07).collect();
        let dy: Vec<f32> = (0..total).map(|i| (i as f32 * 0.31).sin()).collect();
        let eps = 1e-6f32;

        // Per-row variant.
        let mut dx = vec![0.0; total];
        rmsnorm_per_row_backward(&x, Some(&w), &dy, eps, n_rows, n, &mut dx);

        // Reference: loop over rows, call single-row backward.
        let mut dx_ref = vec![0.0; total];
        for r in 0..n_rows {
            let xs = &x[r * n..(r + 1) * n];
            let dys = &dy[r * n..(r + 1) * n];
            let dxs = &mut dx_ref[r * n..(r + 1) * n];
            rmsnorm_backward(xs, Some(&w), dys, eps, dxs);
        }

        for (a, b) in dx.iter().zip(dx_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "per-row vs single-row mismatch: {a} vs {b}");
        }

        // Also unweighted.
        let mut dx_u = vec![0.0; total];
        rmsnorm_per_row_backward(&x, None, &dy, eps, n_rows, n, &mut dx_u);
        let mut dx_u_ref = vec![0.0; total];
        for r in 0..n_rows {
            let xs = &x[r * n..(r + 1) * n];
            let dys = &dy[r * n..(r + 1) * n];
            let dxs = &mut dx_u_ref[r * n..(r + 1) * n];
            rmsnorm_backward(xs, None, dys, eps, dxs);
        }
        for (a, b) in dx_u.iter().zip(dx_u_ref.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// GeGLU backward finite-difference check on both inputs.
    #[test]
    fn geglu_backward_matches_finite_difference() {
        let n = 6;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 2.0) * 0.4).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.9).sin()).collect();

        let mut d_gate = vec![0.0; n];
        let mut d_up = vec![0.0; n];
        geglu_backward(&gate, &up, &dy, &mut d_gate, &mut d_up);

        let mut y = vec![0.0; n];
        let h = 5e-4f32;
        for i in 0..n {
            // Perturb gate[i]
            let mut gp = gate.clone();
            gp[i] += h;
            geglu_split(&gp, &up, &mut y);
            let lp: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let mut gm = gate.clone();
            gm[i] -= h;
            geglu_split(&gm, &up, &mut y);
            let lm: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let num_g = (lp - lm) / (2.0 * h);
            assert!(
                (d_gate[i] - num_g).abs() < 1e-3,
                "gate i={i} analytic={a} numeric={num_g}",
                a = d_gate[i],
            );

            // Perturb up[i]
            let mut upp = up.clone();
            upp[i] += h;
            geglu_split(&gate, &upp, &mut y);
            let lp: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let mut upm = up.clone();
            upm[i] -= h;
            geglu_split(&gate, &upm, &mut y);
            let lm: f32 = y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
            let num_u = (lp - lm) / (2.0 * h);
            assert!(
                (d_up[i] - num_u).abs() < 1e-3,
                "up i={i} analytic={a} numeric={num_u}",
                a = d_up[i],
            );
        }
    }

    /// One-step Adam: hand-compute the expected update and compare.
    #[test]
    fn adam_one_step_matches_hand_calc() {
        let mut p = vec![1.0f32, -0.5];
        let mut m = vec![0.0f32, 0.0];
        let mut v = vec![0.0f32, 0.0];
        let g = vec![0.1f32, -0.2];
        let lr = 0.01;
        let b1 = 0.9;
        let b2 = 0.999;
        let eps = 1e-8;

        adam_step(&g, &mut p, &mut m, &mut v, lr, b1, b2, eps, 0.0, 1);

        // After step 1 with m_prev = v_prev = 0:
        //   m_new = (1 - β₁) g = 0.1 g
        //   v_new = (1 - β₂) g² = 0.001 g²
        //   m̂ = m_new / (1 - β₁¹) = m_new / 0.1 = g
        //   v̂ = v_new / (1 - β₂¹) = v_new / 0.001 = g²
        //   update = lr · g / (|g| + eps)
        // So |update| ≈ lr regardless of |g|.
        let expected_0 = 1.0 - lr * 0.1 / (0.1 + eps);
        let expected_1 = -0.5 - lr * (-0.2) / (0.2 + eps);
        assert!((p[0] - expected_0).abs() < 1e-6, "p[0]={p0} expected={expected_0}", p0 = p[0]);
        assert!((p[1] - expected_1).abs() < 1e-6, "p[1]={p1} expected={expected_1}", p1 = p[1]);
    }

    /// Adam converges on a tiny convex problem: minimize ½ (param - target)²,
    /// gradient = (param - target). 200 steps should drive |param - target|
    /// below 1e-3 from a starting offset of 1.0.
    #[test]
    fn adam_converges_on_quadratic() {
        let mut param = vec![1.0f32];
        let mut m = vec![0.0f32];
        let mut v = vec![0.0f32];
        let target = 0.0f32;
        let lr = 0.05;
        for step in 1..=400u32 {
            let g = vec![param[0] - target];
            adam_step(&g, &mut param, &mut m, &mut v, lr, 0.9, 0.999, 1e-8, 0.0, step);
        }
        assert!((param[0] - target).abs() < 1e-3, "converged to {p}", p = param[0]);
    }

    /// LoRA composed forward + backward — finite-difference check.
    ///
    /// Models a single LoRA-wrapped projection: `y = scale · B @ A @ x`.
    /// Defines `L = Σ dy · y` and verifies that the primitive composition
    ///     z   = A @ x
    ///     y  += scale · B @ z
    ///     u   = Bᵀ @ dy
    ///     dA += scale · outer(u, x)
    ///     dB += scale · outer(dy, z)
    ///     dx += scale · Aᵀ @ u
    /// reproduces the analytical gradients across all of A, B, and x.
    #[test]
    fn lora_forward_backward_matches_finite_difference() {
        let k = 6usize;
        let r = 3usize;
        let n = 5usize;
        let scale = 0.5f32;
        let a: Vec<f32> = (0..r * k).map(|i| (i as f32 * 0.17).sin() * 0.4).collect();
        let b: Vec<f32> = (0..n * r).map(|i| (i as f32 * 0.29).cos() * 0.3).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.31).sin() * 0.5 + 0.1).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.47).cos() * 0.3 + 0.2).collect();

        // Forward: z = A @ x, y = scale · B @ z
        let forward = |a_in: &[f32], b_in: &[f32], x_in: &[f32]| -> Vec<f32> {
            let mut z = vec![0f32; r];
            lora_matmul_row(a_in, x_in, &mut z, k, r, 1.0, false);
            let mut y = vec![0f32; n];
            lora_matmul_row(b_in, &z, &mut y, r, n, scale, false);
            y
        };

        // Analytical backward:
        let mut z = vec![0f32; r];
        lora_matmul_row(&a, &x, &mut z, k, r, 1.0, false);
        let mut u = vec![0f32; r];
        lora_matmul_col(&b, &dy, &mut u, n, r, 1.0, false);
        let mut da = vec![0f32; r * k];
        lora_outer_add(&u, &x, &mut da, scale, false);
        let mut db = vec![0f32; n * r];
        lora_outer_add(&dy, &z, &mut db, scale, false);
        let mut dx = vec![0f32; k];
        lora_matmul_col(&a, &u, &mut dx, r, k, scale, false);

        // Finite-difference reference: perturb each (param, index) and
        // measure Δ(L) / (2h).
        let h = 1e-3f32;
        let loss = |y: &[f32]| -> f32 {
            y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum()
        };

        // dA
        for p in 0..r {
            for ki in 0..k {
                let mut ap = a.clone();
                ap[p * k + ki] += h;
                let yp = forward(&ap, &b, &x);
                let mut am = a.clone();
                am[p * k + ki] -= h;
                let ym = forward(&am, &b, &x);
                let num = (loss(&yp) - loss(&ym)) / (2.0 * h);
                let ana = da[p * k + ki];
                assert!(
                    (num - ana).abs() < 1e-3,
                    "dA[{p}, {ki}] analytic={ana} numeric={num}"
                );
            }
        }
        // dB
        for j in 0..n {
            for p in 0..r {
                let mut bp = b.clone();
                bp[j * r + p] += h;
                let yp = forward(&a, &bp, &x);
                let mut bm = b.clone();
                bm[j * r + p] -= h;
                let ym = forward(&a, &bm, &x);
                let num = (loss(&yp) - loss(&ym)) / (2.0 * h);
                let ana = db[j * r + p];
                assert!(
                    (num - ana).abs() < 1e-3,
                    "dB[{j}, {p}] analytic={ana} numeric={num}"
                );
            }
        }
        // dx
        for ki in 0..k {
            let mut xp = x.clone();
            xp[ki] += h;
            let yp = forward(&a, &b, &xp);
            let mut xm = x.clone();
            xm[ki] -= h;
            let ym = forward(&a, &b, &xm);
            let num = (loss(&yp) - loss(&ym)) / (2.0 * h);
            let ana = dx[ki];
            assert!(
                (num - ana).abs() < 1e-3,
                "dx[{ki}] analytic={ana} numeric={num}"
            );
        }
    }

    /// Finite-difference check for `attention_backward`.
    ///
    /// `L(q, k_hist, v_hist) = Σ d_out · attention(q, k_hist, v_hist)`. Numerically
    /// perturb every element and compare to the analytical gradients.
    #[test]
    fn attention_backward_matches_finite_difference() {
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let history_len = 3usize;
        let q_len = n_heads * head_dim;
        let kv_len = history_len * n_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_len).map(|i| (i as f32 * 0.31).sin() * 0.4).collect();
        let k_hist: Vec<f32> = (0..kv_len).map(|i| (i as f32 * 0.17).cos() * 0.3).collect();
        let v_hist: Vec<f32> = (0..kv_len).map(|i| (i as f32 * 0.23).sin() * 0.5).collect();
        let d_out: Vec<f32> = (0..q_len).map(|i| (i as f32 * 0.47).cos() * 0.3 + 0.1).collect();

        // Forward + save probs
        let mut out = vec![0f32; q_len];
        let mut probs = vec![0f32; n_heads * history_len];
        attention_forward(
            &q, &k_hist, &v_hist, &mut out, &mut probs,
            head_dim, n_heads, n_kv_heads, history_len,
        );

        // Analytical
        let mut d_q = vec![0f32; q_len];
        let mut d_k = vec![0f32; kv_len];
        let mut d_v = vec![0f32; kv_len];
        attention_backward(
            &q, &k_hist, &v_hist, &probs, &d_out,
            &mut d_q, &mut d_k, &mut d_v,
            head_dim, n_heads, n_kv_heads, history_len,
        );

        let loss = |q_in: &[f32], k_in: &[f32], v_in: &[f32]| -> f32 {
            let mut o = vec![0f32; q_len];
            let mut p = vec![0f32; n_heads * history_len];
            attention_forward(
                q_in, k_in, v_in, &mut o, &mut p,
                head_dim, n_heads, n_kv_heads, history_len,
            );
            o.iter().zip(d_out.iter()).map(|(a, b)| a * b).sum::<f32>()
        };

        let h = 1e-3f32;
        let check = |label: &str, ana: &[f32], v: &[f32], idx_fn: &dyn Fn(usize) -> (Vec<f32>, Vec<f32>, Vec<f32>)| {
            for i in 0..v.len() {
                let (qp, kp, vp) = idx_fn(i);
                let lp = loss(&qp, &kp, &vp);
                let (qm, km, vm) = idx_fn(i + v.len()); // marker for minus
                let lm = loss(&qm, &km, &vm);
                let num = (lp - lm) / (2.0 * h);
                let denom = ana[i].abs().max(5e-3);
                assert!(
                    (ana[i] - num).abs() / denom < 1e-1,
                    "{label} i={i} analytic={a} numeric={num}",
                    a = ana[i],
                );
            }
        };

        // q gradient
        check("d_q", &d_q, &q, &|i| {
            let mut perturbed = q.clone();
            let real_i = if i < q.len() { i } else { i - q.len() };
            let sign = if i < q.len() { 1.0 } else { -1.0 };
            perturbed[real_i] += sign * h;
            (perturbed, k_hist.clone(), v_hist.clone())
        });

        // k_hist gradient
        check("d_k", &d_k, &k_hist, &|i| {
            let mut perturbed = k_hist.clone();
            let real_i = if i < k_hist.len() { i } else { i - k_hist.len() };
            let sign = if i < k_hist.len() { 1.0 } else { -1.0 };
            perturbed[real_i] += sign * h;
            (q.clone(), perturbed, v_hist.clone())
        });

        // v_hist gradient
        check("d_v", &d_v, &v_hist, &|i| {
            let mut perturbed = v_hist.clone();
            let real_i = if i < v_hist.len() { i } else { i - v_hist.len() };
            let sign = if i < v_hist.len() { 1.0 } else { -1.0 };
            perturbed[real_i] += sign * h;
            (q.clone(), k_hist.clone(), perturbed)
        });
    }

    /// rope_neox followed by rope_neox_backward at the same `pos` should
    /// restore the original input.
    #[test]
    fn rope_neox_forward_then_backward_is_identity() {
        let head_dim = 8;
        let n_heads = 2;
        let rope_dims = 8;
        let pos = 7;
        let base = 10_000.0f32;
        let mut x: Vec<f32> = (0..head_dim * n_heads).map(|i| (i as f32) * 0.13 - 1.0).collect();
        let orig = x.clone();
        rope_neox(&mut x, head_dim, n_heads, pos, rope_dims, base, None);
        rope_neox_backward(&mut x, head_dim, n_heads, pos, rope_dims, base, None);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "rope roundtrip drift {a} != {b}");
        }
    }

    #[test]
    fn cross_entropy_argmax_target_gives_small_loss() {
        // Strong logit at index 2: softmax(2) close to 1, loss close to 0.
        let logits = vec![0.0f32, 0.0, 10.0, 0.0];
        let mut grad = vec![0.0f32; 4];
        let loss = cross_entropy_backward(&logits, 2, &mut grad);
        assert!(loss < 0.01, "loss {loss} should be near 0");
        // d_logits[2] = softmax[2] - 1 close to 0; others close to 0.
        assert!(grad[2].abs() < 0.01);
        for (i, g) in grad.iter().enumerate() {
            if i != 2 {
                assert!(g.abs() < 0.01, "off-target grad {g} at {i}");
            }
        }
    }
}
