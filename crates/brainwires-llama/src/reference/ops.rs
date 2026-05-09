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
/// Uses the exact (erf-based) GELU, matching ggml's `ggml_geglu_split`.
pub fn geglu_split(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), gate.len());
    const SQRT_HALF: f32 = 0.707_106_77;
    for i in 0..gate.len() {
        let g = gate[i];
        let gelu = 0.5 * g * (1.0 + erf_approx(g * SQRT_HALF));
        out[i] = gelu * up[i];
    }
}

/// Abramowitz & Stegun 7.1.26 erf approximation; max error ~1.5e-7. Matches ggml's GELU.
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f32 = 0.254_829_592;
    const A2: f32 = -0.284_496_736;
    const A3: f32 = 1.421_413_741;
    const A4: f32 = -1.453_152_027;
    const A5: f32 = 1.061_405_429;
    const P:  f32 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
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
}
