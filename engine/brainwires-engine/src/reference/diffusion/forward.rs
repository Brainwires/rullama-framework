//! DiffusionGemma CPU forward primitives.
//!
//! The full canvas-forward (wiring the backbone per the PR's
//! `diffusion-gemma.cpp` graph — reusing the validated gemma4 MoE FFN +
//! q/k/v/rope/norm ops, with the region mask in [`super::mask`] and the dual
//! enc/dec per-layer scales) lands once the `llama-diffusion-cli` parity
//! oracle is built. This module starts with the one genuinely-new op that has
//! no gemma4 analogue: **full-sequence masked attention** (the existing CPU
//! attention is strictly causal + per-token KV append; the canvas forward is
//! non-autoregressive — every position attends per a region mask in one pass).

use super::mask::allowed;
use crate::reference::ops::softmax;

/// Non-autoregressive multi-head attention over a full token sequence with a
/// region mask. Layout: `q` is `[n_tokens, n_heads, head_dim]` row-major,
/// `k`/`v` are `[n_tokens, n_kv_heads, head_dim]` (GQA: each query head `qh`
/// reads kv head `qh / (n_heads/n_kv_heads)`). Score scale is 1.0 (Gemma 4
/// folds it into the Q-norm weights — matches PR's `f_attention_scale=1.0`).
/// Returns `[n_tokens, n_heads, head_dim]`.
///
/// `prompt_len` / `n_swa` / `swa_layer` drive the per-edge mask via
/// [`allowed`]; pass `swa_layer=false` + any `n_swa` for a global layer.
#[allow(clippy::too_many_arguments)]
pub fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    prompt_len: usize,
    n_swa: usize,
    swa_layer: bool,
) -> Vec<f32> {
    let heads_per_kv = (n_heads / n_kv_heads).max(1);
    let mut out = vec![0f32; n_tokens * n_heads * head_dim];
    let mut scores = vec![0f32; n_tokens];

    for qi in 0..n_tokens {
        for qh in 0..n_heads {
            let kvh = qh / heads_per_kv;
            let q_off = (qi * n_heads + qh) * head_dim;

            for (kj, score) in scores.iter_mut().enumerate() {
                if !allowed(qi, kj, prompt_len, n_swa, swa_layer) {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (kj * n_kv_heads + kvh) * head_dim;
                let mut acc = 0f32;
                for d in 0..head_dim {
                    acc += q[q_off + d] * k[k_off + d];
                }
                *score = acc; // scale 1.0
            }
            softmax(&mut scores);

            let o_off = (qi * n_heads + qh) * head_dim;
            for (kj, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let v_off = (kj * n_kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bidirectional (all-canvas, no prompt) attention on a hand-computed
    /// 2-token / 1-head / head_dim-2 case.
    #[test]
    fn masked_attention_bidirectional_hand_case() {
        // q0=[1,0] q1=[0,1]; k0=[1,0] k1=[0,1]; v0=[10,20] v1=[30,40].
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        // prompt_len 0 ⇒ both tokens are canvas ⇒ bidirectional (global).
        let out = masked_attention(&q, &k, &v, 2, 1, 1, 2, 0, 1024, false);

        // token0: softmax([q0·k0, q0·k1]) = softmax([1,0]).
        let (a, b) = {
            let e = 1f32.exp();
            (e / (e + 1.0), 1.0 / (e + 1.0))
        };
        let exp0 = [a * 10.0 + b * 30.0, a * 20.0 + b * 40.0];
        // token1: softmax([0,1]) = (b, a).
        let exp1 = [b * 10.0 + a * 30.0, b * 20.0 + a * 40.0];
        for (i, &e) in exp0.iter().chain(exp1.iter()).enumerate() {
            assert!((out[i] - e).abs() < 1e-5, "out[{i}]={} != {e}", out[i]);
        }
    }

    /// A prompt token (causal) must ignore the canvas; the result equals
    /// attending only over earlier prompt — verifiable by deleting the canvas.
    #[test]
    fn prompt_row_ignores_canvas() {
        // P=1 prompt + 1 canvas. token0 is prompt: sees only itself (causal,
        // never canvas) ⇒ out0 == v0 exactly regardless of canvas content.
        let q = vec![0.5, 0.5, 1.0, 0.0];
        let k = vec![0.3, 0.7, 9.9, 9.9];
        let v = vec![10.0, 20.0, 999.0, 999.0];
        let out = masked_attention(&q, &k, &v, 2, 1, 1, 2, 1, 1024, false);
        assert!((out[0] - 10.0).abs() < 1e-5, "prompt out0[0]={}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-5, "prompt out0[1]={}", out[1]);
    }

    /// GQA: 2 query heads share 1 kv head — both heads read the same K/V.
    #[test]
    fn gqa_two_query_heads_one_kv() {
        // n_tokens=1 (self-attend), 2 heads, 1 kv head, head_dim=1.
        // Single token attends to itself → out = v for both heads.
        let q = vec![3.0, 7.0]; // [tok0][h0,h1]
        let k = vec![1.0];
        let v = vec![42.0];
        let out = masked_attention(&q, &k, &v, 1, 2, 1, 1, 0, 1024, false);
        assert!((out[0] - 42.0).abs() < 1e-5);
        assert!((out[1] - 42.0).abs() < 1e-5);
    }
}
