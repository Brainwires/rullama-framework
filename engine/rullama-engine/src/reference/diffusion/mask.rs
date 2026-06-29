//! Region-aware attention mask for DiffusionGemma's unified [prompt | canvas]
//! forward (the `kv_cache=false` path the entropy-bound sampler uses).
//!
//! Mirrors `llm_graph_input_attn_diffusion::set_input` in llama.cpp PR 24423
//! (`src/models/diffusion-gemma.cpp`) 1:1. For query position `q` and key
//! position `k` over `n_tokens = P + C` (P prompt, C canvas):
//!
//! - **canvas query** (`q >= P`): bidirectional.
//!   - global layer: attends to everything (all prompt + all canvas).
//!   - SWA layer: all canvas + the last `n_swa - 1` prompt positions
//!     (`k >= P - n_swa + 1`).
//! - **prompt query** (`q < P`): causal over earlier prompt, NEVER canvas
//!   (`!k_is_canvas && k <= q`); on SWA layers additionally windowed to
//!   `k > q - n_swa` (ggml's `is_masked_swa`: a key is masked when
//!   `q - k >= n_swa`).
//!
//! `allowed(...)` returns whether attention from `q` to `k` is permitted; the
//! caller adds `-inf` to disallowed score entries before softmax.

/// Per the layer kind, is the query→key attention edge allowed?
#[inline]
pub fn allowed(q: usize, k: usize, prompt_len: usize, n_swa: usize, swa_layer: bool) -> bool {
    let q_is_canvas = q >= prompt_len;
    let k_is_canvas = k >= prompt_len;
    if q_is_canvas {
        if swa_layer {
            // last (n_swa - 1) prompt + all canvas
            k_is_canvas || k + n_swa > prompt_len // k >= prompt_len - (n_swa - 1)
        } else {
            true
        }
    } else {
        // prompt query: causal over earlier prompt, never canvas
        if k_is_canvas || k > q {
            return false;
        }
        if swa_layer {
            // is_masked_swa: masked when q - k >= n_swa  ⇒ allowed when q - k < n_swa
            q - k < n_swa
        } else {
            true
        }
    }
}

/// Build the full `[n_tokens × n_tokens]` additive mask (row-major, query-major:
/// `mask[q * n_tokens + k]`) — `0.0` where allowed, `-inf` where masked.
pub fn build_unified_mask(
    n_tokens: usize,
    prompt_len: usize,
    n_swa: usize,
    swa_layer: bool,
) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; n_tokens * n_tokens];
    for q in 0..n_tokens {
        for k in 0..n_tokens {
            if allowed(q, k, prompt_len, n_swa, swa_layer) {
                m[q * n_tokens + k] = 0.0;
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_rows_are_causal_and_never_see_canvas() {
        let (p, c, n_swa) = (4usize, 3usize, 1024usize);
        let n = p + c;
        // Global layer.
        for q in 0..p {
            for k in 0..n {
                let a = allowed(q, k, p, n_swa, false);
                if k >= p {
                    assert!(!a, "prompt q{q} must not see canvas k{k}");
                } else {
                    assert_eq!(a, k <= q, "prompt q{q}→k{k} must be causal");
                }
            }
        }
    }

    #[test]
    fn canvas_rows_are_bidirectional_global() {
        let (p, c) = (4usize, 3usize);
        let n = p + c;
        for q in p..n {
            for k in 0..n {
                assert!(
                    allowed(q, k, p, 1024, false),
                    "global canvas q{q} must attend everywhere (k{k})"
                );
            }
        }
    }

    #[test]
    fn canvas_rows_swa_window_the_prompt_but_keep_full_canvas() {
        // n_swa = 3 ⇒ canvas keeps the last 2 prompt positions (k >= P-2).
        let (p, c, n_swa) = (5usize, 4usize, 3usize);
        let n = p + c;
        for q in p..n {
            for k in 0..p {
                let a = allowed(q, k, p, n_swa, true);
                assert_eq!(a, k >= p - (n_swa - 1), "swa canvas q{q}→prompt k{k}");
            }
            for k in p..n {
                assert!(allowed(q, k, p, n_swa, true), "swa canvas keeps all canvas");
            }
        }
    }

    #[test]
    fn prompt_rows_swa_are_causal_windowed() {
        // n_swa = 2 ⇒ prompt query sees only k in (q-2, q].
        let (p, n_swa) = (6usize, 2usize);
        for q in 0..p {
            for k in 0..p {
                let a = allowed(q, k, p, n_swa, true);
                let want = k <= q && q - k < n_swa;
                assert_eq!(a, want, "swa prompt q{q}→k{k}");
            }
        }
    }

    #[test]
    fn full_mask_matches_allowed() {
        let m = build_unified_mask(7, 4, 3, true);
        for q in 0..7 {
            for k in 0..7 {
                let v = m[q * 7 + k];
                if allowed(q, k, 4, 3, true) {
                    assert_eq!(v, 0.0);
                } else {
                    assert_eq!(v, f32::NEG_INFINITY);
                }
            }
        }
    }
}
