//! CPU oracle for the Gemma 4 sparse-MoE FFN (`gemma4:26b-a4b`).
//!
//! Mirrors Ollama's `model/models/gemma4/model_text.go` (TextRouter +
//! TextMoEBlock) **1:1** — diff against that Go file, not a llama.cpp port.
//!
//! On MoE layers the expert block runs IN PARALLEL with the dense MLP:
//!
//! ```text
//! hidden ──ffn_norm──▶ dense MLP ──post_ffw_norm_1──┐
//!        ├─router(hidden)─┐                          ├─ sum ─post_ffw_norm─▶ +residual
//!        └─pre_ffw_norm_2─┴▶ top-k experts ─post_ffw_norm_2┘
//! ```
//!
//! The router consumes the RAW post-attention hidden state (not the
//! ffn-normed one): unweighted RMSNorm → ×1/√d_model → ×`ffn_gate_inp.scale`
//! → proj `ffn_gate_inp` → softmax over ALL experts → top-k → renormalize
//! the selected weights by their sum.

use crate::error::Result;
use crate::model::config::Gemma4Config;
use crate::reference::ops::{geglu_split, matvec, rmsnorm, softmax};
use crate::reference::weights::Weights;

/// Pure router math: softmax over all expert scores, pick top-k by routing
/// weight, renormalize the k weights to sum to 1. Returns `(expert, weight)`
/// pairs in descending-weight order. Factored out so the GPU kernel has a
/// slice-level oracle to diff against.
pub fn softmax_topk_renorm(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut probs = scores.to_vec();
    softmax(&mut probs);
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    // Descending by probability; ties resolve to the lower index (matches
    // ggml's argsort-based TopK which is stable on ties).
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
    let top = &idx[..k.min(idx.len())];
    let sum: f32 = top.iter().map(|&i| probs[i]).sum();
    top.iter().map(|&i| (i, probs[i] / sum)).collect()
}

/// Whether layer `i` carries a routed-expert block (tensor presence decides,
/// mirroring Ollama's nil-field checks).
pub fn layer_has_moe(weights: &Weights, i: u32) -> bool {
    weights.has(&format!("blk.{i}.ffn_gate_inp.weight"))
        && (weights.has(&format!("blk.{i}.ffn_gate_up_exps.weight"))
            || (weights.has(&format!("blk.{i}.ffn_gate_exps.weight"))
                && weights.has(&format!("blk.{i}.ffn_up_exps.weight"))))
        && weights.has(&format!("blk.{i}.ffn_down_exps.weight"))
}

/// Run the router for layer `i` on the raw post-attention hidden state.
/// Returns the selected `(expert, weight)` pairs.
pub fn route(
    cfg: &Gemma4Config,
    weights: &Weights,
    i: u32,
    hidden: &[f32],
) -> Result<Vec<(usize, f32)>> {
    let d_model = cfg.d_model as usize;
    let n_experts = cfg.expert_count as usize;
    let prefix = format!("blk.{i}.");

    // Unweighted RMSNorm, then ×1/√d_model.
    let mut x = vec![0f32; d_model];
    rmsnorm(hidden, None, cfg.rms_norm_eps, &mut x);
    let inv_sqrt_d = 1.0 / (d_model as f32).sqrt();
    for v in &mut x {
        *v *= inv_sqrt_d;
    }
    // ×learned per-channel scale, when present.
    if let Some(s) = weights.load_opt(&format!("{prefix}ffn_gate_inp.scale"))? {
        for (v, sv) in x.iter_mut().zip(s.iter()) {
            *v *= sv;
        }
    }
    // Project to expert logits.
    let router_w = weights.load(&format!("{prefix}ffn_gate_inp.weight"))?;
    let mut scores = vec![0f32; n_experts];
    matvec(&router_w, d_model, n_experts, &x, &mut scores);

    Ok(softmax_topk_renorm(&scores, cfg.expert_used_count as usize))
}

/// The expert block for layer `i`: takes the `pre_ffw_norm_2`-normed hidden
/// state plus the router's selection, returns the weighted expert sum
/// (NOT yet `post_ffw_norm_2`-normed — the caller owns the norms).
///
/// Loads one expert slice at a time to bound CPU-oracle peak memory.
pub fn moe_experts(
    cfg: &Gemma4Config,
    weights: &Weights,
    i: u32,
    x: &[f32],
    selected: &[(usize, f32)],
) -> Result<Vec<f32>> {
    let d_model = cfg.d_model as usize;
    let prefix = format!("blk.{i}.");
    let fused_name = format!("{prefix}ffn_gate_up_exps.weight");
    let fused = weights.has(&fused_name);

    // Per-expert down-projection scale (optional; two spellings, per Ollama's
    // `gguf:"ffn_down_exps.scale,alt:ffn_gate_inp.per_expert_scale"`).
    let down_scale = match weights.load_opt(&format!("{prefix}ffn_down_exps.scale"))? {
        Some(s) => Some(s),
        None => weights.load_opt(&format!("{prefix}ffn_gate_inp.per_expert_scale"))?,
    };

    let mut out = vec![0f32; d_model];
    for &(e, w_e) in selected {
        // gate/up — fused [d_model, 2*ffn, E] (gate = rows 0..ffn, up = ffn..2ffn)
        // or split tensors.
        let (gate, up) = if fused {
            let gu = weights.load_expert(&fused_name, e)?;
            let n_ff = gu.len() / d_model / 2;
            // gu is row-major [in=d_model contiguous] × 2*n_ff rows: rows
            // 0..n_ff are gate, n_ff..2n_ff are up (Go slices dim 0 of the
            // OUTPUT, which is the row axis here).
            let mut g = vec![0f32; n_ff];
            let mut u = vec![0f32; n_ff];
            matvec(&gu[..n_ff * d_model], d_model, n_ff, x, &mut g);
            matvec(&gu[n_ff * d_model..], d_model, n_ff, x, &mut u);
            (g, u)
        } else {
            let gw = weights.load_expert(&format!("{prefix}ffn_gate_exps.weight"), e)?;
            let n_ff = gw.len() / d_model;
            let mut g = vec![0f32; n_ff];
            matvec(&gw, d_model, n_ff, x, &mut g);
            drop(gw);
            let uw = weights.load_expert(&format!("{prefix}ffn_up_exps.weight"), e)?;
            let mut u = vec![0f32; n_ff];
            matvec(&uw, d_model, n_ff, x, &mut u);
            (g, u)
        };

        let n_ff = gate.len();
        let mut act = vec![0f32; n_ff];
        geglu_split(&gate, &up, &mut act);
        drop(gate);
        drop(up);

        let dw = weights.load_expert(&format!("{prefix}ffn_down_exps.weight"), e)?;
        let mut down = vec![0f32; d_model];
        matvec(&dw, n_ff, d_model, &act, &mut down);
        drop(dw);

        let scale_e = down_scale.as_ref().map(|s| s[e]).unwrap_or(1.0);
        for (o, d) in out.iter_mut().zip(down.iter()) {
            *o += w_e * scale_e * d;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_topk_renorm_selects_and_renormalizes() {
        // scores → softmax probs ranked 3 > 1 > 0 > 2; top-2 = {3, 1},
        // renormalized to sum 1 with the 3:1 ratio preserved from softmax.
        let scores = [1.0f32, 2.0, 0.0, 3.0];
        let sel = softmax_topk_renorm(&scores, 2);
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].0, 3);
        assert_eq!(sel[1].0, 1);
        let wsum: f32 = sel.iter().map(|(_, w)| w).sum();
        assert!((wsum - 1.0).abs() < 1e-6, "renormalized weights sum to 1");
        // softmax(3)/softmax(2) = e — ratio must survive renormalization.
        let ratio = sel[0].1 / sel[1].1;
        assert!(
            (ratio - std::f32::consts::E).abs() < 1e-4,
            "ratio {ratio} != e"
        );
    }

    #[test]
    fn softmax_topk_renorm_k1_is_argmax_weight_one() {
        let scores = [0.5f32, -1.0, 4.0];
        let sel = softmax_topk_renorm(&scores, 1);
        assert_eq!(sel, vec![(2, 1.0)]);
    }

    #[test]
    fn softmax_topk_renorm_k_ge_n_keeps_plain_softmax() {
        let scores = [1.0f32, 1.0];
        let sel = softmax_topk_renorm(&scores, 5);
        assert_eq!(sel.len(), 2);
        assert!((sel[0].1 - 0.5).abs() < 1e-6);
        assert!((sel[1].1 - 0.5).abs() < 1e-6);
    }
}
