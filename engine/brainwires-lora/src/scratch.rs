//! `TrainingScratch` — per-step GPU scratch buffers for the backward pass.
//!
//! Owns the loss / d_logits buffers shared by every step plus
//! per-layer **activation save** buffers that the forward pass writes
//! to and the backward pass consumes. Allocations are sized off
//! `Gemma4Config` + `max_seq_len` so the lifetime of a scratch is the
//! lifetime of a `TrainingSession` (no per-step alloc).
//!
//! ### M0 simplification — single-position capture
//!
//! M0's NextToken loss only needs activations at the final query
//! position. Per-layer capture buffers are sized for one position
//! (`max_seq_len = 1` effectively). The K/V history is read directly
//! from the model's existing KV cache (`Forward::kv_k[i]`,
//! `Forward::kv_v[i]`) — no separate capture.
//!
//! ### Per-layer activations (what the backward needs)
//!
//! | name            | shape                             | needed by                                  |
//! | --------------- | --------------------------------- | ------------------------------------------ |
//! | `hidden_in`     | `[d_model]`                       | attn rmsnorm backward (input)              |
//! | `norm_x_attn`   | `[d_model]`                       | matmul_q4_k_backward_input for q/k/v       |
//! | `q_pre_norm`    | `[n_heads · head_dim]`            | q_norm rmsnorm backward (input)            |
//! | `q_post_rope`   | `[n_heads · head_dim]`            | attention backward (probs recompute + dkv) |
//! | `k_pre_norm`    | `[n_kv · head_dim]`               | k_norm rmsnorm backward (input)            |
//! | `v_pre_norm`    | `[n_kv · head_dim]`               | v_norm rmsnorm backward (input)            |
//! | `attn_out`      | `[n_heads · head_dim]`            | matmul_q4_k_backward_input (o_proj input)  |
//! | `pre_ffn_rms`   | `[d_model]`                       | ffn rmsnorm backward (input)               |
//! | `norm_x_ffn`    | `[d_model]`                       | matmul_q4_k_backward_input for gate/up     |
//! | `ffn_gate`      | `[ffn_inter]`                     | geglu backward                             |
//! | `ffn_up`        | `[ffn_inter]`                     | geglu backward                             |
//! | `ffn_act`       | `[ffn_inter]`                     | matmul_q4_k_backward_input for ffn_down    |
//!
//! Activations live on the GPU as `wgpu::Buffer`s with
//! `STORAGE | COPY_DST | COPY_SRC`. The forward writes to them via
//! `Forward::step_capture(...)` which threads an
//! `Option<&mut LayerActivations>` through `encode_layer` and emits
//! `copy_buffer_to_buffer` at each capture point. The backward reads
//! them through the dispatchers added in tasks #9–#12.

use std::sync::Arc;

use rullama::backend::WgpuCtx;
use rullama::model::config::Gemma4Config;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

/// Per-layer activation buffer set. One per layer; addressed by layer index.
///
/// All buffers are single-position (final query position). M1 will extend
/// the leading axis to `[seq, …]` for the PerPosition loss path.
pub struct LayerActivations {
    /// `self.hidden` snapshot at the start of the layer (input to attn rmsnorm).
    pub hidden_in: Buffer,
    /// Output of the attn rmsnorm (input to q/k/v matmul + LoRA).
    pub norm_x_attn: Buffer,
    /// `self.q` snapshot (q matmul output, before q_norm rmsnorm).
    pub q_pre_norm: Buffer,
    /// `self.q_norm` snapshot AFTER RoPE (input to attention; reused in dkv pass).
    pub q_post_rope: Buffer,
    /// `self.k` snapshot (k matmul output, before k_norm rmsnorm).
    pub k_pre_norm: Buffer,
    /// `self.v` snapshot (v matmul output, before v_norm rmsnorm — unweighted).
    pub v_pre_norm: Buffer,
    /// Attention output (= input to o_proj matmul).
    pub attn_out: Buffer,
    /// o_proj matmul output (= input to post_attn_norm rmsnorm).
    pub attn_proj: Buffer,
    /// `self.hidden` snapshot after the attn residual add (input to ffn rmsnorm).
    pub pre_ffn_rms: Buffer,
    /// Output of the ffn rmsnorm (input to gate/up matmul + LoRA).
    pub norm_x_ffn: Buffer,
    /// Gate matmul output (one input to GEGLU).
    pub ffn_gate: Buffer,
    /// Up matmul output (one input to GEGLU).
    pub ffn_up: Buffer,
    /// GEGLU output (= input to ffn_down matmul).
    pub ffn_act: Buffer,
    /// ffn_down matmul output (= input to post_ffw_norm rmsnorm).
    pub ffn_out: Buffer,
    /// `self.ple_state` snapshot — output of `inp_gate_w · hidden`,
    /// input (gate branch) to PLE GEGLU. `[ple_dim]`.
    pub ple_state: Buffer,
    /// `self.ple_act` snapshot — output of PLE GEGLU, input to
    /// `proj_w` matmul. `[ple_dim]`.
    pub ple_act: Buffer,
    /// `self.ple_proj` snapshot — output of `proj_w` matmul, input
    /// to PLE rmsnorm (`post_norm_w`). `[d_model]`.
    pub ple_proj: Buffer,
}

/// Top-level scratch for a training step.
///
/// Layout invariants:
/// - Capture is single-position (NextToken / M0). M1 will widen the
///   per-layer buffers along a seq axis.
/// - `d_logits` / `loss` are dispatched once per step.
pub struct TrainingScratch {
    /// `d_logits[vocab]` from cross_entropy_backward.
    pub d_logits: Buffer,
    /// Scalar loss readback (1 f32).
    pub loss: Buffer,
    /// `d_hidden_final[d_model]` — gradient at the final-position hidden
    /// state after the output projection's backward.
    pub d_hidden_final: Buffer,
    /// Per-layer activation captures.
    pub layers: Vec<LayerActivations>,
    /// Running `d_hidden[d_model]` — the per-layer gradient on the
    /// residual stream that walks from the top of the model down to the
    /// embedding. Backward maintains a single running buffer here so
    /// each layer's backward reads from it and writes back to it.
    pub d_hidden: Buffer,
    /// Scratch for in-flight d(something) of d_model shape (e.g. dx out of
    /// rmsnorm_backward before residual_add merges it back into d_hidden).
    pub d_hidden_tmp: Buffer,
    /// Second d_model scratch — pairs with `d_hidden_tmp` for cases
    /// that accumulate two contributions (gate+up → d_norm_x_ffn,
    /// q+k+v → d_norm_x_attn).
    pub d_hidden_tmp2: Buffer,
    /// Scratch `[n_heads · max_history_len]` for the attention probs
    /// recomputed during backward (output of `attention_probs_chained`).
    pub attn_probs: Buffer,
    /// Staging buffer for `d_scores` in attention backward (pass-1 output,
    /// pass-2 input). Sized `[n_heads · max_history_len]`.
    pub attn_d_scores: Buffer,
    /// Scratch `[n_heads · head_dim]` — `d_q` from attention backward,
    /// also reused as d(q_post_rope) input to rope backward.
    pub d_q: Buffer,
    /// Scratch `[max_history_len · n_kv · head_dim]` — `d_k_hist`. For M0
    /// we only consume the row at `pos`, but the kernel writes all rows.
    pub d_k_hist: Buffer,
    /// Same shape as `d_k_hist` — `d_v_hist`.
    pub d_v_hist: Buffer,
    /// Scratch `[n_heads · head_dim]` — d(q before rope) post-rope-back.
    pub d_q_pre_rope: Buffer,
    /// Scratch `[n_kv · head_dim]` — d(k before rope) post-rope-back.
    pub d_k_pre_rope: Buffer,
    /// Scratch `[n_heads · head_dim]` — d(q matmul output).
    pub d_q_pre_norm: Buffer,
    /// Scratch `[n_kv · head_dim]` — d(k matmul output).
    pub d_k_pre_norm: Buffer,
    /// Scratch `[n_kv · head_dim]` — d(v matmul output) (unweighted v_norm).
    pub d_v_pre_norm: Buffer,
    /// Scratch `[ffn_inter]` — running d through ffn block.
    pub d_ffn_a: Buffer,
    /// Scratch `[ffn_inter]` — d_gate output of geglu_back.
    pub d_ffn_b: Buffer,
    /// Scratch `[ffn_inter]` — d_up output of geglu_back.
    pub d_ffn_c: Buffer,
    /// Scratch `[ple_dim]` — d output of PLE geglu_back gate branch.
    /// `0`-sized when the model has no PLE.
    pub d_ple_state: Buffer,
    /// Scratch `[ple_dim]` — d input to PLE geglu_back (= matmul-back
    /// output through `proj_w`).
    pub d_ple_act: Buffer,
    /// Scratch `[ple_dim]` — discarded `d_up` output of PLE
    /// geglu_back. The per-layer input has no LoRA / no grad sink.
    pub d_ple_up_discard: Buffer,
    /// Scratch `[ple_dim]` — copy of `self.per_layer[i*ple_dim..]`
    /// uploaded each PLE backward step to feed geglu_back's `up`
    /// input.
    pub ple_per_layer_tmp: Buffer,
    /// `[d_model]` window into a layer's seq-sized `norm_x_attn`
    /// capture. Pre-copied per backward iteration so kernel
    /// bindings stay single-position (`as_entire_binding`-shaped)
    /// without paying storage-buffer offset-alignment friction.
    pub norm_x_attn_window: Buffer,
    /// `[n_kv_max · head_dim_max]` window into a layer's seq-sized
    /// `k_pre_norm` capture.
    pub k_pre_norm_window: Buffer,
    /// `[n_kv_max · head_dim_max]` window into a layer's seq-sized
    /// `v_pre_norm` capture.
    pub v_pre_norm_window: Buffer,
    /// `[d_model]` window into `hidden_in` capture (input to attn rmsnorm).
    pub hidden_in_window: Buffer,
    /// `[n_heads · head_dim_max]` window into `q_pre_norm` capture.
    pub q_pre_norm_window: Buffer,
    /// `[n_heads · head_dim_max]` window into `q_post_rope` capture
    /// (the rope-rotated, normed Q at the layer's query position).
    pub q_post_rope_window: Buffer,
    /// `[n_heads · head_dim_max]` window into `attn_out` capture.
    pub attn_out_window: Buffer,
    /// `[d_model]` window into `attn_proj` capture.
    pub attn_proj_window: Buffer,
    /// `[d_model]` window into `pre_ffn_rms` capture.
    pub pre_ffn_rms_window: Buffer,
    /// `[d_model]` window into `norm_x_ffn` capture.
    pub norm_x_ffn_window: Buffer,
    /// `[ffn_inter_max]` window into `ffn_gate` capture.
    pub ffn_gate_window: Buffer,
    /// `[ffn_inter_max]` window into `ffn_up` capture.
    pub ffn_up_window: Buffer,
    /// `[ffn_inter_max]` window into `ffn_act` capture.
    pub ffn_act_window: Buffer,
    /// `[d_model]` window into `ffn_out` capture.
    pub ffn_out_window: Buffer,
    /// `[ple_dim]` window into `ple_state` capture.
    pub ple_state_window: Buffer,
    /// `[ple_dim]` window into `ple_act` capture.
    pub ple_act_window: Buffer,
    /// `[d_model]` window into `ple_proj` capture.
    pub ple_proj_window: Buffer,
    /// `[seq · d_model]` — per-position snapshot of `self.hidden`
    /// AFTER the layer stack and BEFORE the final rmsnorm. Captured
    /// by the single-forward PerPosition orchestrator during the
    /// one-shot forward sweep, then read back per active completion
    /// position to drive the final-norm + output projection chain.
    pub seq_pre_final_norm: Buffer,
    /// Configured max sequence length the scratch is sized for.
    pub max_seq_len: u32,
}

impl TrainingScratch {
    /// Allocate all scratch buffers for a `TrainingSession`.
    ///
    /// Sized off the model's `Gemma4Config` and the configured
    /// `max_seq_len`. The scratch is reused across training steps;
    /// the buffer contents are overwritten per step.
    pub fn new(ctx: &Arc<WgpuCtx>, cfg: &Gemma4Config, max_seq_len: u32) -> Self {
        let device = &ctx.device;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        let make = |label: &'static str, elems: u64| -> Buffer {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (elems * 4).max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let d_model_e = cfg.d_model as u64;
        let seq_e = max_seq_len as u64;
        let vocab_e = cfg.vocab_size as u64;
        let n_heads_e = cfg.n_heads as u64;
        let head_dim_max_e = cfg.head_dim_global.max(cfg.head_dim_swa) as u64;
        let n_kv_max_e = cfg.n_kv_heads_global.max(cfg.n_kv_heads_swa) as u64;
        let ffn_inter_max_e = (0..cfg.n_layers).map(|i| cfg.ffn(i)).max().unwrap_or(0) as u64;
        let ple_dim_e = if cfg.has_ple() { cfg.ple_dim as u64 } else { 0 };

        let d_logits = make("scratch.d_logits", vocab_e);
        let loss = make("scratch.loss", 1);
        let d_hidden_final = make("scratch.d_hidden_final", d_model_e);
        let d_hidden = make("scratch.d_hidden", d_model_e);
        let d_hidden_tmp = make("scratch.d_hidden_tmp", d_model_e);
        let d_hidden_tmp2 = make("scratch.d_hidden_tmp2", d_model_e);

        let layers = (0..cfg.n_layers)
            .map(|li| {
                let n_kv = cfg.n_kv_heads(li) as u64;
                let head_dim = cfg.head_dim(li) as u64;
                let n_heads = cfg.n_heads as u64;
                let ffn_inter = cfg.ffn(li) as u64;

                // All captures are seq-sized: forward writes at
                // offset `pos·per_position_size` per position. The
                // final-position backward chain pre-copies the
                // `pos`-slice into single-position windows on
                // `TrainingScratch`. The per-history K/V LoRA loop
                // and the single-forward PerPosition variant
                // re-copy *other* positions into the same windows.
                let ple_d = if ple_dim_e > 0 { d_model_e } else { 0 };
                LayerActivations {
                    hidden_in: make("layer.hidden_in_seq", d_model_e * seq_e),
                    norm_x_attn: make("layer.norm_x_attn_seq", d_model_e * seq_e),
                    q_pre_norm: make("layer.q_pre_norm_seq", n_heads * head_dim * seq_e),
                    q_post_rope: make("layer.q_post_rope_seq", n_heads * head_dim * seq_e),
                    k_pre_norm: make("layer.k_pre_norm_seq", n_kv * head_dim * seq_e),
                    v_pre_norm: make("layer.v_pre_norm_seq", n_kv * head_dim * seq_e),
                    attn_out: make("layer.attn_out_seq", n_heads * head_dim * seq_e),
                    attn_proj: make("layer.attn_proj_seq", d_model_e * seq_e),
                    pre_ffn_rms: make("layer.pre_ffn_rms_seq", d_model_e * seq_e),
                    norm_x_ffn: make("layer.norm_x_ffn_seq", d_model_e * seq_e),
                    ffn_gate: make("layer.ffn_gate_seq", ffn_inter * seq_e),
                    ffn_up: make("layer.ffn_up_seq", ffn_inter * seq_e),
                    ffn_act: make("layer.ffn_act_seq", ffn_inter * seq_e),
                    ffn_out: make("layer.ffn_out_seq", d_model_e * seq_e),
                    ple_state: make("layer.ple_state_seq", ple_dim_e * seq_e),
                    ple_act: make("layer.ple_act_seq", ple_dim_e * seq_e),
                    ple_proj: make("layer.ple_proj_seq", ple_d * seq_e),
                }
            })
            .collect();

        // Max-shape probs/d_scores: at most `n_heads * max_history_len`.
        // history_len at the final position equals `seq_len`.
        let attn_probs = make("scratch.attn_probs", n_heads_e * seq_e);
        let attn_d_scores = make("scratch.attn_d_scores", n_heads_e * seq_e);

        let d_q = make("scratch.d_q", n_heads_e * head_dim_max_e);
        let d_k_hist = make("scratch.d_k_hist", seq_e * n_kv_max_e * head_dim_max_e);
        let d_v_hist = make("scratch.d_v_hist", seq_e * n_kv_max_e * head_dim_max_e);
        let d_q_pre_rope = make("scratch.d_q_pre_rope", n_heads_e * head_dim_max_e);
        let d_k_pre_rope = make("scratch.d_k_pre_rope", n_kv_max_e * head_dim_max_e);
        let d_q_pre_norm = make("scratch.d_q_pre_norm", n_heads_e * head_dim_max_e);
        let d_k_pre_norm = make("scratch.d_k_pre_norm", n_kv_max_e * head_dim_max_e);
        let d_v_pre_norm = make("scratch.d_v_pre_norm", n_kv_max_e * head_dim_max_e);
        let d_ffn_a = make("scratch.d_ffn_a", ffn_inter_max_e);
        let d_ffn_b = make("scratch.d_ffn_b", ffn_inter_max_e);
        let d_ffn_c = make("scratch.d_ffn_c", ffn_inter_max_e);
        // PLE backward scratch — `ple_dim` sized; 4-byte stub if !has_ple
        // (so storage bindings stay valid even though the backward block
        // doesn't fire).
        let d_ple_state = make("scratch.d_ple_state", ple_dim_e);
        let d_ple_act = make("scratch.d_ple_act", ple_dim_e);
        let d_ple_up_discard = make("scratch.d_ple_up_discard", ple_dim_e);
        let ple_per_layer_tmp = make("scratch.ple_per_layer_tmp", ple_dim_e);
        let norm_x_attn_window = make("scratch.norm_x_attn_window", d_model_e);
        let k_pre_norm_window = make("scratch.k_pre_norm_window", n_kv_max_e * head_dim_max_e);
        let v_pre_norm_window = make("scratch.v_pre_norm_window", n_kv_max_e * head_dim_max_e);
        let hidden_in_window = make("scratch.hidden_in_window", d_model_e);
        let q_pre_norm_window = make("scratch.q_pre_norm_window", n_heads_e * head_dim_max_e);
        let q_post_rope_window = make("scratch.q_post_rope_window", n_heads_e * head_dim_max_e);
        let attn_out_window = make("scratch.attn_out_window", n_heads_e * head_dim_max_e);
        let attn_proj_window = make("scratch.attn_proj_window", d_model_e);
        let pre_ffn_rms_window = make("scratch.pre_ffn_rms_window", d_model_e);
        let norm_x_ffn_window = make("scratch.norm_x_ffn_window", d_model_e);
        let ffn_gate_window = make("scratch.ffn_gate_window", ffn_inter_max_e);
        let ffn_up_window = make("scratch.ffn_up_window", ffn_inter_max_e);
        let ffn_act_window = make("scratch.ffn_act_window", ffn_inter_max_e);
        let ffn_out_window = make("scratch.ffn_out_window", d_model_e);
        let ple_state_window = make("scratch.ple_state_window", ple_dim_e);
        let ple_act_window = make("scratch.ple_act_window", ple_dim_e);
        let ple_proj_window = make(
            "scratch.ple_proj_window",
            if ple_dim_e > 0 { d_model_e } else { 0 },
        );
        let seq_pre_final_norm = make("scratch.seq_pre_final_norm", d_model_e * seq_e);

        Self {
            d_logits,
            loss,
            d_hidden_final,
            layers,
            d_hidden,
            d_hidden_tmp,
            d_hidden_tmp2,
            attn_probs,
            attn_d_scores,
            d_q,
            d_k_hist,
            d_v_hist,
            d_q_pre_rope,
            d_k_pre_rope,
            d_q_pre_norm,
            d_k_pre_norm,
            d_v_pre_norm,
            d_ffn_a,
            d_ffn_b,
            d_ffn_c,
            d_ple_state,
            d_ple_act,
            d_ple_up_discard,
            ple_per_layer_tmp,
            norm_x_attn_window,
            k_pre_norm_window,
            v_pre_norm_window,
            hidden_in_window,
            q_pre_norm_window,
            q_post_rope_window,
            attn_out_window,
            attn_proj_window,
            pre_ffn_rms_window,
            norm_x_ffn_window,
            ffn_gate_window,
            ffn_up_window,
            ffn_act_window,
            ffn_out_window,
            ple_state_window,
            ple_act_window,
            ple_proj_window,
            seq_pre_final_norm,
            max_seq_len,
        }
    }

    /// Total byte size of all scratch buffers — useful for logging.
    pub fn byte_size(&self) -> u64 {
        let mut total = self.d_logits.size()
            + self.loss.size()
            + self.d_hidden_final.size()
            + self.d_hidden.size()
            + self.d_hidden_tmp.size()
            + self.d_hidden_tmp2.size()
            + self.attn_probs.size()
            + self.attn_d_scores.size()
            + self.d_q.size()
            + self.d_k_hist.size()
            + self.d_v_hist.size()
            + self.d_q_pre_rope.size()
            + self.d_k_pre_rope.size()
            + self.d_q_pre_norm.size()
            + self.d_k_pre_norm.size()
            + self.d_v_pre_norm.size()
            + self.d_ffn_a.size()
            + self.d_ffn_b.size()
            + self.d_ffn_c.size();
        for l in &self.layers {
            total += l.hidden_in.size()
                + l.norm_x_attn.size()
                + l.q_pre_norm.size()
                + l.q_post_rope.size()
                + l.k_pre_norm.size()
                + l.v_pre_norm.size()
                + l.attn_out.size()
                + l.attn_proj.size()
                + l.pre_ffn_rms.size()
                + l.norm_x_ffn.size()
                + l.ffn_gate.size()
                + l.ffn_up.size()
                + l.ffn_act.size()
                + l.ffn_out.size();
        }
        total
    }
}
