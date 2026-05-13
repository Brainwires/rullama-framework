//! `TrainingScratch` — per-step GPU scratch buffers for the backward pass.
//!
//! Owns the loss / d_logits buffers shared by every step plus
//! per-layer **activation save** buffers that the forward pass writes
//! to and the backward pass consumes. Allocations are sized off
//! `Gemma4Config` + `max_seq_len` so the lifetime of a scratch is the
//! lifetime of a `TrainingSession` (no per-step alloc).
//!
//! ### Per-layer activations (what the backward needs)
//!
//! Per layer, the M0 backward walker consumes:
//!
//! | name            | shape (seq × …)                                 | needed by                                  |
//! | --------------- | ----------------------------------------------- | ------------------------------------------ |
//! | `hidden_in`     | `[seq, d_model]`                                | post-RMSNorm-attn residual backward        |
//! | `pre_attn_rms`  | `[seq, d_model]`                                | rmsnorm_backward (input to attn_norm)      |
//! | `norm_x_attn`   | `[seq, d_model]`                                | matmul_q4_k_backward_input for q/k/v       |
//! | `q_pre_rope`    | `[seq, n_heads · head_dim]`                     | rope_backward (q)                          |
//! | `k_pre_rope`    | `[seq, n_kv · head_dim]`                        | rope_backward (k)                          |
//! | `attn_probs`    | `[seq, n_heads, history_len]`                   | attention_backward (softmax probs)         |
//! | `attn_out`      | `[seq, n_heads · head_dim]`                     | matmul_q4_k_backward_input (o_proj input)  |
//! | `pre_ffn_rms`   | `[seq, d_model]`                                | rmsnorm_backward (input to mlp_norm)       |
//! | `norm_x_ffn`    | `[seq, d_model]`                                | matmul_q4_k_backward_input for gate/up     |
//! | `ffn_gate`      | `[seq, ffn_inter]`                              | geglu_backward                             |
//! | `ffn_up`        | `[seq, ffn_inter]`                              | geglu_backward                             |
//! | `ffn_act`       | `[seq, ffn_inter]`                              | matmul_q4_k_backward_input for ffn_down    |
//!
//! Sizes for Gemma 4 e2b at `seq=128`:
//!   d_model=1536, n_heads=8, head_dim=256, ffn_inter≈8192, history=128.
//!   Per layer total ≈ 12 × {1.5K..1MB} ≈ 1.3 MB. 26 layers ≈ 34 MB.
//!
//! Activations live on the GPU as `wgpu::Buffer`s with
//! `STORAGE | COPY_DST | COPY_SRC`. The forward writes to them via a
//! `Forward::with_activation_capture(...)` mode that task #14b will
//! plumb through `forward_chained.rs::encode_layer`. The backward
//! reads them through the dispatchers added in tasks #9–#12.

use std::sync::Arc;

use rullama::backend::WgpuCtx;
use rullama::model::config::Gemma4Config;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

/// Per-layer activation buffer set. One per layer; addressed by layer index.
pub struct LayerActivations {
    pub hidden_in:    Buffer,
    pub pre_attn_rms: Buffer,
    pub norm_x_attn:  Buffer,
    pub q_pre_rope:   Buffer,
    pub k_pre_rope:   Buffer,
    pub attn_probs:   Buffer,
    pub attn_out:     Buffer,
    pub pre_ffn_rms:  Buffer,
    pub norm_x_ffn:   Buffer,
    pub ffn_gate:     Buffer,
    pub ffn_up:       Buffer,
    pub ffn_act:      Buffer,
}

/// Top-level scratch for a training step.
///
/// Layout invariants:
/// - `max_seq_len` is the upper bound on training-time sequence length;
///   per-step calls may use ≤ that. Buffers are sized for the max.
/// - `d_logits` / `loss` are dispatched once per step (NextToken mode);
///   PerPosition mode (M1) extends `d_logits` to `[seq, vocab]` and
///   `loss` to `[seq]`.
pub struct TrainingScratch {
    /// `d_logits[vocab]` from cross_entropy_backward.
    pub d_logits: Buffer,
    /// Scalar loss readback.
    pub loss: Buffer,
    /// `d_hidden_final[d_model]` — gradient at the final-position hidden
    /// state after the output projection's backward.
    pub d_hidden_final: Buffer,
    /// Per-layer activation captures.
    pub layers: Vec<LayerActivations>,
    /// Staging buffer for `d_scores` in attention backward (pass-1 output,
    /// pass-2 input). Sized `[n_heads, history_len]`.
    pub attn_d_scores: Buffer,
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
        let seq_e     = max_seq_len as u64;
        let vocab_e   = cfg.vocab_size as u64;

        let d_logits       = make("scratch.d_logits", vocab_e);
        let loss           = make("scratch.loss", 1);
        let d_hidden_final = make("scratch.d_hidden_final", d_model_e);

        let layers = (0..cfg.n_layers)
            .map(|li| {
                let n_kv = cfg.n_kv_heads(li) as u64;
                let head_dim = match cfg.layer_kinds[li as usize] {
                    rullama::model::config::LayerKind::SlidingWindow => cfg.head_dim_swa as u64,
                    rullama::model::config::LayerKind::Global => cfg.head_dim_global as u64,
                };
                let n_heads = cfg.n_heads as u64;
                let ffn_inter = cfg.ffn_inter[li as usize] as u64;
                let history = seq_e; // backward at the final position sees the full sequence

                LayerActivations {
                    hidden_in:    make("layer.hidden_in",    seq_e * d_model_e),
                    pre_attn_rms: make("layer.pre_attn_rms", seq_e * d_model_e),
                    norm_x_attn:  make("layer.norm_x_attn",  seq_e * d_model_e),
                    q_pre_rope:   make("layer.q_pre_rope",   seq_e * n_heads * head_dim),
                    k_pre_rope:   make("layer.k_pre_rope",   seq_e * n_kv * head_dim),
                    attn_probs:   make("layer.attn_probs",   seq_e * n_heads * history),
                    attn_out:     make("layer.attn_out",     seq_e * n_heads * head_dim),
                    pre_ffn_rms:  make("layer.pre_ffn_rms",  seq_e * d_model_e),
                    norm_x_ffn:   make("layer.norm_x_ffn",   seq_e * d_model_e),
                    ffn_gate:     make("layer.ffn_gate",     seq_e * ffn_inter),
                    ffn_up:       make("layer.ffn_up",       seq_e * ffn_inter),
                    ffn_act:      make("layer.ffn_act",      seq_e * ffn_inter),
                }
            })
            .collect();

        let attn_d_scores = make("scratch.attn_d_scores", cfg.n_heads as u64 * seq_e);

        Self {
            d_logits,
            loss,
            d_hidden_final,
            layers,
            attn_d_scores,
            max_seq_len,
        }
    }

    /// Total byte size of all scratch buffers — useful for logging.
    pub fn byte_size(&self) -> u64 {
        let mut total = self.d_logits.size() + self.loss.size() + self.d_hidden_final.size() + self.attn_d_scores.size();
        for l in &self.layers {
            total += l.hidden_in.size()
                + l.pre_attn_rms.size()
                + l.norm_x_attn.size()
                + l.q_pre_rope.size()
                + l.k_pre_rope.size()
                + l.attn_probs.size()
                + l.attn_out.size()
                + l.pre_ffn_rms.size()
                + l.norm_x_ffn.size()
                + l.ffn_gate.size()
                + l.ffn_up.size()
                + l.ffn_act.size();
        }
        total
    }
}
