//! ROME (Rank-One Model Editing) infrastructure for surgical fact
//! substitution. Phase 1.1 of the ROME/MEMIT plan
//! (`.claude/plans/write-this-up-formally-delegated-sun.md`).
//!
//! This module owns the GPU buffer infrastructure ROME needs that
//! sits on top of the existing inference path:
//!
//!   • `RomeCapture` — per-layer single-position activation buffers
//!     sized to hold what `Forward::step_capture` writes. Differs
//!     from `brainwires_lora::scratch::LayerActivations` only in
//!     that ROME doesn't need the sequence-dimensional storage that
//!     training uses for the PerPosition backward sweep.
//!
//!   • `RomeCapture::as_captures` — produces a `Vec<LayerCaptureBuffers>`
//!     view that can be passed directly to `Forward::step_capture`.
//!
//!   • `RomeCapture::read_norm_x_ffn` — async readback of the MLP-input
//!     activation (k* in the ROME formulation) at a chosen layer.
//!
//! The full ROME algorithm (k* extraction + v* gradient descent + rank-1
//! safetensors serialization) is implemented in `crate::api::Model`
//! using these primitives. See the plan file for the math.

use std::sync::Arc;

use futures_channel::oneshot;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};
use crate::model::config::Gemma4Config;
use crate::reference::forward_chained::{
    BackwardScratchView, LayerCaptureBuffers, LayerLoraGrads, LayerLoraSlots,
};

/// Per-layer single-position capture buffers. Mirrors the shape
/// requirements of `LayerCaptureBuffers` exactly so the existing
/// `Forward::step_capture` path can write into them.
///
/// Sized for ONE token position (the subject's last token). ROME
/// doesn't need sequence-shaped captures the way per-position
/// training does — we only care about the last-position MLP-input
/// vector for k* extraction and (later) for v* gradient descent.
struct RomeLayerBuffers {
    hidden_in: Buffer,
    norm_x_attn: Buffer,
    q_pre_norm: Buffer,
    q_post_rope: Buffer,
    k_pre_norm: Buffer,
    v_pre_norm: Buffer,
    attn_out: Buffer,
    attn_proj: Buffer,
    pre_ffn_rms: Buffer,
    norm_x_ffn: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    ffn_act: Buffer,
    ffn_out: Buffer,
    ple_state: Buffer,
    ple_act: Buffer,
    ple_proj: Buffer,
}

/// Collection of per-layer capture buffers for a ROME forward pass.
/// Allocate once with [`RomeCapture::new`], then convert to
/// `&[LayerCaptureBuffers]` via [`RomeCapture::as_captures`] each
/// time the caller invokes `Forward::step_capture`.
///
/// Buffers are sized to `seq_len × per_position`. The Forward path
/// (`step_capture`) writes activations at per-position offsets, so a
/// single `[d_model]` buffer overruns. `seq_len` must be ≥ the
/// longest sequence the caller will forward through the captured
/// path.
pub struct RomeCapture {
    ctx: Arc<WgpuCtx>,
    /// `d_model` (= width of the residual stream). Reserved for
    /// Phase 1.2 — v* lives in this dimension so the future
    /// `read_ffn_out` (or backward-gradient readback) will need
    /// `d_model`-sized staging buffers.
    #[allow(dead_code)]
    cfg_d_model: u32,
    /// Per-layer `ffn_inter` (= d_ffn) for sized readback of ffn_act.
    /// Differs across layers in Gemma 4 e2b.
    cfg_ffn_inter: Vec<u32>,
    seq_len: u32,
    layers: Vec<RomeLayerBuffers>,
}

impl RomeCapture {
    /// Allocate per-layer capture buffers sized for sequences up to
    /// `seq_len` tokens. Each buffer is
    /// `STORAGE | COPY_DST | COPY_SRC` so the kernels can write into
    /// it and we can `copy_buffer_to_buffer` the contents to a
    /// `MAP_READ` staging buffer for CPU readback.
    ///
    /// Memory cost is roughly `seq_len × (d_model + ffn_inter + ...)
    /// × 4 × n_layers` bytes. For Gemma 4 e2b at seq=64 that's about
    /// 30-50 MB total — small relative to the model.
    pub fn new(ctx: &Arc<WgpuCtx>, cfg: &Gemma4Config, seq_len: u32) -> Self {
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

        let d_model = cfg.d_model as u64;
        let seq = seq_len as u64;
        let n_heads = cfg.n_heads as u64;
        let head_dim_max = cfg.head_dim_global.max(cfg.head_dim_swa) as u64;
        let n_kv_max = cfg.n_kv_heads_global.max(cfg.n_kv_heads_swa) as u64;
        let ple_dim = if cfg.has_ple() { cfg.ple_dim as u64 } else { 0 };
        let ple_d = if ple_dim > 0 { d_model } else { 0 };

        let layers: Vec<RomeLayerBuffers> = (0..cfg.n_layers)
            .map(|li| {
                let ffn_inter = cfg.ffn(li) as u64;
                RomeLayerBuffers {
                    hidden_in: make("rome.hidden_in", d_model * seq),
                    norm_x_attn: make("rome.norm_x_attn", d_model * seq),
                    q_pre_norm: make("rome.q_pre_norm", n_heads * head_dim_max * seq),
                    q_post_rope: make("rome.q_post_rope", n_heads * head_dim_max * seq),
                    k_pre_norm: make("rome.k_pre_norm", n_kv_max * head_dim_max * seq),
                    v_pre_norm: make("rome.v_pre_norm", n_kv_max * head_dim_max * seq),
                    attn_out: make("rome.attn_out", n_heads * head_dim_max * seq),
                    attn_proj: make("rome.attn_proj", d_model * seq),
                    pre_ffn_rms: make("rome.pre_ffn_rms", d_model * seq),
                    norm_x_ffn: make("rome.norm_x_ffn", d_model * seq),
                    ffn_gate: make("rome.ffn_gate", ffn_inter * seq),
                    ffn_up: make("rome.ffn_up", ffn_inter * seq),
                    ffn_act: make("rome.ffn_act", ffn_inter * seq),
                    ffn_out: make("rome.ffn_out", d_model * seq),
                    ple_state: make("rome.ple_state", ple_dim * seq),
                    ple_act: make("rome.ple_act", ple_dim * seq),
                    ple_proj: make("rome.ple_proj", ple_d * seq),
                }
            })
            .collect();

        let cfg_ffn_inter: Vec<u32> = (0..cfg.n_layers).map(|i| cfg.ffn(i)).collect();
        Self {
            ctx: Arc::clone(ctx),
            cfg_d_model: cfg.d_model,
            cfg_ffn_inter,
            seq_len,
            layers,
        }
    }

    /// View suitable for `Forward::step_capture(&captures, ...)`. The
    /// returned Vec borrows from `self`; valid as long as the caller
    /// holds it (the `step_capture` call itself is short-lived).
    pub fn as_captures(&self) -> Vec<LayerCaptureBuffers<'_>> {
        self.layers
            .iter()
            .map(|l| LayerCaptureBuffers {
                hidden_in: &l.hidden_in,
                norm_x_attn: &l.norm_x_attn,
                q_pre_norm: &l.q_pre_norm,
                q_post_rope: &l.q_post_rope,
                k_pre_norm: &l.k_pre_norm,
                v_pre_norm: &l.v_pre_norm,
                attn_out: &l.attn_out,
                attn_proj: &l.attn_proj,
                pre_ffn_rms: &l.pre_ffn_rms,
                norm_x_ffn: &l.norm_x_ffn,
                ffn_gate: &l.ffn_gate,
                ffn_up: &l.ffn_up,
                ffn_act: &l.ffn_act,
                ffn_out: &l.ffn_out,
                ple_state: &l.ple_state,
                ple_act: &l.ple_act,
                ple_proj: &l.ple_proj,
            })
            .collect()
    }

    /// Read back `ffn_act[target_layer]` at `position` as `[d_ffn]`
    /// f32. **This is ROME's k\*** — the post-GEGLU activation that
    /// is the INPUT to `ffn_down`. Required for a rank-1 LoRA-style
    /// update on `ffn_down.weight` (shape `[d_model × d_ffn]`) where
    /// the rank-1 factor A must be `[d_ffn]`-shaped to compose.
    ///
    /// (We previously read `norm_x_ffn` of shape `[d_model]` — wrong
    /// shape for the rank-1 update on `ffn_down`. The reference
    /// ROME implementation also extracts post-activation, not
    /// pre-MLP-input.)
    ///
    /// Capture buffer is seq-shaped (`[seq_len × ffn_inter]`);
    /// extract slice at byte offset `position × ffn_inter × 4`.
    pub async fn read_ffn_act(&self, target_layer: u32, position: u32) -> Result<Vec<f32>> {
        let layer = target_layer as usize;
        if layer >= self.layers.len() {
            return Err(RullamaError::Inference(format!(
                "read_ffn_act: layer {layer} out of range (have {})",
                self.layers.len()
            )));
        }
        if position >= self.seq_len {
            return Err(RullamaError::Inference(format!(
                "read_ffn_act: position {position} >= seq_len {}",
                self.seq_len
            )));
        }
        let ffn_inter = self.cfg_ffn_inter[layer] as u64;
        let src = &self.layers[layer].ffn_act;
        let elt_bytes = ffn_inter * 4;
        let src_offset = (position as u64) * elt_bytes;
        let bytes = elt_bytes;
        let staging = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rome.staging.ffn_act"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.read_ffn_act"),
            });
        enc.copy_buffer_to_buffer(src, src_offset, &staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
        receiver
            .await
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }

    /// Read back `ffn_out[target_layer]` at `position` as `[d_model]`
    /// f32 — the output of `ffn_down` (post-projection, pre-residual).
    /// Used by ROME Phase 2.b's iterative v\* loop to capture
    /// `target_init = ffn_out[L, subject_last_pos]` once at step 0;
    /// the final `v_star = target_init + δ`.
    pub async fn read_ffn_out(&self, target_layer: u32, position: u32) -> Result<Vec<f32>> {
        let layer = target_layer as usize;
        if layer >= self.layers.len() {
            return Err(RullamaError::Inference(format!(
                "read_ffn_out: layer {layer} out of range (have {})",
                self.layers.len()
            )));
        }
        if position >= self.seq_len {
            return Err(RullamaError::Inference(format!(
                "read_ffn_out: position {position} >= seq_len {}",
                self.seq_len
            )));
        }
        let d_model = self.cfg_d_model as u64;
        let src = &self.layers[layer].ffn_out;
        let elt_bytes = d_model * 4;
        let src_offset = (position as u64) * elt_bytes;
        let staging = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rome.staging.ffn_out"),
            size: elt_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.read_ffn_out"),
            });
        enc.copy_buffer_to_buffer(src, src_offset, &staging, 0, elt_bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
        receiver
            .await
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }
}

/// GPU-resident state for ROME Phase 2.b's iterative v\* optimization.
///
/// Mirrors the four buffers kmeng01/rome's `compute_v.py` keeps alive
/// across Adam steps:
///   * `delta` — the residual perturbation being optimized, shape
///     `[d_model]`
///   * `delta_grad` — accumulator for `∂loss/∂δ`
///   * `adam_m`, `adam_v` — first/second moment estimates for AdamW
///
/// Allocated once per edit; reused across all 25 Adam iterations.
pub struct RomeIterativeState {
    pub delta: wgpu::Buffer,
    pub adam_m: wgpu::Buffer,
    pub adam_v: wgpu::Buffer,
    d_model: u32,
    ctx: Arc<WgpuCtx>,
}

impl RomeIterativeState {
    /// Allocate and zero all four buffers.
    pub fn new(ctx: &Arc<WgpuCtx>, d_model: u32) -> Self {
        let device = &ctx.device;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let alloc = |label: &'static str| -> wgpu::Buffer {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (d_model as u64) * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let delta = alloc("rome.delta");
        let adam_m = alloc("rome.adam_m");
        let adam_v = alloc("rome.adam_v");
        // Zero-init via queue write — wgpu Buffers initialize to zero
        // when mapped_at_creation=false but the write makes the intent
        // explicit and survives any future driver oddities.
        let zeros = vec![0.0f32; d_model as usize];
        ctx.queue
            .write_buffer(&delta, 0, bytemuck::cast_slice(&zeros));
        ctx.queue
            .write_buffer(&adam_m, 0, bytemuck::cast_slice(&zeros));
        ctx.queue
            .write_buffer(&adam_v, 0, bytemuck::cast_slice(&zeros));
        Self {
            delta,
            adam_m,
            adam_v,
            d_model,
            ctx: Arc::clone(ctx),
        }
    }

    /// Read δ back to CPU. Cheap (one `[d_model]` map).
    pub async fn read_delta(&self) -> Result<Vec<f32>> {
        let bytes = (self.d_model as u64) * 4;
        let staging = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rome.staging.delta"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.read_delta"),
            });
        enc.copy_buffer_to_buffer(&self.delta, 0, &staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
        receiver
            .await
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }

    /// Overwrite δ from CPU values (used to apply norm-clamp after an
    /// Adam step, or to reset δ at the top of a new edit).
    pub fn write_delta(&self, vals: &[f32]) -> Result<()> {
        if vals.len() != self.d_model as usize {
            return Err(RullamaError::Inference(format!(
                "write_delta: got {} floats, expected d_model = {}",
                vals.len(),
                self.d_model
            )));
        }
        self.ctx
            .queue
            .write_buffer(&self.delta, 0, bytemuck::cast_slice(vals));
        Ok(())
    }
}

/// Owned GPU buffers for the backward path's per-step scratch. View
/// suitable for `Forward::backward_step` is produced by
/// [`RomeBackwardScratch::view`]. This is the rullama-side mirror of
/// `brainwires_lora::scratch::TrainingScratch`'s scratch fields — we
/// duplicate the allocation here rather than depend on
/// `brainwires-lora` (cycle: brainwires-lora already depends on
/// rullama).
pub struct RomeBackwardScratch {
    d_logits: wgpu::Buffer,
    loss: wgpu::Buffer,
    d_hidden_final: wgpu::Buffer,
    d_hidden: wgpu::Buffer,
    d_hidden_tmp: wgpu::Buffer,
    d_hidden_tmp2: wgpu::Buffer,
    attn_probs: wgpu::Buffer,
    attn_d_scores: wgpu::Buffer,
    d_attn_out: wgpu::Buffer,
    d_q: wgpu::Buffer,
    d_k_hist: wgpu::Buffer,
    d_v_hist: wgpu::Buffer,
    d_q_pre_rope: wgpu::Buffer,
    d_k_pre_rope: wgpu::Buffer,
    d_q_pre_norm: wgpu::Buffer,
    d_k_pre_norm: wgpu::Buffer,
    d_v_pre_norm: wgpu::Buffer,
    d_ffn_a: wgpu::Buffer,
    d_ffn_b: wgpu::Buffer,
    d_ffn_c: wgpu::Buffer,
    d_ple_state: wgpu::Buffer,
    d_ple_act: wgpu::Buffer,
    d_ple_up_discard: wgpu::Buffer,
    ple_per_layer_tmp: wgpu::Buffer,
    norm_x_attn_window: wgpu::Buffer,
    k_pre_norm_window: wgpu::Buffer,
    v_pre_norm_window: wgpu::Buffer,
    hidden_in_window: wgpu::Buffer,
    q_pre_norm_window: wgpu::Buffer,
    q_post_rope_window: wgpu::Buffer,
    attn_out_window: wgpu::Buffer,
    attn_proj_window: wgpu::Buffer,
    pre_ffn_rms_window: wgpu::Buffer,
    norm_x_ffn_window: wgpu::Buffer,
    ffn_gate_window: wgpu::Buffer,
    ffn_up_window: wgpu::Buffer,
    ffn_act_window: wgpu::Buffer,
    ffn_out_window: wgpu::Buffer,
    ple_state_window: wgpu::Buffer,
    ple_act_window: wgpu::Buffer,
    ple_proj_window: wgpu::Buffer,
    cfg_d_model: u32,
}

impl RomeBackwardScratch {
    /// Allocate scratch buffers sized to the max per-layer shapes
    /// across the model. Caller picks `seq_len` (= subject prompt
    /// length); attn_probs and the K/V history buffers are sized for
    /// `n_heads × seq_len` / `seq_len × n_kv × head_dim` accordingly.
    pub fn new(ctx: &WgpuCtx, cfg: &Gemma4Config, seq_len: u32) -> Self {
        let device = &ctx.device;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        let make = |label: &'static str, elems: u64| -> wgpu::Buffer {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (elems * 4).max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let d_model_e = cfg.d_model as u64;
        let seq_e = seq_len as u64;
        let vocab_e = cfg.vocab_size as u64;
        let n_heads_e = cfg.n_heads as u64;
        let head_dim_max_e = cfg.head_dim_global.max(cfg.head_dim_swa) as u64;
        let n_kv_max_e = cfg.n_kv_heads_global.max(cfg.n_kv_heads_swa) as u64;
        let ffn_inter_max_e = (0..cfg.n_layers).map(|i| cfg.ffn(i)).max().unwrap_or(0) as u64;
        let ple_dim_e = if cfg.has_ple() { cfg.ple_dim as u64 } else { 0 };

        Self {
            d_logits: make("scratch.d_logits", vocab_e),
            loss: make("scratch.loss", 1),
            d_hidden_final: make("scratch.d_hidden_final", d_model_e),
            d_hidden: make("scratch.d_hidden", d_model_e),
            d_hidden_tmp: make("scratch.d_hidden_tmp", d_model_e),
            d_hidden_tmp2: make("scratch.d_hidden_tmp2", d_model_e),
            attn_probs: make("scratch.attn_probs", n_heads_e * seq_e),
            attn_d_scores: make("scratch.attn_d_scores", n_heads_e * seq_e),
            d_attn_out: make("scratch.d_attn_out", n_heads_e * head_dim_max_e),
            d_q: make("scratch.d_q", n_heads_e * head_dim_max_e),
            d_k_hist: make("scratch.d_k_hist", seq_e * n_kv_max_e * head_dim_max_e),
            d_v_hist: make("scratch.d_v_hist", seq_e * n_kv_max_e * head_dim_max_e),
            d_q_pre_rope: make("scratch.d_q_pre_rope", n_heads_e * head_dim_max_e),
            d_k_pre_rope: make("scratch.d_k_pre_rope", n_kv_max_e * head_dim_max_e),
            d_q_pre_norm: make("scratch.d_q_pre_norm", n_heads_e * head_dim_max_e),
            d_k_pre_norm: make("scratch.d_k_pre_norm", n_kv_max_e * head_dim_max_e),
            d_v_pre_norm: make("scratch.d_v_pre_norm", n_kv_max_e * head_dim_max_e),
            d_ffn_a: make("scratch.d_ffn_a", ffn_inter_max_e),
            d_ffn_b: make("scratch.d_ffn_b", ffn_inter_max_e),
            d_ffn_c: make("scratch.d_ffn_c", ffn_inter_max_e),
            d_ple_state: make("scratch.d_ple_state", ple_dim_e),
            d_ple_act: make("scratch.d_ple_act", ple_dim_e),
            d_ple_up_discard: make("scratch.d_ple_up_discard", ple_dim_e),
            ple_per_layer_tmp: make("scratch.ple_per_layer_tmp", ple_dim_e),
            norm_x_attn_window: make("scratch.norm_x_attn_window", d_model_e),
            k_pre_norm_window: make("scratch.k_pre_norm_window", n_kv_max_e * head_dim_max_e),
            v_pre_norm_window: make("scratch.v_pre_norm_window", n_kv_max_e * head_dim_max_e),
            hidden_in_window: make("scratch.hidden_in_window", d_model_e),
            q_pre_norm_window: make("scratch.q_pre_norm_window", n_heads_e * head_dim_max_e),
            q_post_rope_window: make("scratch.q_post_rope_window", n_heads_e * head_dim_max_e),
            attn_out_window: make("scratch.attn_out_window", n_heads_e * head_dim_max_e),
            attn_proj_window: make("scratch.attn_proj_window", d_model_e),
            pre_ffn_rms_window: make("scratch.pre_ffn_rms_window", d_model_e),
            norm_x_ffn_window: make("scratch.norm_x_ffn_window", d_model_e),
            ffn_gate_window: make("scratch.ffn_gate_window", ffn_inter_max_e),
            ffn_up_window: make("scratch.ffn_up_window", ffn_inter_max_e),
            ffn_act_window: make("scratch.ffn_act_window", ffn_inter_max_e),
            ffn_out_window: make("scratch.ffn_out_window", d_model_e),
            ple_state_window: make("scratch.ple_state_window", ple_dim_e),
            ple_act_window: make("scratch.ple_act_window", ple_dim_e),
            ple_proj_window: make(
                "scratch.ple_proj_window",
                if ple_dim_e > 0 { d_model_e } else { 0 },
            ),
            cfg_d_model: cfg.d_model,
        }
    }

    /// View suitable for `Forward::backward_step(.., scratch, ..)`.
    pub fn view(&self) -> BackwardScratchView<'_> {
        BackwardScratchView {
            d_logits: &self.d_logits,
            loss: &self.loss,
            d_hidden_final: &self.d_hidden_final,
            d_hidden: &self.d_hidden,
            d_hidden_tmp: &self.d_hidden_tmp,
            d_hidden_tmp2: &self.d_hidden_tmp2,
            attn_probs: &self.attn_probs,
            attn_d_scores: &self.attn_d_scores,
            d_attn_out: &self.d_attn_out,
            d_q: &self.d_q,
            d_k_hist: &self.d_k_hist,
            d_v_hist: &self.d_v_hist,
            d_q_pre_rope: &self.d_q_pre_rope,
            d_k_pre_rope: &self.d_k_pre_rope,
            d_q_pre_norm: &self.d_q_pre_norm,
            d_k_pre_norm: &self.d_k_pre_norm,
            d_v_pre_norm: &self.d_v_pre_norm,
            d_ffn_a: &self.d_ffn_a,
            d_ffn_b: &self.d_ffn_b,
            d_ffn_c: &self.d_ffn_c,
            d_ple_state: &self.d_ple_state,
            d_ple_act: &self.d_ple_act,
            d_ple_up_discard: &self.d_ple_up_discard,
            ple_per_layer_tmp: &self.ple_per_layer_tmp,
            norm_x_attn_window: &self.norm_x_attn_window,
            k_pre_norm_window: &self.k_pre_norm_window,
            v_pre_norm_window: &self.v_pre_norm_window,
            hidden_in_window: &self.hidden_in_window,
            q_pre_norm_window: &self.q_pre_norm_window,
            q_post_rope_window: &self.q_post_rope_window,
            attn_out_window: &self.attn_out_window,
            attn_proj_window: &self.attn_proj_window,
            pre_ffn_rms_window: &self.pre_ffn_rms_window,
            norm_x_ffn_window: &self.norm_x_ffn_window,
            ffn_gate_window: &self.ffn_gate_window,
            ffn_up_window: &self.ffn_up_window,
            ffn_act_window: &self.ffn_act_window,
            ffn_out_window: &self.ffn_out_window,
            ple_state_window: &self.ple_state_window,
            ple_act_window: &self.ple_act_window,
            ple_proj_window: &self.ple_proj_window,
        }
    }

    /// Read back `d_hidden` (the running residual-stream gradient).
    /// After `Forward::backward_step` with
    /// `backward_layer_floor = target_layer + 1`, this contains
    /// `∂loss/∂hidden_input[target_layer + 1]` which by the residual
    /// chain rule equals `∂loss/∂ffn_out[target_layer]` (modulo the
    /// path through `attn_residual[target_layer]` and
    /// `hidden_input[target_layer]`, which are independent of
    /// `ffn_out[target_layer]` in the forward DAG).
    pub async fn read_d_hidden(&self, ctx: &WgpuCtx) -> Result<Vec<f32>> {
        let bytes = (self.cfg_d_model as u64) * 4;
        let staging = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("scratch.read_d_hidden.staging"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scratch.read_d_hidden"),
            });
        enc.copy_buffer_to_buffer(&self.d_hidden, 0, &staging, 0, bytes);
        ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
        receiver
            .await
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }
}

/// In-memory store of per-layer Cholesky factors `L` of the
/// covariance `C + ridge·I`, loaded from a sidecar safetensors file
/// produced by `examples/compute_rome_covariance.rs`.
///
/// Used by full ROME (`Model::rome_edit_native_with_covariance`) to
/// compute the normalized denominator `s = k*ᵀ C⁻¹ k*` without ever
/// materializing `C⁻¹` explicitly. Two triangular solves of `L`
/// produce `x = C⁻¹ k*`, after which `s = k* · x`.
///
/// Memory: one `d_ffn × d_ffn` f32 matrix per loaded layer. For
/// Gemma 4 e2b layer 10 (`d_ffn = 6144`) that's ~144 MB per layer.
/// In practice users calibrate ONE layer (the empirically best one
/// from the Phase 1.5 sweep) and ship a single-layer sidecar.
pub struct RomeCovariance {
    /// `layer index → (d_ffn, L row-major [d_ffn × d_ffn])`. Upper
    /// triangle is zeroed by the calibration tool.
    factors: std::collections::HashMap<u32, (u32, Vec<f32>)>,
}

impl RomeCovariance {
    /// Parse safetensors bytes produced by `compute_rome_covariance`.
    ///
    /// Expected tensor naming: `rome_cov_chol.blk.{layer}.ffn_down`,
    /// shape `[d_ffn, d_ffn]`, dtype f32, lower-triangular content.
    /// Other tensors in the file are ignored — useful if users pack
    /// multiple layers' factors into one sidecar.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| RullamaError::Inference(format!("rome_cov safetensors: {e}")))?;
        let mut factors = std::collections::HashMap::new();
        for (name, view) in st.tensors() {
            let layer = match parse_cov_tensor_name(&name) {
                Some(l) => l,
                None => continue,
            };
            if view.dtype() != safetensors::tensor::Dtype::F32 {
                return Err(RullamaError::Inference(format!(
                    "rome_cov {name}: expected f32, got {:?}",
                    view.dtype()
                )));
            }
            let shape = view.shape();
            if shape.len() != 2 || shape[0] != shape[1] {
                return Err(RullamaError::Inference(format!(
                    "rome_cov {name}: expected square 2D tensor, got {shape:?}"
                )));
            }
            let d = shape[0] as u32;
            let data: &[u8] = view.data();
            if data.len() != (d as usize) * (d as usize) * 4 {
                return Err(RullamaError::Inference(format!(
                    "rome_cov {name}: byte len {} != d² × 4",
                    data.len()
                )));
            }
            let l_vec: Vec<f32> = bytemuck::cast_slice::<u8, f32>(data).to_vec();
            factors.insert(layer, (d, l_vec));
        }
        if factors.is_empty() {
            return Err(RullamaError::Inference(
                "rome_cov: no rome_cov_chol.blk.<layer>.ffn_down tensors found".into(),
            ));
        }
        Ok(Self { factors })
    }

    /// True iff this sidecar has a Cholesky factor for `layer`.
    pub fn has_layer(&self, layer: u32) -> bool {
        self.factors.contains_key(&layer)
    }

    /// Sorted list of layers for which factors are present. Mostly for
    /// diagnostics / CLI listing.
    pub fn layers(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.factors.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Solve `(C + ridge·I) x = k` for x, using the precomputed
    /// Cholesky factor `L L^T = C + ridge·I`. Returns `x` of length
    /// `d_ffn`.
    ///
    /// Two triangular solves:
    ///   1. Forward:  `L y = k`   (lower-triangular)
    ///   2. Back:     `Lᵀ x = y`  (upper-triangular)
    ///
    /// O(d_ffn²) total — ~38M f32 mul-adds at d=6144, well under
    /// 50 ms single-threaded.
    pub fn cov_inv_k(&self, layer: u32, k: &[f32]) -> Result<Vec<f32>> {
        let (d, l) = self.factors.get(&layer).ok_or_else(|| {
            RullamaError::Inference(format!(
                "rome_cov: no factor for layer {layer} (have {:?})",
                self.layers()
            ))
        })?;
        let d = *d as usize;
        if k.len() != d {
            return Err(RullamaError::Inference(format!(
                "rome_cov.cov_inv_k: k len {} != d_ffn {d}",
                k.len()
            )));
        }
        let mut y = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = k[i];
            let row = &l[i * d..i * d + i];
            for j in 0..i {
                sum -= row[j] * y[j];
            }
            let diag = l[i * d + i];
            if diag.abs() < 1e-12 {
                return Err(RullamaError::Inference(format!(
                    "rome_cov: zero diagonal at row {i} (corrupt sidecar?)"
                )));
            }
            y[i] = sum / diag;
        }
        let mut x = vec![0.0f32; d];
        for i in (0..d).rev() {
            let mut sum = y[i];
            // L^T[i,j] = L[j,i], so we walk rows below i in column i.
            for j in (i + 1)..d {
                sum -= l[j * d + i] * x[j];
            }
            let diag = l[i * d + i];
            x[i] = sum / diag;
        }
        Ok(x)
    }
}

/// Parse `rome_cov_chol.blk.<layer>.ffn_down` → `Some(layer)`. Returns
/// None for tensors with any other naming, so unknown entries in a
/// shared sidecar are silently ignored.
fn parse_cov_tensor_name(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("rome_cov_chol.blk.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    if suffix != "ffn_down" {
        return None;
    }
    layer_str.parse().ok()
}

/// Build an `n_layers`-long Vec of all-None LoRA slots — the
/// "no LoRA" shape for ROME's backward pass.
pub fn empty_lora_slots(n_layers: u32) -> Vec<LayerLoraSlots<'static>> {
    (0..n_layers)
        .map(|_| LayerLoraSlots {
            q: None,
            k: None,
            v: None,
            o: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
        })
        .collect()
}

/// Same for grad slots — `Forward::backward_step` checks the slot's
/// LoRA presence, not the grad's, but signature requires matching
/// shapes.
pub fn empty_lora_grads(n_layers: u32) -> Vec<LayerLoraGrads<'static>> {
    (0..n_layers)
        .map(|_| LayerLoraGrads {
            q: None,
            k: None,
            v: None,
            o: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
        })
        .collect()
}
