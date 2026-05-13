//! Chained GPU forward pass: one [`wgpu::CommandEncoder`] per token, one submit,
//! one final logits readback. Targets ≥ 10 tok/s on M-series Mac.
//!
//! Architecture:
//! * All scratch tensors live in persistent `wgpu::Buffer`s allocated at construction
//!   time (sized to the model's max-shape worst case across all layers).
//! * Per-layer K/V caches are full-history GPU buffers; we append at offset =
//!   `kv_lens[i] * n_kv_heads * head_dim * 4` via `copy_buffer_to_buffer` inside the
//!   token's encoder. KV-shared layers alias the donor's `Arc<wgpu::Buffer>`.
//! * The CPU-resident token embedding row is dequantized once per token and
//!   uploaded — too small (single row of Q6_K) to be worth a GPU kernel.
//! * Logits are read back at the end of each token. Sampling stays on CPU.
//!
//! Behaviour mirrors `forward_gpu::forward_token_gpu` op-for-op; that function
//! remains the parity oracle and is now invoked only by `examples/forward_parity`.

use std::sync::Arc;

use crate::backend::dispatch::{
    attention_chained, attention_backward_dkv_chained, attention_backward_dq_chained,
    attention_probs_chained,
    cross_entropy_backward_chained, geglu_backward_chained, geglu_chained,
    lora_matmul_col_chained, lora_matmul_row_chained,
    lora_outer_add_chained,
    make_dummy_storage, matmul_q4_k_backward_input_chained,
    matmul_q4_k_chained, matmul_q6_k_backward_input_chained, matmul_q6_k_chained,
    residual_add_chained, rmsnorm_backward_chained, rmsnorm_chained,
    rmsnorm_per_row_backward_chained, rmsnorm_per_row_chained,
    rope_neox_backward_chained, rope_neox_chained, scale_chained, softcap_chained,
};

/// Activation capture buffers for one transformer layer. Used by the
/// training backward pass to read forward intermediates without
/// recomputing them. Sized for a **single query position** (M0); M1
/// will extend along a seq axis.
///
/// Each buffer must be a STORAGE | COPY_DST | COPY_SRC `wgpu::Buffer`
/// large enough to hold the named tensor at the layer's per-position
/// shape (see `crates/rullama-finetune/src/scratch.rs`).
pub struct LayerCaptureBuffers<'a> {
    /// `self.hidden` snapshot at the start of the layer ([d_model]).
    pub hidden_in:   &'a wgpu::Buffer,
    /// Output of attn rmsnorm ([d_model]).
    pub norm_x_attn: &'a wgpu::Buffer,
    /// q matmul output before q_norm rmsnorm ([n_heads · head_dim]).
    pub q_pre_norm:  &'a wgpu::Buffer,
    /// q after q_norm rmsnorm AND RoPE ([n_heads · head_dim]).
    pub q_post_rope: &'a wgpu::Buffer,
    /// k matmul output before k_norm rmsnorm ([n_kv · head_dim]).
    pub k_pre_norm:  &'a wgpu::Buffer,
    /// v matmul output before v_norm rmsnorm ([n_kv · head_dim]).
    pub v_pre_norm:  &'a wgpu::Buffer,
    /// Attention output, input to o_proj ([n_heads · head_dim]).
    pub attn_out:    &'a wgpu::Buffer,
    /// o_proj matmul output, input to post_attn_norm rmsnorm ([d_model]).
    pub attn_proj:   &'a wgpu::Buffer,
    /// `self.hidden` after the attn residual add ([d_model]).
    pub pre_ffn_rms: &'a wgpu::Buffer,
    /// Output of ffn rmsnorm ([d_model]).
    pub norm_x_ffn:  &'a wgpu::Buffer,
    /// Gate matmul output ([ffn_inter]).
    pub ffn_gate:    &'a wgpu::Buffer,
    /// Up matmul output ([ffn_inter]).
    pub ffn_up:      &'a wgpu::Buffer,
    /// GEGLU output, input to ffn_down ([ffn_inter]).
    pub ffn_act:     &'a wgpu::Buffer,
    /// ffn_down matmul output, input to post_ffw_norm rmsnorm ([d_model]).
    pub ffn_out:     &'a wgpu::Buffer,
}

/// One LoRA wrapper's GPU state — A, B, and a small `z` scratch that
/// the forward correction writes into and the backward reads from.
///
/// Forward: `y[out_dim] = W·x + scale · B · (A·x)`. The `z` buffer
/// holds `A·x` (size `[rank]`) after the forward correction so the
/// backward can build `dB = scale · dy ⊗ z`.
pub struct LoraSlot<'a> {
    pub a: &'a wgpu::Buffer,  // [rank, in_dim]
    pub b: &'a wgpu::Buffer,  // [out_dim, rank]
    pub z: &'a wgpu::Buffer,  // [rank] scratch
    pub rank:  u32,
    pub scale: f32,           // alpha / rank
}

/// Per-layer LoRA slots for the four attention projections. Pass
/// `None` for projections that aren't LoRA-wrapped. (M2 extends with
/// ffn slots.)
pub struct LayerLoraSlots<'a> {
    pub q: Option<LoraSlot<'a>>,
    pub k: Option<LoraSlot<'a>>,
    pub v: Option<LoraSlot<'a>>,
    pub o: Option<LoraSlot<'a>>,
}
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::gguf::GgmlDtype;
use crate::model::config::{Gemma4Config, LayerKind};
use crate::reference::forward::build_donor_map_pub;
use crate::reference::weights::Weights;

use bytemuck::{Pod, Zeroable};
use futures_channel::oneshot;

/// Maximum supported KV history length. Determines per-layer KV buffer size:
/// `MAX_CONTEXT * n_kv_heads(i) * head_dim(i) * 4 bytes` per layer per (K,V).
/// 4096 chosen so a 35-layer Gemma 4 e2b config fits comfortably under 1 GiB.
pub const MAX_CONTEXT: u32 = 4096;

pub struct Forward {
    cfg: Gemma4Config,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    wcache: Arc<WeightCache>,
    weights: Weights,

    // Running residual stream (d_model f32). Layer body writes into this in-place.
    hidden: wgpu::Buffer,

    // Per-layer scratch (max-shape sized).
    norm_x: wgpu::Buffer,    // d_model
    norm_y: wgpu::Buffer,    // d_model
    q: wgpu::Buffer,         // n_heads * head_dim_max
    q_norm: wgpu::Buffer,    // n_heads * head_dim_max (post-norm Q)
    k: wgpu::Buffer,         // n_kv_heads_max * head_dim_max
    k_norm: wgpu::Buffer,
    v: wgpu::Buffer,
    v_norm: wgpu::Buffer,
    attn_out_buf: wgpu::Buffer, // n_heads * head_dim_max
    attn_proj: wgpu::Buffer,    // d_model
    ffn_gate: wgpu::Buffer,     // ffn_inter_max
    ffn_up: wgpu::Buffer,
    ffn_act: wgpu::Buffer,
    ffn_out: wgpu::Buffer,      // d_model

    // PLE prep (computed once per token, then sliced per-layer).
    per_layer_residual: wgpu::Buffer, // n_layers * ple_dim
    per_layer_proj: wgpu::Buffer,
    per_layer: wgpu::Buffer,          // final per-layer inputs

    // PLE per-layer scratch.
    ple_state: wgpu::Buffer,    // ple_dim
    ple_act: wgpu::Buffer,      // ple_dim
    ple_proj: wgpu::Buffer,     // d_model

    // Output projection per-tile scratch (sized to max tile rows). Each output tile
    // matmul writes into this; we then copy_buffer_to_buffer into `logits` at the
    // correct vocab-offset (storage-buffer offset alignment is 256, but
    // copy_buffer_to_buffer alignment is just 4).
    logits_tile: wgpu::Buffer,

    // Output.
    logits: wgpu::Buffer,
    logits_read: wgpu::Buffer,

    // KV cache: one Buffer per layer for K and per layer for V, possibly aliased
    // (KV-shared layers point to the donor's Arc).
    kv_k: Vec<Arc<wgpu::Buffer>>,
    kv_v: Vec<Arc<wgpu::Buffer>>,
    kv_lens: Vec<u32>,
    donor_map: Vec<Option<u32>>,

    // Per-layer output scalar (typically only on global layers; one f32 each).
    // Loaded once at construction so the encoder doesn't have to read from CPU.
    layer_scalars: Vec<Option<f32>>,

    // Bound dummy zero buffer for "no weight" / "no factors" slots.
    dummy: wgpu::Buffer,

    /// Cap the KV cache can grow to (configured at construction). Step
    /// methods bounds-check against this instead of the compile-time
    /// `MAX_CONTEXT`, so a mobile build with a smaller cache can still
    /// surface a clean "context length exceeded" error.
    max_context: u32,

    // Cached scale factor for the final logits softcap dispatch.
    pos: u32,
}

impl Forward {
    /// Default constructor — preallocates KV cache for `MAX_CONTEXT` tokens.
    pub async fn new(
        cfg: Gemma4Config,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        weights: Weights,
        wcache: Arc<WeightCache>,
    ) -> Result<Self> {
        Self::new_with_max_context(cfg, ctx, pipes, weights, wcache, MAX_CONTEXT).await
    }

    /// Variant of [`new`] that lets the caller cap the KV-cache pre-allocation
    /// at fewer than `MAX_CONTEXT` tokens. The KV cache is the dominant GPU
    /// memory cost at load time: per non-donor layer it's
    /// `max_context * n_kv_heads * head_dim * 4 bytes` × 2 (K and V). On
    /// gemma4:e2b a `max_context=4096` cache lands at several hundred MB
    /// before any tensor is uploaded; on iPhone-class shared RAM (8 GB total)
    /// that's enough to push the WebContent process over Jetsam during the
    /// first inference step. Mobile callers pass a smaller value (e.g. 512)
    /// and get a working model that just can't grow past that turn length.
    pub async fn new_with_max_context(
        cfg: Gemma4Config,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        weights: Weights,
        wcache: Arc<WeightCache>,
        max_context: u32,
    ) -> Result<Self> {
        if max_context == 0 || max_context > MAX_CONTEXT {
            return Err(crate::error::RullamaError::Inference(format!(
                "max_context={max_context} out of range (1..={MAX_CONTEXT})"
            )));
        }
        let device = &ctx.device;

        let alloc_storage = |label: &str, n: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let d_model = cfg.d_model as usize;
        let n_heads = cfg.n_heads as usize;
        let head_dim_max = cfg.head_dim_global.max(cfg.head_dim_swa) as usize;
        let n_kv_heads_max = cfg.n_kv_heads_global.max(cfg.n_kv_heads_swa) as usize;
        let ffn_inter_max = (0..cfg.n_layers).map(|i| cfg.ffn(i)).max().unwrap_or(0) as usize;
        let ple_dim = cfg.ple_dim as usize;
        let n_layers = cfg.n_layers as usize;
        let vocab = cfg.vocab_size as usize;

        let hidden = alloc_storage("fwd.hidden", d_model);
        let norm_x = alloc_storage("fwd.norm_x", d_model);
        let norm_y = alloc_storage("fwd.norm_y", d_model);
        let q       = alloc_storage("fwd.q", n_heads * head_dim_max);
        let q_norm  = alloc_storage("fwd.q_norm", n_heads * head_dim_max);
        let k       = alloc_storage("fwd.k", n_kv_heads_max * head_dim_max);
        let k_norm  = alloc_storage("fwd.k_norm", n_kv_heads_max * head_dim_max);
        let v       = alloc_storage("fwd.v", n_kv_heads_max * head_dim_max);
        let v_norm  = alloc_storage("fwd.v_norm", n_kv_heads_max * head_dim_max);
        let attn_out_buf = alloc_storage("fwd.attn_out", n_heads * head_dim_max);
        let attn_proj = alloc_storage("fwd.attn_proj", d_model);
        let ffn_gate = alloc_storage("fwd.ffn_gate", ffn_inter_max);
        let ffn_up   = alloc_storage("fwd.ffn_up", ffn_inter_max);
        let ffn_act  = alloc_storage("fwd.ffn_act", ffn_inter_max);
        let ffn_out  = alloc_storage("fwd.ffn_out", d_model);

        let per_layer_residual = alloc_storage("fwd.per_layer_residual", n_layers * ple_dim.max(1));
        let per_layer_proj     = alloc_storage("fwd.per_layer_proj",     n_layers * ple_dim.max(1));
        let per_layer          = alloc_storage("fwd.per_layer",          n_layers * ple_dim.max(1));

        let ple_state = alloc_storage("fwd.ple_state", ple_dim.max(1));
        let ple_act   = alloc_storage("fwd.ple_act",   ple_dim.max(1));
        let ple_proj  = alloc_storage("fwd.ple_proj",  d_model);

        // Output projection tile scratch: large enough to hold the worst-case tile
        // (MAX_TILE_BYTES / row_bytes rows × 4 bytes per row of f32 logits). 80 MiB
        // tile / 1 byte-per-row-of-Q6_K... actually the tile size is in *weight*
        // bytes, not output bytes. The output is n_rows f32, where n_rows is at
        // most ceil(MAX_TILE_BYTES / row_bytes_of_token_embd). For Gemma 4 e2b
        // that's roughly 80 MiB / 1228 bytes/row ≈ 68 K rows × 4 = 272 KB. We
        // overprovision to vocab_size to keep things simple.
        let logits_tile = alloc_storage("fwd.logits_tile", vocab);

        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd.logits"),
            size: (vocab * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let logits_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd.logits_read"),
            size: (vocab * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // KV cache: alloc owned buffers for non-donor layers, then alias the rest.
        let donor_map = build_donor_map_pub(&cfg);
        let mut kv_k_opt: Vec<Option<Arc<wgpu::Buffer>>> = vec![None; n_layers];
        let mut kv_v_opt: Vec<Option<Arc<wgpu::Buffer>>> = vec![None; n_layers];
        for i in 0..n_layers {
            if donor_map[i].is_none() {
                let n_kv = cfg.n_kv_heads(i as u32) as usize;
                let hd   = cfg.head_dim(i as u32) as usize;
                let bytes = (max_context as usize * n_kv * hd * 4) as u64;
                kv_k_opt[i] = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("fwd.kv_k.{i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })));
                kv_v_opt[i] = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("fwd.kv_v.{i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })));
            }
        }
        for i in 0..n_layers {
            if let Some(d) = donor_map[i] {
                kv_k_opt[i] = kv_k_opt[d as usize].clone();
                kv_v_opt[i] = kv_v_opt[d as usize].clone();
            }
        }
        let kv_k: Vec<Arc<wgpu::Buffer>> = kv_k_opt.into_iter().map(|x| x.unwrap()).collect();
        let kv_v: Vec<Arc<wgpu::Buffer>> = kv_v_opt.into_iter().map(|x| x.unwrap()).collect();
        let kv_lens = vec![0u32; n_layers];

        let dummy = make_dummy_storage(device, "fwd.dummy");

        // Load per-layer output scalars once. The CPU oracle does
        // `weights.load_opt(layer_output_scale.weight)?.first()` per layer per
        // token; we cache the f32 here so the encoder can hand it to scale_chained
        // without an extra GPU↔CPU bounce.
        let mut layer_scalars: Vec<Option<f32>> = Vec::with_capacity(n_layers);
        for i in 0..cfg.n_layers {
            let name = format!("blk.{i}.layer_output_scale.weight");
            let v = weights.load_opt_async(&name).await?;
            layer_scalars.push(v.and_then(|vec| vec.first().copied()));
        }

        Ok(Self {
            cfg, ctx, pipes, wcache, weights,
            hidden, norm_x, norm_y,
            q, q_norm, k, k_norm, v, v_norm,
            attn_out_buf, attn_proj,
            ffn_gate, ffn_up, ffn_act, ffn_out,
            per_layer_residual, per_layer_proj, per_layer,
            ple_state, ple_act, ple_proj,
            logits_tile, logits, logits_read,
            kv_k, kv_v, kv_lens, donor_map,
            layer_scalars,
            dummy,
            max_context,
            pos: 0,
        })
    }

    pub fn cfg(&self) -> &Gemma4Config { &self.cfg }
    pub fn pos(&self) -> u32 { self.pos }
    /// Borrow the GPU context (`WgpuCtx` is internally `Arc`-backed and
    /// cheap to clone). Used by `rullama-finetune` to allocate LoRA and
    /// scratch buffers on the same device + queue as the model.
    pub fn ctx(&self) -> &WgpuCtx { &self.ctx }
    /// Borrow the pipeline cache. The training crate doesn't need this
    /// directly (the backward path goes through `Forward::backward_step`),
    /// but exposing it keeps the surface symmetric for future test code.
    pub fn pipes(&self) -> &std::sync::Arc<Pipelines> { &self.pipes }
    /// Read-only handle on the model's logits buffer (post-forward).
    /// `TrainingSession::step` uses this to feed
    /// `cross_entropy_backward` without exposing the rest of Forward's
    /// scratch.
    pub fn logits_buffer(&self) -> &wgpu::Buffer { &self.logits }

    pub fn reset(&mut self) {
        self.pos = 0;
        for l in self.kv_lens.iter_mut() { *l = 0; }
    }

    /// Run one forward step from a token id. Looks up the token's embedding row,
    /// uploads it to the hidden buffer, then runs the rest of the forward.
    pub async fn step(&mut self, token_id: u32) -> Result<Vec<f32>> {
        self.step_inner(token_id, None, None).await
    }

    /// Run one forward step **with per-layer activation capture** into
    /// the supplied buffers. Used by the training backward pass —
    /// `capture[i]` receives the layer-`i` intermediates needed by the
    /// reverse walker. Pass exactly `cfg.n_layers` entries.
    ///
    /// Capture only emits `copy_buffer_to_buffer` commands inside the
    /// per-token encoder; there is no extra submit. Adds ~12 small
    /// copies per layer (≤ d_model floats each), trivial vs. the
    /// per-layer matmul cost.
    pub async fn step_capture<'a>(
        &mut self,
        token_id: u32,
        capture: &'a [LayerCaptureBuffers<'a>],
        loras: Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        if capture.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_capture: got {} capture layers, expected {}",
                capture.len(), self.cfg.n_layers
            )));
        }
        if let Some(l) = loras {
            if l.len() != self.cfg.n_layers as usize {
                return Err(RullamaError::Inference(format!(
                    "step_capture: got {} lora slots, expected {}",
                    l.len(), self.cfg.n_layers
                )));
            }
        }
        self.step_inner(token_id, Some(capture), loras).await
    }

    /// Run a forward step with LoRA correction enabled but **without**
    /// capturing activations. Used for the prompt-prefill pass during
    /// training (positions 0..N-2 just fill KV; only the final position
    /// is captured + has its loss measured).
    pub async fn step_with_lora<'a>(
        &mut self,
        token_id: u32,
        loras: &'a [LayerLoraSlots<'a>],
    ) -> Result<Vec<f32>> {
        if loras.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_lora: got {} lora slots, expected {}",
                loras.len(), self.cfg.n_layers
            )));
        }
        self.step_inner(token_id, None, Some(loras)).await
    }

    async fn step_inner<'a>(
        &mut self,
        token_id: u32,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras:   Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        if (token_id as u64) >= self.cfg.vocab_size as u64 {
            return Err(RullamaError::Inference(format!(
                "token_id {token_id} >= vocab_size {}", self.cfg.vocab_size
            )));
        }
        if self.pos >= self.max_context {
            return Err(RullamaError::Inference(format!(
                "context length exceeded max_context={}", self.max_context
            )));
        }
        let d_model = self.cfg.d_model as usize;
        let ple_dim = self.cfg.ple_dim as usize;

        // ---- CPU-side per-token preamble: token embed + PLE input dequant + upload ----
        let mut hidden_cpu = self.weights.load_row_async("token_embd.weight", token_id as usize).await?;
        let scale_factor = (d_model as f32).sqrt();
        for v in hidden_cpu.iter_mut() { *v *= scale_factor; }
        self.ctx.queue.write_buffer(&self.hidden, 0, bytemuck::cast_slice(&hidden_cpu));
        drop(hidden_cpu);

        if self.cfg.has_ple() {
            let mut ple_in = self.weights
                .load_row_async("per_layer_token_embd.weight", token_id as usize)
                .await?;
            let s = (ple_dim as f32).sqrt();
            for v in ple_in.iter_mut() { *v *= s; }
            self.ctx.queue.write_buffer(&self.per_layer_residual, 0, bytemuck::cast_slice(&ple_in));
            drop(ple_in);
        }

        self.run_forward_from_hidden(capture, loras).await
    }

    /// Run one forward step from a pre-computed `[d_model]` embedding (vision soft
    /// token, audio soft token, etc.). Skips the `token_embd` lookup; the caller is
    /// responsible for the embedding scale (vision/audio projectors already produce
    /// rmsnorm-normalised outputs).
    ///
    /// PLE prep is run with a zeroed per-layer-residual — there is no
    /// `per_layer_token_embd` lookup possible without a token id; the per-layer
    /// projection from the residual stream still contributes. This matches
    /// Ollama's behaviour: multimodal soft tokens flow through the LM as frozen
    /// inputs and don't get PLE injection.
    pub async fn step_with_embedding(&mut self, embedding: &[f32]) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        if embedding.len() != d_model {
            return Err(RullamaError::Inference(format!(
                "step_with_embedding: got {} f32s, expected d_model = {d_model}",
                embedding.len(),
            )));
        }
        if self.pos >= self.max_context {
            return Err(RullamaError::Inference(format!(
                "context length exceeded max_context={}", self.max_context
            )));
        }
        // Direct upload — caller's embedding is the new hidden state.
        self.ctx.queue.write_buffer(&self.hidden, 0, bytemuck::cast_slice(embedding));

        // Zero out per_layer_residual for this step (no token id → no PLE lookup).
        if self.cfg.has_ple() {
            let n_layers = self.cfg.n_layers as usize;
            let zeros = vec![0f32; n_layers * self.cfg.ple_dim as usize];
            self.ctx.queue.write_buffer(&self.per_layer_residual, 0, bytemuck::cast_slice(&zeros));
        }

        self.run_forward_from_hidden(None, None).await
    }

    /// Forward pass starting from `self.hidden` already populated. Shared by
    /// `step` (token-id path) and `step_with_embedding` (multimodal soft tokens).
    async fn run_forward_from_hidden<'a>(
        &mut self,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras:   Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        let n_layers = self.cfg.n_layers as usize;
        let ple_dim = self.cfg.ple_dim as usize;
        let eps = self.cfg.rms_norm_eps;
        let pos = self.pos;

        // ---- weights we need on GPU before encoder construction ----
        // (WeightCache.buffer_async fetches + uploads on first touch; cached afterwards.)
        let wc = self.wcache.clone();
        let final_norm = wc.buffer_async("output_norm.weight").await?;
        let token_embd_dtype = wc.dtype("token_embd.weight")?;

        // PLE prep weights
        let (ple_proj_w_buf, ple_proj_norm_w_buf, ple_proj_n) = if self.cfg.has_ple() {
            if wc.dtype("per_layer_model_proj.weight")? != GgmlDtype::Q4_K {
                return Err(RullamaError::Inference("per_layer_model_proj expected Q4_K".into()));
            }
            let proj_w = wc.buffer_async("per_layer_model_proj.weight").await?;
            let proj_norm = wc.buffer_async("per_layer_proj_norm.weight").await?;
            (Some(proj_w), Some(proj_norm), n_layers * ple_dim)
        } else {
            (None, None, 0)
        };

        // ---- build the per-token CommandEncoder ----
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fwd.token_encoder"),
        });

        // ---- PLE prep (chained, all GPU) ----
        // per_layer_residual *= sqrt(ple_dim)  → already done CPU-side above (one mul each).
        // proj = matmul(per_layer_model_proj, hidden) → per_layer_proj
        // proj *= 1/sqrt(d_model)
        // per_layer = rmsnorm_per_row(per_layer_proj, per_layer_proj_norm.weight)
        // per_layer += per_layer_residual
        // per_layer *= 1/sqrt(2)
        if self.cfg.has_ple() {
            let proj_w = ple_proj_w_buf.as_ref().unwrap();
            let proj_norm_w = ple_proj_norm_w_buf.as_ref().unwrap();

            matmul_q4_k_chained(&self.ctx, &self.pipes, &mut enc,
                proj_w, &self.hidden, &self.per_layer_proj, d_model, ple_proj_n);
            scale_chained(&self.ctx, &self.pipes, &mut enc,
                &self.per_layer_proj, ple_proj_n, 1.0 / (d_model as f32).sqrt());
            rmsnorm_per_row_chained(&self.ctx, &self.pipes, &mut enc,
                &self.per_layer_proj, Some(proj_norm_w), &self.dummy,
                &self.per_layer, n_layers, ple_dim, eps);
            residual_add_chained(&self.ctx, &self.pipes, &mut enc,
                &self.per_layer, &self.per_layer_residual, ple_proj_n);
            scale_chained(&self.ctx, &self.pipes, &mut enc,
                &self.per_layer, ple_proj_n, 1.0 / 2.0_f32.sqrt());
        }

        // ---- transformer layers ----
        // Per-layer submit + restart. Each flush hands its commands off to the
        // GPU and frees the CPU-side encoder; persistent buffer state on the
        // GPU is unaffected. Empirically anything wider than 1 layer per
        // submit (tried 3) re-introduces the iPhone WebContent crash on the
        // first step — the per-layer cadence is the working strip-line.
        for i in 0..n_layers as u32 {
            let cap = capture.map(|c| &c[i as usize]);
            let lora = loras.map(|l| &l[i as usize]);
            self.encode_layer(&mut enc, i, pos, cap, lora).await?;
            self.ctx.queue.submit(Some(enc.finish()));
            enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd.token_encoder.cont"),
            });
        }

        // ---- final norm (in-place into hidden via norm_y as scratch) ----
        rmsnorm_chained(&self.ctx, &self.pipes, &mut enc,
            &self.hidden, Some(&final_norm), &self.dummy, &self.norm_x, d_model, eps);

        // Flush before the output projection — it's the second-largest concentration
        // of GPU work in the step (262K-row matmul against the embedding) and we
        // don't want it queued behind a still-encoding layer batch.
        self.ctx.queue.submit(Some(enc.finish()));
        enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fwd.out_proj_encoder"),
        });

        // ---- output projection (tiled): tile along vocab axis ----
        // Each tile matmul writes its rows into `logits_tile` starting at offset 0
        // (so it always satisfies the storage-binding alignment), then we copy
        // those bytes into `logits` at offset `row_start * 4` (copy_buffer_to_buffer
        // only needs 4-byte alignment). Submit between tiles too, for the same
        // command-buffer-size reason that we submit between layers.
        // token_embd is the largest single tensor in the model (315 MiB
        // compressed Q6_K for gemma4:e2b). Empirically 80 MiB tiles crash
        // the WebContent process on iPhone 16e mid-step even after the
        // wasm-side per-tile range fetch landed — the issue isn't the
        // staging allocation, it's a single 80 MiB wgpu::Buffer creation
        // on top of ~2 GB of resident layer weights. 8 MiB tiles work.
        const MAX_TILE_BYTES: usize = 8 * 1024 * 1024;
        let tiles = wc.buffer_tiles_async("token_embd.weight", MAX_TILE_BYTES).await?;
        for tile in &tiles {
            run_matmul_into_buf(
                &self.ctx, &self.pipes, &mut enc,
                token_embd_dtype, &tile.buffer, &self.norm_x,
                &self.logits_tile, tile.n_rows, d_model,
                "fwd.output_tile",
            )?;
            enc.copy_buffer_to_buffer(
                &self.logits_tile, 0,
                &self.logits, (tile.row_start as u64) * 4,
                (tile.n_rows as u64) * 4,
            );
            self.ctx.queue.submit(Some(enc.finish()));
            enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd.out_proj_encoder.cont"),
            });
        }

        // ---- softcap ----
        // Out-of-place: read from `logits`, write into `logits_tile`. wgpu
        // disallows binding the same buffer as both read-only and read-write
        // within one dispatch, so we can't softcap in-place.
        let final_src: &wgpu::Buffer = if self.cfg.final_logit_softcap > 0.0 {
            softcap_chained(&self.ctx, &self.pipes, &mut enc,
                &self.logits, &self.logits_tile,
                self.cfg.vocab_size as usize, self.cfg.final_logit_softcap);
            &self.logits_tile
        } else {
            &self.logits
        };

        // ---- copy logits → readback buffer ----
        enc.copy_buffer_to_buffer(final_src, 0, &self.logits_read, 0,
            (self.cfg.vocab_size as u64) * 4);

        // ---- submit + readback ----
        self.ctx.queue.submit(Some(enc.finish()));
        let logits = read_back_f32(&self.ctx.device, &self.logits_read).await?;

        self.pos = self.pos.saturating_add(1);
        Ok(logits)
    }

    async fn encode_layer<'a>(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        i: u32,
        pos: u32,
        capture: Option<&'a LayerCaptureBuffers<'a>>,
        loras:   Option<&'a LayerLoraSlots<'a>>,
    ) -> Result<()> {
        let prefix = format!("blk.{i}.");
        let d_model = self.cfg.d_model as usize;
        let eps = self.cfg.rms_norm_eps;
        let n_heads = self.cfg.n_heads as usize;
        let n_kv_heads = self.cfg.n_kv_heads(i) as usize;
        let head_dim = self.cfg.head_dim(i) as usize;
        let ffn_n = self.cfg.ffn(i) as usize;
        let kind = self.cfg.kind(i);
        let donor = self.donor_map[i as usize];

        // ---- CAPTURE: hidden_in (start-of-layer residual stream) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.hidden, 0, cap.hidden_in, 0, (d_model * 4) as u64);
        }

        // Pre-fetch all weights this layer needs (each is cached after first call).
        let attn_norm_w = self.wcache.buffer_async(&format!("{prefix}attn_norm.weight")).await?;
        let post_attn_w = self.wcache.buffer_async(&format!("{prefix}post_attention_norm.weight")).await?;
        let mlp_norm_w  = self.wcache.buffer_async(&format!("{prefix}ffn_norm.weight")).await?;
        let post_ffw_w  = self.wcache.buffer_async(&format!("{prefix}post_ffw_norm.weight")).await?;

        let q_w = self.wcache.buffer_async(&format!("{prefix}attn_q.weight")).await?;
        let q_norm_w = self.wcache.buffer_async(&format!("{prefix}attn_q_norm.weight")).await?;
        let o_w = self.wcache.buffer_async(&format!("{prefix}attn_output.weight")).await?;

        let (k_w, k_norm_w, v_w, v_w_dtype) = if donor.is_none() {
            let kw = self.wcache.buffer_async(&format!("{prefix}attn_k.weight")).await?;
            let knw = self.wcache.buffer_async(&format!("{prefix}attn_k_norm.weight")).await?;
            let v_name = format!("{prefix}attn_v.weight");
            let vw = self.wcache.buffer_async(&v_name).await?;
            let dt = self.wcache.dtype(&v_name)?;
            (Some(kw), Some(knw), Some(vw), Some(dt))
        } else {
            (None, None, None, None)
        };

        let gate_w = self.wcache.buffer_async(&format!("{prefix}ffn_gate.weight")).await?;
        let up_w   = self.wcache.buffer_async(&format!("{prefix}ffn_up.weight")).await?;
        let down_name = format!("{prefix}ffn_down.weight");
        let down_w = self.wcache.buffer_async(&down_name).await?;
        let down_dtype = self.wcache.dtype(&down_name)?;

        // PLE-injection weights (only when has_ple)
        let (inp_gate_w, proj_w, post_norm_w) = if self.cfg.has_ple() {
            let a = self.wcache.buffer_async(&format!("{prefix}inp_gate.weight")).await?;
            let b = self.wcache.buffer_async(&format!("{prefix}proj.weight")).await?;
            let c = self.wcache.buffer_async(&format!("{prefix}post_norm.weight")).await?;
            (Some(a), Some(b), Some(c))
        } else { (None, None, None) };

        let factors_w = if matches!(kind, LayerKind::Global) {
            // Same RoPE factors tensor across global layers — would benefit from caching;
            // the cache key is the tensor name so it's already a single GPU buffer.
            self.wcache.buffer_opt_async("rope_freqs.weight").await?
        } else { None };

        // ===== ATTENTION =====
        // norm_x = rmsnorm(hidden, attn_norm)
        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, Some(&attn_norm_w), &self.dummy,
            &self.norm_x, d_model, eps);

        // ---- CAPTURE: norm_x_attn (input to q/k/v matmul + LoRA) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.norm_x, 0, cap.norm_x_attn, 0, (d_model * 4) as u64);
        }

        // Q/K/V projections from norm_x
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &q_w, &self.norm_x, &self.q, d_model, n_heads * head_dim);

        // ---- LoRA forward correction (q): self.q += scale · B · (A · norm_x) ----
        if let Some(slot) = loras.and_then(|l| l.q.as_ref()) {
            // z = A · norm_x  ([rank] = [rank, d_model] @ [d_model])
            lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                slot.a, &self.norm_x, slot.z,
                d_model, slot.rank as usize, 1.0, false);
            // self.q += scale · B · z  ([n_heads*head_dim] += [n_heads*head_dim, rank] @ [rank])
            lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                slot.b, slot.z, &self.q,
                slot.rank as usize, n_heads * head_dim, slot.scale, true);
        }

        // ---- CAPTURE: q_pre_norm (q matmul output, input to q_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.q, 0, cap.q_pre_norm, 0, (n_heads * head_dim * 4) as u64);
        }

        // per-head q_norm (weighted)
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.q, Some(&q_norm_w), &self.dummy,
            &self.q_norm, n_heads, head_dim, eps);
        // RoPE in-place into q_norm
        let (rope_base, rope_dims) = match kind {
            LayerKind::SlidingWindow => (self.cfg.rope_freq_base_swa, self.cfg.rope_dim_swa as usize),
            LayerKind::Global        => (self.cfg.rope_freq_base,     self.cfg.rope_dim_global as usize),
        };
        rope_neox_chained(&self.ctx, &self.pipes, enc,
            &self.q_norm, factors_w.as_ref(), &self.dummy,
            head_dim, n_heads, pos as usize, rope_dims, rope_base);

        // ---- CAPTURE: q_post_rope (input to attention; reused in dkv pass) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.q_norm, 0, cap.q_post_rope, 0, (n_heads * head_dim * 4) as u64);
        }

        if donor.is_none() {
            let kw = k_w.as_ref().unwrap();
            let knw = k_norm_w.as_ref().unwrap();
            let vw = v_w.as_ref().unwrap();
            let vdt = v_w_dtype.unwrap();

            matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                kw, &self.norm_x, &self.k, d_model, n_kv_heads * head_dim);

            // ---- LoRA forward correction (k) ----
            if let Some(slot) = loras.and_then(|l| l.k.as_ref()) {
                lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                    slot.a, &self.norm_x, slot.z,
                    d_model, slot.rank as usize, 1.0, false);
                lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                    slot.b, slot.z, &self.k,
                    slot.rank as usize, n_kv_heads * head_dim, slot.scale, true);
            }

            // ---- CAPTURE: k_pre_norm (k matmul output, input to k_norm rmsnorm) ----
            if let Some(cap) = capture {
                enc.copy_buffer_to_buffer(&self.k, 0, cap.k_pre_norm, 0, (n_kv_heads * head_dim * 4) as u64);
            }

            rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
                &self.k, Some(knw), &self.dummy,
                &self.k_norm, n_kv_heads, head_dim, eps);
            rope_neox_chained(&self.ctx, &self.pipes, enc,
                &self.k_norm, factors_w.as_ref(), &self.dummy,
                head_dim, n_kv_heads, pos as usize, rope_dims, rope_base);

            match vdt {
                GgmlDtype::Q6_K => matmul_q6_k_chained(&self.ctx, &self.pipes, enc,
                    vw, &self.norm_x, &self.v, d_model, n_kv_heads * head_dim),
                GgmlDtype::Q4_K => matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                    vw, &self.norm_x, &self.v, d_model, n_kv_heads * head_dim),
                other => return Err(RullamaError::Inference(format!("attn_v dtype {other:?} unsupported"))),
            }

            // ---- LoRA forward correction (v) ----
            if let Some(slot) = loras.and_then(|l| l.v.as_ref()) {
                lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                    slot.a, &self.norm_x, slot.z,
                    d_model, slot.rank as usize, 1.0, false);
                lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                    slot.b, slot.z, &self.v,
                    slot.rank as usize, n_kv_heads * head_dim, slot.scale, true);
            }

            // ---- CAPTURE: v_pre_norm (v matmul output, input to unweighted v_norm rmsnorm) ----
            if let Some(cap) = capture {
                enc.copy_buffer_to_buffer(&self.v, 0, cap.v_pre_norm, 0, (n_kv_heads * head_dim * 4) as u64);
            }

            // V-norm is unweighted
            rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
                &self.v, None, &self.dummy,
                &self.v_norm, n_kv_heads, head_dim, eps);

            // Append rotated K + normed V into this layer's KV cache at offset = kv_lens[i].
            let row_bytes = (n_kv_heads * head_dim * 4) as u64;
            let dst_offset = self.kv_lens[i as usize] as u64 * row_bytes;
            enc.copy_buffer_to_buffer(&self.k_norm, 0, &self.kv_k[i as usize], dst_offset, row_bytes);
            enc.copy_buffer_to_buffer(&self.v_norm, 0, &self.kv_v[i as usize], dst_offset, row_bytes);
            self.kv_lens[i as usize] = self.kv_lens[i as usize].saturating_add(1);
        }

        // attention: kv buffers are kv_k[i], kv_v[i] (alias for donor); history_len from
        // donor's len if shared, else this layer's len (which we just incremented).
        let history_layer = donor.map(|d| d as usize).unwrap_or(i as usize);
        let history_len = self.kv_lens[history_layer] as usize;
        let window = if matches!(kind, LayerKind::SlidingWindow) { self.cfg.sliding_window as usize } else { 0 };

        attention_chained(&self.ctx, &self.pipes, enc,
            &self.q_norm, &self.kv_k[i as usize], &self.kv_v[i as usize], &self.attn_out_buf,
            head_dim, n_heads, n_kv_heads, pos as usize, history_len, window);

        // ---- CAPTURE: attn_out (input to o_proj) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.attn_out_buf, 0, cap.attn_out, 0, (n_heads * head_dim * 4) as u64);
        }

        // attn_proj = matmul(attn_out_buf, attn_output.weight)
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &o_w, &self.attn_out_buf, &self.attn_proj, n_heads * head_dim, d_model);

        // ---- LoRA forward correction (o): self.attn_proj += scale · B · (A · attn_out_buf) ----
        if let Some(slot) = loras.and_then(|l| l.o.as_ref()) {
            lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                slot.a, &self.attn_out_buf, slot.z,
                n_heads * head_dim, slot.rank as usize, 1.0, false);
            lora_matmul_row_chained(&self.ctx, &self.pipes, enc,
                slot.b, slot.z, &self.attn_proj,
                slot.rank as usize, d_model, slot.scale, true);
        }

        // ---- CAPTURE: attn_proj (input to post_attn_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.attn_proj, 0, cap.attn_proj, 0, (d_model * 4) as u64);
        }

        // norm_y = rmsnorm(attn_proj, post_attn_norm.weight)
        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.attn_proj, Some(&post_attn_w), &self.dummy,
            &self.norm_y, d_model, eps);
        // hidden += norm_y
        residual_add_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, &self.norm_y, d_model);

        // ---- CAPTURE: pre_ffn_rms (hidden after attn residual add) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.hidden, 0, cap.pre_ffn_rms, 0, (d_model * 4) as u64);
        }

        // ===== MLP =====
        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, Some(&mlp_norm_w), &self.dummy,
            &self.norm_x, d_model, eps);

        // ---- CAPTURE: norm_x_ffn (input to gate/up matmul + LoRA) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.norm_x, 0, cap.norm_x_ffn, 0, (d_model * 4) as u64);
        }

        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &gate_w, &self.norm_x, &self.ffn_gate, d_model, ffn_n);
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &up_w, &self.norm_x, &self.ffn_up, d_model, ffn_n);

        // ---- CAPTURE: ffn_gate, ffn_up (inputs to GEGLU) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.ffn_gate, 0, cap.ffn_gate, 0, (ffn_n * 4) as u64);
            enc.copy_buffer_to_buffer(&self.ffn_up,   0, cap.ffn_up,   0, (ffn_n * 4) as u64);
        }

        geglu_chained(&self.ctx, &self.pipes, enc,
            &self.ffn_gate, &self.ffn_up, &self.ffn_act, ffn_n);

        // ---- CAPTURE: ffn_act (input to ffn_down matmul) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.ffn_act, 0, cap.ffn_act, 0, (ffn_n * 4) as u64);
        }

        match down_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_chained(&self.ctx, &self.pipes, enc,
                &down_w, &self.ffn_act, &self.ffn_out, ffn_n, d_model),
            GgmlDtype::Q4_K => matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                &down_w, &self.ffn_act, &self.ffn_out, ffn_n, d_model),
            other => return Err(RullamaError::Inference(format!("ffn_down dtype {other:?} unsupported"))),
        }

        // ---- CAPTURE: ffn_out (input to post_ffw_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(&self.ffn_out, 0, cap.ffn_out, 0, (d_model * 4) as u64);
        }

        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.ffn_out, Some(&post_ffw_w), &self.dummy,
            &self.norm_y, d_model, eps);
        residual_add_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, &self.norm_y, d_model);

        // ===== PLE injection =====
        if self.cfg.has_ple() {
            let inp_gate_w = inp_gate_w.unwrap();
            let proj_w = proj_w.unwrap();
            let post_norm_w = post_norm_w.unwrap();
            let ple_dim = self.cfg.ple_dim as usize;

            // ple_state = matmul(hidden, inp_gate_w) [d_model -> ple_dim]
            matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                &inp_gate_w, &self.hidden, &self.ple_state, d_model, ple_dim);
            // Need the per-layer slice of `per_layer` as the second geglu input.
            // geglu_chained currently binds entire buffers — we'd need a sliced bind.
            // For simplicity, do a copy_buffer_to_buffer of the layer-i slice into
            // ple_act before geglu, then run geglu(ple_state, ple_act_copy). One more
            // copy per layer; trivial cost compared to the full forward.
            // Note: ple_act_copy is reused for the geglu output too — geglu does
            //   y = gate * gelu(up); the input `up` is read once before the output write.
            // To keep correctness, write the slice into a separate tmp: reuse ple_proj
            // (since it's not used until later in this block).
            let layer_off = (i as u64) * (ple_dim as u64) * 4;
            let layer_bytes = (ple_dim as u64) * 4;
            enc.copy_buffer_to_buffer(&self.per_layer, layer_off, &self.ple_proj, 0, layer_bytes);
            geglu_chained(&self.ctx, &self.pipes, enc,
                &self.ple_state, &self.ple_proj, &self.ple_act, ple_dim);

            // projected = matmul(ple_act, proj_w) [ple_dim -> d_model]
            matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                &proj_w, &self.ple_act, &self.ple_proj, ple_dim, d_model);
            // norm_y = rmsnorm(ple_proj, post_norm_w)
            rmsnorm_chained(&self.ctx, &self.pipes, enc,
                &self.ple_proj, Some(&post_norm_w), &self.dummy,
                &self.norm_y, d_model, eps);
            // hidden += norm_y
            residual_add_chained(&self.ctx, &self.pipes, enc,
                &self.hidden, &self.norm_y, d_model);
        }

        // Per-layer output scalar (loaded at construction; applied as scale_chained).
        if let Some(s) = self.layer_scalars[i as usize] {
            scale_chained(&self.ctx, &self.pipes, enc, &self.hidden, d_model, s);
        }

        Ok(())
    }
}

// ---------- helpers ----------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatmulParams { k: u32, n: u32, _p0: u32, _p1: u32 }

/// Run a matmul kernel that writes its output rows starting at offset 0 of `dst`.
/// Used for the tiled output projection: caller copies the rows from `dst` into
/// the per-tile slice of the global logits buffer.
fn run_matmul_into_buf(
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    dtype: GgmlDtype,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    n_rows: usize,
    k: usize,
    label: &str,
) -> Result<()> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    // Naive kernel beats tiled here on Apple GPUs (verified empirically on
    // M-series). Tiled pipelines stay built in case future hardware / kernel
    // tuning reverses this — flip these back if perf_bench shows tiled wins.
    let pipeline = match dtype {
        GgmlDtype::Q4_K => &pipes.q4_k_matmul,
        GgmlDtype::Q6_K => &pipes.q6_k_matmul,
        other => return Err(RullamaError::Inference(format!("output proj dtype {other:?} not supported"))),
    };
    let params = MatmulParams { k: k as u32, n: n_rows as u32, _p0: 0, _p1: 0 };
    let p_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}.params")),
        size: std::mem::size_of::<MatmulParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&p_buf, 0, bytemuck::bytes_of(&params));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bg")),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dst.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label), timestamp_writes: None,
    });
    cp.set_pipeline(pipeline);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n_rows as u32).div_ceil(64), 1, 1);
    Ok(())
}

async fn read_back_f32(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<f32>> {
    let slice = buf.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
    receiver
        .await
        .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
        .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
    let data = slice.get_mapped_range();
    let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(v)
}

// =========================================================================
//                            BACKWARD PASS
// =========================================================================
//
// Reverse-mode chained backward, parallel in structure to `encode_layer`.
// Mirrors the forward graph node-for-node — no tape, no autodiff. The
// fully-captured `LayerCaptureBuffers` for each layer plus the live KV
// cache provide every activation the reverse pass needs.
//
// Encoder cadence matches forward: one `wgpu::CommandEncoder` per layer
// (preserves the iPhone WebContent per-encoder workaround), plus single
// encoders for the CE+output-proj+final-norm head and the Adam step.

/// Per-LoRA gradient accumulators. Mirrors `LoraSlot` (the forward
/// view) with the addition of `d_a` and `d_b` — the buffers Adam will
/// step over. Backward writes into these via `lora_outer_add_chained`.
pub struct LoraGradPair<'a> {
    pub a: &'a wgpu::Buffer,    // [rank, in_dim] — read for u = Bᵀ·dy
    pub b: &'a wgpu::Buffer,    // [out_dim, rank] — read for u = Bᵀ·dy
    pub z: &'a wgpu::Buffer,    // [rank] — captured A·x from forward, dB needs it
    pub d_a: &'a wgpu::Buffer,  // [rank, in_dim] — gradient accumulator
    pub d_b: &'a wgpu::Buffer,  // [out_dim, rank] — gradient accumulator
    pub rank: u32,
    pub scale: f32,
}

/// Per-layer LoRA gradient accumulators for the four attention
/// projections. Each pair drives both the LoRA backward (computing
/// dA, dB into d_a, d_b) AND the LoRA contribution to dx
/// (Aᵀ·Bᵀ·dy added to the running input gradient).
pub struct LayerLoraGrads<'a> {
    pub q: Option<LoraGradPair<'a>>,
    pub k: Option<LoraGradPair<'a>>,
    pub v: Option<LoraGradPair<'a>>,
    pub o: Option<LoraGradPair<'a>>,
}

/// All scratch buffers the backward orchestration writes into. Sized
/// at construction time and reused across steps. Allocated by
/// `rullama-finetune::TrainingScratch`.
#[allow(clippy::struct_field_names)]
pub struct BackwardScratchView<'a> {
    /// `[vocab]` — softmax(logits) - one_hot(target).
    pub d_logits: &'a wgpu::Buffer,
    /// `[1]` — scalar CE loss (read back to CPU after backward).
    pub loss: &'a wgpu::Buffer,
    /// `[d_model]` — gradient at the final post-norm hidden (= input
    /// to output projection); used as the running d_hidden after the
    /// output proj backward chains in.
    pub d_hidden_final: &'a wgpu::Buffer,
    /// `[d_model]` — running gradient on the residual stream.
    pub d_hidden: &'a wgpu::Buffer,
    /// `[d_model]` — second d_model scratch (post-attn/post-ffn intermediates).
    pub d_hidden_tmp: &'a wgpu::Buffer,
    /// `[d_model]` — third d_model scratch (sum two contributions).
    pub d_hidden_tmp2: &'a wgpu::Buffer,
    /// `[n_heads · history_len]` — recomputed attention probs.
    pub attn_probs: &'a wgpu::Buffer,
    /// `[n_heads · history_len]` — staged d_scores (pass 1 → pass 2).
    pub attn_d_scores: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` — d_attn_out (input to attn back dq).
    pub d_attn_out: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` — d_q output of attn back dq (= d_q_post_rope).
    pub d_q: &'a wgpu::Buffer,
    /// `[history_len · n_kv · head_dim]` — d_k_hist (only row[pos] consumed in M0).
    pub d_k_hist: &'a wgpu::Buffer,
    /// `[history_len · n_kv · head_dim]` — d_v_hist.
    pub d_v_hist: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` — d after rope_back of q.
    pub d_q_pre_rope: &'a wgpu::Buffer,
    /// `[n_kv · head_dim]` — d after rope_back of k.
    pub d_k_pre_rope: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` — d after q_norm rmsnorm_back.
    pub d_q_pre_norm: &'a wgpu::Buffer,
    /// `[n_kv · head_dim]` — d after k_norm rmsnorm_back.
    pub d_k_pre_norm: &'a wgpu::Buffer,
    /// `[n_kv · head_dim]` — d after v_norm rmsnorm_back.
    pub d_v_pre_norm: &'a wgpu::Buffer,
    /// `[ffn_inter]` — d_ffn_out (matmul_back output, going into geglu_back).
    pub d_ffn_a: &'a wgpu::Buffer,
    /// `[ffn_inter]` — d_ffn_gate (geglu_back output).
    pub d_ffn_b: &'a wgpu::Buffer,
    /// `[ffn_inter]` — d_ffn_up (geglu_back output).
    pub d_ffn_c: &'a wgpu::Buffer,
}

impl Forward {
    /// Full backward pass — produces gradients into `grads` for every
    /// registered LoRA, writes the scalar CE loss into `scratch.loss`,
    /// and returns the loss value.
    ///
    /// Preconditions:
    /// - `step_capture(...)` has just run on this same `Forward`, with
    ///   `capture` and `loras` matching the slices passed here.
    /// - `self.logits` still holds the final-position logits.
    /// - `self.hidden` still holds the pre-final-norm residual stream.
    /// - `self.norm_x` still holds the post-final-norm hidden (input
    ///   to the output projection).
    /// - KV caches `self.kv_k[i]` / `self.kv_v[i]` still hold the
    ///   prompt's K/V (history length = current `pos`).
    /// - `grads[i].*.d_a` and `d_b` are pre-zeroed by the caller (the
    ///   training step's `zero_all_grads` before forward).
    ///
    /// `target_id ≥ vocab_size` masks the gradient (zero loss / zero
    /// gradient at this position).
    #[allow(clippy::too_many_arguments)]
    pub async fn backward_step<'a>(
        &mut self,
        target_id: u32,
        capture: &'a [LayerCaptureBuffers<'a>],
        loras:   &'a [LayerLoraSlots<'a>],
        grads:   &'a [LayerLoraGrads<'a>],
        scratch: &'a BackwardScratchView<'a>,
        history_len: u32,
        pos: u32,
    ) -> Result<f32> {
        let n_layers = self.cfg.n_layers as usize;
        if capture.len() != n_layers || loras.len() != n_layers || grads.len() != n_layers {
            return Err(RullamaError::Inference(
                "backward_step: capture/loras/grads slice length must equal n_layers".into(),
            ));
        }
        let d_model = self.cfg.d_model as usize;
        let vocab = self.cfg.vocab_size as usize;
        let eps = self.cfg.rms_norm_eps;

        // Fetch top-level frozen weights.
        let wc = self.wcache.clone();
        let final_norm = wc.buffer_async("output_norm.weight").await?;
        let token_embd = wc.buffer_async("token_embd.weight").await?;
        let token_embd_dtype = wc.dtype("token_embd.weight")?;

        // ===== Head: CE → output_proj_back → final norm back =====
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bwd.head"),
        });

        // d_logits + scalar loss
        cross_entropy_backward_chained(&self.ctx, &self.pipes, &mut enc,
            &self.logits, scratch.d_logits, scratch.loss, vocab, target_id);

        // d_norm_x_final = embedᵀ @ d_logits → write into scratch.d_hidden_final
        match token_embd_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx, &self.pipes, &mut enc,
                &token_embd, scratch.d_logits, scratch.d_hidden_final,
                d_model, vocab,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx, &self.pipes, &mut enc,
                &token_embd, scratch.d_logits, scratch.d_hidden_final,
                d_model, vocab,
            ),
            other => return Err(RullamaError::Inference(format!(
                "backward_step: token_embd dtype {other:?} unsupported"
            ))),
        }

        // d_hidden (running, top-of-stack) = rmsnorm_back(self.hidden,
        // output_norm.weight, d_norm_x_final).
        rmsnorm_backward_chained(&self.ctx, &self.pipes, &mut enc,
            &self.hidden, &final_norm, scratch.d_hidden_final, scratch.d_hidden,
            d_model, eps, true);

        self.ctx.queue.submit(Some(enc.finish()));

        // ===== Walk layers top-down =====
        for li in (0..n_layers).rev() {
            let i = li as u32;
            let cap = &capture[li];
            let lora = &loras[li];
            let grad = &grads[li];
            let mut lenc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd.layer"),
            });
            self.backward_layer(&mut lenc, i, history_len, pos, cap, lora, grad, scratch).await?;
            self.ctx.queue.submit(Some(lenc.finish()));
        }

        // ===== Loss readback =====
        let loss_read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bwd.loss_read"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut renc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bwd.loss_copy"),
        });
        renc.copy_buffer_to_buffer(scratch.loss, 0, &loss_read, 0, 4);
        self.ctx.queue.submit(Some(renc.finish()));
        let loss_vec = read_back_f32(&self.ctx.device, &loss_read).await?;
        Ok(loss_vec[0])
    }

    /// Backward through one transformer layer. Reads `cap` (forward
    /// activations) and the live KV cache; writes LoRA gradients into
    /// `grad`; carries `d_hidden` running into the next-down layer.
    ///
    /// Skips PLE-injection backward (Gemma 4 e2b's PLE has no LoRA;
    /// the gradient leakage through `inp_gate_w` is dropped — an M0
    /// approximation, documented in `MIGRATION-REPORT.md`).
    #[allow(clippy::too_many_arguments)]
    async fn backward_layer<'a>(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        i: u32,
        history_len: u32,
        pos: u32,
        cap:  &LayerCaptureBuffers<'a>,
        lora: &LayerLoraSlots<'a>,
        grad: &LayerLoraGrads<'a>,
        scratch: &BackwardScratchView<'a>,
    ) -> Result<()> {
        let prefix = format!("blk.{i}.");
        let d_model = self.cfg.d_model as usize;
        let eps = self.cfg.rms_norm_eps;
        let n_heads = self.cfg.n_heads as usize;
        let n_kv_heads = self.cfg.n_kv_heads(i) as usize;
        let head_dim = self.cfg.head_dim(i) as usize;
        let ffn_n = self.cfg.ffn(i) as usize;
        let kind = self.cfg.kind(i);

        // Frozen weights this layer needs (cache hits after the forward).
        let wc = self.wcache.clone();
        let attn_norm_w  = wc.buffer_async(&format!("{prefix}attn_norm.weight")).await?;
        let post_attn_w  = wc.buffer_async(&format!("{prefix}post_attention_norm.weight")).await?;
        let mlp_norm_w   = wc.buffer_async(&format!("{prefix}ffn_norm.weight")).await?;
        let post_ffw_w   = wc.buffer_async(&format!("{prefix}post_ffw_norm.weight")).await?;
        let q_w          = wc.buffer_async(&format!("{prefix}attn_q.weight")).await?;
        let q_norm_w     = wc.buffer_async(&format!("{prefix}attn_q_norm.weight")).await?;
        let o_w          = wc.buffer_async(&format!("{prefix}attn_output.weight")).await?;
        let k_w          = wc.buffer_async(&format!("{prefix}attn_k.weight")).await?;
        let k_norm_w     = wc.buffer_async(&format!("{prefix}attn_k_norm.weight")).await?;
        let v_name       = format!("{prefix}attn_v.weight");
        let v_w          = wc.buffer_async(&v_name).await?;
        let v_w_dtype    = wc.dtype(&v_name)?;
        let gate_w       = wc.buffer_async(&format!("{prefix}ffn_gate.weight")).await?;
        let up_w         = wc.buffer_async(&format!("{prefix}ffn_up.weight")).await?;
        let down_name    = format!("{prefix}ffn_down.weight");
        let down_w       = wc.buffer_async(&down_name).await?;
        let down_dtype   = wc.dtype(&down_name)?;
        let factors_w    = if matches!(kind, LayerKind::Global) {
            wc.buffer_opt_async("rope_freqs.weight").await?
        } else { None };

        // Undo per-layer output scale.
        if let Some(s) = self.layer_scalars[i as usize] {
            scale_chained(&self.ctx, &self.pipes, enc, scratch.d_hidden, d_model, s);
        }

        // PLE injection backward: skipped for M0 (no LoRA on PLE; the
        // gradient leakage through inp_gate_w is dropped). residual_add
        // backward passes d_hidden through unchanged.

        // ----- FFN block backward -----
        // residual_add backward (ffn): d_norm_y_ffn = d_hidden (alias).
        // d_hidden continues as d_pre_ffn_residual (= d_h1 path through residual).
        //
        // post_ffw_norm rmsnorm backward → d_ffn_out into d_hidden_tmp.
        rmsnorm_backward_chained(&self.ctx, &self.pipes, enc,
            cap.ffn_out, &post_ffw_w, scratch.d_hidden, scratch.d_hidden_tmp,
            d_model, eps, true);

        // ffn_down matmul backward: d_ffn_act = down_wᵀ · d_ffn_out → d_ffn_a.
        match down_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx, &self.pipes, enc,
                &down_w, scratch.d_hidden_tmp, scratch.d_ffn_a, ffn_n, d_model,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx, &self.pipes, enc,
                &down_w, scratch.d_hidden_tmp, scratch.d_ffn_a, ffn_n, d_model,
            ),
            other => return Err(RullamaError::Inference(format!("ffn_down dtype {other:?} unsupported in backward"))),
        }

        // geglu backward → d_ffn_gate (d_ffn_b), d_ffn_up (d_ffn_c).
        geglu_backward_chained(&self.ctx, &self.pipes, enc,
            cap.ffn_gate, cap.ffn_up, scratch.d_ffn_a,
            scratch.d_ffn_b, scratch.d_ffn_c, ffn_n);

        // gate matmul backward: d_norm_x_ffn_via_gate = gate_wᵀ · d_ffn_gate → d_hidden_tmp.
        matmul_q4_k_backward_input_chained(&self.ctx, &self.pipes, enc,
            &gate_w, scratch.d_ffn_b, scratch.d_hidden_tmp, d_model, ffn_n);
        // up matmul backward: d_norm_x_ffn_via_up → d_hidden_tmp2.
        matmul_q4_k_backward_input_chained(&self.ctx, &self.pipes, enc,
            &up_w, scratch.d_ffn_c, scratch.d_hidden_tmp2, d_model, ffn_n);
        // d_hidden_tmp += d_hidden_tmp2 (full d_norm_x_ffn).
        residual_add_chained(&self.ctx, &self.pipes, enc,
            scratch.d_hidden_tmp, scratch.d_hidden_tmp2, d_model);

        // mlp_norm rmsnorm backward → d_pre_ffn_rms into d_hidden_tmp2.
        rmsnorm_backward_chained(&self.ctx, &self.pipes, enc,
            cap.pre_ffn_rms, &mlp_norm_w, scratch.d_hidden_tmp, scratch.d_hidden_tmp2,
            d_model, eps, true);
        // Accumulate FFN block branch contribution into running d_hidden.
        residual_add_chained(&self.ctx, &self.pipes, enc,
            scratch.d_hidden, scratch.d_hidden_tmp2, d_model);

        // ----- Attention block backward -----
        // residual_add backward (attn): d_norm_y_attn = d_hidden (alias).
        //
        // post_attn_norm rmsnorm backward → d_attn_proj into d_hidden_tmp.
        rmsnorm_backward_chained(&self.ctx, &self.pipes, enc,
            cap.attn_proj, &post_attn_w, scratch.d_hidden, scratch.d_hidden_tmp,
            d_model, eps, true);

        // o_proj matmul backward: d_attn_out = o_wᵀ · d_attn_proj → scratch.d_attn_out.
        matmul_q4_k_backward_input_chained(&self.ctx, &self.pipes, enc,
            &o_w, scratch.d_hidden_tmp, scratch.d_attn_out,
            n_heads * head_dim, d_model);

        // o LoRA backward: dB += scale·dy⊗z; u=Bᵀ·dy; d_attn_out += scale·Aᵀ·u; dA += scale·u⊗x.
        if let (Some(o_lora), Some(o_grad)) = (lora.o.as_ref(), grad.o.as_ref()) {
            let r = o_lora.rank as usize;
            let s = o_lora.scale;
            // dB_o += s · d_attn_proj ⊗ z_o  (using captured z from forward).
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                scratch.d_hidden_tmp, o_lora.z, o_grad.d_b,
                d_model, r, s, true);
            // u_o = B_oᵀ · d_attn_proj → o_lora.z (overwrite).
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                o_lora.b, scratch.d_hidden_tmp, o_lora.z,
                d_model, r, 1.0, false);
            // d_attn_out += s · A_oᵀ · u_o.
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                o_lora.a, o_lora.z, scratch.d_attn_out,
                r, n_heads * head_dim, s, true);
            // dA_o += s · u_o ⊗ attn_out (= cap.attn_out).
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                o_lora.z, cap.attn_out, o_grad.d_a,
                r, n_heads * head_dim, s, true);
        }

        // Recompute attention probs (from q_post_rope + kv cache) into scratch.attn_probs.
        let window = if matches!(kind, LayerKind::SlidingWindow) {
            self.cfg.sliding_window as usize
        } else { 0 };
        attention_probs_chained(&self.ctx, &self.pipes, enc,
            cap.q_post_rope, &self.kv_k[i as usize], scratch.attn_probs,
            head_dim, n_heads, n_kv_heads,
            pos as usize, history_len as usize, window);

        // Attn backward pass 1: d_q + d_scores (staged).
        attention_backward_dq_chained(&self.ctx, &self.pipes, enc,
            &self.kv_k[i as usize], &self.kv_v[i as usize],
            scratch.attn_probs, scratch.d_attn_out,
            scratch.attn_d_scores, scratch.d_q,
            head_dim, n_heads, n_kv_heads, history_len as usize);
        // Attn backward pass 2: d_k_hist, d_v_hist.
        attention_backward_dkv_chained(&self.ctx, &self.pipes, enc,
            cap.q_post_rope, scratch.attn_probs, scratch.d_attn_out,
            scratch.attn_d_scores, scratch.d_k_hist, scratch.d_v_hist,
            head_dim, n_heads, n_kv_heads, history_len as usize);

        // rope backward of q (in-place into d_q → now d_q_pre_rope's value).
        let (rope_base, rope_dims) = match kind {
            LayerKind::SlidingWindow => (self.cfg.rope_freq_base_swa, self.cfg.rope_dim_swa as usize),
            LayerKind::Global        => (self.cfg.rope_freq_base,     self.cfg.rope_dim_global as usize),
        };
        rope_neox_backward_chained(&self.ctx, &self.pipes, enc,
            scratch.d_q, factors_w.as_ref(), &self.dummy,
            head_dim, n_heads, pos as usize, rope_dims, rope_base);
        // q_norm rmsnorm backward → d_q_pre_norm.
        rmsnorm_per_row_backward_chained(&self.ctx, &self.pipes, enc,
            cap.q_pre_norm, &q_norm_w, scratch.d_q, scratch.d_q_pre_norm,
            n_heads, head_dim, eps, true);
        // q matmul backward: d_norm_x_attn_via_q → d_hidden_tmp (overwrites d_attn_proj).
        matmul_q4_k_backward_input_chained(&self.ctx, &self.pipes, enc,
            &q_w, scratch.d_q_pre_norm, scratch.d_hidden_tmp,
            d_model, n_heads * head_dim);
        // q LoRA backward.
        if let (Some(q_lora), Some(q_grad)) = (lora.q.as_ref(), grad.q.as_ref()) {
            let r = q_lora.rank as usize;
            let s = q_lora.scale;
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                scratch.d_q_pre_norm, q_lora.z, q_grad.d_b,
                n_heads * head_dim, r, s, true);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                q_lora.b, scratch.d_q_pre_norm, q_lora.z,
                n_heads * head_dim, r, 1.0, false);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                q_lora.a, q_lora.z, scratch.d_hidden_tmp,
                r, d_model, s, true);
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                q_lora.z, cap.norm_x_attn, q_grad.d_a,
                r, d_model, s, true);
        }

        // K backward — pull d_k at the final position from d_k_hist.
        // For M0 we only consume the final-position slice (history positions
        // before `pos` get zero LoRA grad contribution — see plan).
        let row_bytes = (n_kv_heads * head_dim * 4) as u64;
        let dk_final_off = pos as u64 * row_bytes;
        enc.copy_buffer_to_buffer(scratch.d_k_hist, dk_final_off,
            scratch.d_k_pre_rope, 0, row_bytes);
        rope_neox_backward_chained(&self.ctx, &self.pipes, enc,
            scratch.d_k_pre_rope, factors_w.as_ref(), &self.dummy,
            head_dim, n_kv_heads, pos as usize, rope_dims, rope_base);
        rmsnorm_per_row_backward_chained(&self.ctx, &self.pipes, enc,
            cap.k_pre_norm, &k_norm_w, scratch.d_k_pre_rope, scratch.d_k_pre_norm,
            n_kv_heads, head_dim, eps, true);
        // d_norm_x_attn_via_k → d_hidden_tmp2.
        matmul_q4_k_backward_input_chained(&self.ctx, &self.pipes, enc,
            &k_w, scratch.d_k_pre_norm, scratch.d_hidden_tmp2,
            d_model, n_kv_heads * head_dim);
        residual_add_chained(&self.ctx, &self.pipes, enc,
            scratch.d_hidden_tmp, scratch.d_hidden_tmp2, d_model);
        if let (Some(k_lora), Some(k_grad)) = (lora.k.as_ref(), grad.k.as_ref()) {
            let r = k_lora.rank as usize;
            let s = k_lora.scale;
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                scratch.d_k_pre_norm, k_lora.z, k_grad.d_b,
                n_kv_heads * head_dim, r, s, true);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                k_lora.b, scratch.d_k_pre_norm, k_lora.z,
                n_kv_heads * head_dim, r, 1.0, false);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                k_lora.a, k_lora.z, scratch.d_hidden_tmp,
                r, d_model, s, true);
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                k_lora.z, cap.norm_x_attn, k_grad.d_a,
                r, d_model, s, true);
        }

        // V backward — pull d_v at the final position from d_v_hist into
        // d_k_pre_norm (free at this point — k backward is done) so it
        // can serve as the rmsnorm_back `dy` without aliasing the `dx`
        // output buffer.
        enc.copy_buffer_to_buffer(scratch.d_v_hist, dk_final_off,
            scratch.d_k_pre_norm, 0, row_bytes);
        // V was passed through unweighted rmsnorm_per_row; do the unweighted backward.
        rmsnorm_per_row_backward_chained(&self.ctx, &self.pipes, enc,
            cap.v_pre_norm, &self.dummy, scratch.d_k_pre_norm, scratch.d_v_pre_norm,
            n_kv_heads, head_dim, eps, false);
        // d_norm_x_attn_via_v → d_hidden_tmp2.
        match v_w_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx, &self.pipes, enc,
                &v_w, scratch.d_v_pre_norm, scratch.d_hidden_tmp2,
                d_model, n_kv_heads * head_dim,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx, &self.pipes, enc,
                &v_w, scratch.d_v_pre_norm, scratch.d_hidden_tmp2,
                d_model, n_kv_heads * head_dim,
            ),
            other => return Err(RullamaError::Inference(format!("attn_v dtype {other:?} unsupported in backward"))),
        }
        residual_add_chained(&self.ctx, &self.pipes, enc,
            scratch.d_hidden_tmp, scratch.d_hidden_tmp2, d_model);
        if let (Some(v_lora), Some(v_grad)) = (lora.v.as_ref(), grad.v.as_ref()) {
            let r = v_lora.rank as usize;
            let s = v_lora.scale;
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                scratch.d_v_pre_norm, v_lora.z, v_grad.d_b,
                n_kv_heads * head_dim, r, s, true);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                v_lora.b, scratch.d_v_pre_norm, v_lora.z,
                n_kv_heads * head_dim, r, 1.0, false);
            lora_matmul_col_chained(&self.ctx, &self.pipes, enc,
                v_lora.a, v_lora.z, scratch.d_hidden_tmp,
                r, d_model, s, true);
            lora_outer_add_chained(&self.ctx, &self.pipes, enc,
                v_lora.z, cap.norm_x_attn, v_grad.d_a,
                r, d_model, s, true);
        }

        // attn_norm rmsnorm backward — flows the attn block contribution
        // into d_hidden_tmp2, then accumulates into running d_hidden.
        rmsnorm_backward_chained(&self.ctx, &self.pipes, enc,
            cap.hidden_in, &attn_norm_w, scratch.d_hidden_tmp, scratch.d_hidden_tmp2,
            d_model, eps, true);
        residual_add_chained(&self.ctx, &self.pipes, enc,
            scratch.d_hidden, scratch.d_hidden_tmp2, d_model);

        Ok(())
    }
}
