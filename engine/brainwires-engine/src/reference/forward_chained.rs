// Per-token / per-layer dispatchers take many dims (d_model, n_heads,
// head_dim, ffn, n_kv, eps, ...) — they mirror the Go reference and
// bundling them into a struct adds boilerplate without clarity.
#![allow(clippy::too_many_arguments)]

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
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::dispatch::{
    attention_backward_dkv_chained, attention_backward_dq_chained, attention_chained,
    attention_probs_chained, cross_entropy_backward_chained, geglu_backward_chained, geglu_chained,
    lora_matmul_col_chained, lora_matmul_row_chained, lora_outer_add_chained, make_dummy_storage,
    matmul_q4_k_backward_input_chained, matmul_q4_k_chained, matmul_q6_k_backward_input_chained,
    matmul_q6_k_chained, residual_add_chained, rmsnorm_backward_chained, rmsnorm_chained,
    rmsnorm_per_row_backward_chained, rmsnorm_per_row_chained, rope_neox_backward_chained,
    rope_neox_chained, scale_chained, softcap_chained,
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
    pub hidden_in: &'a wgpu::Buffer,
    /// Output of attn rmsnorm ([d_model]).
    pub norm_x_attn: &'a wgpu::Buffer,
    /// q matmul output before q_norm rmsnorm ([n_heads · head_dim]).
    pub q_pre_norm: &'a wgpu::Buffer,
    /// q after q_norm rmsnorm AND RoPE ([n_heads · head_dim]).
    pub q_post_rope: &'a wgpu::Buffer,
    /// k matmul output before k_norm rmsnorm ([n_kv · head_dim]).
    pub k_pre_norm: &'a wgpu::Buffer,
    /// v matmul output before v_norm rmsnorm ([n_kv · head_dim]).
    pub v_pre_norm: &'a wgpu::Buffer,
    /// Attention output, input to o_proj ([n_heads · head_dim]).
    pub attn_out: &'a wgpu::Buffer,
    /// o_proj matmul output, input to post_attn_norm rmsnorm ([d_model]).
    pub attn_proj: &'a wgpu::Buffer,
    /// `self.hidden` after the attn residual add ([d_model]).
    pub pre_ffn_rms: &'a wgpu::Buffer,
    /// Output of ffn rmsnorm ([d_model]).
    pub norm_x_ffn: &'a wgpu::Buffer,
    /// Gate matmul output ([ffn_inter]).
    pub ffn_gate: &'a wgpu::Buffer,
    /// Up matmul output ([ffn_inter]).
    pub ffn_up: &'a wgpu::Buffer,
    /// GEGLU output, input to ffn_down ([ffn_inter]).
    pub ffn_act: &'a wgpu::Buffer,
    /// ffn_down matmul output, input to post_ffw_norm rmsnorm ([d_model]).
    pub ffn_out: &'a wgpu::Buffer,
    /// PLE: `inp_gate_w · hidden` (input to PLE GEGLU's gate branch).
    /// Only written when `cfg.has_ple()`. `[ple_dim]`.
    pub ple_state: &'a wgpu::Buffer,
    /// PLE: output of GEGLU (input to `proj_w` matmul). `[ple_dim]`.
    pub ple_act: &'a wgpu::Buffer,
    /// PLE: output of `proj_w` matmul (input to PLE rmsnorm). `[d_model]`.
    pub ple_proj: &'a wgpu::Buffer,
}

/// One LoRA wrapper's GPU state — A, B, and a small `z` scratch that
/// the forward correction writes into and the backward reads from.
///
/// Forward: `y[out_dim] = W·x + scale · B · (A·x)`. The `z` buffer
/// holds `A·x` (size `[rank]`) after the forward correction so the
/// backward can build `dB = scale · dy ⊗ z`.
pub struct LoraSlot<'a> {
    pub a: &'a wgpu::Buffer, // [rank, in_dim]
    pub b: &'a wgpu::Buffer, // [out_dim, rank]
    pub z: &'a wgpu::Buffer, // [rank] scratch
    pub rank: u32,
    pub scale: f32, // alpha / rank
}

/// Per-layer LoRA slots for the four attention projections + three
/// FFN projections. Pass `None` for any projection that isn't
/// LoRA-wrapped.
pub struct LayerLoraSlots<'a> {
    pub q: Option<LoraSlot<'a>>,
    pub k: Option<LoraSlot<'a>>,
    pub v: Option<LoraSlot<'a>>,
    pub o: Option<LoraSlot<'a>>,
    pub ffn_gate: Option<LoraSlot<'a>>,
    pub ffn_up: Option<LoraSlot<'a>>,
    pub ffn_down: Option<LoraSlot<'a>>,
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
    norm_x: wgpu::Buffer, // d_model
    norm_y: wgpu::Buffer, // d_model
    q: wgpu::Buffer,      // n_heads * head_dim_max
    q_norm: wgpu::Buffer, // n_heads * head_dim_max (post-norm Q)
    k: wgpu::Buffer,      // n_kv_heads_max * head_dim_max
    k_norm: wgpu::Buffer,
    v: wgpu::Buffer,
    v_norm: wgpu::Buffer,
    attn_out_buf: wgpu::Buffer, // n_heads * head_dim_max
    attn_proj: wgpu::Buffer,    // d_model
    ffn_gate: wgpu::Buffer,     // ffn_inter_max
    ffn_up: wgpu::Buffer,
    ffn_act: wgpu::Buffer,
    ffn_out: wgpu::Buffer, // d_model

    // PLE prep (computed once per token, then sliced per-layer).
    per_layer_residual: wgpu::Buffer, // n_layers * ple_dim
    per_layer_proj: wgpu::Buffer,
    per_layer: wgpu::Buffer, // final per-layer inputs

    // PLE per-layer scratch.
    ple_state: wgpu::Buffer, // ple_dim
    ple_act: wgpu::Buffer,   // ple_dim
    ple_proj: wgpu::Buffer,  // d_model

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

    /// Cooperative cancel flag for in-flight forward + backward layer
    /// walks. The training cancel button flips this; the per-layer
    /// loops in `run_forward_from_hidden` and `backward_step` check it
    /// after each `encode_layer`. Bounded latency: one layer (~300 ms-
    /// 1 s on browser) instead of one full step (10-30 s). Mirrors
    /// the `Model::encode_cancel` pattern used for multimodal.
    cancel_flag: Arc<AtomicBool>,

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
        let q = alloc_storage("fwd.q", n_heads * head_dim_max);
        let q_norm = alloc_storage("fwd.q_norm", n_heads * head_dim_max);
        let k = alloc_storage("fwd.k", n_kv_heads_max * head_dim_max);
        let k_norm = alloc_storage("fwd.k_norm", n_kv_heads_max * head_dim_max);
        let v = alloc_storage("fwd.v", n_kv_heads_max * head_dim_max);
        let v_norm = alloc_storage("fwd.v_norm", n_kv_heads_max * head_dim_max);
        let attn_out_buf = alloc_storage("fwd.attn_out", n_heads * head_dim_max);
        let attn_proj = alloc_storage("fwd.attn_proj", d_model);
        let ffn_gate = alloc_storage("fwd.ffn_gate", ffn_inter_max);
        let ffn_up = alloc_storage("fwd.ffn_up", ffn_inter_max);
        let ffn_act = alloc_storage("fwd.ffn_act", ffn_inter_max);
        let ffn_out = alloc_storage("fwd.ffn_out", d_model);

        let per_layer_residual = alloc_storage("fwd.per_layer_residual", n_layers * ple_dim.max(1));
        let per_layer_proj = alloc_storage("fwd.per_layer_proj", n_layers * ple_dim.max(1));
        let per_layer = alloc_storage("fwd.per_layer", n_layers * ple_dim.max(1));

        let ple_state = alloc_storage("fwd.ple_state", ple_dim.max(1));
        let ple_act = alloc_storage("fwd.ple_act", ple_dim.max(1));
        let ple_proj = alloc_storage("fwd.ple_proj", d_model);

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
                let hd = cfg.head_dim(i as u32) as usize;
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
            cfg,
            ctx,
            pipes,
            wcache,
            weights,
            hidden,
            norm_x,
            norm_y,
            q,
            q_norm,
            k,
            k_norm,
            v,
            v_norm,
            attn_out_buf,
            attn_proj,
            ffn_gate,
            ffn_up,
            ffn_act,
            ffn_out,
            per_layer_residual,
            per_layer_proj,
            per_layer,
            ple_state,
            ple_act,
            ple_proj,
            logits_tile,
            logits,
            logits_read,
            kv_k,
            kv_v,
            kv_lens,
            donor_map,
            layer_scalars,
            dummy,
            max_context,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            pos: 0,
        })
    }

    /// Flip the cooperative cancel flag. Any in-flight forward or
    /// backward layer walk bails with `RullamaError::Cancelled` at
    /// the next layer boundary. Safe to call when no work is
    /// in-flight — the flag is cleared at the top of each `step` /
    /// `step_with_lora*` / `backward_step` call.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Release);
    }

    /// Clear the cancel flag. Called at the top of each layer-walking
    /// entry point so a stale flag from a previous cancel doesn't
    /// poison the next step.
    fn reset_cancel(&self) {
        self.cancel_flag.store(false, Ordering::Release);
    }

    /// Check the cancel flag — returns `Err(Cancelled)` if it's set.
    /// Called between per-layer encoder submits.
    fn check_cancelled(&self) -> Result<()> {
        if self.cancel_flag.load(Ordering::Acquire) {
            Err(RullamaError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Shared cancel-flag handle so `TrainingSession::cancel` can
    /// reach the flag without taking a `&mut` borrow on the model.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel_flag.clone()
    }

    pub fn cfg(&self) -> &Gemma4Config {
        &self.cfg
    }
    pub fn pos(&self) -> u32 {
        self.pos
    }
    /// Borrow the shared GPU weight cache. Exposed so `Model` can evict
    /// multimodal tower weights between turns.
    pub fn wcache(&self) -> &Arc<WeightCache> {
        &self.wcache
    }
    /// Borrow the GPU context (`WgpuCtx` is internally `Arc`-backed and
    /// cheap to clone). Used by `rullama-finetune` to allocate LoRA and
    /// scratch buffers on the same device + queue as the model.
    pub fn ctx(&self) -> &WgpuCtx {
        &self.ctx
    }
    /// Borrow the pipeline cache. The training crate doesn't need this
    /// directly (the backward path goes through `Forward::backward_step`),
    /// but exposing it keeps the surface symmetric for future test code.
    pub fn pipes(&self) -> &std::sync::Arc<Pipelines> {
        &self.pipes
    }
    /// Read-only handle on the model's logits buffer (post-forward).
    /// `TrainingSession::step` uses this to feed
    /// `cross_entropy_backward` without exposing the rest of Forward's
    /// scratch.
    pub fn logits_buffer(&self) -> &wgpu::Buffer {
        &self.logits
    }

    /// Access the running `hidden` residual buffer. Exposed for the
    /// training crate's single-forward PerPosition orchestrator,
    /// which captures `self.hidden` (= pre-final-norm) per position.
    pub fn hidden_buffer(&self) -> &wgpu::Buffer {
        &self.hidden
    }

    /// Run final rmsnorm + the tiled output projection (no
    /// softcap) over the current `self.hidden`, leaving the result
    /// in `self.logits`. Used by the single-forward PerPosition
    /// backward to compute logits at any captured pre-final-norm
    /// position without re-running the layer stack.
    pub async fn run_final_norm_and_output_proj_only(&mut self) -> Result<()> {
        let d_model = self.cfg.d_model as usize;
        let eps = self.cfg.rms_norm_eps;
        let wc = self.wcache.clone();
        let final_norm = wc.buffer_async("output_norm.weight").await?;
        let token_embd_dtype = wc.dtype("token_embd.weight")?;

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd.out_proj_only"),
            });
        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            &self.hidden,
            Some(&final_norm),
            &self.dummy,
            &self.norm_x,
            d_model,
            eps,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // Tiled output projection — same MAX_TILE_BYTES discipline as
        // the in-line one in `run_forward_from_hidden`.
        const MAX_TILE_BYTES: usize = 8 * 1024 * 1024;
        let tiles = wc
            .buffer_tiles_async("token_embd.weight", MAX_TILE_BYTES)
            .await?;
        for tile in &tiles {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.out_proj_only.tile"),
                });
            run_matmul_into_buf(
                &self.ctx,
                &self.pipes,
                &mut enc,
                token_embd_dtype,
                &tile.buffer,
                &self.norm_x,
                &self.logits_tile,
                tile.n_rows,
                d_model,
                "fwd.out_proj_only_tile",
            )?;
            enc.copy_buffer_to_buffer(
                &self.logits_tile,
                0,
                &self.logits,
                (tile.row_start as u64) * 4,
                (tile.n_rows as u64) * 4,
            );
            self.ctx.queue.submit(Some(enc.finish()));
        }
        Ok(())
    }

    /// Overwrite `self.hidden` from a slice of `src` at byte offset
    /// `src_offset`. Used by the single-forward PerPosition
    /// orchestrator to point the final-norm + output proj at a
    /// previously captured per-position pre-final-norm slice.
    pub fn set_hidden_from(&self, src: &wgpu::Buffer, src_offset: u64) {
        let d_model = self.cfg.d_model as usize;
        let bytes = (d_model as u64) * 4;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd.set_hidden_from"),
            });
        enc.copy_buffer_to_buffer(src, src_offset, &self.hidden, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        for l in self.kv_lens.iter_mut() {
            *l = 0;
        }
    }

    /// Hash of the per-layer KV geometry. Used to refuse a `load_kv` from a
    /// snapshot taken under a different model architecture (e.g. user
    /// switched gemma4 variants between sessions).
    fn kv_layout_hash(&self) -> u32 {
        let mut h: u32 = 0x811C9DC5; // FNV-1a offset basis
        for i in 0..self.cfg.n_layers {
            let nkv = self.cfg.n_kv_heads(i);
            let hd = self.cfg.head_dim(i);
            for byte in nkv.to_le_bytes().iter().chain(hd.to_le_bytes().iter()) {
                h ^= *byte as u32;
                h = h.wrapping_mul(0x01000193);
            }
        }
        h
    }

    /// Snapshot the KV cache + position counter into a versioned byte blob
    /// for suspend/resume. Format (little-endian):
    ///
    /// ```text
    ///   [0..4]    magic = "RLKV"
    ///   [4]       version = 1
    ///   [5]       n_owned_layers (u8) — non-donor layer count
    ///   [6..8]    reserved
    ///   [8..12]   position (u32) — Forward.pos at snapshot time
    ///   [12..16]  layout_hash (u32)
    ///   per owned layer (12 bytes each):
    ///     layer_idx  (u32)
    ///     kv_len     (u32) — tokens, not bytes
    ///     n_kv_heads (u16)
    ///     head_dim   (u16)
    ///   raw payload, same order as headers:
    ///     K bytes [kv_len * n_kv_heads * head_dim * 4]
    ///     V bytes [same]
    /// ```
    ///
    /// Donor layers carry no separate data — on `load_kv` they pick up the
    /// donor's KV via their shared Arc. `kv_lens` is per-layer; donor /
    /// dependent layers' counters stay at 0 by construction.
    pub async fn dump_kv(&self) -> Result<Vec<u8>> {
        let n_layers = self.cfg.n_layers as usize;

        struct Section {
            layer_idx: u32,
            kv_len: u32,
            n_kv_heads: u16,
            head_dim: u16,
            bytes: u64,
        }
        let mut sections: Vec<Section> = Vec::new();
        let mut total_payload: u64 = 0;
        for i in 0..n_layers {
            if self.donor_map[i].is_some() {
                continue;
            }
            let kv_len = self.kv_lens[i];
            if kv_len == 0 {
                continue;
            }
            let nkv = self.cfg.n_kv_heads(i as u32);
            let hd = self.cfg.head_dim(i as u32);
            let bytes = (kv_len as u64) * (nkv as u64) * (hd as u64) * 4;
            sections.push(Section {
                layer_idx: i as u32,
                kv_len,
                n_kv_heads: nkv as u16,
                head_dim: hd as u16,
                bytes,
            });
            total_payload += bytes * 2; // K + V
        }

        let mut header = Vec::<u8>::with_capacity(16 + 12 * sections.len());
        header.extend_from_slice(b"RLKV");
        header.push(1u8);
        header.push(sections.len() as u8);
        header.extend_from_slice(&[0u8, 0u8]);
        header.extend_from_slice(&self.pos.to_le_bytes());
        header.extend_from_slice(&self.kv_layout_hash().to_le_bytes());
        for s in &sections {
            header.extend_from_slice(&s.layer_idx.to_le_bytes());
            header.extend_from_slice(&s.kv_len.to_le_bytes());
            header.extend_from_slice(&s.n_kv_heads.to_le_bytes());
            header.extend_from_slice(&s.head_dim.to_le_bytes());
        }

        if total_payload == 0 {
            return Ok(header);
        }

        // One staging buffer + one encoder for all K/V copies — minimizes
        // submission overhead on the suspension-warning hot path.
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd.kv_dump.staging"),
            size: total_payload,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd.kv_dump.enc"),
            });
        let mut offset: u64 = 0;
        for s in &sections {
            let i = s.layer_idx as usize;
            enc.copy_buffer_to_buffer(&self.kv_k[i], 0, &staging, offset, s.bytes);
            offset += s.bytes;
            enc.copy_buffer_to_buffer(&self.kv_v[i], 0, &staging, offset, s.bytes);
            offset += s.bytes;
        }
        self.ctx.queue.submit(Some(enc.finish()));
        let payload = read_back_bytes(&self.ctx.device, &staging).await?;

        let mut out = header;
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Inverse of [`dump_kv`]. Validates the header (magic, version,
    /// layout_hash), uploads payload bytes back into the existing
    /// pre-allocated K/V buffers, and restores `pos` + `kv_lens`.
    ///
    /// Returns an error (without mutating self) if the snapshot is from a
    /// different model architecture, the format is unknown, or the byte
    /// count doesn't match the headers.
    pub fn load_kv(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 16 {
            return Err(RullamaError::Inference(format!(
                "kv snapshot too short: {} bytes",
                bytes.len()
            )));
        }
        if &bytes[0..4] != b"RLKV" {
            return Err(RullamaError::Inference("kv snapshot: bad magic".into()));
        }
        let version = bytes[4];
        if version != 1 {
            return Err(RullamaError::Inference(format!(
                "kv snapshot: unknown version {version}"
            )));
        }
        let n_owned = bytes[5] as usize;
        let position = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let layout_hash = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let expected_hash = self.kv_layout_hash();
        if layout_hash != expected_hash {
            return Err(RullamaError::Inference(format!(
                "kv snapshot: layout_hash mismatch (snapshot=0x{layout_hash:08X}, model=0x{expected_hash:08X})"
            )));
        }
        let header_size = 16 + 12 * n_owned;
        if bytes.len() < header_size {
            return Err(RullamaError::Inference(
                "kv snapshot: truncated header".into(),
            ));
        }
        if position > self.max_context {
            return Err(RullamaError::Inference(format!(
                "kv snapshot: position {position} exceeds max_context {}",
                self.max_context
            )));
        }

        struct Section {
            layer_idx: u32,
            kv_len: u32,
            bytes: u64,
        }
        let mut sections: Vec<Section> = Vec::with_capacity(n_owned);
        let mut total_payload: u64 = 0;
        for s in 0..n_owned {
            let off = 16 + 12 * s;
            let layer_idx = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let kv_len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let nkv = u16::from_le_bytes(bytes[off + 8..off + 10].try_into().unwrap());
            let hd = u16::from_le_bytes(bytes[off + 10..off + 12].try_into().unwrap());

            if (layer_idx as usize) >= self.kv_lens.len() {
                return Err(RullamaError::Inference(format!(
                    "kv snapshot: layer_idx {layer_idx} out of range"
                )));
            }
            if self.donor_map[layer_idx as usize].is_some() {
                return Err(RullamaError::Inference(format!(
                    "kv snapshot: layer {layer_idx} marked as donor in current model but snapshot has data"
                )));
            }
            let exp_nkv = self.cfg.n_kv_heads(layer_idx) as u16;
            let exp_hd = self.cfg.head_dim(layer_idx) as u16;
            if nkv != exp_nkv || hd != exp_hd {
                return Err(RullamaError::Inference(format!(
                    "kv snapshot: layer {layer_idx} geometry mismatch \
                     (snapshot n_kv={nkv} hd={hd}, model n_kv={exp_nkv} hd={exp_hd})"
                )));
            }
            if kv_len > self.max_context {
                return Err(RullamaError::Inference(format!(
                    "kv snapshot: layer {layer_idx} kv_len {kv_len} exceeds max_context {}",
                    self.max_context
                )));
            }
            let layer_bytes = (kv_len as u64) * (nkv as u64) * (hd as u64) * 4;
            sections.push(Section {
                layer_idx,
                kv_len,
                bytes: layer_bytes,
            });
            total_payload += layer_bytes * 2;
        }
        let payload_off = header_size;
        if (bytes.len() as u64) < (payload_off as u64) + total_payload {
            return Err(RullamaError::Inference(format!(
                "kv snapshot: payload truncated (have {}, need {})",
                bytes.len() - payload_off,
                total_payload,
            )));
        }

        // Validation passed — commit. write_buffer is synchronous from the
        // caller's POV; the queue copies on submit.
        let queue = &self.ctx.queue;
        let mut off: usize = payload_off;
        for s in &sections {
            let i = s.layer_idx as usize;
            let n = s.bytes as usize;
            queue.write_buffer(&self.kv_k[i], 0, &bytes[off..off + n]);
            off += n;
            queue.write_buffer(&self.kv_v[i], 0, &bytes[off..off + n]);
            off += n;
            self.kv_lens[i] = s.kv_len;
        }
        // Clear non-owned layers (donor-dependents stay at 0; non-snapshot
        // owned layers reset to 0 so the model behaves like an empty cache
        // for them).
        for i in 0..self.kv_lens.len() {
            if self.donor_map[i].is_some() {
                continue;
            }
            if !sections.iter().any(|s| s.layer_idx as usize == i) {
                self.kv_lens[i] = 0;
            }
        }
        self.pos = position;
        Ok(())
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
                capture.len(),
                self.cfg.n_layers
            )));
        }
        if let Some(l) = loras
            && l.len() != self.cfg.n_layers as usize
        {
            return Err(RullamaError::Inference(format!(
                "step_capture: got {} lora slots, expected {}",
                l.len(),
                self.cfg.n_layers
            )));
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
                loras.len(),
                self.cfg.n_layers
            )));
        }
        self.step_inner(token_id, None, Some(loras)).await
    }

    /// Same as [`step_with_lora`] but ALSO captures the per-position
    /// seq-shaped activations (`norm_x_attn`, `k_pre_norm`,
    /// `v_pre_norm`) into the supplied capture buffers at offset
    /// `pos·per_position_size`. Used during training prefill so the
    /// per-history K/V LoRA backward can read each position's
    /// activations without re-running forward.
    ///
    /// The 11 non-seq captures (q*, attn_out, attn_proj, hidden_in,
    /// pre_ffn_rms, norm_x_ffn, ffn_*, ple_*) are STILL written by
    /// `encode_layer` at offset 0 — they get overwritten by every
    /// position. Only the seq captures are position-stable.
    pub async fn step_with_lora_seqcap<'a>(
        &mut self,
        token_id: u32,
        loras: &'a [LayerLoraSlots<'a>],
        capture: &'a [LayerCaptureBuffers<'a>],
    ) -> Result<Vec<f32>> {
        if loras.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_lora_seqcap: got {} lora slots, expected {}",
                loras.len(),
                self.cfg.n_layers
            )));
        }
        if capture.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_lora_seqcap: got {} captures, expected {}",
                capture.len(),
                self.cfg.n_layers
            )));
        }
        self.step_inner(token_id, Some(capture), Some(loras)).await
    }

    async fn step_inner<'a>(
        &mut self,
        token_id: u32,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        if (token_id as u64) >= self.cfg.vocab_size as u64 {
            return Err(RullamaError::Inference(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
            )));
        }
        if self.pos >= self.max_context {
            return Err(RullamaError::Inference(format!(
                "context length exceeded max_context={}",
                self.max_context
            )));
        }
        let d_model = self.cfg.d_model as usize;
        let ple_dim = self.cfg.ple_dim as usize;

        // ---- CPU-side per-token preamble: token embed + PLE input dequant + upload ----
        let mut hidden_cpu = self
            .weights
            .load_row_async("token_embd.weight", token_id as usize)
            .await?;
        let scale_factor = (d_model as f32).sqrt();
        for v in hidden_cpu.iter_mut() {
            *v *= scale_factor;
        }
        self.ctx
            .queue
            .write_buffer(&self.hidden, 0, bytemuck::cast_slice(&hidden_cpu));
        drop(hidden_cpu);

        if self.cfg.has_ple() {
            let mut ple_in = self
                .weights
                .load_row_async("per_layer_token_embd.weight", token_id as usize)
                .await?;
            let s = (ple_dim as f32).sqrt();
            for v in ple_in.iter_mut() {
                *v *= s;
            }
            self.ctx
                .queue
                .write_buffer(&self.per_layer_residual, 0, bytemuck::cast_slice(&ple_in));
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
        self.step_with_embedding_inner(embedding, None).await
    }

    /// Variant of [`step_with_embedding`] that applies a LoRA adapter
    /// to every layer's q/k/v/o (+ optional FFN) during the forward.
    /// Used by `Model::step_with_embedding_native` when an inference
    /// adapter is active — without this, image and audio soft-token
    /// steps would silently bypass the loaded adapter while pure-text
    /// steps respect it.
    pub async fn step_with_embedding_with_lora<'a>(
        &mut self,
        embedding: &[f32],
        loras: &'a [LayerLoraSlots<'a>],
    ) -> Result<Vec<f32>> {
        if loras.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_embedding_with_lora: got {} lora slots, expected {}",
                loras.len(),
                self.cfg.n_layers
            )));
        }
        self.step_with_embedding_inner(embedding, Some(loras)).await
    }

    async fn step_with_embedding_inner<'a>(
        &mut self,
        embedding: &[f32],
        loras: Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        let d_model = self.cfg.d_model as usize;
        if embedding.len() != d_model {
            return Err(RullamaError::Inference(format!(
                "step_with_embedding: got {} f32s, expected d_model = {d_model}",
                embedding.len(),
            )));
        }
        if self.pos >= self.max_context {
            return Err(RullamaError::Inference(format!(
                "context length exceeded max_context={}",
                self.max_context
            )));
        }
        // Direct upload — caller's embedding is the new hidden state.
        self.ctx
            .queue
            .write_buffer(&self.hidden, 0, bytemuck::cast_slice(embedding));

        // Zero out per_layer_residual for this step (no token id → no PLE lookup).
        if self.cfg.has_ple() {
            let n_layers = self.cfg.n_layers as usize;
            let zeros = vec![0f32; n_layers * self.cfg.ple_dim as usize];
            self.ctx
                .queue
                .write_buffer(&self.per_layer_residual, 0, bytemuck::cast_slice(&zeros));
        }

        self.run_forward_from_hidden(None, loras).await
    }

    /// Forward pass starting from `self.hidden` already populated. Shared by
    /// `step` (token-id path) and `step_with_embedding` (multimodal soft tokens).
    async fn run_forward_from_hidden<'a>(
        &mut self,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
    ) -> Result<Vec<f32>> {
        // Clear any stale cancel flag from a previous step so this
        // call starts fresh; the per-layer loop below checks it after
        // each `encode_layer`.
        self.reset_cancel();
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
                return Err(RullamaError::Inference(
                    "per_layer_model_proj expected Q4_K".into(),
                ));
            }
            let proj_w = wc.buffer_async("per_layer_model_proj.weight").await?;
            let proj_norm = wc.buffer_async("per_layer_proj_norm.weight").await?;
            (Some(proj_w), Some(proj_norm), n_layers * ple_dim)
        } else {
            (None, None, 0)
        };

        // ---- build the per-token CommandEncoder ----
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

            matmul_q4_k_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                proj_w,
                &self.hidden,
                &self.per_layer_proj,
                d_model,
                ple_proj_n,
            );
            scale_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &self.per_layer_proj,
                ple_proj_n,
                1.0 / (d_model as f32).sqrt(),
            );
            rmsnorm_per_row_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &self.per_layer_proj,
                Some(proj_norm_w),
                &self.dummy,
                &self.per_layer,
                n_layers,
                ple_dim,
                eps,
            );
            residual_add_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &self.per_layer,
                &self.per_layer_residual,
                ple_proj_n,
            );
            scale_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &self.per_layer,
                ple_proj_n,
                1.0 / 2.0_f32.sqrt(),
            );
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
            // Per-layer cancel check. Encoder submits are the natural
            // boundary because the GPU is idle between layers under
            // this submission strategy. Bounded latency: one layer
            // (~300 ms - 1 s on browser) instead of one full step.
            self.check_cancelled()?;
            enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.token_encoder.cont"),
                });
        }

        // ---- final norm (in-place into hidden via norm_y as scratch) ----
        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            &self.hidden,
            Some(&final_norm),
            &self.dummy,
            &self.norm_x,
            d_model,
            eps,
        );

        // Flush before the output projection — it's the second-largest concentration
        // of GPU work in the step (262K-row matmul against the embedding) and we
        // don't want it queued behind a still-encoding layer batch.
        self.ctx.queue.submit(Some(enc.finish()));
        enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        let tiles = wc
            .buffer_tiles_async("token_embd.weight", MAX_TILE_BYTES)
            .await?;
        for tile in &tiles {
            run_matmul_into_buf(
                &self.ctx,
                &self.pipes,
                &mut enc,
                token_embd_dtype,
                &tile.buffer,
                &self.norm_x,
                &self.logits_tile,
                tile.n_rows,
                d_model,
                "fwd.output_tile",
            )?;
            enc.copy_buffer_to_buffer(
                &self.logits_tile,
                0,
                &self.logits,
                (tile.row_start as u64) * 4,
                (tile.n_rows as u64) * 4,
            );
            self.ctx.queue.submit(Some(enc.finish()));
            enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.out_proj_encoder.cont"),
                });
        }

        // ---- softcap ----
        // Out-of-place: read from `logits`, write into `logits_tile`. wgpu
        // disallows binding the same buffer as both read-only and read-write
        // within one dispatch, so we can't softcap in-place.
        let final_src: &wgpu::Buffer = if self.cfg.final_logit_softcap > 0.0 {
            softcap_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &self.logits,
                &self.logits_tile,
                self.cfg.vocab_size as usize,
                self.cfg.final_logit_softcap,
            );
            &self.logits_tile
        } else {
            &self.logits
        };

        // ---- copy logits → readback buffer ----
        enc.copy_buffer_to_buffer(
            final_src,
            0,
            &self.logits_read,
            0,
            (self.cfg.vocab_size as u64) * 4,
        );

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
        loras: Option<&'a LayerLoraSlots<'a>>,
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
            enc.copy_buffer_to_buffer(
                &self.hidden,
                0,
                cap.hidden_in,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        // Pre-fetch all weights this layer needs (each is cached after first call).
        let attn_norm_w = self
            .wcache
            .buffer_async(&format!("{prefix}attn_norm.weight"))
            .await?;
        let post_attn_w = self
            .wcache
            .buffer_async(&format!("{prefix}post_attention_norm.weight"))
            .await?;
        let mlp_norm_w = self
            .wcache
            .buffer_async(&format!("{prefix}ffn_norm.weight"))
            .await?;
        let post_ffw_w = self
            .wcache
            .buffer_async(&format!("{prefix}post_ffw_norm.weight"))
            .await?;

        let q_w = self
            .wcache
            .buffer_async(&format!("{prefix}attn_q.weight"))
            .await?;
        let q_norm_w = self
            .wcache
            .buffer_async(&format!("{prefix}attn_q_norm.weight"))
            .await?;
        let o_w = self
            .wcache
            .buffer_async(&format!("{prefix}attn_output.weight"))
            .await?;

        let (k_w, k_norm_w, v_w, v_w_dtype) = if donor.is_none() {
            let kw = self
                .wcache
                .buffer_async(&format!("{prefix}attn_k.weight"))
                .await?;
            let knw = self
                .wcache
                .buffer_async(&format!("{prefix}attn_k_norm.weight"))
                .await?;
            let v_name = format!("{prefix}attn_v.weight");
            let vw = self.wcache.buffer_async(&v_name).await?;
            let dt = self.wcache.dtype(&v_name)?;
            (Some(kw), Some(knw), Some(vw), Some(dt))
        } else {
            (None, None, None, None)
        };

        let gate_w = self
            .wcache
            .buffer_async(&format!("{prefix}ffn_gate.weight"))
            .await?;
        let up_w = self
            .wcache
            .buffer_async(&format!("{prefix}ffn_up.weight"))
            .await?;
        let down_name = format!("{prefix}ffn_down.weight");
        let down_w = self.wcache.buffer_async(&down_name).await?;
        let down_dtype = self.wcache.dtype(&down_name)?;

        // PLE-injection weights (only when has_ple)
        let (inp_gate_w, proj_w, post_norm_w) = if self.cfg.has_ple() {
            let a = self
                .wcache
                .buffer_async(&format!("{prefix}inp_gate.weight"))
                .await?;
            let b = self
                .wcache
                .buffer_async(&format!("{prefix}proj.weight"))
                .await?;
            let c = self
                .wcache
                .buffer_async(&format!("{prefix}post_norm.weight"))
                .await?;
            (Some(a), Some(b), Some(c))
        } else {
            (None, None, None)
        };

        let factors_w = if matches!(kind, LayerKind::Global) {
            // Same RoPE factors tensor across global layers — would benefit from caching;
            // the cache key is the tensor name so it's already a single GPU buffer.
            self.wcache.buffer_opt_async("rope_freqs.weight").await?
        } else {
            None
        };

        // ===== ATTENTION =====
        // norm_x = rmsnorm(hidden, attn_norm)
        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.hidden,
            Some(&attn_norm_w),
            &self.dummy,
            &self.norm_x,
            d_model,
            eps,
        );

        // ---- CAPTURE: norm_x_attn (input to q/k/v matmul + LoRA) ----
        if let Some(cap) = capture {
            // Per-position seq capture: write at `pos·d_model` offset.
            enc.copy_buffer_to_buffer(
                &self.norm_x,
                0,
                cap.norm_x_attn,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        // Q/K/V projections from norm_x
        matmul_q4_k_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &q_w,
            &self.norm_x,
            &self.q,
            d_model,
            n_heads * head_dim,
        );

        // ---- LoRA forward correction (q): self.q += scale · B · (A · norm_x) ----
        if let Some(slot) = loras.and_then(|l| l.q.as_ref()) {
            // z = A · norm_x  ([rank] = [rank, d_model] @ [d_model])
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                &self.norm_x,
                slot.z,
                d_model,
                slot.rank as usize,
                1.0,
                false,
            );
            // self.q += scale · B · z  ([n_heads*head_dim] += [n_heads*head_dim, rank] @ [rank])
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.b,
                slot.z,
                &self.q,
                slot.rank as usize,
                n_heads * head_dim,
                slot.scale,
                true,
            );
        }

        // ---- CAPTURE: q_pre_norm (q matmul output, input to q_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.q,
                0,
                cap.q_pre_norm,
                (pos as u64) * (n_heads as u64) * (head_dim as u64) * 4,
                (n_heads * head_dim * 4) as u64,
            );
        }

        // per-head q_norm (weighted)
        rmsnorm_per_row_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.q,
            Some(&q_norm_w),
            &self.dummy,
            &self.q_norm,
            n_heads,
            head_dim,
            eps,
        );
        // RoPE in-place into q_norm
        let (rope_base, rope_dims) = match kind {
            LayerKind::SlidingWindow => {
                (self.cfg.rope_freq_base_swa, self.cfg.rope_dim_swa as usize)
            }
            LayerKind::Global => (self.cfg.rope_freq_base, self.cfg.rope_dim_global as usize),
        };
        rope_neox_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.q_norm,
            factors_w.as_ref(),
            &self.dummy,
            head_dim,
            n_heads,
            pos as usize,
            rope_dims,
            rope_base,
        );

        // ---- CAPTURE: q_post_rope (input to attention; reused in dkv pass) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.q_norm,
                0,
                cap.q_post_rope,
                (pos as u64) * (n_heads as u64) * (head_dim as u64) * 4,
                (n_heads * head_dim * 4) as u64,
            );
        }

        if donor.is_none() {
            let kw = k_w.as_ref().unwrap();
            let knw = k_norm_w.as_ref().unwrap();
            let vw = v_w.as_ref().unwrap();
            let vdt = v_w_dtype.unwrap();

            matmul_q4_k_chained(
                &self.ctx,
                &self.pipes,
                enc,
                kw,
                &self.norm_x,
                &self.k,
                d_model,
                n_kv_heads * head_dim,
            );

            // ---- LoRA forward correction (k) ----
            if let Some(slot) = loras.and_then(|l| l.k.as_ref()) {
                lora_matmul_row_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.a,
                    &self.norm_x,
                    slot.z,
                    d_model,
                    slot.rank as usize,
                    1.0,
                    false,
                );
                lora_matmul_row_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.b,
                    slot.z,
                    &self.k,
                    slot.rank as usize,
                    n_kv_heads * head_dim,
                    slot.scale,
                    true,
                );
            }

            // ---- CAPTURE: k_pre_norm (k matmul output, input to k_norm rmsnorm) ----
            if let Some(cap) = capture {
                // Per-position seq capture: write at `pos·(n_kv·head_dim)` offset.
                enc.copy_buffer_to_buffer(
                    &self.k,
                    0,
                    cap.k_pre_norm,
                    (pos as u64) * (n_kv_heads as u64) * (head_dim as u64) * 4,
                    (n_kv_heads * head_dim * 4) as u64,
                );
            }

            rmsnorm_per_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.k,
                Some(knw),
                &self.dummy,
                &self.k_norm,
                n_kv_heads,
                head_dim,
                eps,
            );
            rope_neox_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.k_norm,
                factors_w.as_ref(),
                &self.dummy,
                head_dim,
                n_kv_heads,
                pos as usize,
                rope_dims,
                rope_base,
            );

            match vdt {
                GgmlDtype::Q6_K => matmul_q6_k_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    vw,
                    &self.norm_x,
                    &self.v,
                    d_model,
                    n_kv_heads * head_dim,
                ),
                GgmlDtype::Q4_K => matmul_q4_k_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    vw,
                    &self.norm_x,
                    &self.v,
                    d_model,
                    n_kv_heads * head_dim,
                ),
                other => {
                    return Err(RullamaError::Inference(format!(
                        "attn_v dtype {other:?} unsupported"
                    )));
                }
            }

            // ---- LoRA forward correction (v) ----
            if let Some(slot) = loras.and_then(|l| l.v.as_ref()) {
                lora_matmul_row_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.a,
                    &self.norm_x,
                    slot.z,
                    d_model,
                    slot.rank as usize,
                    1.0,
                    false,
                );
                lora_matmul_row_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.b,
                    slot.z,
                    &self.v,
                    slot.rank as usize,
                    n_kv_heads * head_dim,
                    slot.scale,
                    true,
                );
            }

            // ---- CAPTURE: v_pre_norm (v matmul output, input to unweighted v_norm rmsnorm) ----
            if let Some(cap) = capture {
                // Per-position seq capture.
                enc.copy_buffer_to_buffer(
                    &self.v,
                    0,
                    cap.v_pre_norm,
                    (pos as u64) * (n_kv_heads as u64) * (head_dim as u64) * 4,
                    (n_kv_heads * head_dim * 4) as u64,
                );
            }

            // V-norm is unweighted
            rmsnorm_per_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.v,
                None,
                &self.dummy,
                &self.v_norm,
                n_kv_heads,
                head_dim,
                eps,
            );

            // Append rotated K + normed V into this layer's KV cache at offset = kv_lens[i].
            let row_bytes = (n_kv_heads * head_dim * 4) as u64;
            let dst_offset = self.kv_lens[i as usize] as u64 * row_bytes;
            enc.copy_buffer_to_buffer(
                &self.k_norm,
                0,
                &self.kv_k[i as usize],
                dst_offset,
                row_bytes,
            );
            enc.copy_buffer_to_buffer(
                &self.v_norm,
                0,
                &self.kv_v[i as usize],
                dst_offset,
                row_bytes,
            );
            self.kv_lens[i as usize] = self.kv_lens[i as usize].saturating_add(1);
        }

        // attention: kv buffers are kv_k[i], kv_v[i] (alias for donor); history_len from
        // donor's len if shared, else this layer's len (which we just incremented).
        let history_layer = donor.map(|d| d as usize).unwrap_or(i as usize);
        let history_len = self.kv_lens[history_layer] as usize;
        let window = if matches!(kind, LayerKind::SlidingWindow) {
            self.cfg.sliding_window as usize
        } else {
            0
        };

        attention_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.q_norm,
            &self.kv_k[i as usize],
            &self.kv_v[i as usize],
            &self.attn_out_buf,
            head_dim,
            n_heads,
            n_kv_heads,
            pos as usize,
            history_len,
            window,
        );

        // ---- CAPTURE: attn_out (input to o_proj) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.attn_out_buf,
                0,
                cap.attn_out,
                (pos as u64) * (n_heads as u64) * (head_dim as u64) * 4,
                (n_heads * head_dim * 4) as u64,
            );
        }

        // attn_proj = matmul(attn_out_buf, attn_output.weight)
        matmul_q4_k_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &o_w,
            &self.attn_out_buf,
            &self.attn_proj,
            n_heads * head_dim,
            d_model,
        );

        // ---- LoRA forward correction (o): self.attn_proj += scale · B · (A · attn_out_buf) ----
        if let Some(slot) = loras.and_then(|l| l.o.as_ref()) {
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                &self.attn_out_buf,
                slot.z,
                n_heads * head_dim,
                slot.rank as usize,
                1.0,
                false,
            );
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.b,
                slot.z,
                &self.attn_proj,
                slot.rank as usize,
                d_model,
                slot.scale,
                true,
            );
        }

        // ---- CAPTURE: attn_proj (input to post_attn_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.attn_proj,
                0,
                cap.attn_proj,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        // norm_y = rmsnorm(attn_proj, post_attn_norm.weight)
        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.attn_proj,
            Some(&post_attn_w),
            &self.dummy,
            &self.norm_y,
            d_model,
            eps,
        );
        // hidden += norm_y
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.hidden,
            &self.norm_y,
            d_model,
        );

        // ---- CAPTURE: pre_ffn_rms (hidden after attn residual add) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.hidden,
                0,
                cap.pre_ffn_rms,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        // ===== MLP =====
        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.hidden,
            Some(&mlp_norm_w),
            &self.dummy,
            &self.norm_x,
            d_model,
            eps,
        );

        // ---- CAPTURE: norm_x_ffn (input to gate/up matmul + LoRA) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.norm_x,
                0,
                cap.norm_x_ffn,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        matmul_q4_k_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &gate_w,
            &self.norm_x,
            &self.ffn_gate,
            d_model,
            ffn_n,
        );

        // ---- LoRA forward correction (ffn_gate): ffn_gate += scale · B · (A · norm_x) ----
        if let Some(slot) = loras.and_then(|l| l.ffn_gate.as_ref()) {
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                &self.norm_x,
                slot.z,
                d_model,
                slot.rank as usize,
                1.0,
                false,
            );
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.b,
                slot.z,
                &self.ffn_gate,
                slot.rank as usize,
                ffn_n,
                slot.scale,
                true,
            );
        }

        matmul_q4_k_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &up_w,
            &self.norm_x,
            &self.ffn_up,
            d_model,
            ffn_n,
        );

        // ---- LoRA forward correction (ffn_up): ffn_up += scale · B · (A · norm_x) ----
        if let Some(slot) = loras.and_then(|l| l.ffn_up.as_ref()) {
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                &self.norm_x,
                slot.z,
                d_model,
                slot.rank as usize,
                1.0,
                false,
            );
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.b,
                slot.z,
                &self.ffn_up,
                slot.rank as usize,
                ffn_n,
                slot.scale,
                true,
            );
        }

        // ---- CAPTURE: ffn_gate, ffn_up (inputs to GEGLU) ----
        if let Some(cap) = capture {
            let ffn_pos_off = (pos as u64) * (ffn_n as u64) * 4;
            enc.copy_buffer_to_buffer(
                &self.ffn_gate,
                0,
                cap.ffn_gate,
                ffn_pos_off,
                (ffn_n * 4) as u64,
            );
            enc.copy_buffer_to_buffer(&self.ffn_up, 0, cap.ffn_up, ffn_pos_off, (ffn_n * 4) as u64);
        }

        geglu_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_act,
            ffn_n,
        );

        // ---- CAPTURE: ffn_act (input to ffn_down matmul) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.ffn_act,
                0,
                cap.ffn_act,
                (pos as u64) * (ffn_n as u64) * 4,
                (ffn_n * 4) as u64,
            );
        }

        match down_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &down_w,
                &self.ffn_act,
                &self.ffn_out,
                ffn_n,
                d_model,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &down_w,
                &self.ffn_act,
                &self.ffn_out,
                ffn_n,
                d_model,
            ),
            other => {
                return Err(RullamaError::Inference(format!(
                    "ffn_down dtype {other:?} unsupported"
                )));
            }
        }

        // ---- LoRA forward correction (ffn_down): ffn_out += scale · B · (A · ffn_act) ----
        if let Some(slot) = loras.and_then(|l| l.ffn_down.as_ref()) {
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                &self.ffn_act,
                slot.z,
                ffn_n,
                slot.rank as usize,
                1.0,
                false,
            );
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.b,
                slot.z,
                &self.ffn_out,
                slot.rank as usize,
                d_model,
                slot.scale,
                true,
            );
        }

        // ---- CAPTURE: ffn_out (input to post_ffw_norm rmsnorm) ----
        if let Some(cap) = capture {
            enc.copy_buffer_to_buffer(
                &self.ffn_out,
                0,
                cap.ffn_out,
                (pos as u64) * (d_model as u64) * 4,
                (d_model * 4) as u64,
            );
        }

        rmsnorm_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.ffn_out,
            Some(&post_ffw_w),
            &self.dummy,
            &self.norm_y,
            d_model,
            eps,
        );
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.hidden,
            &self.norm_y,
            d_model,
        );

        // ===== PLE injection =====
        if self.cfg.has_ple() {
            let inp_gate_w = inp_gate_w.unwrap();
            let proj_w = proj_w.unwrap();
            let post_norm_w = post_norm_w.unwrap();
            let ple_dim = self.cfg.ple_dim as usize;

            // ple_state = matmul(hidden, inp_gate_w) [d_model -> ple_dim]
            matmul_q4_k_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &inp_gate_w,
                &self.hidden,
                &self.ple_state,
                d_model,
                ple_dim,
            );

            // ---- CAPTURE: ple_state (input gate branch to PLE GEGLU) ----
            if let Some(cap) = capture {
                enc.copy_buffer_to_buffer(
                    &self.ple_state,
                    0,
                    cap.ple_state,
                    (pos as u64) * (ple_dim as u64) * 4,
                    (ple_dim * 4) as u64,
                );
            }

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
            geglu_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.ple_state,
                &self.ple_proj,
                &self.ple_act,
                ple_dim,
            );

            // ---- CAPTURE: ple_act (input to proj_w matmul) ----
            if let Some(cap) = capture {
                enc.copy_buffer_to_buffer(
                    &self.ple_act,
                    0,
                    cap.ple_act,
                    (pos as u64) * (ple_dim as u64) * 4,
                    (ple_dim * 4) as u64,
                );
            }

            // projected = matmul(ple_act, proj_w) [ple_dim -> d_model]
            matmul_q4_k_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &proj_w,
                &self.ple_act,
                &self.ple_proj,
                ple_dim,
                d_model,
            );

            // ---- CAPTURE: ple_proj (input to PLE rmsnorm) ----
            if let Some(cap) = capture {
                enc.copy_buffer_to_buffer(
                    &self.ple_proj,
                    0,
                    cap.ple_proj,
                    (pos as u64) * (d_model as u64) * 4,
                    (d_model * 4) as u64,
                );
            }

            // norm_y = rmsnorm(ple_proj, post_norm_w)
            rmsnorm_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.ple_proj,
                Some(&post_norm_w),
                &self.dummy,
                &self.norm_y,
                d_model,
                eps,
            );
            // hidden += norm_y
            residual_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &self.hidden,
                &self.norm_y,
                d_model,
            );
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
struct MatmulParams {
    k: u32,
    n: u32,
    _p0: u32,
    _p1: u32,
}

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
        other => {
            return Err(RullamaError::Inference(format!(
                "output proj dtype {other:?} not supported"
            )));
        }
    };
    let params = MatmulParams {
        k: k as u32,
        n: n_rows as u32,
        _p0: 0,
        _p1: 0,
    };
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dst.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipeline);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n_rows as u32).div_ceil(64), 1, 1);
    Ok(())
}

async fn read_buf_stats(ctx: &WgpuCtx, buf: &wgpu::Buffer, n: usize) -> Result<(f32, usize)> {
    let bytes = (n * 4) as u64;
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trace.read"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trace.enc"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let v = read_back_f32(&ctx.device, &read_buf).await?;
    let mut max_abs = 0.0f32;
    let mut nans = 0usize;
    for &x in &v {
        if x.is_nan() {
            nans += 1;
        } else if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    Ok((max_abs, nans))
}

async fn read_back_f32(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<f32>> {
    let slice = buf.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device
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
    buf.unmap();
    Ok(v)
}

/// Same as [`read_back_f32`] but returns raw bytes — for snapshotting the
/// KV cache where we don't care about the f32 alignment, only the byte
/// stream.
async fn read_back_bytes(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<u8>> {
    let slice = buf.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device
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
    let v: Vec<u8> = data.to_vec();
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
    pub a: &'a wgpu::Buffer,   // [rank, in_dim] — read for u = Bᵀ·dy
    pub b: &'a wgpu::Buffer,   // [out_dim, rank] — read for u = Bᵀ·dy
    pub z: &'a wgpu::Buffer,   // [rank] — captured A·x from forward, dB needs it
    pub d_a: &'a wgpu::Buffer, // [rank, in_dim] — gradient accumulator
    pub d_b: &'a wgpu::Buffer, // [out_dim, rank] — gradient accumulator
    pub rank: u32,
    pub scale: f32,
}

/// Per-layer LoRA gradient accumulators for the four attention
/// projections + three FFN projections. Each pair drives both the
/// LoRA backward (computing dA, dB into d_a, d_b) AND the LoRA
/// contribution to dx (Aᵀ·Bᵀ·dy added to the running input gradient).
pub struct LayerLoraGrads<'a> {
    pub q: Option<LoraGradPair<'a>>,
    pub k: Option<LoraGradPair<'a>>,
    pub v: Option<LoraGradPair<'a>>,
    pub o: Option<LoraGradPair<'a>>,
    pub ffn_gate: Option<LoraGradPair<'a>>,
    pub ffn_up: Option<LoraGradPair<'a>>,
    pub ffn_down: Option<LoraGradPair<'a>>,
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
    /// `[ple_dim]` — d_gate output of PLE geglu_back.
    pub d_ple_state: &'a wgpu::Buffer,
    /// `[ple_dim]` — d input to PLE geglu_back (= proj_w matmul-back output).
    pub d_ple_act: &'a wgpu::Buffer,
    /// `[ple_dim]` — discarded `d_up` output of PLE geglu_back.
    pub d_ple_up_discard: &'a wgpu::Buffer,
    /// `[ple_dim]` — staging copy of `self.per_layer[i*ple_dim..]` for
    /// PLE geglu_back's read-only `up` input.
    pub ple_per_layer_tmp: &'a wgpu::Buffer,
    /// `[d_model]` window into a layer's seq-sized `norm_x_attn`
    /// capture — pre-copied per backward iteration.
    pub norm_x_attn_window: &'a wgpu::Buffer,
    /// `[n_kv · head_dim]` window into a layer's seq-sized
    /// `k_pre_norm` capture.
    pub k_pre_norm_window: &'a wgpu::Buffer,
    /// `[n_kv · head_dim]` window into a layer's seq-sized
    /// `v_pre_norm` capture.
    pub v_pre_norm_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `hidden_in` capture.
    pub hidden_in_window: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` window into `q_pre_norm` capture.
    pub q_pre_norm_window: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` window into `q_post_rope` capture.
    pub q_post_rope_window: &'a wgpu::Buffer,
    /// `[n_heads · head_dim]` window into `attn_out` capture.
    pub attn_out_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `attn_proj` capture.
    pub attn_proj_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `pre_ffn_rms` capture.
    pub pre_ffn_rms_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `norm_x_ffn` capture.
    pub norm_x_ffn_window: &'a wgpu::Buffer,
    /// `[ffn_inter]` window into `ffn_gate` capture.
    pub ffn_gate_window: &'a wgpu::Buffer,
    /// `[ffn_inter]` window into `ffn_up` capture.
    pub ffn_up_window: &'a wgpu::Buffer,
    /// `[ffn_inter]` window into `ffn_act` capture.
    pub ffn_act_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `ffn_out` capture.
    pub ffn_out_window: &'a wgpu::Buffer,
    /// `[ple_dim]` window into `ple_state` capture.
    pub ple_state_window: &'a wgpu::Buffer,
    /// `[ple_dim]` window into `ple_act` capture.
    pub ple_act_window: &'a wgpu::Buffer,
    /// `[d_model]` window into `ple_proj` capture.
    pub ple_proj_window: &'a wgpu::Buffer,
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
        loras: &'a [LayerLoraSlots<'a>],
        grads: &'a [LayerLoraGrads<'a>],
        scratch: &'a BackwardScratchView<'a>,
        history_len: u32,
        pos: u32,
        recompute_captures: bool,
    ) -> Result<f32> {
        // Clear any stale cancel flag from a previous step; the layer
        // walk below checks it after each `backward_layer` submit.
        self.reset_cancel();
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
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd.head"),
            });

        // d_logits + scalar loss
        cross_entropy_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            &self.logits,
            scratch.d_logits,
            scratch.loss,
            vocab,
            target_id,
        );

        // d_norm_x_final = embedᵀ @ d_logits → write into scratch.d_hidden_final
        match token_embd_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &token_embd,
                scratch.d_logits,
                scratch.d_hidden_final,
                d_model,
                vocab,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &token_embd,
                scratch.d_logits,
                scratch.d_hidden_final,
                d_model,
                vocab,
            ),
            other => {
                return Err(RullamaError::Inference(format!(
                    "backward_step: token_embd dtype {other:?} unsupported"
                )));
            }
        }

        // d_hidden (running, top-of-stack) = rmsnorm_back(self.hidden,
        // output_norm.weight, d_norm_x_final).
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            &self.hidden,
            &final_norm,
            scratch.d_hidden_final,
            scratch.d_hidden,
            d_model,
            eps,
            true,
        );

        self.ctx.queue.submit(Some(enc.finish()));

        let trace_hidden = std::env::var("RULLAMA_TRACE_DHIDDEN").is_ok();
        // Adaptive max-abs clip on d_hidden between layers. Defaults to
        // 1.0 to keep deep-network gradient flow finite for LoRA
        // fine-tuning of pretrained models, where the backward graph
        // (which the pretrained weights were *not* initialized for) can
        // amplify 100-1500x per layer. Adam normalises to ≈ lr · sign(g)
        // anyway, so absolute magnitude is mostly informational — but
        // preventing overflow is the bare minimum the optimiser needs.
        let clip_max: f32 = std::env::var("RULLAMA_CLIP_DHIDDEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        if trace_hidden {
            let (max_abs, nans) =
                read_buf_stats(&self.ctx, scratch.d_hidden, self.cfg.d_model as usize).await?;
            eprintln!("[trace] after head section: d_hidden max_abs={max_abs:.3e} nan={nans}");
            let (max_abs_f, nans_f) =
                read_buf_stats(&self.ctx, scratch.d_hidden_final, self.cfg.d_model as usize)
                    .await?;
            eprintln!("[trace] d_hidden_final (head): max_abs={max_abs_f:.3e} nan={nans_f}");
            let (max_abs_l, nans_l) =
                read_buf_stats(&self.ctx, scratch.d_logits, self.cfg.vocab_size as usize).await?;
            eprintln!("[trace] d_logits: max_abs={max_abs_l:.3e} nan={nans_l}");
        }
        // ===== Walk layers top-down =====
        let d_model_bytes = (self.cfg.d_model as u64) * 4;
        for li in (0..n_layers).rev() {
            let i = li as u32;
            let cap = &capture[li];
            let lora = &loras[li];
            let grad = &grads[li];

            // Gradient-checkpointing replay: rewrite the per-layer
            // captures by re-running this layer's forward pass.
            // Uses `cap.hidden_in` (saved at the top of the original
            // forward) as the input. The K/V cache write at slot
            // `pos` is idempotent (same value written again);
            // `kv_lens[i]` is save/restored so the cache-count
            // bookkeeping survives the replay.
            if recompute_captures {
                let mut renc =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("bwd.replay"),
                        });
                renc.copy_buffer_to_buffer(
                    cap.hidden_in,
                    (pos as u64) * d_model_bytes,
                    &self.hidden,
                    0,
                    d_model_bytes,
                );
                let saved_len = self.kv_lens[li];
                if self.donor_map[li].is_none() && saved_len > 0 {
                    self.kv_lens[li] = saved_len - 1;
                }
                self.encode_layer(&mut renc, i, pos, Some(cap), Some(lora))
                    .await?;
                // encode_layer's K/V write re-incremented kv_lens[i]; assert.
                debug_assert_eq!(
                    self.kv_lens[li], saved_len,
                    "replay should leave kv_lens unchanged for layer {li}"
                );
                self.ctx.queue.submit(Some(renc.finish()));
            }

            let mut lenc =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bwd.layer"),
                    });
            self.backward_layer(&mut lenc, i, history_len, pos, cap, lora, grad, scratch)
                .await?;
            self.ctx.queue.submit(Some(lenc.finish()));
            // Per-layer cancel check — same boundary the forward loop
            // uses. Cancellation latency is bounded by one
            // `backward_layer` (~300 ms - 1 s on browser).
            self.check_cancelled()?;

            // Adaptive renorm of d_hidden — if max-abs exceeds the
            // configured ceiling, scale d_hidden in-place to bring
            // max-abs back down. Preserves direction (every element
            // scaled by the same factor); Adam doesn't care about
            // absolute scale.
            if clip_max > 0.0 {
                let (max_abs, _) =
                    read_buf_stats(&self.ctx, scratch.d_hidden, self.cfg.d_model as usize).await?;
                if max_abs > clip_max && max_abs.is_finite() {
                    let s = clip_max / max_abs;
                    let mut cenc =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("bwd.clip"),
                            });
                    scale_chained(
                        &self.ctx,
                        &self.pipes,
                        &mut cenc,
                        scratch.d_hidden,
                        self.cfg.d_model as usize,
                        s,
                    );
                    self.ctx.queue.submit(Some(cenc.finish()));
                }
            }

            if trace_hidden {
                let (max_abs, nans) =
                    read_buf_stats(&self.ctx, scratch.d_hidden, self.cfg.d_model as usize).await?;
                eprintln!(
                    "[trace] after layer {li} bwd: d_hidden max_abs={max_abs:.3e} nan={nans}"
                );
            }
        }

        // ===== Loss readback =====
        let loss_read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bwd.loss_read"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut renc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        cap: &LayerCaptureBuffers<'a>,
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
        let attn_norm_w = wc
            .buffer_async(&format!("{prefix}attn_norm.weight"))
            .await?;
        let post_attn_w = wc
            .buffer_async(&format!("{prefix}post_attention_norm.weight"))
            .await?;
        let mlp_norm_w = wc.buffer_async(&format!("{prefix}ffn_norm.weight")).await?;
        let post_ffw_w = wc
            .buffer_async(&format!("{prefix}post_ffw_norm.weight"))
            .await?;
        let q_w = wc.buffer_async(&format!("{prefix}attn_q.weight")).await?;
        let q_norm_w = wc
            .buffer_async(&format!("{prefix}attn_q_norm.weight"))
            .await?;
        let o_w = wc
            .buffer_async(&format!("{prefix}attn_output.weight"))
            .await?;
        let k_w = wc.buffer_async(&format!("{prefix}attn_k.weight")).await?;
        let k_norm_w = wc
            .buffer_async(&format!("{prefix}attn_k_norm.weight"))
            .await?;
        let v_name = format!("{prefix}attn_v.weight");
        let v_w = wc.buffer_async(&v_name).await?;
        let v_w_dtype = wc.dtype(&v_name)?;
        let gate_w = wc.buffer_async(&format!("{prefix}ffn_gate.weight")).await?;
        let up_w = wc.buffer_async(&format!("{prefix}ffn_up.weight")).await?;
        let down_name = format!("{prefix}ffn_down.weight");
        let down_w = wc.buffer_async(&down_name).await?;
        let down_dtype = wc.dtype(&down_name)?;
        let factors_w = if matches!(kind, LayerKind::Global) {
            wc.buffer_opt_async("rope_freqs.weight").await?
        } else {
            None
        };

        // Undo per-layer output scale.
        if let Some(s) = self.layer_scalars[i as usize] {
            scale_chained(&self.ctx, &self.pipes, enc, scratch.d_hidden, d_model, s);
        }

        // Pre-copy the `pos`-slices of the seq-sized captures into
        // single-position windows so the rest of backward_layer can
        // bind them via `as_entire_binding()` without paying offset
        // alignment friction. The per-history K/V LoRA backward and
        // single-forward PerPosition both re-copy *other* positions
        // into the same windows.
        let d_model_bytes = (d_model as u64) * 4;
        let kv_row_bytes = (n_kv_heads as u64) * (head_dim as u64) * 4;
        let n_heads_row_bytes = (n_heads as u64) * (head_dim as u64) * 4;
        let ffn_row_bytes = (ffn_n as u64) * 4;
        let pos_off = pos as u64;
        // Three were already pre-copied (norm_x_attn, k_pre_norm,
        // v_pre_norm) for per-history K/V LoRA backward; the rest
        // (hidden_in, q_pre_norm, q_post_rope, attn_out, attn_proj,
        // pre_ffn_rms, norm_x_ffn, ffn_gate, ffn_up, ffn_act,
        // ffn_out, plus PLE if applicable) are needed for the full
        // backward_layer chain to work uniformly across positions.
        enc.copy_buffer_to_buffer(
            cap.norm_x_attn,
            pos_off * d_model_bytes,
            scratch.norm_x_attn_window,
            0,
            d_model_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.k_pre_norm,
            pos_off * kv_row_bytes,
            scratch.k_pre_norm_window,
            0,
            kv_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.v_pre_norm,
            pos_off * kv_row_bytes,
            scratch.v_pre_norm_window,
            0,
            kv_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.hidden_in,
            pos_off * d_model_bytes,
            scratch.hidden_in_window,
            0,
            d_model_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.q_pre_norm,
            pos_off * n_heads_row_bytes,
            scratch.q_pre_norm_window,
            0,
            n_heads_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.q_post_rope,
            pos_off * n_heads_row_bytes,
            scratch.q_post_rope_window,
            0,
            n_heads_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.attn_out,
            pos_off * n_heads_row_bytes,
            scratch.attn_out_window,
            0,
            n_heads_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.attn_proj,
            pos_off * d_model_bytes,
            scratch.attn_proj_window,
            0,
            d_model_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.pre_ffn_rms,
            pos_off * d_model_bytes,
            scratch.pre_ffn_rms_window,
            0,
            d_model_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.norm_x_ffn,
            pos_off * d_model_bytes,
            scratch.norm_x_ffn_window,
            0,
            d_model_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.ffn_gate,
            pos_off * ffn_row_bytes,
            scratch.ffn_gate_window,
            0,
            ffn_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.ffn_up,
            pos_off * ffn_row_bytes,
            scratch.ffn_up_window,
            0,
            ffn_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.ffn_act,
            pos_off * ffn_row_bytes,
            scratch.ffn_act_window,
            0,
            ffn_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.ffn_out,
            pos_off * d_model_bytes,
            scratch.ffn_out_window,
            0,
            d_model_bytes,
        );
        if self.cfg.has_ple() {
            let ple_dim_bytes = (self.cfg.ple_dim as u64) * 4;
            enc.copy_buffer_to_buffer(
                cap.ple_state,
                pos_off * ple_dim_bytes,
                scratch.ple_state_window,
                0,
                ple_dim_bytes,
            );
            enc.copy_buffer_to_buffer(
                cap.ple_act,
                pos_off * ple_dim_bytes,
                scratch.ple_act_window,
                0,
                ple_dim_bytes,
            );
            enc.copy_buffer_to_buffer(
                cap.ple_proj,
                pos_off * d_model_bytes,
                scratch.ple_proj_window,
                0,
                d_model_bytes,
            );
        }

        // ----- PLE injection backward -----
        //
        // Forward order:
        //   ple_state  = matmul(inp_gate_w, hidden, ple_dim)
        //   ple_act    = geglu(ple_state, per_layer[i*ple_dim..])
        //   ple_proj   = matmul(proj_w, ple_act, d_model)
        //   norm_y     = rmsnorm(ple_proj, post_norm_w)
        //   hidden    += norm_y
        //
        // Reverse: residual_add back (d_norm_y = d_hidden, then
        // accumulate d_hidden_from_ple) → rmsnorm back (post_norm_w)
        // → matmul back (proj_w) → geglu back (drop d_up — per_layer
        // is not a trainable parameter) → matmul back (inp_gate_w) →
        // add into running d_hidden.
        if self.cfg.has_ple() {
            let ple_dim = self.cfg.ple_dim as usize;
            let inp_gate_w = wc.buffer_async(&format!("{prefix}inp_gate.weight")).await?;
            let proj_w = wc.buffer_async("per_layer_model_proj.weight").await?;
            let post_norm_w = wc.buffer_async("per_layer_proj_norm.weight").await?;

            // d_norm_y = d_hidden (residual_add backward — both
            // additive branches carry d_hidden_out through unchanged).
            // post_ffw_norm rmsnorm backward of the PLE rmsnorm:
            // d_ple_proj = rmsnorm_back(cap.ple_proj, post_norm_w, d_hidden) → d_hidden_tmp
            rmsnorm_backward_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.ple_proj_window,
                &post_norm_w,
                scratch.d_hidden,
                scratch.d_hidden_tmp,
                d_model,
                eps,
                true,
            );
            // matmul back through proj_w: d_ple_act = proj_wᵀ · d_ple_proj.
            matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &proj_w,
                scratch.d_hidden_tmp,
                scratch.d_ple_act,
                ple_dim,
                d_model,
            );
            // Copy per_layer[i*ple_dim..] into the staging buf so
            // geglu_back's `up` binding is read-only and distinct
            // from `dy` / `d_gate` / `d_up`.
            let layer_off = (i as u64) * (ple_dim as u64) * 4;
            let layer_bytes = (ple_dim as u64) * 4;
            enc.copy_buffer_to_buffer(
                &self.per_layer,
                layer_off,
                scratch.ple_per_layer_tmp,
                0,
                layer_bytes,
            );
            // geglu back: d_gate → d_ple_state, d_up → d_ple_up_discard.
            geglu_backward_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.ple_state_window,
                scratch.ple_per_layer_tmp,
                scratch.d_ple_act,
                scratch.d_ple_state,
                scratch.d_ple_up_discard,
                ple_dim,
            );
            // matmul back through inp_gate_w: d_hidden_from_ple = inp_gate_wᵀ · d_ple_state
            //   → d_hidden_tmp (safe to overwrite at this point).
            matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &inp_gate_w,
                scratch.d_ple_state,
                scratch.d_hidden_tmp,
                d_model,
                ple_dim,
            );
            // d_hidden += d_hidden_from_ple (residual_add backward
            // combines PLE branch's input grad with the through-path).
            residual_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_hidden,
                scratch.d_hidden_tmp,
                d_model,
            );
        }

        // ----- FFN block backward -----
        // residual_add backward (ffn): d_norm_y_ffn = d_hidden (alias).
        // d_hidden continues as d_pre_ffn_residual (= d_h1 path through residual).
        //
        // post_ffw_norm rmsnorm backward → d_ffn_out into d_hidden_tmp.
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.ffn_out_window,
            &post_ffw_w,
            scratch.d_hidden,
            scratch.d_hidden_tmp,
            d_model,
            eps,
            true,
        );

        // ffn_down matmul backward: d_ffn_act = down_wᵀ · d_ffn_out → d_ffn_a.
        match down_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &down_w,
                scratch.d_hidden_tmp,
                scratch.d_ffn_a,
                ffn_n,
                d_model,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &down_w,
                scratch.d_hidden_tmp,
                scratch.d_ffn_a,
                ffn_n,
                d_model,
            ),
            other => {
                return Err(RullamaError::Inference(format!(
                    "ffn_down dtype {other:?} unsupported in backward"
                )));
            }
        }

        // ffn_down LoRA backward:
        //   dB += s · d_ffn_out ⊗ z;  u = Bᵀ · d_ffn_out;
        //   d_ffn_a += s · Aᵀ · u;    dA += s · u ⊗ cap.ffn_act.
        if let (Some(d_lora), Some(d_grad)) = (lora.ffn_down.as_ref(), grad.ffn_down.as_ref()) {
            let r = d_lora.rank as usize;
            let s = d_lora.scale;
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_hidden_tmp,
                d_lora.z,
                d_grad.d_b,
                d_model,
                r,
                s,
                true,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                d_lora.b,
                scratch.d_hidden_tmp,
                d_lora.z,
                d_model,
                r,
                1.0,
                false,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                d_lora.a,
                d_lora.z,
                scratch.d_ffn_a,
                r,
                ffn_n,
                s,
                true,
            );
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                d_lora.z,
                scratch.ffn_act_window,
                d_grad.d_a,
                r,
                ffn_n,
                s,
                true,
            );
        }

        // geglu backward → d_ffn_gate (d_ffn_b), d_ffn_up (d_ffn_c).
        geglu_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.ffn_gate_window,
            scratch.ffn_up_window,
            scratch.d_ffn_a,
            scratch.d_ffn_b,
            scratch.d_ffn_c,
            ffn_n,
        );

        // gate matmul backward: d_norm_x_ffn_via_gate = gate_wᵀ · d_ffn_gate → d_hidden_tmp.
        matmul_q4_k_backward_input_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &gate_w,
            scratch.d_ffn_b,
            scratch.d_hidden_tmp,
            d_model,
            ffn_n,
        );
        // ffn_gate LoRA backward:
        //   dB += s · d_ffn_gate ⊗ z;  u = Bᵀ · d_ffn_gate;
        //   d_hidden_tmp += s · Aᵀ · u; dA += s · u ⊗ cap.norm_x_ffn.
        if let (Some(g_lora), Some(g_grad)) = (lora.ffn_gate.as_ref(), grad.ffn_gate.as_ref()) {
            let r = g_lora.rank as usize;
            let s = g_lora.scale;
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_ffn_b,
                g_lora.z,
                g_grad.d_b,
                ffn_n,
                r,
                s,
                true,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                g_lora.b,
                scratch.d_ffn_b,
                g_lora.z,
                ffn_n,
                r,
                1.0,
                false,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                g_lora.a,
                g_lora.z,
                scratch.d_hidden_tmp,
                r,
                d_model,
                s,
                true,
            );
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                g_lora.z,
                scratch.norm_x_ffn_window,
                g_grad.d_a,
                r,
                d_model,
                s,
                true,
            );
        }
        // up matmul backward: d_norm_x_ffn_via_up → d_hidden_tmp2.
        matmul_q4_k_backward_input_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &up_w,
            scratch.d_ffn_c,
            scratch.d_hidden_tmp2,
            d_model,
            ffn_n,
        );
        // ffn_up LoRA backward (mirrors gate but accumulates into d_hidden_tmp2).
        if let (Some(u_lora), Some(u_grad)) = (lora.ffn_up.as_ref(), grad.ffn_up.as_ref()) {
            let r = u_lora.rank as usize;
            let s = u_lora.scale;
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_ffn_c,
                u_lora.z,
                u_grad.d_b,
                ffn_n,
                r,
                s,
                true,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                u_lora.b,
                scratch.d_ffn_c,
                u_lora.z,
                ffn_n,
                r,
                1.0,
                false,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                u_lora.a,
                u_lora.z,
                scratch.d_hidden_tmp2,
                r,
                d_model,
                s,
                true,
            );
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                u_lora.z,
                scratch.norm_x_ffn_window,
                u_grad.d_a,
                r,
                d_model,
                s,
                true,
            );
        }
        // d_hidden_tmp += d_hidden_tmp2 (full d_norm_x_ffn).
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.d_hidden_tmp,
            scratch.d_hidden_tmp2,
            d_model,
        );

        // mlp_norm rmsnorm backward → d_pre_ffn_rms into d_hidden_tmp2.
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.pre_ffn_rms_window,
            &mlp_norm_w,
            scratch.d_hidden_tmp,
            scratch.d_hidden_tmp2,
            d_model,
            eps,
            true,
        );
        // Accumulate FFN block branch contribution into running d_hidden.
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.d_hidden,
            scratch.d_hidden_tmp2,
            d_model,
        );

        // ----- Attention block backward -----
        // residual_add backward (attn): d_norm_y_attn = d_hidden (alias).
        //
        // post_attn_norm rmsnorm backward → d_attn_proj into d_hidden_tmp.
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.attn_proj_window,
            &post_attn_w,
            scratch.d_hidden,
            scratch.d_hidden_tmp,
            d_model,
            eps,
            true,
        );

        // o_proj matmul backward: d_attn_out = o_wᵀ · d_attn_proj → scratch.d_attn_out.
        matmul_q4_k_backward_input_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &o_w,
            scratch.d_hidden_tmp,
            scratch.d_attn_out,
            n_heads * head_dim,
            d_model,
        );

        // o LoRA backward: dB += scale·dy⊗z; u=Bᵀ·dy; d_attn_out += scale·Aᵀ·u; dA += scale·u⊗x.
        if let (Some(o_lora), Some(o_grad)) = (lora.o.as_ref(), grad.o.as_ref()) {
            let r = o_lora.rank as usize;
            let s = o_lora.scale;
            // dB_o += s · d_attn_proj ⊗ z_o  (using captured z from forward).
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_hidden_tmp,
                o_lora.z,
                o_grad.d_b,
                d_model,
                r,
                s,
                true,
            );
            // u_o = B_oᵀ · d_attn_proj → o_lora.z (overwrite).
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                o_lora.b,
                scratch.d_hidden_tmp,
                o_lora.z,
                d_model,
                r,
                1.0,
                false,
            );
            // d_attn_out += s · A_oᵀ · u_o.
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                o_lora.a,
                o_lora.z,
                scratch.d_attn_out,
                r,
                n_heads * head_dim,
                s,
                true,
            );
            // dA_o += s · u_o ⊗ attn_out (= cap.attn_out).
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                o_lora.z,
                scratch.attn_out_window,
                o_grad.d_a,
                r,
                n_heads * head_dim,
                s,
                true,
            );
        }

        // Recompute attention probs (from q_post_rope + kv cache) into scratch.attn_probs.
        let window = if matches!(kind, LayerKind::SlidingWindow) {
            self.cfg.sliding_window as usize
        } else {
            0
        };
        attention_probs_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.q_post_rope_window,
            &self.kv_k[i as usize],
            scratch.attn_probs,
            head_dim,
            n_heads,
            n_kv_heads,
            pos as usize,
            history_len as usize,
            window,
        );

        // Attn backward pass 1: d_q + d_scores (staged).
        attention_backward_dq_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &self.kv_k[i as usize],
            &self.kv_v[i as usize],
            scratch.attn_probs,
            scratch.d_attn_out,
            scratch.attn_d_scores,
            scratch.d_q,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len as usize,
        );
        // Attn backward pass 2: d_k_hist, d_v_hist.
        attention_backward_dkv_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.q_post_rope_window,
            scratch.attn_probs,
            scratch.d_attn_out,
            scratch.attn_d_scores,
            scratch.d_k_hist,
            scratch.d_v_hist,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len as usize,
        );

        // rope backward of q (in-place into d_q → now d_q_pre_rope's value).
        let (rope_base, rope_dims) = match kind {
            LayerKind::SlidingWindow => {
                (self.cfg.rope_freq_base_swa, self.cfg.rope_dim_swa as usize)
            }
            LayerKind::Global => (self.cfg.rope_freq_base, self.cfg.rope_dim_global as usize),
        };
        rope_neox_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.d_q,
            factors_w.as_ref(),
            &self.dummy,
            head_dim,
            n_heads,
            pos as usize,
            rope_dims,
            rope_base,
        );
        // q_norm rmsnorm backward → d_q_pre_norm.
        rmsnorm_per_row_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.q_pre_norm_window,
            &q_norm_w,
            scratch.d_q,
            scratch.d_q_pre_norm,
            n_heads,
            head_dim,
            eps,
            true,
        );
        // q matmul backward: d_norm_x_attn_via_q → d_hidden_tmp (overwrites d_attn_proj).
        matmul_q4_k_backward_input_chained(
            &self.ctx,
            &self.pipes,
            enc,
            &q_w,
            scratch.d_q_pre_norm,
            scratch.d_hidden_tmp,
            d_model,
            n_heads * head_dim,
        );
        // q LoRA backward.
        if let (Some(q_lora), Some(q_grad)) = (lora.q.as_ref(), grad.q.as_ref()) {
            let r = q_lora.rank as usize;
            let s = q_lora.scale;
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_q_pre_norm,
                q_lora.z,
                q_grad.d_b,
                n_heads * head_dim,
                r,
                s,
                true,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                q_lora.b,
                scratch.d_q_pre_norm,
                q_lora.z,
                n_heads * head_dim,
                r,
                1.0,
                false,
            );
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                enc,
                q_lora.a,
                q_lora.z,
                scratch.d_hidden_tmp,
                r,
                d_model,
                s,
                true,
            );
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                q_lora.z,
                scratch.norm_x_attn_window,
                q_grad.d_a,
                r,
                d_model,
                s,
                true,
            );
        }

        // K/V backward — only on layers that own their own K/V (i.e.
        // `donor.is_none()`). KV-shared layers (`donor.is_some()`) read
        // K/V from the donor's cache during forward, so they have no
        // K/V matmul or norm of their own to differentiate. Running
        // the chain anyway on donor layers would consume stale captures
        // (cap.k_pre_norm / cap.v_pre_norm carry the donor's last
        // values, not this layer's, because forward never wrote them
        // here). For now the shared layers' contribution to the
        // donor's K/V LoRA gradient is dropped — a small M0
        // approximation; the correct fix is to route d_k_hist /
        // d_v_hist into the donor's grad accumulators.
        let donor = self.donor_map[i as usize];
        if donor.is_none() {
            // K backward — pull d_k at the final position from d_k_hist.
            // For M0 we only consume the final-position slice (history positions
            // before `pos` get zero LoRA grad contribution — see plan).
            let row_bytes = (n_kv_heads * head_dim * 4) as u64;
            let dk_final_off = pos as u64 * row_bytes;
            enc.copy_buffer_to_buffer(
                scratch.d_k_hist,
                dk_final_off,
                scratch.d_k_pre_rope,
                0,
                row_bytes,
            );
            rope_neox_backward_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_k_pre_rope,
                factors_w.as_ref(),
                &self.dummy,
                head_dim,
                n_kv_heads,
                pos as usize,
                rope_dims,
                rope_base,
            );
            rmsnorm_per_row_backward_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.k_pre_norm_window,
                &k_norm_w,
                scratch.d_k_pre_rope,
                scratch.d_k_pre_norm,
                n_kv_heads,
                head_dim,
                eps,
                true,
            );
            // d_norm_x_attn_via_k → d_hidden_tmp2.
            matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                enc,
                &k_w,
                scratch.d_k_pre_norm,
                scratch.d_hidden_tmp2,
                d_model,
                n_kv_heads * head_dim,
            );
            residual_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_hidden_tmp,
                scratch.d_hidden_tmp2,
                d_model,
            );
            if let (Some(k_lora), Some(k_grad)) = (lora.k.as_ref(), grad.k.as_ref()) {
                let r = k_lora.rank as usize;
                let s = k_lora.scale;
                lora_outer_add_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    scratch.d_k_pre_norm,
                    k_lora.z,
                    k_grad.d_b,
                    n_kv_heads * head_dim,
                    r,
                    s,
                    true,
                );
                lora_matmul_col_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    k_lora.b,
                    scratch.d_k_pre_norm,
                    k_lora.z,
                    n_kv_heads * head_dim,
                    r,
                    1.0,
                    false,
                );
                lora_matmul_col_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    k_lora.a,
                    k_lora.z,
                    scratch.d_hidden_tmp,
                    r,
                    d_model,
                    s,
                    true,
                );
                lora_outer_add_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    k_lora.z,
                    scratch.norm_x_attn_window,
                    k_grad.d_a,
                    r,
                    d_model,
                    s,
                    true,
                );
            }

            // V backward — pull d_v at the final position from d_v_hist into
            // d_k_pre_norm (free at this point — k backward is done) so it
            // can serve as the rmsnorm_back `dy` without aliasing the `dx`
            // output buffer.
            enc.copy_buffer_to_buffer(
                scratch.d_v_hist,
                dk_final_off,
                scratch.d_k_pre_norm,
                0,
                row_bytes,
            );
            // V was passed through unweighted rmsnorm_per_row; do the unweighted backward.
            rmsnorm_per_row_backward_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.v_pre_norm_window,
                &self.dummy,
                scratch.d_k_pre_norm,
                scratch.d_v_pre_norm,
                n_kv_heads,
                head_dim,
                eps,
                false,
            );
            // d_norm_x_attn_via_v → d_hidden_tmp2.
            match v_w_dtype {
                GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    &v_w,
                    scratch.d_v_pre_norm,
                    scratch.d_hidden_tmp2,
                    d_model,
                    n_kv_heads * head_dim,
                ),
                GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    &v_w,
                    scratch.d_v_pre_norm,
                    scratch.d_hidden_tmp2,
                    d_model,
                    n_kv_heads * head_dim,
                ),
                other => {
                    return Err(RullamaError::Inference(format!(
                        "attn_v dtype {other:?} unsupported in backward"
                    )));
                }
            }
            residual_add_chained(
                &self.ctx,
                &self.pipes,
                enc,
                scratch.d_hidden_tmp,
                scratch.d_hidden_tmp2,
                d_model,
            );
            if let (Some(v_lora), Some(v_grad)) = (lora.v.as_ref(), grad.v.as_ref()) {
                let r = v_lora.rank as usize;
                let s = v_lora.scale;
                lora_outer_add_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    scratch.d_v_pre_norm,
                    v_lora.z,
                    v_grad.d_b,
                    n_kv_heads * head_dim,
                    r,
                    s,
                    true,
                );
                lora_matmul_col_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    v_lora.b,
                    scratch.d_v_pre_norm,
                    v_lora.z,
                    n_kv_heads * head_dim,
                    r,
                    1.0,
                    false,
                );
                lora_matmul_col_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    v_lora.a,
                    v_lora.z,
                    scratch.d_hidden_tmp,
                    r,
                    d_model,
                    s,
                    true,
                );
                lora_outer_add_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    v_lora.z,
                    scratch.norm_x_attn_window,
                    v_grad.d_a,
                    r,
                    d_model,
                    s,
                    true,
                );
            }

            // ----- Per-history K/V LoRA backward -----
            //
            // For each history position `hp != pos`, accumulate dA/dB
            // contributions into the K and V LoRAs using the
            // per-position seq captures + d_k_hist[hp] / d_v_hist[hp].
            // We do NOT update the running `d_hidden` (which is a
            // single-position scratch carrying the gradient at the
            // FINAL position only); the matmul-back-through-k_w /
            // v_w contributions to d_hidden_at_hp are dropped — that's
            // the per-position-d_hidden story owned by the
            // single-forward PerPosition variant.
            //
            // `z` per LoRA is recomputed inline as A · norm_x_attn[hp]
            // (cheap rank·d_model matmul) so we don't need per-position
            // `z` storage.
            for hp_u in 0..history_len {
                if hp_u == pos {
                    continue;
                }
                let hp = hp_u as usize;
                let p_kv_off = hp_u as u64 * row_bytes;
                let p_dm_off = hp_u as u64 * d_model_bytes;
                // Refresh windows for this history position.
                enc.copy_buffer_to_buffer(
                    cap.norm_x_attn,
                    p_dm_off,
                    scratch.norm_x_attn_window,
                    0,
                    d_model_bytes,
                );
                enc.copy_buffer_to_buffer(
                    cap.k_pre_norm,
                    p_kv_off,
                    scratch.k_pre_norm_window,
                    0,
                    row_bytes,
                );

                // K at history position hp.
                enc.copy_buffer_to_buffer(
                    scratch.d_k_hist,
                    p_kv_off,
                    scratch.d_k_pre_rope,
                    0,
                    row_bytes,
                );
                rope_neox_backward_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    scratch.d_k_pre_rope,
                    factors_w.as_ref(),
                    &self.dummy,
                    head_dim,
                    n_kv_heads,
                    hp,
                    rope_dims,
                    rope_base,
                );
                rmsnorm_per_row_backward_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    scratch.k_pre_norm_window,
                    &k_norm_w,
                    scratch.d_k_pre_rope,
                    scratch.d_k_pre_norm,
                    n_kv_heads,
                    head_dim,
                    eps,
                    true,
                );
                if let (Some(k_lora), Some(k_grad)) = (lora.k.as_ref(), grad.k.as_ref()) {
                    let r = k_lora.rank as usize;
                    let s = k_lora.scale;
                    // z_k[hp] = A_k · norm_x_attn[hp]
                    lora_matmul_row_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        k_lora.a,
                        scratch.norm_x_attn_window,
                        k_lora.z,
                        d_model,
                        r,
                        1.0,
                        false,
                    );
                    lora_outer_add_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        scratch.d_k_pre_norm,
                        k_lora.z,
                        k_grad.d_b,
                        n_kv_heads * head_dim,
                        r,
                        s,
                        true,
                    );
                    lora_matmul_col_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        k_lora.b,
                        scratch.d_k_pre_norm,
                        k_lora.z,
                        n_kv_heads * head_dim,
                        r,
                        1.0,
                        false,
                    );
                    lora_outer_add_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        k_lora.z,
                        scratch.norm_x_attn_window,
                        k_grad.d_a,
                        r,
                        d_model,
                        s,
                        true,
                    );
                }

                // V at history position hp.
                enc.copy_buffer_to_buffer(
                    cap.v_pre_norm,
                    p_kv_off,
                    scratch.v_pre_norm_window,
                    0,
                    row_bytes,
                );
                enc.copy_buffer_to_buffer(
                    scratch.d_v_hist,
                    p_kv_off,
                    scratch.d_k_pre_norm,
                    0,
                    row_bytes,
                );
                rmsnorm_per_row_backward_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    scratch.v_pre_norm_window,
                    &self.dummy,
                    scratch.d_k_pre_norm,
                    scratch.d_v_pre_norm,
                    n_kv_heads,
                    head_dim,
                    eps,
                    false,
                );
                if let (Some(v_lora), Some(v_grad)) = (lora.v.as_ref(), grad.v.as_ref()) {
                    let r = v_lora.rank as usize;
                    let s = v_lora.scale;
                    lora_matmul_row_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        v_lora.a,
                        scratch.norm_x_attn_window,
                        v_lora.z,
                        d_model,
                        r,
                        1.0,
                        false,
                    );
                    lora_outer_add_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        scratch.d_v_pre_norm,
                        v_lora.z,
                        v_grad.d_b,
                        n_kv_heads * head_dim,
                        r,
                        s,
                        true,
                    );
                    lora_matmul_col_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        v_lora.b,
                        scratch.d_v_pre_norm,
                        v_lora.z,
                        n_kv_heads * head_dim,
                        r,
                        1.0,
                        false,
                    );
                    lora_outer_add_chained(
                        &self.ctx,
                        &self.pipes,
                        enc,
                        v_lora.z,
                        scratch.norm_x_attn_window,
                        v_grad.d_a,
                        r,
                        d_model,
                        s,
                        true,
                    );
                }
            }
        }

        // After the per-history loop, the windows hold the LAST
        // history position's values. Restore them to the `pos`-slice
        // so any downstream code that relies on the windows holding
        // the final-position activations (currently only the
        // `attn_norm` backward below, which doesn't read these) sees
        // the right state.
        enc.copy_buffer_to_buffer(
            cap.norm_x_attn,
            (pos as u64) * d_model_bytes,
            scratch.norm_x_attn_window,
            0,
            d_model_bytes,
        );

        // attn_norm rmsnorm backward — flows the attn block contribution
        // into d_hidden_tmp2, then accumulates into running d_hidden.
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.hidden_in_window,
            &attn_norm_w,
            scratch.d_hidden_tmp,
            scratch.d_hidden_tmp2,
            d_model,
            eps,
            true,
        );
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            enc,
            scratch.d_hidden,
            scratch.d_hidden_tmp2,
            d_model,
        );

        Ok(())
    }
}
