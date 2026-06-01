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
    lora_matmul_col_chained, lora_matmul_fused_chained, lora_matmul_row_chained,
    lora_outer_add_chained, make_dummy_storage, matmul_q4_k_backward_input_chained,
    matmul_q4_k_backward_input_tile_chained, matmul_q4_k_chained,
    matmul_q6_k_backward_input_chained, matmul_q6_k_backward_input_tile_chained,
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
    pub b: &'a wgpu::Buffer, // [out_dim, rank]; packed f16 pairs in u32 if `b_is_f16`
    pub z: &'a wgpu::Buffer, // [rank] scratch
    pub rank: u32,
    pub scale: f32, // alpha / rank
    /// When true, `b` is stored as packed f16 (two elements per u32)
    /// and the forward-correction dispatch routes through
    /// `lora_matmul_fused_f16b` instead of `lora_matmul_fused`.
    /// Currently set only for the lm_head global LoRA slot, where the
    /// `vocab × rank` matrix dominates LoRA bandwidth.
    pub b_is_f16: bool,
}

/// Per-layer progress callback fired between encoder submits during
/// a forward + backward layer walk. Signature:
/// `(phase, current, total)` where `phase` is one of `"forward"` /
/// `"backward"` and `current` is 1-based logical layer index. Used
/// by training to drive a VisionProgress-style status strip (see
/// `examples/web/src/components/TrainingProgress.tsx`) — without the
/// per-layer beacons the user stares at a "step 0 / N" counter while
/// a 30 s pipeline-compile + first step grinds in silence.
pub type LayerProgressCb<'a> = dyn Fn(&str, u32, u32) + 'a;

/// ROME residual-stream perturbation injection point. When passed to
/// [`Forward::step_capture_with_rome_delta`], the running `hidden`
/// state is incremented by `delta_buf` (shape `[d_model]`)
/// immediately after `target_layer` writes its contribution to the
/// residual stream, before `target_layer + 1` (or the final norm)
/// reads `hidden`.
///
/// Equivalent to kmeng01/rome's `edit_output_fn` hook: at the
/// subject-last token's position, the optimizer-controlled δ vector
/// is added to the MLP output of the target layer so the model
/// behaves as if `ffn_out[L, subject_last_pos]` had been substituted
/// to produce the target token. Caller is responsible for invoking
/// the perturbed step only on the subject-last position; all other
/// positions use the plain `step_capture`.
pub struct RomeDeltaInjection<'a> {
    pub delta_buf: &'a wgpu::Buffer,
    pub target_layer: u32,
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

/// Model-global LoRA slots — not keyed per layer. Pass `None` for any
/// target that isn't LoRA-wrapped.
///
/// `embed_tokens` injects after the input embedding lookup:
/// `hidden += scale · B_emb · A_emb[:, token_id]` where A_emb has shape
/// `[rank, vocab]` and B_emb has shape `[d_model, rank]`.
///
/// `lm_head` injects after the tiled output projection but before
/// softcap: `logits += scale · B_lmh · (A_lmh · norm_x_final)` where
/// A_lmh has shape `[rank, d_model]` and B_lmh has shape `[vocab, rank]`.
///
/// Even though Gemma 4 uses tied weights (`token_embd.weight` is shared
/// between input embedding and output projection), `embed_tokens` and
/// `lm_head` are two separate LoRA pairs — matches PEFT's
/// `modules_to_save` semantics so input and output distributions can be
/// steered independently. Google's QLoRA Gemma recipe is the canonical
/// reference for this pattern (`ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora`).
#[derive(Default)]
pub struct GlobalLoraSlots<'a> {
    pub embed_tokens: Option<LoraSlot<'a>>,
    pub lm_head: Option<LoraSlot<'a>>,
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

    /// **MeBP-inspired memory mode.** When true, the per-layer forward
    /// loop drains GPU and destroys each layer's weight tiles after its
    /// submit, so peak weight memory during forward = ~1 layer worth
    /// (~40 MiB on e2b) instead of ~all 35 layers (~1417 MiB). Backward's
    /// gradient-checkpointing recompute re-fetches via `buffer_async`
    /// (decompress per block, matching MeBP arxiv 2510.03425's lazy
    /// load). Set true by `TrainingSession::new` on iPhone targets;
    /// false for chat-side inference where the weights need to stay
    /// cached across tokens (per-token re-fetch destroys generation
    /// perf). Costs ~32-42% extra forward time (per MeBP §4.2) in
    /// exchange for fitting under the iOS WebContent jetsam ceiling.
    forward_destroy_per_layer: bool,

    /// Lower bound for backward_layer iteration. Layers `i < floor`
    /// are NOT walked in backward (no LoRA gradient, no recompute).
    /// When `forward_destroy_per_layer` is on, we destroy blk.i's
    /// weights only for `i < floor` — layers at or above the floor
    /// stay cached so backward's recompute hits the WeightCache
    /// instead of re-uploading from OPFS (the re-upload was what
    /// killed iPhone after we removed the head→backward yield).
    /// Set by `TrainingSession::new` from
    /// `TrainingHyperparams::backward_layer_floor`.
    forward_destroy_layer_floor: u32,

    /// **Mobile mode switch** — when true, all the iOS Safari WebGPU
    /// survival workarounds are active: 0 ms event-loop yields at
    /// recompute→backward_layer + backward_layer→epilogue boundaries,
    /// vocab-axis tiling of head_outproj backward matmul, chunked
    /// destroy with yields between chunks, backward-kernel pre-warm
    /// at session start. When false, all of those are no-ops and
    /// training runs the native-fast path (3-5× faster wall time).
    /// Set by `TrainingSession::new` from
    /// `TrainingHyperparams::memory_tight`.
    mobile_mode: bool,

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
            forward_destroy_per_layer: false,
            // u32::MAX = no floor restriction (every layer destroyed
            // when forward_destroy_per_layer is on). TrainingSession::new
            // overrides this to the actual backward_layer_floor.
            forward_destroy_layer_floor: u32::MAX,
            mobile_mode: false,
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

    /// Enable MeBP-style per-layer weight destroy during forward. See
    /// the field doc on `forward_destroy_per_layer`. Call once after
    /// constructing `Forward` for training; never set for chat-side
    /// inference.
    pub fn set_forward_destroy_per_layer(&mut self, on: bool) {
        self.forward_destroy_per_layer = on;
    }

    /// Set the floor used by per-layer forward destroy. Only layers
    /// `i < floor` are destroyed during forward — layers at or above
    /// the floor stay cached so backward's recompute hits the
    /// `WeightCache` instead of re-uploading from OPFS. Pass the
    /// training session's `backward_layer_floor`; for inference (which
    /// never sets `forward_destroy_per_layer = true`) this is a no-op.
    pub fn set_forward_destroy_layer_floor(&mut self, floor: u32) {
        self.forward_destroy_layer_floor = floor;
    }

    /// Toggle the mobile-mode workaround stack. See the field doc on
    /// `mobile_mode` for what this gates. Off by default; the
    /// `TrainingSession` flips it on when
    /// `TrainingHyperparams::memory_tight` is true.
    pub fn set_mobile_mode(&mut self, on: bool) {
        self.mobile_mode = on;
    }

    /// Drop every cached `(uniform, bind_group)` entry in the shared
    /// `BindGroupCache`. Called at end-of-step in training to prevent
    /// cross-step accumulation: `invalidate_buffers` eagerly evicts
    /// entries whose underlying buffer was destroyed, but entries that
    /// still reference live scratch / LoRA / KV buffers accumulate
    /// monotonically (no buffer dies → no invalidation → entry lives
    /// forever). Each entry is small (~32-byte uniform + bind-group
    /// descriptor) but the GPUProcess bind-group table tracks all of
    /// them, and after many steps the table is large enough to
    /// pressure WebKit's bookkeeping. Clearing once per step costs
    /// ~50 cache misses (re-build bind groups for the next step's
    /// first dispatches) which is negligible vs the ~5K hits/step
    /// the cache absorbs.
    ///
    /// Chat-side inference does NOT call this — the cache is meant
    /// to stay warm across tokens.
    pub fn clear_bind_cache(&self) {
        self.ctx.bind_cache.clear();
    }

    /// **0 ms JS event-loop yield (mobile-mode + wasm32 only).** Used
    /// by training callers between bursts of GPU submits
    /// (recompute→backward_layer, backward_layer→epilogue) to let iOS
    /// Safari's GPUProcess message pipe drain a tick of pending IPCs
    /// before the next burst lands. On native this is a no-op `await`;
    /// on Mac browsers (wasm32 but `mobile_mode` off) it's also a
    /// no-op — the GPUProcess can keep up without the assistance and
    /// the yields cost ~5-10 ms each.
    ///
    /// 0 ms specifically (not >0): real-device data showed
    /// `setTimeout(500)` at the head→backward boundary was killing
    /// the Worker (iOS reaped the suspended process). 0 ms releases
    /// the event loop for one tick without exposing us to the
    /// suspended-process reaper.
    pub async fn wasm_yield_zero(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.mobile_mode {
                return;
            }
            use wasm_bindgen::JsCast;
            let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global()
                .dyn_into()
                .expect("training session runs inside a DedicatedWorkerGlobalScope");
            let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                let resolve_fn: js_sys::Function = resolve.into();
                let _ = scope.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve_fn, 0);
            });
            let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
        }
    }

    /// **Pre-warm every backward kernel at session start (Patch 7).**
    ///
    /// Dispatches each backward + optimizer + lora kernel ONCE against
    /// tiny throwaway scratch buffers, all in one submit, then awaits a
    /// readback to force GPU completion before returning. Purpose: any
    /// first-execution Metal state setup (argument-buffer staging,
    /// threadgroup memory reservation, intermediate compilation) happens
    /// HERE — when the GPU process has plenty of headroom and no
    /// weights resident — instead of mid-step 2 when 1.4 GiB of weights
    /// are already loaded and iOS jetsam is one resident-set increment
    /// away.
    ///
    /// Inputs to the warmup are garbage (zero-initialised buffers); the
    /// outputs are discarded. We just need Metal to EXECUTE each kernel
    /// once.
    ///
    /// Cost: ~15 throwaway dispatches in one submit + one tiny
    /// readback. ~50 ms wall time, one-shot at session start.
    ///
    /// The throwaway buffers go out of scope at function end, but the
    /// `BindGroupCache` would otherwise hold them alive via the
    /// CachedDispatch's bind_group strong-refs. We `clear()` at the
    /// end to drop those entries so the throwaway buffers actually die.
    pub async fn warmup_backward_pipelines(&mut self) -> Result<()> {
        use crate::backend::dispatch::*;

        let device = &self.ctx.device;
        let mk = |size: u64, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Distinct throwaway buffer pool. Within one compute dispatch
        // the same wgpu::Buffer cannot be bound as both read and
        // read_write (wgpu validation error: "conflicting usages").
        // Allocate enough distinct buffers that every kernel's binding
        // slots can each take a unique buffer. Each kernel call below
        // explicitly picks distinct buffers per binding.
        //
        // Sizes:
        //   small (4 elements)  — uniforms / scalars / loss_out / Adam state
        //   row   (256 elements) — vector-sized scratch / matmul out / vocab vec
        //   big   (4 k elements) — generously sized buffer for larger output writes
        //   q4k/q6k — single super-block of quantized weight bytes
        let s0 = mk(16, "warmup.small.0");
        let s1 = mk(16, "warmup.small.1");
        let s2 = mk(16, "warmup.small.2");
        let s3 = mk(16, "warmup.small.3");
        let r0 = mk(1024, "warmup.row.0");
        let r1 = mk(1024, "warmup.row.1");
        let r2 = mk(1024, "warmup.row.2");
        let r3 = mk(1024, "warmup.row.3");
        let r4 = mk(1024, "warmup.row.4");
        let r5 = mk(1024, "warmup.row.5");
        let r6 = mk(1024, "warmup.row.6");
        let b0 = mk(16 * 1024, "warmup.big.0");
        let b1 = mk(16 * 1024, "warmup.big.1");
        let q4k = mk(256, "warmup.q4k");
        let q6k = mk(256, "warmup.q6k");
        let dummy = mk(4, "warmup.dummy");

        // Each kernel goes in its own command encoder + submit so wgpu's
        // per-command-buffer usage tracker doesn't pessimise. Cost is 15
        // submits at session start — one-shot, no real wall-time impact
        // (each submit is a 1-dispatch tiny command buffer).
        macro_rules! one_submit {
            ($label:expr, |$enc:ident| $body:block) => {{
                let mut $enc = device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some($label) },
                );
                $body
                self.ctx.queue.submit(Some($enc.finish()));
            }};
        }

        // cross_entropy_backward(logits=read, d_logits=read_write,
        // loss_out=read_write).
        one_submit!("warmup.xent_bwd", |enc| {
            cross_entropy_backward_chained(&self.ctx, &self.pipes, &mut enc, &r0, &r1, &s0, 256, 0);
        });
        // matmul_q4_k_backward_input(weight=read, dy=read, dx=read_write).
        one_submit!("warmup.q4k_bwd", |enc| {
            matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &q4k,
                &s1,
                &r2,
                256,
                1,
            );
        });
        // matmul_q6_k_backward_input — same shape.
        one_submit!("warmup.q6k_bwd", |enc| {
            matmul_q6_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &q6k,
                &s2,
                &r3,
                256,
                1,
            );
        });
        // rmsnorm_backward(x=read, w=read, dy=read, dx=read_write).
        one_submit!("warmup.rms_bwd", |enc| {
            rmsnorm_backward_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &s0,
                &s1,
                &s2,
                &s3,
                4,
                1e-5,
                true,
            );
        });
        // rmsnorm_per_row_backward — same role pattern.
        one_submit!("warmup.rms_pr_bwd", |enc| {
            rmsnorm_per_row_backward_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r0,
                &r1,
                &r2,
                &r3,
                1,
                16,
                1e-5,
                true,
            );
        });
        // geglu_backward(gate=read, up=read, dy=read, d_gate=read_write,
        // d_up=read_write).
        one_submit!("warmup.geglu_bwd", |enc| {
            geglu_backward_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r0,
                &r1,
                &r2,
                &r3,
                &r4,
                16,
            );
        });
        // rope_neox_backward(x=read_write, factors=read|None, dummy).
        one_submit!("warmup.rope_bwd", |enc| {
            rope_neox_backward_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r0,
                None,
                &dummy,
                128,
                1,
                0,
                128,
                10000.0,
            );
        });
        // attention_backward_dq — 6 distinct buffers needed.
        one_submit!("warmup.attn_bwd_dq", |enc| {
            attention_backward_dq_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r0,
                &r1,
                &s0,
                &r2,
                &s1,
                &r3,
                64,
                1,
                1,
                1,
            );
        });
        // attention_backward_dkv — 6 distinct.
        one_submit!("warmup.attn_bwd_dkv", |enc| {
            attention_backward_dkv_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r4,
                &s2,
                &r5,
                &s3,
                &r6,
                &b0,
                64,
                1,
                1,
                1,
            );
        });
        // attention_probs(q=read, k_hist=read, probs=read_write).
        one_submit!("warmup.attn_probs", |enc| {
            attention_probs_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r0,
                &r1,
                &b1,
                64,
                1,
                1,
                0,
                1,
                0,
            );
        });
        // lora_outer_add(dy=read, z=read, dB=read_write).
        one_submit!("warmup.lora_outer", |enc| {
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &s0,
                &s1,
                &r0,
                4,
                4,
                1.0,
                true,
            );
        });
        // lora_matmul_col(W=read, x=read, y=read_write).
        one_submit!("warmup.lora_mm_col", |enc| {
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r1,
                &s2,
                &s3,
                4,
                4,
                1.0,
                false,
            );
        });
        // lora_matmul_row.
        one_submit!("warmup.lora_mm_row", |enc| {
            lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &r2,
                &s0,
                &s1,
                4,
                4,
                1.0,
                false,
            );
        });
        // adam_step(grad=read, param=read_write, m=read_write, v=read_write).
        let adam_cfg = AdamConfig::default();
        one_submit!("warmup.adam", |enc| {
            adam_step_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &s0,
                &s1,
                &s2,
                &s3,
                4,
                adam_cfg,
            );
        });
        // sum_of_squares(input=read, output=read_write).
        one_submit!("warmup.sos", |enc| {
            sum_of_squares_chained(&self.ctx, &self.pipes, &mut enc, &r0, &s0, 16, 1.0);
        });

        // Force GPU to actually run everything before we return.
        // `read_buf_stats` issues a `copy_buffer_to_buffer` from the
        // source buffer into a MAP_READ staging buffer; the source
        // needs `COPY_SRC` usage, which our STORAGE-only warmup pool
        // doesn't have. Use a dedicated drain buffer with the right
        // usage flag; the warmup dispatches above will have queued
        // their work, and one `queue.submit` for this drain buffer's
        // ensuing copy is enough to fence them.
        let drain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warmup.drain"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let _drain = read_buf_stats(&self.ctx, &drain_buf, 1).await?;

        // Drop bind-cache entries created against the throwaway
        // buffers — otherwise the cache holds them alive via bind_group
        // strong refs and they never die.
        self.ctx.bind_cache.clear();

        // Use the buffer handles explicitly so they live across the
        // submits above (the dispatchers only borrow them).
        let _ = (
            &s0, &s1, &s2, &s3, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &b0, &b1, &q4k, &q6k, &dummy,
        );
        Ok(())
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
    /// Borrow the (CPU-side) weights handle. MEMIT uses this to
    /// dequantize ffn_down at each layer for the `R = V − W·K`
    /// residual computation.
    pub fn weights(&self) -> &Weights {
        &self.weights
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

    /// Re-allocate the per-layer KV cache buffers at a smaller `max_context`.
    /// Discards any cached content (kv_lens reset to 0, pos = 0) and returns
    /// the previous `max_context` so the caller can restore on demand.
    ///
    /// Use case: chat sessions reserve `max_context` positions (~600 MB at
    /// 4096 on gemma4:e2b) which training's NextToken loss only needs 1
    /// position of. Calling `shrink_kv(seq_len + 1)` before `TrainingSession::new`
    /// frees the bulk of that allocation back to the WebGPU device for the
    /// training scratch / LoRA / Adam buffers. `trainingFinish` calls
    /// `shrink_kv(original_max)` to put chat back to its full cache.
    ///
    /// Returns an error if `new_max_context` is 0 or larger than the
    /// hardware-cap `MAX_CONTEXT`. Larger-than-current values are allowed
    /// (used by the restore path on `trainingFinish`).
    pub fn shrink_kv(&mut self, new_max_context: u32) -> Result<u32> {
        if new_max_context == 0 || new_max_context > MAX_CONTEXT {
            return Err(RullamaError::Inference(format!(
                "shrink_kv: new_max_context={new_max_context} out of range (1..={MAX_CONTEXT})"
            )));
        }
        let device = &self.ctx.device;
        let n_layers = self.cfg.n_layers as usize;
        let prev = self.max_context;

        // Re-allocate non-donor K/V buffers at the new size, then re-build
        // the donor aliasing the same way the constructor does. Any stale
        // KV content is dropped — that's the contract of shrink (callers
        // call reset implicitly).
        let mut kv_k_opt: Vec<Option<Arc<wgpu::Buffer>>> = vec![None; n_layers];
        let mut kv_v_opt: Vec<Option<Arc<wgpu::Buffer>>> = vec![None; n_layers];
        for i in 0..n_layers {
            if self.donor_map[i].is_none() {
                let n_kv = self.cfg.n_kv_heads(i as u32) as usize;
                let hd = self.cfg.head_dim(i as u32) as usize;
                let bytes = (new_max_context as usize * n_kv * hd * 4) as u64;
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
            if let Some(d) = self.donor_map[i] {
                kv_k_opt[i] = kv_k_opt[d as usize].clone();
                kv_v_opt[i] = kv_v_opt[d as usize].clone();
            }
        }
        self.kv_k = kv_k_opt.into_iter().map(|x| x.unwrap()).collect();
        self.kv_v = kv_v_opt.into_iter().map(|x| x.unwrap()).collect();
        for l in self.kv_lens.iter_mut() {
            *l = 0;
        }
        self.pos = 0;
        self.max_context = new_max_context;
        Ok(prev)
    }

    /// Current `max_context` — the cap on how many tokens the per-layer
    /// K/V buffers can hold. Useful for snapshotting before `shrink_kv`
    /// so the caller can restore.
    pub fn max_context(&self) -> u32 {
        self.max_context
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
        self.step_inner(token_id, None, None, None).await
    }

    /// Run one forward step **with ROME residual perturbation**.
    ///
    /// After layer `rome_delta.target_layer` has finished writing into
    /// `self.hidden`, this function appends a `residual_add` of
    /// `rome_delta.delta_buf` (shape `[d_model]`) to the running
    /// hidden state — *only* on the step that processes the
    /// subject-last token. The caller is responsible for invoking this
    /// method only at the subject-last position; for every other
    /// prompt position, use the plain [`Forward::step_capture`].
    ///
    /// This is ROME Phase 2.b's δ-injection path. Mirrors the
    /// `edit_output_fn` hook in kmeng01/rome's `compute_v.py` where
    /// the optimized residual delta is added at the MLP output of the
    /// target layer for the fact-lookup position.
    pub async fn step_capture_with_rome_delta<'a>(
        &mut self,
        token_id: u32,
        capture: &'a [LayerCaptureBuffers<'a>],
        rome_delta: RomeDeltaInjection<'a>,
    ) -> Result<Vec<f32>> {
        if capture.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_capture_with_rome_delta: got {} capture layers, expected {}",
                capture.len(),
                self.cfg.n_layers
            )));
        }
        if rome_delta.target_layer >= self.cfg.n_layers {
            return Err(RullamaError::Inference(format!(
                "step_capture_with_rome_delta: target_layer {} >= n_layers {}",
                rome_delta.target_layer, self.cfg.n_layers
            )));
        }
        self.step_inner_with_progress(token_id, Some(capture), None, None, None, Some(rome_delta))
            .await
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
        globals: Option<&'a GlobalLoraSlots<'a>>,
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
        self.step_inner(token_id, Some(capture), loras, globals)
            .await
    }

    /// **ROME Phase 2.b auxiliary backward at a non-loss position.**
    ///
    /// Mirrors the K/V projection backward in `backward_layer` but at
    /// `target_pos` (e.g., the subject-last token) instead of the
    /// loss position. Required because the main backward's `d_hidden`
    /// holds gradient only at `hidden_input[target_layer+1, loss_pos]`,
    /// not at `hidden_input[target_layer+1, target_pos]` — and ROME
    /// edits at the subject's residual cell, not the loss-position
    /// residual cell.
    ///
    /// Inputs (all already populated by a prior `backward_step_with_progress`
    /// with `backward_layer_floor = target_layer + 1`):
    ///   * `scratch.d_k_hist` / `scratch.d_v_hist` — gradients at layer
    ///     `target_layer + 1`'s K/V at every history position
    ///   * `captures[target_layer + 1].{k_pre_norm, v_pre_norm,
    ///     norm_x_attn, hidden_in}` — seq-shaped activations from the
    ///     forward
    ///
    /// Writes `∂loss/∂hidden_input[target_layer+1, target_pos]` into
    /// `out_d_hidden` (shape `[d_model]`).
    ///
    /// MVP: only single-layer (target_layer + 1) contribution. Multi-
    /// layer cross-position chain is dropped (an approximation). Donor
    /// (KV-shared) layers fall back to zero. Sufficient as a first
    /// gradient routing improvement; per-history-d_hidden across all
    /// layers above L+1 is a future refinement.
    pub async fn rome_aux_backward_at_position<'a>(
        &mut self,
        captures: &'a [LayerCaptureBuffers<'a>],
        scratch: &BackwardScratchView<'a>,
        target_layer: u32,
        target_pos: u32,
        out_d_hidden: &wgpu::Buffer,
    ) -> Result<()> {
        let i_plus_one = target_layer + 1;
        if i_plus_one >= self.cfg.n_layers {
            return Err(RullamaError::Inference(format!(
                "rome_aux_backward: target_layer+1 = {i_plus_one} >= n_layers = {}",
                self.cfg.n_layers
            )));
        }
        let i_idx = i_plus_one as usize;
        let prefix = format!("blk.{i_plus_one}.");
        let d_model = self.cfg.d_model as usize;
        let n_kv_heads = self.cfg.n_kv_heads(i_plus_one) as usize;
        let head_dim = self.cfg.head_dim(i_plus_one) as usize;
        let kind = self.cfg.kind(i_plus_one);
        let eps = self.cfg.rms_norm_eps;
        let donor = self.donor_map[i_idx];

        // Donor layers don't own K/V — for MVP we just zero the
        // gradient (skipping the auxiliary contribution from this
        // layer). Future: route through the donor's K/V chain.
        if donor.is_some() {
            let zeros = vec![0.0f32; d_model];
            self.ctx
                .queue
                .write_buffer(out_d_hidden, 0, bytemuck::cast_slice(&zeros));
            return Ok(());
        }

        let cap = &captures[i_idx];
        let wc = &self.wcache;

        let k_w = wc.buffer_async(&format!("{prefix}attn_k.weight")).await?;
        let k_norm_w = wc
            .buffer_async(&format!("{prefix}attn_k_norm.weight"))
            .await?;
        let v_w_name = format!("{prefix}attn_v.weight");
        let v_w = wc.buffer_async(&v_w_name).await?;
        let v_w_dtype = wc.dtype(&v_w_name)?;
        let attn_norm_w = wc
            .buffer_async(&format!("{prefix}attn_norm.weight"))
            .await?;
        let factors_w = if matches!(kind, LayerKind::Global) {
            wc.buffer_opt_async("rope_freqs.weight").await?
        } else {
            None
        };

        let (rope_base, rope_dims) = match kind {
            LayerKind::SlidingWindow => {
                (self.cfg.rope_freq_base_swa, self.cfg.rope_dim_swa as usize)
            }
            LayerKind::Global => (self.cfg.rope_freq_base, self.cfg.rope_dim_global as usize),
        };

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.aux_backward"),
            });

        let kv_row_bytes = (n_kv_heads * head_dim * 4) as u64;
        let d_model_bytes = (d_model * 4) as u64;
        let p_kv_off = (target_pos as u64) * kv_row_bytes;
        let p_dm_off = (target_pos as u64) * d_model_bytes;

        // Window captures at target_pos (overwrites whatever the main
        // backward left in the *_window scratch buffers).
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
            kv_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.v_pre_norm,
            p_kv_off,
            scratch.v_pre_norm_window,
            0,
            kv_row_bytes,
        );
        enc.copy_buffer_to_buffer(
            cap.hidden_in,
            p_dm_off,
            scratch.hidden_in_window,
            0,
            d_model_bytes,
        );

        // ---- K backward at target_pos ----
        enc.copy_buffer_to_buffer(
            scratch.d_k_hist,
            p_kv_off,
            scratch.d_k_pre_rope,
            0,
            kv_row_bytes,
        );
        rope_neox_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            scratch.d_k_pre_rope,
            factors_w.as_ref(),
            &self.dummy,
            head_dim,
            n_kv_heads,
            target_pos as usize,
            rope_dims,
            rope_base,
        );
        rmsnorm_per_row_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            scratch.k_pre_norm_window,
            &k_norm_w,
            scratch.d_k_pre_rope,
            scratch.d_k_pre_norm,
            n_kv_heads,
            head_dim,
            eps,
            true,
        );
        matmul_q4_k_backward_input_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            &k_w,
            scratch.d_k_pre_norm,
            scratch.d_hidden_tmp,
            d_model,
            n_kv_heads * head_dim,
        );

        // ---- V backward at target_pos ----
        // d_v at target_pos → temporarily into d_k_pre_norm (K's window
        // is free after K backward completed) to feed rmsnorm_back's dy
        // without aliasing the dx output buffer (mirrors the main
        // backward's pattern).
        enc.copy_buffer_to_buffer(
            scratch.d_v_hist,
            p_kv_off,
            scratch.d_k_pre_norm,
            0,
            kv_row_bytes,
        );
        rmsnorm_per_row_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            scratch.v_pre_norm_window,
            &self.dummy,
            scratch.d_k_pre_norm,
            scratch.d_v_pre_norm,
            n_kv_heads,
            head_dim,
            eps,
            false,
        );
        match v_w_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &v_w,
                scratch.d_v_pre_norm,
                scratch.d_hidden_tmp2,
                d_model,
                n_kv_heads * head_dim,
            ),
            GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                &v_w,
                scratch.d_v_pre_norm,
                scratch.d_hidden_tmp2,
                d_model,
                n_kv_heads * head_dim,
            ),
            other => {
                return Err(RullamaError::Inference(format!(
                    "rome_aux_backward: attn_v dtype {other:?} unsupported"
                )));
            }
        }

        // K + V sum → d_hidden_tmp
        residual_add_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            scratch.d_hidden_tmp,
            scratch.d_hidden_tmp2,
            d_model,
        );

        // attn-norm rmsnorm backward → out_d_hidden
        // The forward applied rmsnorm to hidden[L+1, target_pos] to
        // produce norm_x_attn[L+1, target_pos]. The K/V matmul-back
        // gave us d_norm_x_attn (in scratch.d_hidden_tmp); now we walk
        // back through that rmsnorm to recover gradient on the input
        // (hidden_in at target_pos with δ already applied).
        rmsnorm_backward_chained(
            &self.ctx,
            &self.pipes,
            &mut enc,
            scratch.hidden_in_window,
            &attn_norm_w,
            scratch.d_hidden_tmp,
            out_d_hidden,
            d_model,
            eps,
            true,
        );

        self.ctx.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// ROME δ-injection: adds `delta_buf` (shape `[d_model]`) to
    /// `self.hidden` immediately after the named layer completes, on
    /// exactly the step this is passed to. Used by the iterative v\*
    /// loop (Phase 2.b) to apply the optimizer's current residual
    /// perturbation at the subject-last token's position.
    pub fn rome_delta_buf_alloc(&self) -> wgpu::Buffer {
        let d_model = self.cfg.d_model as u64;
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rome.delta"),
            size: d_model * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Run a forward step with LoRA correction enabled but **without**
    /// capturing activations. Used for the prompt-prefill pass during
    /// training (positions 0..N-2 just fill KV; only the final position
    /// is captured + has its loss measured).
    pub async fn step_with_lora<'a>(
        &mut self,
        token_id: u32,
        loras: &'a [LayerLoraSlots<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
    ) -> Result<Vec<f32>> {
        if loras.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_lora: got {} lora slots, expected {}",
                loras.len(),
                self.cfg.n_layers
            )));
        }
        self.step_inner(token_id, None, Some(loras), globals).await
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
        globals: Option<&'a GlobalLoraSlots<'a>>,
    ) -> Result<Vec<f32>> {
        self.step_with_lora_seqcap_with_progress(token_id, loras, capture, globals, None)
            .await
    }

    /// Variant of [`step_with_lora_seqcap`] that fires
    /// `progress_cb(layer_index, total_layers, "forward")` between
    /// per-layer encoder submits. Used by training to drive a
    /// detailed status indicator without rewriting the existing
    /// callers that don't care about per-layer ticks.
    pub async fn step_with_lora_seqcap_with_progress<'a>(
        &mut self,
        token_id: u32,
        loras: &'a [LayerLoraSlots<'a>],
        capture: &'a [LayerCaptureBuffers<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
        progress_cb: Option<&LayerProgressCb<'_>>,
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
        self.step_inner_with_progress(
            token_id,
            Some(capture),
            Some(loras),
            globals,
            progress_cb,
            None,
        )
        .await
    }

    async fn step_inner<'a>(
        &mut self,
        token_id: u32,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
        globals: Option<&'a GlobalLoraSlots<'a>>,
    ) -> Result<Vec<f32>> {
        self.step_inner_with_progress(token_id, capture, loras, globals, None, None)
            .await
    }

    async fn step_inner_with_progress<'a>(
        &mut self,
        token_id: u32,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
        globals: Option<&'a GlobalLoraSlots<'a>>,
        progress_cb: Option<&LayerProgressCb<'_>>,
        rome_delta: Option<RomeDeltaInjection<'a>>,
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

        // ---- embed_tokens LoRA forward inject ----
        // `hidden += scale · B_emb · A_emb[:, token_id]`. With the input
        // being effectively one_hot(token_id), the matmul `A_emb @ one_hot`
        // reduces to a column extract from A_emb. We capture the column in
        // slot.z so the backward pass can reconstruct the same `z` vector
        // without re-running the column read.
        if let Some(g) = globals
            && let Some(embed) = g.embed_tokens.as_ref()
        {
            let vocab = self.cfg.vocab_size;
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.embed_tokens_lora"),
                });
            crate::backend::dispatch::lora_embed_col_read_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                embed.a,
                embed.z,
                embed.rank,
                vocab,
                token_id,
            );
            // hidden += scale · B_emb · z, where B_emb has shape [d_model, rank].
            crate::backend::dispatch::lora_matmul_row_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                embed.b,
                embed.z,
                &self.hidden,
                embed.rank as usize,
                d_model,
                embed.scale,
                true, // accumulate into hidden
            );
            self.ctx.queue.submit(Some(enc.finish()));
        }

        self.run_forward_from_hidden_with_progress(capture, loras, globals, progress_cb, rome_delta)
            .await
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
        self.step_with_embedding_inner(embedding, None, None).await
    }

    /// Variant of [`step_with_embedding`] that applies a LoRA adapter
    /// to every layer's q/k/v/o (+ optional FFN) during the forward.
    /// Used by `Model::step_with_embedding_native` when an inference
    /// adapter is active — without this, image and audio soft-token
    /// steps would silently bypass the loaded adapter while pure-text
    /// steps respect it.
    ///
    /// `globals.lm_head` is honored (logit correction after the tiled
    /// output projection). `globals.embed_tokens` is ignored: this
    /// path bypasses the `token_embd` lookup, so there's no input-
    /// embedding distribution for the embed_tokens LoRA to perturb.
    pub async fn step_with_embedding_with_lora<'a>(
        &mut self,
        embedding: &[f32],
        loras: &'a [LayerLoraSlots<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
    ) -> Result<Vec<f32>> {
        if loras.len() != self.cfg.n_layers as usize {
            return Err(RullamaError::Inference(format!(
                "step_with_embedding_with_lora: got {} lora slots, expected {}",
                loras.len(),
                self.cfg.n_layers
            )));
        }
        self.step_with_embedding_inner(embedding, Some(loras), globals)
            .await
    }

    async fn step_with_embedding_inner<'a>(
        &mut self,
        embedding: &[f32],
        loras: Option<&'a [LayerLoraSlots<'a>]>,
        globals: Option<&'a GlobalLoraSlots<'a>>,
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

        self.run_forward_from_hidden(None, loras, globals).await
    }

    /// Forward pass starting from `self.hidden` already populated. Shared by
    /// `step` (token-id path) and `step_with_embedding` (multimodal soft tokens).
    async fn run_forward_from_hidden<'a>(
        &mut self,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
        globals: Option<&'a GlobalLoraSlots<'a>>,
    ) -> Result<Vec<f32>> {
        self.run_forward_from_hidden_with_progress(capture, loras, globals, None, None)
            .await
    }

    /// Variant of [`run_forward_from_hidden`] that fires
    /// `progress_cb(layer_index, total_layers, "forward")` between
    /// per-layer encoder submits. Used by training; chat-side
    /// inference passes `None`.
    ///
    /// `rome_delta`: optional ROME residual-stream perturbation.
    /// When `Some(..)`, after the layer matching `target_layer`
    /// completes, `delta_buf` is added to `self.hidden` before the
    /// next layer (or final norm) consumes it.
    ///
    /// `globals`: optional `lm_head` / `embed_tokens` LoRA slots. The
    /// `lm_head` slot is consumed here (added to logits after the
    /// tiled output projection). The `embed_tokens` slot must have
    /// been applied by the caller before populating `self.hidden`
    /// (see `step_inner_with_progress`).
    async fn run_forward_from_hidden_with_progress<'a>(
        &mut self,
        capture: Option<&'a [LayerCaptureBuffers<'a>]>,
        loras: Option<&'a [LayerLoraSlots<'a>]>,
        globals: Option<&'a GlobalLoraSlots<'a>>,
        progress_cb: Option<&LayerProgressCb<'_>>,
        rome_delta: Option<RomeDeltaInjection<'a>>,
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
            // **MeBP-inspired per-layer weight destroy.** When enabled
            // (training mode on memory-tight targets — see
            // forward_destroy_per_layer field doc), drain the GPU
            // (force the just-submitted layer's commands to complete
            // so no bind group still references blk.{i}.* buffers),
            // then destroy this layer's weight tiles. Peak weight cache
            // during forward drops from ~1417 MiB (all 35 layers) to
            // ~40 MiB (one layer at a time). Backward's
            // gradient-checkpointing recompute re-fetches via the same
            // lazy buffer_async path the forward used originally —
            // identical correctness, ~32-42% extra forward time (per
            // MeBP arxiv 2510.03425 §4.2). This is the smallest-change
            // approximation of MeBP's per-block lazy-load architecture
            // adapted to wgpu + OPFS (our analog to their mmap).
            // Only destroy below the backward floor — layers at or above
            // the floor are walked in backward and their weights stay
            // cached so the recompute hits the WeightCache instead of
            // re-uploading. (iPhone real-device test confirmed: with
            // unconditional destroy, recompute alloc churn killed the
            // page immediately after `bwd.post_yield`. With the floor
            // gating, backward layer 34's recompute finds blk.34 still
            // in cache.)
            if self.forward_destroy_per_layer && i < self.forward_destroy_layer_floor {
                let _drain = read_buf_stats(&self.ctx, &self.hidden, 1).await?;
                let _ = self.wcache.drop_blk_layer_range_destroy(i, i + 1);
            }
            // Per-layer progress beacon — fired AFTER the submit so
            // the caller's "layer N done" message correlates with the
            // GPU having actually finished it. `i + 1` is 1-based for
            // "N of n_layers" UX semantics.
            if let Some(cb) = progress_cb {
                cb("forward", i + 1, n_layers as u32);
            }
            enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.token_encoder.cont"),
                });

            // ROME δ injection: append `hidden += delta_buf` to the
            // fresh encoder immediately after `target_layer` settles
            // on the GPU. The next iteration's `encode_layer(i+1)`
            // (or the final norm, if this was the last layer) reads
            // the perturbed hidden. Cross-submit ordering guarantees
            // the previous submit's write to `self.hidden` is visible.
            if let Some(rd) = rome_delta.as_ref()
                && i == rd.target_layer
            {
                residual_add_chained(
                    &self.ctx,
                    &self.pipes,
                    &mut enc,
                    &self.hidden,
                    rd.delta_buf,
                    d_model,
                );
            }
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

        // ---- lm_head LoRA forward inject ----
        // `logits += scale · B_lmh · (A_lmh · norm_x)` where A_lmh has
        // shape [rank, d_model] and B_lmh has shape [vocab, rank]. We
        // capture `z = A_lmh · norm_x` into slot.z so the backward pass
        // can reuse it without re-running the matmul. Injected AFTER the
        // tiled output projection (so the base logits exist) but BEFORE
        // softcap (so the correction sees the same softcap as the base).
        if let Some(g) = globals
            && let Some(lmh) = g.lm_head.as_ref()
        {
            let vocab = self.cfg.vocab_size as usize;
            // Fused: z = A_lmh · norm_x; logits += scale · B_lmh · z.
            // One dispatch instead of two; slot.z is still written for
            // the backward path to consume. When the inference adapter
            // packed B as f16 (vocab × rank ≈ 16 MB → 8 MB), route
            // through the packed-f16 kernel; otherwise use the f32
            // variant. Training never sets `b_is_f16` so the backward
            // path (which reads B as f32) stays correct by construction.
            if lmh.b_is_f16 {
                crate::backend::dispatch::lora_matmul_fused_f16b_chained(
                    &self.ctx,
                    &self.pipes,
                    &mut enc,
                    lmh.a,
                    lmh.b,
                    &self.norm_x,
                    &self.logits,
                    lmh.z,
                    d_model,
                    vocab,
                    lmh.rank as usize,
                    lmh.scale,
                    true,
                );
            } else {
                lora_matmul_fused_chained(
                    &self.ctx,
                    &self.pipes,
                    &mut enc,
                    lmh.a,
                    lmh.b,
                    &self.norm_x,
                    &self.logits,
                    lmh.z,
                    d_model,
                    vocab,
                    lmh.rank as usize,
                    lmh.scale,
                    true,
                );
            }
            self.ctx.queue.submit(Some(enc.finish()));
            enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd.out_proj_encoder.cont2"),
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

        // ---- LoRA forward correction (q) — fused ----
        // Fused into ONE dispatch: z=A·norm_x AND self.q+=scale·B·z.
        // slot.z is still written by the kernel for the backward path.
        if let Some(slot) = loras.and_then(|l| l.q.as_ref()) {
            lora_matmul_fused_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                slot.b,
                &self.norm_x,
                &self.q,
                slot.z,
                d_model,
                n_heads * head_dim,
                slot.rank as usize,
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

            // ---- LoRA forward correction (k) — fused ----
            if let Some(slot) = loras.and_then(|l| l.k.as_ref()) {
                lora_matmul_fused_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.a,
                    slot.b,
                    &self.norm_x,
                    &self.k,
                    slot.z,
                    d_model,
                    n_kv_heads * head_dim,
                    slot.rank as usize,
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

            // ---- LoRA forward correction (v) — fused ----
            if let Some(slot) = loras.and_then(|l| l.v.as_ref()) {
                lora_matmul_fused_chained(
                    &self.ctx,
                    &self.pipes,
                    enc,
                    slot.a,
                    slot.b,
                    &self.norm_x,
                    &self.v,
                    slot.z,
                    d_model,
                    n_kv_heads * head_dim,
                    slot.rank as usize,
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

        // ---- LoRA forward correction (o) — fused ----
        if let Some(slot) = loras.and_then(|l| l.o.as_ref()) {
            lora_matmul_fused_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                slot.b,
                &self.attn_out_buf,
                &self.attn_proj,
                slot.z,
                n_heads * head_dim,
                d_model,
                slot.rank as usize,
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

        // ---- LoRA forward correction (ffn_gate) — fused ----
        if let Some(slot) = loras.and_then(|l| l.ffn_gate.as_ref()) {
            lora_matmul_fused_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                slot.b,
                &self.norm_x,
                &self.ffn_gate,
                slot.z,
                d_model,
                ffn_n,
                slot.rank as usize,
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

        // ---- LoRA forward correction (ffn_up) — fused ----
        if let Some(slot) = loras.and_then(|l| l.ffn_up.as_ref()) {
            lora_matmul_fused_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                slot.b,
                &self.norm_x,
                &self.ffn_up,
                slot.z,
                d_model,
                ffn_n,
                slot.rank as usize,
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

        // ---- LoRA forward correction (ffn_down) — fused ----
        if let Some(slot) = loras.and_then(|l| l.ffn_down.as_ref()) {
            lora_matmul_fused_chained(
                &self.ctx,
                &self.pipes,
                enc,
                slot.a,
                slot.b,
                &self.ffn_act,
                &self.ffn_out,
                slot.z,
                ffn_n,
                d_model,
                slot.rank as usize,
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

/// Model-global LoRA gradient accumulators. The `embed_tokens` pair
/// is updated by a single-column scatter add (since the input is
/// one-hot at the position of the current token). The `lm_head` pair
/// is updated by full matmul backward against d_logits, and feeds an
/// additional `d_norm_x_lmh_tmp` contribution that gets added into
/// the trunk gradient stream (so the per-layer backward sees a
/// d_hidden that already accounts for the lm_head LoRA's chain rule).
#[derive(Default)]
pub struct GlobalLoraGrads<'a> {
    pub embed_tokens: Option<LoraGradPair<'a>>,
    pub lm_head: Option<LoraGradPair<'a>>,
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
        self.backward_step_with_progress(
            target_id,
            capture,
            loras,
            grads,
            None, // no global LoRA slots from this convenience wrapper
            None, // no global LoRA grads from this convenience wrapper
            None, // no embed_token_id; not needed when globals are None
            scratch,
            history_len,
            pos,
            recompute_captures,
            None,
            0, // backward_layer_floor = 0 → backprop all layers (default)
        )
        .await
    }

    /// Backward pass starting from CALLER-PROVIDED `d_logits` (instead
    /// of computing them from a hard-label target via cross-entropy).
    ///
    /// Used by ROME's iterative v\* loop to backpropagate the KL term
    /// `kl_factor · KL(P_base ‖ P_edited)` whose gradient at the edited
    /// logits is `kl_factor · (softmax(edited) − softmax(base))` — a
    /// soft-label CE-like gradient our hard-label kernel can't produce.
    ///
    /// `custom_d_logits` is a length-`vocab_size` f32 slice written
    /// directly into `scratch.d_logits`, after which the rest of the
    /// backward chain (output-proj-back → final-norm-back → layer walk)
    /// runs identically to `backward_step_with_progress`.
    ///
    /// The scalar in `scratch.loss` is NOT populated by this path — the
    /// caller is responsible for computing the loss value on CPU.
    /// Returns 0.0 as a placeholder.
    #[allow(clippy::too_many_arguments)]
    pub async fn backward_step_from_d_logits_with_progress<'a>(
        &mut self,
        custom_d_logits: &[f32],
        capture: &'a [LayerCaptureBuffers<'a>],
        loras: &'a [LayerLoraSlots<'a>],
        grads: &'a [LayerLoraGrads<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
        global_grads: Option<&'a GlobalLoraGrads<'a>>,
        embed_token_id: Option<u32>,
        scratch: &'a BackwardScratchView<'a>,
        history_len: u32,
        pos: u32,
        recompute_captures: bool,
        progress_cb: Option<&LayerProgressCb<'_>>,
        backward_layer_floor: u32,
    ) -> Result<f32> {
        // Upload caller's d_logits into scratch.d_logits, then call the
        // shared inner backward with `target_id = u32::MAX` as a sentinel
        // that means "skip the CE step, d_logits is already populated".
        let vocab = self.cfg.vocab_size as usize;
        if custom_d_logits.len() != vocab {
            return Err(RullamaError::Inference(format!(
                "backward_step_from_d_logits: custom_d_logits len {} != vocab {vocab}",
                custom_d_logits.len()
            )));
        }
        self.ctx
            .queue
            .write_buffer(scratch.d_logits, 0, bytemuck::cast_slice(custom_d_logits));
        self.backward_step_inner(
            u32::MAX,
            capture,
            loras,
            grads,
            globals,
            global_grads,
            embed_token_id,
            scratch,
            history_len,
            pos,
            recompute_captures,
            progress_cb,
            backward_layer_floor,
            true, // skip CE
        )
        .await
    }

    /// Variant of [`backward_step`] that fires
    /// `progress_cb(layer_index, total_layers, "backward")` between
    /// per-layer encoder submits. The layer index in the callback is
    /// the **logical position** (1..=n_layers walking top-down), so
    /// a 35-layer model fires `(1, 35) ... (35, 35)` mirroring the
    /// forward beacon order — friendlier for the UI to render than
    /// the actual reverse-walk index `(n_layers-1) ... 0`.
    #[allow(clippy::too_many_arguments)]
    pub async fn backward_step_with_progress<'a>(
        &mut self,
        target_id: u32,
        capture: &'a [LayerCaptureBuffers<'a>],
        loras: &'a [LayerLoraSlots<'a>],
        grads: &'a [LayerLoraGrads<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
        global_grads: Option<&'a GlobalLoraGrads<'a>>,
        // The token id whose embedding row this backward step corresponds
        // to — required when `globals.embed_tokens` and
        // `global_grads.embed_tokens` are both Some, because the embed
        // gradient is a single-column scatter into `d_A[:, embed_token_id]`.
        // Pass `None` if the embed_tokens LoRA is not in use.
        embed_token_id: Option<u32>,
        scratch: &'a BackwardScratchView<'a>,
        history_len: u32,
        pos: u32,
        recompute_captures: bool,
        progress_cb: Option<&LayerProgressCb<'_>>,
        // **Truncated backward.** When > 0, exit the per-layer
        // reverse walk early once `li < backward_layer_floor`. Layers
        // below the floor get no gradient updates, saving compute +
        // memory transients. 0 keeps every layer trainable (the
        // production default). See `TrainingHyperparams::backward_layer_floor`.
        backward_layer_floor: u32,
    ) -> Result<f32> {
        self.backward_step_inner(
            target_id,
            capture,
            loras,
            grads,
            globals,
            global_grads,
            embed_token_id,
            scratch,
            history_len,
            pos,
            recompute_captures,
            progress_cb,
            backward_layer_floor,
            false, // run CE step
        )
        .await
    }

    /// Shared inner backward, with `skip_ce` toggling whether to call
    /// `cross_entropy_backward_chained` (false → hard-label CE, target_id
    /// drives d_logits) or use pre-populated `scratch.d_logits` (true).
    #[allow(clippy::too_many_arguments)]
    async fn backward_step_inner<'a>(
        &mut self,
        target_id: u32,
        capture: &'a [LayerCaptureBuffers<'a>],
        loras: &'a [LayerLoraSlots<'a>],
        grads: &'a [LayerLoraGrads<'a>],
        globals: Option<&'a GlobalLoraSlots<'a>>,
        global_grads: Option<&'a GlobalLoraGrads<'a>>,
        embed_token_id: Option<u32>,
        scratch: &'a BackwardScratchView<'a>,
        history_len: u32,
        pos: u32,
        recompute_captures: bool,
        progress_cb: Option<&LayerProgressCb<'_>>,
        backward_layer_floor: u32,
        skip_ce: bool,
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

        // **iOS peak-memory cut at the forward→backward boundary.** The
        // forward pass caches every one of the 35 layers' dequantized
        // f32 weight tiles (`blk.{i}.*`) in the GPU WeightCache, and the
        // cache never evicts. By the time backward starts, that resident
        // set + KV cache + per-layer activation captures + the backward
        // scratch/pipelines crosses iOS Safari's ~3-4 GB WebContent
        // ceiling, and jetsam kills the tab during the very first
        // backward dispatch (observed live: forward prefill completes,
        // then the tab dies before the first `head_ce` beacon).
        //
        // Destroy the layer weights now — and this MUST be a real
        // `destroy()`, not a handle-drop. A plain `drop_prefix("blk.")`
        // only releases the Rust `wgpu::Buffer` handles; on iOS Safari
        // WebGPU the ~1417 MiB of `GPUBuffer` memory stays physically
        // resident until GC, which never runs inside a synchronous
        // training step. The on-device beacon trail proved it: forward
        // peaked at gpuMiB=1417, the head then fetched `token_embd`
        // (~637 MiB) on top of those un-reclaimed buffers, and the tab
        // jetsam'd at the head section (~2 GB real RSS) — even though our
        // *tracked* counter had dropped to 668. Handle-drop lies to the
        // accountant; it doesn't free iOS RSS.
        //
        // `destroy()` here is safe specifically because we're at a
        // GPU-idle point: the forward's final act was
        // `read_back_f32(&self.logits_read).await` (logits readback),
        // whose `map_async` only resolves after every prior submit —
        // including the last layer — has completed. No in-flight command
        // references `blk.*` at this instant. (The use-after-destroy we
        // hit before was destroying at the *head→backward* transition,
        // where the just-submitted head dispatches were still pending;
        // that's a different, later point. The head reads `token_embd` /
        // `output_norm` / `per_layer_*` — none match `blk.` — so
        // destroying the block weights here cannot pull a buffer out from
        // under a pending head submit.)
        //
        // **Targeted destroy: drop ONLY layers BELOW the backward floor,
        // keep layers IN the backward walk cached.** Real-device trail
        // showed the page still jetsam'd at the head→backward transition
        // with the original "destroy all blk" strategy because the
        // backward immediately re-fetched (re-allocated) blk.{floor..N-1}
        // — those allocations on top of the head's in-flight Metal state
        // tripped jetsam. Layers 0..floor are never touched in backward
        // (no recompute, no dispatch); destroying them frees the bulk of
        // the forward heap (25 × ~40 MiB = ~1000 MiB on e2b with floor=25).
        // Layers floor..n_layers stay cached so recompute + backward are
        // cache HITS — no re-allocation, no Metal-heap churn at the
        // head→backward boundary.
        let n_layers_u32 = self.cfg.n_layers;
        let floor_u32 = backward_layer_floor.min(n_layers_u32);
        // Single-pass destroy of blk.{0..floor}.*  — replaces a per-layer
        // loop that fired ~floor × HashMap-traversals + ~floor × ~7
        // GPUBuffer.destroy() IPC dispatches on iOS Safari (real-device
        // trail: `forward 35/35 gpuMiB=1417` → jetsam in the gap before
        // head_ce). One traversal, one batch of destroys.
        //
        // **No drain needed here.** Patch 2's BindGroupCache invalidation
        // hooks into `drop_blk_layer_range_destroy` and evicts any
        // cached bind group whose key references one of the buffers
        // about to be destroyed BEFORE the underlying `Buffer::destroy()`
        // runs (see `BindGroupCache::invalidate_buffers` and the call in
        // `WeightCache::drop_blk_layer_range_destroy`). That removes the
        // class of use-after-destroy that an earlier wasm32 drain was
        // guarding against. WebGPU spec §3.4.3.1: commands already
        // encoded using a destroyed buffer continue to execute normally.
        // **Chunked destroy with JS yields between (Patch 8).** The
        // single `drop_blk_layer_range_destroy(0, floor_u32)` call was
        // firing ~floor × ~7 = ~168 `Buffer::destroy()` IPCs in one
        // synchronous burst, right at the moment GPUProcess RSS is at
        // its peak from the just-finished forward. iOS jetsam was
        // killing the WebContent process inside this burst — observed:
        // `forward 35/35 gpuMiB=1417` was the last beacon, with
        // `head_ce` never firing.
        //
        // Split the destroy into chunks of CHUNK_LAYERS layers, with a
        // JS event-loop yield (setTimeout 0) between each chunk. Each
        // chunk fires ~CHUNK_LAYERS × ~7 = ~35 destroys; iOS Metal can
        // process each batch's IPCs before the next batch lands.
        // No-op on native (the yield is wasm32-only).
        // Chunk size is 1 (= a single all-at-once destroy) when mobile
        // mode is off — the chunked-with-yields pattern only buys
        // anything when iOS Metal needs the IPC backlog to drain. On
        // Mac browsers / native the GPUProcess keeps up fine, so the
        // yields are pure latency.
        let chunk_layers: u32 = if self.mobile_mode {
            5
        } else {
            floor_u32.max(1)
        };
        let mut dropped: usize = 0;
        let mut chunk_start = 0u32;
        while chunk_start < floor_u32 {
            let chunk_end = (chunk_start + chunk_layers).min(floor_u32);
            dropped += wc.drop_blk_layer_range_destroy(chunk_start, chunk_end);
            chunk_start = chunk_end;
            // Mobile-mode-only: yield to JS event loop so iOS Metal can
            // drain the destroy IPCs from this chunk before the next
            // batch. On desktop browsers this would just be wasted ms.
            #[cfg(target_arch = "wasm32")]
            if self.mobile_mode && chunk_start < floor_u32 {
                use wasm_bindgen::JsCast;
                let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global()
                    .dyn_into()
                    .expect("training session runs inside a DedicatedWorkerGlobalScope");
                let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                    let resolve_fn: js_sys::Function = resolve.into();
                    let _ =
                        scope.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve_fn, 0);
                });
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            }
        }
        #[cfg(target_arch = "wasm32")]
        if dropped > 0 {
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                "[bwd] evicted {dropped} layer weight tiles before backward (chunked, iOS peak-memory cut)"
            )));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if dropped > 0 && std::env::var("RULLAMA_TRACE_EVICT").is_ok() {
            eprintln!("[bwd] evicted {dropped} layer weight tiles before backward (chunked)");
        }

        let final_norm = wc.buffer_async("output_norm.weight").await?;
        let token_embd = wc.buffer_async("token_embd.weight").await?;
        let token_embd_dtype = wc.dtype("token_embd.weight")?;

        // ===== Head: CE → output_proj_back → lm_head LoRA → final norm =====
        //
        // **iOS-tight invariant: one CommandEncoder per kernel group, with
        // submit() boundaries between groups.** iOS Safari's Metal driver
        // does lazy pipeline codegen — the kernel binary is compiled the
        // first time the pipeline is bound for execution, not when
        // `create_compute_pipeline` returns. If we pack 7+ never-before-
        // dispatched training kernels into ONE submit, Metal must compile
        // all of them before the GPU queue can run, and the transient
        // memory spike on iOS WebContent is enough to trip jetsam (we
        // observed this crash live: prefill completed cleanly but the
        // first head-section dispatch hard-killed the tab).
        //
        // Splitting into per-group submits with `progress_cb` beacons in
        // between gives Metal a chance to compile + reclaim transient
        // memory one pipeline at a time, and gives the post-crash log
        // a phase trail that pinpoints any future regression.

        // **Head section keeps its 4 sub-submits (Patch 4 partial revert).**
        // The first iPhone real-device test of the head-collapse variant
        // died at `step 2 forward 35/35` — earlier than the pre-collapse
        // wall — because the single collapsed head submit packed the
        // 262K × 1536 outproj matmul together with CE_back, the 4
        // lm_head LoRA dispatches, and rmsnorm_back. That single
        // monster submit overflowed iOS Metal's heap reservation
        // window. The OLD 4-submit shape gave each sub-phase (especially
        // the giant outproj) its own command buffer, which Metal handles
        // fine.
        //
        // `backward_layer` per-phase collapse (which removes 5 submits
        // per layer × 10 layers = 50 IPCs/step) is kept — `backward_layer`
        // doesn't have a single-dispatch monster like outproj, just many
        // medium dispatches that benefit from batching into one submit.
        //
        // `token_embd` destroy stays inline after head_outproj (early
        // destroy), restoring the prior shape that reached bwd.loop.enter
        // consistently before this patch.

        // ─── (1) Cross-entropy backward ──────────────────────────────
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd.head.ce"),
            });
        // d_logits + scalar loss — unless caller pre-populated
        // scratch.d_logits (e.g. for KL preservation in ROME).
        if !skip_ce {
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
        }
        self.ctx.queue.submit(Some(enc.finish()));
        if let Some(cb) = progress_cb {
            cb("head_ce", 1, 4);
        }

        // ─── (2) Output projection backward (embedᵀ · d_logits) ──────
        // **Vocab-axis tiled (Patch 6).** The non-tiled dispatch was the
        // largest single instruction in a training step: at vocab=262144
        // it brought ~400 MB of dequantized f32 through Metal's
        // execution path in ONE command buffer, and was the most likely
        // single cause of iOS jetsam in the head section. Tile along
        // the vocab axis: each tile dispatches over j ∈
        // [t*vocab/N, (t+1)*vocab/N), gets its OWN command encoder +
        // submit, so each Metal command buffer carries ~1/N of the
        // working set. arxiv 2604.02344 confirms Safari Metal prefers
        // tiled matmul (2× speedup on the same total work).
        //
        // Math is identical: `Σ_{j=0..n} dy[j] · W[j,i] = Σ_t Σ ...`.
        // Tile 0 writes (accumulate=false); tiles 1..N add into
        // scratch.d_hidden_final.
        // Tile count: 8 in mobile mode (the iOS Metal heap working-set
        // win), 1 on desktop browsers / native (one big command buffer
        // is faster — fewer queue.submit IPCs and the GPU runs at
        // its natural throughput).
        let vocab_tiles: u32 = if self.mobile_mode { 8 } else { 1 };
        let vocab_u32 = vocab as u32;
        let tile_size = vocab_u32.div_ceil(vocab_tiles);
        for t in 0..vocab_tiles {
            let j_start = t * tile_size;
            let j_end = (j_start + tile_size).min(vocab_u32);
            if j_start >= j_end {
                break;
            }
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bwd.head.outproj.tile"),
                });
            let accumulate = t > 0;
            match token_embd_dtype {
                GgmlDtype::Q6_K => matmul_q6_k_backward_input_tile_chained(
                    &self.ctx,
                    &self.pipes,
                    &mut enc,
                    &token_embd,
                    scratch.d_logits,
                    scratch.d_hidden_final,
                    d_model,
                    vocab,
                    j_start,
                    j_end,
                    accumulate,
                ),
                GgmlDtype::Q4_K => matmul_q4_k_backward_input_tile_chained(
                    &self.ctx,
                    &self.pipes,
                    &mut enc,
                    &token_embd,
                    scratch.d_logits,
                    scratch.d_hidden_final,
                    d_model,
                    vocab,
                    j_start,
                    j_end,
                    accumulate,
                ),
                other => {
                    return Err(RullamaError::Inference(format!(
                        "backward_step: token_embd dtype {other:?} unsupported"
                    )));
                }
            }
            self.ctx.queue.submit(Some(enc.finish()));
        }
        if let Some(cb) = progress_cb {
            cb("head_outproj", 2, 4);
        }

        // Destroy token_embd inline after the outproj submit. Patch 2's
        // bind_cache.invalidate_buffers fires inside drop_prefix_destroy
        // BEFORE Buffer::destroy(), so the outproj submit (still being
        // consumed by Metal) keeps its bind-group reference safely.
        // Mac fast path keeps token_embd resident — saves re-fetch.
        if self.mobile_mode {
            let _embd_dropped_early = wc.drop_prefix_destroy("token_embd");
            if let Some(cb) = progress_cb {
                cb("bwd.head.early_destroy_embd", 0, 1);
            }
        }

        // ─── (3) lm_head LoRA backward (optional) ────────────────────
        if let (Some(g), Some(gg)) = (globals, global_grads)
            && let (Some(slot), Some(d_pair)) = (g.lm_head.as_ref(), gg.lm_head.as_ref())
        {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bwd.head.lm_head_lora"),
                });
            let r = slot.rank as usize;
            let s = slot.scale;
            // dB += s · d_logits ⊗ z (shape [vocab, rank])
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                scratch.d_logits,
                slot.z,
                d_pair.d_b,
                vocab,
                r,
                s,
                true,
            );
            // z = Bᵀ · d_logits (rank floats; overwrites the forward-captured z)
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                slot.b,
                scratch.d_logits,
                slot.z,
                vocab,
                r,
                1.0,
                false,
            );
            // d_hidden_final += s · Aᵀ · u
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                slot.a,
                slot.z,
                scratch.d_hidden_final,
                r,
                d_model,
                s,
                true,
            );
            // dA += s · u ⊗ norm_x_final
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                &mut enc,
                slot.z,
                &self.norm_x,
                d_pair.d_a,
                r,
                d_model,
                s,
                true,
            );
            self.ctx.queue.submit(Some(enc.finish()));
        }
        if let Some(cb) = progress_cb {
            cb("head_lm_head_lora", 3, 4);
        }

        // ─── (4) Final rmsnorm backward ──────────────────────────────
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd.head.rmsnorm"),
            });
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
        if let Some(cb) = progress_cb {
            cb("head_rmsnorm", 4, 4);
        }

        // **No drain at head→backward boundary.** Earlier versions of
        // this code did a read_buf_stats(scratch.d_hidden, ...) here to
        // force iOS Metal to settle pending head submits before backward
        // started. Real-device beacon trail proved that drain itself was
        // the jetsam trigger — `head_rmsnorm 4/4 gpuMiB=38` fires (our
        // tracked memory is tiny by then), then 💥. The drain forces a
        // map_async + await on a GPU process that's at the pipeline-
        // compile RSS limit, and the sync push tips it past jetsam.
        //
        // Without the drain, the head's submits and the next layer's
        // recompute submit interleave naturally — Metal handles the
        // queuing. The MeBP per-layer destroy during forward + the early
        // token_embd destroy after head_outproj already keep the
        // *tracked* memory low; the residual jetsam pressure is iOS
        // pipeline-compile RSS, which a drain only makes worse.
        //
        // The late destroy_embd is also gone: token_embd was already
        // destroyed inline after head_outproj's submit (see ~30 lines
        // above), so any further destroy("token_embd") is a no-op.

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
        // Saturate-clamp the floor so an oversized value just means
        // "skip everything" rather than panic; a value of n_layers
        // or larger short-circuits the entire backward sweep.
        let floor = (backward_layer_floor as usize).min(n_layers);
        for li in (0..n_layers).rev() {
            // **Truncated backward gate.** Once `li` drops below the
            // configured floor, no more LoRA grads accumulate. The
            // forward pass already ran every layer (to populate the
            // captures up to the floor); we just stop walking the
            // gradient back. Layers below the floor stay frozen for
            // this step.
            if li < floor {
                break;
            }
            let i = li as u32;
            // **Diagnostic at the absolute top of each iteration** —
            // bracketed by head_rmsnorm (last head beacon) and
            // bwd.layer.recompute. If this fires for layer N but
            // bwd.layer.recompute doesn't, death is in the
            // gradient-checkpointing replay's encode_layer submit
            // (the recompute fetches blk.N.* weights again and
            // submits ~10 forward dispatches). If THIS itself doesn't
            // fire after `head_rmsnorm 4/4`, death is in the trivial
            // env_var / loop-setup code between head end and loop
            // body — which would mean iOS jetsam'd from the head's
            // accumulated Metal state, not from anything we do.
            if let Some(cb) = progress_cb {
                let logical = (n_layers as u32) - i;
                cb("bwd.loop.enter", logical, n_layers as u32);
            }
            // **JS yield REMOVED (Patch 9 diagnostic).** Across all
            // prior iPhone runs `bwd.post_yield` NEVER fired — the
            // page consistently died right at `bwd.loop.enter 1/35`,
            // before this cb could emit. Both 0 ms and 500 ms typed
            // setTimeouts produced the same wall. Tracked memory at
            // crash was 38 MiB (MeBP on) or 415 MiB (MeBP off) —
            // memory is NOT the trigger.
            //
            // If the page now reaches `bwd.post_yield` (with the
            // setTimeout gone), the yield itself was the killer:
            // iOS Safari is jetsam'ing the WebContent process while
            // the Worker is suspended in setTimeout, OR the keepalive
            // beacons from bwd.loop.enter are stacking and pushing
            // a per-request limit.
            //
            // If `bwd.post_yield` still doesn't fire, the issue is
            // immediately downstream — most likely the recompute's
            // first `wcache.buffer_async` call against a freshly-
            // destroyed (MeBP on) blk.N weight, triggering the alloc
            // churn we saw before.
            if let Some(cb) = progress_cb {
                let logical = (n_layers as u32) - i;
                cb("bwd.post_yield", logical, n_layers as u32);
            }
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
                // Diagnostic: recompute (gradient-checkpointing replay)
                // is the last work between layers. If
                // `bwd.layer.recompute` fires but the next per-layer
                // beacon (bwd.layer.entry) doesn't, the recompute submit
                // killed the tab — i.e. encode_layer's forward dispatches
                // tripped jetsam, NOT the backward kernels.
                if let Some(cb) = progress_cb {
                    let logical = (n_layers as u32) - i;
                    cb("bwd.layer.recompute", logical, n_layers as u32);
                }
            }

            // **wasm32: 0 ms yield between recompute submit and
            // backward_layer encoding.** The recompute submit just
            // landed on Metal's command queue — its 10-12 forward
            // dispatches are processing asynchronously. backward_layer
            // is about to encode 16 copy_buffer_to_buffer ops + ~7
            // phases of dispatches on top. Without a yield iOS
            // Safari's GPUProcess is hit with the second-burst
            // immediately, and the cumulative pressure tripped jetsam
            // (last seen beacon was bwd.layer.recompute 1/35, never
            // bwd.layer.entry). A setTimeout(0) releases the event
            // loop for one tick — Metal absorbs the recompute submit
            // before backward_layer floods it.
            //
            // 0 ms (not 500): the 500 ms variant at the head→backward
            // boundary killed the Worker by giving iOS jetsam too
            // much time to reclaim a "suspended" process. The 0 ms
            // tick gives the GPUProcess message-pipe one drain pass
            // without exposing us to suspended-process reaper.
            // Mobile-mode-gated: see Forward::wasm_yield_zero for rationale.
            #[cfg(target_arch = "wasm32")]
            if self.mobile_mode {
                use wasm_bindgen::JsCast;
                let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global()
                    .dyn_into()
                    .expect("training session runs inside a DedicatedWorkerGlobalScope");
                let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                    let resolve_fn: js_sys::Function = resolve.into();
                    let _ =
                        scope.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve_fn, 0);
                });
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            }

            let mut lenc =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bwd.layer"),
                    });
            self.backward_layer(
                &mut lenc,
                i,
                history_len,
                pos,
                cap,
                lora,
                grad,
                scratch,
                progress_cb,
            )
            .await?;
            // Caller's submit handles the FINAL phase (attn_norm rmsnorm +
            // residual_add) — backward_layer flushed phases 1..=4
            // internally. Phase 5's last beacon is the outer
            // "backward N/35" fired below; we don't add a "bwd.attn.merge"
            // beacon to keep the trail compact.
            self.ctx.queue.submit(Some(lenc.finish()));
            // Per-layer cancel check — same boundary the forward loop
            // uses. Cancellation latency is bounded by one
            // `backward_layer` (~300 ms - 1 s on browser).
            self.check_cancelled()?;
            // Per-layer progress beacon. Convert reverse-walk index
            // `li` (counting top-down from n_layers-1) into the
            // logical 1-based position so UI shows "backward 1/35,
            // 2/35, …" mirroring the forward order — easier to read
            // than the underlying reverse walk.
            if let Some(cb) = progress_cb {
                let logical = (n_layers as u32) - i;
                cb("backward", logical, n_layers as u32);
            }

            // **0 ms yield between backward_layer submit and the rest
            // of the per-layer epilogue.** Real-device data: with the
            // floor+5 patch we got bwd.ple/ffn.down/ffn.gateup/
            // attn.proj/attn.qkv all fired (an ENTIRE backward layer
            // ran), then died right after `backward 1/35` cb — before
            // `bwd.layer.end`. The just-submitted backward_layer
            // command buffer was still in Metal's pipeline; the
            // clip's read_buf_stats drain + drop_prefix_destroy
            // racing against it tripped jetsam.
            //
            // Same fix as the head→recompute and recompute→
            // backward_layer transitions: a 1-tick event-loop release
            // lets Metal absorb the backward_layer submit before the
            // epilogue's drain + destroy IPCs land.
            #[cfg(target_arch = "wasm32")]
            self.wasm_yield_zero().await;

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

            // **Per-layer weight reclaim at the inter-layer GPU-idle
            // point.** Without this, every backward_layer re-fetches its
            // weight tiles but the prior layer's stay resident — by the
            // end of a 10-layer walk we have ~10 layers × ~40 MiB
            // stacked on top of the un-reclaimed forward heap, and the
            // real-device beacon trail shows the second backward layer
            // never completing. Per Apple's Metal-heap doc, destroy
            // makes memory aliasable (good) but stacked-up allocations
            // stress the heap geometry; bounding the backward weight
            // set at "one layer at a time" eliminates that stress.
            //
            // Safe by construction:
            //  • `read_buf_stats` above synced the GPU so all of
            //    backward_layer's submits have drained; nothing in-flight
            //    references `blk.{i}.*`.
            //  • The next iteration's `encode_layer(i-1)` recompute
            //    fetches its OWN prefix (`blk.{i-1}.*`), so destroying
            //    `blk.{i}.*` doesn't pull from the next layer's setup.
            //  • Any in-flight renorm-scale submit reads only
            //    `scratch.d_hidden` (not blk weights).
            //
            // **Mac fast path skips this.** On `mobile_mode = false`
            // GPU heap pressure isn't the bottleneck — keeping the
            // 35 backward-layer weight sets resident across the walk
            // saves ~35 re-fetches per step and ~1 GiB of churn.
            if self.mobile_mode {
                let _destroyed_layer = self.wcache.drop_prefix_destroy(&format!("blk.{i}."));
            }
            // Diagnostic: marks per-layer completion. If this fires for
            // layer N but no `bwd.layer.recompute` for layer N-1 does,
            // the death is in the inter-layer transition AFTER destroy
            // (unlikely — destroy is sync, doesn't dispatch).
            if let Some(cb) = progress_cb {
                let logical = (n_layers as u32) - i;
                cb("bwd.layer.end", logical, n_layers as u32);
            }
        }

        // ---- embed_tokens LoRA backward ----
        // After the layer walk, `scratch.d_hidden` holds the gradient at
        // the start of the residual stream (= gradient feeding the input
        // embedding lookup). The embed_tokens LoRA's forward inject was:
        //   hidden += scale · B_emb · A_emb[:, token_id]
        // So:
        //   u = Bᵀ · d_hidden            (rank floats; overwrites slot.z)
        //   d_B += s · d_hidden ⊗ z      (where z was A_emb[:, token_id])
        //   d_A[:, token_id] += s · u    (single-column scatter)
        // The frozen embedding weight has no gradient (one-hot input).
        if let (Some(g), Some(gg), Some(tok)) = (globals, global_grads, embed_token_id)
            && let (Some(slot), Some(d_pair)) = (g.embed_tokens.as_ref(), gg.embed_tokens.as_ref())
        {
            let vocab_u32 = self.cfg.vocab_size;
            let r = slot.rank as usize;
            let s = slot.scale;
            let mut eenc =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bwd.embed_tokens_lora"),
                    });
            // dB += s · d_hidden ⊗ z  (z still holds A_emb[:, token_id] from fwd)
            lora_outer_add_chained(
                &self.ctx,
                &self.pipes,
                &mut eenc,
                scratch.d_hidden,
                slot.z,
                d_pair.d_b,
                d_model,
                r,
                s,
                true,
            );
            // u = Bᵀ · d_hidden  (overwrites slot.z with u, freeing z's prior role)
            lora_matmul_col_chained(
                &self.ctx,
                &self.pipes,
                &mut eenc,
                slot.b,
                scratch.d_hidden,
                slot.z,
                d_model,
                r,
                1.0,
                false,
            );
            // dA[:, tok] += s · u  (single-column scatter)
            crate::backend::dispatch::lora_embed_col_scatter_add_chained(
                &self.ctx,
                &self.pipes,
                &mut eenc,
                slot.z,
                d_pair.d_a,
                slot.rank,
                vocab_u32,
                tok,
                s,
            );
            self.ctx.queue.submit(Some(eenc.finish()));
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

        // **iOS multi-step stability — destroy layer weights at GPU-idle.**
        // The `read_back_f32` above maps the loss buffer, which forces the
        // GPU to drain ALL prior commands (the entire backward sweep is
        // complete and no command references the layer weights anymore).
        // This is the one safe point to call destroy(): doing it earlier
        // (at backward start) is a use-after-destroy because the forward's
        // commands are still in flight, and crashed the tab at the
        // head→backward transition. Here, with the GPU idle, we force
        // prompt VRAM reclaim of every `blk.*` tile so the NEXT step's
        // forward re-cache starts from genuinely freed memory instead of
        // stacking on the previous step's not-yet-GC'd buffers and
        // crossing the iOS WebContent ceiling. The backward-start
        // `drop_prefix` (no destroy) already unreferenced them for in-step
        // reuse; this is the cross-step reclaim. Next step's `buffer_async`
        // re-fetches fresh. Native frees immediately either way.
        // Empty prefix = destroy the ENTIRE weight cache, not just
        // `blk.*`. The on-device trajectory showed step 2 starting at
        // weightCacheMB=637 — that residue is `token_embd` (~637 MiB
        // vocab embed/output weight used by the backward head), not a
        // `blk.` tensor, so the old `blk.`-only eviction left it resident
        // across steps; the next forward stacked on top and tipped iOS
        // over. Destroying everything makes step N start from the same
        // empty cache step 1 had; the next forward re-fetches what it
        // needs. GPU is idle (post loss-readback) so destroy() is safe.
        // Mac fast path: skip — keep the full cache resident across
        // steps. Mac has the RAM and this saves ~1 GiB of re-fetch.
        let dropped_end = if self.mobile_mode {
            self.wcache.drop_prefix_destroy("")
        } else {
            0
        };
        #[cfg(target_arch = "wasm32")]
        if dropped_end > 0 {
            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                "[bwd] destroyed {dropped_end} layer weight tiles at GPU-idle (cross-step reclaim)"
            )));
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = dropped_end;

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
    /// **iOS Metal lazy-compile mitigation.** End the current backward
    /// phase's encoder, submit it, fire a per-phase beacon, open a
    /// fresh encoder for the next phase. Mirrors the per-group submit
    /// pattern in `backward_step_inner`'s head section (see the
    /// "iOS-tight invariant" comment near the head). Background: iOS
    /// Safari WebGPU lazily compiles each kernel's Metal binary at
    /// first dispatch; packing all ~13 distinct backward kernels per
    /// layer into one submit jetsam'd the WebContent process at the
    /// head→backward transition (real-device beacon trail:
    /// `head_rmsnorm 4/4` fires, `backward 1/35` never does, tab dies).
    /// Splitting into per-phase submits gives Metal a beat to compile +
    /// reclaim one phase at a time; per-phase beacons localize any
    /// future regression to the exact phase.
    /// **Beacon-only since Patch 4** — the per-phase `queue.submit` +
    /// `CommandEncoder::finish` pair was REMOVED.
    ///
    /// History: this helper originally split `backward_layer` into 5
    /// per-phase submits under the theory that iOS Metal lazy-compiles
    /// kernels per submit and batched compiles spike GPUProcess RSS.
    /// Verification at `Pipelines::new()` (line ~240) showed that ALL
    /// pipelines are eagerly built at session start, so per-phase
    /// submits cost ~5 × `queue.submit` + 5 × `CommandEncoder::finish`
    /// IPC round-trips per `backward_layer` call (×10 backward layers
    /// = 50 extra IPCs/step) for no measurable RSS benefit. The
    /// WebGPU Dispatch Overhead paper (arxiv 2604.02344) also shows
    /// Safari Metal per-dispatch cost is FAST (31.7 μs); IPC volume
    /// is the GPUProcess bottleneck.
    ///
    /// `enc` and the `new_enc` swap are gone — all phase work
    /// accumulates into the caller's single encoder, submitted once
    /// at end of `backward_layer`. The per-phase `cb(label, ...)`
    /// beacon is preserved so the post-crash log still localizes any
    /// future regression.
    fn flush_backward_phase(
        &self,
        _enc: &mut wgpu::CommandEncoder,
        progress_cb: Option<&LayerProgressCb<'_>>,
        label: &'static str,
        phase_idx: u32,
        total_phases: u32,
    ) {
        if let Some(cb) = progress_cb {
            cb(label, phase_idx, total_phases);
        }
    }

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
        progress_cb: Option<&LayerProgressCb<'_>>,
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

        // Diagnostic: pinpoint inter-layer death. If `bwd.layer.entry`
        // fires for layer N but no later beacon (bwd.ffn.down /
        // bwd.ffn.gateup / bwd.attn.proj / bwd.attn.qkv / backward N/35)
        // does, death is between buffer_async fetches and phase 1's
        // submit — i.e. capture pre-copies or the PLE backward block.
        if let Some(cb) = progress_cb {
            let logical = self.cfg.n_layers - i;
            cb("bwd.layer.entry", logical, self.cfg.n_layers);
        }

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
            // ── flush phase 0/6: PLE backward (Gemma 4 PLE injection
            // gradient). Adds rmsnorm_backward (NEW), matmul_q4_k_backward_input
            // (NEW), geglu_backward (NEW) to Metal's compile queue —
            // 3 new kernels in their own submit so they don't stack with
            // ffn_down's matmul_q6_k_backward_input compile in phase 1.
            // Only fires when the model has a PLE block (Gemma 4 family).
            self.flush_backward_phase(enc, progress_cb, "bwd.ple", 0, 6);
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

        // ── flush phase 1/5: PLE + post_ffw rmsnorm + ffn_down (matmul + LoRA).
        // First batch of NEW-to-Metal kernels (rmsnorm_backward,
        // matmul_q*_backward_input, plus LoRA kernels iff any LoRA targets
        // include ffn_down). Submit before geglu_backward triggers another
        // Metal compile so they don't bundle and spike RSS at once.
        self.flush_backward_phase(enc, progress_cb, "bwd.ffn.down", 1, 6);

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

        // ── flush phase 2/5: FFN gate/up matmul + LoRA + ffn_norm merge.
        // geglu_backward (NEW) + matmul + LoRA + rmsnorm_backward (already
        // compiled in phase 1). Submit before the attention block triggers
        // attention_probs / attention_backward_* compiles.
        self.flush_backward_phase(enc, progress_cb, "bwd.ffn.gateup", 2, 6);

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

        // ── flush phase 3/5: post_attn rmsnorm + o matmul + o LoRA + attention_probs.
        // attention_probs (NEW) just compiled. Submit before the heavy
        // attention-backward-dq/dkv + rope_neox_backward + rmsnorm_per_row_backward
        // wave triggers 3+ more NEW Metal compiles.
        self.flush_backward_phase(enc, progress_cb, "bwd.attn.proj", 3, 6);

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

        // ── flush phase 4/5: attn dq/dkv/rope/q_norm + q/k/v matmul + LoRAs + K/V branches.
        // Final block of NEW-to-Metal compiles for backward
        // (attention_backward_dq, attention_backward_dkv,
        // rope_neox_backward, rmsnorm_per_row_backward). Phase 5
        // (attn_norm rmsnorm + residual_add) is only already-compiled
        // kernels and rides in the caller's per-layer submit.
        self.flush_backward_phase(enc, progress_cb, "bwd.attn.qkv", 4, 6);

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
