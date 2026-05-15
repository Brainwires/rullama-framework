//! Gemma 4 vision tower: image → soft-token embeddings.
//!
//! Mirrors Ollama's `model/models/gemma4/model_vision.go` op-for-op:
//!   image (f32 [3,H,W], normalized to [-1,1])
//!     → patch_embd Conv2D (k=16, s=16)
//!     → + 2D position embeddings (X + Y table lookup)
//!     → 16× ViT blocks (rmsnorm → ClippableLinear Q/K/V → per-head Q/K norm,
//!       unweighted V norm → 2D RoPE (theta=100) → bidirectional attention →
//!       ClippableLinear O → post_attn_norm → residual; then ln2 →
//!       ClippableLinear gate/up → QuickGELU(gate)*up → ClippableLinear down →
//!       post_ffn_norm → residual; optional out_scale)
//!     → AvgPool2D 3×3
//!     → scale by sqrt(hidden_size)
//!     → optional std_bias / std_scale
//!     → ClippableLinear `mm.input_projection` → unweighted RMSNorm
//!   = soft-token embeddings [n_pooled_patches, d_text=1536].
//!
//! Single `wgpu::CommandEncoder` per image; one final readback.

use std::sync::Arc;

use bytemuck::cast_slice;
use futures_channel::oneshot;

use crate::backend::dispatch::{
    avg_pool2d_chained, clamp_chained, conv2d_chained,
    matmul_f16_batched_chained, pos_embed_add_chained, quick_geglu_chained,
    residual_add_chained, rmsnorm_per_row_chained,
    rope_2d_chained, scale_chained, vision_attention_chained, make_dummy_storage,
};
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::gguf::GgufReader;

/// Vision-tower hyperparameters, parsed from `gemma4.vision.*` GGUF metadata keys.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub n_layers:   u32,
    pub hidden:     u32,
    pub ffn_inter:  u32,
    pub n_heads:    u32,
    pub patch_size: u32,
    pub num_channels: u32,
    pub n_merge:    u32,
    pub eps:        f32,
    /// Output dim of `mm.input_projection` — same as the text model's d_model.
    pub d_text:     u32,
    /// First dim of `v.position_embd.weight` (≈ 10240 for gemma4:e2b).
    pub pos_size:   u32,
}

impl VisionConfig {
    pub fn from_gguf(r: &GgufReader, d_text: u32) -> Result<Self> {
        // Optional vision metadata: if absent, this isn't a multimodal GGUF.
        let n_layers = r.get_opt("gemma4.vision.block_count")
            .and_then(|v| v.as_u32().ok())
            .ok_or_else(|| RullamaError::Inference("gemma4.vision.block_count missing — not a multimodal GGUF?".into()))?;
        let hidden = r.get("gemma4.vision.embedding_length")?.as_u32()?;
        let ffn_inter = r.get("gemma4.vision.feed_forward_length")?.as_u32()?;
        let n_heads = r.get("gemma4.vision.attention.head_count")?.as_u32()?;
        let patch_size = r.get_opt("gemma4.vision.patch_size")
            .and_then(|v| v.as_u32().ok()).unwrap_or(16);
        let num_channels = r.get_opt("gemma4.vision.num_channels")
            .and_then(|v| v.as_u32().ok()).unwrap_or(3);
        let n_merge = r.get_opt("gemma4.vision.projector.scale_factor")
            .and_then(|v| v.as_u32().ok()).unwrap_or(3);
        let eps = r.get_opt("gemma4.vision.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32().ok()).unwrap_or(1e-6);

        // Discover pos_size from the position_embd descriptor. Shape is [hidden, pos_size, 2]
        // with dim[0]=hidden as fastest, so pos_size = dims[1].
        let pos_desc = r.tensor("v.position_embd.weight")?;
        let pos_size = pos_desc.dims.get(1).copied().unwrap_or(0) as u32;

        Ok(Self {
            n_layers, hidden, ffn_inter, n_heads,
            patch_size, num_channels, n_merge,
            eps, d_text, pos_size,
        })
    }

    pub fn head_dim(&self) -> u32 { self.hidden / self.n_heads }
}

/// Per-linear input/output clamping (loaded from `v.clamp_data`).
/// Defaults to "unbounded" — `f32::MIN`/`f32::MAX` — when the tensor is absent or
/// the linear's slot is empty.
#[derive(Debug, Clone, Copy)]
pub struct ClampVal {
    pub in_min: f32, pub in_max: f32,
    pub out_min: f32, pub out_max: f32,
}

impl ClampVal {
    pub fn unbounded() -> Self {
        Self { in_min: f32::MIN, in_max: f32::MAX, out_min: f32::MIN, out_max: f32::MAX }
    }
    pub fn has_in_clamp(&self)  -> bool { self.in_min  > f32::MIN || self.in_max  < f32::MAX }
    pub fn has_out_clamp(&self) -> bool { self.out_min > f32::MIN || self.out_max < f32::MAX }
}

/// Indices into `layer_clamps[i]` matching Ollama's `linears` order in
/// `model_vision.go::InitClamp`: Q, K, V, O, Gate, Up, Down.
const CLAMP_Q:    usize = 0;
const CLAMP_K:    usize = 1;
const CLAMP_V:    usize = 2;
const CLAMP_O:    usize = 3;
const CLAMP_GATE: usize = 4;
const CLAMP_UP:   usize = 5;
const CLAMP_DOWN: usize = 6;
const LINEARS_PER_LAYER: usize = 7;

/// Maximum number of patches we allocate scratch for. 2520 is the upper bound from
/// Ollama's `process_image.go` (max 280 output tokens × pool 3×3). Round up to
/// 2560 so workgroup reductions get nice numbers.
pub const MAX_PATCHES: u32 = 2560;
/// Maximum input image dimension on either axis. Aligned to patch_size×n_merge=48.
///
/// This is a *scratch-buffer* cap, not a model cap. The real ceiling is the
/// total pixel budget (`MAX_PATCHES` and `MAX_POOLED`), which the client-side
/// `smartResize` already enforces by scaling proportionally. 1536 covers every
/// realistic phone / laptop aspect ratio (16:9, 19:9, 21:9) at the pixel
/// budget, since smart-resize keeps total pixels ≤ ~645k. The one-time
/// `pixel_buf` allocation grows to ~27 MB — negligible vs the weight set.
pub const MAX_IMG_DIM: u32 = 1536;
/// Maximum pooled patches (d_text-wide soft tokens) — matches Ollama's max.
pub const MAX_POOLED: u32 = 280;

pub struct VisionForward {
    cfg: VisionConfig,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    wcache: Arc<WeightCache>,

    layer_clamps: Vec<[ClampVal; LINEARS_PER_LAYER]>,
    proj_clamp:   ClampVal,
    layer_scalars: Vec<Option<f32>>,

    // GPU-resident weights / lookups (loaded once at construction).
    pos_embd: wgpu::Buffer,
    std_bias:  Option<wgpu::Buffer>,
    std_scale: Option<wgpu::Buffer>,

    // Per-image scratch.
    pixel_buf:     wgpu::Buffer,   // [3, MAX_IMG_DIM, MAX_IMG_DIM]
    pos_x_buf:     wgpu::Buffer,   // [MAX_PATCHES] u32
    pos_y_buf:     wgpu::Buffer,
    hidden_a:      wgpu::Buffer,   // [MAX_PATCHES, hidden] f32
    hidden_b:      wgpu::Buffer,
    q:             wgpu::Buffer,   // [MAX_PATCHES, hidden]
    k:             wgpu::Buffer,
    v:             wgpu::Buffer,
    q_norm:        wgpu::Buffer,
    k_norm:        wgpu::Buffer,
    v_norm:        wgpu::Buffer,
    /// Head-major staging for HPD attention (only allocated when the device
    /// supports subgroups). Each holds `[n_heads, n_patches, head_dim]` f32.
    q_hpd:         wgpu::Buffer,
    k_hpd:         wgpu::Buffer,
    v_hpd:         wgpu::Buffer,
    /// Head-major output of HPD attention; transposed back into `attn_out_buf`.
    attn_hpd:      wgpu::Buffer,
    attn_out_buf:  wgpu::Buffer,
    attn_proj:     wgpu::Buffer,
    ffn_gate:      wgpu::Buffer,   // [MAX_PATCHES, ffn] f32
    ffn_up:        wgpu::Buffer,
    ffn_act:       wgpu::Buffer,
    ffn_out:       wgpu::Buffer,
    pool_buf:      wgpu::Buffer,   // [MAX_POOLED, hidden]
    soft_tokens:   wgpu::Buffer,   // [MAX_POOLED, d_text]
    /// Out-of-place scratch for the final unweighted RMSNorm — wgpu disallows
    /// binding the same buffer as both read-only and read-write inside a dispatch.
    soft_tmp:      wgpu::Buffer,
    soft_tokens_read: wgpu::Buffer,

    dummy: wgpu::Buffer,
}

impl VisionForward {
    pub async fn new(
        cfg: VisionConfig,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        wcache: Arc<WeightCache>,
    ) -> Result<Self> {
        let device = &ctx.device;

        let alloc = |label: &str, n_f32: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n_f32 * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let hidden = cfg.hidden as usize;
        let ffn_inter = cfg.ffn_inter as usize;
        let d_text = cfg.d_text as usize;
        let max_patches = MAX_PATCHES as usize;
        let max_pooled = MAX_POOLED as usize;
        let max_img = MAX_IMG_DIM as usize;

        let pixel_buf  = alloc("vfwd.pixels",   3 * max_img * max_img);
        let pos_x_buf  = alloc("vfwd.pos_x",    max_patches);
        let pos_y_buf  = alloc("vfwd.pos_y",    max_patches);
        let hidden_a   = alloc("vfwd.hidden_a", max_patches * hidden);
        let hidden_b   = alloc("vfwd.hidden_b", max_patches * hidden);
        let q          = alloc("vfwd.q",        max_patches * hidden);
        let k          = alloc("vfwd.k",        max_patches * hidden);
        let v          = alloc("vfwd.v",        max_patches * hidden);
        let q_norm     = alloc("vfwd.q_norm",   max_patches * hidden);
        let k_norm     = alloc("vfwd.k_norm",   max_patches * hidden);
        let v_norm     = alloc("vfwd.v_norm",   max_patches * hidden);
        let q_hpd      = alloc("vfwd.q_hpd",    max_patches * hidden);
        let k_hpd      = alloc("vfwd.k_hpd",    max_patches * hidden);
        let v_hpd      = alloc("vfwd.v_hpd",    max_patches * hidden);
        let attn_hpd   = alloc("vfwd.attn_hpd", max_patches * hidden);
        let attn_out_buf = alloc("vfwd.attn_out", max_patches * hidden);
        let attn_proj  = alloc("vfwd.attn_proj", max_patches * hidden);
        let ffn_gate   = alloc("vfwd.ffn_gate", max_patches * ffn_inter);
        let ffn_up     = alloc("vfwd.ffn_up",   max_patches * ffn_inter);
        let ffn_act    = alloc("vfwd.ffn_act",  max_patches * ffn_inter);
        let ffn_out    = alloc("vfwd.ffn_out",  max_patches * hidden);
        let pool_buf   = alloc("vfwd.pool",     max_pooled  * hidden);
        let soft_tokens = alloc("vfwd.soft",    max_pooled * d_text);
        let soft_tmp    = alloc("vfwd.soft_tmp", max_pooled * d_text);

        let soft_tokens_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vfwd.soft_read"),
            size: (max_pooled * d_text * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Position embeddings: [hidden, pos_size, 2] → split into X (axis=0) + Y (axis=1).
        // Storage order with dim[0]=hidden fastest is: axis-major, idx-major, hidden-fastest.
        // GgufReader returns the raw bytes; for f32 that's the buffer as-is. Upload once.
        let pos_embd = wcache.buffer_async("v.position_embd.weight").await?;

        let std_bias  = wcache.buffer_opt_async("v.std_bias").await?;
        let std_scale = wcache.buffer_opt_async("v.std_scale").await?;

        // Load clamps (CPU-side) from v.clamp_data — packed F32 tensor, layout
        // (n_layers × 7 linears × 4 floats) + (4 floats projector).
        let mut layer_clamps: Vec<[ClampVal; LINEARS_PER_LAYER]> =
            vec![[ClampVal::unbounded(); LINEARS_PER_LAYER]; cfg.n_layers as usize];
        let mut proj_clamp = ClampVal::unbounded();

        if let Ok(_) = wcache.reader().tensor("v.clamp_data") {
            let data: Vec<f32> = crate::gguf::dequant_tensor_to_f32_async(
                wcache.reader(), "v.clamp_data"
            ).await?;
            for layer in 0..cfg.n_layers as usize {
                for li in 0..LINEARS_PER_LAYER {
                    let idx = (layer * LINEARS_PER_LAYER + li) * 4;
                    if idx + 3 < data.len() {
                        layer_clamps[layer][li] = ClampVal {
                            in_min:  data[idx],
                            in_max:  data[idx + 1],
                            out_min: data[idx + 2],
                            out_max: data[idx + 3],
                        };
                    }
                }
            }
            let proj_idx = cfg.n_layers as usize * LINEARS_PER_LAYER * 4;
            if proj_idx + 3 < data.len() {
                proj_clamp = ClampVal {
                    in_min:  data[proj_idx],
                    in_max:  data[proj_idx + 1],
                    out_min: data[proj_idx + 2],
                    out_max: data[proj_idx + 3],
                };
            }
        }

        // Per-layer scalar (one f32 per layer when present).
        let mut layer_scalars: Vec<Option<f32>> = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            let name = format!("v.blk.{i}.out_scale.weight");
            let s = match wcache.reader().tensor(&name) {
                Ok(_) => crate::gguf::dequant_tensor_to_f32_async(wcache.reader(), &name).await
                    .ok()
                    .and_then(|v| v.first().copied()),
                Err(_) => None,
            };
            layer_scalars.push(s);
        }

        let dummy = make_dummy_storage(device, "vfwd.dummy");

        Ok(Self {
            cfg, ctx, pipes, wcache,
            layer_clamps, proj_clamp, layer_scalars,
            pos_embd, std_bias, std_scale,
            pixel_buf, pos_x_buf, pos_y_buf,
            hidden_a, hidden_b,
            q, k, v, q_norm, k_norm, v_norm,
            q_hpd, k_hpd, v_hpd, attn_hpd,
            attn_out_buf, attn_proj,
            ffn_gate, ffn_up, ffn_act, ffn_out,
            pool_buf, soft_tokens, soft_tmp, soft_tokens_read,
            dummy,
        })
    }

    pub fn cfg(&self) -> &VisionConfig { &self.cfg }

    /// Encode an image into soft-token embeddings.
    ///
    /// `pixels`: `[3 * img_h * img_w]` f32, channel-first `[R..., G..., B...]`,
    /// normalized to `[-1, 1]`.
    /// `img_h` / `img_w` must each be a multiple of `patch_size * n_merge` (48).
    /// Returns `[n_pooled_patches * d_text]` f32 — flatten as the soft-token sequence.
    pub async fn encode(
        &self, pixels: &[f32], img_h: usize, img_w: usize,
        progress: Option<&dyn Fn(u32, u32)>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let ps = cfg.patch_size as usize;
        let nm = cfg.n_merge as usize;
        let align = ps * nm;
        if img_h % align != 0 || img_w % align != 0 {
            return Err(RullamaError::Inference(format!(
                "vision encode: ({img_h}×{img_w}) not aligned to patch×merge={align}"
            )));
        }
        if pixels.len() != cfg.num_channels as usize * img_h * img_w {
            return Err(RullamaError::Inference(format!(
                "vision encode: pixel buffer is {} f32s, expected {}",
                pixels.len(), cfg.num_channels as usize * img_h * img_w
            )));
        }
        if img_h > MAX_IMG_DIM as usize || img_w > MAX_IMG_DIM as usize {
            return Err(RullamaError::Inference(format!(
                "vision encode: image {img_h}×{img_w} exceeds MAX_IMG_DIM={}", MAX_IMG_DIM
            )));
        }

        let patches_y = img_h / ps;
        let patches_x = img_w / ps;
        let n_patches = patches_x * patches_y;
        if n_patches > MAX_PATCHES as usize {
            return Err(RullamaError::Inference(format!(
                "vision encode: {n_patches} patches > MAX_PATCHES={}", MAX_PATCHES
            )));
        }
        let pooled_y = patches_y / nm;
        let pooled_x = patches_x / nm;
        let n_pooled = pooled_x * pooled_y;

        let hidden    = cfg.hidden as usize;
        let ffn_inter = cfg.ffn_inter as usize;
        let n_heads   = cfg.n_heads as usize;
        let head_dim  = cfg.head_dim() as usize;
        let d_text    = cfg.d_text as usize;
        let eps       = cfg.eps;

        // ---- CPU prep: upload pixels + position indices ----
        self.ctx.queue.write_buffer(&self.pixel_buf, 0, cast_slice(pixels));

        let mut pos_x: Vec<u32> = Vec::with_capacity(n_patches);
        let mut pos_y: Vec<u32> = Vec::with_capacity(n_patches);
        for i in 0..n_patches {
            pos_x.push((i % patches_x) as u32);
            pos_y.push((i / patches_x) as u32);
        }
        self.ctx.queue.write_buffer(&self.pos_x_buf, 0, cast_slice(&pos_x));
        self.ctx.queue.write_buffer(&self.pos_y_buf, 0, cast_slice(&pos_y));

        // Prefetch both prologue/epilogue weights from cache before recording —
        // this lets the entire encode (prologue + 16 layers + epilogue) live in
        // a single CommandEncoder with a single terminal queue.submit().
        let patch_w = self.wcache.buffer_async("v.patch_embd.weight").await?;
        let proj_w = self.wcache.buffer_async("mm.input_projection.weight").await?;

        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vfwd.encoder"),
        });

        // ---- 1. Patch embedding via Conv2D (k=ps, s=ps, pad=0) ----
        // Output is in channel-LAST layout [patches_y, patches_x, hidden] (per
        // conv2d.wgsl), which we read as [n_patches, hidden] downstream.
        conv2d_chained(
            &self.ctx, &self.pipes, &mut enc,
            &patch_w, &self.pixel_buf, &self.hidden_a,
            cfg.num_channels as usize, img_h, img_w,
            hidden, patches_y, patches_x,
            ps, ps, ps, ps, 0, 0,
        );

        // ---- 2. Add 2D position embeddings ----
        pos_embed_add_chained(
            &self.ctx, &self.pipes, &mut enc,
            &self.hidden_a, &self.pos_embd, &self.pos_x_buf, &self.pos_y_buf,
            n_patches, hidden, cfg.pos_size as usize,
        );

        // ---- 3. Transformer layers, all chained into the same encoder ----
        // The progress callback fires after each layer is *recorded*, not when
        // GPU work completes — all 16 blocks now flush together at the final
        // submit below. This trades fine-grained GPU progress for ~16× fewer
        // CPU↔GPU sync points.
        for i in 0..cfg.n_layers {
            self.encode_layer(&mut enc, i, n_patches, hidden, ffn_inter, n_heads, head_dim, eps).await?;
            if let Some(cb) = progress { cb(i + 1, cfg.n_layers); }
        }

        // ---- 4. AvgPool2D 3×3 ----
        // hidden_a is [patches_y, patches_x, hidden] (patch-major matches our
        // post-conv layout). Pool with k=stride=n_merge → [pooled_y, pooled_x, hidden].
        avg_pool2d_chained(
            &self.ctx, &self.pipes, &mut enc,
            &self.hidden_a, &self.pool_buf,
            patches_y, patches_x, hidden, nm,
        );

        // ---- 5. Scale by sqrt(hidden) ----
        scale_chained(
            &self.ctx, &self.pipes, &mut enc,
            &self.pool_buf, n_pooled * hidden, (hidden as f32).sqrt(),
        );

        // ---- 6. Optional std_bias subtract + std_scale multiply ----
        // (Implement as a small per-row elementwise: we don't have a sub kernel
        // yet; skip for now if std_bias/std_scale are absent. Most Gemma 4 e2b
        // checkpoints don't ship these — Ollama's path also no-ops when nil.)
        // TODO if a checkpoint does ship them: use a residual_add with negated
        // std_bias broadcast across patches, then a multiply broadcast. Same
        // applies to std_scale. Adding a dedicated batched_bias_scale kernel
        // is cleaner. Defer to a follow-up commit.

        // ---- 7. Projector: clamp(in) → matmul mm.input_projection → clamp(out) ----
        if self.proj_clamp.has_in_clamp() {
            clamp_chained(
                &self.ctx, &self.pipes, &mut enc,
                &self.pool_buf, n_pooled * hidden,
                self.proj_clamp.in_min, self.proj_clamp.in_max,
            );
        }
        matmul_f16_batched_chained(
            &self.ctx, &self.pipes, &mut enc,
            &proj_w, &self.pool_buf, &self.soft_tokens,
            hidden, d_text, n_pooled,
        );
        if self.proj_clamp.has_out_clamp() {
            clamp_chained(
                &self.ctx, &self.pipes, &mut enc,
                &self.soft_tokens, n_pooled * d_text,
                self.proj_clamp.out_min, self.proj_clamp.out_max,
            );
        }

        // ---- 8. Final RMSNorm without weight (out-of-place into soft_tmp) ----
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, &mut enc,
            &self.soft_tokens, None, &self.dummy, &self.soft_tmp,
            n_pooled, d_text, eps,
        );

        // ---- 9. Submit + readback (read from soft_tmp) ----
        let out_bytes = (n_pooled * d_text * 4) as u64;
        enc.copy_buffer_to_buffer(&self.soft_tmp, 0, &self.soft_tokens_read, 0, out_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let result = read_back_f32(&self.ctx.device, &self.soft_tokens_read, out_bytes).await?;
        Ok(result)
    }

    /// Run one ViT block. The 13 per-block weight handles are fetched from
    /// the shared `WeightCache` — first encode pays the upload, subsequent
    /// encodes reuse cached Arc handles. Total resident vision weight after
    /// all 16 blocks have been touched is ~3 GB on gemma4:e2b; release via
    /// `Model::release_vision_weights()` when switching modes on a
    /// memory-constrained device.
    async fn encode_layer(
        &self,
        enc: &mut wgpu::CommandEncoder,
        i: u32,
        n_patches: usize,
        hidden: usize,
        ffn_inter: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let prefix = format!("v.blk.{i}.");
        let clamps = &self.layer_clamps[i as usize];

        let ln1_w  = self.wcache.buffer_async(&format!("{prefix}ln1.weight")).await?;
        let ln2_w  = self.wcache.buffer_async(&format!("{prefix}ln2.weight")).await?;
        let post_attn_w = self.wcache.buffer_async(&format!("{prefix}attn_post_norm.weight")).await?;
        let post_ffn_w  = self.wcache.buffer_async(&format!("{prefix}ffn_post_norm.weight")).await?;
        let q_w = self.wcache.buffer_async(&format!("{prefix}attn_q.weight")).await?;
        let k_w = self.wcache.buffer_async(&format!("{prefix}attn_k.weight")).await?;
        let v_w = self.wcache.buffer_async(&format!("{prefix}attn_v.weight")).await?;
        let o_w = self.wcache.buffer_async(&format!("{prefix}attn_out.weight")).await?;
        let q_norm_w = self.wcache.buffer_async(&format!("{prefix}attn_q_norm.weight")).await?;
        let k_norm_w = self.wcache.buffer_async(&format!("{prefix}attn_k_norm.weight")).await?;
        let gate_w = self.wcache.buffer_async(&format!("{prefix}ffn_gate.weight")).await?;
        let up_w   = self.wcache.buffer_async(&format!("{prefix}ffn_up.weight")).await?;
        let down_w = self.wcache.buffer_async(&format!("{prefix}ffn_down.weight")).await?;

        // ---- Pre-attention norm into hidden_b (residual stays in hidden_a) ----
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &self.hidden_a, Some(&ln1_w), &self.dummy, &self.hidden_b,
            n_patches, hidden, eps,
        );

        // ---- Q/K/V via ClippableLinear (clamp in → matmul → clamp out) ----
        // Q
        if clamps[CLAMP_Q].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.hidden_b,
                n_patches * hidden, clamps[CLAMP_Q].in_min, clamps[CLAMP_Q].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &q_w, &self.hidden_b, &self.q, hidden, hidden, n_patches);
        if clamps[CLAMP_Q].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.q,
                n_patches * hidden, clamps[CLAMP_Q].out_min, clamps[CLAMP_Q].out_max);
        }
        // K
        if clamps[CLAMP_K].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.hidden_b,
                n_patches * hidden, clamps[CLAMP_K].in_min, clamps[CLAMP_K].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &k_w, &self.hidden_b, &self.k, hidden, hidden, n_patches);
        if clamps[CLAMP_K].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.k,
                n_patches * hidden, clamps[CLAMP_K].out_min, clamps[CLAMP_K].out_max);
        }
        // V
        if clamps[CLAMP_V].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.hidden_b,
                n_patches * hidden, clamps[CLAMP_V].in_min, clamps[CLAMP_V].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &v_w, &self.hidden_b, &self.v, hidden, hidden, n_patches);
        if clamps[CLAMP_V].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.v,
                n_patches * hidden, clamps[CLAMP_V].out_min, clamps[CLAMP_V].out_max);
        }

        // ---- Per-(patch, head) Q/K/V norms ----
        // q layout is [n_patches, n_heads, head_dim] flat. Norm rows = n_patches × n_heads.
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.q, Some(&q_norm_w), &self.dummy, &self.q_norm,
            n_patches * n_heads, head_dim, eps);
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.k, Some(&k_norm_w), &self.dummy, &self.k_norm,
            n_patches * n_heads, head_dim, eps);
        // V norm is unweighted.
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.v, None, &self.dummy, &self.v_norm,
            n_patches * n_heads, head_dim, eps);

        // ---- 2D RoPE on Q and K (in-place) ----
        rope_2d_chained(&self.ctx, &self.pipes, enc,
            &self.q_norm, &self.pos_x_buf, &self.pos_y_buf,
            head_dim, n_heads, n_patches, 100.0);
        rope_2d_chained(&self.ctx, &self.pipes, enc,
            &self.k_norm, &self.pos_x_buf, &self.pos_y_buf,
            head_dim, n_heads, n_patches, 100.0);

        // ---- Attention (bidirectional batched) ----
        // When the head-major (HPD) subgroup kernel is available, pre-transpose
        // Q/K/V to [n_heads, n_patches, head_dim] so per-WG K/V tile loads
        // coalesce across the head's contiguous slab. Microbench: ~10% over
        // patch-major even with the 4 wrapping transposes counted in.
        // Prefer the f16-LDS HPD variant when SHADER_F16 is available — same
        // numerics within f16-rounding tolerance, half the workgroup storage.
        // Routing precedence (best → worst):
        //   1. subgroup + f16 LDS (AMD GCN + Qualcomm Adreno with SHADER_F16)
        //   2. subgroup + f32 LDS (subgroup-capable adapters w/o SHADER_F16)
        //   3. **subgroup-free + f16 LDS** (Apple Silicon, NVIDIA, Intel
        //      with SHADER_F16 — the iPhone A18 case)
        //   4. fallthrough → original Q=8 barrier-tree kernel
        let hpd_pipe = self.pipes.vision_attention_flash_sub_hpd_f16.as_ref()
            .or(self.pipes.vision_attention_flash_sub_hpd.as_ref())
            .or(self.pipes.vision_attention_flash_hpd_f16.as_ref());
        if let Some(hpd) = hpd_pipe {
            crate::backend::dispatch::transpose_phd_to_hpd_chained(&self.ctx, &self.pipes, enc,
                &self.q_norm, &self.q_hpd, n_patches, n_heads, head_dim);
            crate::backend::dispatch::transpose_phd_to_hpd_chained(&self.ctx, &self.pipes, enc,
                &self.k_norm, &self.k_hpd, n_patches, n_heads, head_dim);
            crate::backend::dispatch::transpose_phd_to_hpd_chained(&self.ctx, &self.pipes, enc,
                &self.v_norm, &self.v_hpd, n_patches, n_heads, head_dim);
            crate::backend::dispatch::vision_attention_flash_sub_hpd_chained(
                &self.ctx, &self.pipes, hpd, enc,
                &self.q_hpd, &self.k_hpd, &self.v_hpd, &self.attn_hpd,
                head_dim, n_heads, n_patches);
            crate::backend::dispatch::transpose_hpd_to_phd_chained(&self.ctx, &self.pipes, enc,
                &self.attn_hpd, &self.attn_out_buf, n_patches, n_heads, head_dim);
        } else {
            vision_attention_chained(&self.ctx, &self.pipes, enc,
                &self.q_norm, &self.k_norm, &self.v_norm, &self.attn_out_buf,
                head_dim, n_heads, n_patches);
        }

        // ---- Output projection (clamp → matmul → clamp) ----
        if clamps[CLAMP_O].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.attn_out_buf,
                n_patches * hidden, clamps[CLAMP_O].in_min, clamps[CLAMP_O].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &o_w, &self.attn_out_buf, &self.attn_proj, hidden, hidden, n_patches);
        if clamps[CLAMP_O].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.attn_proj,
                n_patches * hidden, clamps[CLAMP_O].out_min, clamps[CLAMP_O].out_max);
        }

        // ---- post_attention_norm + residual ----
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.attn_proj, Some(&post_attn_w), &self.dummy, &self.hidden_b,
            n_patches, hidden, eps);
        residual_add_chained(&self.ctx, &self.pipes, enc,
            &self.hidden_a, &self.hidden_b, n_patches * hidden);

        // ---- ln2 ----
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.hidden_a, Some(&ln2_w), &self.dummy, &self.hidden_b,
            n_patches, hidden, eps);

        // ---- MLP gate (clamp → matmul → clamp) ----
        if clamps[CLAMP_GATE].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.hidden_b,
                n_patches * hidden, clamps[CLAMP_GATE].in_min, clamps[CLAMP_GATE].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &gate_w, &self.hidden_b, &self.ffn_gate, hidden, ffn_inter, n_patches);
        if clamps[CLAMP_GATE].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.ffn_gate,
                n_patches * ffn_inter, clamps[CLAMP_GATE].out_min, clamps[CLAMP_GATE].out_max);
        }
        // Up
        if clamps[CLAMP_UP].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.hidden_b,
                n_patches * hidden, clamps[CLAMP_UP].in_min, clamps[CLAMP_UP].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &up_w, &self.hidden_b, &self.ffn_up, hidden, ffn_inter, n_patches);
        if clamps[CLAMP_UP].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.ffn_up,
                n_patches * ffn_inter, clamps[CLAMP_UP].out_min, clamps[CLAMP_UP].out_max);
        }

        // ---- QuickGELU(gate) * up ----
        quick_geglu_chained(&self.ctx, &self.pipes, enc,
            &self.ffn_gate, &self.ffn_up, &self.ffn_act, n_patches * ffn_inter);

        // ---- Down (clamp → matmul → clamp) ----
        if clamps[CLAMP_DOWN].has_in_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.ffn_act,
                n_patches * ffn_inter, clamps[CLAMP_DOWN].in_min, clamps[CLAMP_DOWN].in_max);
        }
        matmul_f16_batched_chained(&self.ctx, &self.pipes, enc,
            &down_w, &self.ffn_act, &self.ffn_out, ffn_inter, hidden, n_patches);
        if clamps[CLAMP_DOWN].has_out_clamp() {
            clamp_chained(&self.ctx, &self.pipes, enc, &self.ffn_out,
                n_patches * hidden, clamps[CLAMP_DOWN].out_min, clamps[CLAMP_DOWN].out_max);
        }

        // ---- post_ffn_norm + residual ----
        rmsnorm_per_row_chained(&self.ctx, &self.pipes, enc,
            &self.ffn_out, Some(&post_ffn_w), &self.dummy, &self.hidden_b,
            n_patches, hidden, eps);
        residual_add_chained(&self.ctx, &self.pipes, enc,
            &self.hidden_a, &self.hidden_b, n_patches * hidden);

        // ---- per-layer output scalar ----
        if let Some(s) = self.layer_scalars[i as usize] {
            scale_chained(&self.ctx, &self.pipes, enc,
                &self.hidden_a, n_patches * hidden, s);
        }

        Ok(())
    }
}

async fn read_back_f32(device: &wgpu::Device, buf: &wgpu::Buffer, n_bytes: u64) -> Result<Vec<f32>> {
    let slice = buf.slice(0..n_bytes);
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
