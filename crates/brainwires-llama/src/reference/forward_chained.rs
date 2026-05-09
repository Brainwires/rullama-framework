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
    attention_chained, geglu_chained, make_dummy_storage,
    matmul_q4_k_chained, matmul_q6_k_chained,
    residual_add_chained, rmsnorm_chained, rmsnorm_per_row_chained,
    rope_neox_chained, scale_chained, softcap_chained,
};
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

    // Cached scale factor for the final logits softcap dispatch.
    pos: u32,
}

impl Forward {
    pub async fn new(
        cfg: Gemma4Config,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        weights: Weights,
        wcache: Arc<WeightCache>,
    ) -> Result<Self> {
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
                let bytes = (MAX_CONTEXT as usize * n_kv * hd * 4) as u64;
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
            pos: 0,
        })
    }

    pub fn cfg(&self) -> &Gemma4Config { &self.cfg }
    pub fn pos(&self) -> u32 { self.pos }

    pub fn reset(&mut self) {
        self.pos = 0;
        for l in self.kv_lens.iter_mut() { *l = 0; }
    }

    /// Run one forward step. Returns logits over the full vocab (post-softcap).
    pub async fn step(&mut self, token_id: u32) -> Result<Vec<f32>> {
        if (token_id as u64) >= self.cfg.vocab_size as u64 {
            return Err(RullamaError::Inference(format!(
                "token_id {token_id} >= vocab_size {}", self.cfg.vocab_size
            )));
        }
        if self.pos >= MAX_CONTEXT {
            return Err(RullamaError::Inference(format!(
                "context length exceeded MAX_CONTEXT={}", MAX_CONTEXT
            )));
        }
        let d_model = self.cfg.d_model as usize;
        let n_layers = self.cfg.n_layers as usize;
        let ple_dim = self.cfg.ple_dim as usize;
        let eps = self.cfg.rms_norm_eps;
        let pos = self.pos;

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
        // We'll need per-layer weights + KV writes; pre-resolve the weight buffers
        // up front (so encoding doesn't fight the borrow checker by awaiting mid-encode).
        for i in 0..n_layers as u32 {
            self.encode_layer(&mut enc, i, pos).await?;
        }

        // ---- final norm (in-place into hidden via norm_y as scratch) ----
        rmsnorm_chained(&self.ctx, &self.pipes, &mut enc,
            &self.hidden, Some(&final_norm), &self.dummy, &self.norm_x, d_model, eps);

        // ---- output projection (tiled): tile along vocab axis ----
        // Each tile matmul writes its rows into `logits_tile` starting at offset 0
        // (so it always satisfies the storage-binding alignment), then we copy
        // those bytes into `logits` at offset `row_start * 4` (copy_buffer_to_buffer
        // only needs 4-byte alignment).
        const MAX_TILE_BYTES: usize = 80 * 1024 * 1024;
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

    async fn encode_layer(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        i: u32,
        pos: u32,
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

        // Q/K/V projections from norm_x
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &q_w, &self.norm_x, &self.q, d_model, n_heads * head_dim);
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

        if donor.is_none() {
            let kw = k_w.as_ref().unwrap();
            let knw = k_norm_w.as_ref().unwrap();
            let vw = v_w.as_ref().unwrap();
            let vdt = v_w_dtype.unwrap();

            matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                kw, &self.norm_x, &self.k, d_model, n_kv_heads * head_dim);
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

        // attn_proj = matmul(attn_out_buf, attn_output.weight)
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &o_w, &self.attn_out_buf, &self.attn_proj, n_heads * head_dim, d_model);
        // norm_y = rmsnorm(attn_proj, post_attn_norm.weight)
        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.attn_proj, Some(&post_attn_w), &self.dummy,
            &self.norm_y, d_model, eps);
        // hidden += norm_y
        residual_add_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, &self.norm_y, d_model);

        // ===== MLP =====
        rmsnorm_chained(&self.ctx, &self.pipes, enc,
            &self.hidden, Some(&mlp_norm_w), &self.dummy,
            &self.norm_x, d_model, eps);
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &gate_w, &self.norm_x, &self.ffn_gate, d_model, ffn_n);
        matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
            &up_w, &self.norm_x, &self.ffn_up, d_model, ffn_n);
        geglu_chained(&self.ctx, &self.pipes, enc,
            &self.ffn_gate, &self.ffn_up, &self.ffn_act, ffn_n);

        match down_dtype {
            GgmlDtype::Q6_K => matmul_q6_k_chained(&self.ctx, &self.pipes, enc,
                &down_w, &self.ffn_act, &self.ffn_out, ffn_n, d_model),
            GgmlDtype::Q4_K => matmul_q4_k_chained(&self.ctx, &self.pipes, enc,
                &down_w, &self.ffn_act, &self.ffn_out, ffn_n, d_model),
            other => return Err(RullamaError::Inference(format!("ffn_down dtype {other:?} unsupported"))),
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
