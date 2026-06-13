//! Gemma 4 sparse-MoE dispatchers (`gemma4:26b-a4b` + DiffusionGemma).
//!
//! The router's expert selection stays GPU-resident (`expert_ids` /
//! `expert_weights` buffers) — the CPU never learns which experts a token
//! picked, preserving the no-readback-per-token rule. Expert matmuls index the
//! stacked 3-D `ffn_*_exps` weight buffer by `ids[slot]` on-GPU
//! (MulmatID-style), one dispatch per selected slot with a distinct per-slot
//! output buffer (distinct buffers ⇒ distinct bind-cache keys ⇒ each dispatch
//! keeps its own uniform — the shared-uniform overwrite hazard in
//! `cached_dispatch` never arises).

use bytemuck::{Pod, Zeroable};

use super::cached_dispatch;
use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;
use crate::error::{Result, RullamaError};
use crate::gguf::GgmlDtype;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiffusionAttnParams {
    n_tokens: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    prompt_len: u32,
    n_swa: u32,
    swa_layer: u32,
    _pad: u32,
}

/// DiffusionGemma region-masked full-sequence attention. Q is
/// `[n_tokens, n_heads, head_dim]`, K/V `[n_tokens, n_kv_heads, head_dim]`,
/// output `[n_tokens, n_heads, head_dim]`. One workgroup per (query, head);
/// the region mask is computed in-kernel (mirrors `mask::allowed`).
#[allow(clippy::too_many_arguments)]
pub fn diffusion_attention_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    o: &wgpu::Buffer,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    prompt_len: usize,
    n_swa: usize,
    swa_layer: bool,
) {
    let params = DiffusionAttnParams {
        n_tokens: n_tokens as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        prompt_len: prompt_len as u32,
        n_swa: n_swa as u32,
        swa_layer: swa_layer as u32,
        _pad: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.diffusion_attention,
        "diffusion_attention",
        &[q, k, v, o],
        &params,
        (n_tokens as u32, n_heads as u32, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeRouterParams {
    d_model: u32,
    n_experts: u32,
    top_k: u32,
    eps: f32,
    has_scale: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Fused router: unweighted RMSNorm → ×1/√d → ×optional per-channel scale →
/// expert scores → softmax → top-k → renorm. One workgroup; writes
/// `expert_ids[k]` (u32) + `expert_weights[k]` (f32) GPU-side.
///
/// `x` is the RAW post-attention hidden state (the router norms internally —
/// NOT the ffn-normed input). `scale` may be a dummy buffer with
/// `has_scale=false`.
#[allow(clippy::too_many_arguments)]
pub fn moe_router_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    scale: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    router_w: &wgpu::Buffer,
    expert_ids: &wgpu::Buffer,
    expert_weights: &wgpu::Buffer,
    d_model: usize,
    n_experts: usize,
    top_k: usize,
    eps: f32,
) {
    let params = MoeRouterParams {
        d_model: d_model as u32,
        n_experts: n_experts as u32,
        top_k: top_k as u32,
        eps,
        has_scale: if scale.is_some() { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_router,
        "moe_router",
        &[
            x,
            scale.unwrap_or(dummy),
            router_w,
            expert_ids,
            expert_weights,
        ],
        &params,
        (1, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeExpertMatmulParams {
    k: u32,
    n: u32,
    slot: u32,
    slice_blocks: u32,
}

/// MulmatID-style expert matmul: `y = W[ids[slot]] · x` against the stacked
/// 3-D expert tensor resident as one buffer. The expert index is read from
/// the GPU-resident `ids` buffer inside the kernel — no CPU readback.
///
/// IMPORTANT: each slot must use its own `y` buffer. `cached_dispatch` keys
/// its uniform on (pipeline, buffers); identical buffer sets across slots in
/// one encoder would all read the LAST slot's params (queue writes land
/// before the encoder runs). Distinct per-slot outputs make distinct keys.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_matmul_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    slot: usize,
    dtype: GgmlDtype,
) -> Result<()> {
    let pipeline = match dtype {
        GgmlDtype::Q4_K => &p.moe_expert_matmul_q4_k,
        GgmlDtype::Q8_0 => &p.moe_expert_matmul_q8_0,
        other => {
            return Err(RullamaError::Inference(format!(
                "moe expert matmul: unsupported quant dtype {other:?} (expected Q4_K or Q8_0)"
            )));
        }
    };
    let blocks_per_row = k / dtype.block_elems();
    let params = MoeExpertMatmulParams {
        k: k as u32,
        n: n as u32,
        slot: slot as u32,
        slice_blocks: (blocks_per_row * n) as u32,
    };
    cached_dispatch(
        ctx,
        enc,
        pipeline,
        "moe_expert_matmul",
        &[w, ids, x, y],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeGegluBatchedParams {
    rows: u32,
    n_ff: u32,
    _p0: u32,
    _p1: u32,
}

/// Batched GeGLU: `act[ps,i] = gelu(gu[ps,i])·gu[ps,i+n_ff]` over all
/// `rows = n_pos*top_k` rows of a fused gate_up buffer.
pub fn moe_geglu_halves_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    gu: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    n_ff: usize,
) {
    let params = MoeGegluBatchedParams {
        rows: rows as u32,
        n_ff: n_ff as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_geglu_halves_batched,
        "moe_geglu_halves_batched",
        &[gu, y],
        &params,
        ((n_ff as u32).div_ceil(64), rows as u32, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeCombineBatchedParams {
    n_pos: u32,
    d_model: u32,
    top_k: u32,
    has_down_scale: u32,
}

/// Batched combine: `y[pos,i] = Σ_s w[pos*top_k+s]·down_scale[ids[..]]·
/// slots[(pos*top_k+s)*d_model+i]` over all positions.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    slots: &wgpu::Buffer,
    expert_ids: &wgpu::Buffer,
    expert_weights: &wgpu::Buffer,
    down_scale: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n_pos: usize,
    d_model: usize,
    top_k: usize,
) {
    let params = MoeCombineBatchedParams {
        n_pos: n_pos as u32,
        d_model: d_model as u32,
        top_k: top_k as u32,
        has_down_scale: if down_scale.is_some() { 1 } else { 0 },
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_combine_batched,
        "moe_combine_batched",
        &[
            slots,
            expert_ids,
            expert_weights,
            down_scale.unwrap_or(dummy),
            y,
        ],
        &params,
        ((d_model as u32).div_ceil(64), n_pos as u32, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeExpertBatchedParams {
    k: u32,
    n: u32,
    top_k: u32,
    slice_blocks: u32,
}

/// Batched MulmatID expert matmul: every (position, slot) applies its own
/// selected expert (`ids[pos*top_k+slot]`) to `x[pos]`, in one dispatch.
/// `x` is `[n_pos, k]`; output `y` is `[n_pos*top_k, n]`. Q4_K (gate_up) or
/// Q8_0 (down) per `dtype`.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_matmul_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n_pos: usize,
    k: usize,
    n: usize,
    top_k: usize,
    dtype: GgmlDtype,
) -> Result<()> {
    let pipeline = match dtype {
        GgmlDtype::Q4_K => &p.moe_expert_matmul_batched_q4_k,
        GgmlDtype::Q8_0 => &p.moe_expert_matmul_batched_q8_0,
        other => {
            return Err(RullamaError::Inference(format!(
                "moe expert matmul (batched): unsupported dtype {other:?} (expected Q4_K or Q8_0)"
            )));
        }
    };
    let blocks_per_row = k / dtype.block_elems();
    let params = MoeExpertBatchedParams {
        k: k as u32,
        n: n as u32,
        top_k: top_k as u32,
        slice_blocks: (blocks_per_row * n) as u32,
    };
    cached_dispatch(
        ctx,
        enc,
        pipeline,
        "moe_expert_matmul_batched",
        &[w, ids, x, y],
        &params,
        ((n as u32).div_ceil(64), (n_pos * top_k) as u32, 1),
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeRouterBatchedParams {
    n_pos: u32,
    d_model: u32,
    n_experts: u32,
    top_k: u32,
    eps: f32,
    has_scale: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Batched router: routes all `n_pos` positions in ONE dispatch (one workgroup
/// per position). `x` is `[n_pos, d_model]`; writes `expert_ids[n_pos, top_k]`
/// (u32) + `expert_weights[n_pos, top_k]` (f32), GPU-resident.
#[allow(clippy::too_many_arguments)]
pub fn moe_router_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    scale: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    router_w: &wgpu::Buffer,
    expert_ids: &wgpu::Buffer,
    expert_weights: &wgpu::Buffer,
    n_pos: usize,
    d_model: usize,
    n_experts: usize,
    top_k: usize,
    eps: f32,
) {
    let params = MoeRouterBatchedParams {
        n_pos: n_pos as u32,
        d_model: d_model as u32,
        n_experts: n_experts as u32,
        top_k: top_k as u32,
        eps,
        has_scale: if scale.is_some() { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_router_batched,
        "moe_router_batched",
        &[
            x,
            scale.unwrap_or(dummy),
            router_w,
            expert_ids,
            expert_weights,
        ],
        &params,
        (n_pos as u32, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeGegluParams {
    n_ff: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GeGLU over the halves of one fused gate_up vector:
/// `y[i] = gelu(gu[i]) * gu[i + n_ff]`.
pub fn moe_geglu_halves_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    gu: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n_ff: usize,
) {
    let params = MoeGegluParams {
        n_ff: n_ff as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_geglu_halves,
        "moe_geglu_halves",
        &[gu, y],
        &params,
        ((n_ff as u32).div_ceil(64), 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeCombineParams {
    d_model: u32,
    top_k: u32,
    has_down_scale: u32,
    _pad0: u32,
}

/// Weighted combine of the k per-slot expert down outputs:
/// `y[i] = Σ_s weights[s] · down_scale[ids[s]] · slots[s*d_model + i]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    slots: &wgpu::Buffer,
    expert_ids: &wgpu::Buffer,
    expert_weights: &wgpu::Buffer,
    down_scale: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    d_model: usize,
    top_k: usize,
) {
    let params = MoeCombineParams {
        d_model: d_model as u32,
        top_k: top_k as u32,
        has_down_scale: if down_scale.is_some() { 1 } else { 0 },
        _pad0: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.moe_combine,
        "moe_combine",
        &[
            slots,
            expert_ids,
            expert_weights,
            down_scale.unwrap_or(dummy),
            y,
        ],
        &params,
        ((d_model as u32).div_ceil(64), 1, 1),
    );
}
