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
