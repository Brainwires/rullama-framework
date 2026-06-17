//! Vision-tower attention dispatchers: positional-embedding add, the reference
//! vision attention, every flash-attention variant (q4/q8/q16, subgroup, HPD,
//! f16-LDS) + the PHD↔HPD transposes, and the audio Conformer's block-local
//! attention. These build bind groups directly (write_uniform) rather than via
//! cached_dispatch. Split out of the monolithic dispatch module.

use super::{
    BlockLocalAttnParams, PosEmbedAddParams, TransposeParams, VisionAttnParams, write_uniform,
};
use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;

/// Add 2D position embeddings to per-patch hidden states (vision tower).
/// hidden[p, d] += pos_embd_X[posX[p], d] + pos_embd_Y[posY[p], d]
pub fn pos_embed_add_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    hidden: &wgpu::Buffer,
    pos_embd: &wgpu::Buffer,
    pos_x: &wgpu::Buffer,
    pos_y: &wgpu::Buffer,
    n_patches: usize,
    hidden_size: usize,
    pos_size: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = PosEmbedAddParams {
        n_patches: n_patches as u32,
        hidden_size: hidden_size as u32,
        pos_size: pos_size as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "posembed.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("posembed.bg"),
        layout: &p.pos_embed_add.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hidden.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pos_embd.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pos_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: pos_y.as_entire_binding(),
            },
        ],
    });
    let total = (n_patches * hidden_size) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("posembed.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.pos_embed_add);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Bidirectional batched self-attention for the vision tower. Reuses the same
/// q/k/v/out layout as text attention but skips causal masking and adds a
/// per-batch-query workgroup dimension.
pub fn vision_attention_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    // The flash kernel (workgroup-shared tiled K/V + online softmax) wins by
    // an order of magnitude over the original sequential per-thread inner
    // loop, provided head_dim fits in the shared q/kv layout. Gemma 4 vision
    // uses head_dim=64, which is the maximum the flash kernel supports.
    // Pick the largest multi-query variant the shape will fill. Q=8 wins
    // marginally over Q=4 in the AMD Pro 555 microbench (1.26 vs 1.34 s on
    // the full 2304-patch shape), Q=4 wins handily over Q=1.
    //
    // When the device exposes `Features::SUBGROUP`, prefer the subgroup-collapsed
    // variant which replaces the per-tile barrier-tree reductions with
    // `subgroupMax` / `subgroupAdd` intrinsics. Numerics match Q=8 within
    // f32-reordering tolerance.
    // TILE_T=64 / Q=8 subgroup variant exists in tree but **NOT ROUTED** —
    // its 16 KB kv_tile drops occupancy below TILE_T=32's break-even point on
    // Radeon Pro 555 (1.67s vs 1.12s). Kept as a reference variant; the next
    // reader should expect the same outcome on similar GCN hardware.
    if head_dim <= 64 && n_patches >= 8 {
        if let Some(sub) = p.vision_attention_flash_subgroup.as_ref() {
            return vision_attention_flash_subgroup_chained(
                ctx, p, sub, enc, q, k, v, out, head_dim, n_heads, n_patches,
            );
        }
        vision_attention_flash_q8_chained(ctx, p, enc, q, k, v, out, head_dim, n_heads, n_patches);
        return;
    }
    if head_dim <= 64 && n_patches >= 4 {
        vision_attention_flash_q4_chained(ctx, p, enc, q, k, v, out, head_dim, n_heads, n_patches);
        return;
    }
    if head_dim <= 64 {
        vision_attention_flash_chained(ctx, p, enc, q, k, v, out, head_dim, n_heads, n_patches);
        return;
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattn.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattn.bg"),
        layout: &p.vision_attention.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattn.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
}

/// Flash-attention-style bidirectional self-attention for vision.
/// Tiles K, V in chunks of 32 patches into workgroup-shared memory and runs
/// online softmax. Same I/O as `vision_attention_chained`.
pub fn vision_attention_flash_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnf.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnf.bg"),
        layout: &p.vision_attention_flash.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnf.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention_flash);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
}

/// Multi-query flash vision attention Q=16. Same idea as Q=8 with double
/// queries-per-WG. Workgroup storage right at the 16 KB WebGPU minimum.
pub fn vision_attention_flash_q16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnq16.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnq16.bg"),
        layout: &p.vision_attention_flash_q16.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq16.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention_flash_q16);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(16);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Multi-query flash vision attention: Q=8 queries per workgroup share one
/// K/V load. Same idea as Q=4 but with 2× more queries; cuts launch count
/// and K/V bandwidth another 2× but uses more workgroup-shared memory
/// (~12.5 KB vs 10.5 KB for Q=4) so occupancy may drop on some adapters.
pub fn vision_attention_flash_q8_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnq8.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnq8.bg"),
        layout: &p.vision_attention_flash_q8.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq8.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention_flash_q8);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(8);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Subgroup-collapsed flash vision attention. Replaces the per-tile barrier
/// tree reductions in the Q=8 variant with `subgroupMax` / `subgroupAdd`.
/// Caller passes the resolved pipeline ref (matched on `WgpuCtx::has_subgroups`).
pub fn vision_attention_flash_subgroup_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    sub: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let _ = p; // reserved for future routing decisions on the kernel set
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnSub.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnSub.bg"),
        layout: &sub.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSub.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(sub);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(8);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Transpose [n_patches, n_heads, head_dim] → [n_heads, n_patches, head_dim].
pub fn transpose_phd_to_hpd_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    n_patches: usize,
    n_heads: usize,
    head_dim: usize,
) {
    transpose_chained(
        ctx,
        &p.transpose_phd_to_hpd,
        "tposePHDtoHPD",
        enc,
        src,
        dst,
        n_patches,
        n_heads,
        head_dim,
    );
}

/// Inverse: head-major → patch-major.
pub fn transpose_hpd_to_phd_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    n_patches: usize,
    n_heads: usize,
    head_dim: usize,
) {
    transpose_chained(
        ctx,
        &p.transpose_hpd_to_phd,
        "tposeHPDtoPHD",
        enc,
        src,
        dst,
        n_patches,
        n_heads,
        head_dim,
    );
}

fn transpose_chained(
    ctx: &WgpuCtx,
    pipe: &wgpu::ComputePipeline,
    label: &str,
    enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    n_patches: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = TransposeParams {
        n_patches: n_patches as u32,
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bg")),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&format!("{label}.pass")),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    let total = (n_patches * n_heads * head_dim) as u32;
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// f16-LDS HPD subgroup flash attention. Same I/O as the f32-LDS variant; the
/// only differences are workgroup-storage type (halves LDS) and the inner
/// product runs through `f32(f16 × f16)` to keep the running sum stable.
pub fn vision_attention_flash_sub_hpd_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    vision_attention_flash_sub_hpd_chained(
        ctx, p, pipe, enc, q, k, v, out, head_dim, n_heads, n_patches,
    );
}

/// HPD + f16-LDS attention WITHOUT subgroups. Bind-group + dispatch shape
/// identical to the subgroup variant — they differ only in WGSL.
pub fn vision_attention_flash_hpd_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    vision_attention_flash_sub_hpd_chained(
        ctx, p, pipe, enc, q, k, v, out, head_dim, n_heads, n_patches,
    );
}

/// Q=16 variant of `vision_attention_flash_sub_hpd_f16_chained`. Dispatches
/// half as many WGs (`ceil(n_patches/16)` per head). Uses the same uniform
/// layout so the dispatcher just wraps the same bind-group construction.
pub fn vision_attention_flash_sub_hpd_f16_q16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnSubHPDQ16.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnSubHPDQ16.bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubHPDQ16.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(16);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Head-major (HPD) subgroup flash attention. Caller must pre-transpose Q/K/V
/// to [n_heads, n_patches, head_dim]; output is written in the same layout.
pub fn vision_attention_flash_sub_hpd_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnSubHPD.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnSubHPD.bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubHPD.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(8);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// TILE_T=64, Q=12 subgroup-collapsed flash attention. K/V tile spans 64
/// patches (all 64 lanes do scoring work, no `tid < tile_size` masking) and
/// each WG handles 12 queries. Needs ≥ 22 KB LDS — pipeline built only when
/// the device exposes that.
pub fn vision_attention_flash_sub_t64_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnSubT64.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnSubT64.bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubT64.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(8);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Multi-query flash vision attention: Q=4 queries per workgroup share one
/// K/V load. Cuts K/V global-memory bandwidth and workgroup launch count 4×
/// over `vision_attention_flash_chained`.
pub fn vision_attention_flash_q4_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = VisionAttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "vattnq4.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vattnq4.bg"),
        layout: &p.vision_attention_flash_q4.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq4.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention_flash_q4);
    cp.set_bind_group(0, &bg, &[]);
    // Each workgroup processes Q_PER_WG=4 queries.
    let n_query_groups = (n_patches as u32).div_ceil(4);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Conformer block-local attention (Gemma 4 audio). Mirrors Ollama's
/// `model_audio.go::AudioConformerBlock.forwardAttention`'s inner loop —
/// the in-tree CPU oracle was removed in M16, Ollama is the parity anchor.
///
/// Inputs are already-prepared:
///   * `q_pad`     [padded_len, hidden] — Q projected and per-dim scaled
///   * `k_padded`  [(pad_left + padded_len + pad_right), hidden] — K projected,
///     k-scale applied, zero-padded
///   * `v_padded`  same shape as `k_padded` — V projected, zero-padded
///   * `pos_proj`  [max_span, hidden] — sinusoidal positions through linear_pos
///
/// Output: `attn_out` [padded_len, hidden]. Caller trims to `seq * hidden`.
///
/// The kernel hard-codes `head_dim = 128` (Gemma 4 audio).
pub fn block_local_attention_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q_pad: &wgpu::Buffer,
    k_padded: &wgpu::Buffer,
    v_padded: &wgpu::Buffer,
    pos_proj: &wgpu::Buffer,
    attn_out: &wgpu::Buffer,
    seq: usize,
    padded_len: usize,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    context_size: usize,
    max_span: usize,
    max_past: usize,
    max_future: usize,
    pad_left: usize,
    logit_cap: f32,
) {
    debug_assert_eq!(
        head_dim, 128,
        "block_local_attention.wgsl is hard-coded to head_dim=128"
    );
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BlockLocalAttnParams {
        seq: seq as u32,
        padded_len: padded_len as u32,
        hidden: hidden as u32,
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        chunk_size: chunk_size as u32,
        context_size: context_size as u32,
        max_span: max_span as u32,
        max_past: max_past as u32,
        max_future: max_future as u32,
        pad_left: pad_left as u32,
        logit_cap,
    };
    let p_buf = write_uniform(device, queue, "blattn.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blattn.bg"),
        layout: &p.block_local_attention.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q_pad.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k_padded.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_padded.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: pos_proj.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: attn_out.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("blattn.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.block_local_attention);
    cp.set_bind_group(0, &bg, &[]);
    // One workgroup per (padded query position, head).
    cp.dispatch_workgroups(padded_len as u32, n_heads as u32, 1);
}
