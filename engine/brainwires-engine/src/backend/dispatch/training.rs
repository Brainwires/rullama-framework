//! Training / backward-pass dispatchers: Adam, LoRA (row/col/outer/embed/fused),
//! and the backward kernels (matmul-input, rmsnorm, geglu, rope, attention,
//! cross-entropy). Split out of the monolithic dispatch module — same kernels,
//! same `&Pipelines` calling convention.

use bytemuck::{Pod, Zeroable};

use super::{
    GegluParams, MatmulBackInputParams, RmsParams, RopeParams, XEntParams, cached_dispatch,
};
use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;
use crate::error::{Result, RullamaError};
use crate::gguf::GgmlDtype;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AdamParams {
    n: u32,
    step: u32,
    offset: u32,
    _pad1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

/// Configuration for one `adam_step` dispatch — bias correction is keyed
/// off `step`, which is 1-based to match PyTorch convention.
#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 1,
        }
    }
}

/// AdamW step over a single parameter buffer. Updates `param`, `m`, `v`
/// in-place. `grad` is read-only.
#[allow(clippy::too_many_arguments)]
pub fn adam_step_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    grad: &wgpu::Buffer,
    param: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
    n: usize,
    cfg: AdamConfig,
) {
    let total_groups = (n as u32).div_ceil(64);
    // Chunk dispatches to stay under wgpu's 65_535 workgroups-per-dim
    // cap. lm_head + embed_tokens LoRA B buffers (vocab × rank ≈ 4.2M
    // f32s) need exactly 65_536 workgroups, JUST over the limit. The
    // bind-group cache key is identical across iterations — chunks
    // after the first are cache hits, only the uniform write differs.
    const MAX_GROUPS_PER_DISPATCH: u32 = 65535;
    let mut groups_done: u32 = 0;
    while groups_done < total_groups {
        let groups_this = (total_groups - groups_done).min(MAX_GROUPS_PER_DISPATCH);
        let params = AdamParams {
            n: n as u32,
            step: cfg.step,
            offset: groups_done * 64,
            _pad1: 0,
            lr: cfg.lr,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            weight_decay: cfg.weight_decay,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        cached_dispatch(
            ctx,
            enc,
            &p.adam_step,
            "adam",
            &[grad, param, m, v],
            &params,
            (groups_this, 1, 1),
        );
        groups_done += groups_this;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SumOfSquaresParams {
    n: u32,
    scale_in: f32,
    _p0: u32,
    _p1: u32,
}

/// Sum-of-squares reduction: `output[0] = Σ (input[i] · scale_in)²`
/// for `i in [0, n)`. Output buffer must hold at least 4 bytes.
/// Single workgroup; no caller-side launch math.
pub fn sum_of_squares_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    n: usize,
    scale_in: f32,
) {
    let params = SumOfSquaresParams {
        n: n as u32,
        scale_in,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.sum_of_squares,
        "sos",
        &[input, output],
        &params,
        (1, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LoraMatmulParams {
    k: u32,
    n: u32,
    accumulate: u32,
    _pad: u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LoraOuterParams {
    outer_a: u32,
    outer_b: u32,
    accumulate: u32,
    _pad: u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

/// Tiny f32 row-major matmul: `y = scale · W @ x` (or `y += scale · W @ x`
/// when `accumulate` is true). `W` shape `[n, k]` row-major.
#[allow(clippy::too_many_arguments)]
pub fn lora_matmul_row_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    scale: f32,
    accumulate: bool,
) {
    let params = LoraMatmulParams {
        k: k as u32,
        n: n as u32,
        accumulate: accumulate as u32,
        _pad: 0,
        scale,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
    };
    let key = crate::backend::CacheKey::three(&p.lora_matmul_row, w, x, y);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_mm_row.params"),
            size: std::mem::size_of::<LoraMatmulParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_mm_row.bg"),
            layout: &p.lora_matmul_row.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
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
                    resource: y.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_mm_row.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_matmul_row);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// Tiny f32 transposed matmul: `y = scale · Wᵀ @ x` (or `+=`).
///
/// `W` is the same physical `[outer, inner]` row-major layout as
/// `lora_matmul_row`; the kernel iterates by column to compute the
/// transposed product. `outer` = the summed-over dimension (rows of W),
/// `inner` = the output length (cols of W).
#[allow(clippy::too_many_arguments)]
pub fn lora_matmul_col_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    inner: usize,
    scale: f32,
    accumulate: bool,
) {
    let params = LoraMatmulParams {
        k: outer as u32,
        n: inner as u32,
        accumulate: accumulate as u32,
        _pad: 0,
        scale,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
    };
    let key = crate::backend::CacheKey::three(&p.lora_matmul_col, w, x, y);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_mm_col.params"),
            size: std::mem::size_of::<LoraMatmulParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_mm_col.bg"),
            layout: &p.lora_matmul_col.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
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
                    resource: y.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_mm_col.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_matmul_col);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups((inner as u32).div_ceil(64), 1, 1);
}

/// Rank-1 outer-product accumulator: `out[i, j] += scale · a[i] · b[j]`
/// (or `=` when `accumulate` is false). `out` is `[outer_a, outer_b]`.
#[allow(clippy::too_many_arguments)]
pub fn lora_outer_add_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    out: &wgpu::Buffer,
    outer_a: usize,
    outer_b: usize,
    scale: f32,
    accumulate: bool,
) {
    let params = LoraOuterParams {
        outer_a: outer_a as u32,
        outer_b: outer_b as u32,
        accumulate: accumulate as u32,
        _pad: 0,
        scale,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
    };
    let key = crate::backend::CacheKey::three(&p.lora_outer_add, a, b, out);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_outer.params"),
            size: std::mem::size_of::<LoraOuterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_outer.bg"),
            layout: &p.lora_outer_add.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_outer.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_outer_add);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups(
        (outer_a as u32).div_ceil(8),
        (outer_b as u32).div_ceil(8),
        1,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LoraEmbedColParams {
    rank: u32,
    vocab: u32,
    col: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LoraEmbedColScatterParams {
    rank: u32,
    vocab: u32,
    col: u32,
    _pad: u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

/// Column extract for embed_tokens LoRA forward:
/// `z[r] = A[r, col]` for r in 0..rank, where A has shape `[rank, vocab]`
/// row-major. Used as the LoRA-side replacement for `A @ one_hot(token_id)`.
pub fn lora_embed_col_read_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    a: &wgpu::Buffer,
    z: &wgpu::Buffer,
    rank: u32,
    vocab: u32,
    col: u32,
) {
    let params = LoraEmbedColParams {
        rank,
        vocab,
        col,
        _pad: 0,
    };
    let key = crate::backend::CacheKey::two(&p.lora_embed_col_read, a, z);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_embed_col_read.params"),
            size: std::mem::size_of::<LoraEmbedColParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_embed_col_read.bg"),
            layout: &p.lora_embed_col_read.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_embed_col_read.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_embed_col_read);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups(rank.div_ceil(64), 1, 1);
}

/// Column scatter-add for embed_tokens LoRA backward:
/// `d_A[r, col] += scale · u[r]` for r in 0..rank. d_A has shape
/// `[rank, vocab]` row-major. Used as the LoRA-side replacement for
/// `d_A += scale · u ⊗ one_hot(token_id)`.
#[allow(clippy::too_many_arguments)]
pub fn lora_embed_col_scatter_add_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    u: &wgpu::Buffer,
    da: &wgpu::Buffer,
    rank: u32,
    vocab: u32,
    col: u32,
    scale: f32,
) {
    let params = LoraEmbedColScatterParams {
        rank,
        vocab,
        col,
        _pad: 0,
        scale,
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
    };
    let key = crate::backend::CacheKey::two(&p.lora_embed_col_scatter_add, u, da);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_embed_col_scatter.params"),
            size: std::mem::size_of::<LoraEmbedColScatterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_embed_col_scatter.bg"),
            layout: &p.lora_embed_col_scatter_add.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: da.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_embed_col_scatter.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_embed_col_scatter_add);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups(rank.div_ceil(64), 1, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LoraFusedParams {
    k: u32,
    n: u32,
    rank: u32,
    accumulate: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Fused LoRA forward correction in a single dispatch:
///   `z = A·x; y = (accumulate ? y : 0) + scale · B·z`
///
/// Replaces the two-dispatch pattern (`lora_matmul_row` twice) used by
/// the inference forward path. Same numerical contract; half the
/// dispatch count. Backward path still uses the un-fused matmul
/// primitives — only the forward inject benefits.
///
/// Shapes:
/// - `a`: `[rank, k]` row-major
/// - `b`: `[n, rank]` row-major
/// - `x`: `[k]`
/// - `y`: `[n]` (in/out; in is read only if `accumulate=true`)
/// - `z_out`: `[rank]` capture buffer for the training backward path
#[allow(clippy::too_many_arguments)]
pub fn lora_matmul_fused_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z_out: &wgpu::Buffer,
    k: usize,
    n: usize,
    rank: usize,
    scale: f32,
    accumulate: bool,
) {
    let params = LoraFusedParams {
        k: k as u32,
        n: n as u32,
        rank: rank as u32,
        accumulate: accumulate as u32,
        scale,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    // Cache key uses (pipeline, A, B, x, y). z_out is deterministically
    // derived from the same LoRA target so it doesn't need to be in
    // the key — every call with the same A/B/x/y also uses the same
    // z_out.
    let key = crate::backend::CacheKey::four(&p.lora_matmul_fused, a, b, x, y);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_fused.params"),
            size: std::mem::size_of::<LoraFusedParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_fused.bg"),
            layout: &p.lora_matmul_fused.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: z_out.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_fused.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_matmul_fused);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// `lora_matmul_fused_chained` variant where `b` is packed f16 in `u32`
/// pairs. Same shapes and semantics as the f32 path — only the on-GPU
/// element type of `b` differs. Used by the lm_head LoRA inject when
/// the slot reports `b_is_f16 = true`.
#[allow(clippy::too_many_arguments)]
pub fn lora_matmul_fused_f16b_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    z_out: &wgpu::Buffer,
    k: usize,
    n: usize,
    rank: usize,
    scale: f32,
    accumulate: bool,
) {
    let params = LoraFusedParams {
        k: k as u32,
        n: n as u32,
        rank: rank as u32,
        accumulate: accumulate as u32,
        scale,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let key = crate::backend::CacheKey::four(&p.lora_matmul_fused_f16b, a, b, x, y);
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_fused_f16b.params"),
            size: std::mem::size_of::<LoraFusedParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lora_fused_f16b.bg"),
            layout: &p.lora_matmul_fused_f16b.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: z_out.as_entire_binding(),
                },
            ],
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(&params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("lora_fused_f16b.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.lora_matmul_fused_f16b);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AttnBackParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    heads_per_kv: u32,
    history_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Attention backward, pass 1 of 2 — writes `d_q` and a staged
/// `d_scores` buffer (size `[n_heads, history_len]`) that pass 2 reads.
/// One workgroup per query head. `q` is intentionally *not* a binding
/// here — pass 1's outputs don't depend on it; it shows up in pass 2.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_dq_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    k_hist: &wgpu::Buffer,
    v_hist: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    d_out: &wgpu::Buffer,
    d_scores: &wgpu::Buffer,
    d_q: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    history_len: usize,
) {
    let heads_per_kv = n_heads / n_kv_heads;
    let params = AttnBackParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        heads_per_kv: heads_per_kv as u32,
        history_len: history_len as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.attention_backward_dq,
        "attn_bwd_dq",
        &[k_hist, v_hist, probs, d_out, d_scores, d_q],
        &params,
        (n_heads as u32, 1, 1),
    );
}

/// Attention backward, pass 2 of 2 — consumes the staged `d_scores`
/// from pass 1 and writes `d_k_hist` and `d_v_hist`.
/// Workgroups dispatched as `(n_kv_heads, history_len, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_dkv_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    d_out: &wgpu::Buffer,
    d_scores: &wgpu::Buffer,
    d_k_hist: &wgpu::Buffer,
    d_v_hist: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    history_len: usize,
) {
    let heads_per_kv = n_heads / n_kv_heads;
    let params = AttnBackParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        heads_per_kv: heads_per_kv as u32,
        history_len: history_len as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.attention_backward_dkv,
        "attn_bwd_dkv",
        &[q, probs, d_out, d_scores, d_k_hist, d_v_hist],
        &params,
        (n_kv_heads as u32, history_len as u32, 1),
    );
}

/// RMSNorm backward w.r.t. the input. Weight `w` is frozen (LoRA
/// convention) — pass a real `w_buf` with `has_weight = true` or any
/// dummy `wgpu::Buffer` with `has_weight = false`.
pub fn rmsnorm_backward_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    n: usize,
    eps: f32,
    has_weight: bool,
) {
    let params = RmsParams {
        n: n as u32,
        eps,
        has_weight: has_weight as u32,
        _p: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.rmsnorm_backward,
        "rms_bwd",
        &[x, w, dy, dx],
        &params,
        (1, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RmsPerRowBackParams {
    n_rows: u32,
    n: u32,
    eps: f32,
    has_weight: u32,
}

/// Per-row RMSNorm backward — one workgroup per row. Mirrors
/// `rmsnorm_per_row_chained` (forward); used by the training backward
/// pass for q/k/v head normalisations.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_per_row_backward_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    n_rows: usize,
    n: usize,
    eps: f32,
    has_weight: bool,
) {
    let params = RmsPerRowBackParams {
        n_rows: n_rows as u32,
        n: n as u32,
        eps,
        has_weight: has_weight as u32,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.rmsnorm_per_row_backward,
        "rms_pr_bwd",
        &[x, w, dy, dx],
        &params,
        (n_rows as u32, 1, 1),
    );
}

/// GeGLU backward — produces `d_gate` and `d_up` from `dy`, `gate`, `up`.
pub fn geglu_backward_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    d_gate: &wgpu::Buffer,
    d_up: &wgpu::Buffer,
    n: usize,
) {
    let params = GegluParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.geglu_backward,
        "geglu_bwd",
        &[gate, up, dy, d_gate, d_up],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
}

/// NeoX RoPE backward — inverse in-place rotation. Reuses the same
/// `RopeParams` layout and call shape as the forward.
pub fn rope_neox_backward_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    factors: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    pos: usize,
    rope_dims: usize,
    base: f32,
) {
    let params = RopeParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        rope_dims: rope_dims as u32,
        pos: pos as u32,
        base,
        has_factors: factors.is_some() as u32,
        _p0: 0,
        _p1: 0,
    };
    let f_buf = factors.unwrap_or(dummy);
    let total = (n_heads * (rope_dims / 2)) as u32;
    cached_dispatch(
        ctx,
        enc,
        &p.rope_neox_backward,
        "rope_bwd",
        &[x, f_buf],
        &params,
        (total.div_ceil(64), 1, 1),
    );
}

/// Backward of `y = matmul_q4_k(W, x)` w.r.t. the input.
///
/// Computes `dx[i] = Σ_j dy[j] * dequant(W)[j, i]` on the GPU. One
/// workgroup per block-row of `k` (256 contiguous output elements).
/// Each thread within a workgroup handles one `i` and iterates over `n`.
///
/// `k` must be divisible by 256 (Q4_K block elements). W is frozen by
/// LoRA convention — no weight gradient is produced.
pub fn matmul_q4_k_backward_input_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    assert!(
        k.is_multiple_of(256),
        "k must be divisible by 256 for Q4_K backward"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start: 0,
        j_end: n as u32,
        accumulate: 0,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q4_k_backward_input,
        "q4k_bwd",
        &[weight, dy, dx],
        &params,
        ((k / 256) as u32, 1, 1),
    );
}

/// **Single-tile variant of `matmul_q4_k_backward_input_chained`
/// (Patch 6).** Dispatches ONE tile of the sum-axis loop with
/// explicit `j_start..j_end` bounds and explicit `accumulate` flag —
/// caller-driven tiling. Used for the head `outproj` backward against
/// Gemma 4's 262144 vocab: the caller loops N times, each iteration
/// creates its own command encoder + submits, so each tile lands as
/// its own Metal command buffer instead of one giant buffer with the
/// full 400 MB dequant working set.
///
/// Caller must arrange `accumulate=false` on the first tile (write)
/// and `accumulate=true` on subsequent tiles (add). Math is identical
/// to the non-tiled kernel: `Σ_{j=0..n} = Σ_t Σ_{j=t*c..(t+1)*c}`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q4_k_backward_input_tile_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
    j_start: u32,
    j_end: u32,
    accumulate: bool,
) {
    assert!(
        k.is_multiple_of(256),
        "k must be divisible by 256 for Q4_K backward"
    );
    assert!(
        j_start <= j_end && (j_end as usize) <= n,
        "tile out of range"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start,
        j_end,
        accumulate: if accumulate { 1 } else { 0 },
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q4_k_backward_input,
        "q4k_bwd_tile",
        &[weight, dy, dx],
        &params,
        ((k / 256) as u32, 1, 1),
    );
}

/// Backward of `y = matmul_q4_0(W, x)` w.r.t. the input — fine-tuning on a Q4_0
/// (QAT) base. `dx[i] = Σ_j dy[j] * dequant(W)[j, i]`, frozen weight. `k` must be
/// divisible by 32 (Q4_0 block elements); dispatches one workgroup per block-row.
pub fn matmul_q4_0_backward_input_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    assert!(
        k.is_multiple_of(32),
        "k must be divisible by 32 for Q4_0 backward"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start: 0,
        j_end: n as u32,
        accumulate: 0,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q4_0_backward_input,
        "q4_0_bwd",
        &[weight, dy, dx],
        &params,
        ((k / 32) as u32, 1, 1),
    );
}

/// Vocab-axis-tiled Q4_0 input backward (j_start/j_end window + optional
/// accumulate), mirroring `matmul_q4_k_backward_input_tile_chained`.
pub fn matmul_q4_0_backward_input_tile_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
    j_start: u32,
    j_end: u32,
    accumulate: bool,
) {
    assert!(
        k.is_multiple_of(32),
        "k must be divisible by 32 for Q4_0 backward"
    );
    assert!(
        j_start <= j_end && (j_end as usize) <= n,
        "tile out of range"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start,
        j_end,
        accumulate: if accumulate { 1 } else { 0 },
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q4_0_backward_input,
        "q4_0_bwd_tile",
        &[weight, dy, dx],
        &params,
        ((k / 32) as u32, 1, 1),
    );
}

/// Dtype-routed input backward — picks the q4_k / q6_k / q4_0 backward kernel
/// from the weight's actual GGUF quant type. Mirrors `matmul_quant_chained` on
/// the forward side so one backward path serves Q4_K_M and QAT (Q4_0) bases.
#[allow(clippy::too_many_arguments)]
pub fn matmul_quant_backward_input_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
    dtype: GgmlDtype,
) -> Result<()> {
    match dtype {
        GgmlDtype::Q4_K => matmul_q4_k_backward_input_chained(ctx, p, enc, weight, dy, dx, k, n),
        GgmlDtype::Q6_K => matmul_q6_k_backward_input_chained(ctx, p, enc, weight, dy, dx, k, n),
        GgmlDtype::Q4_0 => matmul_q4_0_backward_input_chained(ctx, p, enc, weight, dy, dx, k, n),
        other => {
            return Err(RullamaError::Inference(format!(
                "weight backward: unsupported quant dtype {other:?} (expected Q4_0, Q4_K, or Q6_K)"
            )));
        }
    }
    Ok(())
}

/// Vocab-axis-tiled variant of [`matmul_quant_backward_input_chained`].
#[allow(clippy::too_many_arguments)]
pub fn matmul_quant_backward_input_tile_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
    j_start: u32,
    j_end: u32,
    accumulate: bool,
    dtype: GgmlDtype,
) -> Result<()> {
    match dtype {
        GgmlDtype::Q4_K => matmul_q4_k_backward_input_tile_chained(
            ctx, p, enc, weight, dy, dx, k, n, j_start, j_end, accumulate,
        ),
        GgmlDtype::Q6_K => matmul_q6_k_backward_input_tile_chained(
            ctx, p, enc, weight, dy, dx, k, n, j_start, j_end, accumulate,
        ),
        GgmlDtype::Q4_0 => matmul_q4_0_backward_input_tile_chained(
            ctx, p, enc, weight, dy, dx, k, n, j_start, j_end, accumulate,
        ),
        other => {
            return Err(RullamaError::Inference(format!(
                "weight backward (tiled): unsupported quant dtype {other:?}"
            )));
        }
    }
    Ok(())
}

/// Backward of `y = matmul_q6_k(W, x)` w.r.t. the input. Same convention
/// as the Q4_K variant — `dx[i] = Σ_j dy[j] * dequant(W)[j, i]`. Used by
/// the output-projection backward, since Gemma 4's tied `token_embd` is
/// Q6_K. `k` must be divisible by 256 (Q6_K block elements).
pub fn matmul_q6_k_backward_input_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    assert!(
        k.is_multiple_of(256),
        "k must be divisible by 256 for Q6_K backward"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start: 0,
        j_end: n as u32,
        accumulate: 0,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q6_k_backward_input,
        "q6k_bwd",
        &[weight, dy, dx],
        &params,
        ((k / 256) as u32, 1, 1),
    );
}

/// **Single-tile variant of `matmul_q6_k_backward_input_chained`
/// (Patch 6).** See the matching Q4_K function for the full rationale.
/// This is the primary user — Gemma 4's `token_embd` is Q6_K, so the
/// head outproj backward (vocab=262144) routes through this kernel.
#[allow(clippy::too_many_arguments)]
pub fn matmul_q6_k_backward_input_tile_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    dy: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    k: usize,
    n: usize,
    j_start: u32,
    j_end: u32,
    accumulate: bool,
) {
    assert!(
        k.is_multiple_of(256),
        "k must be divisible by 256 for Q6_K backward"
    );
    assert!(
        j_start <= j_end && (j_end as usize) <= n,
        "tile out of range"
    );
    let params = MatmulBackInputParams {
        k: k as u32,
        n: n as u32,
        j_start,
        j_end,
        accumulate: if accumulate { 1 } else { 0 },
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.matmul_q6_k_backward_input,
        "q6k_bwd_tile",
        &[weight, dy, dx],
        &params,
        ((k / 256) as u32, 1, 1),
    );
}

/// Cross-entropy forward + backward over a single logit vector.
///
/// Writes `d_logits = softmax(logits) - one_hot(target)` and the scalar
/// loss into the caller's buffers. `target = u32::MAX` or any value ≥
/// `vocab_size` masks the position: gradient and loss are both zero.
///
/// One workgroup of 256 threads sweeps the vocab in three passes
/// (max-reduce, sum-exp-reduce, write). Dispatch is `(1, 1, 1)`.
pub fn cross_entropy_backward_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    logits: &wgpu::Buffer,
    d_logits: &wgpu::Buffer,
    loss_out: &wgpu::Buffer,
    vocab_size: usize,
    target: u32,
) {
    let params = XEntParams {
        vocab_size: vocab_size as u32,
        target,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.cross_entropy_backward,
        "xent_bwd",
        &[logits, d_logits, loss_out],
        &params,
        (1, 1, 1),
    );
}
