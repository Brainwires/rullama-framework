//! Matmul dispatchers: the basic per-token matvecs (q4_k / q6_k / f16 / bf16)
//! and the batched / tiled variants used by the vision + audio towers. The
//! `use_tiled_batched*` thresholds pick naive-vs-tiled per shape. Shared param
//! structs (MatmulParams, BatchedMatmulParams) and the `cached_dispatch` /
//! `write_uniform` helpers live in the parent module.

use super::{BatchedMatmulParams, MatmulParams, cached_dispatch, write_uniform};
use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;
use crate::error::{Result, RullamaError};
use crate::gguf::GgmlDtype;

fn matmul_chained_inner(
    ctx: &WgpuCtx,
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    let params = MatmulParams {
        k: k as u32,
        n: n as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        pipeline,
        label,
        &[w, x, y],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
}

// Tiled-kernel threshold: empirically the naive matvec (one thread per output)
// beats the tiled kernel on Apple GPUs at every per-layer shape because of L1/L2
// cache + the cost of workgroup barriers. We only switch to tiled when the
// matmul is huge enough that the bandwidth savings clearly dominate — currently
// only the output projection (vocab × d_model) qualifies, and it dispatches
// directly through forward_chained::run_matmul_into_buf.
//
// Keeping the tiled pipelines built so they're available without recompiling
// when we get round to per-shape benchmarking in M8.4 (perf_bench).

pub fn matmul_q4_k_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    // A/B on AMD Pro 555 / Metal, single-token gemma4:e2b "Hi":
    //   non-tiled WG=64  (default): 937 ms/tok
    //   non-tiled WG=256          : 939 ms/tok  (neutral — text is weight-bw bound)
    //   tiled    WG=64            : 996 ms/tok  (-6%)
    //   tiled    WG=64 + f16 LDS  : 975 ms/tok  (-4%)
    // The 3 alternatives are built (when relevant features) but unrouted.
    matmul_chained_inner(ctx, enc, &p.q4_k_matmul, "q4k_chain", w, x, y, k, n);
}

pub fn matmul_q6_k_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.q6_k_matmul, "q6k_chain", w, x, y, k, n);
}

/// Q4_0 weight matmul. Q4_0 is the legacy ggml quant Google ships QAT
/// (quantization-aware-trained) Gemma in — e.g. `gemma4:e2b-it-qat`, whose
/// attn / FFN weights are all Q4_0 (vs the Q4_K mix in the standard Q4_K_M).
pub fn matmul_q4_0_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.q4_0_matmul, "q4_0_chain", w, x, y, k, n);
}

/// Q5_0 weight matmul. Q5_0 is the 5-bit legacy ggml quant DiffusionGemma's
/// Q4_K_M build uses for some down-projections — f16 scale + u32 high bits +
/// 16 nibble bytes per 22-byte block.
pub fn matmul_q5_0_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.q5_0_matmul, "q5_0_chain", w, x, y, k, n);
}

/// Q8_0 weight matmul. Q8_0 is the 8-bit legacy ggml quant the `-it-q8_0`
/// Ollama tags ship — f16 scale + 32 signed int8 per 34-byte block.
pub fn matmul_q8_0_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.q8_0_matmul, "q8_0_chain", w, x, y, k, n);
}

/// Dtype-routed weight matmul: picks the right dequant-matmul pipeline from the
/// weight tensor's actual GGUF quant type. This is what lets one forward path
/// serve both the standard Q4_K_M models (Q4_K / Q6_K weights) and the QAT
/// models (Q4_0 weights) — the per-layer dispatch no longer assumes Q4_K.
pub fn matmul_quant_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    dtype: GgmlDtype,
) -> Result<()> {
    match dtype {
        GgmlDtype::Q4_K => matmul_q4_k_chained(ctx, p, enc, w, x, y, k, n),
        GgmlDtype::Q6_K => matmul_q6_k_chained(ctx, p, enc, w, x, y, k, n),
        GgmlDtype::Q4_0 => matmul_q4_0_chained(ctx, p, enc, w, x, y, k, n),
        GgmlDtype::Q5_0 => matmul_q5_0_chained(ctx, p, enc, w, x, y, k, n),
        GgmlDtype::Q8_0 => matmul_q8_0_chained(ctx, p, enc, w, x, y, k, n),
        // QAT models leave a stray weight (e.g. a projection) in F16.
        GgmlDtype::F16 => matmul_f16_chained(ctx, p, enc, w, x, y, k, n),
        other => {
            return Err(RullamaError::Inference(format!(
                "weight matmul: unsupported quant dtype {other:?} (expected F16, Q4_0, Q5_0, Q8_0, Q4_K, or Q6_K)"
            )));
        }
    }
    Ok(())
}

pub fn matmul_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.f16_matmul, "f16_chain", w, x, y, k, n);
}

/// BF16 weight matmul. Used by the audio Conformer tower (every block
/// linear in `gemma4:e2b`'s audio path is BF16).
#[allow(dead_code)]
pub fn matmul_bf16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    matmul_chained_inner(ctx, enc, &p.bf16_matmul, "bf16_chain", w, x, y, k, n);
}

/// Threshold above which the tiled batched matmul (workgroup-shared 8×8×16
/// tiling) beats the naive one-thread-per-output kernel. Empirically the
/// naive kernel is memory-bound for k ≥ 16, so we route any non-trivial
/// shape through the tiled path.
fn use_tiled_batched(k: usize, n: usize, batch: usize) -> bool {
    k >= 16 && n >= 8 && batch >= 8
}

/// Threshold above which the v2 tiled batched matmul (16×16 output tile,
/// 2×2 register sub-blocks per thread) beats the v1 kernel. Needs both
/// dims ≥ 16 so the output tile is fully populated.
fn use_tiled_batched_v2(k: usize, n: usize, batch: usize) -> bool {
    k >= 16 && n >= 16 && batch >= 16
}

/// V3 tiled batched matmul: 32×32 output tile, 4×4 register sub-blocks per
/// thread. Microbenched 1.6× faster than v2 (129 vs 80 GFLOPS on Pro 555).
/// Needs both dims ≥ 32 for the tile to be fully populated.
fn use_tiled_batched_v3(k: usize, n: usize, batch: usize) -> bool {
    k >= 16 && n >= 32 && batch >= 32
}

/// Batched BF16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i]. Used by the
/// audio Conformer tower so each block linear processes all `seq` frames
/// in a single dispatch instead of `seq` separate ones.
///
/// Routes to the tiled variant when the shape is large enough for it to win;
/// falls back to the naive one-thread-per-output kernel for tiny shapes.
pub fn matmul_bf16_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    if use_tiled_batched_v3(k, n, batch) {
        if let Some(pipe_f) = p.bf16_matmul_batched_tiled_v3_f16lds.as_ref() {
            return matmul_bf16_batched_tiled_v3_f16lds_chained(
                ctx, p, pipe_f, enc, w, x, y, k, n, batch,
            );
        }
        matmul_bf16_batched_tiled_v3_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    if use_tiled_batched_v2(k, n, batch) {
        matmul_bf16_batched_tiled_v2_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    if use_tiled_batched(k, n, batch) {
        matmul_bf16_batched_tiled_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmm.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmm.bg"),
        layout: &p.bf16_matmul_batched.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmm.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched);
    cp.set_bind_group(0, &bg, &[]);
    // 2D dispatch: x = output cols (n / 64), y = batch.
    cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
}

/// Batched f16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i]. Used by the
/// vision tower so each linear processes all `batch` patches in a single dispatch.
///
/// Routes to the tiled variant when the shape is large enough for it to win.
pub fn matmul_f16_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    // v4 (64×32 tile, 8×4 regs) exists in tree but **NOT ROUTED** — the 32
    // accumulators per thread spill on Pro 555 (117 vs 128 GFLOPS for v3 on
    // the ffn_up shape). Kept as reference; the next reader should expect a
    // similar register-pressure regression on similar GCN hardware.
    if use_tiled_batched_v3(k, n, batch) {
        if let Some(pipe_f) = p.f16_matmul_batched_tiled_v3_f16lds.as_ref() {
            return matmul_f16_batched_tiled_v3_f16lds_chained(
                ctx, p, pipe_f, enc, w, x, y, k, n, batch,
            );
        }
        matmul_f16_batched_tiled_v3_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    if use_tiled_batched_v2(k, n, batch) {
        matmul_f16_batched_tiled_v2_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    if use_tiled_batched(k, n, batch) {
        matmul_f16_batched_tiled_chained(ctx, p, enc, w, x, y, k, n, batch);
        return;
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmm.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmm.bg"),
        layout: &p.f16_matmul_batched.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmm.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched);
    cp.set_bind_group(0, &bg, &[]);
    // 2D dispatch: x = output cols (n / 64), y = batch.
    cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
}

/// Tiled batched f16-weight matmul. Same I/O as `matmul_f16_batched_chained`
/// but uses workgroup-shared tiling (TILE_M × TILE_N × TILE_K = 8 × 8 × 16).
/// Reduces memory bandwidth by ~8× on the vision shapes where the naive
/// kernel is memory-bound.
pub fn matmul_f16_batched_tiled_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt.bg"),
        layout: &p.f16_matmul_batched_tiled.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled);
    cp.set_bind_group(0, &bg, &[]);
    // Tile = 8×8 outputs per workgroup.
    cp.dispatch_workgroups((n as u32).div_ceil(8), (batch as u32).div_ceil(8), 1);
}

/// Tiled batched bf16-weight matmul. Audio analogue of the f16 tiled variant.
pub fn matmul_bf16_batched_tiled_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt.bg"),
        layout: &p.bf16_matmul_batched_tiled.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(8), (batch as u32).div_ceil(8), 1);
}

/// V2 tiled batched f16-weight matmul: 16×16 output tile per workgroup with
/// each thread computing a 2×2 register sub-block. ~2× arithmetic intensity
/// over the v1 kernel on shapes where both n and batch ≥ 16.
pub fn matmul_f16_batched_tiled_v2_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt2.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt2.bg"),
        layout: &p.f16_matmul_batched_tiled_v2.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt2.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled_v2);
    cp.set_bind_group(0, &bg, &[]);
    // Tile = 16×16 outputs per workgroup.
    cp.dispatch_workgroups((n as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
}

/// V3 tiled batched f16-weight matmul: 32×32 output tile, 4×4 register
/// sub-blocks per thread. ~2× arithmetic intensity over v2 on shapes where
/// both n and batch are ≥ 32.
pub fn matmul_f16_batched_tiled_v3_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt3.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt3.bg"),
        layout: &p.f16_matmul_batched_tiled_v3.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt3.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled_v3);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V3 tiled batched bf16 matmul with f16 LDS + f16 arithmetic. Caller-supplied
/// pipeline since it's only built when SHADER_F16 is available. Used by the
/// audio Conformer tower; bf16's wider exponent is reduced to f16 in LDS,
/// which is safe for typical activation ranges (verified by the audio_smoke
/// tower output staying numerically stable).
pub fn matmul_bf16_batched_tiled_v3_f16lds_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt3f.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt3f.bg"),
        layout: &pipe.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt3f.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V3 tiled batched matmul with f16 LDS + f16 arithmetic. Caller-supplied
/// pipeline since it's only built when SHADER_F16 is available.
pub fn matmul_f16_batched_tiled_v3_f16lds_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt3f.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt3f.bg"),
        layout: &pipe.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt3f.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V4 tiled batched f16-weight matmul: 64×32 output tile, 8×4 register
/// sub-blocks per thread. Higher arithmetic intensity than v3 (2.67 vs 2.0
/// MACs/load) and 2× more outputs per WG launch.
pub fn matmul_f16_batched_tiled_v4_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt4.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt4.bg"),
        layout: &p.f16_matmul_batched_tiled_v4.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt4.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled_v4);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(64), 1);
}

/// V3 tiled batched bf16-weight matmul: 32×32 output tile, 4×4 register
/// sub-blocks per thread. Audio analogue of the f16 v3 variant.
pub fn matmul_bf16_batched_tiled_v3_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt3.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt3.bg"),
        layout: &p.bf16_matmul_batched_tiled_v3.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt3.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled_v3);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V2 tiled batched bf16-weight matmul. Audio analogue of the f16 v2 variant.
pub fn matmul_bf16_batched_tiled_v2_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32,
        n: n as u32,
        batch: batch as u32,
        _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt2.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt2.bg"),
        layout: &p.bf16_matmul_batched_tiled_v2.get_bind_group_layout(0),
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
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt2.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled_v2);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
}
