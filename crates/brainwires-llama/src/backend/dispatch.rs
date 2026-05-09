//! Cached-pipeline dispatchers. Same kernels as `matmul.rs` / `elementwise.rs` but
//! they take a `&Pipelines` reference instead of compiling a fresh module each call.
//!
//! Buffers are still created per call (no pool yet) — a future optimization is to
//! pre-allocate scratch buffers sized to the model's max op shapes. For M3 closure
//! that's a perf detail; correctness is what we're chasing here.

use bytemuck::{Pod, Zeroable};
use futures_channel::oneshot;

use crate::backend::pipelines::Pipelines;
use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};

// ---------- shared param structs ----------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatmulParams { k: u32, n: u32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RmsParams { n: u32, eps: f32, has_weight: u32, _p: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct CapParams { n: u32, cap: f32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GegluParams { n: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RopeParams {
    head_dim: u32, n_heads: u32, rope_dims: u32, pos: u32,
    base: f32, has_factors: u32, _p0: u32, _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AttnParams {
    head_dim: u32, n_heads: u32, n_kv_heads: u32, heads_per_kv: u32,
    pos: u32, history_len: u32, window: u32, _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ResAddParams { n: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScaleParams { n: u32, s: f32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RmsPerRowParams {
    n_rows: u32,
    row_dim: u32,
    eps: f32,
    has_weight: u32,
}

// ---------- helpers ----------

fn write_uniform<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &T) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
    buf
}

fn write_storage(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buf, 0, bytes);
    }
    buf
}

fn write_storage_f32(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, x: &[f32]) -> wgpu::Buffer {
    write_storage(device, queue, label, bytemuck::cast_slice(x))
}

fn make_output_pair(device: &wgpu::Device, label: &str, n_bytes: u64) -> (wgpu::Buffer, wgpu::Buffer) {
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}.out")),
        size: n_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}.read")),
        size: n_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (out, read)
}

async fn read_back_f32(device: &wgpu::Device, read_buf: &wgpu::Buffer) -> Result<Vec<f32>> {
    let slice = read_buf.slice(..);
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
    read_buf.unmap();
    Ok(v)
}

// ---------- matmuls ----------

async fn run_matmul(
    ctx: &WgpuCtx,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = MatmulParams { k: k as u32, n: n as u32, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let w_buf = write_storage(device, queue, &format!("{label}.W"), w_bytes);
    let x_buf = write_storage_f32(device, queue, &format!("{label}.x"), x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, label, n_bytes);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bg")),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{label}.encoder")),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

pub async fn matmul_q4_k_cached(ctx: &WgpuCtx, p: &Pipelines, w_bytes: &[u8], x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul(ctx, &p.q4_k_matmul, "q4k_matmul", w_bytes, x, k, n).await
}
pub async fn matmul_q6_k_cached(ctx: &WgpuCtx, p: &Pipelines, w_bytes: &[u8], x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul(ctx, &p.q6_k_matmul, "q6k_matmul", w_bytes, x, k, n).await
}
#[allow(dead_code)]
pub async fn matmul_f16_cached(ctx: &WgpuCtx, p: &Pipelines, w_bytes: &[u8], x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul(ctx, &p.f16_matmul, "f16_matmul", w_bytes, x, k, n).await
}

// ---------- buffer-based matmul: weight already on GPU ----------
//
// These take a pre-uploaded `&wgpu::Buffer` instead of `&[u8]`, avoiding the
// per-call upload. Used by `forward_token_gpu_cached` with a [`WeightCache`].

async fn run_matmul_buf(
    ctx: &WgpuCtx,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    w_buf: &wgpu::Buffer,
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = MatmulParams { k: k as u32, n: n as u32, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let x_buf = write_storage_f32(device, queue, &format!("{label}.x"), x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, label, n_bytes);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bg")),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{label}.encoder")),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

pub async fn matmul_q4_k_buf(ctx: &WgpuCtx, p: &Pipelines, w: &wgpu::Buffer, x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.q4_k_matmul, "q4k_matmul_buf", w, x, k, n).await
}
pub async fn matmul_q6_k_buf(ctx: &WgpuCtx, p: &Pipelines, w: &wgpu::Buffer, x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.q6_k_matmul, "q6k_matmul_buf", w, x, k, n).await
}
#[allow(dead_code)]
pub async fn matmul_f16_buf(ctx: &WgpuCtx, p: &Pipelines, w: &wgpu::Buffer, x: &[f32], k: usize, n: usize) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.f16_matmul, "f16_matmul_buf", w, x, k, n).await
}

// ---------- rmsnorm ----------

pub async fn rmsnorm_cached(ctx: &WgpuCtx, p: &Pipelines, x: &[f32], weight: Option<&[f32]>, eps: f32) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 { return Ok(Vec::new()); }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RmsParams { n: n as u32, eps, has_weight: if weight.is_some() { 1 } else { 0 }, _p: 0 };
    let p_buf = write_uniform(device, queue, "rms.params", &params);
    let x_buf = write_storage_f32(device, queue, "rms.x", x);
    let w_buf = match weight {
        Some(w) => write_storage_f32(device, queue, "rms.w", w),
        None    => write_storage(device, queue, "rms.w_dummy", &[0u8; 4]),
    };
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "rms", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms.bg"),
        layout: &p.rmsnorm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rms.enc") });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rms.pass"), timestamp_writes: None });
        cp.set_pipeline(&p.rmsnorm);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

// ---------- softcap ----------

pub async fn softcap_cached(ctx: &WgpuCtx, p: &Pipelines, x: &[f32], cap: f32) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 { return Ok(Vec::new()); }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = CapParams { n: n as u32, cap, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "cap.params", &params);
    let x_buf = write_storage_f32(device, queue, "cap.x", x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "cap", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cap.bg"),
        layout: &p.softcap.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cap.enc") });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cap.pass"), timestamp_writes: None });
        cp.set_pipeline(&p.softcap);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

// ---------- geglu ----------

pub async fn geglu_cached(ctx: &WgpuCtx, p: &Pipelines, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(RullamaError::Inference("geglu: gate/up length mismatch".into()));
    }
    let n = gate.len();
    if n == 0 { return Ok(Vec::new()); }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "geglu.params", &params);
    let gate_buf = write_storage_f32(device, queue, "geglu.gate", gate);
    let up_buf = write_storage_f32(device, queue, "geglu.up", up);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "geglu", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("geglu.bg"),
        layout: &p.geglu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gate_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("geglu.enc") });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("geglu.pass"), timestamp_writes: None });
        cp.set_pipeline(&p.geglu);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

// ---------- rope ----------

pub async fn rope_neox_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    x: &[f32],
    head_dim: usize,
    n_heads: usize,
    pos: usize,
    rope_dims: usize,
    base: f32,
    factors: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if x.len() != head_dim * n_heads {
        return Err(RullamaError::Inference("rope: shape mismatch".into()));
    }
    if rope_dims > head_dim || rope_dims % 2 != 0 {
        return Err(RullamaError::Inference("rope: bad rope_dims".into()));
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RopeParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        rope_dims: rope_dims as u32, pos: pos as u32,
        base, has_factors: if factors.is_some() { 1 } else { 0 },
        _p0: 0, _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "rope.params", &params);
    let x_bytes = (x.len() * 4) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.x"),
        size: x_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));
    let factors_buf = match factors {
        Some(f) => write_storage_f32(device, queue, "rope.factors", f),
        None    => write_storage(device, queue, "rope.factors_dummy", &[0u8; 4]),
    };
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.read"),
        size: x_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope.bg"),
        layout: &p.rope_neox.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: factors_buf.as_entire_binding() },
        ],
    });
    let total = (n_heads * (rope_dims / 2)) as u32;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rope.enc") });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rope.pass"), timestamp_writes: None });
        cp.set_pipeline(&p.rope_neox);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&x_buf, 0, &read_buf, 0, x_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

// ============================================================================
// Chained dispatchers (M7).
//
// These do *not* submit, copy-to-readback, or block. They append a single
// compute dispatch onto the caller's encoder using only pre-allocated buffers.
// One CommandEncoder per token; one queue.submit at the end. The result is
// that a 35-layer Gemma 4 forward goes from ~420 round-trips per token (~5 s)
// to a single submit (~tens of ms on M-series).
//
// All output buffers must be created with STORAGE | COPY_DST (and COPY_SRC if
// the caller wants to copy them at the end). Input buffers must be readable
// (STORAGE). Uniform/param buffers are still created per-call because they're
// tiny — caching them per shape is a future optimization.
// ============================================================================

/// Create a 4-byte zero buffer to bind into "weight is optional" slots
/// (e.g. unweighted rmsnorm) without an extra param toggle.
pub fn make_dummy_storage(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn matmul_chained_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    k: usize,
    n: usize,
) {
    let params = MatmulParams { k: k as u32, n: n as u32, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}.bg")),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label), timestamp_writes: None,
    });
    cp.set_pipeline(pipeline);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, k: usize, n: usize,
) {
    matmul_chained_inner(&ctx.device, &ctx.queue, enc, &p.q4_k_matmul, "q4k_chain", w, x, y, k, n);
}

pub fn matmul_q6_k_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, k: usize, n: usize,
) {
    matmul_chained_inner(&ctx.device, &ctx.queue, enc, &p.q6_k_matmul, "q6k_chain", w, x, y, k, n);
}

#[allow(dead_code)]
pub fn matmul_f16_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, k: usize, n: usize,
) {
    matmul_chained_inner(&ctx.device, &ctx.queue, enc, &p.f16_matmul, "f16_chain", w, x, y, k, n);
}

/// Chained RMSNorm. `weight` of None binds a dummy zero buffer + sets `has_weight=0`,
/// matching the WGSL layout's optional-weight contract.
pub fn rmsnorm_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, weight: Option<&wgpu::Buffer>, dummy: &wgpu::Buffer,
    y: &wgpu::Buffer, n: usize, eps: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RmsParams { n: n as u32, eps, has_weight: weight.is_some() as u32, _p: 0 };
    let p_buf = write_uniform(device, queue, "rms_chain.params", &params);
    let w_buf = weight.unwrap_or(dummy);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms_chain.bg"),
        layout: &p.rmsnorm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rms_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rmsnorm);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(1, 1, 1);
}

/// Chained per-row RMSNorm: dispatches one workgroup per row, each computing
/// `y[r,:] = x[r,:] / rms(x[r,:]) * (w[:] if has_weight else 1)`. Used for
/// per-head Q/K/V norm and per-layer PLE proj_norm — the cases where the old
/// path looped per head/layer with one CPU readback each.
pub fn rmsnorm_per_row_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, weight: Option<&wgpu::Buffer>, dummy: &wgpu::Buffer,
    y: &wgpu::Buffer, n_rows: usize, row_dim: usize, eps: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RmsPerRowParams {
        n_rows: n_rows as u32, row_dim: row_dim as u32, eps,
        has_weight: weight.is_some() as u32,
    };
    let p_buf = write_uniform(device, queue, "rmspr_chain.params", &params);
    let w_buf = weight.unwrap_or(dummy);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmspr_chain.bg"),
        layout: &p.rmsnorm_per_row.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rmspr_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rmsnorm_per_row);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_rows as u32, 1, 1);
}

/// Chained softcap: in-place would be ideal, but the WGSL has separate `x`, `y`
/// bindings — so caller passes both. Output buffer can equal input on the host
/// side (alias the same wgpu::Buffer through both bindings).
pub fn softcap_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize, cap: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = CapParams { n: n as u32, cap, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "cap_chain.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cap_chain.bg"),
        layout: &p.softcap.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("cap_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.softcap);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

pub fn geglu_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer, up: &wgpu::Buffer, y: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "geglu_chain.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("geglu_chain.bg"),
        layout: &p.geglu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("geglu_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.geglu);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// Chained NeoX RoPE. The WGSL writes in-place into the `x` buffer.
pub fn rope_neox_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, factors: Option<&wgpu::Buffer>, dummy: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, pos: usize, rope_dims: usize, base: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RopeParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        rope_dims: rope_dims as u32, pos: pos as u32,
        base, has_factors: factors.is_some() as u32,
        _p0: 0, _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "rope_chain.params", &params);
    let f_buf = factors.unwrap_or(dummy);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_chain.bg"),
        layout: &p.rope_neox.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: f_buf.as_entire_binding() },
        ],
    });
    let total = (n_heads * (rope_dims / 2)) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rope_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rope_neox);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained residual_add: x[i] += y[i], in-place into `x`.
pub fn residual_add_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ResAddParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "resadd_chain.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resadd_chain.bg"),
        layout: &p.residual_add.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("resadd_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.residual_add);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// Chained scale: x[i] *= s, in-place into `x`.
pub fn scale_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, n: usize, s: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ScaleParams { n: n as u32, s, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "scale_chain.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale_chain.bg"),
        layout: &p.scale.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("scale_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.scale);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

pub fn attention_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k_hist: &wgpu::Buffer, v_hist: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_kv_heads: usize,
    pos: usize, history_len: usize, window: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = AttnParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32, heads_per_kv: (n_heads / n_kv_heads) as u32,
        pos: pos as u32, history_len: history_len as u32, window: window as u32, _p: 0,
    };
    let p_buf = write_uniform(device, queue, "attn_chain.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_chain.bg"),
        layout: &p.attention.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_hist.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_hist.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_chain.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.attention);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_heads as u32, 1, 1);
}

// ---------- attention ----------

pub async fn attention_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    q: &[f32],
    k_hist: &[f32],
    v_hist: &[f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pos: usize,
    history_len: usize,
    window: usize,
) -> Result<Vec<f32>> {
    if q.len() != n_heads * head_dim {
        return Err(RullamaError::Inference("attn: q shape".into()));
    }
    if k_hist.len() != history_len * n_kv_heads * head_dim || v_hist.len() != history_len * n_kv_heads * head_dim {
        return Err(RullamaError::Inference("attn: kv shape".into()));
    }
    if n_heads % n_kv_heads != 0 {
        return Err(RullamaError::Inference("attn: n_heads % n_kv_heads".into()));
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = AttnParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32, heads_per_kv: (n_heads / n_kv_heads) as u32,
        pos: pos as u32, history_len: history_len as u32, window: window as u32, _p: 0,
    };
    let p_buf = write_uniform(device, queue, "attn.params", &params);
    let q_buf = write_storage_f32(device, queue, "attn.q", q);
    let k_buf = write_storage_f32(device, queue, "attn.k", k_hist);
    let v_buf = write_storage_f32(device, queue, "attn.v", v_hist);
    let out_bytes = (n_heads * head_dim * 4) as u64;
    let (out_buf, read_buf) = make_output_pair(device, "attn", out_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn.bg"),
        layout: &p.attention.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("attn.enc") });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn.pass"), timestamp_writes: None });
        cp.set_pipeline(&p.attention);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(n_heads as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}
