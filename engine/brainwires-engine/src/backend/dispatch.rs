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
struct XEntParams { vocab_size: u32, target: u32, _p0: u32, _p1: u32 }

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Conv2dParams {
    in_c:  u32, in_h: u32, in_w: u32,
    out_c: u32, out_h: u32, out_w: u32,
    k_h: u32, k_w: u32,
    s_h: u32, s_w: u32,
    p_h: u32, p_w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ClampParams { n: u32, lo: f32, hi: f32, _p: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AvgPool2dParams {
    in_h: u32, in_w: u32,
    out_h: u32, out_w: u32,
    channels: u32, k: u32, _p0: u32, _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Rope2dParams { head_dim: u32, n_heads: u32, n_patches: u32, base: f32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub(crate) struct BatchedMatmulParams { pub k: u32, pub n: u32, pub batch: u32, pub _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct TransposeParams { n_patches: u32, n_heads: u32, head_dim: u32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct PosEmbedAddParams {
    n_patches: u32,
    hidden_size: u32,
    pos_size: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct VisionAttnParams { head_dim: u32, n_heads: u32, n_patches: u32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GluSplitParams { seq: u32, inner: u32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct DepthwiseConv1dParams { seq: u32, channels: u32, kernel: u32, _p: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScalarNParams { n: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct BlockLocalAttnParams {
    seq:          u32,
    padded_len:   u32,
    hidden:       u32,
    n_heads:      u32,
    head_dim:     u32,
    chunk_size:   u32,
    context_size: u32,
    max_span:     u32,
    max_past:     u32,
    max_future:   u32,
    pad_left:     u32,
    logit_cap:    f32,
}

// ---------- helpers ----------

/// Compute 2D dispatch dimensions for a 1D-elementwise kernel processing
/// `n_elements` items at `wg_size`. WebGPU mandates max workgroups per
/// dimension ≥ 65535; for very large `n_elements` (vision FFW is ~7M scalars
/// = 110K workgroups at wg_size=64) we wrap into the y-axis.
///
/// The corresponding kernel must compute its index as
/// `i = gid.y * 4194240u + gid.x` (where 4194240 = 65535 × 64).
fn dispatch_dims_1d(n_elements: u32, wg_size: u32) -> (u32, u32, u32) {
    const MAX_WG_PER_DIM: u32 = 65535;
    let total = n_elements.div_ceil(wg_size);
    if total <= MAX_WG_PER_DIM {
        (total, 1, 1)
    } else {
        (MAX_WG_PER_DIM, total.div_ceil(MAX_WG_PER_DIM), 1)
    }
}

pub(crate) fn write_uniform<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &T) -> wgpu::Buffer {
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

// ---------- Q4_K backward w.r.t. input (parity-test convenience) ----------

/// Async helper: allocate buffers, run `matmul_q4_k_backward_input`,
/// read `dx` back. Parity-test convenience; hot training paths should
/// use the chained dispatcher with pre-allocated persistent buffers.
pub async fn matmul_q4_k_backward_input_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w_bytes: &[u8],
    dy: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if k == 0 || n == 0 {
        return Ok(vec![0.0; k]);
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let w_buf = write_storage(device, queue, "q4k_bwd.w", w_bytes);
    let dy_buf = write_storage_f32(device, queue, "q4k_bwd.dy", dy);
    let n_bytes = (k * 4) as u64;
    let (dx_buf, dx_read) = make_output_pair(device, "q4k_bwd.dx", n_bytes);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("q4k_bwd.enc"),
    });
    matmul_q4_k_backward_input_chained(
        ctx, p, &mut enc, &w_buf, &dy_buf, &dx_buf, k, n,
    );
    enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &dx_read).await
}

// ---------- cross-entropy backward (parity-test convenience) ----------

/// Async helper: build buffers, dispatch `cross_entropy_backward`, read
/// the gradient and loss back to the host. Useful for parity tests and
/// occasional host-side instrumentation. Hot training paths should use
/// `cross_entropy_backward_chained` with pre-allocated buffers and avoid
/// the per-call readback.
pub async fn cross_entropy_backward_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    logits: &[f32],
    target: u32,
) -> Result<(Vec<f32>, f32)> {
    let n = logits.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let logits_buf = write_storage_f32(device, queue, "xent.logits", logits);
    let n_bytes = (n * 4) as u64;
    let (d_logits_buf, d_logits_read) = make_output_pair(device, "xent.dlog", n_bytes);
    let (loss_buf, loss_read) = make_output_pair(device, "xent.loss", 4);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xent.enc"),
    });
    cross_entropy_backward_chained(
        ctx, p, &mut enc, &logits_buf, &d_logits_buf, &loss_buf, n, target,
    );
    enc.copy_buffer_to_buffer(&d_logits_buf, 0, &d_logits_read, 0, n_bytes);
    enc.copy_buffer_to_buffer(&loss_buf, 0, &loss_read, 0, 4);
    queue.submit(Some(enc.finish()));
    let d_logits = read_back_f32(device, &d_logits_read).await?;
    let loss_vec = read_back_f32(device, &loss_read).await?;
    Ok((d_logits, loss_vec[0]))
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

/// Same as `matmul_chained_inner` but dispatches with WG-size-256 stride so
/// the WG count uses ceil(n/256) instead of ceil(n/64). The kernel must
/// declare `@workgroup_size(256)`.
fn matmul_chained_inner_wg256(
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
    cp.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
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
    // A/B on AMD Pro 555 / Metal, single-token gemma4:e2b "Hi":
    //   non-tiled WG=64  (default): 937 ms/tok
    //   non-tiled WG=256          : 939 ms/tok  (neutral — text is weight-bw bound)
    //   tiled    WG=64            : 996 ms/tok  (-6%)
    //   tiled    WG=64 + f16 LDS  : 975 ms/tok  (-4%)
    // The 3 alternatives are built (when relevant features) but unrouted.
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

/// BF16 weight matmul. Used by the audio Conformer tower (every block
/// linear in `gemma4:e2b`'s audio path is BF16).
#[allow(dead_code)]
pub fn matmul_bf16_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, k: usize, n: usize,
) {
    matmul_chained_inner(&ctx.device, &ctx.queue, enc, &p.bf16_matmul, "bf16_chain", w, x, y, k, n);
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

/// Half-residual add: x[i] = x[i] + 0.5 * y[i] (Conformer FFW).
pub fn half_residual_add_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ScalarNParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "halfres.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("halfres.bg"),
        layout: &p.half_residual_add.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("halfres.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.half_residual_add);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// In-place SiLU: x[i] = x[i] * sigmoid(x[i]).
pub fn silu_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ScalarNParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "silu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu.bg"),
        layout: &p.silu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("silu.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.silu);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// GLU split: y[t, d] = x[t, d] * sigmoid(x[t, inner + d]).
/// `x` is `[seq, 2 * inner]`, `y` is `[seq, inner]`.
pub fn glu_split_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, y: &wgpu::Buffer, seq: usize, inner: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GluSplitParams { seq: seq as u32, inner: inner as u32, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "glu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glu.bg"),
        layout: &p.glu_split.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        ],
    });
    let total = (seq * inner) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("glu.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.glu_split);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Depthwise 1D convolution along the time axis (Conformer LightConv).
/// `x`: `[seq, channels]` f32. `w`: `[channels, kernel]` f32. `y`: `[seq, channels]`.
pub fn depthwise_conv1d_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer,
    seq: usize, channels: usize, kernel: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = DepthwiseConv1dParams {
        seq: seq as u32, channels: channels as u32, kernel: kernel as u32, _p: 0,
    };
    let p_buf = write_uniform(device, queue, "dwconv.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dwconv.bg"),
        layout: &p.depthwise_conv1d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let total = (seq * channels) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("dwconv.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.depthwise_conv1d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Add 2D position embeddings to per-patch hidden states (vision tower).
/// hidden[p, d] += pos_embd_X[posX[p], d] + pos_embd_Y[posY[p], d]
pub fn pos_embed_add_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    hidden: &wgpu::Buffer, pos_embd: &wgpu::Buffer, pos_x: &wgpu::Buffer, pos_y: &wgpu::Buffer,
    n_patches: usize, hidden_size: usize, pos_size: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: hidden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pos_embd.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pos_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pos_y.as_entire_binding() },
        ],
    });
    let total = (n_patches * hidden_size) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("posembed.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.pos_embed_add);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Bidirectional batched self-attention for the vision tower. Reuses the same
/// q/k/v/out layout as text attention but skips causal masking and adds a
/// per-batch-query workgroup dimension.
pub fn vision_attention_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            return vision_attention_flash_subgroup_chained(ctx, p, sub, enc, q, k, v, out, head_dim, n_heads, n_patches);
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattn.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
}

/// Flash-attention-style bidirectional self-attention for vision.
/// Tiles K, V in chunks of 32 patches into workgroup-shared memory and runs
/// online softmax. Same I/O as `vision_attention_chained`.
pub fn vision_attention_flash_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnf.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.vision_attention_flash);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
}

/// Multi-query flash vision attention Q=16. Same idea as Q=8 with double
/// queries-per-WG. Workgroup storage right at the 16 KB WebGPU minimum.
pub fn vision_attention_flash_q16_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq16.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq8.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, sub: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSub.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(sub);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(8);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Transpose [n_patches, n_heads, head_dim] → [n_heads, n_patches, head_dim].
pub fn transpose_phd_to_hpd_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer, dst: &wgpu::Buffer,
    n_patches: usize, n_heads: usize, head_dim: usize,
) {
    transpose_chained(ctx, &p.transpose_phd_to_hpd, "tposePHDtoHPD", enc, src, dst, n_patches, n_heads, head_dim);
}

/// Inverse: head-major → patch-major.
pub fn transpose_hpd_to_phd_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer, dst: &wgpu::Buffer,
    n_patches: usize, n_heads: usize, head_dim: usize,
) {
    transpose_chained(ctx, &p.transpose_hpd_to_phd, "tposeHPDtoPHD", enc, src, dst, n_patches, n_heads, head_dim);
}

fn transpose_chained(
    ctx: &WgpuCtx, pipe: &wgpu::ComputePipeline, label: &str,
    enc: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer, dst: &wgpu::Buffer,
    n_patches: usize, n_heads: usize, head_dim: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&format!("{label}.pass")), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
) {
    vision_attention_flash_sub_hpd_chained(ctx, p, pipe, enc, q, k, v, out, head_dim, n_heads, n_patches);
}

/// HPD + f16-LDS attention WITHOUT subgroups. Bind-group + dispatch shape
/// identical to the subgroup variant — they differ only in WGSL.
pub fn vision_attention_flash_hpd_f16_chained(
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
) {
    vision_attention_flash_sub_hpd_chained(ctx, p, pipe, enc, q, k, v, out, head_dim, n_heads, n_patches);
}

/// Q=16 variant of `vision_attention_flash_sub_hpd_f16_chained`. Dispatches
/// half as many WGs (`ceil(n_patches/16)` per head). Uses the same uniform
/// layout so the dispatcher just wraps the same bind-group construction.
pub fn vision_attention_flash_sub_hpd_f16_q16_chained(
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubHPDQ16.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    let n_query_groups = (n_patches as u32).div_ceil(16);
    cp.dispatch_workgroups(n_query_groups, n_heads as u32, 1);
}

/// Head-major (HPD) subgroup flash attention. Caller must pre-transpose Q/K/V
/// to [n_heads, n_patches, head_dim]; output is written in the same layout.
pub fn vision_attention_flash_sub_hpd_chained(
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubHPD.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnSubT64.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer, out: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize,
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
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("vattnq4.pass"), timestamp_writes: None,
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
///                                       k-scale applied, zero-padded
///   * `v_padded`  same shape as `k_padded` — V projected, zero-padded
///   * `pos_proj`  [max_span, hidden] — sinusoidal positions through linear_pos
///
/// Output: `attn_out` [padded_len, hidden]. Caller trims to `seq * hidden`.
///
/// The kernel hard-codes `head_dim = 128` (Gemma 4 audio).
pub fn block_local_attention_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q_pad: &wgpu::Buffer, k_padded: &wgpu::Buffer, v_padded: &wgpu::Buffer,
    pos_proj: &wgpu::Buffer, attn_out: &wgpu::Buffer,
    seq: usize, padded_len: usize, hidden: usize, n_heads: usize, head_dim: usize,
    chunk_size: usize, context_size: usize, max_span: usize,
    max_past: usize, max_future: usize, pad_left: usize, logit_cap: f32,
) {
    debug_assert_eq!(head_dim, 128, "block_local_attention.wgsl is hard-coded to head_dim=128");
    let device = &ctx.device;
    let queue  = &ctx.queue;
    let params = BlockLocalAttnParams {
        seq:          seq as u32,
        padded_len:   padded_len as u32,
        hidden:       hidden as u32,
        n_heads:      n_heads as u32,
        head_dim:     head_dim as u32,
        chunk_size:   chunk_size as u32,
        context_size: context_size as u32,
        max_span:     max_span as u32,
        max_past:     max_past as u32,
        max_future:   max_future as u32,
        pad_left:     pad_left as u32,
        logit_cap,
    };
    let p_buf = write_uniform(device, queue, "blattn.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blattn.bg"),
        layout: &p.block_local_attention.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_pad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_padded.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_padded.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pos_proj.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: attn_out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("blattn.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.block_local_attention);
    cp.set_bind_group(0, &bg, &[]);
    // One workgroup per (padded query position, head).
    cp.dispatch_workgroups(padded_len as u32, n_heads as u32, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScalePerInnerDimParams { n: u32, inner_dim: u32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AddBiasBatchedParams { n: u32, batch: u32, _p0: u32, _p1: u32 }

/// In-place per-inner-dim scale: x[i] *= s[i % inner_dim]. Used by the
/// audio Conformer attention to apply per-dim Q scaling across all heads.
pub fn scale_per_inner_dim_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, s: &wgpu::Buffer, n: usize, inner_dim: usize,
) {
    let device = &ctx.device;
    let queue  = &ctx.queue;
    let params = ScalePerInnerDimParams {
        n: n as u32, inner_dim: inner_dim as u32, _p0: 0, _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "scale_pd.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale_pd.bg"),
        layout: &p.scale_per_inner_dim.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: s.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("scale_pd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.scale_per_inner_dim);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// In-place per-output-dim bias add: y[b, j] += bias[j]. Used by the audio
/// projector's FC linear which has a learned bias.
pub fn add_bias_batched_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    y: &wgpu::Buffer, bias: &wgpu::Buffer, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue  = &ctx.queue;
    let params = AddBiasBatchedParams {
        n: n as u32, batch: batch as u32, _p0: 0, _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "addbias.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("addbias.bg"),
        layout: &p.add_bias_batched.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bias.as_entire_binding() },
        ],
    });
    let total = (n * batch) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("addbias.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.add_bias_batched);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
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

/// V4 tiled batched matmul: 64×32 output tile, 8×4 register sub-blocks per
/// thread. Larger output tile (2× v3) + better arithmetic intensity
/// (2.67 vs 2.0 MACs/load). Needs batch ≥ 64 to fill the 64-row dim.
fn use_tiled_batched_v4(k: usize, n: usize, batch: usize) -> bool {
    k >= 16 && n >= 32 && batch >= 64
}

/// Batched BF16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i]. Used by the
/// audio Conformer tower so each block linear processes all `seq` frames
/// in a single dispatch instead of `seq` separate ones.
///
/// Routes to the tiled variant when the shape is large enough for it to win;
/// falls back to the naive one-thread-per-output kernel for tiny shapes.
pub fn matmul_bf16_batched_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    if use_tiled_batched_v3(k, n, batch) {
        if let Some(pipe_f) = p.bf16_matmul_batched_tiled_v3_f16lds.as_ref() {
            return matmul_bf16_batched_tiled_v3_f16lds_chained(ctx, p, pipe_f, enc, w, x, y, k, n, batch);
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
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmm.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmm.bg"),
        layout: &p.bf16_matmul_batched.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmm.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    // v4 (64×32 tile, 8×4 regs) exists in tree but **NOT ROUTED** — the 32
    // accumulators per thread spill on Pro 555 (117 vs 128 GFLOPS for v3 on
    // the ffn_up shape). Kept as reference; the next reader should expect a
    // similar register-pressure regression on similar GCN hardware.
    if use_tiled_batched_v3(k, n, batch) {
        if let Some(pipe_f) = p.f16_matmul_batched_tiled_v3_f16lds.as_ref() {
            return matmul_f16_batched_tiled_v3_f16lds_chained(ctx, p, pipe_f, enc, w, x, y, k, n, batch);
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
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmm.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmm.bg"),
        layout: &p.f16_matmul_batched.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmm.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt.bg"),
        layout: &p.f16_matmul_batched_tiled.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled);
    cp.set_bind_group(0, &bg, &[]);
    // Tile = 8×8 outputs per workgroup.
    cp.dispatch_workgroups((n as u32).div_ceil(8), (batch as u32).div_ceil(8), 1);
}

/// Tiled batched bf16-weight matmul. Audio analogue of the f16 tiled variant.
pub fn matmul_bf16_batched_tiled_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt.bg"),
        layout: &p.bf16_matmul_batched_tiled.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(8), (batch as u32).div_ceil(8), 1);
}

/// V2 tiled batched f16-weight matmul: 16×16 output tile per workgroup with
/// each thread computing a 2×2 register sub-block. ~2× arithmetic intensity
/// over the v1 kernel on shapes where both n and batch ≥ 16.
pub fn matmul_f16_batched_tiled_v2_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt2.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt2.bg"),
        layout: &p.f16_matmul_batched_tiled_v2.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt2.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt3.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt3.bg"),
        layout: &p.f16_matmul_batched_tiled_v3.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt3.pass"), timestamp_writes: None,
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
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt3f.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt3f.bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt3f.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V3 tiled batched matmul with f16 LDS + f16 arithmetic. Caller-supplied
/// pipeline since it's only built when SHADER_F16 is available.
pub fn matmul_f16_batched_tiled_v3_f16lds_chained(
    ctx: &WgpuCtx, p: &Pipelines, pipe: &wgpu::ComputePipeline,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let _ = p;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt3f.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt3f.bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt3f.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(pipe);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V4 tiled batched f16-weight matmul: 64×32 output tile, 8×4 register
/// sub-blocks per thread. Higher arithmetic intensity than v3 (2.67 vs 2.0
/// MACs/load) and 2× more outputs per WG launch.
pub fn matmul_f16_batched_tiled_v4_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "f16bmmt4.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16bmmt4.bg"),
        layout: &p.f16_matmul_batched_tiled_v4.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("f16bmmt4.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.f16_matmul_batched_tiled_v4);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(64), 1);
}

/// V3 tiled batched bf16-weight matmul: 32×32 output tile, 4×4 register
/// sub-blocks per thread. Audio analogue of the f16 v3 variant.
pub fn matmul_bf16_batched_tiled_v3_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt3.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt3.bg"),
        layout: &p.bf16_matmul_batched_tiled_v3.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt3.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled_v3);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(32), (batch as u32).div_ceil(32), 1);
}

/// V2 tiled batched bf16-weight matmul. Audio analogue of the f16 v2 variant.
pub fn matmul_bf16_batched_tiled_v2_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = BatchedMatmulParams {
        k: k as u32, n: n as u32, batch: batch as u32, _pad: 0,
    };
    let p_buf = write_uniform(device, queue, "bf16bmmt2.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bf16bmmt2.bg"),
        layout: &p.bf16_matmul_batched_tiled_v2.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bf16bmmt2.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.bf16_matmul_batched_tiled_v2);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
}

/// Chained 2D convolution. Generic stride/padding so the same kernel handles
/// vision patch embed (k=16, s=16, p=0) and audio SSCP (k=3, s=2, p=1).
///
/// Layouts:
/// * `x`: f32 [in_c, in_h, in_w]
/// * `w`: f16 [out_c, in_c, k_h, k_w] (packed 2× per u32)
/// * `y`: f32 [out_c, out_h, out_w]
pub fn conv2d_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer,
    in_c: usize, in_h: usize, in_w: usize,
    out_c: usize, out_h: usize, out_w: usize,
    k_h: usize, k_w: usize, s_h: usize, s_w: usize, pad_h: usize, pad_w: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = Conv2dParams {
        in_c:  in_c as u32, in_h: in_h as u32, in_w: in_w as u32,
        out_c: out_c as u32, out_h: out_h as u32, out_w: out_w as u32,
        k_h: k_h as u32, k_w: k_w as u32,
        s_h: s_h as u32, s_w: s_w as u32,
        p_h: pad_h as u32, p_w: pad_w as u32,
    };
    let p_buf = write_uniform(device, queue, "conv2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv2d.bg"),
        layout: &p.conv2d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let total = (out_c * out_h * out_w) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("conv2d.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.conv2d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained in-place clamp: x[i] = clamp(x[i], lo, hi).
pub fn clamp_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, n: usize, lo: f32, hi: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ClampParams { n: n as u32, lo, hi, _p: 0 };
    let p_buf = write_uniform(device, queue, "clamp.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clamp.bg"),
        layout: &p.clamp.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("clamp.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.clamp);
    cp.set_bind_group(0, &bg, &[]);
    let (dx, dy, dz) = dispatch_dims_1d(n as u32, 64);
    cp.dispatch_workgroups(dx, dy, dz);
}

/// Chained QuickGELU split: y[i] = quick_gelu(gate[i]) * up[i].
pub fn quick_geglu_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer, up: &wgpu::Buffer, y: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "qgeglu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qgeglu.bg"),
        layout: &p.quick_geglu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("qgeglu.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.quick_geglu);
    cp.set_bind_group(0, &bg, &[]);
    let (dx, dy, dz) = dispatch_dims_1d(n as u32, 64);
    cp.dispatch_workgroups(dx, dy, dz);
}

/// Chained 2D average pool with kernel = stride (vision token merge).
/// Layout: x = [in_h, in_w, channels], y = [out_h, out_w, channels]; out = in / k.
pub fn avg_pool2d_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, y: &wgpu::Buffer,
    in_h: usize, in_w: usize, channels: usize, k: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let out_h = in_h / k;
    let out_w = in_w / k;
    let params = AvgPool2dParams {
        in_h: in_h as u32, in_w: in_w as u32,
        out_h: out_h as u32, out_w: out_w as u32,
        channels: channels as u32, k: k as u32, _p0: 0, _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "pool2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pool2d.bg"),
        layout: &p.avg_pool2d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        ],
    });
    let total = (out_h * out_w * channels) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pool2d.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.avg_pool2d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained 2D NeoX RoPE for the vision tower: head_dim split — first half rotates
/// by `pos_x`, second half by `pos_y`. In-place into `x`. `pos_x`/`pos_y` are
/// `array<u32>` buffers of length n_patches.
pub fn rope_2d_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, pos_x: &wgpu::Buffer, pos_y: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_patches: usize, base: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = Rope2dParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        n_patches: n_patches as u32, base,
    };
    let p_buf = write_uniform(device, queue, "rope2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope2d.bg"),
        layout: &p.rope_2d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pos_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pos_y.as_entire_binding() },
        ],
    });
    // Total threads: n_patches * n_heads * (head_dim/2) where each handles both halves.
    let total = (n_patches * n_heads * (head_dim / 2)) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rope2d.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rope_2d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AttnBackParams {
    head_dim:     u32,
    n_heads:      u32,
    n_kv_heads:   u32,
    heads_per_kv: u32,
    history_len:  u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

/// Attention backward, pass 1 of 2 — writes `d_q` and a staged
/// `d_scores` buffer (size `[n_heads, history_len]`) that pass 2 reads.
/// One workgroup per query head. `q` is intentionally *not* a binding
/// here — pass 1's outputs don't depend on it; it shows up in pass 2.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_dq_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    k_hist: &wgpu::Buffer, v_hist: &wgpu::Buffer,
    probs: &wgpu::Buffer, d_out: &wgpu::Buffer,
    d_scores: &wgpu::Buffer, d_q: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_kv_heads: usize, history_len: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let heads_per_kv = n_heads / n_kv_heads;
    let params = AttnBackParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32, heads_per_kv: heads_per_kv as u32,
        history_len: history_len as u32,
        _pad0: 0, _pad1: 0, _pad2: 0,
    };
    let p_buf = write_uniform(device, queue, "attn_bwd_dq.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bwd_dq.bg"),
        layout: &p.attention_backward_dq.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_hist.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_hist.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: probs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: d_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: d_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: d_q.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_bwd_dq.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.attention_backward_dq);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_heads as u32, 1, 1);
}

/// Attention backward, pass 2 of 2 — consumes the staged `d_scores`
/// from pass 1 and writes `d_k_hist` and `d_v_hist`.
/// Workgroups dispatched as `(n_kv_heads, history_len, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_dkv_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer, probs: &wgpu::Buffer, d_out: &wgpu::Buffer,
    d_scores: &wgpu::Buffer,
    d_k_hist: &wgpu::Buffer, d_v_hist: &wgpu::Buffer,
    head_dim: usize, n_heads: usize, n_kv_heads: usize, history_len: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let heads_per_kv = n_heads / n_kv_heads;
    let params = AttnBackParams {
        head_dim: head_dim as u32, n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32, heads_per_kv: heads_per_kv as u32,
        history_len: history_len as u32,
        _pad0: 0, _pad1: 0, _pad2: 0,
    };
    let p_buf = write_uniform(device, queue, "attn_bwd_dkv.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bwd_dkv.bg"),
        layout: &p.attention_backward_dkv.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: probs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: d_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: d_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: d_k_hist.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: d_v_hist.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_bwd_dkv.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.attention_backward_dkv);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(n_kv_heads as u32, history_len as u32, 1);
}

/// RMSNorm backward w.r.t. the input. Weight `w` is frozen (LoRA
/// convention) — pass a real `w_buf` with `has_weight = true` or any
/// dummy `wgpu::Buffer` with `has_weight = false`.
pub fn rmsnorm_backward_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer, w: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer,
    n: usize, eps: f32, has_weight: bool,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RmsParams {
        n: n as u32, eps,
        has_weight: has_weight as u32, _p: 0,
    };
    let p_buf = write_uniform(device, queue, "rms_bwd.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms_bwd.bg"),
        layout: &p.rmsnorm_backward.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dx.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rms_bwd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rmsnorm_backward);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(1, 1, 1);
}

/// GeGLU backward — produces `d_gate` and `d_up` from `dy`, `gate`, `up`.
pub fn geglu_backward_chained(
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer, up: &wgpu::Buffer, dy: &wgpu::Buffer,
    d_gate: &wgpu::Buffer, d_up: &wgpu::Buffer, n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams { n: n as u32, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = write_uniform(device, queue, "geglu_bwd.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("geglu_bwd.bg"),
        layout: &p.geglu_backward.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: up.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: d_gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: d_up.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("geglu_bwd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.geglu_backward);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// NeoX RoPE backward — inverse in-place rotation. Reuses the same
/// `RopeParams` layout and call shape as the forward.
pub fn rope_neox_backward_chained(
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
    let p_buf = write_uniform(device, queue, "rope_bwd.params", &params);
    let f_buf = factors.unwrap_or(dummy);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_bwd.bg"),
        layout: &p.rope_neox_backward.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: f_buf.as_entire_binding() },
        ],
    });
    let total = (n_heads * (rope_dims / 2)) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rope_bwd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.rope_neox_backward);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer,
    k: usize, n: usize,
) {
    assert!(k % 256 == 0, "k must be divisible by 256 for Q4_K backward");
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = MatmulParams { k: k as u32, n: n as u32, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "q4k_bwd.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4k_bwd.bg"),
        layout: &p.matmul_q4_k_backward_input.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dx.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("q4k_bwd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.matmul_q4_k_backward_input);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((k / 256) as u32, 1, 1);
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
    ctx: &WgpuCtx, p: &Pipelines, enc: &mut wgpu::CommandEncoder,
    logits: &wgpu::Buffer, d_logits: &wgpu::Buffer, loss_out: &wgpu::Buffer,
    vocab_size: usize, target: u32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = XEntParams { vocab_size: vocab_size as u32, target, _p0: 0, _p1: 0 };
    let p_buf = write_uniform(device, queue, "xent_bwd.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xent_bwd.bg"),
        layout: &p.cross_entropy_backward.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: d_logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: loss_out.as_entire_binding() },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("xent_bwd.pass"), timestamp_writes: None,
    });
    cp.set_pipeline(&p.cross_entropy_backward);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(1, 1, 1);
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


#[cfg(test)]
mod tests {
    use super::*;

    /// GPU vs CPU parity for `cross_entropy_backward` on a vocab vector with a
    /// real (non-masked) target. Verifies both the gradient and the scalar
    /// loss agree within f32 noise.
    #[test]
    fn cross_entropy_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        // Deterministic logits — keep magnitude modest so the f32 softmax
        // sum stays well-conditioned across CPU and GPU.
        let vocab = 4096usize;
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16_777_216.0) - 0.5
        };
        let logits: Vec<f32> = (0..vocab).map(|_| next() * 4.0).collect();
        let target: u32 = 137;

        // CPU oracle
        let mut cpu_grad = vec![0.0f32; vocab];
        let cpu_loss =
            crate::reference::ops::cross_entropy_backward(&logits, target, &mut cpu_grad);

        // GPU
        let (gpu_grad, gpu_loss) =
            pollster::block_on(cross_entropy_backward_cached(&ctx, &p, &logits, target))
                .expect("gpu");

        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-3,
            "loss cpu={cpu_loss} gpu={gpu_loss}"
        );
        let mut max_diff = 0.0f32;
        for (c, g) in cpu_grad.iter().zip(gpu_grad.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff < 1e-5, "d_logits max_diff = {max_diff}");
    }

    /// GPU vs CPU parity for `matmul_q4_k_backward_input`. Synthesizes a
    /// small Q4_K weight buffer from a deterministic byte stream — Q4_K
    /// block bytes are unconstrained (any pattern parses), so this exercises
    /// the dequant + transposed-matvec path without needing the local
    /// Gemma 4 GGUF fixture.
    #[test]
    fn matmul_q4_k_backward_input_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        // Small but real: k=256 (one Q4_K block per row), n=16 rows.
        let k = 256usize;
        let n = 16usize;
        let row_bytes = (k / 256) * 144;
        let total_bytes = n * row_bytes;
        let mut w_bytes = vec![0u8; total_bytes];
        let mut state: u32 = 0xDEAD_BEEF;
        for b in w_bytes.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 16) as u8;
        }
        // Clamp the f16 scale fields so the dequanted values stay bounded
        // — random f16 bit patterns can be NaN/Inf and would propagate.
        for j in 0..n {
            let off = j * row_bytes;
            // d at offset 0..2, dmin at 2..4. Use small positive f16s
            // (0.0625 and 0.03125) for repeatable, finite magnitudes.
            w_bytes[off + 0] = 0x00;
            w_bytes[off + 1] = 0x2C; // f16(0.0625)
            w_bytes[off + 2] = 0x00;
            w_bytes[off + 3] = 0x28; // f16(0.03125)
        }

        // Deterministic dy.
        let dy: Vec<f32> = (0..n).map(|j| ((j as i32 - 8) as f32) * 0.25).collect();

        // CPU oracle
        let mut cpu_dx = vec![0.0f32; k];
        crate::reference::ops::matmul_q4_k_backward_input(
            &w_bytes, &dy, k, n, &mut cpu_dx,
        );

        // GPU
        let gpu_dx = pollster::block_on(matmul_q4_k_backward_input_cached(
            &ctx, &p, &w_bytes, &dy, k, n,
        ))
        .expect("gpu");

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for (c, g) in cpu_dx.iter().zip(gpu_dx.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
            let denom = c.abs().max(1e-6);
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
        assert!(
            max_diff < 1e-3 && max_rel < 1e-3,
            "q4_k_bwd_input max_abs={max_diff} max_rel={max_rel}"
        );
    }

    /// GPU vs CPU parity for `rmsnorm_backward` with a real weight vector.
    #[test]
    fn rmsnorm_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 64usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 30.0) * 0.05).collect();
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() * 0.3 + 1.0).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 0.5).collect();
        let eps = 1e-6f32;

        let mut cpu_dx = vec![0.0f32; n];
        crate::reference::ops::rmsnorm_backward(&x, Some(&w), &dy, eps, &mut cpu_dx);

        let device = &ctx.device;
        let queue = &ctx.queue;
        let x_buf  = write_storage_f32(device, queue, "x", &x);
        let w_buf  = write_storage_f32(device, queue, "w", &w);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dx_buf, dx_read) = make_output_pair(device, "dx", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_bwd.enc"),
        });
        rmsnorm_backward_chained(&ctx, &p, &mut enc, &x_buf, &w_buf, &dy_buf, &dx_buf, n, eps, true);
        enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dx = pollster::block_on(read_back_f32(device, &dx_read)).expect("readback");

        let mut max_diff = 0.0f32;
        for (c, g) in cpu_dx.iter().zip(gpu_dx.iter()) {
            let d = (c - g).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 1e-4, "rmsnorm_bwd max_diff = {max_diff}");
    }

    /// GPU vs CPU parity for `geglu_backward`.
    #[test]
    fn geglu_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 64usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 30.0) * 0.05).collect();
        let up:   Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let dy:   Vec<f32> = (0..n).map(|i| (i as f32 * 0.4).sin()).collect();

        let mut cpu_dg = vec![0.0f32; n];
        let mut cpu_du = vec![0.0f32; n];
        crate::reference::ops::geglu_backward(&gate, &up, &dy, &mut cpu_dg, &mut cpu_du);

        let device = &ctx.device;
        let queue = &ctx.queue;
        let g_buf  = write_storage_f32(device, queue, "gate", &gate);
        let u_buf  = write_storage_f32(device, queue, "up",   &up);
        let dy_buf = write_storage_f32(device, queue, "dy",   &dy);
        let (dg_buf, dg_read) = make_output_pair(device, "dg", (n * 4) as u64);
        let (du_buf, du_read) = make_output_pair(device, "du", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("geglu_bwd.enc"),
        });
        geglu_backward_chained(&ctx, &p, &mut enc, &g_buf, &u_buf, &dy_buf, &dg_buf, &du_buf, n);
        enc.copy_buffer_to_buffer(&dg_buf, 0, &dg_read, 0, (n * 4) as u64);
        enc.copy_buffer_to_buffer(&du_buf, 0, &du_read, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dg = pollster::block_on(read_back_f32(device, &dg_read)).expect("dg readback");
        let gpu_du = pollster::block_on(read_back_f32(device, &du_read)).expect("du readback");

        let mut max_dg = 0.0f32;
        let mut max_du = 0.0f32;
        for i in 0..n {
            max_dg = max_dg.max((cpu_dg[i] - gpu_dg[i]).abs());
            max_du = max_du.max((cpu_du[i] - gpu_du[i]).abs());
        }
        assert!(max_dg < 1e-5 && max_du < 1e-5, "geglu_bwd max_dg={max_dg} max_du={max_du}");
    }

    /// GPU two-pass attention backward vs. CPU oracle.
    ///
    /// Uses the CPU `attention_forward` to generate inputs + probs, then
    /// compares both the CPU and GPU backward against each other on the
    /// same `d_out`. Small shapes (n_heads=2, n_kv_heads=1, head_dim=8,
    /// history_len=5) — keeps the test fast and exercises the GQA
    /// aggregation (heads_per_kv = 2).
    #[test]
    fn attention_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let history_len = 5usize;
        let q_len = n_heads * head_dim;
        let kv_len = history_len * n_kv_heads * head_dim;

        // Deterministic inputs.
        let q: Vec<f32> = (0..q_len)
            .map(|i| (i as f32 * 0.31).sin() * 0.4)
            .collect();
        let k_hist: Vec<f32> = (0..kv_len)
            .map(|i| (i as f32 * 0.17).cos() * 0.3)
            .collect();
        let v_hist: Vec<f32> = (0..kv_len)
            .map(|i| (i as f32 * 0.23).sin() * 0.5)
            .collect();
        let d_out: Vec<f32> = (0..q_len)
            .map(|i| (i as f32 * 0.47).cos() * 0.3 + 0.1)
            .collect();

        // Forward (CPU) — gives us probs to feed back into both backwards.
        let mut out_unused = vec![0f32; q_len];
        let mut probs = vec![0f32; n_heads * history_len];
        crate::reference::ops::attention_forward(
            &q, &k_hist, &v_hist, &mut out_unused, &mut probs,
            head_dim, n_heads, n_kv_heads, history_len,
        );

        // CPU backward
        let mut cpu_dq = vec![0f32; q_len];
        let mut cpu_dk = vec![0f32; kv_len];
        let mut cpu_dv = vec![0f32; kv_len];
        crate::reference::ops::attention_backward(
            &q, &k_hist, &v_hist, &probs, &d_out,
            &mut cpu_dq, &mut cpu_dk, &mut cpu_dv,
            head_dim, n_heads, n_kv_heads, history_len,
        );

        // GPU backward — two passes.
        let device = &ctx.device;
        let queue = &ctx.queue;
        let q_buf      = write_storage_f32(device, queue, "q",      &q);
        let k_buf      = write_storage_f32(device, queue, "k_hist", &k_hist);
        let v_buf      = write_storage_f32(device, queue, "v_hist", &v_hist);
        let probs_buf  = write_storage_f32(device, queue, "probs",  &probs);
        let dout_buf   = write_storage_f32(device, queue, "d_out",  &d_out);
        let (ds_buf, _)  = make_output_pair(device, "d_scores", (n_heads * history_len * 4) as u64);
        let (dq_buf, dq_read) = make_output_pair(device, "d_q", (q_len * 4) as u64);
        let (dk_buf, dk_read) = make_output_pair(device, "d_k_hist", (kv_len * 4) as u64);
        let (dv_buf, dv_read) = make_output_pair(device, "d_v_hist", (kv_len * 4) as u64);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_bwd.enc"),
        });
        attention_backward_dq_chained(
            &ctx, &p, &mut enc,
            &k_buf, &v_buf, &probs_buf, &dout_buf,
            &ds_buf, &dq_buf,
            head_dim, n_heads, n_kv_heads, history_len,
        );
        attention_backward_dkv_chained(
            &ctx, &p, &mut enc,
            &q_buf, &probs_buf, &dout_buf, &ds_buf,
            &dk_buf, &dv_buf,
            head_dim, n_heads, n_kv_heads, history_len,
        );
        enc.copy_buffer_to_buffer(&dq_buf, 0, &dq_read, 0, (q_len * 4) as u64);
        enc.copy_buffer_to_buffer(&dk_buf, 0, &dk_read, 0, (kv_len * 4) as u64);
        enc.copy_buffer_to_buffer(&dv_buf, 0, &dv_read, 0, (kv_len * 4) as u64);
        queue.submit(Some(enc.finish()));

        let gpu_dq = pollster::block_on(read_back_f32(device, &dq_read)).expect("dq");
        let gpu_dk = pollster::block_on(read_back_f32(device, &dk_read)).expect("dk");
        let gpu_dv = pollster::block_on(read_back_f32(device, &dv_read)).expect("dv");

        let max = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let dq_diff = max(&cpu_dq, &gpu_dq);
        let dk_diff = max(&cpu_dk, &gpu_dk);
        let dv_diff = max(&cpu_dv, &gpu_dv);
        assert!(
            dq_diff < 1e-5 && dk_diff < 1e-5 && dv_diff < 1e-5,
            "attn_bwd diffs dq={dq_diff} dk={dk_diff} dv={dv_diff}"
        );
    }

    /// GPU forward+backward round-trip restores the input (rotations are
    /// orthogonal — fwd · bwd = identity at the same `pos`).
    #[test]
    fn rope_neox_forward_then_backward_gpu_is_identity() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let head_dim = 16usize;
        let n_heads = 4usize;
        let rope_dims = 16usize;
        let pos = 11usize;
        let base = 10_000.0f32;
        let total = head_dim * n_heads;
        let orig: Vec<f32> = (0..total).map(|i| (i as f32) * 0.07 - 1.5).collect();

        let device = &ctx.device;
        let queue = &ctx.queue;
        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope.x"),
            size: (total * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&orig));
        let dummy = write_storage_f32(device, queue, "dummy", &[0.0]);
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope.read"),
            size: (total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope.enc"),
        });
        rope_neox_chained(&ctx, &p, &mut enc, &x_buf, None, &dummy,
                          head_dim, n_heads, pos, rope_dims, base);
        rope_neox_backward_chained(&ctx, &p, &mut enc, &x_buf, None, &dummy,
                                   head_dim, n_heads, pos, rope_dims, base);
        enc.copy_buffer_to_buffer(&x_buf, 0, &read_buf, 0, (total * 4) as u64);
        queue.submit(Some(enc.finish()));
        let out = pollster::block_on(read_back_f32(device, &read_buf)).expect("readback");

        let mut max_drift = 0.0f32;
        for (o, n) in orig.iter().zip(out.iter()) {
            let d = (o - n).abs();
            if d > max_drift { max_drift = d; }
        }
        assert!(max_drift < 1e-4, "rope fwd+bwd drift = {max_drift}");
    }

    /// Masked target (`u32::MAX`) emits zero gradient and zero loss on the
    /// GPU, matching the CPU oracle's masking behavior.
    #[test]
    fn cross_entropy_backward_gpu_masked_target_is_zero() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let logits: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let (gpu_grad, gpu_loss) =
            pollster::block_on(cross_entropy_backward_cached(&ctx, &p, &logits, u32::MAX))
                .expect("gpu");
        assert_eq!(gpu_loss, 0.0);
        for g in &gpu_grad {
            assert_eq!(*g, 0.0);
        }
    }
}
