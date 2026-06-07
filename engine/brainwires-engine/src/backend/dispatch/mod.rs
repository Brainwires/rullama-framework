//! Cached-pipeline dispatchers. Same kernels as `matmul.rs` / `elementwise.rs` but
//! they take a `&Pipelines` reference instead of compiling a fresh module each call.
//!
//! Buffers are still created per call (no pool yet) — a future optimization is to
//! pre-allocate scratch buffers sized to the model's max op shapes. For M3 closure
//! that's a perf detail; correctness is what we're chasing here.

// Kernel-dispatcher functions map argument-for-argument to WGSL uniform/binding
// slots — the apparent arg counts (8-11) are the actual shape of the GPU op,
// not a code-smell. Bundling them into a struct just shuffles the boilerplate
// around. Similarly, this module owns matmul/ffn/attention math whose loop
// indices feed multiple parallel arrays at once; `for i in 0..n` is genuinely
// clearer than zip chains there.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use bytemuck::{Pod, Zeroable};
use futures_channel::oneshot;

use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;
use crate::error::{Result, RullamaError};

// Kernel-category submodules. Each holds a cohesive group of `*_chained`
// dispatchers + their param structs; `pub use` re-exports keep the flat
// `dispatch::<fn>` paths every call site already uses. Shared helpers
// (cached_dispatch, write_storage*, wg_grid, …) stay here and are visible to
// the children as parent-private items.
mod attention;
pub use attention::*;
mod training;
pub use training::*;
mod matmul;
pub use matmul::*;
mod vocoder;
pub use vocoder::*;

/// Block (asynchronously) until all currently-submitted GPU work on `queue`
/// has finished. Used between per-layer encoder submits in the multimodal
/// towers so the next submit sees a fully-drained queue — on iOS Safari
/// WebGPU, a single CommandEncoder that spans every block of an encoder
/// (vision: 16 blocks, audio: 12 blocks) records hundreds of dispatches +
/// bind-group changes against transient resources and pushes WebKit's
/// per-encoder budget hard enough that the *next* operation (typically
/// the first text `step()`) dies silently. Splitting each block into its
/// own encoder + submit, with this fence between submits, drains the GPU
/// in small chunks and clears WebKit's working set between blocks.
///
/// On wasm32 this resolves via `GPUQueue.onSubmittedWorkDone()` (Promise),
/// also letting the JS event loop service UI updates between blocks; on
/// native the executor drives the poll.
pub async fn fence_submitted_work(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = tx.send(());
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
    rx.await
        .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
    Ok(())
}

// ---------- shared param structs ----------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatmulParams {
    k: u32,
    n: u32,
    _p0: u32,
    _p1: u32,
}

/// Params for the `matmul_q*_backward_input` kernels (Patch 6).
///
/// `j_start..j_end` bounds the sum-axis loop so callers can tile a big
/// matmul into N submits. Non-tiled callers pass `j_start=0, j_end=n,
/// accumulate=0` and get the same behavior as the pre-Patch-6 kernels.
/// Tiled callers set `accumulate=0` on the first tile (write) and
/// `accumulate=1` on tiles 1..N (add to `dx`). Used by the head
/// `outproj` backward where the single dispatch over vocab=262144 was
/// the largest Metal heap working-set spike in a training step.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatmulBackInputParams {
    k: u32,
    n: u32,
    j_start: u32,
    j_end: u32,
    accumulate: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RmsParams {
    n: u32,
    eps: f32,
    has_weight: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct CapParams {
    n: u32,
    cap: f32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct XEntParams {
    vocab_size: u32,
    target: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GegluParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RopeParams {
    head_dim: u32,
    n_heads: u32,
    rope_dims: u32,
    pos: u32,
    base: f32,
    has_factors: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AttnParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    heads_per_kv: u32,
    pos: u32,
    history_len: u32,
    window: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ResAddParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScaleParams {
    n: u32,
    s: f32,
    offset: u32,
    _p1: u32,
}

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
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    k_h: u32,
    k_w: u32,
    s_h: u32,
    s_w: u32,
    p_h: u32,
    p_w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ClampParams {
    n: u32,
    lo: f32,
    hi: f32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AvgPool2dParams {
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    channels: u32,
    k: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Rope2dParams {
    head_dim: u32,
    n_heads: u32,
    n_patches: u32,
    base: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub(crate) struct BatchedMatmulParams {
    pub k: u32,
    pub n: u32,
    pub batch: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct TransposeParams {
    n_patches: u32,
    n_heads: u32,
    head_dim: u32,
    _pad: u32,
}

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
struct VisionAttnParams {
    head_dim: u32,
    n_heads: u32,
    n_patches: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GluSplitParams {
    seq: u32,
    inner: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct DepthwiseConv1dParams {
    seq: u32,
    channels: u32,
    kernel: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScalarNParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct BlockLocalAttnParams {
    seq: u32,
    padded_len: u32,
    hidden: u32,
    n_heads: u32,
    head_dim: u32,
    chunk_size: u32,
    context_size: u32,
    max_span: u32,
    max_past: u32,
    max_future: u32,
    pad_left: u32,
    logit_cap: f32,
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

pub(crate) fn write_uniform<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &T,
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
    buf
}

fn write_storage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
) -> wgpu::Buffer {
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

pub(crate) fn write_storage_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    x: &[f32],
) -> wgpu::Buffer {
    write_storage(device, queue, label, bytemuck::cast_slice(x))
}

/// f32 → f16 storage buffer (same row-major layout). For the f16-weight matmul kernels.
pub(crate) fn write_storage_f16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    x: &[f32],
) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(x.len() * 2);
    for &v in x {
        bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    write_storage(device, queue, label, &bytes)
}

/// Raw f16 bit patterns → storage buffer, padded to an even element count so
/// the byte length is a multiple of 4 (the f16-weight kernels bind it as
/// `array<u32>`, two halves per word). The padding half is never read. Source
/// for the f16-resident weight path (host weights stored as `Vec<u16>`).
pub(crate) fn write_storage_f16_bits(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bits: &[u16],
) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity((bits.len() + 1) * 2);
    for &b in bits {
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    if bits.len() & 1 == 1 {
        bytes.extend_from_slice(&[0u8, 0u8]); // pad to even → 4-byte aligned
    }
    write_storage(device, queue, label, &bytes)
}

/// Read-write storage buffer (STORAGE | COPY_SRC | COPY_DST), zero-initialized.
/// For intermediate activations that are written by one kernel and read by the next
/// (and optionally copied out for readback).
pub(crate) fn make_storage_rw(device: &wgpu::Device, label: &str, n_floats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n_floats * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_output_pair(
    device: &wgpu::Device,
    label: &str,
    n_bytes: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
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

pub(crate) async fn read_back_f32(
    device: &wgpu::Device,
    read_buf: &wgpu::Buffer,
) -> Result<Vec<f32>> {
    let slice = read_buf.slice(..);
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
    let params = MatmulParams {
        k: k as u32,
        n: n as u32,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let w_buf = write_storage(device, queue, &format!("{label}.W"), w_bytes);
    let x_buf = write_storage_f32(device, queue, &format!("{label}.x"), x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, label, n_bytes);

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
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{label}.encoder")),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

pub async fn matmul_q4_k_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    run_matmul(ctx, &p.q4_k_matmul, "q4k_matmul", w_bytes, x, k, n).await
}
pub async fn matmul_q6_k_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    run_matmul(ctx, &p.q6_k_matmul, "q6k_matmul", w_bytes, x, k, n).await
}
#[allow(dead_code)]
pub async fn matmul_f16_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
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
    let params = MatmulParams {
        k: k as u32,
        n: n as u32,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, &format!("{label}.params"), &params);
    let x_buf = write_storage_f32(device, queue, &format!("{label}.x"), x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, label, n_bytes);

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
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{label}.encoder")),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

pub async fn matmul_q4_k_buf(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w: &wgpu::Buffer,
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.q4_k_matmul, "q4k_matmul_buf", w, x, k, n).await
}
pub async fn matmul_q6_k_buf(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w: &wgpu::Buffer,
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.q6_k_matmul, "q6k_matmul_buf", w, x, k, n).await
}
#[allow(dead_code)]
pub async fn matmul_f16_buf(
    ctx: &WgpuCtx,
    p: &Pipelines,
    w: &wgpu::Buffer,
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    run_matmul_buf(ctx, &p.f16_matmul, "f16_matmul_buf", w, x, k, n).await
}

// ---------- rmsnorm ----------

pub async fn rmsnorm_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    x: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RmsParams {
        n: n as u32,
        eps,
        has_weight: if weight.is_some() { 1 } else { 0 },
        _p: 0,
    };
    let p_buf = write_uniform(device, queue, "rms.params", &params);
    let x_buf = write_storage_f32(device, queue, "rms.x", x);
    let w_buf = match weight {
        Some(w) => write_storage_f32(device, queue, "rms.w", w),
        None => write_storage(device, queue, "rms.w_dummy", &[0u8; 4]),
    };
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "rms", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms.bg"),
        layout: &p.rmsnorm.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rms.enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms.pass"),
            timestamp_writes: None,
        });
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
    matmul_q4_k_backward_input_chained(ctx, p, &mut enc, &w_buf, &dy_buf, &dx_buf, k, n);
    enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &dx_read).await
}

// ---------- Q6_K backward w.r.t. input (parity-test convenience) ----------

/// Async helper for parity-testing `matmul_q6_k_backward_input`.
pub async fn matmul_q6_k_backward_input_cached(
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
    let w_buf = write_storage(device, queue, "q6k_bwd.w", w_bytes);
    let dy_buf = write_storage_f32(device, queue, "q6k_bwd.dy", dy);
    let n_bytes = (k * 4) as u64;
    let (dx_buf, dx_read) = make_output_pair(device, "q6k_bwd.dx", n_bytes);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("q6k_bwd.enc"),
    });
    matmul_q6_k_backward_input_chained(ctx, p, &mut enc, &w_buf, &dy_buf, &dx_buf, k, n);
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
        ctx,
        p,
        &mut enc,
        &logits_buf,
        &d_logits_buf,
        &loss_buf,
        n,
        target,
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
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = CapParams {
        n: n as u32,
        cap,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "cap.params", &params);
    let x_buf = write_storage_f32(device, queue, "cap.x", x);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "cap", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cap.bg"),
        layout: &p.softcap.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cap.enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cap.pass"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&p.softcap);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, n_bytes);
    queue.submit(Some(enc.finish()));
    read_back_f32(device, &read_buf).await
}

// ---------- geglu ----------

pub async fn geglu_cached(
    ctx: &WgpuCtx,
    p: &Pipelines,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(RullamaError::Inference(
            "geglu: gate/up length mismatch".into(),
        ));
    }
    let n = gate.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let p_buf = write_uniform(device, queue, "geglu.params", &params);
    let gate_buf = write_storage_f32(device, queue, "geglu.gate", gate);
    let up_buf = write_storage_f32(device, queue, "geglu.up", up);
    let n_bytes = (n * 4) as u64;
    let (y_buf, read_buf) = make_output_pair(device, "geglu", n_bytes);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("geglu.bg"),
        layout: &p.geglu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gate_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: up_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("geglu.enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("geglu.pass"),
            timestamp_writes: None,
        });
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
    if rope_dims > head_dim || !rope_dims.is_multiple_of(2) {
        return Err(RullamaError::Inference("rope: bad rope_dims".into()));
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = RopeParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        rope_dims: rope_dims as u32,
        pos: pos as u32,
        base,
        has_factors: if factors.is_some() { 1 } else { 0 },
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "rope.params", &params);
    let x_bytes = (x.len() * 4) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.x"),
        size: x_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));
    let factors_buf = match factors {
        Some(f) => write_storage_f32(device, queue, "rope.factors", f),
        None => write_storage(device, queue, "rope.factors_dummy", &[0u8; 4]),
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: factors_buf.as_entire_binding(),
            },
        ],
    });
    let total = (n_heads * (rope_dims / 2)) as u32;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rope.enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope.pass"),
            timestamp_writes: None,
        });
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

/// Workgroup grid for `threads` elements at workgroup_size 64, split into 2D when the 1D
/// count would exceed wgpu's 65535-per-dimension cap (large TTS sequences hit this). Kernels
/// that use it reconstruct the linear index as `gid.y * num_workgroups.x * 64 + gid.x`
/// (a no-op when y == 1, so existing 1D callers are unaffected).
pub fn wg_grid(threads: usize) -> (u32, u32, u32) {
    let wg = (threads as u32).div_ceil(64);
    if wg <= 65535 {
        (wg, 1, 1)
    } else {
        let y = wg.div_ceil(65535);
        (wg.div_ceil(y), y, 1)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Conv2dChfParams {
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    groups: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
}

/// Channel-first conv2d (StyleTTS2 style encoder): f32 weights, optional bias, groups
/// (depthwise = groups == in_c). `w [out_c, in_c/groups, kh, kw]`, x/y channel-first.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_chf_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    groups: usize,
) {
    let params = Conv2dChfParams {
        in_c: in_c as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_c: out_c as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: sh as u32,
        sw: sw as u32,
        ph: ph as u32,
        pw: pw as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
    };
    let b = bias.unwrap_or(dummy);
    cached_dispatch(
        ctx,
        enc,
        &p.conv2d_chf,
        "conv2d_chf",
        &[w, x, b, y],
        &params,
        wg_grid(out_c * out_h * out_w),
    );
}

/// f16-weight variant of [`conv2d_chf_chained`]. `w` is an f16-packed weight
/// buffer (2 halves per u32, `write_storage_f16` layout); x/bias/y stay f32.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_chf_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    groups: usize,
) {
    let params = Conv2dChfParams {
        in_c: in_c as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_c: out_c as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        kh: kh as u32,
        kw: kw as u32,
        sh: sh as u32,
        sw: sw as u32,
        ph: ph as u32,
        pw: pw as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
    };
    let b = bias.unwrap_or(dummy);
    cached_dispatch(
        ctx,
        enc,
        &p.conv2d_chf_f16,
        "conv2d_chf_f16",
        &[w, x, b, y],
        &params,
        wg_grid(out_c * out_h * out_w),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AvgPoolHalfChfParams {
    c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// Channel-first 2×2 average pool with odd-width last-column repeat (StyleTTS2 DownSample).
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_half_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) {
    let params = AvgPoolHalfChfParams {
        c: c as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.avg_pool2d_half_chf,
        "avgpool_chf",
        &[x, y],
        &params,
        wg_grid(c * out_h * out_w),
    );
}

/// **The hot path for every chained dispatcher.** Looks up (or builds
/// on first miss) the cached `(uniform, bind_group)` for this
/// `pipeline` × `buffers` combo, writes the per-call `params` into the
/// persistent uniform, then dispatches.
///
/// On iOS Safari WebGPU every `create_buffer` and `create_bind_group`
/// is an IPC round-trip to GPUProcess + descriptor bookkeeping. A
/// training step does ~30,000 of each without caching — directly
/// matching the WebKit-bug-302711 GPUProcess pressure pattern that
/// jetsam'd the WebContent tab in our iPhone tests. Cache hits skip
/// both — only the `write_buffer` of fresh params runs, which is what
/// the GPUProcess is designed for at scale.
///
/// Bindings 1..=buffers.len() — binding 0 is always the uniform.
/// Supports 2..=7 storage buffers (covers every shape in this file —
/// `attention_backward_dq_chained` at 6 is the verified worst case).
fn cached_dispatch<T: bytemuck::Pod>(
    ctx: &WgpuCtx,
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    buffers: &[&wgpu::Buffer],
    params: &T,
    wg: (u32, u32, u32),
) {
    let key = match buffers.len() {
        1 => crate::backend::CacheKey::one(pipeline, buffers[0]),
        2 => crate::backend::CacheKey::two(pipeline, buffers[0], buffers[1]),
        3 => crate::backend::CacheKey::three(pipeline, buffers[0], buffers[1], buffers[2]),
        4 => {
            crate::backend::CacheKey::four(pipeline, buffers[0], buffers[1], buffers[2], buffers[3])
        }
        5 => crate::backend::CacheKey::five(
            pipeline, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        ),
        6 => crate::backend::CacheKey::six(
            pipeline, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
        ),
        7 => crate::backend::CacheKey::seven(
            pipeline, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
            buffers[6],
        ),
        n => panic!("cached_dispatch supports 1..=7 storage buffers, got {n}"),
    };
    let cached = ctx.bind_cache.get_or_create(key, || {
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}.params")),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(buffers.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        });
        for (i, b) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}.bg")),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        crate::backend::CachedDispatch {
            uniform,
            bind_group,
        }
    });
    ctx.queue
        .write_buffer(&cached.uniform, 0, bytemuck::bytes_of(params));
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cp.set_pipeline(pipeline);
    cp.set_bind_group(0, &cached.bind_group, &[]);
    cp.dispatch_workgroups(wg.0, wg.1, wg.2);
}

/// Chained RMSNorm. `weight` of None binds a dummy zero buffer + sets `has_weight=0`,
/// matching the WGSL layout's optional-weight contract.
pub fn rmsnorm_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
    eps: f32,
) {
    let params = RmsParams {
        n: n as u32,
        eps,
        has_weight: weight.is_some() as u32,
        _p: 0,
    };
    let w_buf = weight.unwrap_or(dummy);
    cached_dispatch(
        ctx,
        enc,
        &p.rmsnorm,
        "rms_chain",
        &[x, w_buf, y],
        &params,
        (1, 1, 1),
    );
}

/// Half-residual add: x[i] = x[i] + 0.5 * y[i] (Conformer FFW).
pub fn half_residual_add_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ScalarNParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let p_buf = write_uniform(device, queue, "halfres.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("halfres.bg"),
        layout: &p.half_residual_add.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("halfres.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.half_residual_add);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// In-place SiLU: x[i] = x[i] * sigmoid(x[i]).
pub fn silu_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ScalarNParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let p_buf = write_uniform(device, queue, "silu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu.bg"),
        layout: &p.silu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("silu.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.silu);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
}

/// GLU split: y[t, d] = x[t, d] * sigmoid(x[t, inner + d]).
/// `x` is `[seq, 2 * inner]`, `y` is `[seq, inner]`.
pub fn glu_split_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    seq: usize,
    inner: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GluSplitParams {
        seq: seq as u32,
        inner: inner as u32,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "glu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glu.bg"),
        layout: &p.glu_split.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.as_entire_binding(),
            },
        ],
    });
    let total = (seq * inner) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("glu.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.glu_split);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Depthwise 1D convolution along the time axis (Conformer LightConv).
/// `x`: `[seq, channels]` f32. `w`: `[channels, kernel]` f32. `y`: `[seq, channels]`.
pub fn depthwise_conv1d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    seq: usize,
    channels: usize,
    kernel: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = DepthwiseConv1dParams {
        seq: seq as u32,
        channels: channels as u32,
        kernel: kernel as u32,
        _p: 0,
    };
    let p_buf = write_uniform(device, queue, "dwconv.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dwconv.bg"),
        layout: &p.depthwise_conv1d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y.as_entire_binding(),
            },
        ],
    });
    let total = (seq * channels) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("dwconv.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.depthwise_conv1d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ScalePerInnerDimParams {
    n: u32,
    inner_dim: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AddBiasBatchedParams {
    n: u32,
    batch: u32,
    _p0: u32,
    _p1: u32,
}

/// In-place per-inner-dim scale: x[i] *= s[i % inner_dim]. Used by the
/// audio Conformer attention to apply per-dim Q scaling across all heads.
pub fn scale_per_inner_dim_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    s: &wgpu::Buffer,
    n: usize,
    inner_dim: usize,
) {
    let params = ScalePerInnerDimParams {
        n: n as u32,
        inner_dim: inner_dim as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.scale_per_inner_dim,
        "scale_pd",
        &[x, s],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
}

/// In-place per-output-dim bias add: y[b, j] += bias[j]. Used by the audio
/// projector's FC linear which has a learned bias.
pub fn add_bias_batched_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    y: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    n: usize,
    batch: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = AddBiasBatchedParams {
        n: n as u32,
        batch: batch as u32,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "addbias.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("addbias.bg"),
        layout: &p.add_bias_batched.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias.as_entire_binding(),
            },
        ],
    });
    let total = (n * batch) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("addbias.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.add_bias_batched);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained 2D convolution. Generic stride/padding so the same kernel handles
/// vision patch embed (k=16, s=16, p=0) and audio SSCP (k=3, s=2, p=1).
///
/// Layouts:
/// * `x`: f32 [in_c, in_h, in_w]
/// * `w`: f16 [out_c, in_c, k_h, k_w] (packed 2× per u32)
/// * `y`: f32 [out_c, out_h, out_w]
pub fn conv2d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    k_h: usize,
    k_w: usize,
    s_h: usize,
    s_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = Conv2dParams {
        in_c: in_c as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_c: out_c as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        k_h: k_h as u32,
        k_w: k_w as u32,
        s_h: s_h as u32,
        s_w: s_w as u32,
        p_h: pad_h as u32,
        p_w: pad_w as u32,
    };
    let p_buf = write_uniform(device, queue, "conv2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv2d.bg"),
        layout: &p.conv2d.get_bind_group_layout(0),
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
    let total = (out_c * out_h * out_w) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("conv2d.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.conv2d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained in-place clamp: x[i] = clamp(x[i], lo, hi).
pub fn clamp_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    n: usize,
    lo: f32,
    hi: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = ClampParams {
        n: n as u32,
        lo,
        hi,
        _p: 0,
    };
    let p_buf = write_uniform(device, queue, "clamp.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clamp.bg"),
        layout: &p.clamp.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("clamp.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.clamp);
    cp.set_bind_group(0, &bg, &[]);
    let (dx, dy, dz) = dispatch_dims_1d(n as u32, 64);
    cp.dispatch_workgroups(dx, dy, dz);
}

/// Chained QuickGELU split: y[i] = quick_gelu(gate[i]) * up[i].
pub fn quick_geglu_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let p_buf = write_uniform(device, queue, "qgeglu.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qgeglu.bg"),
        layout: &p.quick_geglu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gate.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: up.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y.as_entire_binding(),
            },
        ],
    });
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("qgeglu.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.quick_geglu);
    cp.set_bind_group(0, &bg, &[]);
    let (dx, dy, dz) = dispatch_dims_1d(n as u32, 64);
    cp.dispatch_workgroups(dx, dy, dz);
}

/// Chained 2D average pool with kernel = stride (vision token merge).
/// Layout: x = [in_h, in_w, channels], y = [out_h, out_w, channels]; out = in / k.
pub fn avg_pool2d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    in_h: usize,
    in_w: usize,
    channels: usize,
    k: usize,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let out_h = in_h / k;
    let out_w = in_w / k;
    let params = AvgPool2dParams {
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        channels: channels as u32,
        k: k as u32,
        _p0: 0,
        _p1: 0,
    };
    let p_buf = write_uniform(device, queue, "pool2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pool2d.bg"),
        layout: &p.avg_pool2d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y.as_entire_binding(),
            },
        ],
    });
    let total = (out_h * out_w * channels) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pool2d.pass"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&p.avg_pool2d);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(total.div_ceil(64), 1, 1);
}

/// Chained 2D NeoX RoPE for the vision tower: head_dim split — first half rotates
/// by `pos_x`, second half by `pos_y`. In-place into `x`. `pos_x`/`pos_y` are
/// `array<u32>` buffers of length n_patches.
pub fn rope_2d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    pos_x: &wgpu::Buffer,
    pos_y: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_patches: usize,
    base: f32,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = Rope2dParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_patches: n_patches as u32,
        base,
    };
    let p_buf = write_uniform(device, queue, "rope2d.params", &params);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope2d.bg"),
        layout: &p.rope_2d.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pos_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pos_y.as_entire_binding(),
            },
        ],
    });
    // Total threads: n_patches * n_heads * (head_dim/2) where each handles both halves.
    let total = (n_patches * n_heads * (head_dim / 2)) as u32;
    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rope2d.pass"),
        timestamp_writes: None,
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
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n_rows: usize,
    row_dim: usize,
    eps: f32,
) {
    let params = RmsPerRowParams {
        n_rows: n_rows as u32,
        row_dim: row_dim as u32,
        eps,
        has_weight: weight.is_some() as u32,
    };
    let w_buf = weight.unwrap_or(dummy);
    cached_dispatch(
        ctx,
        enc,
        &p.rmsnorm_per_row,
        "rmspr_chain",
        &[x, w_buf, y],
        &params,
        (n_rows as u32, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LayerNormAffineParams {
    n_rows: u32,
    row_dim: u32,
    eps: f32,
    has_affine: u32,
}

/// Per-row LayerNorm (mean-subtraction + bias) with optional (gamma, beta) affine.
/// Buffers: x, gamma, beta, y. Pass `None` affine (with a dummy) for plain LN.
#[allow(clippy::too_many_arguments)]
pub fn layernorm_affine_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    gamma: Option<&wgpu::Buffer>,
    beta: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n_rows: usize,
    row_dim: usize,
    eps: f32,
) {
    let params = LayerNormAffineParams {
        n_rows: n_rows as u32,
        row_dim: row_dim as u32,
        eps,
        has_affine: gamma.is_some() as u32,
    };
    let g = gamma.unwrap_or(dummy);
    let b = beta.unwrap_or(dummy);
    cached_dispatch(
        ctx,
        enc,
        &p.layernorm_affine,
        "ln_affine",
        &[x, g, b, y],
        &params,
        (n_rows as u32, 1, 1),
    );
}

/// Chained softcap: in-place would be ideal, but the WGSL has separate `x`, `y`
/// bindings — so caller passes both. Output buffer can equal input on the host
/// side (alias the same wgpu::Buffer through both bindings).
pub fn softcap_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
    cap: f32,
) {
    let params = CapParams {
        n: n as u32,
        cap,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.softcap,
        "cap_chain",
        &[x, y],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
}

pub fn geglu_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    y: &wgpu::Buffer,
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
        &p.geglu,
        "geglu_chain",
        &[gate, up, y],
        &params,
        ((n as u32).div_ceil(64), 1, 1),
    );
}

/// Chained NeoX RoPE. The WGSL writes in-place into the `x` buffer.
pub fn rope_neox_chained(
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
        &p.rope_neox,
        "rope_chain",
        &[x, f_buf],
        &params,
        (total.div_ceil(64), 1, 1),
    );
}

/// Chained residual_add: x[i] += y[i], in-place into `x`.
pub fn residual_add_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
) {
    let params = ResAddParams {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.residual_add,
        "resadd_chain",
        &[x, y],
        &params,
        wg_grid(n),
    );
}

/// Chained scale: x[i] *= s, in-place into `x`.
pub fn scale_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    n: usize,
    s: f32,
) {
    let total_groups = (n as u32).div_ceil(64);
    // wgpu hard caps dispatch_workgroups at 65535 per dimension. For
    // very large buffers (lm_head / embed_tokens LoRA B = vocab × rank
    // ≈ 4.2M f32s → 65_536 workgroups, JUST over the cap), chunk the
    // dispatch across multiple submissions and use `offset` in the
    // shader to keep linear indexing correct.
    //
    // The bind-group cache key is identical across iterations (same
    // pipeline, same `x` buffer), so all loop iterations after the
    // first are cache hits — only the per-call uniform write differs.
    const MAX_GROUPS_PER_DISPATCH: u32 = 65535;
    let mut groups_done: u32 = 0;
    while groups_done < total_groups {
        let groups_this = (total_groups - groups_done).min(MAX_GROUPS_PER_DISPATCH);
        let params = ScaleParams {
            n: n as u32,
            s,
            offset: groups_done * 64,
            _p1: 0,
        };
        cached_dispatch(
            ctx,
            enc,
            &p.scale,
            "scale_chain",
            &[x],
            &params,
            (groups_this, 1, 1),
        );
        groups_done += groups_this;
    }
}

pub fn attention_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k_hist: &wgpu::Buffer,
    v_hist: &wgpu::Buffer,
    out: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pos: usize,
    history_len: usize,
    window: usize,
) {
    let params = AttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        heads_per_kv: (n_heads / n_kv_heads) as u32,
        pos: pos as u32,
        history_len: history_len as u32,
        window: window as u32,
        _p: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.attention,
        "attn_chain",
        &[q, k_hist, v_hist, out],
        &params,
        (n_heads as u32, 1, 1),
    );
}

/// Compute attention softmax probabilities only (no V multiply). Mirrors
/// `attention_chained`'s scoring + softmax math and writes
/// `probs[n_heads, history_len]`. Used by the training backward pass to
/// reconstruct probs from the captured `q_post_rope` and the existing KV
/// cache, so the forward attention kernel can stay unchanged.
#[allow(clippy::too_many_arguments)]
pub fn attention_probs_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    k_hist: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pos: usize,
    history_len: usize,
    window: usize,
) {
    let params = AttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        heads_per_kv: (n_heads / n_kv_heads) as u32,
        pos: pos as u32,
        history_len: history_len as u32,
        window: window as u32,
        _p: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.attention_probs,
        "attn_probs",
        &[q, k_hist, probs],
        &params,
        (n_heads as u32, 1, 1),
    );
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
    if k_hist.len() != history_len * n_kv_heads * head_dim
        || v_hist.len() != history_len * n_kv_heads * head_dim
    {
        return Err(RullamaError::Inference("attn: kv shape".into()));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(RullamaError::Inference("attn: n_heads % n_kv_heads".into()));
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = AttnParams {
        head_dim: head_dim as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        heads_per_kv: (n_heads / n_kv_heads) as u32,
        pos: pos as u32,
        history_len: history_len as u32,
        window: window as u32,
        _p: 0,
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn.enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn.pass"),
            timestamp_writes: None,
        });
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

    #[test]
    fn conv1d_gpu_vs_cpu() {
        use crate::reference::kokoro::convblocks::conv1d as cpu_conv1d;
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");

        // (cin, tin, cout, k, stride, pad, dilation, groups): same-len k5, downsample
        // stride-2, dilated, and depthwise-grouped — the Kokoro conv shapes.
        let cases = [
            (
                4usize, 20usize, 6usize, 5usize, 1usize, 2usize, 1usize, 1usize,
            ),
            (3, 31, 1, 3, 2, 1, 1, 1),
            (8, 17, 8, 3, 1, 2, 2, 1),
            (6, 13, 6, 3, 1, 1, 1, 6),
        ];
        for (ci, (cin, tin, cout, k, stride, pad, dil, groups)) in cases.into_iter().enumerate() {
            let xs: Vec<f32> = (0..cin * tin)
                .map(|i| ((i as i32 % 19 - 9) as f32) * 0.07)
                .collect();
            let ws: Vec<f32> = (0..cout * (cin / groups) * k)
                .map(|i| (i as f32 * 0.11).sin() * 0.4)
                .collect();
            let bs: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.05 - 0.1).collect();
            let (cpu, tout) = cpu_conv1d(
                &xs,
                cin,
                tin,
                &ws,
                Some(&bs),
                cout,
                k,
                stride,
                pad,
                dil,
                groups,
            );

            let xb = write_storage_f32(device, queue, "x", &xs);
            let wb = write_storage_f32(device, queue, "w", &ws);
            let bb = write_storage_f32(device, queue, "b", &bs);
            let (yb, yr) = make_output_pair(device, "y", (cout * tout * 4) as u64);
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c1d") });
            let gt = conv1d_chained(
                &ctx,
                &p,
                &mut enc,
                &xb,
                &wb,
                Some(&bb),
                &dummy,
                &yb,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                dil,
                groups,
            );
            enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (cout * tout * 4) as u64);
            queue.submit(Some(enc.finish()));
            let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("readback");
            assert_eq!(gt, tout, "case {ci} tout");
            let md = cpu
                .iter()
                .zip(&gpu)
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(md < 1e-4, "conv1d case {ci} max_diff = {md}");
        }
    }

    #[test]
    fn conv1d_f16_gpu_vs_cpu() {
        // f16-weight conv1d vs the same conv with f16-ROUNDED weights on CPU.
        // Rounding the oracle's weights through f16 isolates kernel correctness
        // (reading packed f16 via unpack2x16float) from the f16 storage
        // precision loss, which is the intended trade — not a kernel bug.
        use crate::reference::kokoro::convblocks::conv1d as cpu_conv1d;
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");

        // Same shapes as conv1d_gpu_vs_cpu; case 1 has an ODD weight-element
        // count (1*3*3=9) to exercise the u32 even-pad path.
        let cases = [
            (
                4usize, 20usize, 6usize, 5usize, 1usize, 2usize, 1usize, 1usize,
            ),
            (3, 31, 1, 3, 2, 1, 1, 1),
            (8, 17, 8, 3, 1, 2, 2, 1),
            (6, 13, 6, 3, 1, 1, 1, 6),
        ];
        for (ci, (cin, tin, cout, k, stride, pad, dil, groups)) in cases.into_iter().enumerate() {
            let xs: Vec<f32> = (0..cin * tin)
                .map(|i| ((i as i32 % 19 - 9) as f32) * 0.07)
                .collect();
            let ws: Vec<f32> = (0..cout * (cin / groups) * k)
                .map(|i| (i as f32 * 0.11).sin() * 0.4)
                .collect();
            let bs: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.05 - 0.1).collect();
            // f16 weight bits + their exact f32 values for the oracle.
            let ws16_bits: Vec<u16> = ws
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let ws16_f32: Vec<f32> = ws16_bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let (cpu, tout) = cpu_conv1d(
                &xs,
                cin,
                tin,
                &ws16_f32,
                Some(&bs),
                cout,
                k,
                stride,
                pad,
                dil,
                groups,
            );

            let xb = write_storage_f32(device, queue, "x", &xs);
            let wb = write_storage_f16_bits(device, queue, "w16", &ws16_bits);
            let bb = write_storage_f32(device, queue, "b", &bs);
            let (yb, yr) = make_output_pair(device, "y", (cout * tout * 4) as u64);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("c1d16"),
            });
            let gt = conv1d_f16_chained(
                &ctx,
                &p,
                &mut enc,
                &xb,
                &wb,
                Some(&bb),
                &dummy,
                &yb,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                dil,
                groups,
            );
            enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (cout * tout * 4) as u64);
            queue.submit(Some(enc.finish()));
            let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("readback");
            assert_eq!(gt, tout, "case {ci} tout");
            let md = cpu
                .iter()
                .zip(&gpu)
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(md < 1e-4, "conv1d_f16 case {ci} max_diff = {md}");
        }
    }

    #[test]
    fn conv_transpose1d_f16_gpu_vs_cpu() {
        use crate::reference::kokoro::convblocks::conv_transpose1d as cpu_ct;
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");
        // groups=1 ISTFTNet ups shape: 4->6, k5 stride2 pad2 outpad0.
        let (cin, tin, cout, k, stride, pad, op) =
            (4usize, 9usize, 6usize, 5usize, 2usize, 2usize, 0usize);
        let xs: Vec<f32> = (0..cin * tin)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let ws: Vec<f32> = (0..cin * cout * k)
            .map(|i| (i as f32 * 0.09).cos() * 0.3)
            .collect();
        let bs: Vec<f32> = (0..cout).map(|i| i as f32 * 0.02).collect();
        let wbits: Vec<u16> = ws
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let w16f: Vec<f32> = wbits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        let (cpu, tout) = cpu_ct(&xs, cin, tin, &w16f, Some(&bs), cout, k, stride, pad, op);
        let xb = write_storage_f32(device, queue, "x", &xs);
        let wb = write_storage_f16_bits(device, queue, "w16", &wbits);
        let bb = write_storage_f32(device, queue, "b", &bs);
        let (yb, yr) = make_output_pair(device, "y", (cout * tout * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ct16"),
        });
        let gt = conv_transpose1d_f16_chained(
            &ctx,
            &p,
            &mut enc,
            &xb,
            &wb,
            Some(&bb),
            &dummy,
            &yb,
            cin,
            tin,
            cout,
            k,
            stride,
            pad,
            op,
            1,
        );
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (cout * tout * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("readback");
        assert_eq!(gt, tout);
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "conv_transpose1d_f16 max_diff = {md}");
    }

    #[test]
    fn conv2d_chf_f16_gpu_vs_cpu() {
        use crate::reference::styletts2::{Map, conv2d as cpu_conv2d};
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");
        let (in_c, in_h, in_w, out_c, kh, kw, stride, pad, groups) = (
            3usize, 7usize, 5usize, 4usize, 3usize, 3usize, 1usize, 1usize, 1usize,
        );
        let xs: Vec<f32> = (0..in_c * in_h * in_w)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.06)
            .collect();
        let ws: Vec<f32> = (0..out_c * (in_c / groups) * kh * kw)
            .map(|i| (i as f32 * 0.13).sin() * 0.35)
            .collect();
        let bs: Vec<f32> = (0..out_c).map(|i| i as f32 * 0.03 - 0.05).collect();
        let wbits: Vec<u16> = ws
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let w16f: Vec<f32> = wbits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        let xm = Map::new(xs.clone(), in_c, in_h, in_w);
        let cpu = cpu_conv2d(&xm, &w16f, Some(&bs), out_c, kh, kw, stride, pad, groups);
        let (out_h, out_w) = (cpu.h, cpu.w);
        let xb = write_storage_f32(device, queue, "x", &xs);
        let wb = write_storage_f16_bits(device, queue, "w16", &wbits);
        let bb = write_storage_f32(device, queue, "b", &bs);
        let (yb, yr) = make_output_pair(device, "y", (out_c * out_h * out_w * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("c2d16"),
        });
        conv2d_chf_f16_chained(
            &ctx,
            &p,
            &mut enc,
            &wb,
            &xb,
            Some(&bb),
            &dummy,
            &yb,
            in_c,
            in_h,
            in_w,
            out_c,
            out_h,
            out_w,
            kh,
            kw,
            stride,
            stride,
            pad,
            pad,
            groups,
        );
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (out_c * out_h * out_w * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("readback");
        let md = cpu
            .data
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "conv2d_chf_f16 max_diff = {md}");
    }

    #[test]
    fn conv_transpose1d_gpu_vs_cpu() {
        use crate::reference::kokoro::convblocks::{
            conv_transpose1d as cpu_ct, conv_transpose1d_depthwise as cpu_ctd,
        };
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let dummy = make_dummy_storage(device, "dummy");

        // (a) general groups=1, like ISTFTNet ups: 4->6, k5 stride2 pad2 outpad0
        {
            let (cin, tin, cout, k, stride, pad, op) =
                (4usize, 9usize, 6usize, 5usize, 2usize, 2usize, 0usize);
            let xs: Vec<f32> = (0..cin * tin)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                .collect();
            let ws: Vec<f32> = (0..cin * cout * k)
                .map(|i| (i as f32 * 0.09).cos() * 0.3)
                .collect();
            let bs: Vec<f32> = (0..cout).map(|i| i as f32 * 0.02).collect();
            let (cpu, tout) = cpu_ct(&xs, cin, tin, &ws, Some(&bs), cout, k, stride, pad, op);
            let xb = write_storage_f32(device, queue, "x", &xs);
            let wb = write_storage_f32(device, queue, "w", &ws);
            let bb = write_storage_f32(device, queue, "b", &bs);
            let (yb, yr) = make_output_pair(device, "y", (cout * tout * 4) as u64);
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ct") });
            let gt = conv_transpose1d_chained(
                &ctx,
                &p,
                &mut enc,
                &xb,
                &wb,
                Some(&bb),
                &dummy,
                &yb,
                cin,
                tin,
                cout,
                k,
                stride,
                pad,
                op,
                1,
            );
            enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (cout * tout * 4) as u64);
            queue.submit(Some(enc.finish()));
            let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
            assert_eq!(gt, tout);
            let md = cpu
                .iter()
                .zip(&gpu)
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(md < 1e-4, "convT general max_diff = {md}");
        }
        // (b) depthwise groups=C, like StyleTTS2 pool: C5, k3 stride2 pad1 outpad1
        {
            let (c, tin, k, stride, pad, op) = (5usize, 7usize, 3usize, 2usize, 1usize, 1usize);
            let xs: Vec<f32> = (0..c * tin)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.06)
                .collect();
            let ws: Vec<f32> = (0..c * k).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
            let bs: Vec<f32> = (0..c).map(|i| i as f32 * 0.03).collect();
            let (cpu, tout) = cpu_ctd(&xs, c, tin, &ws, Some(&bs), k, stride, pad, op);
            let xb = write_storage_f32(device, queue, "x", &xs);
            let wb = write_storage_f32(device, queue, "w", &ws);
            let bb = write_storage_f32(device, queue, "b", &bs);
            let (yb, yr) = make_output_pair(device, "y", (c * tout * 4) as u64);
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ctd") });
            let gt = conv_transpose1d_chained(
                &ctx,
                &p,
                &mut enc,
                &xb,
                &wb,
                Some(&bb),
                &dummy,
                &yb,
                c,
                tin,
                c,
                k,
                stride,
                pad,
                op,
                c,
            );
            enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (c * tout * 4) as u64);
            queue.submit(Some(enc.finish()));
            let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
            assert_eq!(gt, tout);
            let md = cpu
                .iter()
                .zip(&gpu)
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(md < 1e-4, "convT depthwise max_diff = {md}");
        }
    }

    #[test]
    fn leaky_relu_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let n = 200usize;
        let slope = 0.2f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 100.0) * 0.03).collect();
        let mut cpu = x.clone();
        crate::reference::kokoro::ops::leaky_relu(&mut cpu, slope);
        // in-place buffer needs COPY_SRC to read it back in the test (production
        // feeds it straight to the next kernel, so this usage is test-only).
        let yb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&yb, 0, bytemuck::cast_slice(&x));
        let (_y_unused, yr) = make_output_pair(device, "yr", (n * 4) as u64);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lr") });
        leaky_relu_chained(&ctx, &p, &mut enc, &yb, n, slope);
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-6, "leaky_relu max_diff = {md}");
    }

    #[test]
    fn snake_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let (c, t) = (6usize, 23usize);
        let x: Vec<f32> = (0..c * t).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let alpha: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.2).collect();
        let mut cpu = x.clone();
        crate::reference::kokoro::convblocks::snake(&mut cpu, c, t, &alpha);
        let xb = write_storage_f32(device, queue, "x", &x);
        let ab = write_storage_f32(device, queue, "a", &alpha);
        let (yb, yr) = make_output_pair(device, "y", (c * t * 4) as u64);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sn") });
        snake_chained(&ctx, &p, &mut enc, &xb, &ab, &yb, c, t);
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (c * t * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "snake max_diff = {md}");
    }

    #[test]
    fn adain_gpu_vs_cpu() {
        use crate::reference::kokoro::ops::linear;
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let (c, t, sd) = (8usize, 19usize, 5usize);
        let x: Vec<f32> = (0..c * t).map(|i| ((i % 13) as f32 - 6.0) * 0.07).collect();
        let style: Vec<f32> = (0..sd).map(|i| (i as f32 * 0.3).sin()).collect();
        let fc_w: Vec<f32> = (0..2 * c * sd)
            .map(|i| (i as f32 * 0.05).cos() * 0.2)
            .collect();
        let fc_b: Vec<f32> = (0..2 * c).map(|i| i as f32 * 0.01).collect();
        // CPU oracle (norm affine absent → identity)
        let cpu = crate::reference::kokoro::convblocks::adain1d(
            &x, c, t, None, None, &fc_w, &fc_b, &style, sd,
        );
        // GPU: precompute gamma/beta = chunk(fc(style))
        let gb = linear(&style, 1, sd, &fc_w, Some(&fc_b), 2 * c);
        let (gamma, beta) = gb.split_at(c);
        let xb = write_storage_f32(device, queue, "x", &x);
        let gbuf = write_storage_f32(device, queue, "g", gamma);
        let bbuf = write_storage_f32(device, queue, "b", beta);
        let (yb, yr) = make_output_pair(device, "y", (c * t * 4) as u64);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ad") });
        adain_chained(&ctx, &p, &mut enc, &xb, &gbuf, &bbuf, &yb, c, t, 1e-5);
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (c * t * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "adain max_diff = {md}");
    }

    #[test]
    fn transpose2d_gpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let (rows, cols) = (5usize, 7usize);
        let x: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let mut cpu = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                cpu[c * rows + r] = x[r * cols + c];
            }
        }
        let xb = write_storage_f32(device, queue, "x", &x);
        let (yb, yr) = make_output_pair(device, "y", (rows * cols * 4) as u64);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tr") });
        transpose2d_chained(&ctx, &p, &mut enc, &xb, &yb, rows, cols);
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (rows * cols * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
        assert_eq!(cpu, gpu);
    }

    #[test]
    fn istft_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let (nbins, frames, nfft, hop) = (11usize, 30usize, 20usize, 5usize);
        let spec: Vec<f32> = (0..nbins * frames)
            .map(|i| ((i * 7 % 13) as f32) * 0.1 + 0.05)
            .collect();
        let phase: Vec<f32> = (0..nbins * frames)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();
        let cpu =
            crate::reference::kokoro::generator::istft(&spec, &phase, nbins, frames, nfft, hop);

        let sb = write_storage_f32(device, queue, "spec", &spec);
        let pb = write_storage_f32(device, queue, "phase", &phase);
        let (yb, yr) = make_output_pair(device, "y", (cpu.len() * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("istft"),
        });
        let ol = istft_chained(&ctx, &p, &mut enc, &sb, &pb, &yb, nbins, frames, nfft, hop);
        enc.copy_buffer_to_buffer(&yb, 0, &yr, 0, (cpu.len() * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &yr)).expect("rb");
        assert_eq!(ol, cpu.len());
        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "istft max_diff = {md}");
    }

    #[test]
    fn layernorm_affine_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;
        let n_rows = 7usize;
        let dim = 53usize;
        let total = n_rows * dim;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i as i32 - 100) as f32) * 0.013)
            .collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * 0.2).sin() * 0.5 + 1.0)
            .collect();
        let beta: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos() * 0.3).collect();

        let cpu = crate::reference::kokoro::ops::layer_norm(&x, n_rows, dim, &gamma, &beta, eps);

        let x_buf = write_storage_f32(device, queue, "x", &x);
        let g_buf = write_storage_f32(device, queue, "g", &gamma);
        let b_buf = write_storage_f32(device, queue, "b", &beta);
        let (y_buf, y_read) = make_output_pair(device, "y", (total * 4) as u64);
        let dummy = make_dummy_storage(device, "dummy");
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ln") });
        layernorm_affine_chained(
            &ctx,
            &p,
            &mut enc,
            &x_buf,
            Some(&g_buf),
            Some(&b_buf),
            &dummy,
            &y_buf,
            n_rows,
            dim,
            eps,
        );
        enc.copy_buffer_to_buffer(&y_buf, 0, &y_read, 0, (total * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &y_read)).expect("readback");

        let md = cpu
            .iter()
            .zip(&gpu)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "layernorm_affine max_diff = {md}");
    }

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

    /// **D5 — fuzz-magnitude cross-entropy backward.** Same kernel,
    /// inputs scaled to logit magnitudes a barely-trained or
    /// diverging model produces (some outputs at ±50, most ~0). The
    /// stability concern is the LogSumExp in CE-loss: a single large
    /// logit can blow up the softmax sum if the GPU implementation
    /// isn't doing the max-subtract trick. Tolerance loosened for
    /// the larger floor on f32 round-off at these scales.
    #[test]
    fn cross_entropy_backward_gpu_vs_cpu_wide_magnitude() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        let vocab = 4096usize;
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16_777_216.0) - 0.5
        };
        // Most logits modest, scatter ~5% of them out to ±50 to
        // exercise the LogSumExp stability path.
        let logits: Vec<f32> = (0..vocab)
            .map(|i| {
                let n = next();
                if i % 19 == 0 { n * 100.0 } else { n * 4.0 }
            })
            .collect();
        let target: u32 = 137;

        let mut cpu_grad = vec![0.0f32; vocab];
        let cpu_loss =
            crate::reference::ops::cross_entropy_backward(&logits, target, &mut cpu_grad);

        let (gpu_grad, gpu_loss) =
            pollster::block_on(cross_entropy_backward_cached(&ctx, &p, &logits, target))
                .expect("gpu");

        // Loss tolerance loosened because LogSumExp at ±50 input range
        // produces larger absolute round-off than at ±4 (the synthetic
        // test's range). Bug-level failures would be orders bigger.
        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-2,
            "wide-magnitude loss cpu={cpu_loss} gpu={gpu_loss}"
        );
        let mut max_diff = 0.0f32;
        for (c, g) in cpu_grad.iter().zip(gpu_grad.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-4,
            "wide-magnitude d_logits max_diff = {max_diff}"
        );
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
            w_bytes[off] = 0x00;
            w_bytes[off + 1] = 0x2C; // f16(0.0625)
            w_bytes[off + 2] = 0x00;
            w_bytes[off + 3] = 0x28; // f16(0.03125)
        }

        // Deterministic dy.
        let dy: Vec<f32> = (0..n).map(|j| ((j as i32 - 8) as f32) * 0.25).collect();

        // CPU oracle
        let mut cpu_dx = vec![0.0f32; k];
        crate::reference::ops::matmul_q4_k_backward_input(&w_bytes, &dy, k, n, &mut cpu_dx);

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

    /// GPU vs CPU parity for `matmul_q6_k_backward_input`. Same approach
    /// as the Q4_K parity test — synthesize a small Q6_K weight buffer
    /// from a deterministic byte stream with f16 scale field clamped to
    /// a small finite value, then compare GPU vs CPU oracle.
    #[test]
    fn matmul_q6_k_backward_input_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        let k = 256usize;
        let n = 16usize;
        // Q6_K: 210 bytes per 256-element block.
        let block_bytes = 210usize;
        let row_bytes = (k / 256) * block_bytes;
        let total_bytes = n * row_bytes;
        let mut w_bytes = vec![0u8; total_bytes];
        let mut state: u32 = 0xCAFEBABE;
        for b in w_bytes.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 16) as u8;
        }
        // Clamp the super-block scale `d` (last 2 bytes of each 210-byte
        // block) so the dequant stays bounded — random f16 bit patterns
        // can land in NaN/Inf and propagate everywhere.
        for j in 0..n {
            for b in 0..(k / 256) {
                let off = j * row_bytes + b * block_bytes;
                w_bytes[off + 208] = 0x00;
                w_bytes[off + 209] = 0x28; // f16(0.03125) — small, positive, finite
            }
        }

        let dy: Vec<f32> = (0..n).map(|j| ((j as i32 - 8) as f32) * 0.25).collect();

        let mut cpu_dx = vec![0.0f32; k];
        crate::reference::ops::matmul_q6_k_backward_input(&w_bytes, &dy, k, n, &mut cpu_dx);
        let gpu_dx = pollster::block_on(matmul_q6_k_backward_input_cached(
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
            "q6_k_bwd_input max_abs={max_diff} max_rel={max_rel}"
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
        let x_buf = write_storage_f32(device, queue, "x", &x);
        let w_buf = write_storage_f32(device, queue, "w", &w);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dx_buf, dx_read) = make_output_pair(device, "dx", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_bwd.enc"),
        });
        rmsnorm_backward_chained(
            &ctx, &p, &mut enc, &x_buf, &w_buf, &dy_buf, &dx_buf, n, eps, true,
        );
        enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dx = pollster::block_on(read_back_f32(device, &dx_read)).expect("readback");

        let mut max_diff = 0.0f32;
        for (c, g) in cpu_dx.iter().zip(gpu_dx.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff < 1e-4, "rmsnorm_bwd max_diff = {max_diff}");
    }

    /// **D5 — fuzz-magnitude rmsnorm backward.** Production residual
    /// streams can hit hidden-state norms in the ±20 range and weight
    /// scales near 0.05 (small) or 5.0 (large). The variance
    /// reduction inside rmsnorm is the place where extreme magnitudes
    /// produce f32 round-off concentration. Larger tolerance to
    /// reflect the genuine f32 limit at this scale; a real bug would
    /// produce mismatches orders of magnitude larger.
    #[test]
    fn rmsnorm_backward_gpu_vs_cpu_wide_magnitude() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 256usize;
        // Wider, asymmetric range; mix of small and large weights.
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 - 128.0) * 0.15).sin() * 20.0)
            .collect();
        let w: Vec<f32> = (0..n)
            .map(|i| if i % 4 == 0 { 5.0 } else { 0.05 })
            .collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 10.0).collect();
        let eps = 1e-6f32;

        let mut cpu_dx = vec![0.0f32; n];
        crate::reference::ops::rmsnorm_backward(&x, Some(&w), &dy, eps, &mut cpu_dx);

        let device = &ctx.device;
        let queue = &ctx.queue;
        let x_buf = write_storage_f32(device, queue, "x", &x);
        let w_buf = write_storage_f32(device, queue, "w", &w);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dx_buf, dx_read) = make_output_pair(device, "dx", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_bwd_wide.enc"),
        });
        rmsnorm_backward_chained(
            &ctx, &p, &mut enc, &x_buf, &w_buf, &dy_buf, &dx_buf, n, eps, true,
        );
        enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dx = pollster::block_on(read_back_f32(device, &dx_read)).expect("readback");

        let mut max_diff = 0.0f32;
        for (c, g) in cpu_dx.iter().zip(gpu_dx.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 5e-3,
            "rmsnorm_bwd wide-magnitude max_diff = {max_diff}"
        );
    }

    /// GPU vs CPU parity for `rmsnorm_per_row_backward`.
    #[test]
    fn rmsnorm_per_row_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n_rows = 4usize;
        let n = 32usize;
        let total = n_rows * n;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i as i32 - 30) as f32) * 0.05)
            .collect();
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() * 0.3 + 1.0).collect();
        let dy: Vec<f32> = (0..total).map(|i| (i as f32 * 0.7).cos() * 0.5).collect();
        let eps = 1e-6f32;

        let mut cpu_dx = vec![0.0f32; total];
        crate::reference::ops::rmsnorm_per_row_backward(
            &x,
            Some(&w),
            &dy,
            eps,
            n_rows,
            n,
            &mut cpu_dx,
        );

        let device = &ctx.device;
        let queue = &ctx.queue;
        let x_buf = write_storage_f32(device, queue, "x", &x);
        let w_buf = write_storage_f32(device, queue, "w", &w);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dx_buf, dx_read) = make_output_pair(device, "dx", (total * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_pr_bwd.enc"),
        });
        rmsnorm_per_row_backward_chained(
            &ctx, &p, &mut enc, &x_buf, &w_buf, &dy_buf, &dx_buf, n_rows, n, eps, true,
        );
        enc.copy_buffer_to_buffer(&dx_buf, 0, &dx_read, 0, (total * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dx = pollster::block_on(read_back_f32(device, &dx_read)).expect("readback");

        let mut max_diff = 0.0f32;
        for (c, g) in cpu_dx.iter().zip(gpu_dx.iter()) {
            let d = (c - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff < 1e-4, "rmsnorm_per_row_bwd max_diff = {max_diff}");

        // Unweighted variant.
        let mut cpu_dx_u = vec![0.0f32; total];
        crate::reference::ops::rmsnorm_per_row_backward(
            &x,
            None,
            &dy,
            eps,
            n_rows,
            n,
            &mut cpu_dx_u,
        );
        let dummy = make_dummy_storage(device, "dummy");
        let (dx_u_buf, dx_u_read) = make_output_pair(device, "dx_u", (total * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_pr_bwd_u.enc"),
        });
        rmsnorm_per_row_backward_chained(
            &ctx, &p, &mut enc, &x_buf, &dummy, &dy_buf, &dx_u_buf, n_rows, n, eps, false,
        );
        enc.copy_buffer_to_buffer(&dx_u_buf, 0, &dx_u_read, 0, (total * 4) as u64);
        queue.submit(Some(enc.finish()));
        let gpu_dx_u = pollster::block_on(read_back_f32(device, &dx_u_read)).expect("readback");
        let mut max_diff_u = 0.0f32;
        for (c, g) in cpu_dx_u.iter().zip(gpu_dx_u.iter()) {
            let d = (c - g).abs();
            if d > max_diff_u {
                max_diff_u = d;
            }
        }
        assert!(
            max_diff_u < 1e-4,
            "rmsnorm_per_row_bwd unweighted max_diff = {max_diff_u}"
        );
    }

    /// GPU vs CPU parity for the single-workgroup sum-of-squares
    /// reduction. Tests three sizes (under, equal to, and over the
    /// 256-thread workgroup) plus the `scale_in` path.
    #[test]
    fn sum_of_squares_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let device = &ctx.device;
        let queue = &ctx.queue;

        for &n in &[63usize, 256usize, 1024usize, 4097usize] {
            let x: Vec<f32> = (0..n).map(|i| ((i as i32 - 100) as f32) * 0.03).collect();
            let cpu_sos: f32 = x.iter().map(|&v| v * v).sum();

            let x_buf = write_storage_f32(device, queue, "x", &x);
            let (out_buf, out_read) = make_output_pair(device, "sos", 4);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sos.enc"),
            });
            sum_of_squares_chained(&ctx, &p, &mut enc, &x_buf, &out_buf, n, 1.0);
            enc.copy_buffer_to_buffer(&out_buf, 0, &out_read, 0, 4);
            queue.submit(Some(enc.finish()));
            let gpu = pollster::block_on(read_back_f32(device, &out_read)).expect("readback")[0];
            let denom = cpu_sos.abs().max(1e-6);
            let rel = (cpu_sos - gpu).abs() / denom;
            assert!(rel < 1e-4, "n={n} cpu={cpu_sos} gpu={gpu} rel={rel}");
        }

        // scale_in path: scale every input by 0.5 before squaring, so
        // sos becomes 0.25× the unscaled sos.
        let n = 256usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let cpu_sos: f32 = x.iter().map(|&v| (v * 0.5) * (v * 0.5)).sum();
        let x_buf = write_storage_f32(device, queue, "x", &x);
        let (out_buf, out_read) = make_output_pair(device, "sos.scaled", 4);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sos.scaled.enc"),
        });
        sum_of_squares_chained(&ctx, &p, &mut enc, &x_buf, &out_buf, n, 0.5);
        enc.copy_buffer_to_buffer(&out_buf, 0, &out_read, 0, 4);
        queue.submit(Some(enc.finish()));
        let gpu = pollster::block_on(read_back_f32(device, &out_read)).expect("readback")[0];
        let rel = (cpu_sos - gpu).abs() / cpu_sos.abs().max(1e-6);
        assert!(rel < 1e-4, "scaled cpu={cpu_sos} gpu={gpu} rel={rel}");
    }

    /// GPU vs CPU parity for `geglu_backward`.
    #[test]
    fn geglu_backward_gpu_vs_cpu() {
        // **D5 — production-magnitude variant lives below as
        // `geglu_backward_gpu_vs_cpu_wide_magnitude`.** The
        // synthetic-range test here (gate ∈ [-1.5, 1.65]) is a cheap
        // CI gate; the wide variant exercises the regime real
        // checkpoints actually hit (±20-40 gate values), which is
        // what the May 2026 tanh-clamp bug needed to surface.
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 64usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 30.0) * 0.05).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4).sin()).collect();

        let mut cpu_dg = vec![0.0f32; n];
        let mut cpu_du = vec![0.0f32; n];
        crate::reference::ops::geglu_backward(&gate, &up, &dy, &mut cpu_dg, &mut cpu_du);

        let device = &ctx.device;
        let queue = &ctx.queue;
        let g_buf = write_storage_f32(device, queue, "gate", &gate);
        let u_buf = write_storage_f32(device, queue, "up", &up);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dg_buf, dg_read) = make_output_pair(device, "dg", (n * 4) as u64);
        let (du_buf, du_read) = make_output_pair(device, "du", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("geglu_bwd.enc"),
        });
        geglu_backward_chained(
            &ctx, &p, &mut enc, &g_buf, &u_buf, &dy_buf, &dg_buf, &du_buf, n,
        );
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
        assert!(
            max_dg < 1e-5 && max_du < 1e-5,
            "geglu_bwd max_dg={max_dg} max_du={max_du}"
        );
    }

    /// **D5 — fuzz-magnitude geglu backward.** Re-runs the same kernel
    /// vs the CPU oracle with inputs scaled to the empirical range
    /// production checkpoints actually hit (gates reaching ±40 on
    /// gemma4:e2b — that's where the May 2026 tanh-clamp bug surfaced).
    /// The synthetic-range test above runs in CI as a cheap gate; this
    /// one is the regression net for magnitude-dependent kernel bugs.
    /// Bigger tolerance than the synthetic test (5e-4 vs 1e-5) because
    /// at large magnitudes f32 round-off in the tanh-saturated tail is
    /// no longer negligible, but a clamp-missing-style bug would
    /// produce diffs orders of magnitude larger.
    #[test]
    fn geglu_backward_gpu_vs_cpu_wide_magnitude() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 128usize;
        // Gate values that exercise both saturation tails of GELU/tanh.
        let gate: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / (n as f32)) - 0.5) * 80.0)
            .collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() * 10.0).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).cos() * 5.0).collect();

        let mut cpu_dg = vec![0.0f32; n];
        let mut cpu_du = vec![0.0f32; n];
        crate::reference::ops::geglu_backward(&gate, &up, &dy, &mut cpu_dg, &mut cpu_du);

        let device = &ctx.device;
        let queue = &ctx.queue;
        let g_buf = write_storage_f32(device, queue, "gate", &gate);
        let u_buf = write_storage_f32(device, queue, "up", &up);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (dg_buf, dg_read) = make_output_pair(device, "dg", (n * 4) as u64);
        let (du_buf, du_read) = make_output_pair(device, "du", (n * 4) as u64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("geglu_bwd_wide.enc"),
        });
        geglu_backward_chained(
            &ctx, &p, &mut enc, &g_buf, &u_buf, &dy_buf, &dg_buf, &du_buf, n,
        );
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
        // At gate=±40 the GELU/tanh saturation tail makes d_gate large in
        // MAGNITUDE (≈ dy·up·gelu'(gate) ~ O(50) here), so an absolute tolerance
        // is the wrong gauge — f32 rounding there produces absolute diffs ~1e-2
        // that are still only a ~1e-3 RELATIVE error. Use a relative tolerance
        // for d_gate (the up-path d_up stays small, so absolute is fine for it).
        // A real clamp bug would blow the relative error up to O(1) or produce
        // NaN/inf — orders of magnitude past this gate.
        let scale_dg = cpu_dg.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1.0);
        let rel_dg = max_dg / scale_dg;
        assert!(
            rel_dg < 1e-3 && max_du < 5e-4,
            "geglu_bwd wide-magnitude rel_dg={rel_dg} (max_dg={max_dg}, scale={scale_dg}) max_du={max_du}"
        );
    }

    /// GPU vs CPU Adam step. Initializes random params + grads + zeros for
    /// (m, v), runs the kernel once, and compares against the CPU oracle.
    #[test]
    fn adam_step_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);
        let n = 128usize;

        let mut param: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).cos() * 0.1).collect();
        let mut m_cpu = vec![0.0f32; n];
        let mut v_cpu = vec![0.0f32; n];
        let mut param_cpu = param.clone();

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let step = 1u32;

        crate::reference::ops::adam_step(
            &grad,
            &mut param_cpu,
            &mut m_cpu,
            &mut v_cpu,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step,
        );

        // GPU
        let device = &ctx.device;
        let queue = &ctx.queue;
        let grad_buf = write_storage_f32(device, queue, "g", &grad);
        let param_buf = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("param"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&param));
            buf
        };
        let m_init = vec![0.0f32; n];
        let v_init = vec![0.0f32; n];
        let m_buf = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("m"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&m_init));
            buf
        };
        let v_buf = {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("v"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&v_init));
            buf
        };
        let param_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("param.read"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adam.enc"),
        });
        adam_step_chained(
            &ctx,
            &p,
            &mut enc,
            &grad_buf,
            &param_buf,
            &m_buf,
            &v_buf,
            n,
            AdamConfig {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay: wd,
                step,
            },
        );
        enc.copy_buffer_to_buffer(&param_buf, 0, &param_read, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));

        let gpu_param = pollster::block_on(read_back_f32(device, &param_read)).unwrap();
        param = gpu_param;

        let max_diff = param
            .iter()
            .zip(param_cpu.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "adam max_diff = {max_diff}");
    }

    /// GPU vs CPU for the three LoRA primitives composed into a full
    /// forward+backward sequence (matches what `TrainingSession::step`
    /// will eventually run, minus optimizer + activation capture).
    ///
    /// Exercises `lora_matmul_row` × 2 (forward correction's z=A·x and
    /// y+=s·B·z), `lora_matmul_col` × 2 (u=Bᵀ·dy and dx+=s·Aᵀ·u), and
    /// `lora_outer_add` × 2 (dA and dB) — all six call sites that
    /// `TrainingSession::step` will use per LoRA-wrapped projection.
    #[test]
    fn lora_forward_backward_gpu_vs_cpu() {
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let p = Pipelines::new(&ctx.device);

        let k = 16usize;
        let r = 4usize;
        let n = 12usize;
        let scale = 0.5f32;
        let a: Vec<f32> = (0..r * k).map(|i| (i as f32 * 0.17).sin() * 0.4).collect();
        let b: Vec<f32> = (0..n * r).map(|i| (i as f32 * 0.29).cos() * 0.3).collect();
        let x: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.31).sin() * 0.5 + 0.1)
            .collect();
        let dy: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.47).cos() * 0.3 + 0.2)
            .collect();

        // CPU reference path
        let mut z_cpu = vec![0f32; r];
        crate::reference::ops::lora_matmul_row(&a, &x, &mut z_cpu, k, r, 1.0, false);
        let mut y_cpu = vec![0f32; n];
        crate::reference::ops::lora_matmul_row(&b, &z_cpu, &mut y_cpu, r, n, scale, false);
        let mut u_cpu = vec![0f32; r];
        crate::reference::ops::lora_matmul_col(&b, &dy, &mut u_cpu, n, r, 1.0, false);
        let mut da_cpu = vec![0f32; r * k];
        crate::reference::ops::lora_outer_add(&u_cpu, &x, &mut da_cpu, scale, false);
        let mut db_cpu = vec![0f32; n * r];
        crate::reference::ops::lora_outer_add(&dy, &z_cpu, &mut db_cpu, scale, false);
        let mut dx_cpu = vec![0f32; k];
        crate::reference::ops::lora_matmul_col(&a, &u_cpu, &mut dx_cpu, r, k, scale, false);

        // GPU path — all dispatches in one encoder.
        let device = &ctx.device;
        let queue = &ctx.queue;
        let a_buf = write_storage_f32(device, queue, "A", &a);
        let b_buf = write_storage_f32(device, queue, "B", &b);
        let x_buf = write_storage_f32(device, queue, "x", &x);
        let dy_buf = write_storage_f32(device, queue, "dy", &dy);
        let (z_buf, z_read) = make_output_pair(device, "z", (r * 4) as u64);
        let (y_buf, y_read) = make_output_pair(device, "y", (n * 4) as u64);
        let (u_buf, u_read) = make_output_pair(device, "u", (r * 4) as u64);
        let (da_buf, da_read) = make_output_pair(device, "dA", (r * k * 4) as u64);
        let (db_buf, db_read) = make_output_pair(device, "dB", (n * r * 4) as u64);
        let (dx_buf, dx_read) = make_output_pair(device, "dx", (k * 4) as u64);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lora_fb.enc"),
        });
        // Forward: z = A @ x, y = scale · B @ z
        lora_matmul_row_chained(&ctx, &p, &mut enc, &a_buf, &x_buf, &z_buf, k, r, 1.0, false);
        lora_matmul_row_chained(
            &ctx, &p, &mut enc, &b_buf, &z_buf, &y_buf, r, n, scale, false,
        );
        // Backward: u = Bᵀ @ dy
        lora_matmul_col_chained(
            &ctx, &p, &mut enc, &b_buf, &dy_buf, &u_buf, n, r, 1.0, false,
        );
        // dA = scale · outer(u, x), dB = scale · outer(dy, z)
        lora_outer_add_chained(
            &ctx, &p, &mut enc, &u_buf, &x_buf, &da_buf, r, k, scale, false,
        );
        lora_outer_add_chained(
            &ctx, &p, &mut enc, &dy_buf, &z_buf, &db_buf, n, r, scale, false,
        );
        // dx = scale · Aᵀ @ u
        lora_matmul_col_chained(
            &ctx, &p, &mut enc, &a_buf, &u_buf, &dx_buf, r, k, scale, false,
        );

        for (src, sz, dst) in [
            (&z_buf, (r * 4) as u64, &z_read),
            (&y_buf, (n * 4) as u64, &y_read),
            (&u_buf, (r * 4) as u64, &u_read),
            (&da_buf, (r * k * 4) as u64, &da_read),
            (&db_buf, (n * r * 4) as u64, &db_read),
            (&dx_buf, (k * 4) as u64, &dx_read),
        ] {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, sz);
        }
        queue.submit(Some(enc.finish()));

        let z_gpu = pollster::block_on(read_back_f32(device, &z_read)).unwrap();
        let y_gpu = pollster::block_on(read_back_f32(device, &y_read)).unwrap();
        let u_gpu = pollster::block_on(read_back_f32(device, &u_read)).unwrap();
        let da_gpu = pollster::block_on(read_back_f32(device, &da_read)).unwrap();
        let db_gpu = pollster::block_on(read_back_f32(device, &db_read)).unwrap();
        let dx_gpu = pollster::block_on(read_back_f32(device, &dx_read)).unwrap();

        let max = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(max(&z_cpu, &z_gpu) < 1e-5);
        assert!(max(&y_cpu, &y_gpu) < 1e-5);
        assert!(max(&u_cpu, &u_gpu) < 1e-5);
        assert!(max(&da_cpu, &da_gpu) < 1e-5);
        assert!(max(&db_cpu, &db_gpu) < 1e-5);
        assert!(max(&dx_cpu, &dx_gpu) < 1e-5);
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
        let q: Vec<f32> = (0..q_len).map(|i| (i as f32 * 0.31).sin() * 0.4).collect();
        let k_hist: Vec<f32> = (0..kv_len).map(|i| (i as f32 * 0.17).cos() * 0.3).collect();
        let v_hist: Vec<f32> = (0..kv_len).map(|i| (i as f32 * 0.23).sin() * 0.5).collect();
        let d_out: Vec<f32> = (0..q_len)
            .map(|i| (i as f32 * 0.47).cos() * 0.3 + 0.1)
            .collect();

        // Forward (CPU) — gives us probs to feed back into both backwards.
        let mut out_unused = vec![0f32; q_len];
        let mut probs = vec![0f32; n_heads * history_len];
        crate::reference::ops::attention_forward(
            &q,
            &k_hist,
            &v_hist,
            &mut out_unused,
            &mut probs,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len,
        );

        // CPU backward
        let mut cpu_dq = vec![0f32; q_len];
        let mut cpu_dk = vec![0f32; kv_len];
        let mut cpu_dv = vec![0f32; kv_len];
        crate::reference::ops::attention_backward(
            &q,
            &k_hist,
            &v_hist,
            &probs,
            &d_out,
            &mut cpu_dq,
            &mut cpu_dk,
            &mut cpu_dv,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len,
        );

        // GPU backward — two passes.
        let device = &ctx.device;
        let queue = &ctx.queue;
        let q_buf = write_storage_f32(device, queue, "q", &q);
        let k_buf = write_storage_f32(device, queue, "k_hist", &k_hist);
        let v_buf = write_storage_f32(device, queue, "v_hist", &v_hist);
        let probs_buf = write_storage_f32(device, queue, "probs", &probs);
        let dout_buf = write_storage_f32(device, queue, "d_out", &d_out);
        let (ds_buf, _) = make_output_pair(device, "d_scores", (n_heads * history_len * 4) as u64);
        let (dq_buf, dq_read) = make_output_pair(device, "d_q", (q_len * 4) as u64);
        let (dk_buf, dk_read) = make_output_pair(device, "d_k_hist", (kv_len * 4) as u64);
        let (dv_buf, dv_read) = make_output_pair(device, "d_v_hist", (kv_len * 4) as u64);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_bwd.enc"),
        });
        attention_backward_dq_chained(
            &ctx,
            &p,
            &mut enc,
            &k_buf,
            &v_buf,
            &probs_buf,
            &dout_buf,
            &ds_buf,
            &dq_buf,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len,
        );
        attention_backward_dkv_chained(
            &ctx,
            &p,
            &mut enc,
            &q_buf,
            &probs_buf,
            &dout_buf,
            &ds_buf,
            &dk_buf,
            &dv_buf,
            head_dim,
            n_heads,
            n_kv_heads,
            history_len,
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
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
        rope_neox_chained(
            &ctx, &p, &mut enc, &x_buf, None, &dummy, head_dim, n_heads, pos, rope_dims, base,
        );
        rope_neox_backward_chained(
            &ctx, &p, &mut enc, &x_buf, None, &dummy, head_dim, n_heads, pos, rope_dims, base,
        );
        enc.copy_buffer_to_buffer(&x_buf, 0, &read_buf, 0, (total * 4) as u64);
        queue.submit(Some(enc.finish()));
        let out = pollster::block_on(read_back_f32(device, &read_buf)).expect("readback");

        let mut max_drift = 0.0f32;
        for (o, n) in orig.iter().zip(out.iter()) {
            let d = (o - n).abs();
            if d > max_drift {
                max_drift = d;
            }
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
