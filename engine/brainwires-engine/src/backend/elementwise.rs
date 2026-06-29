// Elementwise dispatchers (RMSNorm, softcap, GeGLU, RoPE) each carry the
// kernel's WGSL uniform layout in their signature; bundling them buys
// nothing here.
#![allow(clippy::too_many_arguments)]
// CPU oracle math (softmax, masking, repetition penalty) walks parallel
// index spaces — `for i in 0..n` is clearer than zipped iterators here.
#![allow(clippy::needless_range_loop)]

//! Elementwise / per-row WGSL kernels: RMSNorm, softcap, GeGLU, RoPE.
//!
//! Same single-shot dispatch pattern as `matmul.rs` — these don't cache pipelines or
//! buffers across calls. Cached versions land in M3's full-forward integration.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};
use crate::kernels;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RmsNormParams {
    n: u32,
    eps: f32,
    has_weight: u32,
    _pad: u32,
}

/// `y = x / sqrt(mean(x²) + eps) * (w | 1)`. If `weight` is `None`, the kernel
/// multiplies by 1.0 (unweighted RMSNorm — matches Gemma 4's V-norm).
pub async fn rmsnorm(
    ctx: &WgpuCtx,
    x: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if let Some(w) = weight
        && w.len() != n
    {
        return Err(RullamaError::Inference(format!(
            "rmsnorm weight len {} != x len {}",
            w.len(),
            n
        )));
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    let params = RmsNormParams {
        n: n as u32,
        eps,
        has_weight: if weight.is_some() { 1 } else { 0 },
        _pad: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.params"),
        size: std::mem::size_of::<RmsNormParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let bytes_n = (n * 4) as u64;

    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.x"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));

    // Always allocate a w buffer so the bind group has something to bind, even when
    // the kernel ignores it (has_weight=0).
    let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.w"),
        size: bytes_n.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if let Some(w) = weight {
        queue.write_buffer(&w_buf, 0, bytemuck::cast_slice(w));
    }

    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.y"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm.read"),
        size: bytes_n,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rmsnorm.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::RMSNORM)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rmsnorm.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmsnorm.bg"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rmsnorm.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // One workgroup handles the whole row.
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, bytes_n);
    queue.submit(Some(encoder.finish()));

    let slice = read_buf.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
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
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    Ok(out)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct CapParams {
    n: u32,
    cap: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Logit softcap: `y = cap * tanh(x / cap)`. If `cap <= 0` the kernel is identity.
pub async fn softcap(ctx: &WgpuCtx, x: &[f32], cap: f32) -> Result<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = CapParams {
        n: n as u32,
        cap,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softcap.params"),
        size: std::mem::size_of::<CapParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
    let bytes_n = (n * 4) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softcap.x"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softcap.y"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softcap.read"),
        size: bytes_n,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("softcap.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::SOFTCAP)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("softcap.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softcap.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("softcap.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softcap.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, bytes_n);
    queue.submit(Some(encoder.finish()));
    readback(device, &read_buf).await
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GegluParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `y = gelu(gate) * up`. Erf-based GELU matches the CPU reference exactly.
pub async fn geglu(ctx: &WgpuCtx, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(RullamaError::Inference(format!(
            "geglu: gate len {} != up len {}",
            gate.len(),
            up.len()
        )));
    }
    let n = gate.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;
    let params = GegluParams {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geglu.params"),
        size: std::mem::size_of::<GegluParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
    let bytes_n = (n * 4) as u64;
    let gate_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geglu.gate"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&gate_buf, 0, bytemuck::cast_slice(gate));
    let up_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geglu.up"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&up_buf, 0, bytemuck::cast_slice(up));
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geglu.y"),
        size: bytes_n,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geglu.read"),
        size: bytes_n,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("geglu.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::GEGLU)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("geglu.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("geglu.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("geglu.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("geglu.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, bytes_n);
    queue.submit(Some(encoder.finish()));
    readback(device, &read_buf).await
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
    _pad0: u32,
    _pad1: u32,
}

/// In-place NeoX RoPE on a `[head_dim, n_heads]` flattened tensor. Returns the rotated
/// tensor as a fresh `Vec<f32>` (the kernel writes back to the storage buffer; we
/// readback the modified vector).
pub async fn rope_neox(
    ctx: &WgpuCtx,
    x: &[f32],
    head_dim: usize,
    n_heads: usize,
    pos: usize,
    rope_dims: usize,
    base: f32,
    factors: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if x.len() != head_dim * n_heads {
        return Err(RullamaError::Inference(format!(
            "rope: x.len() {} != head_dim*n_heads {}",
            x.len(),
            head_dim * n_heads
        )));
    }
    if rope_dims > head_dim || !rope_dims.is_multiple_of(2) {
        return Err(RullamaError::Inference(format!(
            "rope: rope_dims={rope_dims} must be even and ≤ head_dim={head_dim}"
        )));
    }
    if let Some(f) = factors
        && f.len() != rope_dims / 2
    {
        return Err(RullamaError::Inference(format!(
            "rope: factors.len() {} != rope_dims/2 {}",
            f.len(),
            rope_dims / 2
        )));
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
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.params"),
        size: std::mem::size_of::<RopeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let bytes_x = (x.len() * 4) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.x"),
        size: bytes_x,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));

    let factors_bytes = (rope_dims / 2 * 4).max(4) as u64;
    let factors_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.factors"),
        size: factors_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if let Some(f) = factors {
        queue.write_buffer(&factors_buf, 0, bytemuck::cast_slice(f));
    }

    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope.read"),
        size: bytes_x,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rope.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::ROPE_NEOX)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rope.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rope.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(total.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&x_buf, 0, &read_buf, 0, bytes_x);
    queue.submit(Some(encoder.finish()));
    readback(device, &read_buf).await
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
    _pad: u32,
}

/// Multi-head softmax attention over a KV history. Returns `[n_heads * head_dim]`.
///
/// `q` has shape `[n_heads, head_dim]`, `k_hist`/`v_hist` have shape
/// `[history_len, n_kv_heads, head_dim]`. `window=0` selects global causal attention;
/// `window>0` enables sliding-window masking on top of causal.
pub async fn attention(
    ctx: &WgpuCtx,
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
        return Err(RullamaError::Inference(format!(
            "attn: q.len()={} != {}",
            q.len(),
            n_heads * head_dim
        )));
    }
    if k_hist.len() != history_len * n_kv_heads * head_dim
        || v_hist.len() != history_len * n_kv_heads * head_dim
    {
        return Err(RullamaError::Inference(format!(
            "attn: kv shape mismatch (history_len={history_len}, kvh={n_kv_heads}, hd={head_dim})"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(RullamaError::Inference(format!(
            "attn: n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
        )));
    }
    if history_len > 1024 {
        return Err(RullamaError::Inference(format!(
            "attn: history_len {history_len} > MAX_HISTORY=1024"
        )));
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
        _pad: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.params"),
        size: std::mem::size_of::<AttnParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let q_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.q"),
        size: (q.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&q_buf, 0, bytemuck::cast_slice(q));
    let k_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.k"),
        size: (k_hist.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&k_buf, 0, bytemuck::cast_slice(k_hist));
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.v"),
        size: (v_hist.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(v_hist));

    let out_bytes = (n_heads * head_dim * 4) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn.read"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attn.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::ATTENTION)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("attn.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(n_heads as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    queue.submit(Some(encoder.finish()));
    readback(device, &read_buf).await
}

async fn readback(device: &wgpu::Device, read_buf: &wgpu::Buffer) -> Result<Vec<f32>> {
    let slice = read_buf.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
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
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    Ok(out)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::reference::ops::rmsnorm as cpu_rmsnorm;

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 16777216.0) - 0.5
            })
            .collect()
    }

    fn check(cpu: &[f32], gpu: &[f32], tol_abs: f32) {
        let mut max_abs = 0f32;
        for i in 0..cpu.len() {
            let d = (cpu[i] - gpu[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!("rmsnorm diff max_abs={max_abs:e} (n={})", cpu.len());
        assert!(max_abs < tol_abs, "max_abs {max_abs} >= {tol_abs}");
    }

    #[test]
    fn rmsnorm_unweighted_n256() {
        let x = rand_vec(256, 0xAAAA_5555);
        let mut cpu = vec![0f32; 256];
        cpu_rmsnorm(&x, None, 1e-6, &mut cpu);
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu = pollster::block_on(rmsnorm(&ctx, &x, None, 1e-6)).unwrap();
        check(&cpu, &gpu, 1e-5);
    }

    #[test]
    fn rmsnorm_weighted_n1536() {
        let x = rand_vec(1536, 0xCAFE_BEEF);
        let w = rand_vec(1536, 0xBEEF_CAFE);
        let mut cpu = vec![0f32; 1536];
        cpu_rmsnorm(&x, Some(&w), 1e-6, &mut cpu);
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu = pollster::block_on(rmsnorm(&ctx, &x, Some(&w), 1e-6)).unwrap();
        check(&cpu, &gpu, 1e-5);
    }

    #[test]
    fn softcap_matches_cpu() {
        let x = rand_vec(4096, 0xFEED_F00D);
        let cap = 30.0;
        let mut cpu_y = x.clone();
        crate::reference::ops::softcap(&mut cpu_y, cap);
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu_y = pollster::block_on(softcap(&ctx, &x, cap)).unwrap();
        check(&cpu_y, &gpu_y, 1e-5);
    }

    #[test]
    fn geglu_matches_cpu() {
        let n = 6144;
        let gate = rand_vec(n, 0x5A5A_3C3C);
        let up = rand_vec(n, 0x33CC_99FF);
        let mut cpu_y = vec![0f32; n];
        crate::reference::ops::geglu_split(&gate, &up, &mut cpu_y);
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu_y = pollster::block_on(geglu(&ctx, &gate, &up)).unwrap();
        // The erf approx is the same A&S 7.1.26 polynomial on both sides; the only
        // diff is internal f32 ordering. Tight tolerance.
        check(&cpu_y, &gpu_y, 1e-5);
    }

    /// Reference attention computation matching the M1 CPU forward.
    fn cpu_attention(
        q: &[f32],
        k_hist: &[f32],
        v_hist: &[f32],
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        pos: usize,
        history_len: usize,
        window: usize,
    ) -> Vec<f32> {
        let heads_per_kv = n_heads / n_kv_heads;
        let earliest: usize = if window == 0 {
            0
        } else {
            (pos + 1).saturating_sub(window)
        };

        let mut out = vec![0f32; n_heads * head_dim];
        let mut scores = vec![0f32; history_len];
        for qh in 0..n_heads {
            let kvh = qh / heads_per_kv;
            let q_off = qh * head_dim;
            for t in 0..history_len {
                if t < earliest || t > pos {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (t * n_kv_heads + kvh) * head_dim;
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += q[q_off + d] * k_hist[k_off + d];
                }
                scores[t] = s;
            }
            // softmax
            let mut maxv = f32::NEG_INFINITY;
            for &s in &scores {
                if s > maxv {
                    maxv = s;
                }
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = if *s == f32::NEG_INFINITY {
                    0.0
                } else {
                    (*s - maxv).exp()
                };
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in scores.iter_mut() {
                *s *= inv;
            }
            // weighted V
            let out_off = qh * head_dim;
            for d in 0..head_dim {
                out[out_off + d] = 0.0;
            }
            for t in 0..history_len {
                let w = scores[t];
                if w == 0.0 {
                    continue;
                }
                let v_off = (t * n_kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] += w * v_hist[v_off + d];
                }
            }
        }
        out
    }

    #[test]
    fn attention_global_history_3() {
        let head_dim = 256;
        let n_heads = 8;
        let n_kv_heads = 1;
        let history_len = 3;
        let pos = 2;
        let window = 0; // global

        let q = rand_vec(n_heads * head_dim, 0xA1A1_B2B2);
        let k = rand_vec(history_len * n_kv_heads * head_dim, 0xC3C3_D4D4);
        let v = rand_vec(history_len * n_kv_heads * head_dim, 0xE5E5_F6F6);

        let cpu = cpu_attention(
            &q,
            &k,
            &v,
            head_dim,
            n_heads,
            n_kv_heads,
            pos,
            history_len,
            window,
        );
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu = pollster::block_on(attention(
            &ctx,
            &q,
            &k,
            &v,
            head_dim,
            n_heads,
            n_kv_heads,
            pos,
            history_len,
            window,
        ))
        .unwrap();
        check(&cpu, &gpu, 1e-4);
    }

    #[test]
    fn attention_swa_window_clamps_history() {
        // window=2 with history_len=5, pos=4 → only positions [3, 4] are visible.
        let head_dim = 256;
        let n_heads = 8;
        let n_kv_heads = 1;
        let history_len = 5;
        let pos = 4;
        let window = 2;

        let q = rand_vec(n_heads * head_dim, 0x1010_2020);
        let k = rand_vec(history_len * n_kv_heads * head_dim, 0x3030_4040);
        let v = rand_vec(history_len * n_kv_heads * head_dim, 0x5050_6060);

        let cpu = cpu_attention(
            &q,
            &k,
            &v,
            head_dim,
            n_heads,
            n_kv_heads,
            pos,
            history_len,
            window,
        );
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu = pollster::block_on(attention(
            &ctx,
            &q,
            &k,
            &v,
            head_dim,
            n_heads,
            n_kv_heads,
            pos,
            history_len,
            window,
        ))
        .unwrap();
        check(&cpu, &gpu, 1e-4);
    }

    #[test]
    fn rope_swa_full_rotation_pos8() {
        // SWA case: full rotation (rope_dims = head_dim), base = 10000, no factors.
        let head_dim = 256;
        let n_heads = 8;
        let pos = 8;
        let base = 10_000.0_f32;
        let x = rand_vec(head_dim * n_heads, 0xABBA_5050);

        let mut cpu_x = x.clone();
        crate::reference::ops::rope_neox(&mut cpu_x, head_dim, n_heads, pos, head_dim, base, None);

        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu_x = pollster::block_on(rope_neox(
            &ctx, &x, head_dim, n_heads, pos, head_dim, base, None,
        ))
        .unwrap();
        check(&cpu_x, &gpu_x, 1e-5);
    }

    #[test]
    fn rope_global_with_factors_pos1024() {
        // Global case: rope_dims = head_dim_global = 512, base = 1e6,
        // freq_factors = a real-shape vector; first 25% are 1.0, rest are 1e30 → no rotation.
        let head_dim = 512;
        let n_heads = 8;
        let pos = 1024;
        let base = 1_000_000.0_f32;
        let x = rand_vec(head_dim * n_heads, 0x1357_2468);

        let half = head_dim / 2;
        let rotated_pairs = head_dim / 4; // first 25% of head_dim, but in pairs
        let mut factors = vec![1.0_f32; half];
        for i in rotated_pairs..half {
            factors[i] = 1e30;
        }

        let mut cpu_x = x.clone();
        crate::reference::ops::rope_neox(
            &mut cpu_x,
            head_dim,
            n_heads,
            pos,
            head_dim,
            base,
            Some(&factors),
        );

        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu_x = pollster::block_on(rope_neox(
            &ctx,
            &x,
            head_dim,
            n_heads,
            pos,
            head_dim,
            base,
            Some(&factors),
        ))
        .unwrap();
        // Looser tolerance than other kernels: at pos=1024, base=1e6, GPU `pow()` and
        // `cos/sin` precision drift gives diffs around 3-4e-5. Real-magnitude inputs
        // are scaled vectors, not raw samples in [-0.5, 0.5], so the absolute drift
        // on actual hidden states is similar — a few tens of microvolts is fine.
        check(&cpu_x, &gpu_x, 1e-4);
    }

    /// CPU port of `audio.rs::AudioForward::forward_attention`'s inner loop:
    /// takes already-prepared (Q per-dim-scaled, K k-scaled) Q/K/V/pos_proj
    /// and computes `attn_out` exactly as the GPU kernel should.
    #[allow(clippy::too_many_arguments)]
    fn cpu_block_local_attention(
        q_pad: &[f32],
        k_padded: &[f32],
        v_padded: &[f32],
        pos_proj: &[f32],
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
    ) -> Vec<f32> {
        let mut out = vec![0f32; padded_len * hidden];
        let num_chunks = padded_len / chunk_size;
        for u in 0..num_chunks {
            for r in 0..chunk_size {
                for h in 0..n_heads {
                    let q_off = (u * chunk_size + r) * hidden + h * head_dim;
                    let mut logits = vec![f32::NEG_INFINITY; context_size];
                    let mut max_logit = f32::NEG_INFINITY;
                    for c in 0..context_size {
                        let actual_t = (u * chunk_size) as i64 + c as i64 - pad_left as i64;
                        let valid = actual_t >= 0 && actual_t < seq as i64;
                        let causal = c >= r && c <= r + max_past + max_future;
                        if !valid || !causal {
                            continue;
                        }
                        let k_off = (u * chunk_size + c) * hidden + h * head_dim;
                        let mut ac = 0f32;
                        for d in 0..head_dim {
                            ac += q_pad[q_off + d] * k_padded[k_off + d];
                        }
                        let p_signed = max_past as i64 + r as i64 - c as i64;
                        let bd = if p_signed >= 0 && (p_signed as usize) < max_span {
                            let pos_off = p_signed as usize * hidden + h * head_dim;
                            let mut bd = 0f32;
                            for d in 0..head_dim {
                                bd += q_pad[q_off + d] * pos_proj[pos_off + d];
                            }
                            bd
                        } else {
                            0.0
                        };
                        let mut score = ac + bd;
                        score = (score / logit_cap).tanh() * logit_cap;
                        logits[c] = score;
                        if score > max_logit {
                            max_logit = score;
                        }
                    }
                    let mut sum_exp = 0f32;
                    for c in 0..context_size {
                        if logits[c] == f32::NEG_INFINITY {
                            logits[c] = 0.0;
                            continue;
                        }
                        let e = (logits[c] - max_logit).exp();
                        logits[c] = e;
                        sum_exp += e;
                    }
                    let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    let out_off = (u * chunk_size + r) * hidden + h * head_dim;
                    for d in 0..head_dim {
                        let mut acc = 0f32;
                        for c in 0..context_size {
                            if logits[c] == 0.0 {
                                continue;
                            }
                            let weight = logits[c] * inv;
                            let v_off = (u * chunk_size + c) * hidden + h * head_dim;
                            acc += weight * v_padded[v_off + d];
                        }
                        out[out_off + d] = acc;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn block_local_attention_matches_cpu_oracle() {
        // Realistic Gemma 4 audio config: hidden=1024, 8 heads × 128 dim.
        let hidden = 1024;
        let n_heads = 8;
        let head_dim = 128;
        let chunk_size = 12;
        let max_past = 12;
        let max_future = 0;
        let context_size = max_past + chunk_size + max_future; // 24
        let max_span = max_past + max_future + 1; // 13
        let pad_left = max_past; // 12
        let pad_right = max_future + chunk_size - 1; // 11
        let seq: usize = 25; // ~1 s of audio
        let num_chunks = seq.div_ceil(chunk_size);
        let padded_len = num_chunks * chunk_size; // 36
        let k_padded_len = pad_left + padded_len + pad_right; // 59
        let logit_cap = 50.0f32;

        let q_pad = rand_vec(padded_len * hidden, 0xC0DE_F00D);
        let k_inner = rand_vec(padded_len * hidden, 0xDEAD_BEEF);
        let v_inner = rand_vec(padded_len * hidden, 0xFEED_FACE);
        let pos_proj = rand_vec(max_span * hidden, 0xCAFE_BABE);

        // Pad K/V to k_padded_len with zeros on left/right.
        let mut k_padded = vec![0f32; k_padded_len * hidden];
        let mut v_padded = vec![0f32; k_padded_len * hidden];
        k_padded[pad_left * hidden..(pad_left + padded_len) * hidden].copy_from_slice(&k_inner);
        v_padded[pad_left * hidden..(pad_left + padded_len) * hidden].copy_from_slice(&v_inner);

        // CPU reference.
        let cpu = cpu_block_local_attention(
            &q_pad,
            &k_padded,
            &v_padded,
            &pos_proj,
            seq,
            padded_len,
            hidden,
            n_heads,
            head_dim,
            chunk_size,
            context_size,
            max_span,
            max_past,
            max_future,
            pad_left,
            logit_cap,
        );

        // GPU dispatch via the chained pipeline.
        let ctx = pollster::block_on(crate::backend::WgpuCtx::new()).unwrap();
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let q_buf = upload_storage(&ctx, "test.q", &q_pad);
        let k_buf = upload_storage(&ctx, "test.k", &k_padded);
        let v_buf = upload_storage(&ctx, "test.v", &v_padded);
        let pp_buf = upload_storage(&ctx, "test.pp", &pos_proj);
        let out_size = (padded_len * hidden * 4) as u64;
        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.read"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test.enc"),
            });
        crate::backend::dispatch::block_local_attention_chained(
            &ctx,
            &pipes,
            &mut enc,
            &q_buf,
            &k_buf,
            &v_buf,
            &pp_buf,
            &out_buf,
            seq,
            padded_len,
            hidden,
            n_heads,
            head_dim,
            chunk_size,
            context_size,
            max_span,
            max_past,
            max_future,
            pad_left,
            logit_cap,
        );
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
        ctx.queue.submit(Some(enc.finish()));
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();

        // Compare. Block-local attention with softcap accumulates lots of
        // f32 ops; allow a generous tolerance but it should be well within.
        check(&cpu, &gpu, 1e-3);
    }

    fn upload_storage(ctx: &crate::backend::WgpuCtx, label: &str, data: &[f32]) -> wgpu::Buffer {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    #[test]
    fn rmsnorm_real_attn_norm_layer0() {
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available");
            return;
        }
        // Header-only reader + per-tensor file read: reading the whole 7.16 GB
        // blob just to dequant one 1536-float norm tensor used to peak this
        // test at 6.2 GB resident — enough to OOM the suite on a 16 GB machine
        // once the other GPU tests' driver allocations share the process.
        let r = crate::gguf::tensor::reader_from_file_header(path).unwrap();
        let desc = r.tensor("blk.0.attn_norm.weight").unwrap().clone();
        assert_eq!(
            desc.dtype,
            crate::gguf::GgmlDtype::F32,
            "norm weight is F32"
        );
        let raw = crate::gguf::tensor::read_tensor_raw(path, &r, "blk.0.attn_norm.weight").unwrap();
        let mut w = vec![0f32; desc.elem_count() as usize];
        crate::gguf::quant::dequant_into_f32(desc.dtype, &raw, &mut w).unwrap();

        let x = rand_vec(w.len(), 0xFEEDFACE);
        let mut cpu = vec![0f32; w.len()];
        cpu_rmsnorm(&x, Some(&w), 1e-6, &mut cpu);
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let gpu = pollster::block_on(rmsnorm(&ctx, &x, Some(&w), 1e-6)).unwrap();
        check(&cpu, &gpu, 1e-3);
    }
}
