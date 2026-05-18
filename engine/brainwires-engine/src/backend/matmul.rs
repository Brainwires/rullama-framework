//! GPU matmul dispatchers, paired with WGSL kernels under `crate::kernels`.
//!
//! Each function takes the weight bytes as they appear in GGUF, an input vector `x`,
//! the matmul shape `(k, n)`, and returns `y = x @ W` of length `n` as a `Vec<f32>`.
//! Designed for correctness and parity testing, not throughput — every call creates
//! buffers + pipeline. M3 will introduce a cached pipeline + persistent buffers.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};
use crate::kernels;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct MatmulParams {
    k: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Run `y = x @ W` on the GPU where `W` is stored as BF16 row-major bytes
/// (length `k * n * 2`). BF16 = the upper 16 bits of an F32; the kernel
/// reconstructs each element as `bitcast<f32>(u32(bf16) << 16)`.
pub async fn matmul_bf16(
    ctx: &WgpuCtx,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if w_bytes.len() != k * n * 2 {
        return Err(RullamaError::Inference(format!(
            "bf16 W bytes {} != k*n*2 = {}",
            w_bytes.len(),
            k * n * 2
        )));
    }
    if x.len() != k {
        return Err(RullamaError::Inference(format!(
            "x.len() {} != k {}",
            x.len(),
            k
        )));
    }
    if !k.is_multiple_of(2) {
        return Err(RullamaError::Inference(format!(
            "k {k} must be even for bf16 matmul"
        )));
    }
    dispatch_matmul(ctx, kernels::BF16_MATMUL, w_bytes, x, k, n).await
}

/// Run `y = x @ W` on the GPU where `W` is stored as F16 row-major bytes (length
/// `k * n * 2`).
pub async fn matmul_f16(
    ctx: &WgpuCtx,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if w_bytes.len() != k * n * 2 {
        return Err(RullamaError::Inference(format!(
            "f16 W bytes {} != k*n*2 = {}",
            w_bytes.len(),
            k * n * 2
        )));
    }
    if x.len() != k {
        return Err(RullamaError::Inference(format!(
            "x.len() {} != k {}",
            x.len(),
            k
        )));
    }
    if !k.is_multiple_of(2) {
        return Err(RullamaError::Inference(format!(
            "k {k} must be even for f16 matmul"
        )));
    }

    dispatch_matmul(ctx, kernels::F16_MATMUL, w_bytes, x, k, n).await
}

/// Run `y = x @ W` on the GPU where `W` is stored as Q4_K-packed row-major bytes.
/// Each row has `k/256` super-blocks of 144 bytes.
pub async fn matmul_q4_k(
    ctx: &WgpuCtx,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if !k.is_multiple_of(256) {
        return Err(RullamaError::Inference(format!(
            "k {k} must be a multiple of 256 for Q4_K matmul"
        )));
    }
    let row_bytes = (k / 256) * 144;
    let expected = row_bytes * n;
    if w_bytes.len() != expected {
        return Err(RullamaError::Inference(format!(
            "Q4_K W bytes {} != (k/256)*144*n = {}",
            w_bytes.len(),
            expected
        )));
    }
    if x.len() != k {
        return Err(RullamaError::Inference(format!(
            "x.len() {} != k {}",
            x.len(),
            k
        )));
    }
    if !row_bytes.is_multiple_of(4) {
        return Err(RullamaError::Inference(format!(
            "Q4_K row_bytes {row_bytes} not multiple of 4 (k={k})"
        )));
    }
    dispatch_matmul(ctx, kernels::Q4_K_DEQUANT_MATMUL, w_bytes, x, k, n).await
}

/// Run `y = x @ W` on the GPU where `W` is stored as Q6_K-packed row-major bytes.
/// Each row has `k/256` super-blocks of 210 bytes, so total weight bytes = `(k/256)*210*n`.
pub async fn matmul_q6_k(
    ctx: &WgpuCtx,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if !k.is_multiple_of(256) {
        return Err(RullamaError::Inference(format!(
            "k {k} must be a multiple of 256 for Q6_K matmul"
        )));
    }
    let row_bytes = (k / 256) * 210;
    let expected = row_bytes * n;
    if w_bytes.len() != expected {
        return Err(RullamaError::Inference(format!(
            "Q6_K W bytes {} != (k/256)*210*n = {}",
            w_bytes.len(),
            expected
        )));
    }
    if x.len() != k {
        return Err(RullamaError::Inference(format!(
            "x.len() {} != k {}",
            x.len(),
            k
        )));
    }
    // Each row's byte length must be a multiple of 4 for u32-indexed weight storage.
    // For k=1536 → 1260 bytes/row (÷4 = 315), k=6144 → 5040 (÷4 = 1260): both fine.
    if !row_bytes.is_multiple_of(4) {
        return Err(RullamaError::Inference(format!(
            "Q6_K row_bytes {row_bytes} not multiple of 4 (k={k})"
        )));
    }
    dispatch_matmul(ctx, kernels::Q6_K_DEQUANT_MATMUL, w_bytes, x, k, n).await
}

/// Internal: create buffers, pipeline, dispatch, read back.
async fn dispatch_matmul(
    ctx: &WgpuCtx,
    wgsl: &str,
    w_bytes: &[u8],
    x: &[f32],
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    // ---- buffers ----
    let params = MatmulParams {
        k: k as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul.params"),
        size: std::mem::size_of::<MatmulParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul.W"),
        size: w_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&w_buf, 0, w_bytes);

    let x_bytes_len = (x.len() * 4) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul.x"),
        size: x_bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));

    let y_bytes_len = (n * 4) as u64;
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul.y"),
        size: y_bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul.read"),
        size: y_bytes_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ---- pipeline ----
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul.bg"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
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

    // ---- dispatch ----
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, y_bytes_len);
    queue.submit(Some(encoder.finish()));

    // ---- readback ----
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
    use half::f16;

    /// Pack a 1-D f32 vector of length `k*n` (row-major, n rows of length k) into
    /// f16 little-endian bytes.
    fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
        }
        out
    }

    fn cpu_matmul_f32(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n];
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += w[j * k + i] * x[i];
            }
            y[j] = acc;
        }
        y
    }

    #[test]
    fn f16_matmul_3x4_eye() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Identity-ish: W[j, i] = 1 if i == j else 0; n=k=4.
        let k = 4;
        let n = 4;
        let mut w = vec![0f32; n * k];
        for j in 0..n {
            w[j * k + j] = 1.0;
        }
        let x: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let w_f16 = f32_to_f16_bytes(&w);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let y = pollster::block_on(matmul_f16(&ctx, &w_f16, &x, k, n)).expect("matmul");
        for i in 0..n {
            assert!(
                (y[i] - x[i]).abs() < 1e-4,
                "y[{i}]={} != x[{i}]={}",
                y[i],
                x[i]
            );
        }
    }

    /// Layer-0 fragment integration: run all four projection matmuls of layer 0
    /// (Q, K, V, O — three Q4_K and one Q6_K) on GPU and on CPU using the dequant
    /// path, and confirm they agree pairwise. This is the closest thing we have to
    /// "layer-1 GPU vs CPU diff" before we wire a full backend-parameterized forward
    /// pass in M3.
    #[test]
    fn layer0_qkv_o_proj_gpu_vs_cpu() {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available at {path}");
            return;
        }
        let bytes = std::fs::read(path).expect("read");
        let r = crate::gguf::GgufReader::new(bytes).expect("parse");

        // Deterministic input: a normalized-ish vector of length d_model = 1536.
        let d_model = 1536usize;
        let mut state: u32 = 0xCAFE_BABE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let x_qkv: Vec<f32> = (0..d_model).map(|_| next()).collect();

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");

        // ---- Q proj: Q4_K, [d_model, n_q=2048] ----
        let q_desc = r.tensor("blk.0.attn_q.weight").expect("Q desc");
        let q_bytes = r.tensor_bytes("blk.0.attn_q.weight").expect("Q bytes");
        let n_q = q_desc.dims[1] as usize;
        let mut q_w_f32 = vec![0f32; d_model * n_q];
        crate::gguf::quant::dequant_q4_k(q_bytes, &mut q_w_f32).expect("Q dequant");
        let cpu_q = cpu_matmul_f32(&q_w_f32, &x_qkv, d_model, n_q);
        let gpu_q =
            pollster::block_on(matmul_q4_k(&ctx, q_bytes, &x_qkv, d_model, n_q)).expect("Q gpu");

        // ---- K proj: Q4_K, [d_model, n_k=256] ----
        let k_desc = r.tensor("blk.0.attn_k.weight").expect("K desc");
        let k_bytes = r.tensor_bytes("blk.0.attn_k.weight").expect("K bytes");
        let n_k = k_desc.dims[1] as usize;
        let mut k_w_f32 = vec![0f32; d_model * n_k];
        crate::gguf::quant::dequant_q4_k(k_bytes, &mut k_w_f32).expect("K dequant");
        let cpu_k = cpu_matmul_f32(&k_w_f32, &x_qkv, d_model, n_k);
        let gpu_k =
            pollster::block_on(matmul_q4_k(&ctx, k_bytes, &x_qkv, d_model, n_k)).expect("K gpu");

        // ---- V proj: Q6_K, [d_model, n_v=256] ----
        let v_desc = r.tensor("blk.0.attn_v.weight").expect("V desc");
        let v_bytes = r.tensor_bytes("blk.0.attn_v.weight").expect("V bytes");
        let n_v = v_desc.dims[1] as usize;
        let mut v_w_f32 = vec![0f32; d_model * n_v];
        crate::gguf::quant::dequant_q6_k(v_bytes, &mut v_w_f32).expect("V dequant");
        let cpu_v = cpu_matmul_f32(&v_w_f32, &x_qkv, d_model, n_v);
        let gpu_v =
            pollster::block_on(matmul_q6_k(&ctx, v_bytes, &x_qkv, d_model, n_v)).expect("V gpu");

        // ---- O proj: Q4_K, [n_q, d_model] applied to "attention" vector of length n_q.
        // Synthesize an attention-output stand-in (we don't compute real attention here).
        let attn_out: Vec<f32> = (0..n_q).map(|_| next()).collect();
        let o_desc = r.tensor("blk.0.attn_output.weight").expect("O desc");
        let o_bytes = r.tensor_bytes("blk.0.attn_output.weight").expect("O bytes");
        assert_eq!(o_desc.dims, vec![n_q as u64, d_model as u64]);
        let mut o_w_f32 = vec![0f32; n_q * d_model];
        crate::gguf::quant::dequant_q4_k(o_bytes, &mut o_w_f32).expect("O dequant");
        let cpu_o = cpu_matmul_f32(&o_w_f32, &attn_out, n_q, d_model);
        let gpu_o =
            pollster::block_on(matmul_q4_k(&ctx, o_bytes, &attn_out, n_q, d_model)).expect("O gpu");

        for (name, c, g) in [
            ("Q [1536,2048]", &cpu_q, &gpu_q),
            ("K [1536, 256]", &cpu_k, &gpu_k),
            ("V [1536, 256]", &cpu_v, &gpu_v),
            ("O [2048,1536]", &cpu_o, &gpu_o),
        ] {
            let mut max_abs = 0f32;
            let mut max_rel = 0f32;
            for i in 0..c.len() {
                let abs = (g[i] - c[i]).abs();
                let rel = if c[i].abs() > 1e-3 {
                    abs / c[i].abs()
                } else {
                    0.0
                };
                if abs > max_abs {
                    max_abs = abs;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!("layer0 {name}: max_abs={max_abs:.5e}, max_rel={max_rel:.5e}");
            assert!(max_abs < 1e-2, "{name} max_abs {max_abs} exceeds 1e-2");
        }
    }

    #[test]
    fn q4_k_matmul_real_layer0_attn_q() {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available at {path}");
            return;
        }
        let bytes = std::fs::read(path).expect("read");
        let r = crate::gguf::GgufReader::new(bytes).expect("parse");

        // attn_q.weight is Q4_K with shape [1536, 2048] in our E2B fixture.
        let name = "blk.0.attn_q.weight";
        let desc = r.tensor(name).expect("tensor");
        assert!(matches!(desc.dtype, crate::gguf::GgmlDtype::Q4_K));
        assert_eq!(desc.dims, vec![1536, 2048]);
        let k = desc.dims[0] as usize;
        let n = desc.dims[1] as usize;
        let w_bytes = r.tensor_bytes(name).expect("bytes");

        let mut state: u32 = 0xC0FF_EE42;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let x: Vec<f32> = (0..k).map(|_| next()).collect();

        let mut w_f32 = vec![0f32; k * n];
        crate::gguf::quant::dequant_q4_k(w_bytes, &mut w_f32).expect("dequant");
        let cpu_y = cpu_matmul_f32(&w_f32, &x, k, n);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let gpu_y = pollster::block_on(matmul_q4_k(&ctx, w_bytes, &x, k, n)).expect("matmul");

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in 0..n {
            let abs = (gpu_y[i] - cpu_y[i]).abs();
            let rel = if cpu_y[i].abs() > 1e-3 {
                abs / cpu_y[i].abs()
            } else {
                0.0
            };
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "q4_k matmul real layer-0 attn_q: max_abs={max_abs:.5}, max_rel={max_rel:.5}, k={k}, n={n}"
        );
        assert!(
            max_abs < 1e-2,
            "Q4_K matmul GPU/CPU disagreement: max_abs={max_abs}"
        );
    }

    #[test]
    fn q6_k_matmul_real_layer0_attn_v() {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available at {path}");
            return;
        }
        let bytes = std::fs::read(path).expect("read");
        let r = crate::gguf::GgufReader::new(bytes).expect("parse");

        // attn_v.weight is Q6_K with shape [1536, 256] in our E2B fixture:
        //   k = 1536, n = 256
        let name = "blk.0.attn_v.weight";
        let desc = r.tensor(name).expect("tensor");
        assert!(matches!(desc.dtype, crate::gguf::GgmlDtype::Q6_K));
        assert_eq!(desc.dims, vec![1536, 256]);
        let k = desc.dims[0] as usize;
        let n = desc.dims[1] as usize;
        let w_bytes = r.tensor_bytes(name).expect("bytes");

        // Deterministic input.
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let x: Vec<f32> = (0..k).map(|_| next()).collect();

        // CPU reference: full dequant then matvec.
        let mut w_f32 = vec![0f32; k * n];
        crate::gguf::quant::dequant_q6_k(w_bytes, &mut w_f32).expect("dequant");
        let cpu_y = cpu_matmul_f32(&w_f32, &x, k, n);

        // GPU.
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let gpu_y = pollster::block_on(matmul_q6_k(&ctx, w_bytes, &x, k, n)).expect("matmul");

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in 0..n {
            let abs = (gpu_y[i] - cpu_y[i]).abs();
            let rel = if cpu_y[i].abs() > 1e-3 {
                abs / cpu_y[i].abs()
            } else {
                0.0
            };
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!("q6_k matmul real layer-0 attn_v: max_abs={max_abs:.5}, max_rel={max_rel:.5}");
        // Tolerance: should match exactly to within f32 rounding since both paths do
        // identical arithmetic (scalar f32 dequant + accumulate). Allow a tiny epsilon
        // for the order-of-operations difference between row-major CPU dequant + matvec
        // and the kernel's interleaved dequant-then-multiply.
        assert!(
            max_abs < 1e-3,
            "Q6_K matmul GPU/CPU disagreement: max_abs={max_abs}"
        );
    }

    /// Convert F32 values to packed BF16 bytes (truncate to high 16 bits, pack
    /// two per u32 little-endian).
    fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let bits = v.to_bits();
            let bf = (bits >> 16) as u16;
            out.extend_from_slice(&bf.to_le_bytes());
        }
        out
    }

    #[test]
    fn bf16_matmul_random_64x128() {
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 64;
        let n = 128;
        let mut state: u32 = 0xDEAD_F00D;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|_| next()).collect();

        // Round-trip CPU reference: BF16 has only 8 bits of mantissa, so
        // reproduce the truncation in the CPU side too.
        let w_bf16_bytes = f32_to_bf16_bytes(&w_f32);
        let w_round_tripped: Vec<f32> = w_f32
            .iter()
            .map(|&v| {
                let bits = v.to_bits();
                f32::from_bits(bits & 0xFFFF0000)
            })
            .collect();
        let cpu_y = cpu_matmul_f32(&w_round_tripped, &x, k, n);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let gpu_y = pollster::block_on(matmul_bf16(&ctx, &w_bf16_bytes, &x, k, n)).expect("matmul");

        let mut max_abs = 0f32;
        for i in 0..n {
            let diff = (gpu_y[i] - cpu_y[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            assert!(
                diff < 1e-4,
                "bf16 y[{i}] cpu={} gpu={} diff={}",
                cpu_y[i],
                gpu_y[i],
                diff
            );
        }
        eprintln!("bf16_matmul max_abs_diff over {n} outputs = {max_abs:e}");
    }

    #[test]
    fn bf16_matmul_batched_matches_cpu() {
        // Same numerics as the single-row test, but processed as a batch in
        // one dispatch — mirrors how the audio block FFW will use it
        // (seq frames, hidden→ffn projection).
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 64;
        let n = 32;
        let batch = 6;
        let mut state: u32 = 0xCAFEFACE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();

        let w_bf16_bytes = f32_to_bf16_bytes(&w_f32);
        let w_round_tripped: Vec<f32> = w_f32
            .iter()
            .map(|&v| f32::from_bits(v.to_bits() & 0xFFFF0000))
            .collect();

        // CPU reference: per-row matmul with the rounded weights.
        let mut cpu_y = vec![0f32; batch * n];
        for b in 0..batch {
            let row = cpu_matmul_f32(&w_round_tripped, &x_batch[b * k..(b + 1) * k], k, n);
            cpu_y[b * n..(b + 1) * n].copy_from_slice(&row);
        }

        // GPU: one batched dispatch.
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.w"),
            size: w_bf16_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_bf16_bytes);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let y_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.y"),
            size: y_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.read"),
            size: y_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test.enc"),
            });
        crate::backend::dispatch::matmul_bf16_batched_chained(
            &ctx, &pipes, &mut enc, &w_buf, &x_buf, &y_buf, k, n, batch,
        );
        enc.copy_buffer_to_buffer(&y_buf, 0, &read_buf, 0, y_size);
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
        let gpu_y: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();

        let mut max_abs = 0f32;
        for i in 0..gpu_y.len() {
            let d = (gpu_y[i] - cpu_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "bf16_matmul_batched max_abs over {} outputs = {max_abs:e}",
            gpu_y.len()
        );
        assert!(max_abs < 1e-4, "bf16 batched diff: {max_abs}");
    }

    /// Tiled f16 batched matmul must match the naive one bit-for-bit (modulo
    /// floating-point accumulation order — the inner loop sums in a different
    /// order so a tiny epsilon is acceptable, but the kernel uses the same
    /// f32 accumulator so it should still round to within ~1e-5 on these shapes).
    #[test]
    fn f16_matmul_batched_tiled_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Real vision shape slice: k=hidden=768, n=ffn=3072 is too big for a
        // quick test; downscale but stay above the tiled threshold and
        // include a non-multiple of TILE_K to exercise the bounds path.
        let k = 80; // 80 / TILE_K(16) = 5 exactly
        let n = 24; // 24 / TILE_N(8)  = 3 exactly
        let batch = 17; // 17 / TILE_M(8)  = 2 with 1 leftover → bounds check
        let mut state: u32 = 0xA5A5A5A5;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_f16 = f32_to_f16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft.w"),
            size: w_f16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_f16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mk_read = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ft.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = mk_read();
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };

        let y_naive = mk_y("ft.y_naive");
        let y_tiled = mk_y("ft.y_tiled");

        // Naive (bypass routing — call dispatcher directly via the pipeline path).
        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Call the tiled function directly, then bypass into naive by using
        // shapes the router would skip; but we want both on the SAME shape.
        // Simplest: call the naive dispatcher inline (replicate body) by
        // using a shape that triggers naive. But shapes must match. Instead:
        // use the naive batched kernel directly through the pipeline.
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "ft.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ft.naive.bg"),
                layout: &pipes.f16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.f16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_f16_batched_tiled_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_tiled, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let tiled_y = read(&y_tiled);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - tiled_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "f16 tiled vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "tiled vs naive diff: {max_abs}");
    }

    /// Same shape-parity test for the bf16 tiled batched kernel.
    #[test]
    fn bf16_matmul_batched_tiled_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 80;
        let n = 24;
        let batch = 17;
        let mut state: u32 = 0xBEEFCAFE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_bf16 = f32_to_bf16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft.w"),
            size: w_bf16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_bf16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mk_read = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bft.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = mk_read();
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };

        let y_naive = mk_y("bft.y_naive");
        let y_tiled = mk_y("bft.y_tiled");

        // Naive path (bypass routing).
        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "bft.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bft.naive.bg"),
                layout: &pipes.bf16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.bf16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_bf16_batched_tiled_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_tiled, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let tiled_y = read(&y_tiled);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - tiled_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "bf16 tiled vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "tiled vs naive diff: {max_abs}");
    }

    /// V3 tiled f16 batched matmul (32×32 output tile, 4×4 register sub-blocks).
    /// Same parity bar as v2 — bit-identical under fp accumulation order.
    #[test]
    fn f16_matmul_batched_tiled_v3_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Use shapes that hit v3 threshold and include non-multiples of 32.
        let k = 80;
        let n = 72; // 72 / 32 = 2.25 → tail
        let batch = 35; // 35 / 32 = 1.09 → tail
        let mut state: u32 = 0x33333333;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_f16 = f32_to_f16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft3.w"),
            size: w_f16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_f16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft3.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ft3.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };
        let y_naive = mk_y("ft3.y_naive");
        let y_v3 = mk_y("ft3.y_v3");

        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "ft3.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ft3.naive.bg"),
                layout: &pipes.f16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.f16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_f16_batched_tiled_v3_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_v3, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let v3_y = read(&y_v3);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - v3_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "f16 v3 vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "v3 vs naive diff: {max_abs}");
    }

    /// V3 bf16 batched parity test (mirrors f16_matmul_batched_tiled_v3).
    #[test]
    fn bf16_matmul_batched_tiled_v3_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 80;
        let n = 72;
        let batch = 35;
        let mut state: u32 = 0x44444444;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_bf16 = f32_to_bf16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft3.w"),
            size: w_bf16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_bf16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft3.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bft3.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };
        let y_naive = mk_y("bft3.y_naive");
        let y_v3 = mk_y("bft3.y_v3");

        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "bft3.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bft3.naive.bg"),
                layout: &pipes.bf16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.bf16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_bf16_batched_tiled_v3_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_v3, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let v3_y = read(&y_v3);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - v3_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "bf16 v3 vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "v3 vs naive diff: {max_abs}");
    }

    /// V2 tiled f16 batched matmul (16×16 output tile, 2×2 register sub-blocks)
    /// must match the naive kernel within fp accumulation tolerance.
    #[test]
    fn f16_matmul_batched_tiled_v2_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Shape that hits the v2 path: k, n, batch ≥ 16. Include non-multiples
        // of TILE_M=TILE_N=16 to exercise bounds checks.
        let k = 80; // 80 / 16 = 5
        let n = 40; // 40 / 16 = 2.5 → tail row
        let batch = 19; // 19 / 16 = 1.18 → tail row
        let mut state: u32 = 0xA5A5A5A5;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_f16 = f32_to_f16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft2.w"),
            size: w_f16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_f16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ft2.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mk_read = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ft2.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = mk_read();
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };

        let y_naive = mk_y("ft2.y_naive");
        let y_v2 = mk_y("ft2.y_v2");

        // Naive (bypass routing).
        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "ft2.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ft2.naive.bg"),
                layout: &pipes.f16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.f16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_f16_batched_tiled_v2_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_v2, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let v2_y = read(&y_v2);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - v2_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "f16 v2 vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "v2 vs naive diff: {max_abs}");
    }

    /// Same shape-parity test for the bf16 v2 tiled kernel.
    #[test]
    fn bf16_matmul_batched_tiled_v2_matches_naive() {
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 80;
        let n = 40;
        let batch = 19;
        let mut state: u32 = 0xBEEFCAFE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x_batch: Vec<f32> = (0..batch * k).map(|_| next()).collect();
        let w_bf16 = f32_to_bf16_bytes(&w_f32);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft2.w"),
            size: w_bf16.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&w_buf, 0, &w_bf16);
        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bft2.x"),
            size: (x_batch.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x_batch));
        let y_size = (batch * n * 4) as u64;
        let mk_y = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: y_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mk_read = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bft2.read"),
                size: y_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = mk_read();
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, y_size);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };

        let y_naive = mk_y("bft2.y_naive");
        let y_v2 = mk_y("bft2.y_v2");

        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let params = crate::backend::dispatch::BatchedMatmulParams {
                k: k as u32,
                n: n as u32,
                batch: batch as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "bft2.naive.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bft2.naive.bg"),
                layout: &pipes.bf16_matmul_batched.get_bind_group_layout(0),
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
                        resource: y_naive.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.bf16_matmul_batched);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((n as u32).div_ceil(64), batch as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::matmul_bf16_batched_tiled_v2_chained(
            &ctx, &pipes, &mut e2, &w_buf, &x_buf, &y_v2, k, n, batch,
        );
        ctx.queue.submit(Some(e2.finish()));

        let naive_y = read(&y_naive);
        let v2_y = read(&y_v2);
        let mut max_abs = 0f32;
        for i in 0..naive_y.len() {
            let d = (naive_y[i] - v2_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "bf16 v2 vs naive max_abs over {} outputs = {max_abs:e}",
            naive_y.len()
        );
        assert!(max_abs < 1e-4, "v2 vs naive diff: {max_abs}");
    }

    #[test]
    fn f16_matmul_random_64x128() {
        let _ = env_logger::builder().is_test(true).try_init();
        let k = 64;
        let n = 128;
        // Deterministic pseudo-random (LCG) so the test is stable.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let w: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|_| next()).collect();
        let w_f16 = f32_to_f16_bytes(&w);
        let cpu_y = cpu_matmul_f32(&w, &x, k, n);

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let gpu_y = pollster::block_on(matmul_f16(&ctx, &w_f16, &x, k, n)).expect("matmul");

        // Tolerance accounts for f16 round-trip on each weight (~6e-4 relative) plus
        // accumulator order differences.
        let mut max_abs = 0f32;
        for i in 0..n {
            let diff = (gpu_y[i] - cpu_y[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            assert!(
                diff < 1e-2,
                "y[{i}] cpu={} gpu={} diff={}",
                cpu_y[i],
                gpu_y[i],
                diff
            );
        }
        eprintln!("f16_matmul max_abs_diff over {n} outputs = {max_abs:e}");
    }

    /// Flash vision attention must match the original kernel within numerical
    /// tolerance. Both produce the same softmax result; differences are pure
    /// floating-point accumulation order (tiled sums vs straight sums).
    #[test]
    fn vision_attention_flash_matches_original() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Use a non-multiple of TILE_T=32 to exercise tail handling.
        let n_patches = 100;
        let n_heads = 3;
        let head_dim = 64;
        let total = n_patches * n_heads * head_dim;

        let mut state: u32 = 0xFEEDFACE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let q: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let mkbuf = |label: &'static str, data: &[f32]| {
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            buf
        };
        let q_buf = mkbuf("vat.q", &q);
        let k_buf = mkbuf("vat.k", &k);
        let v_buf = mkbuf("vat.v", &v);
        let mk_out = |label: &'static str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let out_orig = mk_out("vat.out_orig");
        let out_flash = mk_out("vat.out_flash");

        // Run ORIGINAL kernel directly (bypassing the routing).
        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct Params {
                head_dim: u32,
                n_heads: u32,
                n_patches: u32,
                _pad: u32,
            }
            let params = Params {
                head_dim: head_dim as u32,
                n_heads: n_heads as u32,
                n_patches: n_patches as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "vat.orig.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vat.orig.bg"),
                layout: &pipes.vision_attention.get_bind_group_layout(0),
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
                        resource: out_orig.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.vision_attention);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        // Run FLASH kernel.
        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::vision_attention_flash_chained(
            &ctx, &pipes, &mut e2, &q_buf, &k_buf, &v_buf, &out_flash, head_dim, n_heads, n_patches,
        );
        ctx.queue.submit(Some(e2.finish()));

        // Read both back.
        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vat.read"),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, (total * 4) as u64);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };
        let o_orig = read(&out_orig);
        let o_flash = read(&out_flash);

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in 0..total {
            let d = (o_orig[i] - o_flash[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
            let denom = o_orig[i].abs().max(1e-6);
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
        eprintln!("vision_attention flash vs original: max_abs={max_abs:e} max_rel={max_rel:e}");
        assert!(max_abs < 1e-4, "flash diverges: max_abs={max_abs}");
    }

    /// Q4 multi-query flash vision attention must match the original within
    /// fp tolerance. Uses a non-multiple of Q_PER_WG=4 in n_patches to exercise
    /// the per-workgroup query-count clamp.
    #[test]
    fn vision_attention_flash_q4_matches_original() {
        let _ = env_logger::builder().is_test(true).try_init();
        let n_patches = 102; // 102 / 4 = 25.5 → tail workgroup with 2 queries
        let n_heads = 3;
        let head_dim = 64;
        let total = n_patches * n_heads * head_dim;

        let mut state: u32 = 0xC0FFEE13;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let q: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let mkbuf = |label: &'static str, data: &[f32]| {
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            buf
        };
        let q_buf = mkbuf("vatq.q", &q);
        let k_buf = mkbuf("vatq.k", &k);
        let v_buf = mkbuf("vatq.v", &v);
        let mk_out = |label: &'static str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let out_orig = mk_out("vatq.out_orig");
        let out_q4 = mk_out("vatq.out_q4");

        // Run ORIGINAL kernel (Q=1).
        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                head_dim: u32,
                n_heads: u32,
                n_patches: u32,
                _pad: u32,
            }
            let params = P {
                head_dim: head_dim as u32,
                n_heads: n_heads as u32,
                n_patches: n_patches as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "vatq.orig.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vatq.orig.bg"),
                layout: &pipes.vision_attention.get_bind_group_layout(0),
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
                        resource: out_orig.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.vision_attention);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::vision_attention_flash_q4_chained(
            &ctx, &pipes, &mut e2, &q_buf, &k_buf, &v_buf, &out_q4, head_dim, n_heads, n_patches,
        );
        ctx.queue.submit(Some(e2.finish()));

        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vatq.read"),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, (total * 4) as u64);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };
        let o_orig = read(&out_orig);
        let o_q4 = read(&out_q4);

        let mut max_abs = 0f32;
        for i in 0..total {
            let d = (o_orig[i] - o_q4[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!("vision_attention Q4 vs original: max_abs={max_abs:e}");
        assert!(max_abs < 1e-4, "Q4 diverges: max_abs={max_abs}");
    }

    /// Q8 multi-query flash variant — same parity check as Q4 with a non-multiple
    /// of 8 in n_patches to exercise the tail-workgroup clamp.
    #[test]
    fn vision_attention_flash_q8_matches_original() {
        let _ = env_logger::builder().is_test(true).try_init();
        let n_patches = 103; // 103 / 8 = 12.875 → tail workgroup
        let n_heads = 3;
        let head_dim = 64;
        let total = n_patches * n_heads * head_dim;

        let mut state: u32 = 0xC0DEFACE;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0) - 0.5
        };
        let q: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|_| next() * 0.1).collect();

        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = crate::backend::Pipelines::new(&ctx.device);

        let mkbuf = |label: &'static str, data: &[f32]| {
            let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            buf
        };
        let q_buf = mkbuf("vat8.q", &q);
        let k_buf = mkbuf("vat8.k", &k);
        let v_buf = mkbuf("vat8.v", &v);
        let mk_out = |label: &'static str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let out_orig = mk_out("vat8.out_orig");
        let out_q8 = mk_out("vat8.out_q8");

        let mut e1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                head_dim: u32,
                n_heads: u32,
                n_patches: u32,
                _pad: u32,
            }
            let params = P {
                head_dim: head_dim as u32,
                n_heads: n_heads as u32,
                n_patches: n_patches as u32,
                _pad: 0,
            };
            let p_buf = crate::backend::dispatch::write_uniform(
                &ctx.device,
                &ctx.queue,
                "vat8.orig.params",
                &params,
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vat8.orig.bg"),
                layout: &pipes.vision_attention.get_bind_group_layout(0),
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
                        resource: out_orig.as_entire_binding(),
                    },
                ],
            });
            let mut cp = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipes.vision_attention);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(n_patches as u32, n_heads as u32, 1);
        }
        ctx.queue.submit(Some(e1.finish()));

        let mut e2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        crate::backend::dispatch::vision_attention_flash_q8_chained(
            &ctx, &pipes, &mut e2, &q_buf, &k_buf, &v_buf, &out_q8, head_dim, n_heads, n_patches,
        );
        ctx.queue.submit(Some(e2.finish()));

        let read = |buf: &wgpu::Buffer| -> Vec<f32> {
            let r = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vat8.read"),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e.copy_buffer_to_buffer(buf, 0, &r, 0, (total * 4) as u64);
            ctx.queue.submit(Some(e.finish()));
            let slice = r.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |x| {
                tx.send(x).unwrap();
            });
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            r.unmap();
            out
        };
        let o_orig = read(&out_orig);
        let o_q8 = read(&out_q8);

        let mut max_abs = 0f32;
        for i in 0..total {
            let d = (o_orig[i] - o_q8[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!("vision_attention Q8 vs original: max_abs={max_abs:e}");
        assert!(max_abs < 1e-4, "Q8 diverges: max_abs={max_abs}");
    }
}
