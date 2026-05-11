//! iOS bench harness — exposes a single C-ABI entry point that runs the
//! representative matmul + attention shapes and prints results to stdout.
//! Apps embedding this crate get the same numbers `examples/matmul_bench`
//! prints on macOS, but evaluated on the iPhone's Metal driver.

use std::ffi::c_char;
use std::time::Instant;

use bytemuck::Pod;
use bytemuck::Zeroable;
use rullama::backend::{Pipelines, WgpuCtx, dispatch};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params { k: u32, n: u32, batch: u32, _pad: u32 }

/// Run a small representative perf sweep, printing to stdout. Returns 0
/// on success, non-zero on init failure.
#[unsafe(no_mangle)]
pub extern "C" fn rullama_run_bench() -> i32 {
    // wgpu init + bench loop, all sync via pollster.
    let Ok(ctx) = pollster::block_on(WgpuCtx::new()) else {
        println!("rullama-bench: ERROR — wgpu adapter request failed");
        return 1;
    };
    let info = ctx.adapter.get_info();
    let limits = ctx.adapter.limits();
    println!("rullama-bench: adapter = {} / {:?}", info.name, info.backend);
    println!("rullama-bench:   subgroup range = [{}, {}]",
        info.subgroup_min_size, info.subgroup_max_size);
    println!("rullama-bench:   features: subgroups={}, has_f16={}",
        ctx.has_subgroups, ctx.has_f16);
    println!("rullama-bench:   max_storage_buffer_binding_size = {}",
        limits.max_storage_buffer_binding_size);

    let pipes = Pipelines::new_with_features(&ctx.device, ctx.has_subgroups, ctx.has_f16);

    // Vision-tower-representative matmul shapes.
    let shapes: [(&str, usize, usize, usize); 4] = [
        ("attn QKV  768x768  ", 768,  768, 2304),
        ("attn out  768x768  ", 768,  768, 2304),
        ("ffn up    768x3072 ", 768, 3072, 2304),
        ("ffn down  3072x768 ", 3072, 768, 2304),
    ];
    for (label, k, n, batch) in shapes {
        let (w, x, y) = make_buffers(&ctx, k, n, batch);
        // Warmup
        for _ in 0..3 {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::matmul_f16_batched_chained(&ctx, &pipes, &mut enc, &w, &x, &y, k, n, batch);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        }
        let n_iters = 5usize;
        let t = Instant::now();
        for _ in 0..n_iters {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::matmul_f16_batched_chained(&ctx, &pipes, &mut enc, &w, &x, &y, k, n, batch);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        let per = t.elapsed() / n_iters as u32;
        let flops = 2.0 * k as f64 * n as f64 * batch as f64;
        let gflops = flops / per.as_secs_f64() / 1e9;
        println!("rullama-bench: {label:<22} k={k:>4} n={n:>4} batch={batch:>4} | {:?}/iter  {gflops:>6.1} GFLOPS", per);
    }

    // Vision attention bench.
    let n_patches = 2304usize;
    let n_heads = 12usize;
    let head_dim = 64usize;
    let qkv_len = n_patches * n_heads * head_dim;
    let qkv = vec![0.01f32; qkv_len];
    let q_buf = upload_storage(&ctx, "q", &qkv);
    let k_buf = upload_storage(&ctx, "k", &qkv);
    let v_buf = upload_storage(&ctx, "v", &qkv);
    let out_buf = alloc_storage(&ctx, "out", qkv_len);

    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    }
    let t = Instant::now();
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    let per = t.elapsed() / 3;
    println!("rullama-bench: vision_attention (router): {:?}/iter  (×16 layers ≈ {:?})", per, per * 16);

    println!("rullama-bench: done");
    0
}

/// Optional convenience: returns the adapter description as a static
/// C-string. Useful for logging from Swift without re-querying.
#[unsafe(no_mangle)]
pub extern "C" fn rullama_describe_adapter() -> *const c_char {
    static ADAPTER: &[u8] = b"rullama-ios-bench\0";
    ADAPTER.as_ptr() as *const c_char
}

fn make_buffers(ctx: &WgpuCtx, k: usize, n: usize, batch: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    // f16 weight packed as u32: k * n / 2 u32s.
    let w_pairs = (k * n / 2).max(1);
    let mut w_bytes = vec![0u32; w_pairs];
    // Fill with small values to avoid Inf/NaN.
    for v in w_bytes.iter_mut() { *v = 0x33333333u32; }
    let w_buf = upload_storage_u32(ctx, "w", &w_bytes);
    let x = vec![0.01f32; batch * k];
    let x_buf = upload_storage(ctx, "x", &x);
    let y_buf = alloc_storage(ctx, "y", batch * n);
    (w_buf, x_buf, y_buf)
}

fn upload_storage(ctx: &WgpuCtx, label: &str, data: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn upload_storage_u32(ctx: &WgpuCtx, label: &str, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn alloc_storage(ctx: &WgpuCtx, label: &str, n_f32: usize) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n_f32 * 4).max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
