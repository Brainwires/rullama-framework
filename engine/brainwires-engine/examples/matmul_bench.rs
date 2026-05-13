//! Microbenchmark for the batched f16 matmul on vision-representative
//! shapes. Times naive, v1 tiled, and v2 tiled at the same shape so we
//! can see how much of the 51 s vision encode is actually matmul.
//!
//! The big vision matmuls are:
//!   • qkv:  k=768,  n=768,  batch=2304  (×3 q,k,v separately)
//!   • attn: k=768,  n=768,  batch=2304
//!   • ffn_up/gate:  k=768,  n=3072, batch=2304
//!   • ffn_down:     k=3072, n=768,  batch=2304
//!
//! 16 blocks of those = ~96 matmuls per encode. If each takes ~500 ms
//! the encode is matmul-bound. If each takes ~50 ms, something else
//! is eating the time.

use std::time::Instant;

use rullama::backend::{Pipelines, WgpuCtx, dispatch};

fn run_shape(
    label: &str,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    w_buf: &wgpu::Buffer,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
    n_iters: usize,
) {
    // Warmup.
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        dispatch::matmul_f16_batched_chained(ctx, pipes, &mut enc, w_buf, x_buf, y_buf, k, n, batch);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }

    let t = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        dispatch::matmul_f16_batched_chained(ctx, pipes, &mut enc, w_buf, x_buf, y_buf, k, n, batch);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    let gflops = 2.0 * (k * n * batch) as f64 / per_iter.as_secs_f64() / 1e9;
    println!("{label:30} k={k:5} n={n:5} batch={batch:5}: {per_iter:?}/iter   {gflops:.2} GFLOPS");
}

fn run_shape_force(
    label: &str,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dispatch_x: u32, dispatch_y: u32,
    k: usize, n: usize, batch: usize,
    n_iters: usize,
) {
    // Warmup.
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bind_group, &[]);
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }

    let t = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bind_group, &[]);
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    let gflops = 2.0 * (k * n * batch) as f64 / per_iter.as_secs_f64() / 1e9;
    let _ = pipes; // suppress unused
    let _ = n; let _ = batch;
    println!("{label:30}                                       : {per_iter:?}/iter   {gflops:.2} GFLOPS");
}

fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 2];
    for (i, &v) in values.iter().enumerate() {
        let h = half::f16::from_f32(v).to_bits();
        out[i*2]     = (h & 0xFF) as u8;
        out[i*2 + 1] = (h >> 8)   as u8;
    }
    out
}

fn make_buffers(ctx: &WgpuCtx, k: usize, n: usize, batch: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let mut state: u32 = 0xCAFEFACE;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / 16777216.0) - 0.5
    };
    let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.05).collect();
    let x: Vec<f32> = (0..batch * k).map(|_| next() * 0.5).collect();
    let w_bytes = f32_to_f16_bytes(&w_f32);

    let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.w"), size: w_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&w_buf, 0, &w_bytes);
    let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.x"), size: (x.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
    let y_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.y"), size: (batch * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (w_buf, x_buf, y_buf)
}

fn main() {
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu init failed");
    let pipes = Pipelines::new_with_features(&ctx.device, ctx.has_subgroups, ctx.has_f16);
    let info = ctx.adapter.get_info();
    println!("Adapter: {} / {:?}  (subgroups: {})", info.name, info.backend, ctx.has_subgroups);

    // Real vision shapes.
    let shapes = [
        ("attn QKV  768x768",     768, 768,  2304),
        ("attn out  768x768",     768, 768,  2304),
        ("ffn up    768x3072",    768, 3072, 2304),
        ("ffn down  3072x768",   3072, 768,  2304),
    ];

    // Pre-allocate buffers for each shape (separately, since k,n,batch vary).
    println!("\n=== full router (whatever variant fires) ===");
    for (label, k, n, batch) in shapes {
        let (w, x, y) = make_buffers(&ctx, k, n, batch);
        run_shape(label, &ctx, &pipes, &w, &x, &y, k, n, batch, 5);
    }

    // Now force each variant for the biggest shape (ffn_up) so we can see
    // the spread directly.
    println!("\n=== forced variants on ffn_up shape (k=768, n=3072, batch=2304) ===");
    let k = 768;
    let n = 3072;
    let batch = 2304;
    let (w, x, y) = make_buffers(&ctx, k, n, batch);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params { k: u32, n: u32, batch: u32, _pad: u32 }
    let params = Params { k: k as u32, n: n as u32, batch: batch as u32, _pad: 0 };
    let p_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.p"), size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&p_buf, 0, bytemuck::bytes_of(&params));

    let mk_bg = |pipeline: &wgpu::ComputePipeline| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
            ],
        })
    };

    let bg_naive = mk_bg(&pipes.f16_matmul_batched);
    let bg_v1    = mk_bg(&pipes.f16_matmul_batched_tiled);
    let bg_v2    = mk_bg(&pipes.f16_matmul_batched_tiled_v2);

    run_shape_force("naive", &ctx, &pipes, &pipes.f16_matmul_batched,
        &bg_naive, (n as u32).div_ceil(64), batch as u32, k, n, batch, 5);
    run_shape_force("v1 tiled 8×8×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled,
        &bg_v1, (n as u32).div_ceil(8), (batch as u32).div_ceil(8), k, n, batch, 5);
    run_shape_force("v2 tiled 16×16×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled_v2,
        &bg_v2, (n as u32).div_ceil(16), (batch as u32).div_ceil(16), k, n, batch, 5);
    let bg_v3 = mk_bg(&pipes.f16_matmul_batched_tiled_v3);
    run_shape_force("v3 tiled 32×32×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled_v3,
        &bg_v3, (n as u32).div_ceil(32), (batch as u32).div_ceil(32), k, n, batch, 5);
    let bg_v4 = mk_bg(&pipes.f16_matmul_batched_tiled_v4);
    run_shape_force("v4 tiled 64×32×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled_v4,
        &bg_v4, (n as u32).div_ceil(32), (batch as u32).div_ceil(64), k, n, batch, 5);
    if let Some(pipe_f) = pipes.f16_matmul_batched_tiled_v3_f16lds.as_ref() {
        let bg_vf = mk_bg(pipe_f);
        run_shape_force("v3 f16-LDS 32×32×16", &ctx, &pipes, pipe_f,
            &bg_vf, (n as u32).div_ceil(32), (batch as u32).div_ceil(32), k, n, batch, 5);
    }

    // Per-block estimate: 6 matmuls per block × 16 blocks = 96 matmuls.
    // Avg matmul ~ middle of the shapes above.
    println!("\n96 matmuls × per-iter from above ≈ expected total matmul time per encode.");

    // ---- vision_attention bench ----
    println!("\n=== vision_attention (n_patches=2304, n_heads=12, head_dim=64) ===");
    let n_patches = 2304usize;
    let n_heads = 12usize;
    let head_dim = 64usize;
    let qkv_size = (n_patches * n_heads * head_dim * 4) as u64;
    let mkbuf = |label| ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: qkv_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let q_buf = mkbuf("attn.q");
    let k_buf = mkbuf("attn.k");
    let v_buf = mkbuf("attn.v");
    let out_buf = mkbuf("attn.out");
    let zeros = vec![0f32; n_patches * n_heads * head_dim];
    ctx.queue.write_buffer(&q_buf, 0, bytemuck::cast_slice(&zeros));
    ctx.queue.write_buffer(&k_buf, 0, bytemuck::cast_slice(&zeros));
    ctx.queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&zeros));

    // Warmup.
    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }
    let t = Instant::now();
    let n_iters = 5;
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    println!("vision_attention (router → Q8): {per_iter:?}/iter  (×16 blocks ≈ {:?} total)", per_iter * 16);

    // Bench the original (Q=1) for comparison.
    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }
    let t2 = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let per_iter2 = t2.elapsed() / n_iters as u32;
    println!("vision_attention_flash (Q=1)   : {per_iter2:?}/iter  (×16 blocks ≈ {:?} total)", per_iter2 * 16);

    // Bench Q=8 directly.
    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_q8_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }
    let t3 = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_q8_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let per_iter3 = t3.elapsed() / n_iters as u32;
    println!("vision_attention_flash (Q=8)   : {per_iter3:?}/iter  (×16 blocks ≈ {:?} total)", per_iter3 * 16);

    // Bench Q=16 directly.
    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_q16_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }
    let t4 = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_flash_q16_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let per_iter4 = t4.elapsed() / n_iters as u32;
    println!("vision_attention_flash (Q=16)  : {per_iter4:?}/iter  (×16 blocks ≈ {:?} total)", per_iter4 * 16);

    // Bench head-major (HPD) subgroup variant: transpose Q/K/V, run attn, transpose back.
    // Measures end-to-end (3 transposes-in + attention + 1 transpose-out).
    if let Some(sub_hpd) = pipes.vision_attention_flash_sub_hpd.as_ref() {
        let q_t = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_t"), size: (n_patches * n_heads * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let k_t = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_t"), size: (n_patches * n_heads * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let v_t = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_t"), size: (n_patches * n_heads * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let out_t = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_t"), size: (n_patches * n_heads * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        for _ in 0..2 {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &q_buf, &q_t, n_patches, n_heads, head_dim);
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &k_buf, &k_t, n_patches, n_heads, head_dim);
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &v_buf, &v_t, n_patches, n_heads, head_dim);
            dispatch::vision_attention_flash_sub_hpd_chained(&ctx, &pipes, sub_hpd, &mut enc, &q_t, &k_t, &v_t, &out_t,
                head_dim, n_heads, n_patches);
            dispatch::transpose_hpd_to_phd_chained(&ctx, &pipes, &mut enc, &out_t, &out_buf, n_patches, n_heads, head_dim);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let th = Instant::now();
        for _ in 0..n_iters {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &q_buf, &q_t, n_patches, n_heads, head_dim);
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &k_buf, &k_t, n_patches, n_heads, head_dim);
            dispatch::transpose_phd_to_hpd_chained(&ctx, &pipes, &mut enc, &v_buf, &v_t, n_patches, n_heads, head_dim);
            dispatch::vision_attention_flash_sub_hpd_chained(&ctx, &pipes, sub_hpd, &mut enc, &q_t, &k_t, &v_t, &out_t,
                head_dim, n_heads, n_patches);
            dispatch::transpose_hpd_to_phd_chained(&ctx, &pipes, &mut enc, &out_t, &out_buf, n_patches, n_heads, head_dim);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let per_iterh = th.elapsed() / n_iters as u32;
        // Also bench attention-only (cost without the wrap transposes), with same warmup.
        let ta = Instant::now();
        for _ in 0..n_iters {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::vision_attention_flash_sub_hpd_chained(&ctx, &pipes, sub_hpd, &mut enc, &q_t, &k_t, &v_t, &out_t,
                head_dim, n_heads, n_patches);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let per_itera = ta.elapsed() / n_iters as u32;
        println!("vision_attention_flash (sub+HPD): {per_iterh:?}/iter  (with 3+1 transposes; ×16 ≈ {:?})", per_iterh * 16);
        println!("vision_attention_flash (HPD only): {per_itera:?}/iter  (attn-only; ×16 ≈ {:?})", per_itera * 16);

        if let Some(sub_hpd_f16_q16) = pipes.vision_attention_flash_sub_hpd_f16_q16.as_ref() {
            for _ in 0..2 {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                dispatch::vision_attention_flash_sub_hpd_f16_q16_chained(&ctx, &pipes, sub_hpd_f16_q16, &mut enc, &q_t, &k_t, &v_t, &out_t,
                    head_dim, n_heads, n_patches);
                ctx.queue.submit(Some(enc.finish()));
                ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            let taq = Instant::now();
            for _ in 0..n_iters {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                dispatch::vision_attention_flash_sub_hpd_f16_q16_chained(&ctx, &pipes, sub_hpd_f16_q16, &mut enc, &q_t, &k_t, &v_t, &out_t,
                    head_dim, n_heads, n_patches);
                ctx.queue.submit(Some(enc.finish()));
            }
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let per_iteraq = taq.elapsed() / n_iters as u32;
            println!("vision_attention_flash (HPD f16 Q=16): {per_iteraq:?}/iter  (attn-only; ×16 ≈ {:?})", per_iteraq * 16);
        }

        if let Some(sub_hpd_f16) = pipes.vision_attention_flash_sub_hpd_f16.as_ref() {
            for _ in 0..2 {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                dispatch::vision_attention_flash_sub_hpd_chained(&ctx, &pipes, sub_hpd_f16, &mut enc, &q_t, &k_t, &v_t, &out_t,
                    head_dim, n_heads, n_patches);
                ctx.queue.submit(Some(enc.finish()));
                ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            let taf = Instant::now();
            for _ in 0..n_iters {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                dispatch::vision_attention_flash_sub_hpd_chained(&ctx, &pipes, sub_hpd_f16, &mut enc, &q_t, &k_t, &v_t, &out_t,
                    head_dim, n_heads, n_patches);
                ctx.queue.submit(Some(enc.finish()));
            }
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let per_iteraf = taf.elapsed() / n_iters as u32;
            println!("vision_attention_flash (HPD f16): {per_iteraf:?}/iter  (attn-only; ×16 ≈ {:?})", per_iteraf * 16);
        }
    }

    // Bench TILE_T=64 / Q=12 subgroup variant if available.
    if let Some(sub64) = pipes.vision_attention_flash_sub_t64.as_ref() {
        for _ in 0..2 {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::vision_attention_flash_sub_t64_chained(&ctx, &pipes, sub64, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
                head_dim, n_heads, n_patches);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let t6 = Instant::now();
        for _ in 0..n_iters {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::vision_attention_flash_sub_t64_chained(&ctx, &pipes, sub64, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
                head_dim, n_heads, n_patches);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let per_iter6 = t6.elapsed() / n_iters as u32;
        println!("vision_attention_flash (T64/Q12): {per_iter6:?}/iter  (×16 blocks ≈ {:?} total)", per_iter6 * 16);
    }

    // Bench subgroup-collapsed flash attention, if available.
    if let Some(sub) = pipes.vision_attention_flash_subgroup.as_ref() {
        for _ in 0..2 {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::vision_attention_flash_subgroup_chained(&ctx, &pipes, sub, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
                head_dim, n_heads, n_patches);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let t5 = Instant::now();
        for _ in 0..n_iters {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch::vision_attention_flash_subgroup_chained(&ctx, &pipes, sub, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
                head_dim, n_heads, n_patches);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let per_iter5 = t5.elapsed() / n_iters as u32;
        println!("vision_attention_flash (sub)   : {per_iter5:?}/iter  (×16 blocks ≈ {:?} total)", per_iter5 * 16);
    }

    // ---- Small-op benches at vision shapes ----
    // Vision encode for 768×528 has n_patches=2304, hidden=768, ffn=3072.
    // Per-layer call counts (counted by reading vision.rs):
    //   rmsnorm:        4 calls @ 2304×768
    //   rmsnorm_per_row 2 calls (Q,K) @ 27648×64
    //   clamp:          ~6 calls @ 2304×3072 (worst case ffn)
    //   quick_geglu:    1 call  @ 2304×3072 → 7M elements
    //   residual_add:   2 calls @ 2304×768
    //   rope_2d:        1 call  @ 27648 elements
    //   pos_embed_add:  1 call  total (not per-layer)
    println!("\n=== small-op benches at vision shapes ===");
    let hidden = 768usize;
    let ffn = 3072usize;
    let n_patches_v = 2304usize;
    let n_heads_v = 12usize;
    let head_dim_v = 64usize;

    let bench_n = 5;
    let bench = |label: &str, f: &dyn Fn(&mut wgpu::CommandEncoder)| {
        // Warmup.
        for _ in 0..2 {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            f(&mut enc);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let t = Instant::now();
        for _ in 0..bench_n {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            f(&mut enc);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let per_iter = t.elapsed() / bench_n as u32;
        println!("{label:36}: {per_iter:?}/iter");
    };

    // Allocate buffers for the small ops.
    let mk_buf = |label: &'static str, n: usize| ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let scratch_hidden = mk_buf("sc.hidden", n_patches_v * hidden);
    let scratch_hidden2 = mk_buf("sc.hidden2", n_patches_v * hidden);
    let scratch_ffn = mk_buf("sc.ffn", n_patches_v * ffn);
    let scratch_ffn2 = mk_buf("sc.ffn2", n_patches_v * ffn);
    let scratch_ffn3 = mk_buf("sc.ffn3", n_patches_v * ffn);
    let scratch_norm = mk_buf("sc.norm", hidden);
    let scratch_qk = mk_buf("sc.qk", n_patches_v * n_heads_v * head_dim_v);
    let scratch_qk2 = mk_buf("sc.qk2", n_patches_v * n_heads_v * head_dim_v);
    let _scratch_freq = mk_buf("sc.freq", head_dim_v / 2);
    let scratch_pos_x = mk_buf("sc.pos_x", n_patches_v);
    let scratch_pos_y = mk_buf("sc.pos_y", n_patches_v);

    bench("rmsnorm_per_row hidden", &|enc| {
        dispatch::rmsnorm_per_row_chained(&ctx, &pipes, enc, &scratch_hidden, None, &scratch_norm, &scratch_hidden2,
            n_patches_v, hidden, 1e-6);
    });
    bench("rmsnorm_per_row Q/K (per-head)", &|enc| {
        dispatch::rmsnorm_per_row_chained(&ctx, &pipes, enc, &scratch_qk, None, &scratch_norm, &scratch_qk2,
            n_patches_v * n_heads_v, head_dim_v, 1e-6);
    });
    bench("clamp on ffn buf (7M)", &|enc| {
        dispatch::clamp_chained(&ctx, &pipes, enc, &scratch_ffn, n_patches_v * ffn, -10.0, 10.0);
    });
    bench("clamp on hidden buf (1.7M)", &|enc| {
        dispatch::clamp_chained(&ctx, &pipes, enc, &scratch_hidden, n_patches_v * hidden, -10.0, 10.0);
    });
    bench("quick_geglu (7M)", &|enc| {
        dispatch::quick_geglu_chained(&ctx, &pipes, enc, &scratch_ffn, &scratch_ffn2, &scratch_ffn3, n_patches_v * ffn);
    });
    bench("residual_add hidden", &|enc| {
        dispatch::residual_add_chained(&ctx, &pipes, enc, &scratch_hidden, &scratch_hidden2,
            n_patches_v * hidden);
    });
    bench("rope_2d Q", &|enc| {
        dispatch::rope_2d_chained(&ctx, &pipes, enc, &scratch_qk, &scratch_pos_x, &scratch_pos_y,
            head_dim_v, n_heads_v, n_patches_v, 100.0);
    });

    println!("\n(rough per-encode totals: ×16 for per-layer ops, ×1 for one-shot ops)");
}
