//! Phase-B GPU gate: run the NEW sparse-MoE FFN sub-path (router → per-slot
//! fused-gate_up/GeGLU/down → combine → post-norm) on REAL `gemma4:26b` blk.N
//! weights, GPU vs the CPU oracle in reference/moe.rs, on the same hidden
//! vector. Streams only that layer's MoE tensors (~420 MB) — the full 18 GB
//! model can't be GPU-resident on a 16 GB machine, so this is the
//! real-weights validation gate for the encode_layer MoE branch.
//!
//!   cargo run -p rullama --release --example moe_layer_parity -- \
//!       ~/.ollama/models/blobs/sha256-<digest> [layer]

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

use rullama::backend::dispatch::{
    make_dummy_storage, moe_combine_chained, moe_expert_matmul_chained, moe_geglu_halves_chained,
    moe_router_chained, rmsnorm_chained,
};
use rullama::backend::{BindGroupCache, Pipelines, WeightCache, WgpuCtx};
use rullama::gguf::{FileFetcher, GgufReader};
use rullama::model::config::Gemma4Config;
use rullama::reference::Weights;
use rullama::reference::moe::{moe_experts, route};
use rullama::reference::ops::rmsnorm;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: moe_layer_parity <path-to-gguf> [layer]");
        return ExitCode::from(2);
    };
    let layer: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let fetcher = FileFetcher::open(std::path::Path::new(&path)).expect("open");
    let r = pollster::block_on(GgufReader::new_streaming(Arc::new(fetcher))).expect("gguf");
    let cfg = Gemma4Config::from_gguf(&r).expect("config");
    assert!(cfg.has_moe(), "expects an MoE checkpoint (gemma4:26b)");
    let d_model = cfg.d_model as usize;
    let top_k = cfg.expert_used_count as usize;
    let e_ffn = cfg.expert_ffn as usize;
    let n_experts = cfg.expert_count as usize;
    let eps = cfg.rms_norm_eps;
    let prefix = format!("blk.{layer}.");

    // Deterministic "hidden state" with realistic scale.
    let hidden: Vec<f32> = (0..d_model)
        .map(|i| ((i as f32) * 0.137).sin() * 2.0)
        .collect();

    let r_arc = Arc::new(r);
    let weights = Weights::new(r_arc.clone());

    // ---- CPU oracle ----
    let selected = route(&cfg, &weights, layer, &hidden).expect("route");
    println!("cpu selected experts: {selected:?}");
    let pre2 = weights
        .load(&format!("{prefix}pre_ffw_norm_2.weight"))
        .expect("pre_ffw_norm_2");
    let mut moe_x = vec![0f32; d_model];
    rmsnorm(&hidden, Some(&pre2), eps, &mut moe_x);
    let cpu_moe = moe_experts(&cfg, &weights, layer, &moe_x, &selected).expect("experts");
    let post2 = weights
        .load(&format!("{prefix}post_ffw_norm_2.weight"))
        .expect("post_ffw_norm_2");
    let mut cpu_out = vec![0f32; d_model];
    rmsnorm(&cpu_moe, Some(&post2), eps, &mut cpu_out);

    // ---- GPU path (the exact encode_layer sub-graph) ----
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let bind_cache = Arc::new(BindGroupCache::default());
    let wcache = WeightCache::new(
        r_arc.clone(),
        ctx.device.clone(),
        ctx.queue.clone(),
        bind_cache,
    );

    let router_w =
        pollster::block_on(wcache.buffer_async(&format!("{prefix}ffn_gate_inp.weight"))).unwrap();
    println!(
        "router buffer: {} bytes (expect {})",
        router_w.size(),
        d_model * n_experts * 4
    );
    let router_w =
        pollster::block_on(wcache.buffer_async(&format!("{prefix}ffn_gate_inp.weight"))).unwrap();
    let router_scale =
        pollster::block_on(wcache.buffer_opt_async(&format!("{prefix}ffn_gate_inp.scale")))
            .unwrap();
    let gu_name = format!("{prefix}ffn_gate_up_exps.weight");
    let gate_up = pollster::block_on(wcache.buffer_async(&gu_name)).unwrap();
    let gate_up_dt = wcache.dtype(&gu_name).unwrap();
    let d_name = format!("{prefix}ffn_down_exps.weight");
    let down_w = pollster::block_on(wcache.buffer_async(&d_name)).unwrap();
    let down_dt = wcache.dtype(&d_name).unwrap();
    let down_scale =
        pollster::block_on(wcache.buffer_opt_async(&format!("{prefix}ffn_down_exps.scale")))
            .unwrap();
    let pre2_b =
        pollster::block_on(wcache.buffer_async(&format!("{prefix}pre_ffw_norm_2.weight"))).unwrap();
    let post2_b =
        pollster::block_on(wcache.buffer_async(&format!("{prefix}post_ffw_norm_2.weight")))
            .unwrap();

    let device = &ctx.device;
    let queue = &ctx.queue;
    let dummy = make_dummy_storage(device, "dummy");
    let mk = |label: &str, n: usize| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let hidden_b = mk("h", d_model);
    queue.write_buffer(&hidden_b, 0, bytemuck::cast_slice(&hidden));
    let ids_b = mk("ids", top_k);
    let w_b = mk("w", top_k);
    let moe_x_b = mk("mx", d_model);
    let gu_bufs: Vec<_> = (0..top_k)
        .map(|s| mk(&format!("gu{s}"), 2 * e_ffn))
        .collect();
    let act_bufs: Vec<_> = (0..top_k).map(|s| mk(&format!("a{s}"), e_ffn)).collect();
    let down_bufs: Vec<_> = (0..top_k).map(|s| mk(&format!("d{s}"), d_model)).collect();
    let slots_b = mk("slots", top_k * d_model);
    let moe_out_b = mk("mo", d_model);
    let out_b = mk("out", d_model);
    let read_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: (d_model * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let ids_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ids_read"),
        size: (top_k * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("moe_layer"),
    });
    moe_router_chained(
        &ctx,
        &pipes,
        &mut enc,
        &hidden_b,
        router_scale.as_ref(),
        &dummy,
        &router_w,
        &ids_b,
        &w_b,
        d_model,
        n_experts,
        top_k,
        eps,
    );
    rmsnorm_chained(
        &ctx,
        &pipes,
        &mut enc,
        &hidden_b,
        Some(&pre2_b),
        &dummy,
        &moe_x_b,
        d_model,
        eps,
    );
    for s in 0..top_k {
        moe_expert_matmul_chained(
            &ctx,
            &pipes,
            &mut enc,
            &gate_up,
            &ids_b,
            &moe_x_b,
            &gu_bufs[s],
            d_model,
            2 * e_ffn,
            s,
            gate_up_dt,
        )
        .unwrap();
        moe_geglu_halves_chained(&ctx, &pipes, &mut enc, &gu_bufs[s], &act_bufs[s], e_ffn);
        moe_expert_matmul_chained(
            &ctx,
            &pipes,
            &mut enc,
            &down_w,
            &ids_b,
            &act_bufs[s],
            &down_bufs[s],
            e_ffn,
            d_model,
            s,
            down_dt,
        )
        .unwrap();
        enc.copy_buffer_to_buffer(
            &down_bufs[s],
            0,
            &slots_b,
            (s * d_model * 4) as u64,
            (d_model * 4) as u64,
        );
    }
    moe_combine_chained(
        &ctx,
        &pipes,
        &mut enc,
        &slots_b,
        &ids_b,
        &w_b,
        down_scale.as_ref(),
        &dummy,
        &moe_out_b,
        d_model,
        top_k,
    );
    rmsnorm_chained(
        &ctx,
        &pipes,
        &mut enc,
        &moe_out_b,
        Some(&post2_b),
        &dummy,
        &out_b,
        d_model,
        eps,
    );
    enc.copy_buffer_to_buffer(&out_b, 0, &read_b, 0, (d_model * 4) as u64);
    enc.copy_buffer_to_buffer(&ids_b, 0, &ids_read, 0, (top_k * 4) as u64);
    queue.submit(Some(enc.finish()));

    let map = |b: &wgpu::Buffer| -> Vec<f32> {
        let slice = b.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        b.unmap();
        v
    };
    let gpu_ids: Vec<u32> = map(&ids_read).iter().map(|f| f.to_bits()).collect();
    let gpu_out = map(&read_b);

    println!("gpu selected experts: {gpu_ids:?}");
    let cpu_ids: Vec<u32> = selected.iter().map(|&(e, _)| e as u32).collect();
    assert_eq!(gpu_ids, cpu_ids, "expert selection mismatch");

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..d_model {
        let abs = (gpu_out[i] - cpu_out[i]).abs();
        let rel = if cpu_out[i].abs() > 1e-3 {
            abs / cpu_out[i].abs()
        } else {
            0.0
        };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    println!("layer {layer} MoE FFN sub-path: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    assert!(max_abs < 1e-2, "GPU/CPU disagreement: max_abs={max_abs}");
    println!("PASS");
    ExitCode::SUCCESS
}
