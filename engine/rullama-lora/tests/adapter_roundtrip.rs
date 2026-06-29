//! Integration test: LoRA adapter save → load round-trip is bit-identical
//! (f32) and within f16 quantisation tolerance (mixed_precision path).
//!
//! Promoted from `session.rs`'s in-module tests so the gate runs from
//! `cargo test -p rullama-lora` against the published API without
//! requiring a GGUF blob. The full pipeline gate ("save adapter from a
//! real training run, reload into a fresh Model, assert logits match")
//! lives in `examples/eval_adapter.rs` because it needs a model.

use std::sync::Arc;

use rullama_engine::backend::WgpuCtx;
use rullama_lora::load_adapter_into_state;
use rullama_lora::lora::{LoraKey, LoraState};
use safetensors::tensor::{Dtype, TensorView};

const A_VALS: [f32; 16] = [
    0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000, 1.125, 1.250, 1.375, 1.500,
    1.625, 1.750, 1.875,
];
const B_VALS: [f32; 8] = [0.50, 0.25, 0.00, -0.25, -0.50, -0.75, -1.00, -1.25];

#[test]
fn adapter_f32_round_trip_is_bit_identical() {
    let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));

    let mut state = LoraState::new(Arc::clone(&ctx));
    state
        .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
        .unwrap();
    state
        .insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 2)
        .unwrap();

    {
        let layer = state.get(&LoraKey::new(0, "attn_q")).unwrap();
        ctx.queue
            .write_buffer(&layer.a, 0, bytemuck::cast_slice(&A_VALS));
        ctx.queue
            .write_buffer(&layer.b, 0, bytemuck::cast_slice(&B_VALS));
    }
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    write_f32_pair(&ctx, &state, &path);

    let mut state_b = LoraState::new(Arc::clone(&ctx));
    state_b
        .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 99)
        .unwrap();
    state_b
        .insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 100)
        .unwrap();
    let loaded = load_adapter_into_state(&mut state_b, &path).unwrap();
    assert_eq!(loaded, 4, "expected to load 4 tensors (A+B for q and k)");

    let layer_q = state_b.get(&LoraKey::new(0, "attn_q")).unwrap();
    let a_round = read_buf_f32(&ctx, &layer_q.a, 16);
    let b_round = read_buf_f32(&ctx, &layer_q.b, 8);
    assert_eq!(
        a_round,
        A_VALS.to_vec(),
        "f32 A round-trip must be bit-identical"
    );
    assert_eq!(
        b_round,
        B_VALS.to_vec(),
        "f32 B round-trip must be bit-identical"
    );
}

#[test]
fn adapter_f16_round_trip_within_quantisation_tolerance() {
    let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));

    let mut state = LoraState::new(Arc::clone(&ctx));
    state
        .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
        .unwrap();

    {
        let layer = state.get(&LoraKey::new(0, "attn_q")).unwrap();
        ctx.queue
            .write_buffer(&layer.a, 0, bytemuck::cast_slice(&A_VALS));
        ctx.queue
            .write_buffer(&layer.b, 0, bytemuck::cast_slice(&B_VALS));
    }
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    write_f16_pair(&ctx, &state, &path);

    let mut state_b = LoraState::new(Arc::clone(&ctx));
    state_b
        .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 99)
        .unwrap();
    let loaded = load_adapter_into_state(&mut state_b, &path).unwrap();
    assert_eq!(loaded, 2, "expected to load 2 f16 tensors");

    let layer_q = state_b.get(&LoraKey::new(0, "attn_q")).unwrap();
    let a_round = read_buf_f32(&ctx, &layer_q.a, 16);
    let b_round = read_buf_f32(&ctx, &layer_q.b, 8);
    for (orig, round) in A_VALS.iter().zip(a_round.iter()) {
        assert!(
            (orig - round).abs() < 1e-3,
            "A f16 round-trip drift too large: {orig} vs {round}"
        );
    }
    for (orig, round) in B_VALS.iter().zip(b_round.iter()) {
        assert!(
            (orig - round).abs() < 1e-3,
            "B f16 round-trip drift too large: {orig} vs {round}"
        );
    }
}

#[test]
fn adapter_load_returns_zero_on_unknown_key() {
    // A loader given a safetensors with no `lora.blk.*` entries should
    // not blow up; it returns 0 loaded tensors and leaves the state
    // alone.
    let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
    let mut state = LoraState::new(Arc::clone(&ctx));
    state
        .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let payload: Vec<f32> = vec![0.0; 16];
    let bytes = bytemuck::cast_slice::<f32, u8>(&payload).to_vec();
    let view = TensorView::new(Dtype::F32, vec![2usize, 8usize], &bytes).unwrap();
    let mut views: std::collections::HashMap<&str, TensorView<'_>> =
        std::collections::HashMap::new();
    views.insert("unrelated.tensor", view);
    safetensors::serialize_to_file(&views, &None, &path).unwrap();

    let loaded = load_adapter_into_state(&mut state, &path).unwrap();
    assert_eq!(loaded, 0, "no lora.blk.* tensors → nothing loaded");
}

fn write_f32_pair(ctx: &WgpuCtx, state: &LoraState, path: &std::path::Path) {
    let q = state.get(&LoraKey::new(0, "attn_q")).unwrap();
    let k = state.get(&LoraKey::new(0, "attn_k")).unwrap();
    let a_q = read_buf_f32(ctx, &q.a, 16);
    let b_q = read_buf_f32(ctx, &q.b, 8);
    let a_k = read_buf_f32(ctx, &k.a, 16);
    let b_k = read_buf_f32(ctx, &k.b, 8);
    let a_q_bytes = bytemuck::cast_slice::<f32, u8>(&a_q).to_vec();
    let b_q_bytes = bytemuck::cast_slice::<f32, u8>(&b_q).to_vec();
    let a_k_bytes = bytemuck::cast_slice::<f32, u8>(&a_k).to_vec();
    let b_k_bytes = bytemuck::cast_slice::<f32, u8>(&b_k).to_vec();
    let v_a_q = TensorView::new(Dtype::F32, vec![2usize, 8usize], &a_q_bytes).unwrap();
    let v_b_q = TensorView::new(Dtype::F32, vec![4usize, 2usize], &b_q_bytes).unwrap();
    let v_a_k = TensorView::new(Dtype::F32, vec![2usize, 8usize], &a_k_bytes).unwrap();
    let v_b_k = TensorView::new(Dtype::F32, vec![4usize, 2usize], &b_k_bytes).unwrap();
    let mut views: std::collections::HashMap<&str, TensorView<'_>> =
        std::collections::HashMap::new();
    views.insert("lora.blk.0.attn_q.A", v_a_q);
    views.insert("lora.blk.0.attn_q.B", v_b_q);
    views.insert("lora.blk.0.attn_k.A", v_a_k);
    views.insert("lora.blk.0.attn_k.B", v_b_k);
    safetensors::serialize_to_file(&views, &None, path).unwrap();
}

fn write_f16_pair(ctx: &WgpuCtx, state: &LoraState, path: &std::path::Path) {
    let q = state.get(&LoraKey::new(0, "attn_q")).unwrap();
    let a_q = read_buf_f32(ctx, &q.a, 16);
    let b_q = read_buf_f32(ctx, &q.b, 8);
    let a_h: Vec<half::f16> = a_q.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b_h: Vec<half::f16> = b_q.iter().map(|&x| half::f16::from_f32(x)).collect();
    let a_bytes = bytemuck::cast_slice::<half::f16, u8>(&a_h).to_vec();
    let b_bytes = bytemuck::cast_slice::<half::f16, u8>(&b_h).to_vec();
    let v_a = TensorView::new(Dtype::F16, vec![2usize, 8usize], &a_bytes).unwrap();
    let v_b = TensorView::new(Dtype::F16, vec![4usize, 2usize], &b_bytes).unwrap();
    let mut views: std::collections::HashMap<&str, TensorView<'_>> =
        std::collections::HashMap::new();
    views.insert("lora.blk.0.attn_q.A", v_a);
    views.insert("lora.blk.0.attn_q.B", v_b);
    safetensors::serialize_to_file(&views, &None, path).unwrap();
}

fn read_buf_f32(ctx: &WgpuCtx, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let bytes = (n * 4) as u64;
    let read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test.read"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.read.enc"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &read, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let v: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    read.unmap();
    v
}
