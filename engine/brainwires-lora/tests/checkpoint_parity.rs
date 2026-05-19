//! Integration test: gradient checkpointing produces equivalent LoRA
//! gradients to the standard per-layer-scratch path.
//!
//! With `gradient_checkpointing=true`, `TrainingScratch` allocates one
//! shared `LayerActivations` (cloned into every layer slot) instead of
//! N independent sets, and `backward_step` replays each layer's
//! forward into the shared buffers before reading them. Replay walks
//! the same fp32 matmuls in the same order, so the gradients should
//! match the non-checkpointed run within tight tolerance — small
//! drift only from floating-point non-associativity of buffer-write
//! ordering.
//!
//! Tolerance: max-abs ≤ 5e-4 per LoRA gradient element. Loose enough
//! to absorb FP reordering, tight enough that an actual divergence
//! (e.g. shared scratch not being recomputed before read) shows up.
//!
//! Skipped (with a printed message) when `RULLAMA_TEST_GGUF` is unset.

use std::env;
use std::fs;

use rullama::api::Model;
use rullama_finetune::TrainingSession;
use rullama_finetune::lora::LoraState;
use rullama_finetune::shared::config::{LoraConfig, TrainingHyperparams};

const PROMPT: &str = "The quick brown fox";
const TARGET: &str = " jumps";

#[test]
fn checkpoint_parity_matches_standard_grads() {
    let gguf_path = match env::var("RULLAMA_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[skip] checkpoint_parity: RULLAMA_TEST_GGUF unset \
                 (point at a gemma4 e2b/e4b blob to enable)"
            );
            return;
        }
    };

    pollster::block_on(async move {
        let gguf = fs::read(&gguf_path).expect("read gguf");

        let std_grads = run_one_step(&gguf, false).await;
        let ckpt_grads = run_one_step(&gguf, true).await;

        assert_eq!(
            std_grads.len(),
            ckpt_grads.len(),
            "different LoRA layer count"
        );

        // Compare per-layer per-LoRA gradient tensors.
        let mut max_diff = 0.0f32;
        let mut total_compared = 0usize;
        for ((sk, sv), (ck, cv)) in std_grads.iter().zip(ckpt_grads.iter()) {
            assert_eq!(sk, ck, "LoraKey mismatch: std={sk:?} ckpt={ck:?}");
            assert_eq!(sv.da.len(), cv.da.len(), "dA len mismatch on {sk:?}");
            assert_eq!(sv.db.len(), cv.db.len(), "dB len mismatch on {sk:?}");
            for (i, (a, b)) in sv.da.iter().zip(cv.da.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
                total_compared += 1;
                assert!(
                    d < 5e-4,
                    "dA diff at {sk:?}[{i}]: std={a} ckpt={b} diff={d}",
                );
            }
            for (i, (a, b)) in sv.db.iter().zip(cv.db.iter()).enumerate() {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
                total_compared += 1;
                assert!(
                    d < 5e-4,
                    "dB diff at {sk:?}[{i}]: std={a} ckpt={b} diff={d}",
                );
            }
        }
        eprintln!(
            "[checkpoint_parity] compared {total_compared} grad elements; max_diff={max_diff:.3e}"
        );
    });
}

struct GradPair {
    da: Vec<f32>,
    db: Vec<f32>,
}

async fn run_one_step(
    gguf: &[u8],
    gradient_checkpointing: bool,
) -> Vec<(rullama_finetune::lora::LoraKey, GradPair)> {
    let model = Model::load_native(gguf.to_vec()).await.expect("load model");
    let input_tokens = model.encode_tokens(PROMPT);
    let target_tokens = model.encode_tokens(TARGET);
    let target_id = *target_tokens.first().expect("target tokenised");

    let lora_cfg = LoraConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec!["attn_q".into(), "attn_o".into()],
    };
    // Use the same seed for LoRA A init in both runs so the only
    // input difference is the scratch layout.
    let hp = TrainingHyperparams {
        learning_rate: 1e-3,
        weight_decay: 0.0,
        max_seq_len: input_tokens.len().max(32),
        seed: 0xC0FFEE,
        gradient_checkpointing,
        max_grad_norm: 0.0,
        ..Default::default()
    };

    let mut session = TrainingSession::new(model, lora_cfg, hp).expect("session");

    // forward_backward leaves the freshly-accumulated gradients in
    // the LoRA buffers (no Adam step yet). Read them out — that's the
    // signal we compare.
    session.zero_grads();
    session
        .forward_backward(&input_tokens, target_id)
        .await
        .expect("forward_backward");

    read_lora_grads(session.lora_state()).await
}

async fn read_lora_grads(state: &LoraState) -> Vec<(rullama_finetune::lora::LoraKey, GradPair)> {
    let ctx = state.ctx();
    let mut out = Vec::new();
    for (key, layer) in state.iter() {
        let da = read_buf_f32(ctx, &layer.da, layer.a_len()).await;
        let db = read_buf_f32(ctx, &layer.db, layer.b_len()).await;
        out.push((key.clone(), GradPair { da, db }));
    }
    out
}

async fn read_buf_f32(ctx: &rullama::backend::WgpuCtx, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let bytes = (n * 4) as u64;
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("checkpoint_parity.read"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("checkpoint_parity.read.enc"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let v: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    read_buf.unmap();
    v
}
