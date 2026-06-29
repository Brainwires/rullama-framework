//! Reproduce the iPhone "Memory-tight preset" training crash NATIVELY.
//!
//! The iPhone fine-tune crash happens only with the Memory-tight preset
//! config, which `overfit_one` (rank-4, q/k/v/o, NextToken loss, full
//! backward, no checkpointing) never exercises. This example runs the
//! EXACT Memory-tight preset — rank 1, attn_q+attn_v only, PerPosition
//! loss, backward_layer_floor=25, gradient_checkpointing=true — across
//! several multi-token examples, to see whether the bug reproduces off
//! the iOS memory ceiling (i.e. is a config/logic bug, not jetsam).
//!
//! Usage:
//!   cargo run -p rullama-finetune --example mem_tight_repro --release -- \
//!       /path/to/gemma4-e2b.gguf
//! Env: RULLAMA_TRACE_MEM=1 to dump the GPU-allocation ledger.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::Model;
use rullama_finetune::TrainingSession;
use rullama_finetune::shared::config::{LoraConfig, LossMode, TrainingHyperparams};

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let gguf_path: PathBuf = env::args()
        .nth(1)
        .ok_or_else(|| -> BoxError { "usage: mem_tight_repro <gguf-path>".into() })?
        .into();
    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[load] model ready (vocab={})", model.vocab_size_native());

    // EXACT Memory-tight (iPhone-safe) preset.
    let lora_cfg = LoraConfig {
        rank: 1,
        alpha: 2.0,
        dropout: 0.0,
        target_modules: vec!["attn_q".into(), "attn_v".into()],
        target_layers: None,
    };
    let hp = TrainingHyperparams {
        learning_rate: 3e-4,
        weight_decay: 0.0,
        max_seq_len: 32,
        seed: 12648430,
        max_grad_norm: 1.0,
        loss_mode: LossMode::PerPosition,
        gradient_checkpointing: true,
        backward_layer_floor: 25,
        // The example tests the iPhone-Memory-tight code path
        // (per-layer MeBP destroy, tiled outproj, kernel warmup,
        // per-step yields). On native the yields are no-ops, but
        // MeBP destroy + tiled outproj exercise the full stack so
        // we catch any regression to the iPhone path natively.
        memory_tight: true,
        ..Default::default()
    };
    eprintln!(
        "[cfg] Memory-tight: rank=1 alpha=2 targets=attn_q+attn_v \
         loss=PerPosition ckpt=true floor=25 seq=32"
    );

    let mut session = TrainingSession::new(model, lora_cfg, hp)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[init] params={}", session.parameter_count());

    // Multi-token examples (mirrors the 3-example iPhone dataset:
    // ~21, ~18, ~12 tokens). The session already owns the model so we
    // can't tokenize real text here; the crash is about the backward
    // memory/compute pattern, not the specific token values, so use
    // fixed in-vocab token sequences of the right lengths. PerPosition
    // needs input_ids + per-position targets (next-token shift).
    let seqs: Vec<Vec<u32>> = vec![
        (0..21).map(|i| 100 + i as u32).collect(),
        (0..18).map(|i| 200 + i as u32).collect(),
        (0..12).map(|i| 300 + i as u32).collect(),
    ];

    let mut step = 0;
    for epoch in 0..3 {
        for seq in &seqs {
            step += 1;
            // PerPosition targets: next-token shift, last position targets
            // itself (matches the worker's construction).
            let mut targets = vec![0u32; seq.len()];
            let last = seq.len() - 1;
            targets[..last].copy_from_slice(&seq[1..]);
            targets[last] = seq[last];
            eprintln!(
                "[step {step}] (epoch {epoch}) per_position inputLen={} …",
                seq.len()
            );
            match session.step_per_position(seq, &targets).await {
                Ok(loss) => eprintln!("[step {step}] done loss={loss:.4}"),
                Err(e) => {
                    eprintln!("[step {step}] ERROR: {e:?}");
                    return Err(format!("step {step} failed: {e:?}").into());
                }
            }
        }
    }
    eprintln!("[done] all steps completed — no native crash with Memory-tight config");
    Ok(())
}
