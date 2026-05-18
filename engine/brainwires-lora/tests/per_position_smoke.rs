//! Integration test: PerPosition loss objective drops measurably in a
//! few steps on a single example. Regression gate for
//! `TrainingSession::forward_backward_per_position`.
//!
//! The threshold (≥30% drop in 3 steps) is intentionally generous —
//! the goal is "did the backward path produce useful gradients?",
//! not "did we hit the same number we hit last time on a different
//! machine." A broken backward NaNs or stays flat; anything ≥30% means
//! the per-position chain is working.
//!
//! Skipped (with a printed message) when `RULLAMA_TEST_GGUF` is unset.

use std::env;
use std::fs;

use rullama::api::Model;
use rullama_finetune::shared::config::{LoraConfig, LossMode, TrainingHyperparams};
use rullama_finetune::TrainingSession;

const PROMPT: &str = "The quick brown fox jumps over";
const TARGET: &str = " the lazy dog";
const N_STEPS: u32 = 3;
const REQUIRED_DROP: f32 = 0.30;

#[test]
fn per_position_smoke_drops_loss_sharply() {
    let gguf_path = match env::var("RULLAMA_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[skip] per_position_smoke: RULLAMA_TEST_GGUF unset \
                 (point at a gemma4 e2b/e4b blob to enable)"
            );
            return;
        }
    };

    pollster::block_on(async move {
        let bytes = fs::read(&gguf_path).expect("read gguf");
        let model = Model::load_native(bytes).await.expect("load model");

        // Build input + targets: prompt tokens have target u32::MAX
        // (masked); completion tokens predict the next token.
        let prompt_tokens = model.encode_tokens(PROMPT);
        let completion_tokens = model.encode_tokens(TARGET);
        let mut input_ids = prompt_tokens.clone();
        input_ids.extend_from_slice(&completion_tokens);

        let mut targets: Vec<u32> = input_ids.iter().enumerate().map(|(i, _)| {
            // Predict-next: target[i] = input[i+1]; last position has no
            // next-token target → mask.
            if i + 1 < input_ids.len() && i + 1 >= prompt_tokens.len() {
                input_ids[i + 1]
            } else {
                u32::MAX
            }
        }).collect();
        // Mask the final position (no next token).
        if let Some(last) = targets.last_mut() { *last = u32::MAX; }

        let lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![
                "attn_q".into(), "attn_k".into(),
                "attn_v".into(), "attn_o".into(),
            ],
        };
        let mut hp = TrainingHyperparams::default();
        hp.learning_rate = 1e-3;
        hp.weight_decay = 0.0;
        hp.max_seq_len = input_ids.len().max(32);
        hp.seed = 0xC0FFEE;
        hp.loss_mode = LossMode::PerPosition;

        let mut session = TrainingSession::new(model, lora_cfg, hp).expect("session");

        let mut first_loss = f32::NAN;
        let mut last_loss = f32::NAN;
        for step in 1..=N_STEPS {
            let loss = session.step_per_position(&input_ids, &targets).await
                .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
            if step == 1 { first_loss = loss; }
            last_loss = loss;
        }

        assert!(first_loss.is_finite() && last_loss.is_finite(),
            "non-finite loss: first={first_loss} last={last_loss}");
        let ratio = last_loss / first_loss.max(1e-6);
        assert!(ratio <= 1.0 - REQUIRED_DROP,
            "PerPosition loss did not drop enough: first={first_loss:.4} \
             last={last_loss:.4} ratio={ratio:.3} (need ≤ {:.3})",
            1.0 - REQUIRED_DROP);
    });
}
