//! Integration test: short overfit-one run must show clear loss drop.
//!
//! The 200-step ≥90% drop acceptance lives in the `overfit_one` example
//! and isn't appropriate for `cargo test` (multi-minute runtime). This
//! smoke runs 20 steps and asserts ≥50% drop — fast enough for CI,
//! strong enough to catch a broken backward path.
//!
//! Requires a GGUF blob. Skipped (with a printed message) when
//! `RULLAMA_TEST_GGUF` is unset, so the test suite stays green on
//! machines without a model checked out.

use std::env;
use std::fs;

use rullama::api::Model;
use rullama_finetune::shared::config::{LoraConfig, TrainingHyperparams};
use rullama_finetune::TrainingSession;

const PROMPT: &str = "The quick brown fox";
const TARGET: &str = " jumps";
const N_STEPS: u32 = 20;
const REQUIRED_DROP: f32 = 0.50;

#[test]
fn overfit_one_smoke_drops_loss_by_half() {
    let gguf_path = match env::var("RULLAMA_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[skip] overfit_one_smoke: RULLAMA_TEST_GGUF unset \
                 (point at a gemma4 e2b/e4b blob to enable)"
            );
            return;
        }
    };

    pollster::block_on(async move {
        let bytes = fs::read(&gguf_path).expect("read gguf");
        let model = Model::load_native(bytes).await.expect("load model");
        let input_tokens = model.encode_tokens(PROMPT);
        let target_tokens = model.encode_tokens(TARGET);
        let target_id = *target_tokens.first().expect("target tokenised");

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
        hp.max_seq_len = input_tokens.len().max(32);
        hp.seed = 0xC0FFEE;

        let mut session = TrainingSession::new(model, lora_cfg, hp).expect("session");

        let mut first_loss = f32::NAN;
        let mut last_loss = f32::NAN;
        for step in 1..=N_STEPS {
            let loss = session.step(&input_tokens, target_id).await
                .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
            if step == 1 { first_loss = loss; }
            last_loss = loss;
        }

        assert!(first_loss.is_finite() && last_loss.is_finite(),
            "non-finite loss: first={first_loss} last={last_loss}");
        let ratio = last_loss / first_loss.max(1e-6);
        assert!(ratio <= 1.0 - REQUIRED_DROP,
            "loss did not drop enough: first={first_loss:.4} last={last_loss:.4} \
             ratio={ratio:.3} (need ≤ {:.3})", 1.0 - REQUIRED_DROP);
    });
}
