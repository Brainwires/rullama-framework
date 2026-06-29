//! Integration test: truncated backward (training only the top layers)
//! must still drop loss meaningfully, even if not as fast as full backward.
//!
//! The full backward proves `overfit_one_smoke` halves the loss in 20
//! steps. Truncated backward at `floor = n_layers - 5` (last 5 layers
//! only) is the iPhone-safe configuration. The adapter can no longer
//! re-shape the bottom of the network, so we expect a smaller but
//! still positive drop. Acceptance bar: 20% relative drop in 20 steps.
//! Anything less suggests the floor gate is broken (e.g. accidentally
//! freezing all layers, or seeding garbage gradients).
//!
//! Same skip-on-missing-fixture pattern as the other smoke tests.

use std::env;
use std::fs;

use brainwires_engine::api::Model;
use brainwires_lora::TrainingSession;
use brainwires_lora::shared::config::{LoraConfig, TrainingHyperparams};

const PROMPT: &str = "The quick brown fox";
const TARGET: &str = " jumps";
const N_STEPS: u32 = 20;
// Looser acceptance than the full-backward smoke (which requires 50%)
// because only the top 5 layers have trainable LoRA grads. Empirically
// the top layers carry most task-specific signal, so 20% is reachable
// in 20 steps; if this fails it's a real regression in the floor logic.
const REQUIRED_DROP: f32 = 0.20;

#[test]
fn truncated_backward_smoke_still_drops_loss() {
    let gguf_path = match env::var("RULLAMA_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[skip] truncated_backward_smoke: RULLAMA_TEST_GGUF unset \
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
                "attn_q".into(),
                "attn_k".into(),
                "attn_v".into(),
                "attn_o".into(),
            ],
            target_layers: None,
        };
        // Floor 30 leaves the top 5 layers trainable on gemma4:e2b
        // (35 layers). The backward path saturate-clamps floor to
        // n_layers so over-large values are harmless. For other model
        // sizes the test runs with whatever effective top-N this
        // produces.
        let floor: u32 = 30;
        let hp = TrainingHyperparams {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            max_seq_len: input_tokens.len().max(32),
            seed: 0xC0FFEE,
            backward_layer_floor: floor,
            ..Default::default()
        };

        let mut session = TrainingSession::new(model, lora_cfg, hp).expect("session");

        let mut first_loss = f32::NAN;
        let mut last_loss = f32::NAN;
        for step in 1..=N_STEPS {
            let loss = session
                .step(&input_tokens, target_id)
                .await
                .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
            if step == 1 {
                first_loss = loss;
            }
            last_loss = loss;
        }

        assert!(
            first_loss.is_finite() && last_loss.is_finite(),
            "non-finite loss: first={first_loss} last={last_loss}"
        );
        let ratio = last_loss / first_loss.max(1e-6);
        assert!(
            ratio <= 1.0 - REQUIRED_DROP,
            "truncated backward (floor={floor}) did not drop loss enough: \
             first={first_loss:.4} last={last_loss:.4} ratio={ratio:.3} \
             (need ≤ {:.3})",
            1.0 - REQUIRED_DROP
        );
        eprintln!(
            "truncated_backward_smoke: floor={floor} \
             first={first_loss:.4} last={last_loss:.4} ratio={ratio:.3}"
        );
    });
}
