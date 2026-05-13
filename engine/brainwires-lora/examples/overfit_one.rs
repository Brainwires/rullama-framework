//! Overfit a single (prompt, target) pair with rank-4 attention LoRAs.
//!
//! M0 acceptance test: run 200 Adam steps on the same example and
//! verify that the cross-entropy loss drops by ≥ 90%. Loss should
//! start around `log(vocab_size)` ≈ 12.5 for the base Gemma 4 e2b
//! (262 144 vocab) and decay smoothly as the LoRAs absorb the
//! prompt → target association.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-finetune --example overfit_one --release -- \
//!     /path/to/gemma4-e2b.gguf
//! ```
//!
//! On the dev machine the GGUF lives at
//! `/Users/$USER/.ollama/models/blobs/sha256-…` — `ollama show
//! --modelfile gemma4:e2b` reports the path.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::Model;
use rullama_finetune::shared::config::{LoraConfig, TrainingHyperparams};
use rullama_finetune::TrainingSession;

type BoxError = Box<dyn Error + Send + Sync>;

const DEFAULT_N_STEPS: u32 = 200;
const PROMPT: &str = "The quick brown fox";
const TARGET: &str = " jumps";

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let gguf_path: PathBuf = env::args()
        .nth(1)
        .ok_or_else(|| -> BoxError { "usage: overfit_one <gguf-path>".into() })?
        .into();
    // `RULLAMA_OVERFIT_STEPS=<n>` lets the smoke test run a short
    // session (e.g. 2 steps for a "does it crash?" check) without a
    // rebuild. Defaults to the M0 acceptance target of 200.
    let n_steps: u32 = env::var("RULLAMA_OVERFIT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_STEPS);
    let assert_drop: bool = n_steps >= DEFAULT_N_STEPS / 2;
    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[load] model ready (vocab={})", model.vocab_size_native());

    // Tokenize the example.
    let input_tokens = model.encode_tokens(PROMPT);
    let target_tokens = model.encode_tokens(TARGET);
    let target_id = *target_tokens.first()
        .ok_or_else(|| -> BoxError { "target tokenized to zero tokens".into() })?;
    eprintln!(
        "[encode] prompt → {} toks, target first-id = {}",
        input_tokens.len(),
        target_id,
    );

    // Rank-4 LoRA over q/k/v/o on every layer.
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
    };
    let mut hp = TrainingHyperparams::default();
    hp.learning_rate = 1e-3;
    hp.weight_decay = 0.0;
    hp.max_seq_len = input_tokens.len().max(32) as usize;
    hp.seed = 0xC0FFEE;
    let mut session = TrainingSession::new(model, lora_cfg, hp)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!(
        "[init] training {} LoRA parameters across all attn projections",
        session.parameter_count()
    );

    let mut first_loss: Option<f32> = None;
    let mut last_loss = f32::NAN;
    for step in 1..=n_steps {
        let loss = session
            .step(&input_tokens, target_id)
            .await
            .map_err(|e| -> BoxError { format!("step {step}: {e:?}").into() })?;
        if first_loss.is_none() {
            first_loss = Some(loss);
        }
        last_loss = loss;
        if step <= 5 || step % 10 == 0 || step == n_steps {
            eprintln!("[step {step:>3}] loss = {loss:.4}");
        }
    }

    let l0 = first_loss.unwrap();
    let drop_pct = (l0 - last_loss) / l0.max(1e-6) * 100.0;
    eprintln!(
        "[done] start={l0:.4}, end={last_loss:.4}, drop={drop_pct:.1}%"
    );

    if !assert_drop {
        // Smoke test with fewer steps — don't enforce the 90% drop
        // assertion. We only report.
        eprintln!("[smoke] short run ({n_steps} steps); skipping drop-assert");
        return Ok(());
    }
    if drop_pct >= 90.0 {
        eprintln!("[PASS] loss drop ≥ 90%");
        Ok(())
    } else {
        Err(format!(
            "loss drop only {drop_pct:.1}% (target ≥ 90%) — backward may be incorrect"
        ).into())
    }
}
