//! Train rullama-finetune on a JSONL dataset.
//!
//! Surface: `train_jsonl <gguf> <jsonl>`. Reads `(prompt, completion)`
//! examples, tokenizes via the model's BPE, and runs LoRA SGD over
//! `attn_q`/`k`/`v`/`o` with **NextToken** cross-entropy on the first
//! completion token. Gradient accumulation is honored via
//! `RULLAMA_TRAIN_ACCUM` (default 1) — within each optimizer step the
//! loop calls `zero_grads → N × forward_backward → optimizer_step`.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-finetune --example train_jsonl --release -- \
//!     /path/to/gemma4-e2b.gguf \
//!     crates/rullama-finetune/examples/data/echo.jsonl
//! ```
//!
//! Env knobs (all optional):
//!   - `RULLAMA_TRAIN_STEPS`     — optimizer steps                  (default 100)
//!   - `RULLAMA_TRAIN_LR`        — learning rate                    (default 1e-3)
//!   - `RULLAMA_TRAIN_ACCUM`     — gradient accumulation steps      (default 1)
//!   - `RULLAMA_TRAIN_RANK`      — LoRA rank                        (default 8)
//!   - `RULLAMA_TRAIN_ALPHA`     — LoRA alpha                       (default 16)
//!   - `RULLAMA_TRAIN_SEED`      — RNG seed for LoRA A init         (default 0xC0FFEE)
//!   - `RULLAMA_TRAIN_LOG_EVERY` — print every N optimizer steps    (default 5)
//!   - `RULLAMA_ADAPTER_PATH`    — write adapter here when done     (default unset)
//!   - plus the backward-side knobs honored by `Forward::backward_step`
//!     (`RULLAMA_CLIP_DHIDDEN`, `RULLAMA_DEBUG_GRADS`,
//!     `RULLAMA_TRACE_DHIDDEN`).
//!
//! M0 loss surface is NextToken — the trainer predicts the *first*
//! completion token given the full prompt. PerPosition (predict every
//! position of the completion, averaged) is the M1.4 follow-up.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::Model;
use rullama_finetune::dataset_loader::TrainingDataset;
use rullama_finetune::shared::config::{LoraConfig, TrainingHyperparams};
use rullama_finetune::TrainingSession;

type BoxError = Box<dyn Error + Send + Sync>;

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "usage: train_jsonl <gguf-path> <jsonl-path>".into() })?
        .into();
    let jsonl_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "usage: train_jsonl <gguf-path> <jsonl-path>".into() })?
        .into();

    let n_steps = env_u32("RULLAMA_TRAIN_STEPS", 100);
    let lr = env_f32("RULLAMA_TRAIN_LR", 1e-3) as f64;
    let accum = env_u32("RULLAMA_TRAIN_ACCUM", 1).max(1);
    let rank = env_u32("RULLAMA_TRAIN_RANK", 8);
    let alpha = env_f32("RULLAMA_TRAIN_ALPHA", 16.0);
    let seed = env_u64("RULLAMA_TRAIN_SEED", 0xC0FFEE);
    let log_every = env_u32("RULLAMA_TRAIN_LOG_EVERY", 5).max(1);

    eprintln!("[load] gguf = {}", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[load] model ready (vocab={})", model.vocab_size_native());

    eprintln!("[load] dataset = {}", jsonl_path.display());
    let dataset = TrainingDataset::load_jsonl(&jsonl_path)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[load] {} examples in dataset", dataset.len());

    // Tokenize up front so the training loop is just forward+backward,
    // no string handling. Skip examples whose prompt or completion
    // tokenizes to nothing.
    let mut tokenized: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut max_prompt_len = 0usize;
    for ex in &dataset.examples {
        let prompt = model.encode_tokens(&ex.prompt);
        let completion = model.encode_tokens(&ex.completion);
        if prompt.is_empty() || completion.is_empty() {
            continue;
        }
        max_prompt_len = max_prompt_len.max(prompt.len());
        tokenized.push((prompt, completion[0]));
    }
    if tokenized.is_empty() {
        return Err("dataset has no usable examples after tokenization".into());
    }
    eprintln!(
        "[tok] {} examples kept, longest prompt = {} tokens",
        tokenized.len(),
        max_prompt_len,
    );

    let lora_cfg = LoraConfig {
        rank,
        alpha,
        dropout: 0.0,
        target_modules: vec![
            "attn_q".into(),
            "attn_k".into(),
            "attn_v".into(),
            "attn_o".into(),
        ],
    };
    let mut hp = TrainingHyperparams::default();
    hp.learning_rate = lr;
    hp.weight_decay = 0.0;
    hp.max_seq_len = max_prompt_len.max(32);
    hp.seed = seed;
    eprintln!(
        "[hp] lr={lr:.3e} rank={rank} alpha={alpha} accum={accum} steps={n_steps}"
    );
    let mut session = TrainingSession::new(model, lora_cfg, hp)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!(
        "[init] training {} LoRA parameters",
        session.parameter_count()
    );

    // Training loop. One outer iteration = one optimizer.step() =
    // `accum` calls of forward_backward(). Examples are drawn
    // round-robin in dataset order.
    let mut first_loss: Option<f32> = None;
    let mut last_loss = f32::NAN;
    let mut idx = 0usize;
    for step in 1..=n_steps {
        session.zero_grads();
        let mut accum_loss = 0.0f32;
        for _ in 0..accum {
            let (input_ids, target_id) = &tokenized[idx % tokenized.len()];
            idx += 1;
            let loss = session
                .forward_backward(input_ids, *target_id)
                .await
                .map_err(|e| -> BoxError { format!("step {step}: {e:?}").into() })?;
            accum_loss += loss;
        }
        session.optimizer_step();
        let avg_loss = accum_loss / accum as f32;
        if first_loss.is_none() {
            first_loss = Some(avg_loss);
        }
        last_loss = avg_loss;
        if step == 1 || step % log_every == 0 || step == n_steps {
            eprintln!("[step {step:>4}/{n_steps}] loss = {avg_loss:.4}");
        }
    }

    let l0 = first_loss.unwrap();
    let drop_pct = (l0 - last_loss) / l0.max(1e-6) * 100.0;
    eprintln!("[done] start={l0:.4}, end={last_loss:.4}, drop={drop_pct:.1}%");

    if let Ok(path_s) = env::var("RULLAMA_ADAPTER_PATH") {
        let path = PathBuf::from(&path_s);
        session
            .save_adapter(&path)
            .await
            .map_err(|e| -> BoxError { format!("save_adapter: {e:?}").into() })?;
        let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!("[save] adapter → {} ({} bytes)", path.display(), bytes);
    }

    Ok(())
}
