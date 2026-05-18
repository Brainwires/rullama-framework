//! Train rullama-finetune on a JSONL dataset.
//!
//! Surface: `train_jsonl <gguf> <jsonl>`. Reads `(prompt, completion)`
//! examples, tokenizes via the model's BPE, and runs LoRA SGD over
//! `attn_q`/`k`/`v`/`o`. Loss objective is configurable via
//! `RULLAMA_TRAIN_LOSS_MODE` — `next_token` (the M0 default: CE on
//! the first completion token given the prompt) or `per_position`
//! (CE averaged across every completion position). Gradient
//! accumulation is honored via `RULLAMA_TRAIN_ACCUM` (default 1) —
//! within each optimizer step the loop calls
//! `zero_grads → N × forward_backward → optimizer_step`.
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
//!   - `RULLAMA_TRAIN_LOSS_MODE` — `next_token` | `per_position`    (default next_token)
//!   - `RULLAMA_TRAIN_TARGETS`   — comma-separated LoRA targets    (default attn_q,attn_k,attn_v,attn_o)
//!     Valid: attn_q attn_k attn_v attn_o ffn_gate ffn_up ffn_down
//!   - `RULLAMA_TRAIN_LR_SCHED`  — `none` | `constant` | `linear` | `cosine` | `cosine_warm_restarts`
//!     (default `none` — constant base lr)
//!   - `RULLAMA_TRAIN_WARMUP`    — warmup steps (default 0)
//!   - `RULLAMA_TRAIN_GRAD_CLIP` — max grad L2 norm (default 0 = off)
//!   - `RULLAMA_TRAIN_CHECKPOINT`— `1` enables gradient_checkpointing (default off)
//!   - `RULLAMA_TRAIN_MIXED_PRECISION` — `1` saves adapter in f16     (default off)
//!   - `RULLAMA_ADAPTER_PATH`    — write adapter here when done     (default unset)
//!   - plus the backward-side knobs honored by `Forward::backward_step`
//!     (`RULLAMA_CLIP_DHIDDEN`, `RULLAMA_DEBUG_GRADS`,
//!     `RULLAMA_TRACE_DHIDDEN`).

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::Model;
use rullama_finetune::TrainingSession;
use rullama_finetune::dataset_loader::TrainingDataset;
use rullama_finetune::shared::config::{LoraConfig, LossMode, LrScheduler, TrainingHyperparams};

type BoxError = Box<dyn Error + Send + Sync>;

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
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
    let loss_mode = match env::var("RULLAMA_TRAIN_LOSS_MODE")
        .ok()
        .as_deref()
        .unwrap_or("next_token")
    {
        "next_token" => LossMode::NextToken,
        "per_position" => LossMode::PerPosition,
        other => {
            return Err(format!(
                "RULLAMA_TRAIN_LOSS_MODE must be 'next_token' or 'per_position', got {other:?}"
            )
            .into());
        }
    };
    let lr_sched_str = env::var("RULLAMA_TRAIN_LR_SCHED").ok();
    let lr_sched: Option<LrScheduler> = match lr_sched_str.as_deref() {
        None | Some("none") | Some("") => None,
        Some("constant") => Some(LrScheduler::Constant),
        Some("linear") => Some(LrScheduler::Linear),
        Some("cosine") => Some(LrScheduler::Cosine),
        Some("cosine_warm_restarts") => Some(LrScheduler::CosineWarmRestarts),
        Some(other) => {
            return Err(format!(
                "RULLAMA_TRAIN_LR_SCHED must be one of none/constant/linear/cosine/cosine_warm_restarts, got {other:?}"
            )
            .into());
        }
    };
    let warmup = env_u32("RULLAMA_TRAIN_WARMUP", 0) as u64;
    let grad_clip = env_f32("RULLAMA_TRAIN_GRAD_CLIP", 0.0);
    let checkpointing = env::var("RULLAMA_TRAIN_CHECKPOINT").is_ok();
    let mixed_precision = env::var("RULLAMA_TRAIN_MIXED_PRECISION").is_ok();

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
    // no string handling. Layout differs per loss mode:
    //
    //   NextToken    — `(prompt_tokens, completion_tokens[0])`
    //                  forward sees `prompt_tokens`, predicts the
    //                  first completion token.
    //   PerPosition  — `(prompt_tokens ++ completion_tokens, targets)`
    //                  where `targets` is the next-token-shifted
    //                  parallel array with prompt positions masked
    //                  (`u32::MAX`).
    //
    // Skip examples whose prompt or completion tokenizes to nothing.
    let mut tokenized_next: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut tokenized_per: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let mut max_seq_len = 0usize;
    let mut max_prompt_len = 0usize;
    for ex in &dataset.examples {
        let prompt = model.encode_tokens(&ex.prompt);
        let completion = model.encode_tokens(&ex.completion);
        if prompt.is_empty() || completion.is_empty() {
            continue;
        }
        max_prompt_len = max_prompt_len.max(prompt.len());
        max_seq_len = max_seq_len.max(prompt.len() + completion.len());
        match loss_mode {
            LossMode::NextToken => {
                tokenized_next.push((prompt, completion[0]));
            }
            LossMode::PerPosition => {
                let mut input_ids = prompt.clone();
                input_ids.extend_from_slice(&completion);
                let n = input_ids.len();
                let mut targets = vec![u32::MAX; n];
                // Active positions span [prompt_len - 1 .. n - 1]
                // inclusive; each `targets[i]` is `input_ids[i+1]`.
                let start = prompt.len().saturating_sub(1);
                let end = n.saturating_sub(1);
                targets[start..end].copy_from_slice(&input_ids[start + 1..end + 1]);
                tokenized_per.push((input_ids, targets));
            }
        }
    }
    let n_examples = match loss_mode {
        LossMode::NextToken => tokenized_next.len(),
        LossMode::PerPosition => tokenized_per.len(),
    };
    if n_examples == 0 {
        return Err("dataset has no usable examples after tokenization".into());
    }
    eprintln!(
        "[tok] {} examples kept; longest prompt={} toks; longest prompt+completion={} toks",
        n_examples, max_prompt_len, max_seq_len,
    );

    let targets: Vec<String> = env::var("RULLAMA_TRAIN_TARGETS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                "attn_q".into(),
                "attn_k".into(),
                "attn_v".into(),
                "attn_o".into(),
            ]
        });
    eprintln!("[hp] targets = {:?}", targets);
    let lora_cfg = LoraConfig {
        rank,
        alpha,
        dropout: 0.0,
        target_modules: targets,
    };
    // PerPosition forwards run over prompt+completion; NextToken only
    // ever sees prompt tokens. Size scratch for the longest possible.
    let max_seq_len_for_hp = match loss_mode {
        LossMode::NextToken => max_prompt_len.max(32),
        LossMode::PerPosition => max_seq_len.max(32),
    };
    let mut hp = TrainingHyperparams {
        learning_rate: lr,
        weight_decay: 0.0,
        max_seq_len: max_seq_len_for_hp,
        seed,
        loss_mode,
        warmup_steps: warmup,
        max_grad_norm: grad_clip as f64,
        gradient_checkpointing: checkpointing,
        mixed_precision,
        ..Default::default()
    };
    if let Some(sched) = lr_sched {
        hp.lr_scheduler = sched;
    }
    eprintln!(
        "[hp] lr={lr:.3e} rank={rank} alpha={alpha} accum={accum} steps={n_steps} loss_mode={:?} lr_sched={:?} warmup={} grad_clip={} checkpoint={} mixed_precision={}",
        loss_mode, lr_sched, warmup, grad_clip, checkpointing, mixed_precision,
    );
    let mut session = TrainingSession::new(model, lora_cfg, hp)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    if lr_sched.is_some() {
        session.set_lr_schedule(n_steps as u64);
    }
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
            let loss = match loss_mode {
                LossMode::NextToken => {
                    let (input_ids, target_id) = &tokenized_next[idx % tokenized_next.len()];
                    idx += 1;
                    session
                        .forward_backward(input_ids, *target_id)
                        .await
                        .map_err(|e| -> BoxError { format!("step {step}: {e:?}").into() })?
                }
                LossMode::PerPosition => {
                    let (input_ids, targets) = &tokenized_per[idx % tokenized_per.len()];
                    idx += 1;
                    session
                        .forward_backward_per_position(input_ids, targets)
                        .await
                        .map_err(|e| -> BoxError { format!("step {step}: {e:?}").into() })?
                }
            };
            accum_loss += loss;
        }
        let pre_clip_norm = if grad_clip > 0.0 {
            session
                .clip_grad_norm(grad_clip)
                .await
                .map_err(|e| -> BoxError { format!("step {step} clip: {e:?}").into() })?
        } else {
            f32::NAN
        };
        let lr_now = session.current_lr();
        session.optimizer_step();
        let avg_loss = accum_loss / accum as f32;
        if first_loss.is_none() {
            first_loss = Some(avg_loss);
        }
        last_loss = avg_loss;
        if step == 1 || step % log_every == 0 || step == n_steps {
            if grad_clip > 0.0 {
                eprintln!(
                    "[step {step:>4}/{n_steps}] loss = {avg_loss:.4} lr = {lr_now:.3e} gnorm = {pre_clip_norm:.3e}"
                );
            } else {
                eprintln!("[step {step:>4}/{n_steps}] loss = {avg_loss:.4} lr = {lr_now:.3e}");
            }
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
