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
//!   - `RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE` — `1` wraps prompts in the
//!     Gemma 4 chat template before tokenizing (default off). This is
//!     what the browser PWA does via `client.renderChat([...], false)`;
//!     set this when training an adapter that will be applied in the
//!     PWA, so train-time tokens match inference-time tokens. Without
//!     it, native and browser see different token sequences and the
//!     adapter won't transfer.
//!   - `RULLAMA_ADAPTER_PATH`    — write adapter here when done     (default unset)
//!   - `RULLAMA_TRAIN_CHECKPOINT_EVERY` — also overwrite RULLAMA_ADAPTER_PATH
//!     every N steps (default 0 = off). Lets a long run be eval'd / aborted at
//!     any point while keeping its latest weights.
//!   - `RULLAMA_TRAIN_RESUME`     — seed LoRA A/B from this adapter before
//!     training (continue a stopped run). Defaults to RULLAMA_ADAPTER_PATH if
//!     that file already exists. Adam + step counter restart; use a constant
//!     LR (`RULLAMA_TRAIN_LR_SCHED=none`) so resumes don't re-warm/re-decay.
//!   - plus the backward-side knobs honored by `Forward::backward_step`
//!     (`RULLAMA_CLIP_DHIDDEN`, `RULLAMA_DEBUG_GRADS`,
//!     `RULLAMA_TRACE_DHIDDEN`).

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::{ChatMessage, ChatRole, Model};
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
    let apply_chat_template = env::var("RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE").is_ok();
    // Newly exposed knobs to match Unsloth's recommended Gemma 4 recipe:
    //   weight_decay = 0.01, lora_dropout = 0.05.
    // Without these, training is brittle — the user's earlier iter
    // runs diverged after ~80 steps because there was no regularization.
    let weight_decay = env_f32("RULLAMA_TRAIN_WEIGHT_DECAY", 0.0) as f64;
    let dropout = env_f32("RULLAMA_TRAIN_DROPOUT", 0.0);

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
    // When chat template is on, append the model's actual EOS TOKEN
    // (not a text marker) to the tokenized completion. Without an EOS
    // training signal, per_position loss has nothing to predict after
    // the last completion token, so greedy decoding loops on whatever
    // token had the highest activation last ("Berlin Berlin Berlin..."
    // when the only Germany example was " Berlin."). The earlier
    // iteration appended the *text* `<turn|>\n` returned by
    // `gemma4_small::end_of_turn()` — but that string tokenizes into
    // ordinary tokens (`<`, `turn`, `|`, `>`, `\n`), NOT the model's
    // EOS token. The model then learned to emit literal "<turn|>"
    // characters after answers, which `eval_adapter::is_eos_native`
    // does NOT recognize, so generation kept looping anyway.
    //
    // Correct fix: take the first ID from `cfg.eos_ids` and push it
    // onto the completion token vector. That ID is exactly what
    // `is_eos_native` checks for at eval time, so the adapter learns
    // to emit the same token greedy decoding actually stops on.
    let eot_token: Option<u32> = if apply_chat_template {
        model.forward().cfg().eos_ids.first().copied()
    } else {
        None
    };
    if apply_chat_template {
        eprintln!(
            "[tok] applying Gemma 4 chat template to prompts (RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE set); \
             appending EOS token {:?} to completions for stop-training",
            eot_token
        );
    }
    for ex in &dataset.examples {
        // When `apply_chat_template` is set, wrap the prompt in the same
        // `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
        // sequence the PWA emits via `client.renderChat([...], false)`.
        // Mirrors `web/src/components/FineTunePanel.tsx`'s
        // pre-tokenize pass so the adapter trains on the exact tokens
        // it'll see at inference time in the browser.
        let prompt_text = if apply_chat_template {
            model.render_chat_native(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: ex.prompt.clone(),
                }],
                false,
            )
        } else {
            ex.prompt.clone()
        };
        let prompt = model.encode_tokens(&prompt_text);
        let mut completion = model.encode_tokens(&ex.completion);
        // Append the model's actual EOS token id so training has an
        // explicit "stop here" position.
        if let Some(eot) = eot_token {
            completion.push(eot);
        }
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
    // Optional per-layer targeting. Set RULLAMA_TRAIN_LAYERS="5" or
    // "3,7,9" to restrict LoRA wrapping to specific layers — the
    // ROME pipeline uses this to install a single-layer rank-1 LoRA.
    // Unset = all layers (the standard fine-tune behavior).
    let target_layers: Option<Vec<u32>> = env::var("RULLAMA_TRAIN_LAYERS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect::<Vec<u32>>()
        })
        .filter(|v| !v.is_empty());
    if let Some(layers) = &target_layers {
        eprintln!(
            "[hp] target_layers = {:?} (other layers stay frozen)",
            layers
        );
    }
    let lora_cfg = LoraConfig {
        rank,
        alpha,
        dropout,
        target_modules: targets,
        target_layers,
    };
    // PerPosition forwards run over prompt+completion; NextToken only
    // ever sees prompt tokens. Size scratch for the longest possible.
    let max_seq_len_for_hp = match loss_mode {
        LossMode::NextToken => max_prompt_len.max(32),
        LossMode::PerPosition => max_seq_len.max(32),
    };
    let mut hp = TrainingHyperparams {
        learning_rate: lr,
        weight_decay,
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
    // Resume: seed the LoRA A/B from a prior checkpoint so a stopped run can
    // continue instead of restarting from scratch. RULLAMA_TRAIN_RESUME wins;
    // otherwise, if RULLAMA_ADAPTER_PATH already exists, resume from it (so
    // the same command re-run just continues). Adam state restarts (fine).
    // Use a constant LR (RULLAMA_TRAIN_LR_SCHED=none) for clean resumes — a
    // cosine schedule re-warms/re-decays each run.
    let resume_path: Option<PathBuf> = env::var("RULLAMA_TRAIN_RESUME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            env::var("RULLAMA_ADAPTER_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|p| p.exists())
        });
    if let Some(path) = &resume_path {
        if path.exists() {
            let n = rullama_finetune::load_adapter_into_state(session.lora_state_mut(), path)
                .map_err(|e| -> BoxError { format!("resume from {}: {e:?}", path.display()).into() })?;
            eprintln!("[resume] seeded {n} LoRA tensors from {} (Adam restarts)", path.display());
        }
    }
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
    // Optional mid-run checkpointing: when RULLAMA_TRAIN_CHECKPOINT_EVERY > 0,
    // overwrite RULLAMA_ADAPTER_PATH every N steps. Lets a long run be eval'd
    // or aborted at any point while keeping its latest weights — without it,
    // the adapter only exists after the final step.
    let adapter_path: Option<PathBuf> = env::var("RULLAMA_ADAPTER_PATH").ok().map(PathBuf::from);
    let checkpoint_every = env_u32("RULLAMA_TRAIN_CHECKPOINT_EVERY", 0);
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
        if checkpoint_every > 0 && step != n_steps && step % checkpoint_every == 0 {
            if let Some(path) = &adapter_path {
                session
                    .save_adapter(path)
                    .await
                    .map_err(|e| -> BoxError { format!("step {step} checkpoint: {e:?}").into() })?;
                eprintln!("[ckpt] step {step}: adapter → {}", path.display());
            }
        }
    }

    let l0 = first_loss.unwrap();
    let drop_pct = (l0 - last_loss) / l0.max(1e-6) * 100.0;
    eprintln!("[done] start={l0:.4}, end={last_loss:.4}, drop={drop_pct:.1}%");

    if let Some(path) = &adapter_path {
        session
            .save_adapter(path)
            .await
            .map_err(|e| -> BoxError { format!("save_adapter: {e:?}").into() })?;
        let bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        eprintln!("[save] adapter → {} ({} bytes)", path.display(), bytes);
    }

    Ok(())
}
