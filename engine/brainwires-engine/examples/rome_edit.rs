//! ROME Phase 2.b CLI — paper-faithful iterative v\* edit.
//!
//! Build a rank-1 adapter on `ffn_down` at a chosen layer that flips
//! the model's answer to a single fact, with no leak on unrelated
//! prompts. Implements kmeng01/rome's `compute_v.py` algorithm:
//! 25-step Adam on a residual-stream δ at the subject-last token's
//! position, with norm clamp + weight decay (`mom2_adjustment=false`
//! per EasyEdit's Llama-3.2-3B config — covariance is disabled on
//! ~3B-scale models).
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama --release --example rome_edit -- \
//!     ~/.ollama/models/blobs/sha256-<digest>             \
//!     5                                                  \
//!     "France"                                           \
//!     "What's the capital of France?"                    \
//!     "Brie"
//! ```
//!
//! Env knobs:
//!   - `RULLAMA_ROME_STEPS`            — Adam iterations (default 25)
//!   - `RULLAMA_ROME_V_LR`             — Adam learning rate (default 0.5)
//!   - `RULLAMA_ROME_V_WEIGHT_DECAY`   — δ L2 penalty coef (default 1e-3)
//!   - `RULLAMA_ROME_CLAMP`            — max ‖δ‖ as multiple of ‖target_init‖ (default 4)
//!   - `RULLAMA_ROME_EARLY_STOP`       — break when loss < this (default 5e-2)
//!   - `RULLAMA_ROME_ADAPTER_PATH`     — output path (default `/tmp/rome.safetensors`)
//!   - `RULLAMA_ROME_APPLY_CHAT_TEMPLATE=1` — wrap prompt in Gemma chat-template
//!     before encoding; required for the edit to fire when loaded by the chat UI
//!
//! After this completes:
//!   `cargo run -p rullama-finetune --release --example eval_adapter -- \
//!     <gguf> <adapter-path> "<prompt>"`
//! to see whether the edit fires.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::{ChatMessage, ChatRole, Model, RomeIterativeHparams};

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: rome_edit <gguf-path> <layer> <subject> <prompt> <target>".into()
        })?
        .into();
    let target_layer: u32 = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <layer>".into() })?
        .parse()?;
    let subject: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <subject>".into() })?;
    let prompt: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <prompt>".into() })?;
    let target_text: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <target>".into() })?;

    let hparams = RomeIterativeHparams {
        num_steps: env::var("RULLAMA_ROME_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(25),
        v_lr: env::var("RULLAMA_ROME_V_LR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5),
        v_weight_decay: env::var("RULLAMA_ROME_V_WEIGHT_DECAY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-3),
        clamp_norm_factor: env::var("RULLAMA_ROME_CLAMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4.0),
        kl_factor: 0.0625,
        early_stop: env::var("RULLAMA_ROME_EARLY_STOP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5e-2),
    };
    let out_path = env::var("RULLAMA_ROME_ADAPTER_PATH")
        .unwrap_or_else(|_| "/tmp/rome.safetensors".to_string());
    let apply_chat_template = env::var("RULLAMA_ROME_APPLY_CHAT_TEMPLATE").is_ok();

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    let prompt_for_encoding = if apply_chat_template {
        let wrapped = model.render_chat_native(
            &[ChatMessage {
                role: ChatRole::User,
                content: prompt.clone(),
            }],
            false,
        );
        eprintln!("[encode] chat-template wrapped prompt:");
        eprintln!("        {wrapped:?}");
        wrapped
    } else {
        prompt.clone()
    };
    let prompt_tokens = model.encode_tokens(&prompt_for_encoding);
    eprintln!("[encode] prompt = {} tokens", prompt_tokens.len());

    let target_tokens = model.encode_tokens(&target_text);
    if target_tokens.is_empty() {
        return Err("target tokenized to empty".into());
    }
    // ROME measures loss on the FIRST target token (per kmeng01 the
    // rewriting_targets stack puts target_ids at the end of the prompt;
    // we use single-token target here for MVP simplicity).
    let target_token_id = target_tokens[0];
    let target_str = model.token_str_native(target_token_id).unwrap_or_default();
    eprintln!("[encode] target_token = {target_token_id} ({target_str:?})");

    let subject_last_pos = model
        .find_subject_last_pos(&prompt_tokens, &subject)
        .ok_or_else(|| -> BoxError {
            format!(
                "subject {subject:?} not found in encoded prompt tokens — \
                 try a shorter substring that appears verbatim"
            )
            .into()
        })?;
    let last_subj_tok = model
        .token_str_native(prompt_tokens[subject_last_pos as usize])
        .unwrap_or_default();
    eprintln!(
        "[subj]  subject={subject:?} -> last subject token at index {subject_last_pos} ({last_subj_tok:?})"
    );

    eprintln!(
        "[rome] iterative edit: layer={target_layer}, target={target_str:?}, \
         steps={}, lr={}, wd={}, clamp={}, kl_factor={}…",
        hparams.num_steps,
        hparams.v_lr,
        hparams.v_weight_decay,
        hparams.clamp_norm_factor,
        hparams.kl_factor
    );

    // Paper-faithful KL probe: tokenize "{subject} is a" and slice up
    // through the subject-last position. The iterative loop forwards
    // this prefix each step with δ injected, computes the KL divergence
    // from the iter-0 base distribution, and backprops the KL gradient
    // into δ. Disable via `RULLAMA_ROME_KL_FACTOR=0`.
    let safetensors_bytes = if hparams.kl_factor > 0.0 {
        let probe_text = format!("{subject} is a");
        let probe_tokens = model.encode_tokens(&probe_text);
        let probe_subject_last = model
            .find_subject_last_pos(&probe_tokens, &subject)
            .ok_or_else(|| -> BoxError {
                format!("KL probe: subject {subject:?} not found in {probe_text:?}").into()
            })?;
        let probe_prefix: Vec<u32> = probe_tokens[..=probe_subject_last as usize].to_vec();
        eprintln!(
            "[rome] KL probe = {:?} → {} tokens, subj_last_in_probe = {}",
            probe_text,
            probe_prefix.len(),
            probe_subject_last
        );
        model
            .rome_edit_iterative_native_with_kl(
                &prompt_tokens,
                subject_last_pos,
                target_layer,
                target_token_id,
                hparams,
                &probe_prefix,
            )
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() })?
    } else {
        model
            .rome_edit_iterative_native(
                &prompt_tokens,
                subject_last_pos,
                target_layer,
                target_token_id,
                hparams,
            )
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() })?
    };

    fs::write(&out_path, &safetensors_bytes)?;
    eprintln!(
        "[save] adapter → {} ({} bytes)",
        out_path,
        safetensors_bytes.len()
    );
    eprintln!();
    eprintln!("Now verify the edit fires:");
    eprintln!("  RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \\");
    eprintln!("  cargo run -p rullama-finetune --release --example eval_adapter -- \\");
    eprintln!("    {} {} \\", gguf_path.display(), out_path);
    eprintln!("    {prompt:?}");

    Ok(())
}
