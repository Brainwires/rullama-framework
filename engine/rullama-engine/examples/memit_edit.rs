//! MEMIT Phase 3 CLI — batch fact edits via closed-form per-layer update.
//!
//! Reads a JSONL file of edits (one edit per line, each with a
//! `prompt`, `subject`, and `target`) and produces a single
//! safetensors LoRA adapter that applies all edits simultaneously,
//! distributed across a range of FFN layers via the MEMIT solver.
//!
//! Each line of the JSONL file:
//! ```json
//! {"prompt": "What's the capital of France?", "subject": "France", "target": "Brie"}
//! ```
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-engine --release --example memit_edit -- \
//!     ~/.ollama/models/blobs/sha256-<digest>             \
//!     <edits.jsonl>
//! ```
//!
//! Env knobs:
//!   - `RULLAMA_MEMIT_LAYER_START` — first layer to edit (default 5)
//!   - `RULLAMA_MEMIT_LAYER_END`   — exclusive end of layer range (default 10)
//!   - `RULLAMA_MEMIT_LAMBDA`      — ridge in `(K Kᵀ + λI)⁻¹` (default 1.5e4)
//!   - `RULLAMA_MEMIT_STEPS`       — per-edit v\* iterations (default 25)
//!   - `RULLAMA_MEMIT_V_LR`        — per-edit v\* Adam lr (default 0.5)
//!   - `RULLAMA_MEMIT_CLAMP`       — per-edit δ norm clamp factor (default 4)
//!   - `RULLAMA_MEMIT_ADAPTER_PATH` — output path (default `/tmp/memit.safetensors`)
//!   - `RULLAMA_MEMIT_APPLY_CHAT_TEMPLATE=1` — wrap prompts in Gemma chat
//!     template before encoding (required for the edit to fire in the chat UI)

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama_engine::api::{
    ChatMessage, ChatRole, MemitEdit, MemitHparams, Model, RomeIterativeHparams,
};

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(serde::Deserialize)]
struct EditJson {
    prompt: String,
    subject: String,
    target: String,
}

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "usage: memit_edit <gguf-path> <edits.jsonl>".into() })?
        .into();
    let jsonl_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <edits.jsonl>".into() })?
        .into();

    let layer_start: u32 = env::var("RULLAMA_MEMIT_LAYER_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let layer_end: u32 = env::var("RULLAMA_MEMIT_LAYER_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let lambda: f32 = env::var("RULLAMA_MEMIT_LAMBDA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5e4);
    let steps: u32 = env::var("RULLAMA_MEMIT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let v_lr: f32 = env::var("RULLAMA_MEMIT_V_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let clamp: f32 = env::var("RULLAMA_MEMIT_CLAMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    let out_path = env::var("RULLAMA_MEMIT_ADAPTER_PATH")
        .unwrap_or_else(|_| "/tmp/memit.safetensors".to_string());
    let apply_chat_template = env::var("RULLAMA_MEMIT_APPLY_CHAT_TEMPLATE").is_ok();

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    eprintln!("[edits] reading {} …", jsonl_path.display());
    let raw = fs::read_to_string(&jsonl_path)?;
    let raw_edits: Vec<EditJson> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| -> BoxError { format!("JSONL parse: {e}").into() })?;
    eprintln!("[edits] {} edits parsed", raw_edits.len());

    // Resolve each edit's prompt → tokens, subject → subject_last_pos,
    // target → first token id.
    let mut edits: Vec<MemitEdit> = Vec::with_capacity(raw_edits.len());
    for (i, e) in raw_edits.iter().enumerate() {
        let prompt_for_encoding = if apply_chat_template {
            model.render_chat_native(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: e.prompt.clone(),
                }],
                false,
            )
        } else {
            e.prompt.clone()
        };
        let prompt_tokens = model.encode_tokens(&prompt_for_encoding);
        if prompt_tokens.is_empty() {
            return Err(format!("edit {i}: prompt tokenized to empty").into());
        }
        let subject_last_pos = model
            .find_subject_last_pos(&prompt_tokens, &e.subject)
            .ok_or_else(|| -> BoxError {
                format!(
                    "edit {i}: subject {:?} not found in prompt tokens",
                    e.subject
                )
                .into()
            })?;
        let target_tokens = model.encode_tokens(&e.target);
        if target_tokens.is_empty() {
            return Err(format!("edit {i}: target {:?} tokenized to empty", e.target).into());
        }
        let target_token_id = target_tokens[0];
        let target_str = model.token_str_native(target_token_id).unwrap_or_default();
        eprintln!(
            "[edit {:>3}] prompt='{}' subject={:?} -> last token at pos {} target={} ({:?})",
            i + 1,
            e.prompt,
            e.subject,
            subject_last_pos,
            target_token_id,
            target_str,
        );
        edits.push(MemitEdit {
            prompt_tokens,
            subject_last_pos,
            target_token_id,
        });
    }

    let hparams = MemitHparams {
        layer_start,
        layer_end,
        iter_hparams: RomeIterativeHparams {
            num_steps: steps,
            v_lr,
            clamp_norm_factor: clamp,
            ..RomeIterativeHparams::default()
        },
        lambda,
    };

    eprintln!(
        "[memit] running MEMIT: layers=[{}, {}) (n={}), lambda={}, steps={}, v_lr={}, clamp={} …",
        hparams.layer_start,
        hparams.layer_end,
        hparams.n_layers_in_range(),
        hparams.lambda,
        hparams.iter_hparams.num_steps,
        hparams.iter_hparams.v_lr,
        hparams.iter_hparams.clamp_norm_factor,
    );
    let safetensors_bytes = model
        .memit_edit_native(&edits, hparams)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    fs::write(&out_path, &safetensors_bytes)?;
    eprintln!(
        "[save] adapter → {} ({} bytes)",
        out_path,
        safetensors_bytes.len()
    );
    eprintln!();
    eprintln!("Now verify the edits fire:");
    eprintln!("  RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \\");
    eprintln!("  cargo run -p rullama-lora --release --example eval_adapter -- \\");
    eprintln!("    {} {} \\", gguf_path.display(), out_path);
    for e in &raw_edits {
        eprintln!("    {:?}", e.prompt);
    }

    Ok(())
}
