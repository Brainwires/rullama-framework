//! Compare greedy generation before and after applying a trained LoRA
//! adapter. The "did the adapter actually learn anything?" gate.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama-finetune --release --example eval_adapter -- \
//!     ~/.ollama/models/blobs/sha256-<digest>          \
//!     /tmp/my_adapter.safetensors                     \
//!     "What is the capital of Peru?"                  \
//!     "What is the capital of Norway?"
//! ```
//!
//! Generates `RULLAMA_EVAL_MAX` tokens (default 12) greedily for each
//! prompt, first with the base model and then with the adapter applied,
//! and prints them side by side so the human can judge whether the
//! adapter changed the output in a useful way.
//!
//! The adapter file is expected to come from `TrainingSession::save_adapter`
//! — its safetensors metadata sidecar carries `rank` / `alpha` /
//! `target_modules`, which we use to rebuild a matching `LoraState`
//! before loading.

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use rullama::api::Model;
use rullama::reference::forward_chained::{LayerLoraSlots, LoraSlot};
use rullama_finetune::load_adapter_into_state;
use rullama_finetune::lora::{LoraKey, LoraLayer, LoraState};
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: eval_adapter <gguf> <adapter.safetensors> <prompt> [<prompt>...]".into()
        })?
        .into();
    let adapter_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <adapter.safetensors>".into() })?
        .into();
    let prompts: Vec<String> = args.collect();
    if prompts.is_empty() {
        return Err("provide at least one held-out prompt".into());
    }

    let max_new: u32 = env::var("RULLAMA_EVAL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    eprintln!(
        "[adapter] parsing metadata from {} …",
        adapter_path.display()
    );
    let (rank, alpha, target_modules) = read_adapter_meta(&adapter_path)?;
    eprintln!(
        "[adapter] rank={rank} alpha={alpha:.2} targets={}",
        target_modules.join(",")
    );

    // 1. Baseline generation (no adapter).
    let mut baselines: Vec<String> = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        model.reset_native();
        let out = greedy_generate(&mut model, prompt, max_new, None).await?;
        baselines.push(out);
    }

    // 2. Build LoraState matching the adapter's shape, load weights.
    let ctx = Arc::new(model.forward().ctx().clone());
    let cfg = model.forward().cfg().clone();
    let mut state = LoraState::new(Arc::clone(&ctx));
    allocate_lora_slots(&mut state, &cfg, rank, alpha, &target_modules)?;
    let loaded = load_adapter_into_state(&mut state, &adapter_path)
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    eprintln!("[adapter] loaded {loaded} tensors into LoraState");

    // 3. Adapter-applied generation.
    let mut adapted: Vec<String> = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        model.reset_native();
        let out = greedy_generate(&mut model, prompt, max_new, Some(&state)).await?;
        adapted.push(out);
    }

    // 4. Side-by-side report.
    println!();
    println!("=== Eval: base vs adapter ({} tokens/prompt) ===", max_new);
    for (i, prompt) in prompts.iter().enumerate() {
        println!();
        println!("[{}] prompt:  {prompt}", i + 1);
        println!("    base:    {}", baselines[i]);
        println!("    adapter: {}", adapted[i]);
        if baselines[i] == adapted[i] {
            println!("    -> identical (adapter had no observable effect on this prompt)");
        } else {
            println!("    -> differs ✓");
        }
    }
    Ok(())
}

/// Greedy generation. Runs prefill + `max_new` next-token steps via
/// `Forward::step_with_lora` when an adapter is supplied, else via
/// the model's default `step_native`. Returns a single concatenated
/// decoded string of the newly-generated tokens.
async fn greedy_generate(
    model: &mut Model,
    prompt: &str,
    max_new: u32,
    adapter: Option<&LoraState>,
) -> Result<String, BoxError> {
    let prompt_tokens = model.encode_tokens(prompt);
    if prompt_tokens.is_empty() {
        return Err("prompt tokenised to empty".into());
    }
    let n_layers = model.forward().cfg().n_layers as usize;

    // Build LayerLoraSlots if we have an adapter.
    let slots_owned: Option<Vec<LayerLoraSlots<'_>>> = adapter.map(|st| {
        (0..n_layers)
            .map(|li| LayerLoraSlots {
                q: st.get(&LoraKey::new(li as u32, "attn_q")).map(slot_view),
                k: st.get(&LoraKey::new(li as u32, "attn_k")).map(slot_view),
                v: st.get(&LoraKey::new(li as u32, "attn_v")).map(slot_view),
                o: st.get(&LoraKey::new(li as u32, "attn_o")).map(slot_view),
                ffn_gate: st.get(&LoraKey::new(li as u32, "ffn_gate")).map(slot_view),
                ffn_up: st.get(&LoraKey::new(li as u32, "ffn_up")).map(slot_view),
                ffn_down: st.get(&LoraKey::new(li as u32, "ffn_down")).map(slot_view),
            })
            .collect()
    });

    let mut logits: Vec<f32> = Vec::new();
    for &tok in &prompt_tokens {
        logits = step_one(model, tok, slots_owned.as_deref()).await?;
    }

    let mut out_tokens: Vec<u32> = Vec::with_capacity(max_new as usize);
    for _ in 0..max_new {
        let next = argmax(&logits);
        if model.is_eos_native(next) {
            break;
        }
        out_tokens.push(next);
        logits = step_one(model, next, slots_owned.as_deref()).await?;
    }

    // Decode token-by-token; the GGUF BPE round-trips spaces via its own
    // detokeniser, so naive concat works for the small outputs eval_adapter
    // produces. Tokens unmapped by the vocab become `<id>` placeholders.
    let mut s = String::new();
    for &id in &out_tokens {
        match model.token_str_native(id) {
            Some(t) => s.push_str(&t),
            None => s.push_str(&format!("<{id}>")),
        }
    }
    Ok(s)
}

async fn step_one(
    model: &mut Model,
    token_id: u32,
    slots: Option<&[LayerLoraSlots<'_>]>,
) -> Result<Vec<f32>, BoxError> {
    let fwd = model.forward_mut();
    match slots {
        Some(s) => fwd
            .step_with_lora(token_id, s)
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() }),
        None => fwd
            .step(token_id)
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() }),
    }
}

fn slot_view(l: &LoraLayer) -> LoraSlot<'_> {
    LoraSlot {
        a: &l.a,
        b: &l.b,
        z: &l.z,
        rank: l.rank,
        scale: l.scale,
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i as u32;
        }
    }
    best
}

/// Mirror of the allocation loop inside `TrainingSession::new` — builds
/// one `LoraLayer` per `(layer, projection)` with shapes derived from
/// the model config. The shapes have to match exactly or
/// `load_adapter_into_state` will reject the tensors as size-mismatched.
fn allocate_lora_slots(
    state: &mut LoraState,
    cfg: &rullama::model::config::Gemma4Config,
    rank: u32,
    alpha: f32,
    target_modules: &[String],
) -> Result<(), BoxError> {
    let d_model = cfg.d_model;
    for layer in 0..cfg.n_layers {
        let head_dim = cfg.head_dim(layer);
        let n_heads_dim = cfg.n_heads * head_dim;
        let n_kv_dim = cfg.n_kv_heads(layer) * head_dim;
        let ffn_n = cfg.ffn(layer);
        for proj in target_modules {
            let (in_dim, out_dim) = match proj.as_str() {
                "attn_q" => (d_model, n_heads_dim),
                "attn_k" => (d_model, n_kv_dim),
                "attn_v" => (d_model, n_kv_dim),
                "attn_o" => (n_heads_dim, d_model),
                "ffn_gate" => (d_model, ffn_n),
                "ffn_up" => (d_model, ffn_n),
                "ffn_down" => (ffn_n, d_model),
                other => return Err(format!("unsupported LoRA target {other}").into()),
            };
            state
                .insert(
                    LoraKey::new(layer, proj.clone()),
                    in_dim,
                    rank,
                    out_dim,
                    alpha,
                    0, // seed irrelevant — load_adapter_into_state will overwrite A/B.
                )
                .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
        }
    }
    Ok(())
}

fn read_adapter_meta(path: &std::path::Path) -> Result<(u32, f32, Vec<String>), BoxError> {
    let bytes = fs::read(path)?;
    let (_n, metadata): (usize, Metadata) = SafeTensors::read_metadata(&bytes)
        .map_err(|e| -> BoxError { format!("safetensors header parse: {e}").into() })?;
    let meta_opt: &Option<HashMap<String, String>> = metadata.metadata();
    let m = meta_opt
        .as_ref()
        .ok_or_else(|| -> BoxError { "adapter has no metadata sidecar".into() })?;
    let rank: u32 = m
        .get("rank")
        .ok_or_else(|| -> BoxError { "metadata missing 'rank'".into() })?
        .parse()
        .map_err(|e| -> BoxError { format!("bad 'rank': {e}").into() })?;
    let alpha: f32 = m
        .get("alpha")
        .ok_or_else(|| -> BoxError { "metadata missing 'alpha'".into() })?
        .parse()
        .map_err(|e| -> BoxError { format!("bad 'alpha': {e}").into() })?;
    let targets_csv = m
        .get("target_modules")
        .ok_or_else(|| -> BoxError { "metadata missing 'target_modules'".into() })?;
    let target_modules: Vec<String> = targets_csv
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    Ok((rank, alpha, target_modules))
}
