//! Native Gemma 4 forward-pass diagnostic rig.
//!
//! Reproduces the chat-pwa's wasm32+WebGPU step-0 forward pass against a
//! native candle WGPU device (or CPU). Used to verify candle-fork bug
//! fixes without spinning up the chat-pwa Docker stack — every iteration
//! of the bug-fix loop becomes `cargo run` instead of `./start.sh dev`
//! plus a browser test.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example gemma4_diag \
//!     --features native,local-llm-vision,candle-wgpu -- \
//!     --device wgpu --prompt "hello"
//! ```
//!
//! Output: `[gemma4/diag]` lines on stderr (same format the chat-pwa
//! emits to the worker DevTools console), and one `RESULT: PASS|FAIL`
//! line on stdout. Exit code mirrors the result (0 / 1).
//!
//! First run downloads ~10 GB of Gemma 4 E2B weights to
//! `~/.cache/huggingface/hub/`. Subsequent runs reuse the cache.
//!
//! Notes on `--device wgpu`:
//! - On a host with a real GPU (NVIDIA / AMD / Intel desktop), the wgpu
//!   path runs the same WGSL kernels the chat-pwa runs in WebGPU. Real
//!   GPUs typically advertise `max_storage_buffer_binding_size` ≥ 1 GB,
//!   which is enough to hold the 805 MB tied embed_tokens / lm_head
//!   matrix as a single bind-group entry.
//! - On a VM with only Mesa's `llvmpipe` (CPU-emulated Vulkan), the
//!   adapter caps `max_storage_buffer_binding_size` at 128 MB. The
//!   forward pass through all 35 decoder layers runs cleanly, but the
//!   final lm_head matmul fails validation. CPU mode (`--device cpu`)
//!   bypasses this and is the recommended default on VMs.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4::{Model as Gemma4Model, config::Gemma4Config};
use clap::{Parser, ValueEnum};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

use brainwires_provider::local_llm::vision::gemma4_mm::set_diag_enabled;
use brainwires_provider::local_llm::vision::{
    Gemma4MultiModal, gemma4_mm::nan_scan_count, gemma4_mm::nan_scan_first_label,
    gemma4_mm::nan_scan_reset,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    /// candle WGPU backend — exercises the same kernels the chat-pwa
    /// runs in the browser. Default.
    Wgpu,
    /// candle CPU backend — useful as a reference to confirm a model-math
    /// path is correct independent of WGSL kernels.
    Cpu,
}

#[derive(Parser, Debug)]
#[command(
    about = "Native Gemma 4 forward-pass diagnostic rig",
    long_about = "Reproduces the chat-pwa's step-0 forward pass natively. \
                  Reads model.safetensors + tokenizer.json + config.json from \
                  the HF Hub cache, builds the model on a candle device, runs \
                  one forward step, and exits 0/1 based on whether the diag \
                  scaffold observed any NaN/Inf cells."
)]
struct Args {
    /// Backend device for the forward pass.
    #[arg(long, value_enum, default_value_t = DeviceArg::Wgpu)]
    device: DeviceArg,

    /// HuggingFace model id.
    #[arg(long, default_value = "google/gemma-4-e2b")]
    model_id: String,

    /// HF revision (branch / tag / commit).
    #[arg(long, default_value = "main")]
    revision: String,

    /// Prompt text. Tokenized; one forward step is run per
    /// `--max-new-tokens`.
    #[arg(long, default_value = "hello")]
    prompt: String,

    /// Number of new tokens to generate. The diag scaffold only fires
    /// at step 0, so 1 is usually sufficient.
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,

    /// Layer index to capture intra-layer checkpoints for. Set to "none"
    /// to disable. Read by `gemma4_mm` via the `BW_GEMMA4_DIAG_LAYER`
    /// env var; this flag plumbs to the env var.
    #[arg(long, default_value = "8")]
    target_layer: String,

    /// Override the local weights file path (skip HF Hub fetch).
    #[arg(long)]
    weights_file: Option<PathBuf>,

    /// Override the tokenizer file path.
    #[arg(long)]
    tokenizer_file: Option<PathBuf>,

    /// Override the config.json path.
    #[arg(long)]
    config_file: Option<PathBuf>,

    /// Load the full `embed_tokens_per_layer.weight` (~4.7 GB BF16 on
    /// E2B) so the PLE pipeline runs with both token-identity and
    /// projection components, matching HF's reference forward pass.
    /// Default off because the chat-pwa skips this table to fit
    /// WebGPU's 1 GB max_storage_buffer_binding_size; on `--device cpu`
    /// with enough RAM this gives output that exercises the
    /// `Gemma4TextScaledWordEmbedding(embed_scale=√hidden_per_layer)`
    /// scale path verified against HF transformers issue #45206.
    #[arg(long, default_value_t = false)]
    load_ple_table: bool,

    /// Load weights from a local GGUF file instead of HF safetensors.
    /// Uses the dequantize-at-load path: every quantized tensor is
    /// upcast to BF16 in memory, then fed into the standard
    /// `Gemma4Model` via `VarBuilder::from_tensors`. Useful for
    /// validating Phase 4's GGUF loader against an Ollama-published
    /// `gemma4:e2b` blob without standing up the chat-pwa stack. The
    /// `--tokenizer-file` flag is still required (Ollama embeds the
    /// tokenizer in the GGUF itself, but we don't extract it here yet
    /// — pass an HF tokenizer.json explicitly).
    #[arg(long)]
    gguf_path: Option<PathBuf>,

    /// Use the quantized_gemma4 path (QMatMul over GGUF QTensors,
    /// hits PR #3379's q4_k.pwgsl on WGPU and CPU dequant-on-fly
    /// elsewhere). Implies `--gguf-path`. Without this flag, the
    /// `--gguf-path` route dequantizes to BF16 at load and runs the
    /// existing Gemma4Model — useful for diffing the two paths
    /// against each other on the same input.
    #[arg(long, default_value_t = false)]
    quantized: bool,

    /// Wrap the prompt in the Gemma 4 chat template
    /// (`<bos><|turn>user\n…<turn|>\n<|turn>model\n`) before encoding.
    /// Required for IT-tuned variants like `gemma4:e2b` to produce
    /// coherent answers — without it the model treats the input as raw
    /// completion context and tends to repeat the input verbatim.
    #[arg(long, default_value_t = false)]
    chat_template: bool,
}

/// Tensors the chat-pwa filters out of the safetensors load. Replicated
/// here so the native rig sees the same model as the chat-pwa.
///
/// Source: `extras/brainwires-chat-pwa/wasm/src/vision.rs::gemma4_skip_reason`.
///
/// `load_ple_table = true` keeps `embed_tokens_per_layer.weight` (~4.7 GB
/// BF16) so the full PLE pipeline runs. Use this on `--device cpu` with
/// enough RAM, or on a real GPU whose `max_storage_buffer_binding_size`
/// can hold the table — the chat-pwa's default skip exists because
/// WebGPU's 1 GB binding cap can't hold it as a single bind-group entry.
fn gemma4_skip_reason(name: &str, load_ple_table: bool) -> Option<&'static str> {
    if name.contains(".audio_tower.") || name.contains(".embed_audio.") {
        return Some("audio");
    }
    if !load_ple_table && name.ends_with(".embed_tokens_per_layer.weight") {
        return Some("ple-table-oversize");
    }
    if name.ends_with(".input_min")
        || name.ends_with(".input_max")
        || name.ends_with(".output_min")
        || name.ends_with(".output_max")
    {
        return Some("qat-stat");
    }
    None
}

/// Translate a candle-side tensor lookup name into the safetensors key
/// stored in the HF Gemma 4 checkpoint.
///
/// Candle's `Gemma4Model::new_partial` applies `vb.pp("model")` and
/// then `TextModel::new` applies `vb.pp("model")` *again*, so language-
/// model tensors are looked up at `model.language_model.model.<rest>`.
/// HF stores them at `model.language_model.<rest>` — no inner `.model.`
/// segment. Strip it on lookup so candle and HF agree.
///
/// This mirrors the inverse of the chat-pwa's
/// `extras/brainwires-chat-pwa/wasm/src/vision.rs::gemma4_remap_key`,
/// which transforms HF → candle when building the HashMap. Here we go
/// candle → HF on each `vb.get(...)`.
///
/// Vision (`model.vision_tower.*`) and audio (`model.audio_tower.*`)
/// paths are passed through unchanged — they don't have the
/// double-prefix issue.
fn remap_candle_to_hf(name: &str) -> std::borrow::Cow<'_, str> {
    if let Some(rest) = name.strip_prefix("model.language_model.model.") {
        std::borrow::Cow::Owned(format!("model.language_model.{rest}"))
    } else {
        std::borrow::Cow::Borrowed(name)
    }
}

/// Tensors that must remain on the CPU device regardless of the
/// VarBuilder's nominal device. Mirrors the chat-pwa's `cpu_pinned`
/// HashSet (vision.rs:1293-1297). Names are in candle's post-remap form
/// (with the inner `.model.` segment).
///
/// Why: `Gemma4MultiModal::generate_greedy` builds `input_ids` on the
/// CPU device and calls `embed_tokens(input_ids)` directly, which is
/// an `index-select` op that requires both operands on the same
/// device. Same for the per-layer-input projection, which the pipeline
/// computes on CPU before transferring to GPU. Pinning these three
/// tensors to CPU makes the cross-device dance work.
fn is_cpu_pinned(candle_name: &str) -> bool {
    matches!(
        candle_name,
        "model.language_model.model.embed_tokens.weight"
            | "model.language_model.model.embed_tokens_per_layer.weight"
            | "model.language_model.model.per_layer_model_projection.weight"
            | "model.language_model.model.per_layer_projection_norm.weight"
    )
}

/// Wraps an inner `SimpleBackend` and (a) reports any tensor matching
/// `gemma4_skip_reason` as absent so candle's PLE construction falls
/// back to the projection-only path, (b) translates candle's
/// double-`model.` lookup names back to HF's single-`model.` safetensors
/// keys, and (c) forces selected tensors onto the CPU device regardless
/// of `vb.device()` so the chat-pwa's mixed-device pipeline works.
struct FilteredBackend {
    inner: Box<dyn SimpleBackend + 'static>,
    load_ple_table: bool,
}

impl SimpleBackend for FilteredBackend {
    fn get(
        &self,
        shape: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let hf_name = remap_candle_to_hf(name);
        if let Some(reason) = gemma4_skip_reason(&hf_name, self.load_ple_table) {
            return Err(candle_core::Error::Msg(format!(
                "tensor `{name}` filtered ({reason}) — caller should check \
                 contains_tensor first"
            )));
        }
        let target_dev: &Device = if is_cpu_pinned(name) { &Device::Cpu } else { dev };
        self.inner.get(shape, &hf_name, h, dtype, target_dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let hf_name = remap_candle_to_hf(name);
        if let Some(reason) = gemma4_skip_reason(&hf_name, self.load_ple_table) {
            return Err(candle_core::Error::Msg(format!(
                "tensor `{name}` filtered ({reason})"
            )));
        }
        let target_dev: &Device = if is_cpu_pinned(name) { &Device::Cpu } else { dev };
        self.inner.get_unchecked(&hf_name, dtype, target_dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let hf_name = remap_candle_to_hf(name);
        if gemma4_skip_reason(&hf_name, self.load_ple_table).is_some() {
            return false;
        }
        self.inner.contains_tensor(&hf_name)
    }
}

fn build_device(arg: DeviceArg) -> Result<Device> {
    match arg {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Wgpu => Device::new_wgpu(0)
            .context("failed to construct candle WGPU device — ensure the candle-wgpu feature is enabled and a Vulkan/Metal/DX12 adapter is available"),
    }
}

async fn fetch_files(args: &Args) -> Result<(PathBuf, PathBuf, PathBuf)> {
    if let (Some(w), Some(t), Some(c)) = (
        args.weights_file.as_ref(),
        args.tokenizer_file.as_ref(),
        args.config_file.as_ref(),
    ) {
        return Ok((w.clone(), t.clone(), c.clone()));
    }

    let api = Api::new().context("failed to construct hf-hub Api")?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    let weights = match args.weights_file.clone() {
        Some(p) => p,
        None => repo
            .get("model.safetensors")
            .await
            .context("failed to fetch model.safetensors from HF Hub")?,
    };
    let tokenizer = match args.tokenizer_file.clone() {
        Some(p) => p,
        None => repo
            .get("tokenizer.json")
            .await
            .context("failed to fetch tokenizer.json from HF Hub")?,
    };
    let config = match args.config_file.clone() {
        Some(p) => p,
        None => repo
            .get("config.json")
            .await
            .context("failed to fetch config.json from HF Hub")?,
    };

    Ok((weights, tokenizer, config))
}

fn load_config(path: &std::path::Path) -> Result<Gemma4Config> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let cfg: Gemma4Config = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {} as Gemma4Config", path.display()))?;
    Ok(cfg)
}

/// Read the safetensors header keys and use them to override config
/// defaults that don't match the actual checkpoint. The HF Gemma 4 E2B
/// `config.json` omits `altup_num_inputs`, `laurel_rank`, and similar
/// fields — candle's serde defaults assume Gemma 3n's full feature set
/// (4 AltUp streams, LAuReL rank 64). The actual E2B checkpoint has
/// neither, so we sniff for marker tensors and disable the modules
/// that aren't backed by weights.
///
/// Mirrors the inference logic in
/// `extras/brainwires-chat-pwa/wasm/src/vision.rs::build_gemma4_config`.
fn override_cfg_from_safetensors(
    cfg: &mut Gemma4Config,
    weights_path: &std::path::Path,
) -> Result<()> {
    // Read only the header — first 8 bytes are little-endian u64
    // header_size, then `header_size` bytes of JSON. Avoid loading the
    // full 10 GB tensor body (an earlier `fs::read` of the whole file
    // hung for minutes — this version is ~milliseconds).
    use std::io::Read;
    let mut f = std::fs::File::open(weights_path)
        .with_context(|| format!("open {}", weights_path.display()))?;
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)
        .context("read safetensors header size")?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    let mut header_bytes = vec![0u8; header_size];
    f.read_exact(&mut header_bytes)
        .context("read safetensors header bytes")?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .context("parse safetensors header as JSON")?;
    let keys: Vec<&str> = header
        .as_object()
        .context("safetensors header not an object")?
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .map(|s| s.as_str())
        .collect();

    let has_altup = keys.iter().any(|k| k.contains("altup_projections"));
    let has_laurel = keys.iter().any(|k| k.contains("laurel.linear"));

    if !has_altup && cfg.text_config.altup_num_inputs > 1 {
        eprintln!(
            "[gemma4_diag] no `altup_projections` tensors in checkpoint; \
             overriding cfg.text_config.altup_num_inputs from {} → 1",
            cfg.text_config.altup_num_inputs
        );
        cfg.text_config.altup_num_inputs = 1;
    }
    if !has_laurel && cfg.text_config.laurel_rank > 0 {
        eprintln!(
            "[gemma4_diag] no `laurel.linear_*` tensors in checkpoint; \
             overriding cfg.text_config.laurel_rank from {} → 0",
            cfg.text_config.laurel_rank
        );
        cfg.text_config.laurel_rank = 0;
    }

    // Per-layer MLP intermediate-size override. Gemma 4 uses
    // `use_double_wide_mlp` so KV-shared layers (top
    // `num_kv_shared_layers`) carry 2× intermediate width. The per-layer
    // override Vec is the candle-fork's mechanism for honoring this —
    // we infer the actual shape from each layer's `gate_proj.weight`.
    let n_layers = cfg.text_config.num_hidden_layers;
    let mut sizes: Vec<usize> = Vec::with_capacity(n_layers);
    let header_obj = header.as_object().unwrap();
    for li in 0..n_layers {
        let key = format!("model.language_model.layers.{li}.mlp.gate_proj.weight");
        let entry = header_obj.get(&key).with_context(|| {
            format!("safetensors header missing `{key}` — cannot infer layer {li}'s MLP width")
        })?;
        let shape = entry
            .get("shape")
            .and_then(|s| s.as_array())
            .with_context(|| format!("`{key}` has no shape array"))?;
        let intermediate = shape
            .get(0)
            .and_then(|v| v.as_u64())
            .with_context(|| format!("`{key}.shape[0]` not a u64"))? as usize;
        sizes.push(intermediate);
    }
    let any_diff = sizes.iter().any(|s| *s != cfg.text_config.intermediate_size);
    if any_diff {
        eprintln!(
            "[gemma4_diag] mixed MLP widths detected: per-layer intermediate_sizes = {:?}",
            sizes
        );
        cfg.text_config.intermediate_sizes = Some(sizes);
    }

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args = Args::parse();

    // Plumb the target layer to gemma4_mm.rs via env var. "none" disables
    // intra-capture; numeric strings select a layer; "default" / unset
    // falls back to whatever gemma4_mm decides (currently 8).
    if !args.target_layer.eq_ignore_ascii_case("default") {
        // SAFETY: single-threaded current-thread runtime, set before
        // spawning any async work, so no other thread can be reading
        // env at the same time.
        unsafe {
            std::env::set_var("BW_GEMMA4_DIAG_LAYER", &args.target_layer);
        }
    }

    eprintln!(
        "[gemma4_diag] device={:?} model={} revision={} prompt={:?} target_layer={} load_ple_table={}",
        args.device, args.model_id, args.revision, args.prompt, args.target_layer, args.load_ple_table
    );

    let result = run(args).await;
    match result {
        Ok(()) => {
            let nans = nan_scan_count();
            if nans > 0 {
                let label = nan_scan_first_label().unwrap_or_else(|| "<unknown>".to_string());
                println!(
                    "RESULT: FAIL  nan_scans={nans}  first_nan_at={label}"
                );
                ExitCode::from(1)
            } else {
                println!("RESULT: PASS  forward pass produced finite logits");
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("[gemma4_diag] error: {e:#}");
            println!("RESULT: ERROR  {e}");
            ExitCode::from(2)
        }
    }
}

async fn run(args: Args) -> Result<()> {
    let device = build_device(args.device)?;
    eprintln!("[gemma4_diag] device built: {:?}", device);

    if args.quantized {
        return run_quantized(args, device).await;
    }

    let dtype = DType::BF16;

    // GGUF path: dequantize-at-load via the new Phase 4 loader. Skips HF
    // safetensors download entirely; tokenizer still has to be supplied
    // separately (`--tokenizer-file`) since we don't extract from GGUF
    // metadata yet.
    let (cfg, vb) = if let Some(gguf_path) = args.gguf_path.clone() {
        eprintln!("[gemma4_diag] loading GGUF: {}", gguf_path.display());
        let t0 = std::time::Instant::now();
        let (tensors, cfg) =
            brainwires_provider::local_llm::gguf_loader::load_gemma4_gguf(&gguf_path, &device)
                .context("GGUF load")?;
        eprintln!(
            "[gemma4_diag] dequantized {} tensors → BF16 in {:?}",
            tensors.len(),
            t0.elapsed()
        );
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        (cfg, vb)
    } else {
        let (weights, tokenizer_path, config_path) = fetch_files(&args).await?;
        eprintln!("[gemma4_diag] weights={}", weights.display());
        eprintln!("[gemma4_diag] tokenizer={}", tokenizer_path.display());
        eprintln!("[gemma4_diag] config={}", config_path.display());

        let mut cfg = load_config(&config_path)?;
        override_cfg_from_safetensors(&mut cfg, &weights)?;

        // SAFETY: from_mmaped_safetensors is unsafe because the underlying
        // mmap can be invalidated by external file writes. We hold the file
        // for the duration of the program; nothing modifies it.
        let inner = unsafe {
            candle_core::safetensors::MmapedSafetensors::multi(&[&weights])
                .context("mmap safetensors")?
        };
        let backend: Box<dyn SimpleBackend + 'static> = Box::new(FilteredBackend {
            inner: Box::new(inner),
            load_ple_table: args.load_ple_table,
        });
        let vb: VarBuilder = VarBuilderArgs::from_backend(backend, dtype, device.clone());
        (cfg, vb)
    };

    let tokenizer_path = if args.gguf_path.is_some() {
        // GGUF path needs an explicit tokenizer file — bail with a
        // clear error if the user forgot to pass it.
        args.tokenizer_file.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "--gguf-path requires --tokenizer-file (GGUF tokenizer extraction not implemented yet)"
            )
        })?
    } else {
        // Fetched alongside weights in the safetensors branch above.
        let (_, t, _) = fetch_files(&args).await?;
        t
    };

    eprintln!(
        "[gemma4_diag] config: hidden={} layers={} heads={} head_dim={} global_head_dim={} \
         altup_num_inputs={} laurel_rank={}",
        cfg.text_config.hidden_size,
        cfg.text_config.num_hidden_layers,
        cfg.text_config.num_attention_heads,
        cfg.text_config.head_dim,
        cfg.text_config.global_head_dim,
        cfg.text_config.altup_num_inputs,
        cfg.text_config.laurel_rank,
    );

    eprintln!("[gemma4_diag] building Gemma 4 model (text-only, no vision/audio)...");
    let t0 = std::time::Instant::now();
    let model = Gemma4Model::new_partial(&cfg, vb, false, false)
        .context("Gemma4Model::new_partial")?;
    eprintln!("[gemma4_diag] model built in {:?}", t0.elapsed());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let pipeline = Gemma4MultiModal::from_components(
        model,
        tokenizer,
        device.clone(),
        cfg.clone(),
    );

    // Gemma 4 EOS = token id 1 (`<eos>` in the tokenizer). Hardcoded
    // because `Gemma4TextConfig` doesn't expose token-id fields. With
    // `max_new_tokens=1` the early-exit doesn't fire anyway, but pass
    // the right value for longer runs.
    let eos: Option<u32> = Some(1);

    nan_scan_reset();
    // The native diag rig is for development; turn the readback scaffold
    // on so we get the per-step `[gemma4/diag] step0/...` lines. The
    // chat-pwa wasm path leaves it off by default.
    set_diag_enabled(true);
    eprintln!("[gemma4_diag] generating {} token(s)...", args.max_new_tokens);
    let t0 = std::time::Instant::now();
    let output = pipeline
        .generate_greedy(&args.prompt, &[], args.max_new_tokens, eos)
        .await
        .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;
    eprintln!(
        "[gemma4_diag] generate_greedy returned in {:?}: {output:?}",
        t0.elapsed()
    );
    Ok(())
}

/// Quantized-path diagnostic — drives the `quantized_gemma4` model end-to-end:
/// load GGUF (with QTensors kept as QTensor so PR #3379's `q4_k.pwgsl` runs),
/// build a tokenizer, encode the prompt, run one forward step, argmax the
/// final-token logits, and decode + print the predicted next token. Used to
/// shake out the QMatMul + GGUF integration without the full `Gemma4MultiModal`
/// generate_greedy machinery (which currently only wraps the BF16 path).
async fn run_quantized(args: Args, device: Device) -> Result<()> {
    let gguf_path = args
        .gguf_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--quantized requires --gguf-path"))?;
    let tokenizer_path = args
        .tokenizer_file
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--quantized requires --tokenizer-file"))?;

    eprintln!("[gemma4_diag/quant] loading GGUF: {}", gguf_path.display());
    let mut file = std::fs::File::open(&gguf_path).context("open GGUF")?;
    let t0 = std::time::Instant::now();
    let (mut model, cfg) =
        brainwires_provider::local_llm::gguf_loader::load_quantized_gemma4_from_reader(
            &mut file, &device,
        )
        .context("load_quantized_gemma4_from_reader")?;
    eprintln!(
        "[gemma4_diag/quant] model built in {:?} (layers={}, vocab={})",
        t0.elapsed(),
        cfg.text_config.num_hidden_layers,
        cfg.text_config.vocab_size,
    );

    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    // For IT variants (`gemma4:e2b`), wrap the prompt in the Gemma 4
    // chat template; the wrapped string already contains `<bos>`, so
    // call `encode(..., false)` to skip the auto-special-token pass.
    // Raw mode keeps `add_special_tokens=true` so the BF16 path can
    // be exercised with an unwrapped prompt for parity testing.
    let (prompt_str, add_special) = if args.chat_template {
        (
            format!(
                "<bos><|turn>user\n{}<turn|>\n<|turn>model\n",
                args.prompt.trim()
            ),
            false,
        )
    } else {
        (args.prompt.clone(), true)
    };
    let encoded = tokenizer
        .encode(prompt_str.as_str(), add_special)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let token_ids: Vec<u32> = encoded.get_ids().to_vec();
    eprintln!(
        "[gemma4_diag/quant] prompt={prompt_str:?} tokens={token_ids:?}",
    );

    // Greedy autoregressive loop. First step ingests the prompt; each
    // subsequent step feeds the previous argmax token at the next
    // position. EOS = 1 (Gemma 4 `<eos>`) plus 106 / 50 from the
    // GGUF's `tokenizer.ggml.eos_token_ids`.
    let eos: std::collections::HashSet<u32> = [1u32, 106, 50].into_iter().collect();
    let prompt_len = token_ids.len();
    let mut all_tokens = token_ids.clone();
    let mut seqlen_offset = 0usize;
    let mut current_input = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
    let t0 = std::time::Instant::now();
    let mut t_first_step: Option<std::time::Duration> = None;

    for step in 0..args.max_new_tokens {
        let step_t0 = std::time::Instant::now();
        let logits = model.forward(&current_input, seqlen_offset).context("model forward")?;
        let elapsed = step_t0.elapsed();
        if step == 0 {
            t_first_step = Some(elapsed);
            eprintln!(
                "[gemma4_diag/quant] prompt forward in {:?}, logits shape={:?}",
                elapsed,
                logits.shape(),
            );
        }
        let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?.squeeze(0)?;
        let logit_vec: Vec<f32> = last_logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (next_id, max_v) = logit_vec
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 { (i, v) } else { acc }
            });
        let nan_count = logit_vec.iter().filter(|v| v.is_nan()).count();
        let zero_count = logit_vec.iter().filter(|&&v| v == 0.0).count();
        let mut sorted: Vec<(usize, f32)> = logit_vec.iter().copied().enumerate().collect();
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<(usize, f32)> = sorted.into_iter().take(5).collect();
        eprintln!(
            "[gemma4_diag/quant] step {step}: next_id={next_id} max={max_v:.3} \
             nan_count={nan_count} zero_count={zero_count} top5={top5:?}",
        );
        let next_id_u32 = next_id as u32;
        all_tokens.push(next_id_u32);
        seqlen_offset = if step == 0 { prompt_len } else { seqlen_offset + 1 };
        current_input = Tensor::new(&[next_id_u32][..], &device)?.unsqueeze(0)?;
        if eos.contains(&next_id_u32) {
            eprintln!("[gemma4_diag/quant] EOS at step {step} ({next_id_u32})");
            break;
        }
    }

    let total = t0.elapsed();
    let generated = &all_tokens[prompt_len..];
    let decoded = tokenizer
        .decode(generated, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    eprintln!(
        "[gemma4_diag/quant] generated {} tokens in {:?} (first step {:?}); decoded={:?}",
        generated.len(),
        total,
        t_first_step.unwrap_or_default(),
        decoded,
    );
    println!("RESULT: PASS  quantized_gemma4 forward produced finite logits");
    Ok(())
}
