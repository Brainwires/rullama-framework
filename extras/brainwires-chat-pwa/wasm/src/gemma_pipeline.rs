//! Gemma-family multimodal pipeline glue for the chat PWA.
//!
//! Despite "multimodal" in the name, this module is the entry point for
//! Gemma 3/4 inference whether or not images are attached — the
//! `Gemma4MultiModal::generate_greedy(prompt, &[], …)` (empty pixel_values)
//! path is what the chat-pwa uses for *all* Gemma 4 chat, because that's
//! where the streaming callback, OPFS PLE row reader, and chunked loader
//! plumbing live.
//!
//! What's actually in here:
//!
//! - **Chunked safetensors loader** — splits a 10 GB model file into
//!   per-tensor materialization so we never allocate the whole thing
//!   in wasm32 linear memory.
//! - **`OpfsPerLayerEmbedTable`** — Per-Layer Embedding row reader
//!   backed by OPFS sync-access reads. The PLE table is a 4.7 GB text-
//!   path tensor that doesn't fit in a WebGPU buffer, so we stream rows
//!   on demand. (Not vision; it's part of the Gemma 3n decoder.)
//! - **`build_gemma_chat_prompt`** — Gemma 4 chat-template formatter.
//!   Pure text. Lives here because the streaming entry point does too.
//! - **Vision pipeline (Gemma 3 SigLIP / Gemma 4 native vision tower)** —
//!   only used when images are attached; gated by `gemma4_skip_reason`
//!   filters so the vision-tower tensors don't load in text-only mode.
//!
//! Companion to the text-only [`crate::LocalModelHandle`] surface in `lib.rs`.
//! Exposes a JS-callable `local_chat_stream_with_image(handle, messages_json,
//! params_json)` that emits the same NDJSON `ReadableStream<Uint8Array>`
//! shape the text path uses — making the JS-side dispatcher in
//! `local-worker.js` route-agnostic.
//!
//! Supports two model architectures, auto-detected from safetensors
//! tensor names during chunked loading:
//!
//! **Gemma-3** (SigLIP-based): SigLIP tower + MM projector + vendored Gemma3 decoder
//! **Gemma-4** (native vision tower): Gemma4 VisionTower + MultimodalEmbedder + upstream Gemma4 TextModel

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use brainwires_provider::local_llm::candle_provider::default_gemma_e2b_config;
use brainwires_provider::local_llm::vision::{
    Gemma3MultiModal, ImageInput, MmPipelineError, MultiModalProjector, PROJECTOR_DEFAULT_EPS,
    SiglipVisionTower, preprocess_image_bytes,
};
use brainwires_provider::local_llm::vision::gemma3_mm::{
    Config as Gemma3MmConfig, Model as Gemma3MmModel,
};
use brainwires_provider::local_llm::vision::{
    Gemma4MultiModal, Gemma4PipelineError, preprocess_image_for_gemma4,
};
use brainwires_provider::gemma4::config::Gemma4Config;
use brainwires_provider::gemma4::Model as Gemma4Model;
use brainwires_provider::{
    CandleDType as DType, CandleDevice as Device, CandleTensor as Tensor, CandleVarBuilder,
};
use candle_nn::Activation;
use js_sys::{Function, Object, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{ReadableStream, ReadableStreamDefaultController};

use crate::{StTensorInfo, call_read_fn, js_err_to_string, load_tensor_to_gpu, st_dtype_to_candle};

// ---------------------------------------------------------------------------
// Multimodal handle
// ---------------------------------------------------------------------------

/// Detected model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Gemma3,
    Gemma4,
}

/// Inner pipeline — either Gemma-3 (SigLIP-based) or Gemma-4 (native vision tower).
enum MultimodalInner {
    Gemma3(Arc<Gemma3MultiModal>),
    Gemma4 {
        pipeline: Arc<Gemma4MultiModal>,
        gpu_device: Device,
    },
}

/// Per-handle state that survives init for on-demand `attach_vision` /
/// `attach_audio` calls. Holds enough context to re-read tensors from the
/// original safetensors file/blob and merge them into the loaded model.
///
/// Lives only on the wasm-bindgen handle (single-threaded), because
/// [`js_sys::Function`] is not `Send` and cannot sit behind `Arc<Mutex<…>>`.
struct Gemma4LazyState {
    read_fn: Function,
    tensor_meta: Vec<(String, StTensorInfo)>,
    data_start: u64,
    cfg: brainwires_provider::gemma4::config::Gemma4Config,
    device: Device,
    wgpu_dev: Option<brainwires_provider::WgpuDevice>,
}

/// Multimodal Gemma handle. Loaded separately from the text-only
/// [`crate::LocalModelHandle`] because the safetensors file structure differs
/// (text-only vs full vision-language weights). The JS-side worker tracks
/// which shape was loaded and routes `chat` vs `vision_chat` accordingly.
///
/// Disposal: wasm-bindgen autogenerates a JS-side `free()`. Calling it
/// drops the inner pipeline (and, if last reference, all model weights).
#[wasm_bindgen]
pub struct LocalMultiModalHandle {
    inner: MultimodalInner,
    model_id: String,
    /// Present iff this is a Gemma4 handle whose vision and/or audio tower
    /// was deferred at init. Consumed to drive `attach_vision`/`attach_audio`.
    lazy: Option<Gemma4LazyState>,
    /// `"hf"` for safetensors-loaded handles, `"gguf"` for ones built via
    /// `init_local_multimodal_gguf`. Read by JS to render the right UI
    /// badge / route quantization-specific diagnostics.
    source: &'static str,
}

#[wasm_bindgen]
impl LocalMultiModalHandle {
    #[wasm_bindgen(getter)]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn device_type(&self) -> String {
        match &self.inner {
            MultimodalInner::Gemma3(p) => {
                match p.device().location() {
                    brainwires_provider::CandleDeviceLocation::Cpu => "cpu".into(),
                    brainwires_provider::CandleDeviceLocation::Wgpu { .. } => "webgpu".into(),
                    _ => "unknown".into(),
                }
            }
            MultimodalInner::Gemma4 { gpu_device, .. } => {
                match gpu_device.location() {
                    brainwires_provider::CandleDeviceLocation::Cpu => "cpu".into(),
                    brainwires_provider::CandleDeviceLocation::Wgpu { .. } => "webgpu".into(),
                    _ => "unknown".into(),
                }
            }
        }
    }

    #[wasm_bindgen(getter)]
    pub fn is_multimodal(&self) -> bool {
        true
    }

    /// Whether the vision tower is currently attached. Returns `true` for
    /// Gemma3 (always loaded eagerly) and reflects actual state for Gemma4.
    #[wasm_bindgen(getter)]
    pub fn has_vision(&self) -> bool {
        match &self.inner {
            MultimodalInner::Gemma3(_) => true,
            MultimodalInner::Gemma4 { pipeline, .. } => pipeline.has_vision(),
        }
    }

    /// Whether the audio tower is currently attached.
    #[wasm_bindgen(getter)]
    pub fn has_audio(&self) -> bool {
        match &self.inner {
            MultimodalInner::Gemma3(_) => false,
            MultimodalInner::Gemma4 { pipeline, .. } => pipeline.has_audio(),
        }
    }

    /// `"hf"` (safetensors) or `"gguf"` (Ollama-format dequant-at-load).
    #[wasm_bindgen(getter)]
    pub fn source(&self) -> String {
        self.source.into()
    }

    /// Stream the vision-tower tensors from the original safetensors file
    /// and attach them to the loaded Gemma4 model. Idempotent; a no-op if
    /// vision is already attached.
    ///
    /// Errors if the handle is Gemma3 (vision is always eager there) or if
    /// no lazy state was retained at init.
    pub async fn attach_vision(&self) -> Result<(), JsValue> {
        let pipeline = match &self.inner {
            MultimodalInner::Gemma3(_) => {
                return Err(JsValue::from_str(
                    "attach_vision is only supported for Gemma4 handles",
                ));
            }
            MultimodalInner::Gemma4 { pipeline, .. } => pipeline.clone(),
        };
        if pipeline.has_vision() {
            return Ok(());
        }
        let lazy = self.lazy.as_ref().ok_or_else(|| {
            JsValue::from_str("attach_vision: handle has no retained lazy state")
        })?;
        let vb = build_subset_var_builder(lazy, is_vision_tensor)?;
        pipeline
            .attach_vision(vb)
            .map_err(|e| JsValue::from_str(&format!("attach_vision: {e}")))?;
        web_sys::console::log_1(&"[wasm/mm] vision tower attached".into());
        Ok(())
    }

    /// Stream the audio-tower tensors from the original safetensors file
    /// and attach them to the loaded Gemma4 model. Idempotent.
    ///
    /// Currently errors with "audio config not inferred" — `build_gemma4_config`
    /// synthesizes `audio_config: None`, so the model has no audio shapes
    /// to validate against. Wiring this end-to-end requires inferring the
    /// `Gemma4AudioConfig` from `audio_tower.*` tensor shapes; the candle-fork
    /// side (`Gemma4Model::attach_audio`) is already in place for when that
    /// inference lands.
    pub async fn attach_audio(&self) -> Result<(), JsValue> {
        let pipeline = match &self.inner {
            MultimodalInner::Gemma3(_) => {
                return Err(JsValue::from_str(
                    "attach_audio is only supported for Gemma4 handles",
                ));
            }
            MultimodalInner::Gemma4 { pipeline, .. } => pipeline.clone(),
        };
        if pipeline.has_audio() {
            return Ok(());
        }
        let lazy = self.lazy.as_ref().ok_or_else(|| {
            JsValue::from_str("attach_audio: handle has no retained lazy state")
        })?;
        if lazy.cfg.audio_config.is_none() {
            return Err(JsValue::from_str(
                "attach_audio: audio_config not inferred from tensor shapes — \
                 audio support is not yet wired through the Gemma4 config builder",
            ));
        }
        let vb = build_subset_var_builder(lazy, is_audio_tensor)?;
        pipeline
            .attach_audio(vb)
            .map_err(|e| JsValue::from_str(&format!("attach_audio: {e}")))?;
        web_sys::console::log_1(&"[wasm/mm] audio tower attached".into());
        Ok(())
    }
}

/// Stream the subset of tensors matching `predicate` from the safetensors
/// file pointed at by `lazy.read_fn`, run them through `gemma4_remap_key`,
/// and return a `CandleVarBuilder` ready for `Model::attach_vision/audio`.
fn build_subset_var_builder<'a>(
    lazy: &'a Gemma4LazyState,
    predicate: fn(&str) -> bool,
) -> Result<CandleVarBuilder<'a>, JsValue> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut total_bytes: u64 = 0;
    let mut count: usize = 0;
    for (name, info) in &lazy.tensor_meta {
        if !predicate(name) {
            continue;
        }
        if gemma4_skip_reason(name).is_some() {
            // QAT activation stats etc. — skip even on lazy attach.
            continue;
        }
        let tensor = load_one_tensor(
            name,
            info,
            lazy.data_start,
            &lazy.read_fn,
            &lazy.device,
            lazy.wgpu_dev.as_ref(),
            false,
        )?;
        let key = gemma4_remap_key(name);
        let length = info.data_offsets.1 - info.data_offsets.0;
        total_bytes += length;
        count += 1;
        tensors.insert(key, tensor);
    }
    web_sys::console::log_1(
        &format!(
            "[wasm/mm] attach: streamed {count} tensors ({:.2} MB)",
            total_bytes as f64 / 1_048_576.0,
        )
        .into(),
    );
    Ok(CandleVarBuilder::from_tensors(
        tensors,
        DType::BF16,
        &lazy.device,
    ))
}

// ---------------------------------------------------------------------------
// Loader (bulk-read only)
// ---------------------------------------------------------------------------

/// Conservative default Gemma 4 E2B vision-language config. Mirrors
/// [`default_gemma_e2b_config`] but as the vendored
/// [`Gemma3MmConfig`] type — they have the same fields, but live in
/// different crates (the vendored decoder needed an `embed_tokens()` accessor
/// the upstream `Model` doesn't expose).
fn default_gemma_e2b_mm_config() -> Gemma3MmConfig {
    let txt = default_gemma_e2b_config();
    Gemma3MmConfig {
        attention_bias: txt.attention_bias,
        head_dim: txt.head_dim,
        // Upstream `Activation` type is the same one re-exported by
        // candle-nn; Gemma 4 E2B uses GeluPytorchTanh.
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size: txt.hidden_size,
        intermediate_size: txt.intermediate_size,
        num_attention_heads: txt.num_attention_heads,
        num_hidden_layers: txt.num_hidden_layers,
        num_key_value_heads: txt.num_key_value_heads,
        rms_norm_eps: txt.rms_norm_eps,
        rope_theta: txt.rope_theta,
        rope_local_base_freq: txt.rope_local_base_freq,
        vocab_size: txt.vocab_size,
        final_logit_softcapping: txt.final_logit_softcapping,
        attn_logit_softcapping: txt.attn_logit_softcapping,
        query_pre_attn_scalar: txt.query_pre_attn_scalar,
        sliding_window: txt.sliding_window,
        sliding_window_pattern: txt.sliding_window_pattern,
        max_position_embeddings: txt.max_position_embeddings,
    }
}

/// SigLIP-So400m hidden size used by the `paligemma_3b_896` preset that the
/// [`SiglipVisionTower`] wraps. Hardcoded because the preset itself is
/// hardcoded — no field on the wrapper exposes it pre-load.
const SIGLIP_HIDDEN: usize = 1152;

// ---------------------------------------------------------------------------
// Model type detection + Gemma4 config inference
// ---------------------------------------------------------------------------

fn detect_model_type(tensor_meta: &[(String, StTensorInfo)]) -> ModelType {
    let has_gemma4_vision = tensor_meta
        .iter()
        .any(|(n, _)| n.contains("vision_tower.patch_embedder"));
    if has_gemma4_vision {
        ModelType::Gemma4
    } else {
        ModelType::Gemma3
    }
}

/// Build a [`Gemma4Config`] by inspecting tensor shapes in the safetensors header.
///
/// `metadata` is the safetensors `__metadata__` blob (or an empty map when
/// the file omits it) — `activation_sparsity_pattern` is read from there if
/// the trainer embedded it, otherwise we fall back to the canonical
/// `[0.95; 10] + [0.0; rest]` Gemma 3n heuristic.
fn build_gemma4_config(
    tensor_meta: &[(String, StTensorInfo)],
    metadata: &HashMap<String, String>,
    options: &LoadOptions,
) -> Result<Gemma4Config, String> {
    let find = |suffix: &str| -> Option<&Vec<usize>> {
        tensor_meta
            .iter()
            .find(|(n, _)| n.ends_with(suffix))
            .map(|(_, info)| &info.shape)
    };

    // hidden_size from embed_tokens.weight [vocab_size, hidden_size]
    let embed_shape = find("embed_tokens.weight")
        .ok_or("missing embed_tokens.weight")?;
    let vocab_size = embed_shape[0];
    let hidden_size = embed_shape[1];

    // intermediate_size from first layer's gate_proj [intermediate_size, hidden_size].
    // Gemma4-E2B (Gemma 3n) ships with elastic MLP widths — some layers
    // are 6144-wide, others 12288-wide — so we also build a per-layer
    // override table below. The scalar `intermediate_size` is the layer-0
    // value and is used as a fallback for any layer whose gate_proj is
    // missing from the safetensors index (shouldn't happen in practice).
    let intermediate_size = find("language_model.layers.0.mlp.gate_proj.weight")
        .map(|s| s[0])
        .ok_or("missing layers.0.mlp.gate_proj.weight")?;

    // num_hidden_layers: count unique layer indices
    let num_hidden_layers = tensor_meta
        .iter()
        .filter_map(|(n, _)| {
            let rest = n.strip_prefix("model.language_model.layers.")?;
            rest.split('.').next()?.parse::<usize>().ok()
        })
        .max()
        .map(|m| m + 1)
        .ok_or("no decoder layers found")?;

    // num_attention_heads from q_proj.weight shape[0] / head_dim
    // For Gemma4, global layers use global_head_dim and sliding layers use head_dim.
    // Detect from layer 0's q_proj.
    let q_proj_shape = find("language_model.layers.0.self_attn.q_proj.weight")
        .ok_or("missing layers.0 q_proj")?;
    let kv_proj_shape = find("language_model.layers.0.self_attn.k_proj.weight")
        .ok_or("missing layers.0 k_proj")?;

    // Infer layer_types by checking each layer's q_proj shape.
    // Sliding layers have q_proj [num_heads * head_dim, hidden_size]
    // Global layers have q_proj [num_heads * global_head_dim, hidden_size]
    // The first layer is typically sliding in Gemma4's alternating pattern.
    let layer0_q_out = q_proj_shape[0];

    // Gemma 3n's layer alternation is 4 sliding + 1 full. Layer 0 /
    // layer 1 are both sliding, so comparing just those two q_proj
    // widths can't distinguish sliding `head_dim` from the global
    // `head_dim` used by full_attention layers. Instead, fix layer 0
    // as the sliding sample and scan every other layer's q_proj for
    // the first one whose width differs — that's the global. If no
    // layer differs, the model is uniformly head_dim and there is no
    // separate `global_head_dim`.
    const NUM_HEADS: usize = 8; // Gemma 3n / Gemma 4 default
    let head_dim = layer0_q_out / NUM_HEADS;
    let mut global_head_dim = head_dim;
    for i in 1..num_hidden_layers {
        let key = format!("language_model.layers.{i}.self_attn.q_proj.weight");
        if let Some(shape) = tensor_meta
            .iter()
            .find(|(n, _)| n.ends_with(&key))
            .map(|(_, info)| &info.shape)
        {
            let q_out = shape[0];
            if q_out != layer0_q_out {
                global_head_dim = q_out / NUM_HEADS;
                break;
            }
        }
    }
    let num_attention_heads = NUM_HEADS;

    let num_key_value_heads = kv_proj_shape[0] / head_dim;

    // Build layer_types: check each layer's q_proj output dimension
    let sliding_q_out = num_attention_heads * head_dim;
    let mut layer_types = Vec::with_capacity(num_hidden_layers);
    for i in 0..num_hidden_layers {
        let key = format!("language_model.layers.{i}.self_attn.q_proj.weight");
        let is_sliding = tensor_meta
            .iter()
            .find(|(n, _)| n.ends_with(&key))
            .map(|(_, info)| info.shape[0] == sliding_q_out)
            .unwrap_or(i % 2 == 0); // fallback: even=sliding
        layer_types.push(if is_sliding {
            "sliding_attention".to_string()
        } else {
            "full_attention".to_string()
        });
    }

    // Build the per-layer intermediate_size table by reading each layer's
    // gate_proj.weight shape — the first dim is that layer's MLP width.
    // Required for Gemma4-E2B's elastic MLP layout (mix of 6144 and 12288).
    let mut intermediate_sizes_vec = Vec::with_capacity(num_hidden_layers);
    for i in 0..num_hidden_layers {
        let key = format!("language_model.layers.{i}.mlp.gate_proj.weight");
        let size = tensor_meta
            .iter()
            .find(|(n, _)| n.ends_with(&key))
            .map(|(_, info)| info.shape[0])
            .unwrap_or(intermediate_size);
        intermediate_sizes_vec.push(size);
    }
    // Only populate the override when widths actually vary; otherwise the
    // scalar field carries the same information and we keep the config
    // payload smaller.
    let intermediate_sizes = if intermediate_sizes_vec
        .iter()
        .any(|&s| s != intermediate_size)
    {
        Some(intermediate_sizes_vec)
    } else {
        None
    };

    use brainwires_provider::gemma4::config::*;

    Ok(Gemma4Config {
        text_config: Gemma4TextConfig {
            attention_bias: false,
            head_dim,
            hidden_activation: Activation::GeluPytorchTanh,
            hidden_size,
            intermediate_size,
            intermediate_sizes,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size,
            // Per the canonical Gemma 3n / Gemma 4 config.json. The earlier
            // value (4096) was a Gemma 2 carry-over and over-extended the
            // RotatingKvCache + sliding-attention mask span on local layers.
            sliding_window: 512,
            // Real config: `final_logit_softcapping: 30.0` (the Gemma 2
            // soft-cap returned in 3n alongside QK-norm). Without it the
            // last-layer logits aren't squashed to the trained range and
            // the sampler sees out-of-distribution magnitudes.
            final_logit_softcapping: Some(30.0),
            query_pre_attn_scalar: head_dim,
            max_position_embeddings: 32768,
            tie_word_embeddings: true,
            // 4 sliding + 1 full per group (Gemma 3n / Gemma 4 layer_types
            // pattern), down from Gemma 3's 5+1.
            sliding_window_pattern: 5,
            layer_types,
            global_head_dim,
            num_global_key_value_heads: None,
            rope_parameters: None,
            use_bidirectional_attention: None,
            use_flash_attn: false,
            // Gemma 3n PLE — width of the per-layer auxiliary input
            // (256 for E2B / E4B per the canonical config). When the
            // safetensors index includes `embed_tokens_per_layer.weight`
            // we infer the actual width from its shape; otherwise we
            // disable PLE so non-Gemma3n configs keep working.
            hidden_size_per_layer_input: find("language_model.embed_tokens_per_layer.weight")
                .map(|s| s[1] / num_hidden_layers),
            vocab_size_per_layer_input: find("language_model.embed_tokens_per_layer.weight")
                .map(|s| s[0]),
            // Gemma 3n AltUp — 4 parallel hidden streams. When the
            // safetensors index includes the `altup_projections.0.weight`
            // tensor we know the model carries AltUp; otherwise we
            // disable it (`altup_num_inputs = 1` collapses the stack
            // path back to the classic single-stream forward).
            altup_num_inputs: find("language_model.altup_projections.0.weight")
                .map(|_| 4)
                .unwrap_or(1),
            altup_active_idx: 0,
            altup_correct_scale: true,
            altup_coef_clip: Some(120.0),
            // Gemma 3n LAuReL low-rank residual — `laurel_rank: 64` per
            // the canonical config.
            laurel_rank: find("language_model.layers.0.laurel.linear_left.weight")
                .map(|s| s[0])
                .unwrap_or(64),
            // Gemma 3n activation sparsity — first 10 layers train at
            // 0.95 (zero the bottom 95% of gate_proj per-token), the
            // rest run dense. Only enable when AltUp is also wired
            // (i.e. we're loading a real Gemma 3n checkpoint).
            // Prefer the metadata pattern (trainer-supplied) when present.
            // Falls back to the canonical Gemma 3n heuristic — first 10
            // layers at 0.95 sparsity, rest at 0 — which matches the E2B
            // checkpoint's published config and is only emitted when AltUp
            // is wired (`altup_projections.0.weight` present) and the model
            // is wide enough for the cutoff to make sense.
            activation_sparsity_pattern: metadata
                .get("activation_sparsity_pattern")
                .and_then(|s| serde_json::from_str::<Vec<f64>>(s).ok())
                .filter(|v| v.len() == num_hidden_layers)
                .or_else(|| {
                    if find("language_model.altup_projections.0.weight").is_some()
                        && num_hidden_layers >= 10
                    {
                        let mut v = Vec::with_capacity(num_hidden_layers);
                        for i in 0..num_hidden_layers {
                            v.push(if i < 10 { 0.95 } else { 0.0 });
                        }
                        Some(v)
                    } else {
                        None
                    }
                }),
            // Bisection kill-switches default off; surfaced via LoadOptions
            // so the JS side can flip them in-browser when chasing a regression.
            disable_altup: options.disable_altup,
            disable_laurel: options.disable_laurel,
            disable_per_layer_input_gate: options.disable_per_layer_input_gate,
            // KV-cache sharing: last N layers re-use K/V from earlier
            // donor layers of the same attention type (sliding vs full).
            // Per the trained-checkpoint config:
            //   - Gemma 4 E2B / E4B (35 layers) → 20 shared (donors 0..14)
            //   - Gemma 3n E2B (30 layers)      → 10 shared (donors 0..19)
            //   - other layouts                 → 0 (KV-share off)
            // Picking the wrong value silently re-routes layer 15 from
            // the receiver branch to the donor branch and the model
            // produces gibberish on the WGPU/AMD path because layer-15
            // k_proj weights in the safetensors are receiver-shape
            // placeholders that the donor branch then matmuls against
            // the wrong inputs. Mac diag binary parses config.json
            // directly so it gets the right value; the wasm side has
            // to derive it from the layout.
            num_kv_shared_layers: match num_hidden_layers {
                35 => 20,
                30 => 10,
                _ => 0,
            },
        },
        vision_config: Gemma4VisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 16,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            hidden_activation: Activation::GeluPytorchTanh,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            position_embedding_size: 10240,
            pooling_kernel_size: 3,
            default_output_length: 280,
            standardize: false,
            rope_parameters: None,
            // Canonical Gemma 3n vision encoder is MobileNetV5 (300M).
            // Read the value from safetensors `__metadata__` if the
            // trainer embedded it, otherwise default to the canonical
            // string so `attach_vision` can route by architecture.
            architecture: Some(
                metadata
                    .get("vision_architecture")
                    .cloned()
                    .unwrap_or_else(|| "mobilenetv5_300m_enc".to_string()),
            ),
        },
        audio_config: None,
        image_token_id: 258880,
        audio_token_id: 258881,
        video_token_id: 258884,
    })
}

/// Tensors present in the Gemma 3n / Gemma4 QAT safetensors that the
/// candle-fork [`Gemma4Model`] either does not reference or cannot
/// physically load on this target. Skipping them avoids loading audio
/// tower weights when audio is disabled, dropping the QAT input/output
/// min/max statistics (training-time only), and dropping the multi-GB
/// `embed_tokens_per_layer` tensor that exceeds WebGPU's 1 GB
/// max-buffer-size and wasm32's 2 GB `isize::MAX` Vec limit.
///
/// PLE companion tensors (`per_layer_input_gate`, `per_layer_projection`,
/// `post_per_layer_input_norm`, `per_layer_model_projection`,
/// `per_layer_projection_norm`, `layer_scalar`) flow through the loader
/// normally — only the giant per-layer embedding *table* is dropped.
/// The candle-fork `PerLayerEmbedding` handles the missing table by
/// degrading the merge to `per_layer_input = per_layer_proj * rsqrt(2)`.
///
/// Returns `Some(reason)` if the tensor should be skipped, `None` otherwise.
fn gemma4_skip_reason(name: &str) -> Option<&'static str> {
    if name.contains(".audio_tower.") {
        return Some("audio");
    }
    // The per-layer embedding table is `vocab × num_layers × hidden_per_layer`
    // — ~4.7 GB bf16 on the E2B checkpoint. wasm32 + WebGPU can't hold it
    // as a single contiguous buffer.
    if name.ends_with(".embed_tokens_per_layer.weight") {
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

/// Map a tensor name from the HF Gemma 4 safetensors layout to the path the
/// vendored candle `gemma4::Model` expects.
///
/// Two transformations:
///
/// 1. QAT `.linear.weight` → `.weight`. HF QAT layout wraps each `nn.Linear`
///    so the underlying weight is stored at `.../linear.weight`. The
///    candle-fork uses plain `linear_no_bias`, which expects `.../weight`.
///
/// 2. Insert the missing inner `.model.` segment under `language_model`.
///    `Gemma4Model::new_partial` applies `vb.pp("model")`, then
///    `TextModel::new` applies `vb.pp("model")` *again* — so the candle
///    lookup path for the decoder is `model.language_model.model.<sub>`.
///    The HF safetensors file omits that inner segment (paths are
///    `model.language_model.embed_tokens.weight`, `…layers.X.…`,
///    `…norm.weight`). Insert it here for those three roots. `lm_head` is
///    loaded one level up (`vb.pp("lm_head")`), so it stays as-is.
fn gemma4_remap_key(name: &str) -> String {
    let s = if let Some(stripped) = name.strip_suffix(".linear.weight") {
        format!("{stripped}.weight")
    } else {
        name.to_string()
    };
    if let Some(rest) = s.strip_prefix("model.language_model.") {
        // Anything that lives directly on `Gemma3nTextModel` in HF needs
        // the inner `.model.` segment inserted to match the candle
        // double-`vb.pp("model")` nesting.
        let needs_remap = rest.starts_with("layers.")
            || rest.starts_with("embed_tokens")
            || rest.starts_with("norm.")
            // Gemma 3n top-level tensors (Phase 2 PLE + Phase 3 AltUp).
            || rest.starts_with("per_layer_model_projection")
            || rest.starts_with("per_layer_projection_norm")
            || rest.starts_with("altup_projections.")
            || rest.starts_with("altup_unembed_projections.");
        if needs_remap {
            return format!("model.language_model.model.{rest}");
        }
    }
    s
}

/// Tensors that belong to the Gemma4 vision tower (encoder + projector).
/// Used to either skip them at init (lazy) or load them on `attach_vision`.
fn is_vision_tensor(name: &str) -> bool {
    name.contains(".vision_tower.") || name.contains(".embed_vision.")
}

// ── OPFS-backed per-layer-embedding table ──────────────────────────────────
//
// The Gemma 3n `embed_tokens_per_layer.weight` tensor for the E2B
// checkpoint is `~262_144 × 30 × 256 × bf16 ≈ 4.7 GB` — too big for a
// single WebGPU buffer (1 GB cap) or a wasm32 `Vec` (`isize::MAX` ≈
// 2 GB cap). We can't load it into RAM at all on this target.
//
// Instead, we keep the safetensors `read_fn` callback alive past
// `init_local_multimodal_chunked` and read the matching row on every
// PLE lookup. Per-token cost is one OPFS sync read of
// `num_layers * hidden_per_layer * sizeof(dtype)` bytes (≈ 15 KB for E2B),
// which is well below the per-step budget.
//
// We do keep a bounded in-memory row cache (`row_cache`) so that
// repeated tokens don't re-hit OPFS. Typical chat sessions touch a few
// thousand unique tokens out of the 262k vocab — at ~15 KB per row,
// caching ~16k rows costs ~240 MB. JS↔WASM bridge calls have measurable
// per-call overhead, so even a cold prefill of N tokens benefits when
// the same row is reread by attention/diag passes downstream. The
// cache bound is enforced by capacity check; oldest insertion order
// is preserved via a FIFO companion deque.

/// Streaming `PerLayerEmbedTable` impl backed by OPFS sync-access reads
/// against the original safetensors blob.
///
/// `Send`/`Sync` are unsafely asserted because `js_sys::Function` is
/// `!Send`/`!Sync` by default. wasm32 is single-threaded — there is no
/// scenario where the table can be read from another thread, so the
/// asserts are sound for the only target this type compiles for.
/// Max number of rows cached in `OpfsPerLayerEmbedTable.row_cache`.
/// At ~15 KB/row for E2B this caps RAM at roughly 240 MB.
const PLE_ROW_CACHE_MAX: usize = 16_384;

struct OpfsPerLayerEmbedTable {
    read_fn: Function,
    /// Absolute byte offset of the PLE table inside the safetensors file
    /// (`data_start + entry.data_offsets.0`).
    table_offset: u64,
    vocab_size: usize,
    /// `num_hidden_layers * hidden_per_layer` — the row width in
    /// elements (not bytes).
    row_elements: usize,
    /// Bytes per row (`row_elements * sizeof(dtype)`).
    row_bytes: u64,
    /// The dtype the safetensors file stores the table as. Typically
    /// `BF16` for Gemma 3n.
    src_dtype: DType,
    /// The dtype the merge in `PerLayerEmbedding::forward` expects —
    /// matches the model's compute dtype (BF16). Cast on lookup if
    /// `src_dtype != target_dtype`.
    target_dtype: DType,
    /// Cached rows keyed by clamped token id, plus a FIFO of insertion
    /// order so we can evict the oldest row when we hit
    /// `PLE_ROW_CACHE_MAX`. Behind a `Mutex` because the trait-object
    /// `lookup` takes `&self`. Single-threaded wasm: lock is uncontested.
    row_cache: std::sync::Mutex<RowCache>,
}

struct RowCache {
    map: std::collections::HashMap<u64, std::sync::Arc<Vec<u8>>>,
    fifo: std::collections::VecDeque<u64>,
    hits: u64,
    misses: u64,
}

unsafe impl Send for OpfsPerLayerEmbedTable {}
unsafe impl Sync for OpfsPerLayerEmbedTable {}

impl std::fmt::Debug for OpfsPerLayerEmbedTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpfsPerLayerEmbedTable")
            .field("vocab_size", &self.vocab_size)
            .field("row_elements", &self.row_elements)
            .field("src_dtype", &self.src_dtype)
            .field("target_dtype", &self.target_dtype)
            .finish()
    }
}

impl candle_transformers::models::gemma4::PerLayerEmbedTable
    for OpfsPerLayerEmbedTable
{
    fn lookup(
        &self,
        input_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let ids_cpu = input_ids.to_device(&Device::Cpu)?;
        let (b, t) = ids_cpu.dims2()?;
        // Different chat paths build input_ids in different integer
        // dtypes (U32 from the chat-pwa, I64 from native). Read as
        // whichever dtype the caller supplied; everything else converts
        // to a common `u64` index space below.
        let flat = ids_cpu.flatten_all()?;
        let ids_u64: Vec<u64> = match flat.dtype() {
            DType::U32 => flat.to_vec1::<u32>()?.into_iter().map(|x| x as u64).collect(),
            DType::I64 => flat
                .to_vec1::<i64>()?
                .into_iter()
                .map(|x| x.max(0) as u64)
                .collect(),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "OPFS PLE lookup: unsupported input_ids dtype {other:?} \
                     (expected U32 or I64)"
                )));
            }
        };

        let total_bytes = (b * t) * (self.row_bytes as usize);
        let mut buf: Vec<u8> = Vec::with_capacity(total_bytes);
        let mut cache = self.row_cache.lock().unwrap();
        for id in &ids_u64 {
            // Clamp into vocab range — out-of-range ids in PLE tables get
            // the sentinel row (matches `candle_nn::Embedding::forward`'s
            // behavior; clamp at the lookup boundary).
            let id_clamped = (*id).min(self.vocab_size as u64 - 1);
            if let Some(row) = cache.map.get(&id_clamped).cloned() {
                cache.hits += 1;
                buf.extend_from_slice(&row);
                continue;
            }
            cache.misses += 1;
            let offset = self.table_offset + id_clamped * self.row_bytes;
            let row = call_read_fn(&self.read_fn, offset, self.row_bytes)
                .map_err(|e| candle_core::Error::Msg(format!(
                    "OPFS PLE row read failed at id={id_clamped}: {}",
                    js_err_to_string(&e),
                )))?;
            buf.extend_from_slice(&row);
            // Insert into cache, evicting oldest if at capacity.
            if cache.map.len() >= PLE_ROW_CACHE_MAX {
                if let Some(oldest) = cache.fifo.pop_front() {
                    cache.map.remove(&oldest);
                }
            }
            cache.map.insert(id_clamped, std::sync::Arc::new(row));
            cache.fifo.push_back(id_clamped);
        }
        drop(cache);

        let raw = Tensor::from_raw_buffer(
            &buf,
            self.src_dtype,
            &[b, t, self.row_elements],
            &Device::Cpu,
        )?;
        if raw.dtype() == self.target_dtype {
            Ok(raw)
        } else {
            raw.to_dtype(self.target_dtype)
        }
    }
}

/// Build an `OpfsPerLayerEmbedTable` from the parsed safetensors header,
/// or `None` when the checkpoint doesn't carry the PLE table at all.
///
/// The companion `gemma4_skip_reason("ple-table-oversize")` filter
/// already drops this tensor at load time so it never enters the
/// VarBuilder — this function reads the same `tensor_meta` entry that
/// the loader skipped to build a streaming-source replacement.
fn build_opfs_per_layer_table(
    tensor_meta: &[(String, StTensorInfo)],
    read_fn: Function,
    data_start: u64,
    target_dtype: DType,
) -> Result<Option<Arc<dyn candle_transformers::models::gemma4::PerLayerEmbedTable>>, String> {
    let entry = tensor_meta
        .iter()
        .find(|(n, _)| n.ends_with(".embed_tokens_per_layer.weight"));
    let (_, info) = match entry {
        Some(e) => e,
        None => return Ok(None),
    };
    if info.shape.len() != 2 {
        return Err(format!(
            "embed_tokens_per_layer.weight: expected rank-2 shape, got {:?}",
            info.shape,
        ));
    }
    let vocab_size = info.shape[0];
    let row_elements = info.shape[1];
    let src_dtype = st_dtype_to_candle(&info.dtype)
        .map_err(|e| format!("PLE table dtype: {e}"))?;
    let elem_bytes: usize = match src_dtype {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::F64 | DType::I64 => 8,
        DType::U8 => 1,
        other => {
            return Err(format!(
                "PLE table: unsupported dtype {other:?} for OPFS streaming"
            ));
        }
    };
    let row_bytes = (row_elements * elem_bytes) as u64;
    let table_offset = data_start + info.data_offsets.0;

    Ok(Some(Arc::new(OpfsPerLayerEmbedTable {
        read_fn,
        table_offset,
        vocab_size,
        row_elements,
        row_bytes,
        src_dtype,
        target_dtype,
        row_cache: std::sync::Mutex::new(RowCache {
            map: std::collections::HashMap::new(),
            fifo: std::collections::VecDeque::new(),
            hits: 0,
            misses: 0,
        }),
    })))
}

/// Tensors that belong to the Gemma4 audio tower (encoder + projector).
fn is_audio_tensor(name: &str) -> bool {
    name.contains(".audio_tower.") || name.contains(".embed_audio.")
}

/// Options accepted by `init_local_multimodal_chunked`. Defaults to eager
/// vision (current behavior) and lazy audio (audio is unsupported in the
/// chat UI today, and `build_gemma4_config` synthesizes a `None` audio
/// config regardless).
#[derive(Debug, Clone, Default, Deserialize)]
struct LoadOptions {
    #[serde(default)]
    lazy_vision: bool,
    #[serde(default = "default_true")]
    lazy_audio: bool,
    /// Bisection kill-switches for the Gemma 3n modules. JS callers can
    /// flip these (e.g. via a `?disable_altup=1` URL hack handed into
    /// the worker's load message) to bypass AltUp / LAuReL / per-layer-
    /// input-gate at construction time. Defaults all-off so existing
    /// callers see no change.
    #[serde(default)]
    disable_altup: bool,
    #[serde(default)]
    disable_laurel: bool,
    #[serde(default)]
    disable_per_layer_input_gate: bool,
    /// Skip the OPFS-streaming PLE table install entirely. With this
    /// set the candle-fork PLE module degrades to projection-only
    /// (`per_layer_input = proj * rsqrt(2)`), losing the trained
    /// per-layer-embedding signal but bypassing the streaming reads
    /// completely. Bisection: if forward logits go from NaN to
    /// numerically reasonable when this is on, the OPFS PLE row
    /// reader is producing corrupt bytes.
    #[serde(default)]
    disable_ple_streaming: bool,
    /// Enable the candle-side diagnostic scaffold (per-layer abs_max +
    /// head[..4] readbacks). Adds ~120 GPU→CPU readbacks per generation
    /// step → 1–2 s/token of overhead. Off by default; turn on for one
    /// debug run by passing `{ diag: true }` from the JS loader to
    /// capture the per-layer trace under `[gemma4/diag]` lines.
    #[serde(default)]
    diag: bool,
    /// When `diag` is on, target this decoder layer index for
    /// intra-layer captures (`post_input_layernorm`, `post_self_attn`,
    /// ple/post_*, etc). Defaults to layer 8. Set to `Some(15)` to
    /// inspect the first KV-shared / double-wide-MLP receiver layer.
    /// Negative values disable intra-capture entirely.
    #[serde(default)]
    diag_target_layer: Option<i32>,
}

fn default_true() -> bool {
    true
}

async fn try_webgpu_device() -> Result<Device, String> {
    let has_gpu = js_sys::Reflect::get(
        &js_sys::global(),
        &JsValue::from_str("navigator"),
    )
    .ok()
    .and_then(|nav| js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")).ok())
    .map_or(false, |gpu| !gpu.is_undefined() && !gpu.is_null());

    if !has_gpu {
        return Err("navigator.gpu not available".into());
    }

    Device::new_wgpu_async(0)
        .await
        .map_err(|e| format!("{e}"))
}

/// Build a [`LocalMultiModalHandle`] from JS-supplied byte buffers.
///
/// `weights` is the contents of a single safetensors file containing the
/// vision tower (`vision_tower.vision_model.*`), the projector
/// (`multi_modal_projector.*`), and the Gemma decoder (`model.*`,
/// `lm_head.*`). One [`CandleVarBuilder`] is built over the whole buffer
/// and sub-prefixed for each component.
///
/// Bulk-read only: the chunked-loader path used for the text-only handle
/// would need prefix-aware tensor routing for vision_tower vs decoder, which
/// is non-trivial enough to defer. With Gemma 4 E2B + SigLIP-So400m the
/// safetensors fits in a single allocation on the `Device::Cpu` path the
/// browser already uses for the text-only chunked fallback (~5 GB for the
/// vision-language weights vs ~10 GB for the standalone decoder).
#[wasm_bindgen]
pub async fn init_local_multimodal(
    weights: Vec<u8>,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalMultiModalHandle, JsValue> {
    // Try WebGPU first; fall back to CPU. Same policy as the text-only path.
    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"[wasm/mm] using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm/mm] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    let cfg = default_gemma_e2b_mm_config();
    let hidden_size = cfg.hidden_size;

    // One VarBuilder over the full safetensors. `from_buffered_safetensors`
    // takes the bytes by value, so we pass `weights` directly — no clone.
    let vb = CandleVarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
        .map_err(|e| JsValue::from_str(&format!("safetensors load: {e}")))?;

    // Build the three sub-models, each rooted at the appropriate prefix.
    let vision = SiglipVisionTower::load(vb.pp("vision_tower").pp("vision_model"), device.clone())
        .map_err(|e| JsValue::from_str(&format!("siglip load: {e}")))?;

    let projector = MultiModalProjector::load(
        vb.pp("multi_modal_projector"),
        SIGLIP_HIDDEN,
        hidden_size,
        PROJECTOR_DEFAULT_EPS,
    )
    .map_err(|e| JsValue::from_str(&format!("projector load: {e}")))?;

    let decoder = Gemma3MmModel::new(false, &cfg, vb)
        .map_err(|e| JsValue::from_str(&format!("decoder load: {e}")))?;

    let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e}")))?;

    let pipeline =
        Gemma3MultiModal::from_components(vision, projector, decoder, tokenizer, device, cfg);

    Ok(LocalMultiModalHandle {
        inner: MultimodalInner::Gemma3(Arc::new(pipeline)),
        model_id,
        lazy: None,
        source: "hf",
    })
}

/// Read one tensor from the JS-backed safetensors file and materialize it
/// onto `device` (or stream it directly to GPU for tensors larger than
/// wasm32's `isize::MAX`). Shared by the bulk init loop and the on-demand
/// `attach_*` methods so the routing logic stays in one place.
fn load_one_tensor(
    name: &str,
    info: &StTensorInfo,
    data_start: u64,
    read_fn: &Function,
    device: &Device,
    wgpu_dev: Option<&brainwires_provider::WgpuDevice>,
    force_cpu: bool,
) -> Result<Tensor, JsValue> {
    let offset = data_start + info.data_offsets.0;
    let length = info.data_offsets.1 - info.data_offsets.0;
    let src_dtype = st_dtype_to_candle(&info.dtype)
        .map_err(|e| JsValue::from_str(&format!("tensor {name}: {e}")))?;

    let needs_gpu_stream = length > (isize::MAX as u64);

    // candle's WGPU backend doesn't support BF16 storage (only F16 /
    // F32 / U32 / U8 / I64 / F64 — see `candle-core/src/wgpu_backend/
    // mod.rs:34` which panics on the unsupported dtype). HF Gemma 4
    // safetensors are BF16. Detect the BF16-on-WGPU combo and cast
    // through CPU F16 before transfer. F16 has slightly less exponent
    // range than BF16, but for inference weights (typically clamped
    // to [-3, 3] for trained models) the loss is below numerical
    // noise — Apple silicon, llama.cpp, and most quantization tools
    // do the same conversion.
    //
    // The `needs_gpu_stream` path (tensors > isize::MAX bytes, ~2 GB
    // on wasm32) is unaffected: HF Gemma 4's largest BF16 tensor is
    // the embedding table at ~800 MB, well below the threshold. If a
    // future model needs BF16-on-WGPU streaming we'll need to also
    // teach `load_tensor_to_gpu` to convert chunkwise.
    let bf16_on_wgpu = !device.is_cpu() && matches!(src_dtype, DType::BF16);

    if force_cpu && needs_gpu_stream {
        let w = wgpu_dev.ok_or_else(|| {
            JsValue::from_str(&format!(
                "tensor {name} is {length} bytes — too large for CPU and no \
                 WebGPU device available"
            ))
        })?;
        load_tensor_to_gpu(read_fn, offset, length, src_dtype, &info.shape, w)
    } else if force_cpu {
        let bytes = call_read_fn(read_fn, offset, length)?;
        Tensor::from_raw_buffer(&bytes, src_dtype, &info.shape, &Device::Cpu)
            .map_err(|e| JsValue::from_str(&format!("tensor {name}: {e}")))
    } else if needs_gpu_stream {
        let w = wgpu_dev.ok_or_else(|| {
            JsValue::from_str(&format!(
                "tensor {name} is {length} bytes — too large for wasm32 and no \
                 WebGPU device available for direct upload"
            ))
        })?;
        load_tensor_to_gpu(read_fn, offset, length, src_dtype, &info.shape, w)
    } else if bf16_on_wgpu {
        // BF16 → CPU → cast to F16 → transfer to WGPU.
        let bytes = call_read_fn(read_fn, offset, length)?;
        let cpu_t = Tensor::from_raw_buffer(&bytes, src_dtype, &info.shape, &Device::Cpu)
            .map_err(|e| JsValue::from_str(&format!("tensor {name} (cpu BF16): {e}")))?;
        let f16_t = cpu_t
            .to_dtype(DType::F16)
            .map_err(|e| JsValue::from_str(&format!("tensor {name} (BF16→F16): {e}")))?;
        f16_t
            .to_device(device)
            .map_err(|e| JsValue::from_str(&format!("tensor {name} (CPU→WGPU): {e}")))
    } else {
        let bytes = call_read_fn(read_fn, offset, length)?;
        Tensor::from_raw_buffer(&bytes, src_dtype, &info.shape, device)
            .map_err(|e| JsValue::from_str(&format!("tensor {name}: {e}")))
    }
}

/// Chunked variant of [`init_local_multimodal`]. Reads tensors one at a time
/// via a JS callback, avoiding a single multi-GB allocation.
///
/// `options_js` is an optional JS object: `{lazy_vision?: bool, lazy_audio?: bool}`.
/// When `lazy_vision` is `true`, the vision tower is not loaded at init time;
/// call [`LocalMultiModalHandle::attach_vision`] before sending an image.
#[wasm_bindgen]
pub async fn init_local_multimodal_chunked(
    read_fn: Function,
    file_size: f64,
    tokenizer_json: Vec<u8>,
    model_id: String,
    options_js: JsValue,
) -> Result<LocalMultiModalHandle, JsValue> {
    let options: LoadOptions = if options_js.is_null() || options_js.is_undefined() {
        LoadOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options_js)
            .map_err(|e| JsValue::from_str(&format!("invalid options: {e}")))?
    };

    if options.diag {
        brainwires_provider::local_llm::vision::gemma4_mm::set_diag_enabled(true);
        let target_layer: Option<usize> = match options.diag_target_layer {
            None => None, // fall back to default (layer 8)
            Some(v) if v < 0 => {
                brainwires_provider::local_llm::vision::gemma4_mm::set_diag_target_layer(None);
                None
            }
            Some(v) => {
                let l = v as usize;
                brainwires_provider::local_llm::vision::gemma4_mm::set_diag_target_layer(Some(l));
                Some(l)
            }
        };
        web_sys::console::log_1(
            &format!(
                "[wasm/mm] diag scaffold enabled — per-layer abs_max readback active (intra-layer target: {})",
                match target_layer {
                    Some(l) => l.to_string(),
                    None => "default(8)".to_string(),
                },
            )
            .into(),
        );
    }

    let file_size = file_size as u64;
    web_sys::console::log_1(
        &format!(
            "[wasm/mm] chunked load: file_size={file_size}, model={model_id}, \
             lazy_vision={}, lazy_audio={}",
            options.lazy_vision, options.lazy_audio,
        )
        .into(),
    );

    let header_size_bytes = call_read_fn(&read_fn, 0, 8)?;
    if header_size_bytes.len() < 8 {
        return Err(JsValue::from_str("failed to read safetensors header size"));
    }
    let header_size =
        u64::from_le_bytes(header_size_bytes[..8].try_into().unwrap());

    let header_bytes = call_read_fn(&read_fn, 8, header_size)?;
    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| JsValue::from_str(&format!("invalid header UTF-8: {e}")))?;

    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| JsValue::from_str(&format!("invalid safetensors header: {e}")))?;

    let mut tensor_meta: Vec<(String, StTensorInfo)> = Vec::new();
    let mut metadata: HashMap<String, String> = HashMap::new();
    for (name, value) in &raw {
        if name == "__metadata__" {
            // safetensors carries optional `__metadata__` as an object of
            // string→string pairs. Extract for downstream consumers
            // (e.g. `activation_sparsity_pattern` for Gemma 3n).
            if let Some(obj) = value.as_object() {
                for (k, v) in obj {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let info: StTensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
            JsValue::from_str(&format!("bad tensor info for {name}: {e}"))
        })?;
        tensor_meta.push((name.clone(), info));
    }
    tensor_meta.sort_by_key(|(_, info)| info.data_offsets.0);

    let total = tensor_meta.len();
    web_sys::console::log_1(
        &format!("[wasm/mm] parsed {total} tensor entries").into(),
    );

    let data_start: u64 = 8 + header_size;

    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"[wasm/mm] chunked load: using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm/mm] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    let wgpu_dev = match &device {
        Device::Wgpu(w) => Some(w.clone()),
        _ => None,
    };

    let model_type = detect_model_type(&tensor_meta);
    web_sys::console::log_1(
        &format!("[wasm/mm] detected model type: {model_type:?}").into(),
    );

    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(total);
    // (audio_unused, qat-stat, ple-table-oversize, lazy_vision, lazy_audio, total bytes deferred)
    let mut skipped = (0usize, 0usize, 0usize, 0usize, 0usize, 0u64);
    for (idx, (name, info)) in tensor_meta.iter().enumerate() {
        let length = info.data_offsets.1 - info.data_offsets.0;

        if model_type == ModelType::Gemma4 {
            if let Some(reason) = gemma4_skip_reason(name) {
                match reason {
                    "audio" => skipped.0 += 1,
                    "qat-stat" => skipped.1 += 1,
                    "ple-table-oversize" => skipped.2 += 1,
                    _ => {}
                }
                skipped.5 += length;
                continue;
            }
            if options.lazy_vision && is_vision_tensor(name) {
                skipped.3 += 1;
                skipped.5 += length;
                continue;
            }
            if options.lazy_audio && is_audio_tensor(name) {
                // Audio is also caught by gemma4_skip_reason("audio") above,
                // but keep this guarded in case that filter is relaxed.
                skipped.4 += 1;
                skipped.5 += length;
                continue;
            }
        }

        // Force-CPU keeps these tensors in host memory so they don't
        // try to occupy WebGPU buffers. embed_tokens.weight (~800 MB)
        // exceeds the 1 GB WebGPU max-buffer-size limit; lm_head shares
        // it via tied weights so it tags along. PLE companion tensors
        // (per_layer_model_projection + per_layer_projection_norm) stay
        // on CPU as well so the PerLayerEmbedding.forward merge happens
        // entirely on CPU and Gemma4MultiModal::generate_greedy moves
        // the resulting table to GPU in one shot.
        // `embed_tokens_per_layer.weight` itself is *skipped* by
        // gemma4_skip_reason — it's too big for both WebGPU buffers
        // (1 GB) and wasm32 Vec (2 GB). The PLE signal is restored via
        // `OpfsPerLayerEmbedTable` (see `build_opfs_per_layer_table` and
        // the `set_per_layer_embed_table` injection in the loader tail);
        // rows are streamed on every forward pass.
        let force_cpu = model_type == ModelType::Gemma4
            && (name.ends_with("embed_tokens.weight")
                || name.ends_with("lm_head.weight")
                || name.ends_with("per_layer_model_projection.weight")
                || name.ends_with("per_layer_projection_norm.weight"));

        let tensor = load_one_tensor(
            name,
            info,
            data_start,
            &read_fn,
            &device,
            wgpu_dev.as_ref(),
            force_cpu,
        )?;

        let key = if model_type == ModelType::Gemma4 {
            gemma4_remap_key(name)
        } else {
            name.strip_prefix("model.").unwrap_or(name).to_string()
        };
        tensors.insert(key, tensor);

        let needs_gpu_stream = length > (isize::MAX as u64);
        if idx % 20 == 0 || idx == total - 1 || needs_gpu_stream || force_cpu || length > 100_000_000 {
            let tag = if needs_gpu_stream {
                " [gpu-direct]"
            } else if force_cpu {
                " [cpu]"
            } else {
                ""
            };
            web_sys::console::log_1(
                &format!(
                    "[wasm/mm] loaded tensor {}/{total}: {name} {:?} [{}] ({length} bytes){tag}",
                    idx + 1,
                    info.shape,
                    info.dtype,
                )
                .into(),
            );
        }
    }

    if model_type == ModelType::Gemma4 {
        let (audio, qat, ple_oversize, lazy_v, lazy_a, bytes) = skipped;
        web_sys::console::log_1(
            &format!(
                "[wasm/mm] skipped {} tensors ({} audio-unused, {} QAT-stat, \
                 {} PLE-table-oversize, {} deferred-vision, {} deferred-audio), \
                 saved {:.2} GB",
                audio + qat + ple_oversize + lazy_v + lazy_a,
                audio, qat, ple_oversize, lazy_v, lazy_a,
                bytes as f64 / 1_073_741_824.0,
            )
            .into(),
        );
    }
    web_sys::console::log_1(
        &format!("[wasm/mm] all {total} tensors loaded, building {model_type:?} model...").into(),
    );

    let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e}")))?;

    let (inner, lazy) = match model_type {
        ModelType::Gemma3 => {
            let cfg = default_gemma_e2b_mm_config();
            let hidden_size = cfg.hidden_size;
            let vb = CandleVarBuilder::from_tensors(tensors, DType::F32, &device);

            let vision = SiglipVisionTower::load(
                vb.pp("vision_tower").pp("vision_model"),
                device.clone(),
            )
            .map_err(|e| JsValue::from_str(&format!("siglip load: {e}")))?;

            let projector = MultiModalProjector::load(
                vb.pp("multi_modal_projector"),
                SIGLIP_HIDDEN,
                hidden_size,
                PROJECTOR_DEFAULT_EPS,
            )
            .map_err(|e| JsValue::from_str(&format!("projector load: {e}")))?;

            let decoder = Gemma3MmModel::new(false, &cfg, vb)
                .map_err(|e| JsValue::from_str(&format!("decoder load: {e}")))?;

            let pipeline = Gemma3MultiModal::from_components(
                vision, projector, decoder, tokenizer, device, cfg,
            );
            (MultimodalInner::Gemma3(Arc::new(pipeline)), None)
        }
        ModelType::Gemma4 => {
            let cfg = build_gemma4_config(&tensor_meta, &metadata, &options)
                .map_err(|e| JsValue::from_str(&format!("gemma4 config: {e}")))?;

            web_sys::console::log_1(
                &format!(
                    "[wasm/mm] Gemma4 config: hidden={}, layers={}, heads={}, head_dim={}, global_head_dim={}",
                    cfg.text_config.hidden_size,
                    cfg.text_config.num_hidden_layers,
                    cfg.text_config.num_attention_heads,
                    cfg.text_config.head_dim,
                    cfg.text_config.global_head_dim,
                )
                .into(),
            );

            // Gemma4-E2B's embed_tokens / lm_head weights weigh ~800 MB
            // each in bf16. They were loaded onto CPU above (force_cpu in
            // load_one_tensor) so they don't eat WebGPU memory, but the
            // default `HashMap`-backed VarBuilder always calls
            // `tensor.to_device(dev)` on every fetch — silently undoing the
            // CPU placement and causing the runtime device-mismatch we hit
            // in `index_select` (`embed_tokens` weight on Wgpu, the input
            // ids constructed on Cpu in `Gemma4MultiModal::generate_greedy`).
            //
            // `Gemma4MultiModal::generate_greedy` is built around mixed-
            // device execution: embed_tokens / lm_head on CPU, decoder on
            // GPU, with explicit `to_device` shuffles around `forward_embeds_hidden`.
            // To make that design actually take effect we route the
            // VarBuilder through a small backend that honors the loaded
            // device for a pinned set of names and falls back to the
            // default to-device behavior for everything else.
            // The HashMap key is the post-`gemma4_remap_key` name; the
            // VarBuilder path (`vb.pp("model").pp("language_model")
            // .pp("model").pp("embed_tokens")`) lands at exactly the same
            // string. With `tie_word_embeddings: true` (the Gemma4-E2B
            // default) `lm_head` shares this same tensor — pinning it once
            // covers both. PLE projection / norm tensors are also pinned
            // so the merge in `PerLayerEmbedding.forward` happens
            // entirely host-side. (`embed_tokens_per_layer.weight` itself
            // is dropped by `gemma4_skip_reason` — too big for any
            // single buffer — and the candle-fork PLE module degrades
            // gracefully when the table is absent.)
            let cpu_pinned: HashSet<String> = [
                "model.language_model.model.embed_tokens.weight",
                "model.language_model.model.per_layer_model_projection.weight",
                "model.language_model.model.per_layer_projection_norm.weight",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let vb = CandleVarBuilder::from_backend(
                Box::new(CpuPinnedBackend {
                    inner: tensors,
                    cpu_pinned,
                }),
                DType::BF16,
                device.clone(),
            );

            // `lazy_audio` defaults to true and `cfg.audio_config` is currently
            // synthesized as `None`, so audio is never built at init regardless
            // of the flag. The flag exists for symmetry with vision.
            let with_vision = !options.lazy_vision;
            let with_audio = !options.lazy_audio && cfg.audio_config.is_some();
            let mut model = Gemma4Model::new_partial(&cfg, vb, with_vision, with_audio)
                .map_err(|e| JsValue::from_str(&format!("gemma4 model load: {e}")))?;

            // Restore the per-layer-embedding signal via OPFS-backed
            // streaming. The 4.7 GB PLE table is too big to load (filtered
            // out by `gemma4_skip_reason("ple-table-oversize")`); we read
            // matching rows on every forward pass through `read_fn`,
            // which the lazy state keeps alive past load.
            //
            // `disable_ple_streaming` short-circuits this block — the
            // candle-fork PLE module then runs the projection-only merge,
            // which lets us prove whether step-0 NaN logits originate in
            // the OPFS row reader or further downstream.
            if options.disable_ple_streaming {
                web_sys::console::warn_1(
                    &"[wasm/mm] PLE OPFS streaming disabled by kill-switch \
                      — per-layer merge degrades to projection-only".into(),
                );
            } else {
            match build_opfs_per_layer_table(
                &tensor_meta,
                read_fn.clone(),
                data_start,
                DType::BF16,
            ) {
                Ok(Some(table)) => {
                    if let Err(e) =
                        model.language_model.set_per_layer_embed_table(table)
                    {
                        web_sys::console::warn_1(&format!(
                            "[wasm/mm] PLE OPFS table not attached: {e}"
                        ).into());
                    } else {
                        web_sys::console::log_1(
                            &"[wasm/mm] PLE table backed by OPFS streaming".into(),
                        );
                    }
                }
                Ok(None) => {
                    web_sys::console::log_1(
                        &"[wasm/mm] no embed_tokens_per_layer.weight in checkpoint; \
                          per-layer merge degrades to projection-only"
                            .into(),
                    );
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!(
                        "[wasm/mm] PLE OPFS table build failed: {e}"
                    ).into());
                }
            }
            }

            let pipeline = Gemma4MultiModal::from_components(
                model,
                tokenizer,
                device.clone(),
                cfg.clone(),
            );
            // Retain enough state to honor a later attach_vision/attach_audio call.
            let lazy_state = Gemma4LazyState {
                read_fn: read_fn.clone(),
                tensor_meta: tensor_meta.clone(),
                data_start,
                cfg,
                device: device.clone(),
                wgpu_dev: wgpu_dev.clone(),
            };
            (
                MultimodalInner::Gemma4 {
                    pipeline: Arc::new(pipeline),
                    gpu_device: device,
                },
                Some(lazy_state),
            )
        }
    };

    Ok(LocalMultiModalHandle {
        inner,
        model_id,
        lazy,
        source: "hf",
    })
}

// ---------------------------------------------------------------------------
// Streaming chat
// ---------------------------------------------------------------------------

/// Subset of [`brainwires_core::provider::ChatOptions`] the JS side passes
/// through. Cloud-only fields (cache strategy, etc.) do not apply here.
#[derive(Debug, Clone, Default, Deserialize)]
struct VisionStreamParams {
    #[serde(default)]
    max_tokens: Option<u32>,
}

/// JS-side message shape. `content` is either a plain string OR an array of
/// `{type: 'text'|'image', ...}` parts. We accept both via `untagged` and the
/// `JsContent` enum below.
#[derive(Debug, Clone, Deserialize)]
struct JsMessage {
    role: String,
    content: JsContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum JsContent {
    Text(String),
    Parts(Vec<JsPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum JsPart {
    Text {
        text: String,
    },
    Image {
        #[allow(dead_code)]
        #[serde(default)]
        media_type: Option<String>,
        #[allow(dead_code)]
        #[serde(default, rename = "mediaType")]
        media_type_camel: Option<String>,
        data: String,
    },
}

/// Wire-format chunk emitted into the [`ReadableStream`]. Mirrors the
/// text-only path's `WireChunk` — same fields, same NDJSON contract.
#[derive(Debug, Clone, Default, Serialize)]
struct VisionWireChunk<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    finished: bool,
}

/// Drive a multimodal chat against a loaded [`LocalMultiModalHandle`].
///
/// `messages_json` is a JSON array of `{role, content}` where `content` is
/// either a string OR `[{type:'text'|'image', ...}]`.
/// `params_json` is `{ max_tokens? }`.
///
/// Returns a `ReadableStream<Uint8Array>` of NDJSON-encoded
/// [`VisionWireChunk`]s — the same shape `local_chat_stream` produces, so
/// the JS-side reader in `local-worker.js#runChatStream` is reused
/// verbatim.
#[wasm_bindgen]
pub fn local_chat_stream_with_image(
    handle: &LocalMultiModalHandle,
    messages_json: String,
    params_json: String,
) -> Result<ReadableStream, JsValue> {
    let messages: Vec<JsMessage> = serde_json::from_str(&messages_json)
        .map_err(|e| JsValue::from_str(&format!("messages_json parse: {e}")))?;
    let params: VisionStreamParams = if params_json.trim().is_empty() {
        VisionStreamParams::default()
    } else {
        serde_json::from_str(&params_json)
            .map_err(|e| JsValue::from_str(&format!("params_json parse: {e}")))?
    };

    let inner = match &handle.inner {
        MultimodalInner::Gemma3(p) => StreamInner::Gemma3(p.clone()),
        MultimodalInner::Gemma4 { pipeline, .. } => StreamInner::Gemma4(pipeline.clone()),
    };

    let underlying = Object::new();
    let start_cb = Closure::once_into_js(move |controller: JsValue| {
        let controller: ReadableStreamDefaultController = match controller.dyn_into() {
            Ok(c) => c,
            Err(_) => return,
        };
        spawn_local(run_vision_stream(inner, messages, params, controller));
    });
    Reflect::set(&underlying, &JsValue::from_str("start"), &start_cb)
        .map_err(|_| JsValue::from_str("failed to set ReadableStream start callback"))?;

    ReadableStream::new_with_underlying_source(&underlying)
}

/// Cloneable inner for streaming — avoids moving the full handle into spawn_local.
enum StreamInner {
    Gemma3(Arc<Gemma3MultiModal>),
    Gemma4(Arc<Gemma4MultiModal>),
}

/// Runs greedy, one-shot generation and pushes a `delta` + `finished`
/// chunk into the controller. Errors surface as a `{error: "..."}` chunk
/// followed by `controller.error_with_e`, matching the text-only path.
async fn run_vision_stream(
    inner: StreamInner,
    messages: Vec<JsMessage>,
    params: VisionStreamParams,
    controller: ReadableStreamDefaultController,
) {
    let result = match &inner {
        StreamInner::Gemma3(pipeline) => {
            // Gemma3 path is non-streaming for now — emits the full
            // result as a single chunk at completion.
            build_and_generate_gemma3(pipeline, &messages, &params).map_err(|e| format!("{e}"))
        }
        StreamInner::Gemma4(pipeline) => {
            // Gemma 4 streams per-token deltas via the controller. The
            // returned `String` here is just for completeness; the UI
            // already received the deltas during generation.
            build_and_stream_gemma4(pipeline, &messages, &params, &controller)
                .await
                .map_err(|e| format!("{e}"))
        }
    };
    match result {
        Ok(text) => {
            // For Gemma 3 (non-streaming), emit the full text now. For
            // Gemma 4, deltas were already emitted during generation —
            // sending the full text again would duplicate everything,
            // so only the `finished` chunk is emitted here.
            if matches!(inner, StreamInner::Gemma3(_)) {
                enqueue_vision_chunk(
                    &controller,
                    &VisionWireChunk {
                        delta: Some(&text),
                        ..Default::default()
                    },
                );
            }
            enqueue_vision_chunk(
                &controller,
                &VisionWireChunk {
                    finished: true,
                    ..Default::default()
                },
            );
            let _ = controller.close();
        }
        Err(e) => {
            let msg = format!("local_chat_stream_with_image: {e}");
            enqueue_vision_chunk(
                &controller,
                &VisionWireChunk {
                    error: Some(msg.clone()),
                    finished: true,
                    ..Default::default()
                },
            );
            controller.error_with_e(&JsValue::from_str(&msg));
        }
    }
}

/// Gemma-3 generation: extract text/images, run SigLIP + projector + decoder.
fn build_and_generate_gemma3(
    pipeline: &Gemma3MultiModal,
    messages: &[JsMessage],
    params: &VisionStreamParams,
) -> Result<String, MmPipelineError> {
    if messages.is_empty() {
        return Err(MmPipelineError::InvalidInput("empty messages".into()));
    }

    pipeline.clear_kv_cache();

    let last = &messages[messages.len() - 1];
    let mut text_segments: Vec<String> = Vec::new();
    let mut image_bytes: Vec<Vec<u8>> = Vec::new();

    let prefix = build_history_prefix(&messages[..messages.len() - 1], &last.role);

    match &last.content {
        JsContent::Text(t) => {
            text_segments.push(format!("{prefix}{t}"));
        }
        JsContent::Parts(parts) => {
            let mut current = String::new();
            current.push_str(&prefix);
            for p in parts {
                match p {
                    JsPart::Text { text } => current.push_str(text),
                    JsPart::Image { data, .. } => {
                        text_segments.push(std::mem::take(&mut current));
                        let bytes = BASE64
                            .decode(data.as_bytes())
                            .map_err(|e| MmPipelineError::InvalidInput(format!("base64: {e}")))?;
                        image_bytes.push(bytes);
                    }
                }
            }
            text_segments.push(current);
        }
    }

    let pixel_tensors: Vec<Tensor> = image_bytes
        .iter()
        .map(|b| {
            preprocess_image_bytes(b, pipeline.device())
                .map_err(|e| MmPipelineError::InvalidInput(format!("preprocess: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let images: Vec<ImageInput> = pixel_tensors
        .iter()
        .map(|t| ImageInput { pixel_values: t })
        .collect();

    let segs_ref: Vec<&str> = text_segments.iter().map(|s| s.as_str()).collect();
    let max_new = params.max_tokens.unwrap_or(256) as usize;
    let eos: Option<u32> = None;

    pipeline.generate_greedy(&segs_ref, &images, max_new, eos)
}

/// Gemma-4 generation: extract text/images, run native vision tower + embedder + decoder.
/// Streaming Gemma 4 generation. Per-token deltas are emitted into
/// `controller` via `enqueue_vision_chunk` as they're produced. Returns
/// the full text on completion (which `run_vision_stream` discards
/// because the deltas have already gone over the wire).
async fn build_and_stream_gemma4(
    pipeline: &Gemma4MultiModal,
    messages: &[JsMessage],
    params: &VisionStreamParams,
    controller: &ReadableStreamDefaultController,
) -> Result<String, Gemma4PipelineError> {
    if messages.is_empty() {
        return Err(Gemma4PipelineError::InvalidInput("empty messages".into()));
    }

    pipeline.clear_kv_cache();

    let mut image_bytes: Vec<Vec<u8>> = Vec::new();
    let prompt_text = build_gemma_chat_prompt(messages, &mut image_bytes)
        .map_err(Gemma4PipelineError::InvalidInput)?;

    let target_size = 768u32;
    let pixel_tensors: Vec<Tensor> = image_bytes
        .iter()
        .map(|b| {
            preprocess_image_for_gemma4(b, &Device::Cpu, target_size)
                .map_err(|e| Gemma4PipelineError::InvalidInput(format!("preprocess: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let max_new = params.max_tokens.unwrap_or(256) as usize;
    let eos: Option<u32> = Some(1);

    pipeline
        .generate_greedy_streaming(
            &prompt_text,
            &pixel_tensors,
            max_new,
            eos,
            |_token_id, delta| {
                enqueue_vision_chunk(
                    controller,
                    &VisionWireChunk {
                        delta: Some(delta),
                        ..Default::default()
                    },
                );
            },
        )
        .await
}

async fn build_and_generate_gemma4(
    pipeline: &Gemma4MultiModal,
    messages: &[JsMessage],
    params: &VisionStreamParams,
) -> Result<String, Gemma4PipelineError> {
    if messages.is_empty() {
        return Err(Gemma4PipelineError::InvalidInput("empty messages".into()));
    }

    pipeline.clear_kv_cache();

    let mut image_bytes: Vec<Vec<u8>> = Vec::new();
    let prompt_text = build_gemma_chat_prompt(messages, &mut image_bytes)
        .map_err(Gemma4PipelineError::InvalidInput)?;

    // Preprocess images to [1, 3, target, target] f32 in [0,1].
    // Gemma4 default vision input is 768px (48 patches of 16).
    let target_size = 768u32;
    let pixel_tensors: Vec<Tensor> = image_bytes
        .iter()
        .map(|b| {
            preprocess_image_for_gemma4(b, &Device::Cpu, target_size)
                .map_err(|e| Gemma4PipelineError::InvalidInput(format!("preprocess: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let max_new = params.max_tokens.unwrap_or(256) as usize;
    let eos: Option<u32> = Some(1); // Gemma EOS token

    pipeline
        .generate_greedy(&prompt_text, &pixel_tensors, max_new, eos)
        .await
}

/// Build the Gemma 4 chat-template prompt for a full message list.
///
/// Gemma 4 uses **different chat tokens than Gemma 3**:
///   - `<|turn>` (id 105) — begin-of-turn (previously `<start_of_turn>`)
///   - `<turn|>` (id 106) — end-of-turn   (previously `<end_of_turn>`)
///
/// Note pipe placement: `<|turn>` and `<turn|>`, not `<|turn|>` or
/// `<start_of_turn>`. These strings are in the tokenizer's
/// `added_tokens` array with `special: true`, so `encode()` matches
/// them as single units verbatim (no runtime registration needed).
///
/// Role string for the assistant turn is **`"model"`** (not `"assistant"`).
/// Per the official `chat_template.jinja`:
/// `{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}`
/// and the generation prompt is `<|turn>model\n`.
///
/// Format:
///
/// ```text
/// <bos><|turn>user
/// hello<turn|>
/// <|turn>model
/// hi<turn|>
/// <|turn>user
/// how are you<turn|>
/// <|turn>model
/// ```
///
/// The trailing `<|turn>model\n` is the generation prompt that
/// cues the model to begin its response.
///
/// `Image` parts are emitted as the `<|image|>` literal marker
/// (id 258880). The downstream `Gemma4MultiModal::generate_greedy`
/// finds those positions in the encoded prompt by `cfg.image_token_id`
/// and splices in the vision-embedder output.
fn build_gemma_chat_prompt(
    messages: &[JsMessage],
    image_bytes: &mut Vec<Vec<u8>>,
) -> Result<String, String> {
    let mut buf = String::from("<bos>");
    for m in messages {
        // Gemma 4's chat_template.jinja maps assistant → model.
        let role: &str = if m.role == "assistant" { "model" } else { m.role.as_str() };
        let text = match &m.content {
            JsContent::Text(t) => t.clone(),
            JsContent::Parts(parts) => {
                let mut out = String::new();
                for p in parts {
                    match p {
                        JsPart::Text { text } => out.push_str(text),
                        JsPart::Image { data, .. } => {
                            let bytes = BASE64
                                .decode(data.as_bytes())
                                .map_err(|e| format!("base64: {e}"))?;
                            image_bytes.push(bytes);
                            // `<|image|>` (id 258880) is the canonical
                            // image marker per the tokenizer config.
                            out.push_str("<|image|>");
                        }
                    }
                }
                out
            }
        };
        buf.push_str("<|turn>");
        buf.push_str(role);
        buf.push('\n');
        buf.push_str(text.trim());
        buf.push_str("<turn|>\n");
    }
    // Generation cue. Per chat_template.jinja: `<|turn>model\n`.
    buf.push_str("<|turn>model\n");
    Ok(buf)
}

/// Build a `<role>: <text>\n…` prefix from earlier turns. Plain join — same
/// formatter the text-only path uses for `format_prompt`.
fn build_history_prefix(history: &[JsMessage], current_role: &str) -> String {
    let mut buf = String::new();
    for m in history {
        let text = match &m.content {
            JsContent::Text(t) => t.clone(),
            JsContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    JsPart::Text { text } => Some(text.as_str()),
                    JsPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };
        buf.push_str(&m.role);
        buf.push_str(": ");
        buf.push_str(&text);
        buf.push('\n');
    }
    if !buf.is_empty() {
        buf.push_str(current_role);
        buf.push_str(": ");
    }
    buf
}

fn enqueue_vision_chunk(
    controller: &ReadableStreamDefaultController,
    chunk: &VisionWireChunk<'_>,
) {
    let mut bytes = match serde_json::to_vec(chunk) {
        Ok(b) => b,
        Err(_) => return,
    };
    bytes.push(b'\n');
    let view = Uint8Array::from(bytes.as_slice());
    let _ = controller.enqueue_with_chunk(&view);
}

// ── Per-tensor device-pinning VarBuilder backend ─────────────────────
//
// `candle_nn`'s default `SimpleBackend for HashMap<String, Tensor>` calls
// `tensor.to_device(dev)` on every fetch, which silently moves CPU-loaded
// weights onto the VarBuilder's GPU device — defeating the chat-pwa's
// `force_cpu` placement for Gemma4-E2B's 800 MB embed_tokens / lm_head
// table. This backend honors the loaded device for a small allow-list of
// pinned names so the table stays where it was loaded; everything else
// keeps the default to-device behavior.
struct CpuPinnedBackend {
    inner: HashMap<String, Tensor>,
    cpu_pinned: HashSet<String>,
}

impl candle_nn::var_builder::SimpleBackend for CpuPinnedBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self
            .inner
            .get(name)
            .ok_or_else(|| {
                candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .clone();
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        if self.cpu_pinned.contains(name) {
            // Honor the loaded device; don't move to `dev`.
            tensor.to_dtype(dtype)
        } else {
            tensor.to_device(dev)?.to_dtype(dtype)
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let tensor = self
            .inner
            .get(name)
            .ok_or_else(|| {
                candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .clone();
        if self.cpu_pinned.contains(name) {
            tensor.to_dtype(dtype)
        } else {
            tensor.to_device(dev)?.to_dtype(dtype)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// GGUF (Ollama) loading — Phase 4 part 3
// ---------------------------------------------------------------------------

/// Build a Gemma 4 [`LocalMultiModalHandle`] from a Q4_K_M GGUF blob.
///
/// Reads via [`brainwires_provider::local_llm::gguf_loader`], dequantizes
/// every quantized tensor to BF16 in memory, builds a
/// [`CandleVarBuilder`] over the resulting tensor map, and constructs a
/// text-only [`Gemma4Model`] feeding into a
/// [`Gemma4MultiModal`] pipeline. The vision/audio towers are disabled
/// — Ollama's `gemma4:e2b` is text-only and the GGUF doesn't carry the
/// SigLIP / audio weights.
///
/// **No perf win on its own** — this is the dequant-at-load path. A
/// `quantized_gemma4::ModelWeights` model that consumes QTensors via
/// QMatMul end-to-end now exists in candle (and is reachable via
/// `gguf_loader::load_quantized_gemma4_from_reader`), but the wasm
/// pipeline here doesn't wire it up yet — the `Gemma4MultiModal`
/// generate_greedy machinery currently only wraps the BF16 path.
/// Adding a parallel `Gemma4QuantizedMultiModal` wrapper is the
/// follow-up needed to make the chat-pwa actually run on the
/// `q4_k.pwgsl` quantized matmul kernels. Until then, GGUF saves
/// download bytes (~6× smaller than HF safetensors) but inference
/// runs at the same BF16 tok/s as the safetensors path.
#[wasm_bindgen]
pub async fn init_local_multimodal_gguf(
    weights: Vec<u8>,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalMultiModalHandle, JsValue> {
    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"[wasm/gguf] using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm/gguf] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    web_sys::console::log_1(
        &format!(
            "[wasm/gguf] loading GGUF blob ({} bytes), model_id={model_id}",
            weights.len()
        )
        .into(),
    );

    let mut cursor = std::io::Cursor::new(weights);
    let (tensors, cfg) =
        brainwires_provider::local_llm::gguf_loader::load_gemma4_gguf_from_reader(
            &mut cursor,
            &device,
        )
        .map_err(|e| JsValue::from_str(&format!("GGUF dequant load failed: {e}")))?;

    web_sys::console::log_1(
        &format!(
            "[wasm/gguf] dequantized {} tensors → BF16, layers={}",
            tensors.len(),
            cfg.text_config.num_hidden_layers,
        )
        .into(),
    );

    let vb = CandleVarBuilder::from_tensors(tensors, DType::BF16, &device);
    let model = Gemma4Model::new_partial(&cfg, vb, false, false)
        .map_err(|e| JsValue::from_str(&format!("Gemma4Model::new_partial: {e}")))?;

    let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e}")))?;

    let pipeline =
        Gemma4MultiModal::from_components(model, tokenizer, device.clone(), cfg.clone());

    Ok(LocalMultiModalHandle {
        inner: MultimodalInner::Gemma4 {
            pipeline: Arc::new(pipeline),
            gpu_device: device,
        },
        model_id,
        lazy: None,
        source: "gguf",
    })
}

// ── Quantized GGUF entry points — Phase 5 perf path ───────────────────────

/// `Read + Seek` adapter that pulls bytes through a JS callback. Used by
/// the chunked GGUF init (and any other reader-based loader) so we never
/// have to allocate the whole file as a single `Vec<u8>` in wasm linear
/// memory — Ollama Q4_K_M `gemma4:e2b` is a ~7.2 GB blob on disk
/// (LM + vision + audio); even its LM-only subset (~1.6 GB) overflows
/// `new Uint8Array(N)` in Chrome with "Array buffer allocation failed",
/// and reading the rest of the file is required to walk through GGUF
/// metadata anyway.
struct JsCallbackReader {
    read_fn: js_sys::Function,
    file_size: u64,
    cursor: u64,
}

impl JsCallbackReader {
    fn new(read_fn: js_sys::Function, file_size: u64) -> Self {
        Self { read_fn, file_size, cursor: 0 }
    }
}

impl std::io::Read for JsCallbackReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cursor >= self.file_size {
            return Ok(0);
        }
        let want = std::cmp::min(buf.len() as u64, self.file_size - self.cursor);
        if want == 0 {
            return Ok(0);
        }
        let bytes = crate::call_read_fn(&self.read_fn, self.cursor, want).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("JS read_fn failed at offset {}: {:?}", self.cursor, e),
            )
        })?;
        let n = bytes.len();
        if n == 0 {
            return Ok(0);
        }
        buf[..n].copy_from_slice(&bytes);
        self.cursor += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for JsCallbackReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            std::io::SeekFrom::Start(p) => p as i64,
            std::io::SeekFrom::End(p) => self.file_size as i64 + p,
            std::io::SeekFrom::Current(p) => self.cursor as i64 + p,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "JsCallbackReader: seek to negative position",
            ));
        }
        self.cursor = new_pos as u64;
        Ok(self.cursor)
    }
}


/// Build a text-only Gemma 4 handle from an Ollama Q4_K_M GGUF that
/// keeps weights as `QTensor` end-to-end. Inference runs on PR #3379's
/// `q4_k.pwgsl` quantized matmul kernel on WGPU and CPU dequant-on-fly
/// elsewhere — this is the path that actually realises Phase 5's
/// projected ~3-4× decode speedup.
///
/// Vision/audio are unavailable on this handle (Ollama gemma4:e2b
/// is text-only). For the multimodal path use
/// `init_local_multimodal_gguf` which dequantizes to BF16 at load and
/// wraps the existing `Gemma4MultiModal`.
#[wasm_bindgen]
pub async fn init_local_multimodal_gguf_quantized(
    weights: Vec<u8>,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalQuantizedHandle, JsValue> {
    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"[wasm/gguf-q] using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm/gguf-q] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    web_sys::console::log_1(
        &format!(
            "[wasm/gguf-q] loading GGUF blob ({} bytes), model_id={model_id}",
            weights.len()
        )
        .into(),
    );

    let mut cursor = std::io::Cursor::new(weights);
    let (model, cfg) =
        brainwires_provider::local_llm::gguf_loader::load_quantized_gemma4_from_reader(
            &mut cursor,
            &device,
        )
        .map_err(|e| JsValue::from_str(&format!("quantized_gemma4 load: {e}")))?;

    web_sys::console::log_1(
        &format!(
            "[wasm/gguf-q] quantized model built, layers={}",
            cfg.text_config.num_hidden_layers,
        )
        .into(),
    );

    let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e}")))?;

    let pipeline =
        brainwires_provider::local_llm::quantized_gemma4_pipeline::Gemma4QuantizedTextOnly::from_components(
            model, tokenizer, device, cfg,
        );

    Ok(LocalQuantizedHandle {
        inner: Arc::new(pipeline),
        model_id,
    })
}

/// Chunked variant of [`init_local_multimodal_gguf_quantized`] for large
/// Ollama blobs. Takes a JS callback `read_fn(offset, length) -> Uint8Array`
/// instead of a pre-loaded `Vec<u8>`, so the GGUF blob never has to be
/// materialised as a single allocation in either the JS heap or wasm
/// linear memory.
///
/// Caller (chat-pwa local-worker.js) holds the OPFS sync access handle
/// and drives `read_fn` from it; `gguf_loader::load_quantized_gemma4_from_reader`
/// walks the file once, reading metadata + each tensor on demand.
#[wasm_bindgen]
pub async fn init_local_multimodal_gguf_quantized_chunked(
    read_fn: js_sys::Function,
    file_size: f64,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalQuantizedHandle, JsValue> {
    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"[wasm/gguf-q-chunked] using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm/gguf-q-chunked] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    let file_size = file_size as u64;
    web_sys::console::log_1(
        &format!(
            "[wasm/gguf-q-chunked] streaming GGUF, file_size={file_size}, model_id={model_id}",
        )
        .into(),
    );

    let mut reader = JsCallbackReader::new(read_fn, file_size);
    let (model, cfg) =
        brainwires_provider::local_llm::gguf_loader::load_quantized_gemma4_from_reader(
            &mut reader,
            &device,
        )
        .map_err(|e| JsValue::from_str(&format!("quantized_gemma4 load: {e}")))?;

    web_sys::console::log_1(
        &format!(
            "[wasm/gguf-q-chunked] quantized model built, layers={}",
            cfg.text_config.num_hidden_layers,
        )
        .into(),
    );

    let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e}")))?;

    let pipeline =
        brainwires_provider::local_llm::quantized_gemma4_pipeline::Gemma4QuantizedTextOnly::from_components(
            model, tokenizer, device, cfg,
        );

    Ok(LocalQuantizedHandle {
        inner: Arc::new(pipeline),
        model_id,
    })
}

/// Text-only Gemma 4 handle backed by `Gemma4QuantizedTextOnly`. Mirrors
/// `LocalMultiModalHandle` in shape but doesn't carry a vision/audio
/// state machine. The `local_chat_stream_quantized` function below
/// drives generation through this handle.
#[wasm_bindgen]
pub struct LocalQuantizedHandle {
    inner: Arc<
        brainwires_provider::local_llm::quantized_gemma4_pipeline::Gemma4QuantizedTextOnly,
    >,
    model_id: String,
}

#[wasm_bindgen]
impl LocalQuantizedHandle {
    /// Model id (e.g. `"gemma4:e2b"`).
    #[wasm_bindgen(getter)]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// `"webgpu"` or `"cpu"` — which device the quantized weights live on.
    #[wasm_bindgen(getter)]
    pub fn device_type(&self) -> String {
        match self.inner.device().location() {
            brainwires_provider::CandleDeviceLocation::Cpu => "cpu".into(),
            brainwires_provider::CandleDeviceLocation::Wgpu { .. } => "webgpu".into(),
            _ => "unknown".into(),
        }
    }

    /// Always `false` — Ollama gemma4:e2b GGUF is text-only.
    #[wasm_bindgen(getter)]
    pub fn has_vision(&self) -> bool {
        false
    }

    /// Always `false`.
    #[wasm_bindgen(getter)]
    pub fn has_audio(&self) -> bool {
        false
    }

    /// Always `"gguf"` — paired with `LocalMultiModalHandle.source` so
    /// JS UIs can render a single badge.
    #[wasm_bindgen(getter)]
    pub fn source(&self) -> String {
        "gguf".into()
    }
}

/// Drive a text-only chat against a [`LocalQuantizedHandle`]. Mirrors
/// `local_chat_stream_with_image` (NDJSON `VisionWireChunk` framing
/// over a `ReadableStream<Uint8Array>`) so the JS-side reader can
/// consume both streams identically. Image parts in `messages_json`
/// are silently dropped — text-only model.
#[wasm_bindgen]
pub fn local_chat_stream_quantized(
    handle: &LocalQuantizedHandle,
    messages_json: String,
    params_json: String,
) -> Result<ReadableStream, JsValue> {
    let messages: Vec<JsMessage> = serde_json::from_str(&messages_json)
        .map_err(|e| JsValue::from_str(&format!("messages_json parse: {e}")))?;
    let params: VisionStreamParams = if params_json.trim().is_empty() {
        VisionStreamParams::default()
    } else {
        serde_json::from_str(&params_json)
            .map_err(|e| JsValue::from_str(&format!("params_json parse: {e}")))?
    };

    // Render messages into a Gemma 4 chat-template prompt. Image parts
    // are dropped (text-only model). The downstream pipeline calls
    // `tokenizer.encode(prompt, false)` so we include `<bos>` literally
    // here. Role mapping: `assistant` → `model` (Gemma 4's
    // chat_template.jinja convention; the model wasn't trained on the
    // bare `assistant` role string).
    let mut prompt = String::with_capacity(256);
    prompt.push_str("<bos>");
    for m in &messages {
        let role: &str = if m.role == "assistant" { "model" } else { m.role.as_str() };
        let text = match &m.content {
            JsContent::Text(s) => s.clone(),
            JsContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    JsPart::Text { text } => Some(text.clone()),
                    JsPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };
        // Gemma 4 chat tokens — registered in tokenizer.json
        // `added_tokens` array as `<|turn>` (105) / `<turn|>` (106).
        prompt.push_str("<|turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(text.trim());
        prompt.push_str("<turn|>\n");
    }
    prompt.push_str("<|turn>model\n");

    let pipeline = handle.inner.clone();
    let max_new_tokens = params.max_tokens.unwrap_or(512) as usize;

    let underlying = Object::new();
    let start_cb = Closure::once_into_js(move |controller: JsValue| {
        let controller: ReadableStreamDefaultController = match controller.dyn_into() {
            Ok(c) => c,
            Err(_) => return,
        };
        let pipeline = pipeline.clone();
        let prompt_owned = prompt.clone();
        spawn_local(async move {
            let send_chunk = |c: &ReadableStreamDefaultController, chunk: &VisionWireChunk<'_>| {
                if let Ok(json) = serde_json::to_string(chunk) {
                    let bytes = json.into_bytes();
                    let arr = Uint8Array::new_with_length(bytes.len() as u32);
                    arr.copy_from(&bytes);
                    let _ = c.enqueue_with_chunk(&arr);
                }
            };

            // Gemma 4 IT publishes three EOS tokens: 1 (`<eos>`), 106
            // (`<turn|>`, end-of-turn), and 50. Stop on any of them so
            // generation halts at the natural turn boundary instead of
            // running through `max_new_tokens`.
            let eos: &[u32] = &[1, 106, 50];
            let result = pipeline
                .generate_greedy_streaming(&prompt_owned, max_new_tokens, eos, |_, delta| {
                    send_chunk(
                        &controller,
                        &VisionWireChunk {
                            delta: Some(delta),
                            ..Default::default()
                        },
                    );
                })
                .await;

            match result {
                Ok(_) => {
                    send_chunk(
                        &controller,
                        &VisionWireChunk {
                            finished: true,
                            ..Default::default()
                        },
                    );
                }
                Err(e) => {
                    send_chunk(
                        &controller,
                        &VisionWireChunk {
                            error: Some(format!("{e}")),
                            finished: true,
                            ..Default::default()
                        },
                    );
                }
            }
            let _ = controller.close();
        });
    });
    Reflect::set(&underlying, &JsValue::from_str("start"), &start_cb)?;
    ReadableStream::new_with_underlying_source(&underlying)
}
