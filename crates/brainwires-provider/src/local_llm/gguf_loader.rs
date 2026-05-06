//! GGUF → BF16 weight-map loader for Gemma 4 (Phase 4 part 3).
//!
//! Reads an Ollama-format Gemma 4 GGUF, dequantizes every tensor to BF16,
//! translates each GGUF tensor name (`blk.0.attn_q.weight`) to the HF
//! safetensors equivalent (`model.language_model.layers.0.self_attn.q_proj.weight`)
//! the existing `gemma4::text::TextModel` / `Gemma4Config` consumers expect,
//! and returns a `HashMap<String, Tensor>` ready to wrap in
//! `VarBuilder::from_tensors`.
//!
//! **No perf gain on its own** — this is the dequantize-at-load path. Q4_K_M
//! weights become BF16 in memory, so VRAM/RAM usage matches the safetensors
//! path. The win is download size (~6× smaller). Phase 5's WGPU Q4_K_M
//! dequant matmul kernel (already shipped in PR #3379) becomes reachable
//! once we swap `QTensor → Linear(bf16)` for `QTensor → QMatMul`, which is
//! follow-up work.
//!
//! Tensor name mapping follows llama.cpp / ollama convention. The fields
//! Gemma 4 needs that aren't in the standard `gemma3` GGUF naming
//! (per-layer embeddings, AltUp projections, Laurel, layer_scalar) are
//! mapped from their ollama-published names — see [`gguf_to_hf_name`] for
//! the full table. Missing tensors are tolerated: the loader emits only
//! the ones present, and `Gemma4TextConfig` flags
//! (`disable_altup`, `disable_laurel`, `disable_per_layer_input_gate`)
//! gate which tower paths actually run.
//!
//! Native-only — gated to non-wasm so the chat-pwa wasm bundle doesn't
//! pull in `std::fs`. The wasm side has its own `Read+Seek` shim around
//! the JS-supplied byte stream and reuses [`gguf_to_hf_name`] +
//! [`build_gemma4_config_from_gguf`] from this module.

#![cfg(feature = "local-llm-candle")]

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::models::gemma4::config::{
    Gemma4Config, Gemma4RopeLayerParams, Gemma4RopeParameters, Gemma4TextConfig,
    Gemma4VisionConfig,
};

/// Translate one GGUF tensor name into the HF safetensors name the
/// existing `gemma4::text::TextModel` VarBuilder lookups expect.
///
/// Returns `None` for GGUF tensors that don't have an HF equivalent
/// (e.g. `rope_freqs.weight` — RoPE tables are computed at runtime
/// from the config, not loaded from weights). Caller skips those.
pub fn gguf_to_hf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.language_model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.language_model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        "rope_freqs.weight" => return None,
        // PLE top-level tensors. Ollama gemma4:e2b uses bare names;
        // older Gemma 3n publications used `_projection` / `_norm`.
        "per_layer_model_proj.weight" | "per_layer_model_projection.weight" => {
            return Some("model.language_model.per_layer_model_projection.weight".into())
        }
        "per_layer_proj_norm.weight" | "per_layer_projection_norm.weight" => {
            return Some("model.language_model.per_layer_projection_norm.weight".into())
        }
        "per_layer_token_embd.weight" | "embed_tokens_per_layer.weight" => {
            return Some("model.language_model.embed_tokens_per_layer.weight".into())
        }
        _ => {}
    }

    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let (layer_str, suffix) = rest.split_at(dot);
    let layer_idx: usize = layer_str.parse().ok()?;
    let suffix = &suffix[1..];

    let hf_suffix = match suffix {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
        "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // Gemma 4 PLE — Ollama-published gemma4:e2b uses bare names
        // (`inp_gate`, `proj`, `post_norm`, `layer_output_scale`).
        // Older llama.cpp gemma3n style is `per_layer_inp_gate` /
        // `per_layer_proj` etc; we accept both for compatibility.
        "inp_gate.weight" | "per_layer_inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" | "per_layer_proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" | "post_per_layer_input_norm.weight" => {
            "post_per_layer_input_norm.weight"
        }
        "layer_output_scale.weight" | "layer_scalar" => "layer_scalar",
        // AltUp / Laurel — only present in some Gemma 3n publications;
        // gemma4:e2b on Ollama doesn't ship these, so the model simply
        // skips them at load time.
        "altup_proj.weight" => "altup_projections.weight",
        "altup_unembd_proj.weight" => "altup_unembed_projections.weight",
        "laurel_l.weight" => "laurel.linear_left.weight",
        "laurel_r.weight" => "laurel.linear_right.weight",
        _ => return None,
    };
    Some(format!(
        "model.language_model.layers.{layer_idx}.{hf_suffix}"
    ))
}

/// Read a Gemma 4 GGUF file from a local path. Native-only convenience
/// wrapper over [`load_gemma4_gguf_from_reader`]; the wasm chat-pwa
/// path uses an in-memory `Cursor` over the OPFS blob instead.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_gemma4_gguf(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, Gemma4Config)> {
    let mut file = std::fs::File::open(path.as_ref())?;
    load_gemma4_gguf_from_reader(&mut file, device)
}

/// Read a Gemma 4 GGUF from any `Read + Seek` source (file on native,
/// `Cursor<Vec<u8>>` on wasm), dequantize every tensor to BF16, and
/// translate GGUF tensor names to the HF safetensors keys the existing
/// `Gemma4Model` consumer expects.
pub fn load_gemma4_gguf_from_reader<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, Gemma4Config)> {
    let content = gguf_file::Content::read(reader)?;
    let cfg = build_gemma4_config_from_gguf(&content)?;

    let mut out = HashMap::with_capacity(content.tensor_infos.len());
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for gguf_name in names {
        let Some(hf_name) = gguf_to_hf_name(&gguf_name) else {
            continue;
        };
        let qtensor = content.tensor(reader, &gguf_name, device)?;
        let tensor = qtensor.dequantize(device)?.to_dtype(DType::BF16)?;
        out.insert(hf_name, tensor);
    }
    Ok((out, cfg))
}

/// Read a Gemma 4 GGUF and build a `quantized_gemma4::ModelWeights`
/// directly — keeps weights as `QTensor` end-to-end so PR #3379's
/// quantized matmul kernels (`q4_k.pwgsl` / `q5_k.pwgsl` / etc on
/// WGPU, CPU dequant-on-fly elsewhere) carry the inference workload.
/// This is the perf-bearing path; `load_gemma4_gguf_from_reader`
/// above is the dequant-at-load fallback that runs the existing BF16
/// model.
///
/// **Limitation:** the basic decoder only — auxiliary towers (PLE /
/// AltUp / LAuReL / KV-share / layer_scalar / activation sparsity)
/// are gated off until reference-verified. Output won't bit-match a
/// canonical Gemma 4 forward pass.
pub fn load_quantized_gemma4_from_reader<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    device: &Device,
) -> Result<(
    candle_transformers::models::quantized_gemma4::ModelWeights,
    Gemma4Config,
)> {
    let content = gguf_file::Content::read(reader)?;
    let cfg = build_gemma4_config_from_gguf(&content)?;
    let model = candle_transformers::models::quantized_gemma4::ModelWeights::from_gguf(
        content,
        reader,
        device,
        &cfg.text_config,
    )?;
    Ok((model, cfg))
}

/// Build a [`Gemma4Config`] from a GGUF metadata kv-store.
///
/// Falls back to canonical Gemma 4 E2B values when an optional key is
/// absent. AltUp / Laurel / PLE are *disabled by default* on this path —
/// Ollama GGUFs may not publish the full Gemma 4 schema, so disabling
/// the auxiliary towers lets us at least produce coherent text from the
/// classic decoder. The flags can be re-enabled once we verify which
/// metadata keys the Ollama publication actually uses.
pub fn build_gemma4_config_from_gguf(ct: &gguf_file::Content) -> Result<Gemma4Config> {
    use candle_core::bail;

    let prefix = ["gemma4", "gemma3"]
        .iter()
        .find(|p| {
            ct.metadata
                .contains_key(&format!("{p}.attention.head_count"))
        })
        .copied()
        .unwrap_or("gemma4");

    let md_get = |s: &str| {
        let key = format!("{prefix}.{s}");
        match ct.metadata.get(&key) {
            None => bail!("cannot find {key} in GGUF metadata"),
            Some(v) => Ok(v),
        }
    };
    let md_get_opt = |s: &str| {
        let key = format!("{prefix}.{s}");
        ct.metadata.get(&key).cloned()
    };

    let num_attention_heads = md_get("attention.head_count")?.to_u32()? as usize;
    let num_key_value_heads = md_get("attention.head_count_kv")?.to_u32()? as usize;
    let num_hidden_layers = md_get("block_count")?.to_u32()? as usize;
    let hidden_size = md_get("embedding_length")?.to_u32()? as usize;

    // `feed_forward_length` is published as either a single u32 (older
    // gemma3 schema) or an Array of i32 with one entry per layer
    // (Gemma 4's elastic MLP — E2B uses 6144 for layers 0-14 and 12288
    // for 15-34). Handle both.
    let (intermediate_size, intermediate_sizes) = match md_get("feed_forward_length")? {
        gguf_file::Value::Array(arr) => {
            let mut sizes = Vec::with_capacity(arr.len());
            for v in arr {
                let n = match v {
                    gguf_file::Value::I32(n) => *n as usize,
                    gguf_file::Value::U32(n) => *n as usize,
                    other => bail!(
                        "feed_forward_length array entry has unexpected type: {other:?}"
                    ),
                };
                sizes.push(n);
            }
            let first = sizes.first().copied().unwrap_or(0);
            (first, Some(sizes))
        }
        v => (v.to_u32()? as usize, None),
    };

    // Sliding (SWA) head_dim from `attention.key_length_swa`, full-attn
    // head_dim from `attention.key_length`. Older gemma3 publications
    // don't separate these, so fall back to a single key_length.
    let head_dim_swa = md_get_opt("attention.key_length_swa")
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize);
    let head_dim_full = md_get("attention.key_length")?.to_u32()? as usize;
    let head_dim = head_dim_swa.unwrap_or(head_dim_full);
    let global_head_dim = head_dim_full;

    let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let max_position_embeddings = md_get("context_length")?.to_u32()? as usize;

    // RoPE theta: full-attn uses `rope.freq_base`, sliding uses
    // `rope.freq_base_swa`. Stored on the per-rope-type config.
    let rope_theta_full = md_get_opt("rope.freq_base")
        .and_then(|m| m.to_f32().ok())
        .unwrap_or(1_000_000.0) as f64;
    let rope_theta_swa = md_get_opt("rope.freq_base_swa")
        .and_then(|m| m.to_f32().ok())
        .unwrap_or(10_000.0) as f64;

    // Vocab size from the embedding tensor shape (more reliable than
    // metadata, which may not carry it).
    let vocab_size = ct
        .tensor_infos
        .get("token_embd.weight")
        .map(|ti| ti.shape.dims()[0])
        .ok_or_else(|| {
            candle_core::Error::Msg("missing token_embd.weight in GGUF".into())
        })?;

    let sliding_window = md_get_opt("attention.sliding_window")
        .and_then(|m| m.to_u32().ok())
        .unwrap_or(512) as usize;

    // `attention.sliding_window_pattern` is published as an Array of Bool
    // in Gemma 4 (true = sliding, false = full). Older Gemma 3 schemas
    // used `attention.sliding_window_type` as a scalar period (every Nth
    // layer is full). Support both, derive `layer_types` accordingly.
    let layer_types: Vec<String> = match md_get_opt("attention.sliding_window_pattern") {
        Some(gguf_file::Value::Array(arr)) => {
            let mut types = Vec::with_capacity(arr.len());
            for v in &arr {
                let is_sliding = matches!(v, gguf_file::Value::Bool(true));
                types.push(if is_sliding {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                });
            }
            // Pad/truncate to num_hidden_layers if metadata length
            // disagrees (defensive — should always match).
            types.resize(num_hidden_layers, "sliding_attention".to_string());
            types
        }
        _ => {
            let period = md_get_opt("attention.sliding_window_type")
                .and_then(|m| m.to_u32().ok())
                .unwrap_or(5) as usize;
            (0..num_hidden_layers)
                .map(|i| {
                    let is_sliding = (i + 1) % period > 0;
                    if is_sliding {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect()
        }
    };
    // For the legacy `sliding_window_pattern: usize` field, derive a
    // period from the actual pattern (count between non-sliding layers).
    let sliding_window_pattern = layer_types
        .iter()
        .position(|t| t == "full_attention")
        .map(|p| p + 1)
        .unwrap_or(num_hidden_layers);

    // KV-share. Gemma 4 uses `attention.shared_kv_layers`; older
    // gemma3 publications used `attention.kv_shared_layers`. Fall back
    // to canonical heuristics when neither key is present.
    let num_kv_shared_layers = md_get_opt("attention.shared_kv_layers")
        .or_else(|| md_get_opt("attention.kv_shared_layers"))
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or_else(|| match num_hidden_layers {
            35 => 20,
            30 => 10,
            _ => 0,
        });

    let final_logit_softcapping = md_get_opt("final_logit_softcapping")
        .and_then(|m| m.to_f32().ok())
        .map(|v| v as f64);

    let hidden_size_per_layer_input = md_get_opt("embedding_length_per_layer_input")
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize);

    // Surface dual-RoPE base frequencies through the `rope_parameters`
    // sub-config so `Gemma4TextConfig::rope_local_base_freq()` and
    // `partial_rotary_factor()` resolve to the right values.
    let rope_parameters = Some(Gemma4RopeParameters {
        full_attention: Some(Gemma4RopeLayerParams {
            rope_theta: Some(rope_theta_full),
            rope_type: None,
            partial_rotary_factor: Some(0.25),
        }),
        sliding_attention: Some(Gemma4RopeLayerParams {
            rope_theta: Some(rope_theta_swa),
            rope_type: None,
            partial_rotary_factor: None,
        }),
        rope_theta: Some(rope_theta_full),
        rope_type: None,
        partial_rotary_factor: Some(0.25),
    });

    // PLE is enabled when `embedding_length_per_layer_input` is present
    // and the GGUF carries the per-layer model projection tensor.
    let has_ple = hidden_size_per_layer_input.is_some()
        && (ct.tensor_infos.contains_key("per_layer_model_proj.weight")
            || ct
                .tensor_infos
                .contains_key("per_layer_model_projection.weight"));

    let text_config = Gemma4TextConfig {
        attention_bias: false,
        head_dim,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size,
        intermediate_size,
        intermediate_sizes,
        num_attention_heads,
        num_hidden_layers,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta: rope_theta_full,
        vocab_size,
        sliding_window,
        final_logit_softcapping,
        // Gemma 4 sets pre-softmax scale to 1.0 (commit b38856b1).
        // The struct stores it as a usize divisor under sqrt; 1
        // corresponds to scale = 1/sqrt(1) = 1.0.
        query_pre_attn_scalar: 1,
        max_position_embeddings,
        tie_word_embeddings: true,
        sliding_window_pattern,
        layer_types,
        global_head_dim,
        num_global_key_value_heads: None,
        rope_parameters,
        use_bidirectional_attention: None,
        use_flash_attn: false,
        hidden_size_per_layer_input,
        vocab_size_per_layer_input: None,
        // Gemma 4 E2B doesn't ship AltUp / Laurel in the Ollama
        // publication — disable both. PLE turns on iff the GGUF
        // carries the projection tensor.
        altup_num_inputs: 1,
        altup_active_idx: 0,
        altup_correct_scale: false,
        altup_coef_clip: None,
        laurel_rank: 0,
        activation_sparsity_pattern: None,
        disable_altup: true,
        disable_laurel: true,
        disable_per_layer_input_gate: !has_ple,
        num_kv_shared_layers,
    };

    // Vision config — every field has a serde default, so deserialise
    // from an empty object to get the canonical Gemma 3n vision shape.
    // Ollama's text-only Gemma 4 GGUFs won't carry vision tensors, so
    // this stays unused at runtime; the field is required by the
    // struct definition though.
    let vision_config: Gemma4VisionConfig =
        serde_json::from_value(serde_json::json!({})).map_err(|e| {
            candle_core::Error::Msg(format!("failed to build default vision config: {e}"))
        })?;

    Ok(Gemma4Config {
        text_config,
        vision_config,
        audio_config: None,
        image_token_id: 258880,
        audio_token_id: 258881,
        video_token_id: 258884,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_top_level() {
        assert_eq!(
            gguf_to_hf_name("token_embd.weight").as_deref(),
            Some("model.language_model.embed_tokens.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("output_norm.weight").as_deref(),
            Some("model.language_model.norm.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("output.weight").as_deref(),
            Some("lm_head.weight"),
        );
        assert_eq!(gguf_to_hf_name("rope_freqs.weight"), None);
    }

    #[test]
    fn name_mapping_per_layer_attention() {
        assert_eq!(
            gguf_to_hf_name("blk.5.attn_q.weight").as_deref(),
            Some("model.language_model.layers.5.self_attn.q_proj.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_k_norm.weight").as_deref(),
            Some("model.language_model.layers.0.self_attn.k_norm.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("blk.34.attn_output.weight").as_deref(),
            Some("model.language_model.layers.34.self_attn.o_proj.weight"),
        );
    }

    #[test]
    fn name_mapping_per_layer_mlp_and_norm() {
        assert_eq!(
            gguf_to_hf_name("blk.7.ffn_gate.weight").as_deref(),
            Some("model.language_model.layers.7.mlp.gate_proj.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("blk.7.attn_norm.weight").as_deref(),
            Some("model.language_model.layers.7.input_layernorm.weight"),
        );
        assert_eq!(
            gguf_to_hf_name("blk.7.post_ffw_norm.weight").as_deref(),
            Some("model.language_model.layers.7.post_feedforward_layernorm.weight"),
        );
    }

    #[test]
    fn name_mapping_unknown_returns_none() {
        assert_eq!(gguf_to_hf_name("blk.foo.bar"), None);
        assert_eq!(gguf_to_hf_name("not_a_blk_tensor"), None);
    }

    /// Hand-roll a `gguf_file::Content` with the metadata fields the
    /// loader actually reads. This bypasses the file-format parsing so
    /// we can exercise `build_gemma4_config_from_gguf` without an
    /// actual GGUF file. Useful for catching regressions in field
    /// names / types / fallback logic.
    fn synthetic_gemma4_e2b_content() -> gguf_file::Content {
        use candle_core::Shape;
        use candle_core::quantized::GgmlDType;
        use gguf_file::{TensorInfo, Value, VersionedMagic};

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("gemma4.attention.head_count".into(), Value::U32(8));
        metadata.insert("gemma4.attention.head_count_kv".into(), Value::U32(2));
        metadata.insert("gemma4.block_count".into(), Value::U32(35));
        metadata.insert("gemma4.embedding_length".into(), Value::U32(1536));
        metadata.insert("gemma4.feed_forward_length".into(), Value::U32(8192));
        metadata.insert("gemma4.attention.key_length".into(), Value::U32(256));
        metadata.insert(
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-6),
        );
        metadata.insert("gemma4.context_length".into(), Value::U32(8192));
        metadata.insert("gemma4.rope.freq_base".into(), Value::F32(1_000_000.0));

        // Need token_embd.weight to exist so vocab_size lookup works.
        let mut tensor_infos = std::collections::HashMap::new();
        tensor_infos.insert(
            "token_embd.weight".into(),
            TensorInfo {
                ggml_dtype: GgmlDType::Q4K,
                shape: Shape::from((262144, 1536)),
                offset: 0,
            },
        );

        gguf_file::Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn config_from_gguf_gemma4_e2b_canonical() {
        let content = synthetic_gemma4_e2b_content();
        let cfg = build_gemma4_config_from_gguf(&content)
            .expect("config build should succeed for canonical metadata");
        assert_eq!(cfg.text_config.num_attention_heads, 8);
        assert_eq!(cfg.text_config.num_key_value_heads, 2);
        assert_eq!(cfg.text_config.num_hidden_layers, 35);
        assert_eq!(cfg.text_config.hidden_size, 1536);
        assert_eq!(cfg.text_config.intermediate_size, 8192);
        assert_eq!(cfg.text_config.head_dim, 256);
        assert_eq!(cfg.text_config.vocab_size, 262144);
        assert_eq!(cfg.text_config.max_position_embeddings, 8192);
        assert_eq!(cfg.text_config.rope_theta, 1_000_000.0);
        // E2B canonical: 35 layers → 20 KV-shared.
        assert_eq!(cfg.text_config.num_kv_shared_layers, 20);
        // Auxiliary towers default-disabled on the GGUF path.
        assert!(cfg.text_config.disable_altup);
        assert!(cfg.text_config.disable_laurel);
        assert!(cfg.text_config.disable_per_layer_input_gate);
    }

    #[test]
    fn config_from_gguf_kv_share_30_layer_fallback() {
        let mut content = synthetic_gemma4_e2b_content();
        content.metadata.insert(
            "gemma4.block_count".into(),
            gguf_file::Value::U32(30),
        );
        let cfg = build_gemma4_config_from_gguf(&content).unwrap();
        // 30-layer fallback: 10 KV-shared (Gemma 3n canonical).
        assert_eq!(cfg.text_config.num_kv_shared_layers, 10);
    }

    #[test]
    fn config_from_gguf_kv_share_explicit_metadata_wins() {
        let mut content = synthetic_gemma4_e2b_content();
        // Even with the canonical 35-layer count, an explicit
        // `kv_shared_layers` metadata key should override.
        content.metadata.insert(
            "gemma4.attention.kv_shared_layers".into(),
            gguf_file::Value::U32(15),
        );
        let cfg = build_gemma4_config_from_gguf(&content).unwrap();
        assert_eq!(cfg.text_config.num_kv_shared_layers, 15);
    }

    #[test]
    fn config_from_gguf_missing_required_metadata_errors() {
        let mut content = synthetic_gemma4_e2b_content();
        content.metadata.remove("gemma4.attention.head_count");
        let err = build_gemma4_config_from_gguf(&content).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("attention.head_count"),
            "error should mention the missing key, got: {msg}"
        );
    }
}
