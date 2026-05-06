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
    Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig,
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
        // Gemma 4 PLE / AltUp / Laurel — names follow llama.cpp's
        // gemma3n/gemma4 naming. If an Ollama publication uses
        // different suffixes the model will simply not load that
        // tower and the relevant disable_* flags should be set.
        "per_layer_inp_gate.weight" => "per_layer_input_gate.weight",
        "per_layer_proj.weight" => "per_layer_projection.weight",
        "per_layer_model_proj.weight" => "per_layer_model_projection.weight",
        "altup_proj.weight" => "altup_projections.weight",
        "altup_unembd_proj.weight" => "altup_unembed_projections.weight",
        "laurel_l.weight" => "laurel.linear_left.weight",
        "laurel_r.weight" => "laurel.linear_right.weight",
        "layer_scalar" => "layer_scalar",
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
    let intermediate_size = md_get("feed_forward_length")?.to_u32()? as usize;
    let head_dim = md_get("attention.key_length")?.to_u32()? as usize;
    let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let max_position_embeddings = md_get("context_length")?.to_u32()? as usize;
    let rope_theta = md_get_opt("rope.freq_base")
        .and_then(|m| m.to_f32().ok())
        .unwrap_or(1_000_000.0) as f64;

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
        .unwrap_or(4096) as usize;
    let sliding_window_pattern = md_get_opt("attention.sliding_window_type")
        .and_then(|m| m.to_u32().ok())
        .unwrap_or(5) as usize;
    let layer_types: Vec<String> = (0..num_hidden_layers)
        .map(|i| {
            // Mirror the llama.cpp convention used by quantized_gemma3:
            // sliding when (i + 1) % sliding_window_pattern > 0.
            let is_sliding = (i + 1) % sliding_window_pattern > 0;
            if is_sliding {
                "sliding_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect();

    // KV-share. Use GGUF metadata if present, else apply canonical
    // heuristic (matches the chat-pwa wasm fix in commit dca60315).
    let num_kv_shared_layers = md_get_opt("attention.kv_shared_layers")
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or_else(|| match num_hidden_layers {
            35 => 20,
            30 => 10,
            _ => 0,
        });

    let text_config = Gemma4TextConfig {
        attention_bias: false,
        head_dim,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size,
        intermediate_size,
        intermediate_sizes: None,
        num_attention_heads,
        num_hidden_layers,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta,
        vocab_size,
        sliding_window,
        final_logit_softcapping: None,
        // Gemma 4 sets pre-softmax scale to 1.0 (commit b38856b1).
        // The struct stores it as a usize divisor under sqrt; 1
        // corresponds to scale = 1/sqrt(1) = 1.0.
        query_pre_attn_scalar: 1,
        max_position_embeddings,
        tie_word_embeddings: true,
        sliding_window_pattern,
        layer_types,
        global_head_dim: head_dim,
        num_global_key_value_heads: None,
        rope_parameters: None,
        use_bidirectional_attention: None,
        use_flash_attn: false,
        // PLE / AltUp / Laurel / sparsity / layer_scalar default-off
        // until we verify Ollama GGUF publishes the relevant
        // metadata + tensors.
        hidden_size_per_layer_input: None,
        vocab_size_per_layer_input: None,
        altup_num_inputs: 1,
        altup_active_idx: 0,
        altup_correct_scale: false,
        altup_coef_clip: None,
        laurel_rank: 0,
        activation_sparsity_pattern: None,
        disable_altup: true,
        disable_laurel: true,
        disable_per_layer_input_gate: true,
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
}
