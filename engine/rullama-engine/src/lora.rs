//! Inference-time LoRA adapter state.
//!
//! Lean cousin of `rullama-lora::lora::LoraState` — holds just the
//! frozen `A`, `B`, and a per-LoRA `z` scratch buffer. No gradient
//! buffers, no Adam moments; those live with the trainable adapter on
//! the finetune side. The on-disk safetensors format is shared, so a
//! training session can save an adapter and an inference `Model` can
//! load it without a roundtrip through the finetune crate.
//!
//! Tensor naming: `lora.blk.{layer}.{projection}.{A|B}`, where
//! `projection` is one of `attn_q`/`attn_k`/`attn_v`/`attn_o`/
//! `ffn_gate`/`ffn_up`/`ffn_down`. Metadata sidecar must carry
//! `rank` / `alpha` / `target_modules` so the loader can rebuild
//! shapes from the model config without external context.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};
use crate::model::config::Gemma4Config;
use crate::reference::forward_chained::{GlobalLoraSlots, LayerLoraSlots, LoraSlot};

/// Identifies one LoRA wrapper.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoraKey {
    pub layer: u32,
    pub projection: String,
}

impl LoraKey {
    pub fn new(layer: u32, projection: impl Into<String>) -> Self {
        Self {
            layer,
            projection: projection.into(),
        }
    }
}

/// One LoRA wrapper's inference-time state.
pub struct InferenceLoraLayer {
    pub in_dim: u32,
    pub rank: u32,
    pub out_dim: u32,
    /// `alpha / rank` — runtime scale.
    pub scale: f32,
    /// A matrix, `[rank, in_dim]` row-major, f32.
    pub a: Buffer,
    /// B matrix, `[out_dim, rank]` row-major. f32 by default; when
    /// `b_is_f16` is true the buffer is half-sized and holds packed
    /// f16 pairs (two elements per `u32`).
    pub b: Buffer,
    /// `[rank]` scratch holding `A·x` from the forward correction.
    pub z: Buffer,
    /// `true` iff `b` is packed f16. Currently set only for the
    /// `lm_head` global LoRA slot to halve bandwidth on the
    /// `vocab × rank` matmul. Requires `rank` to be even.
    pub b_is_f16: bool,
}

/// Loaded inference adapter — a collection of `InferenceLoraLayer`
/// keyed by `(layer_idx, projection)`.
///
/// Held on `Model` as `Option<InferenceAdapter>`. When `Some`, the
/// regular `Model::step_native` path automatically routes through
/// `Forward::step_with_lora` instead of `Forward::step`.
pub struct InferenceAdapter {
    layers: BTreeMap<LoraKey, InferenceLoraLayer>,
}

impl InferenceAdapter {
    /// Build an adapter from a safetensors byte buffer + the target
    /// model's config. The metadata sidecar's `rank` / `alpha` /
    /// `target_modules` drive shape allocation; `Gemma4Config`
    /// provides the per-layer `d_model`, `n_heads * head_dim`, KV
    /// dims, and FFN width.
    pub fn from_safetensors_bytes(
        ctx: Arc<WgpuCtx>,
        cfg: &Gemma4Config,
        bytes: &[u8],
    ) -> Result<Self> {
        let (_n, header) = SafeTensors::read_metadata(bytes)
            .map_err(|e| RullamaError::Inference(format!("safetensors header: {e}")))?;
        let meta_opt: &Option<HashMap<String, String>> = header.metadata();
        let m = meta_opt
            .as_ref()
            .ok_or_else(|| RullamaError::Inference("adapter has no metadata sidecar".into()))?;
        let rank: u32 = m
            .get("rank")
            .ok_or_else(|| RullamaError::Inference("metadata missing 'rank'".into()))?
            .parse()
            .map_err(|e| RullamaError::Inference(format!("bad 'rank': {e}")))?;
        let alpha: f32 = m
            .get("alpha")
            .ok_or_else(|| RullamaError::Inference("metadata missing 'alpha'".into()))?
            .parse()
            .map_err(|e| RullamaError::Inference(format!("bad 'alpha': {e}")))?;
        let targets_csv = m
            .get("target_modules")
            .ok_or_else(|| RullamaError::Inference("metadata missing 'target_modules'".into()))?;
        let target_modules: Vec<String> = targets_csv
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if target_modules.is_empty() {
            return Err(RullamaError::Inference(
                "target_modules metadata is empty".into(),
            ));
        }

        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| RullamaError::Inference(format!("safetensors parse: {e}")))?;

        // Allocate shape-matched slots for every (layer, projection).
        // `lm_head` and `embed_tokens` are "global" — allocated once at
        // layer=0 by convention; matches the training-side keying in
        // `rullama-lora::session::build_lora_state`.
        let mut layers: BTreeMap<LoraKey, InferenceLoraLayer> = BTreeMap::new();
        let d_model = cfg.d_model;
        let vocab = cfg.vocab_size;
        const GLOBAL_TARGETS: &[&str] = &["lm_head", "embed_tokens"];
        for li in 0..cfg.n_layers {
            let head_dim = cfg.head_dim(li);
            let n_heads_dim = cfg.n_heads * head_dim;
            let n_kv_dim = cfg.n_kv_heads(li) * head_dim;
            let ffn_n = cfg.ffn(li);
            for proj in &target_modules {
                if GLOBAL_TARGETS.contains(&proj.as_str()) {
                    continue; // global pass below
                }
                let (in_dim, out_dim) = match proj.as_str() {
                    "attn_q" => (d_model, n_heads_dim),
                    "attn_k" => (d_model, n_kv_dim),
                    "attn_v" => (d_model, n_kv_dim),
                    "attn_o" => (n_heads_dim, d_model),
                    "ffn_gate" => (d_model, ffn_n),
                    "ffn_up" => (d_model, ffn_n),
                    "ffn_down" => (ffn_n, d_model),
                    other => {
                        return Err(RullamaError::Inference(format!(
                            "unsupported LoRA target '{other}'"
                        )));
                    }
                };
                let layer = InferenceLoraLayer::alloc(&ctx, in_dim, rank, out_dim, alpha, false);
                layers.insert(LoraKey::new(li, proj.clone()), layer);
            }
        }
        // Global targets — allocate once each (keyed at layer=0).
        for proj in &target_modules {
            if !GLOBAL_TARGETS.contains(&proj.as_str()) {
                continue;
            }
            let (in_dim, out_dim) = match proj.as_str() {
                "lm_head" => (d_model, vocab),
                "embed_tokens" => (vocab, d_model),
                _ => unreachable!("filter above admits only GLOBAL_TARGETS"),
            };
            // Only `lm_head` enables f16-packed B: its [vocab, rank]
            // matrix (~16 MB at f32 for Gemma 4 vocab=262 144, rank=16)
            // dominates LoRA bandwidth per token; quantizing halves the
            // phase-2 B·z read. embed_tokens uses the column-indexed
            // path, not the fused kernel — leave it f32.
            let b_is_f16 = proj == "lm_head" && rank.is_multiple_of(2);
            let layer = InferenceLoraLayer::alloc(&ctx, in_dim, rank, out_dim, alpha, b_is_f16);
            layers.insert(LoraKey::new(0, proj.clone()), layer);
        }

        // Upload every matching tensor from the file.
        for (name, tensor) in st.tensors() {
            if !name.starts_with("lora.blk.") {
                continue;
            }
            let suffix = &name["lora.blk.".len()..];
            let (layer_str, rest) = match suffix.split_once('.') {
                Some(p) => p,
                None => continue,
            };
            let layer_idx: u32 = match layer_str.parse() {
                Ok(n) => n,
                Err(_) => continue,
            };
            let (projection, ab) = match rest.rsplit_once('.') {
                Some(p) => p,
                None => continue,
            };
            let key = LoraKey::new(layer_idx, projection.to_string());
            let layer = match layers.get(&key) {
                Some(l) => l,
                None => continue,
            };
            let (buf, target_packed_f16) = match ab {
                "A" => (&layer.a, false),
                "B" => (&layer.b, layer.b_is_f16),
                _ => continue,
            };
            let data = tensor.data();
            // Element count of the LoRA tensor regardless of buffer-side
            // packing. `buf.size()` is in BYTES — f32 is 4/elem, packed
            // f16 is 2/elem.
            let n_elems = if target_packed_f16 {
                (buf.size() / 2) as usize
            } else {
                (buf.size() / 4) as usize
            };
            let upload_bytes: Vec<u8> = match tensor.dtype() {
                Dtype::F32 => {
                    if data.len() != n_elems * 4 {
                        return Err(RullamaError::Inference(format!(
                            "tensor {name} f32 size mismatch: file={} expected={}",
                            data.len(),
                            n_elems * 4
                        )));
                    }
                    if target_packed_f16 {
                        // f32 -> f16 quantize, then memcpy. The kernel
                        // reads each u32 as two consecutive f16 elements
                        // via unpack2x16float (little-endian: .x = low 16
                        // bits → even index).
                        let src: &[f32] = bytemuck::cast_slice(data);
                        let packed = pack_f32_to_f16_pairs(src);
                        bytemuck::cast_slice::<u32, u8>(&packed).to_vec()
                    } else {
                        data.to_vec()
                    }
                }
                Dtype::F16 => {
                    if data.len() != n_elems * 2 {
                        return Err(RullamaError::Inference(format!(
                            "tensor {name} f16 size mismatch: file={} expected={}",
                            data.len(),
                            n_elems * 2
                        )));
                    }
                    if target_packed_f16 {
                        // Already f16 on disk; reinterpret as u32 pairs.
                        data.to_vec()
                    } else {
                        let h: &[half::f16] = bytemuck::cast_slice(data);
                        let f: Vec<f32> = h.iter().map(|&x| x.to_f32()).collect();
                        bytemuck::cast_slice::<f32, u8>(&f).to_vec()
                    }
                }
                other => {
                    return Err(RullamaError::Inference(format!(
                        "tensor {name} unsupported dtype {other:?}"
                    )));
                }
            };
            ctx.queue.write_buffer(buf, 0, &upload_bytes);
        }

        Ok(Self { layers })
    }

    /// Build per-layer `LayerLoraSlots` borrowed views — fed into
    /// `Forward::step_with_lora` for adapter-aware inference.
    pub fn layer_slots(&self, n_layers: u32) -> Vec<LayerLoraSlots<'_>> {
        (0..n_layers)
            .map(|li| LayerLoraSlots {
                q: self.layers.get(&LoraKey::new(li, "attn_q")).map(slot_view),
                k: self.layers.get(&LoraKey::new(li, "attn_k")).map(slot_view),
                v: self.layers.get(&LoraKey::new(li, "attn_v")).map(slot_view),
                o: self.layers.get(&LoraKey::new(li, "attn_o")).map(slot_view),
                ffn_gate: self
                    .layers
                    .get(&LoraKey::new(li, "ffn_gate"))
                    .map(slot_view),
                ffn_up: self.layers.get(&LoraKey::new(li, "ffn_up")).map(slot_view),
                ffn_down: self
                    .layers
                    .get(&LoraKey::new(li, "ffn_down"))
                    .map(slot_view),
            })
            .collect()
    }

    /// Build the model-global `GlobalLoraSlots` view (lm_head + embed_tokens).
    /// Either slot is `None` when the adapter doesn't include that target.
    /// Returns `GlobalLoraSlots::default()` if neither was loaded.
    pub fn global_slots(&self) -> GlobalLoraSlots<'_> {
        GlobalLoraSlots {
            embed_tokens: self
                .layers
                .get(&LoraKey::new(0, "embed_tokens"))
                .map(slot_view),
            lm_head: self.layers.get(&LoraKey::new(0, "lm_head")).map(slot_view),
        }
    }

    /// Number of LoRA slots loaded.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// True iff no slots loaded.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl InferenceLoraLayer {
    fn alloc(
        ctx: &WgpuCtx,
        in_dim: u32,
        rank: u32,
        out_dim: u32,
        alpha: f32,
        b_is_f16: bool,
    ) -> Self {
        let scale = alpha / rank as f32;
        let device = &ctx.device;
        let a_bytes = (in_dim as usize * rank as usize * 4) as u64;
        // B is half-size in bytes when packed f16. Total element count
        // stays `out_dim * rank`; storage layout is `(out_dim * rank) / 2`
        // u32 words. Requires even rank.
        let b_elem_bytes = if b_is_f16 { 2 } else { 4 };
        let b_bytes = (out_dim as usize * rank as usize * b_elem_bytes) as u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let a = device.create_buffer(&BufferDescriptor {
            label: Some("infer.lora.A"),
            size: a_bytes,
            usage,
            mapped_at_creation: false,
        });
        let b = device.create_buffer(&BufferDescriptor {
            label: if b_is_f16 {
                Some("infer.lora.B.f16")
            } else {
                Some("infer.lora.B")
            },
            size: b_bytes,
            usage,
            mapped_at_creation: false,
        });
        let z = device.create_buffer(&BufferDescriptor {
            label: Some("infer.lora.z"),
            size: (rank as usize * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        // A/B come from the file; z is scratch. wgpu zero-fills buffers
        // at allocation, so no explicit clear here.
        Self {
            in_dim,
            rank,
            out_dim,
            scale,
            a,
            b,
            z,
            b_is_f16,
        }
    }
}

fn slot_view(l: &InferenceLoraLayer) -> LoraSlot<'_> {
    LoraSlot {
        a: &l.a,
        b: &l.b,
        z: &l.z,
        rank: l.rank,
        scale: l.scale,
        b_is_f16: l.b_is_f16,
    }
}

/// Quantize f32 → f16 and pack two consecutive elements per `u32` for
/// the f16-B fused kernel. `src.len()` must be even (kernel constraint
/// is even `rank`; `out_dim * rank` is therefore always even).
fn pack_f32_to_f16_pairs(src: &[f32]) -> Vec<u32> {
    debug_assert!(src.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(src.len() / 2);
    for pair in src.chunks_exact(2) {
        let lo = half::f16::from_f32(pair[0]).to_bits() as u32;
        let hi = half::f16::from_f32(pair[1]).to_bits() as u32;
        out.push((hi << 16) | lo);
    }
    out
}
