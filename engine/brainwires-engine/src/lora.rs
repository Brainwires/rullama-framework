//! Inference-time LoRA adapter state.
//!
//! Lean cousin of `rullama-finetune::lora::LoraState` — holds just the
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
use crate::reference::forward_chained::{LayerLoraSlots, LoraSlot};

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
    /// B matrix, `[out_dim, rank]` row-major, f32.
    pub b: Buffer,
    /// `[rank]` scratch holding `A·x` from the forward correction.
    pub z: Buffer,
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
        let mut layers: BTreeMap<LoraKey, InferenceLoraLayer> = BTreeMap::new();
        let d_model = cfg.d_model;
        for li in 0..cfg.n_layers {
            let head_dim = cfg.head_dim(li);
            let n_heads_dim = cfg.n_heads * head_dim;
            let n_kv_dim = cfg.n_kv_heads(li) * head_dim;
            let ffn_n = cfg.ffn(li);
            for proj in &target_modules {
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
                let layer = InferenceLoraLayer::alloc(&ctx, in_dim, rank, out_dim, alpha);
                layers.insert(LoraKey::new(li, proj.clone()), layer);
            }
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
            let buf = match ab {
                "A" => &layer.a,
                "B" => &layer.b,
                _ => continue,
            };
            let data = tensor.data();
            let upload_bytes: Vec<u8> = match tensor.dtype() {
                Dtype::F32 => {
                    if data.len() != buf.size() as usize {
                        return Err(RullamaError::Inference(format!(
                            "tensor {name} f32 size mismatch: file={} expected={}",
                            data.len(),
                            buf.size()
                        )));
                    }
                    data.to_vec()
                }
                Dtype::F16 => {
                    let n_elems = (buf.size() / 4) as usize;
                    if data.len() != n_elems * 2 {
                        return Err(RullamaError::Inference(format!(
                            "tensor {name} f16 size mismatch: file={} expected={}",
                            data.len(),
                            n_elems * 2
                        )));
                    }
                    let h: &[half::f16] = bytemuck::cast_slice(data);
                    let f: Vec<f32> = h.iter().map(|&x| x.to_f32()).collect();
                    bytemuck::cast_slice::<f32, u8>(&f).to_vec()
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
    fn alloc(ctx: &WgpuCtx, in_dim: u32, rank: u32, out_dim: u32, alpha: f32) -> Self {
        let scale = alpha / rank as f32;
        let device = &ctx.device;
        let a_bytes = (in_dim as usize * rank as usize * 4) as u64;
        let b_bytes = (out_dim as usize * rank as usize * 4) as u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let a = device.create_buffer(&BufferDescriptor {
            label: Some("infer.lora.A"),
            size: a_bytes,
            usage,
            mapped_at_creation: false,
        });
        let b = device.create_buffer(&BufferDescriptor {
            label: Some("infer.lora.B"),
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
    }
}
