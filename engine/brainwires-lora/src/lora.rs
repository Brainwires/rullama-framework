//! Per-LoRA GPU state: A and B matrices for each wrapped projection.
//!
//! `LoraState` keys by `(layer_index, projection_name)` — e.g.
//! `(0, "attn_q")`. Each entry owns the f32 GPU buffers for that LoRA's
//! A (`[r, k]`) and B (`[n, r]`) matrices.
//!
//! Adam optimizer state (m, v) lives in [`crate::optim`] alongside `A`
//! and `B`; this module only owns the trainable parameters themselves.
//!
//! Standard LoRA init: A is Kaiming-ish small Gaussian, B is zero, so
//! the LoRA contribution at step 0 is exactly zero (the base model is
//! the starting point of training).

use std::collections::BTreeMap;
use std::sync::Arc;

use rullama::backend::WgpuCtx;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::shared::error::TrainingError;

/// Identifies one LoRA wrapper. Layer index is 0-based; `projection` is a
/// GGUF tensor stem (`attn_q`, `attn_k`, `attn_v`, `attn_o`, `ffn_gate`,
/// `ffn_up`, `ffn_down`).
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

/// Trainable A and B for one LoRA wrapper plus their shape.
///
/// Forward: `y[j] += (alpha/r) · Σ_p B[j, p] · Σ_i A[p, i] · x[i]`.
/// Backward (M0): A/B grads flow back through `dispatch::lora_*_chained`.
pub struct LoraLayer {
    /// Input dim of the wrapped projection (k).
    pub in_dim: u32,
    /// LoRA rank (r).
    pub rank: u32,
    /// Output dim of the wrapped projection (n).
    pub out_dim: u32,
    /// `α / r` — the runtime scale applied to the LoRA correction. Constant
    /// for the layer's lifetime; baked in at construction so the per-step
    /// dispatchers don't have to recompute.
    pub scale: f32,
    /// A matrix, shape `[r, in_dim]` row-major. STORAGE + COPY_DST + COPY_SRC.
    pub a: Buffer,
    /// B matrix, shape `[out_dim, r]` row-major. Same usage flags as A.
    pub b: Buffer,
}

impl LoraLayer {
    /// Allocate fresh A and B buffers, initialize A with a deterministic
    /// small-variance pattern (seeded), B with zeros.
    pub fn new(
        ctx: &WgpuCtx,
        in_dim: u32,
        rank: u32,
        out_dim: u32,
        alpha: f32,
        seed: u64,
    ) -> Self {
        let scale = alpha / rank as f32;
        let device = &ctx.device;

        let a_bytes = (in_dim as usize * rank as usize * 4) as u64;
        let b_bytes = (out_dim as usize * rank as usize * 4) as u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let a = device.create_buffer(&BufferDescriptor {
            label: Some("lora.A"),
            size: a_bytes,
            usage,
            mapped_at_creation: false,
        });
        let b = device.create_buffer(&BufferDescriptor {
            label: Some("lora.B"),
            size: b_bytes,
            usage,
            mapped_at_creation: false,
        });

        // Deterministic pseudo-Kaiming init for A. Tiny LCG keyed by `seed`
        // (good enough for M0; replace with a real PRNG when `seed` is
        // wired into `TrainingHyperparams`).
        let mut a_init = vec![0f32; in_dim as usize * rank as usize];
        let mut state: u64 = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let scale_init = 1.0f32 / (in_dim as f32).sqrt();
        for slot in a_init.iter_mut() {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            // [-1, 1] uniform → scaled.
            let u = ((bits as f32) / (u32::MAX as f32 / 2.0)) - 1.0;
            *slot = u * scale_init;
        }
        ctx.queue
            .write_buffer(&a, 0, bytemuck::cast_slice(&a_init));

        // B starts at zero; the buffer was created with COPY_DST so just
        // queue a zero-fill (wgpu zeroes uninitialized buffers but we
        // make it explicit for clarity).
        let b_init = vec![0f32; out_dim as usize * rank as usize];
        ctx.queue
            .write_buffer(&b, 0, bytemuck::cast_slice(&b_init));

        Self {
            in_dim,
            rank,
            out_dim,
            scale,
            a,
            b,
        }
    }
}

/// Collection of LoRA layers keyed by `(layer_index, projection)`.
///
/// `Arc`-wrapped so a `TrainingSession` and an inference `Model` can
/// share state — the same A/B can be consulted from a `Forward` call
/// (correction added during inference) and a `TrainingSession::step`
/// call (gradients accumulated into A/B's gradient buffers, then Adam
/// applied to update A/B in place).
pub struct LoraState {
    layers: BTreeMap<LoraKey, LoraLayer>,
    ctx: Arc<WgpuCtx>,
}

impl LoraState {
    pub fn new(ctx: Arc<WgpuCtx>) -> Self {
        Self {
            layers: BTreeMap::new(),
            ctx,
        }
    }

    /// Register one LoRA wrapper. Errors if the key is already present.
    pub fn insert(
        &mut self,
        key: LoraKey,
        in_dim: u32,
        rank: u32,
        out_dim: u32,
        alpha: f32,
        seed: u64,
    ) -> Result<(), TrainingError> {
        if self.layers.contains_key(&key) {
            return Err(TrainingError::Config(format!(
                "LoRA already registered for {:?}",
                key
            )));
        }
        let layer = LoraLayer::new(&self.ctx, in_dim, rank, out_dim, alpha, seed);
        self.layers.insert(key, layer);
        Ok(())
    }

    pub fn get(&self, key: &LoraKey) -> Option<&LoraLayer> {
        self.layers.get(key)
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Number of trainable f32 elements across all registered LoRAs.
    /// Useful for parameter-count logging and gradient-norm budgets.
    pub fn parameter_count(&self) -> u64 {
        self.layers
            .values()
            .map(|l| {
                (l.in_dim as u64 * l.rank as u64) + (l.out_dim as u64 * l.rank as u64)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test — exercises the LoRA buffer allocation path and confirms
    /// B starts at zero (the LoRA invariant that the wrapped projection's
    /// behavior is unchanged at training step 0).
    #[test]
    fn lora_state_inserts_zero_b_buffer() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
        let mut state = LoraState::new(Arc::clone(&ctx));
        let key = LoraKey::new(0, "attn_q");
        state.insert(key.clone(), 16, 4, 12, 8.0, 42).unwrap();

        let layer = state.get(&key).expect("layer");
        assert_eq!(layer.in_dim, 16);
        assert_eq!(layer.rank, 4);
        assert_eq!(layer.out_dim, 12);
        assert!((layer.scale - 2.0).abs() < 1e-6, "alpha/rank = 8/4 = 2");

        // Read B back and confirm it's all-zero.
        let read = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("read"),
            size: (12 * 4 * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read.enc"),
        });
        enc.copy_buffer_to_buffer(&layer.b, 0, &read, 0, (12 * 4 * 4) as u64);
        ctx.queue.submit(Some(enc.finish()));

        // Quick blocking readback.
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("poll");
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        let b_vals: &[f32] = bytemuck::cast_slice(&view);
        for &v in b_vals {
            assert_eq!(v, 0.0, "B should be zero-initialized");
        }
    }

    #[test]
    fn lora_state_parameter_count() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
        let mut state = LoraState::new(Arc::clone(&ctx));
        state.insert(LoraKey::new(0, "attn_q"), 16, 4, 12, 8.0, 1).unwrap();
        state.insert(LoraKey::new(0, "attn_k"), 16, 4, 8, 8.0, 2).unwrap();
        // 16*4 + 12*4 = 64 + 48 = 112 for first
        // 16*4 + 8*4  = 64 + 32 =  96 for second
        // total 208
        assert_eq!(state.parameter_count(), 208);
    }
}
