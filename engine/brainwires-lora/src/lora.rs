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

use brainwires_engine::backend::WgpuCtx;
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

/// Trainable A and B plus per-parameter gradient and Adam state.
///
/// Forward: `y[j] += (alpha/r) · Σ_p B[j, p] · Σ_i A[p, i] · x[i]`.
/// Backward: gradients accumulate into `da` / `db` via
/// `dispatch::lora_outer_add_chained`. Adam consumes those grads to
/// update `a` and `b` via `dispatch::adam_step_chained`.
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

    /// Gradient buffer for A — accumulated across the backward sweep,
    /// reset by [`LoraLayer::zero_grads`] at the start of each step.
    pub da: Buffer,
    /// Gradient buffer for B.
    pub db: Buffer,

    /// Adam first-moment estimate for A.
    pub m_a: Buffer,
    /// Adam second-moment estimate for A.
    pub v_a: Buffer,
    /// Adam first-moment estimate for B.
    pub m_b: Buffer,
    /// Adam second-moment estimate for B.
    pub v_b: Buffer,

    /// Scratch buffer holding `z = A · x` from the most recent forward
    /// LoRA correction. Reused in the backward pass to build
    /// `dB = scale · dy ⊗ z` (the dB gradient depends on the captured
    /// `z`, not the input `x`). Size is `[rank]` f32 — trivially small.
    pub z: Buffer,

    /// 4-byte scratch holding the sum-of-squares of dA after a
    /// `sum_of_squares_chained` dispatch. Read by `clip_grad_norm` to
    /// compute the global L2 norm without reading dA itself back to
    /// host.
    pub sos_a: Buffer,
    /// 4-byte scratch holding the sum-of-squares of dB.
    pub sos_b: Buffer,
}

impl LoraLayer {
    /// Allocate fresh A and B buffers, initialize A with a deterministic
    /// small-variance pattern (seeded), B with zeros.
    pub fn new(ctx: &WgpuCtx, in_dim: u32, rank: u32, out_dim: u32, alpha: f32, seed: u64) -> Self {
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

        // Deterministic pseudo-Kaiming init for A. Tiny LCG keyed by
        // `seed`. The seed is plumbed from
        // `TrainingHyperparams::seed` via
        // `TrainingSession::new` (mixed with layer/projection index
        // for per-LoRA decorrelation). Determinism is locked in by
        // `lora_seed_determinism_same_seed_same_init`.
        let mut a_init = vec![0f32; in_dim as usize * rank as usize];
        let mut state: u64 = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let scale_init = 1.0f32 / (in_dim as f32).sqrt();
        for slot in a_init.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            // [-1, 1] uniform → scaled.
            let u = ((bits as f32) / (u32::MAX as f32 / 2.0)) - 1.0;
            *slot = u * scale_init;
        }
        ctx.queue.write_buffer(&a, 0, bytemuck::cast_slice(&a_init));

        // B starts at zero; the buffer was created with COPY_DST so just
        // queue a zero-fill (wgpu zeroes uninitialized buffers but we
        // make it explicit for clarity).
        let b_init = vec![0f32; out_dim as usize * rank as usize];
        ctx.queue.write_buffer(&b, 0, bytemuck::cast_slice(&b_init));

        // Gradient + Adam buffers, all f32, all zero-initialized. Same
        // usage flags as A/B so they can participate in copy operations
        // (gradient clearing, checkpoint readback, etc.).
        let make_zero = |label: &str, size_bytes: u64| -> Buffer {
            let buf = device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: size_bytes,
                usage,
                mapped_at_creation: false,
            });
            // Single zero-fill upload — wgpu doesn't have a buffer-clear
            // command in the public API at this version, so a small Vec
            // suffices (it's freed once write_buffer returns).
            let zeros = vec![0u8; size_bytes as usize];
            ctx.queue.write_buffer(&buf, 0, &zeros);
            buf
        };

        let da = make_zero("lora.dA", a_bytes);
        let db = make_zero("lora.dB", b_bytes);
        let m_a = make_zero("lora.mA", a_bytes);
        let v_a = make_zero("lora.vA", a_bytes);
        let m_b = make_zero("lora.mB", b_bytes);
        let v_b = make_zero("lora.vB", b_bytes);
        let z = make_zero("lora.z", (rank as usize * 4) as u64);
        let sos_a = make_zero("lora.sos_a", 4);
        let sos_b = make_zero("lora.sos_b", 4);

        Self {
            in_dim,
            rank,
            out_dim,
            scale,
            a,
            b,
            da,
            db,
            m_a,
            v_a,
            m_b,
            v_b,
            z,
            sos_a,
            sos_b,
        }
    }

    /// Number of f32 elements in A. (= `rank * in_dim`.)
    pub fn a_len(&self) -> usize {
        self.rank as usize * self.in_dim as usize
    }

    /// Number of f32 elements in B. (= `out_dim * rank`.)
    pub fn b_len(&self) -> usize {
        self.out_dim as usize * self.rank as usize
    }

    /// Clear the gradient buffers in-place. Called at the start of every
    /// training step (or every micro-batch boundary) before the backward
    /// sweep accumulates fresh gradients.
    pub fn zero_grads(&self, ctx: &WgpuCtx) {
        let zeros_a = vec![0u8; self.a_len() * 4];
        let zeros_b = vec![0u8; self.b_len() * 4];
        ctx.queue.write_buffer(&self.da, 0, &zeros_a);
        ctx.queue.write_buffer(&self.db, 0, &zeros_b);
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

    /// Borrow the GPU context — used by `load_adapter_into_state` to
    /// upload tensor bytes into the existing A/B buffers without
    /// requiring the caller to pass `ctx` separately.
    pub fn ctx(&self) -> &WgpuCtx {
        &self.ctx
    }

    /// Iterate over all `(key, layer)` pairs in deterministic
    /// (`BTreeMap`) order.
    pub fn iter(&self) -> impl Iterator<Item = (&LoraKey, &LoraLayer)> {
        self.layers.iter()
    }

    /// Clear every LoRA's gradient buffers — call at the start of each
    /// training step (or each gradient-accumulation micro-batch).
    pub fn zero_all_grads(&self) {
        for layer in self.layers.values() {
            layer.zero_grads(&self.ctx);
        }
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
            .map(|l| (l.in_dim as u64 * l.rank as u64) + (l.out_dim as u64 * l.rank as u64))
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
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read.enc"),
            });
        enc.copy_buffer_to_buffer(&layer.b, 0, &read, 0, (12 * 4 * 4) as u64);
        ctx.queue.submit(Some(enc.finish()));

        // Quick blocking readback.
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("poll");
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        let b_vals: &[f32] = bytemuck::cast_slice(&view);
        for &v in b_vals {
            assert_eq!(v, 0.0, "B should be zero-initialized");
        }
    }

    #[test]
    fn lora_layer_carries_grad_and_adam_buffers() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
        let mut state = LoraState::new(Arc::clone(&ctx));
        state
            .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
            .unwrap();
        let layer = state.get(&LoraKey::new(0, "attn_q")).unwrap();
        assert_eq!(layer.a_len(), 16);
        assert_eq!(layer.b_len(), 8);
        // Existence and size: every aux buffer matches A or B exactly.
        for buf in [&layer.da, &layer.m_a, &layer.v_a] {
            assert_eq!(buf.size() as usize, layer.a_len() * 4);
        }
        for buf in [&layer.db, &layer.m_b, &layer.v_b] {
            assert_eq!(buf.size() as usize, layer.b_len() * 4);
        }
    }

    /// Two `LoraLayer::new` calls with identical `(shape, seed)` must
    /// produce bit-identical A buffers. Determinism on this path is
    /// the contract behind `TrainingHyperparams::seed` — two training
    /// sessions with the same seed should diverge only through
    /// floating-point ordering effects in the GPU dispatches, not
    /// through the LoRA init RNG.
    #[test]
    fn lora_seed_determinism_same_seed_same_init() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
        let key = LoraKey::new(0, "attn_q");
        // A is `[rank, in_dim] = [4, 16] = 64` elements; readback those bytes.
        let mut state_a = LoraState::new(Arc::clone(&ctx));
        let mut state_b = LoraState::new(Arc::clone(&ctx));
        state_a
            .insert(key.clone(), 16, 4, 12, 8.0, 0xC0FFEE)
            .unwrap();
        state_b
            .insert(key.clone(), 16, 4, 12, 8.0, 0xC0FFEE)
            .unwrap();
        let layer_a = state_a.get(&key).unwrap();
        let layer_b = state_b.get(&key).unwrap();
        let a_vals = read_a(&ctx, &layer_a.a, layer_a.a_len());
        let b_vals = read_a(&ctx, &layer_b.a, layer_b.a_len());
        assert_eq!(a_vals, b_vals, "same seed must produce identical A init");

        // And a different seed must produce a different init (otherwise
        // the LCG would be ignoring `seed`).
        let mut state_c = LoraState::new(Arc::clone(&ctx));
        state_c
            .insert(key.clone(), 16, 4, 12, 8.0, 0xC0FFEE ^ 1)
            .unwrap();
        let layer_c = state_c.get(&key).unwrap();
        let c_vals = read_a(&ctx, &layer_c.a, layer_c.a_len());
        assert_ne!(
            a_vals, c_vals,
            "different seed must produce different A init"
        );
    }

    fn read_a(ctx: &WgpuCtx, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let bytes = (n * 4) as u64;
        let read = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("read.A"),
            size: bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read.enc"),
            });
        enc.copy_buffer_to_buffer(buf, 0, &read, 0, bytes);
        ctx.queue.submit(Some(enc.finish()));
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        read.unmap();
        v
    }

    #[test]
    fn lora_state_parameter_count() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));
        let mut state = LoraState::new(Arc::clone(&ctx));
        state
            .insert(LoraKey::new(0, "attn_q"), 16, 4, 12, 8.0, 1)
            .unwrap();
        state
            .insert(LoraKey::new(0, "attn_k"), 16, 4, 8, 8.0, 2)
            .unwrap();
        // 16*4 + 12*4 = 64 + 48 = 112 for first
        // 16*4 + 8*4  = 64 + 32 =  96 for second
        // total 208
        assert_eq!(state.parameter_count(), 208);
    }
}
