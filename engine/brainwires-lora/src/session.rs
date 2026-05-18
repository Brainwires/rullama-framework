// Training loop functions take many dims (rank, n_layers, d_model, batch,
// etc.) per call — bundling them just moves boilerplate around.
#![allow(clippy::too_many_arguments)]

//! `TrainingSession` — drives one training step end-to-end:
//! forward (with LoRA correction + activation capture) → loss →
//! backward → Adam update.
//!
//! M0 surface: single-example, NextToken cross-entropy. The session
//! resets the model's KV cache at the top of every step (each call is
//! a fresh sequence), zeroes LoRA gradient buffers, runs the prompt
//! tokens through `step_with_lora`, then runs the final token through
//! `step_capture` so the last position's activations land in
//! `TrainingScratch`. `Forward::backward_step` then walks the
//! captured graph backward, accumulating into LoRA d_a / d_b, and
//! `adam_step_chained` applies them in place.

use std::sync::Arc;

use rullama::api::Model;
use rullama::backend::dispatch::{
    AdamConfig, adam_step_chained, scale_chained, sum_of_squares_chained,
};
use rullama::reference::forward_chained::{
    BackwardScratchView, LayerCaptureBuffers, LayerLoraGrads, LayerLoraSlots, LoraGradPair,
    LoraSlot,
};

use crate::lora::{LoraKey, LoraState};
use crate::lr_schedule::LrSchedule;
use crate::scratch::TrainingScratch;
use crate::shared::config::{LoraConfig, LossMode, TrainingHyperparams};
use crate::shared::error::TrainingError;

/// Per-step progress callback fired at phase boundaries inside a
/// training step. Signature: `(phase, current, total)` where:
///
/// - `phase = "prefill"`: prompt-token forward sweep. `current` is the
///   1-based prompt-token index, `total` is the number of prefill
///   tokens (`input_ids.len() - 1`).
/// - `phase = "forward"`: final-position forward + activation capture.
///   Single tick `(total_layers, total_layers, "forward")` at the
///   end of the forward pass — coarse signal that the prefill+capture
///   phase finished and backward is about to start.
/// - `phase = "backward"`: per-layer backward sweep. `current` is the
///   1-based logical layer index (top-down), `total` is `n_layers`.
/// - `phase = "clip"`: gradient clipping just finished. `(0, 1, "clip")`.
/// - `phase = "optimizer"`: Adam step just finished. `(0, 1, "optimizer")`.
///
/// The browser worker translates this into a `trainingProgress`
/// notify the UI subscribes to; the chat-side `VisionProgress`
/// component is the visual template (see `TrainingProgress.tsx`).
pub type TrainingProgressCb<'a> = dyn Fn(&str, u32, u32) + 'a;

/// One LoRA fine-tuning session over a loaded model.
pub struct TrainingSession {
    model: Model,
    loras: LoraState,
    scratch: TrainingScratch,
    adam_cfg: AdamConfig,
    /// Loss objective for `forward_backward`. Picked at session
    /// construction from `TrainingHyperparams::loss_mode`.
    loss_mode: LossMode,
    /// Optional LR schedule. When `Some`, `optimizer_step` overrides
    /// `adam_cfg.lr` with `schedule.get_lr(step_num)` per step.
    /// `None` keeps `hp.learning_rate` constant. Cached
    /// `lr_scheduler` + `warmup_steps` from `hp` go into this when
    /// the caller opts in via `set_lr_schedule(total_steps)`.
    lr_schedule: Option<LrSchedule>,
    /// Persisted from `TrainingHyperparams` so `set_lr_schedule` can
    /// build the schedule without forcing the caller to re-pass them.
    base_lr: f64,
    warmup_steps: u64,
    lr_scheduler: crate::shared::config::LrScheduler,
    /// Global gradient-norm cap applied inside `step()` between
    /// `forward_backward` and `optimizer_step`. `0.0` disables
    /// (matches PyTorch convention). The manual driver path
    /// (`zero_grads → forward_backward × N → optimizer_step`)
    /// ignores this — call [`clip_grad_norm`] explicitly when
    /// driving accumulation by hand.
    max_grad_norm: f32,
    /// Gradient checkpointing — when true, `forward_backward`
    /// passes `recompute_captures=true` to `Forward::backward_step`,
    /// which replays each layer's forward right before its
    /// backward (using the saved `hidden_in` as the input). Memory
    /// layout is unchanged in this revision (per-layer captures
    /// still allocated); compute cost is +1× forward during
    /// backward. The flag exists so the recompute mechanism is
    /// proven end-to-end, ready for a future revision that drops
    /// per-layer non-`hidden_in` captures in favor of one shared
    /// scratch.
    gradient_checkpointing: bool,
    /// When true, `save_adapter` writes f16 instead of f32, halving
    /// adapter file size. In-memory storage on the GPU stays fp32
    /// in this revision; full bf16 kernel variants are a future
    /// optimization.
    mixed_precision: bool,
    /// 1-based step counter used by Adam's bias correction. Increments
    /// at the end of every successful `step()` call.
    step_num: u32,
}

/// Shape table the LoRA inserter walks. Pulled out so [`build_lora_state`]
/// and the probe path see the same per-layer shapes without duplicating
/// the match arms.
fn lora_projection_dims(
    cfg: &rullama::model::config::Gemma4Config,
    layer: u32,
    proj: &str,
) -> Result<(u32, u32), TrainingError> {
    let d_model = cfg.d_model;
    let head_dim = cfg.head_dim(layer);
    let n_heads_dim = cfg.n_heads * head_dim;
    let n_kv_dim = cfg.n_kv_heads(layer) * head_dim;
    let ffn_n = cfg.ffn(layer);
    Ok(match proj {
        "attn_q" => (d_model, n_heads_dim),
        "attn_k" => (d_model, n_kv_dim),
        "attn_v" => (d_model, n_kv_dim),
        "attn_o" => (n_heads_dim, d_model),
        "ffn_gate" => (d_model, ffn_n),
        "ffn_up" => (d_model, ffn_n),
        "ffn_down" => (ffn_n, d_model),
        other => {
            return Err(TrainingError::Config(format!(
                "supported LoRA targets: attn_q/k/v/o + ffn_gate/up/down, got {other}"
            )));
        }
    })
}

/// Build a fresh `LoraState` for every `(layer, projection)` pair in
/// `lora_cfg.target_modules`. Shared by [`TrainingSession::new`] and
/// the probe path so they allocate identical shapes.
fn build_lora_state(
    ctx: Arc<rullama::backend::WgpuCtx>,
    cfg: &rullama::model::config::Gemma4Config,
    lora_cfg: &LoraConfig,
    seed_base: u64,
) -> Result<LoraState, TrainingError> {
    let mut loras = LoraState::new(ctx);
    for layer in 0..cfg.n_layers {
        for proj in &lora_cfg.target_modules {
            let (in_dim, out_dim) = lora_projection_dims(cfg, layer, proj)?;
            // Deterministic seed per (layer, proj) so reruns are
            // reproducible without an extra RNG.
            let proj_idx = [
                "attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down",
            ]
            .iter()
            .position(|p| *p == proj.as_str())
            .unwrap_or(0) as u64;
            let seed = seed_base
                .wrapping_add(layer as u64 * 7919)
                .wrapping_add(proj_idx * 17);
            loras.insert(
                LoraKey::new(layer, proj.clone()),
                in_dim,
                lora_cfg.rank,
                out_dim,
                lora_cfg.alpha,
                seed,
            )?;
        }
    }
    Ok(loras)
}

/// Coarse estimate of the GPU bytes a training session would consume:
/// LoRA A/B/grad/Adam-moments per `(layer, projection)` + per-layer
/// activation captures sized for `max_seq_len`. Used by the probe so
/// the UI can show "this would need X MB" before committing the Model.
pub fn estimate_training_bytes(
    cfg: &rullama::model::config::Gemma4Config,
    lora_cfg: &LoraConfig,
    hp: &TrainingHyperparams,
) -> u64 {
    let seq = hp.max_seq_len as u64;
    let d_model = cfg.d_model as u64;
    let mut bytes: u64 = 0;
    // LoRA state — A, B, dA, dB, m_A, v_A, m_B, v_B (= 4× A + 4× B).
    for layer in 0..cfg.n_layers {
        for proj in &lora_cfg.target_modules {
            if let Ok((in_dim, out_dim)) = lora_projection_dims(cfg, layer, proj) {
                let a_elems = (in_dim as u64) * (lora_cfg.rank as u64);
                let b_elems = (out_dim as u64) * (lora_cfg.rank as u64);
                bytes += 4 * a_elems * 4 + 4 * b_elems * 4; // 4-buffer set ×4 bytes f32
            }
        }
    }
    // Activation captures per layer × n_layers. Dominated by FFN intermediates.
    // Per layer: 17 seq-sized buffers, the big ones are ffn_gate/up/act/down each
    // `ffn_inter × seq × 4`. Sum approximation per layer.
    for layer in 0..cfg.n_layers {
        let ffn = cfg.ffn(layer) as u64;
        let head_dim = cfg.head_dim(layer) as u64;
        let n_heads_dim = (cfg.n_heads as u64) * head_dim;
        let n_kv_dim = (cfg.n_kv_heads(layer) as u64) * head_dim;
        let ple_dim = if cfg.has_ple() { cfg.ple_dim as u64 } else { 0 };
        let per_layer = (3 * ffn + 2 * n_heads_dim + 2 * n_kv_dim + 6 * d_model + 3 * ple_dim) * seq * 4;
        bytes += per_layer;
    }
    bytes
}

impl TrainingSession {
    /// Allocate all training state for `model`:
    /// - One `LoraLayer` per `(layer, projection)` pair specified in
    ///   `lora_cfg.target_modules`. Each carries A, B, dA, dB, mA, vA,
    ///   mB, vB, and a `z` scratch.
    /// - One `TrainingScratch` sized for `hp.max_seq_len`.
    /// - An `AdamConfig` initialised from `hp` (lr, weight decay, etc.).
    pub fn new(
        model: Model,
        lora_cfg: LoraConfig,
        hp: TrainingHyperparams,
    ) -> Result<Self, TrainingError> {
        let cfg = model.forward().cfg().clone();
        let ctx = Arc::new(model.forward().ctx().clone());
        let max_seq_len = hp.max_seq_len as u32;
        let scratch = TrainingScratch::new(&ctx, &cfg, max_seq_len);
        let loras = build_lora_state(Arc::clone(&ctx), &cfg, &lora_cfg, hp.seed)?;

        let adam_cfg = AdamConfig {
            lr: hp.learning_rate as f32,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: hp.weight_decay as f32,
            step: 1,
        };

        Ok(Self {
            model,
            loras,
            scratch,
            adam_cfg,
            loss_mode: hp.loss_mode,
            lr_schedule: None,
            base_lr: hp.learning_rate,
            warmup_steps: hp.warmup_steps,
            lr_scheduler: hp.lr_scheduler,
            max_grad_norm: hp.max_grad_norm as f32,
            gradient_checkpointing: hp.gradient_checkpointing,
            mixed_precision: hp.mixed_precision,
            step_num: 1,
        })
    }

    /// Try the allocations a real session would do — `TrainingScratch`
    /// + `LoraState` — against a *borrowed* Model, then drop them.
    /// Returns the estimated GPU bytes on success; `Err` if any
    /// allocation fails or wgpu surfaces an OOM error during the
    /// trial. Used by the UI to refuse a `TrainingSession::new` call
    /// that would consume the Model and then fail.
    pub async fn probe(
        model: &Model,
        lora_cfg: &LoraConfig,
        hp: &TrainingHyperparams,
    ) -> Result<u64, TrainingError> {
        let cfg = model.forward().cfg().clone();
        let ctx = Arc::new(model.forward().ctx().clone());
        let estimated = estimate_training_bytes(&cfg, lora_cfg, hp);

        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let alloc_result: Result<(), TrainingError> = (|| {
            let _scratch = TrainingScratch::new(&ctx, &cfg, hp.max_seq_len as u32);
            let _loras = build_lora_state(Arc::clone(&ctx), &cfg, lora_cfg, hp.seed)?;
            drop(_scratch);
            drop(_loras);
            Ok(())
        })();
        let oom = scope.pop().await;

        match (alloc_result, oom) {
            (Ok(()), None) => Ok(estimated),
            (Err(e), _) => Err(e),
            (_, Some(err)) => Err(TrainingError::Backend(format!(
                "GPU rejected training scratch allocation (need ~{} MB): {err}",
                estimated / (1024 * 1024),
            ))),
        }
    }

    /// True iff this session was constructed with
    /// `TrainingHyperparams::gradient_checkpointing = true`.
    pub fn gradient_checkpointing(&self) -> bool {
        self.gradient_checkpointing
    }

    /// True iff this session was constructed with
    /// `TrainingHyperparams::mixed_precision = true`. Adapter
    /// serialization writes f16 in that mode.
    pub fn mixed_precision(&self) -> bool {
        self.mixed_precision
    }

    /// The loss objective this session was constructed with.
    pub fn loss_mode(&self) -> LossMode {
        self.loss_mode
    }

    /// Opt into LR scheduling for the next `total_steps` optimizer
    /// steps. The schedule respects `TrainingHyperparams::warmup_steps`
    /// and `lr_scheduler` (Constant / Linear / Cosine /
    /// CosineWarmRestarts). With no call, `optimizer_step` uses the
    /// constant `hp.learning_rate` from `new()`.
    ///
    /// `total_steps` should reflect optimizer steps, not micro-batches.
    /// Calling this resets the schedule's `total_steps` if it had been
    /// set previously.
    pub fn set_lr_schedule(&mut self, total_steps: u64) {
        self.lr_schedule = Some(LrSchedule::new(
            self.base_lr,
            self.warmup_steps,
            total_steps,
            self.lr_scheduler,
        ));
    }

    /// Drop any previously-set LR schedule and revert to constant
    /// `hp.learning_rate`.
    pub fn clear_lr_schedule(&mut self) {
        self.lr_schedule = None;
    }

    /// Current learning rate (for logging). If a schedule is set,
    /// returns `schedule.get_lr(step_num)`; otherwise the constant
    /// base rate. `step_num` is 1-based — call this before
    /// `optimizer_step` to get the lr that will be applied at the
    /// next step, or after to get the lr that was applied.
    pub fn current_lr(&self) -> f64 {
        match &self.lr_schedule {
            Some(s) => s.get_lr(self.step_num as u64),
            None => self.base_lr,
        }
    }

    /// Zero every LoRA's `dA` / `dB` gradient buffers. Call at the
    /// start of a gradient-accumulation cycle (or `step()` does this
    /// for you for the single-example case).
    pub fn zero_grads(&self) {
        self.loras.zero_all_grads();
    }

    /// Compute the L2 norm `||g||` over every LoRA's `dA` and `dB`
    /// gradient buffer; if `||g|| > max_norm` scale every gradient
    /// in-place by `max_norm / ||g||`. Mirrors PyTorch's
    /// `clip_grad_norm_` semantics: norm is over the concatenation
    /// of all gradient tensors, not per-tensor.
    ///
    /// Returns the pre-clip L2 norm, useful for logging /
    /// `RULLAMA_DEBUG_GRAD_NORM=1`. A non-positive `max_norm` is a
    /// no-op (clipping disabled) — caller forwards
    /// `hp.max_grad_norm` whose `0.0` default opts out.
    ///
    /// Sequencing: call after the gradient accumulation cycle
    /// (`forward_backward(...) × N`) and before `optimizer_step()`.
    pub async fn clip_grad_norm(&mut self, max_norm: f32) -> Result<f32, TrainingError> {
        if max_norm <= 0.0 || !max_norm.is_finite() {
            return Ok(0.0);
        }
        let ctx = self.model.forward().ctx().clone();
        let pipes = self.model.forward().pipes().clone();

        // Pass 1: per-LoRA sum-of-squares into each layer's sos_a / sos_b.
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("train.sos"),
            });
        for (_key, layer) in self.loras.iter() {
            sum_of_squares_chained(
                &ctx,
                &pipes,
                &mut enc,
                &layer.da,
                &layer.sos_a,
                layer.a_len(),
                1.0,
            );
            sum_of_squares_chained(
                &ctx,
                &pipes,
                &mut enc,
                &layer.db,
                &layer.sos_b,
                layer.b_len(),
                1.0,
            );
        }
        // Gather all sos scalars into one readback buffer (4 bytes per
        // grad buffer, two per LoRA).
        let n_loras = self.loras.len();
        let read_bytes = (n_loras * 2 * 4) as u64;
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("train.sos.read"),
            size: read_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (i, (_key, layer)) in self.loras.iter().enumerate() {
            let off_a = (i * 2) as u64 * 4;
            let off_b = off_a + 4;
            enc.copy_buffer_to_buffer(&layer.sos_a, 0, &read_buf, off_a, 4);
            enc.copy_buffer_to_buffer(&layer.sos_b, 0, &read_buf, off_b, 4);
        }
        ctx.queue.submit(Some(enc.finish()));

        let slice = read_buf.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| TrainingError::Backend(format!("poll: {e:?}")))?;
        rx.await
            .map_err(|e| TrainingError::Backend(format!("rx: {e:?}")))?
            .map_err(|e| TrainingError::Backend(format!("map: {e:?}")))?;
        let view = slice.get_mapped_range();
        let sos_vals: &[f32] = bytemuck::cast_slice(&view);
        let total_sos: f64 = sos_vals.iter().map(|&x| x as f64).sum();
        drop(view);
        read_buf.unmap();

        if !total_sos.is_finite() {
            // Numerical disaster — let the caller see NaN/Inf
            // (Adam will already pollute the params next step). Return
            // the bogus norm; clipping won't help.
            return Ok(total_sos as f32);
        }
        let l2 = total_sos.sqrt() as f32;
        if l2 <= max_norm {
            return Ok(l2);
        }
        let s = max_norm / l2;

        // Pass 2: scale every grad buffer by `s` in-place.
        let mut enc2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("train.gradclip.scale"),
            });
        for (_key, layer) in self.loras.iter() {
            scale_chained(&ctx, &pipes, &mut enc2, &layer.da, layer.a_len(), s);
            scale_chained(&ctx, &pipes, &mut enc2, &layer.db, layer.b_len(), s);
        }
        ctx.queue.submit(Some(enc2.finish()));
        Ok(l2)
    }

    /// Apply Adam to every registered LoRA's `(A, m_a, v_a)` and
    /// `(B, m_b, v_b)`. Bumps the internal step counter (1-based,
    /// drives the bias-correction). Call once at the end of each
    /// gradient-accumulation cycle.
    pub fn optimizer_step(&mut self) {
        let lr = match &self.lr_schedule {
            Some(s) => s.get_lr(self.step_num as u64) as f32,
            None => self.adam_cfg.lr,
        };
        let adam = AdamConfig {
            step: self.step_num,
            lr,
            ..self.adam_cfg
        };
        let ctx = self.model.forward().ctx().clone();
        let pipes = self.model.forward().pipes().clone();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("train.adam"),
            });
        for (_key, layer) in self.loras.iter() {
            adam_step_chained(
                &ctx,
                &pipes,
                &mut enc,
                &layer.da,
                &layer.a,
                &layer.m_a,
                &layer.v_a,
                layer.a_len(),
                adam,
            );
            adam_step_chained(
                &ctx,
                &pipes,
                &mut enc,
                &layer.db,
                &layer.b,
                &layer.m_b,
                &layer.v_b,
                layer.b_len(),
                adam,
            );
        }
        ctx.queue.submit(Some(enc.finish()));
        self.step_num = self.step_num.saturating_add(1);
    }

    /// Run a forward + backward sweep that **accumulates** gradients
    /// into each LoRA's `dA` / `dB` buffers without applying Adam or
    /// zeroing first. The driver of a gradient-accumulation loop
    /// calls this N times after a single `zero_grads()`, then one
    /// `optimizer_step()` to consume the summed gradients.
    pub async fn forward_backward(
        &mut self,
        input_ids: &[u32],
        target_id: u32,
    ) -> Result<f32, TrainingError> {
        self.forward_backward_with_progress(input_ids, target_id, None).await
    }

    /// Variant of [`forward_backward`] that fires `progress_cb` at
    /// phase boundaries: `"prefill"` per prompt token, `"forward"`
    /// once at end of capture step, `"backward"` per layer (top-down),
    /// `"clip"` once after gradient clip. The optimizer step is
    /// reported by [`step_with_progress`]; manual gradient
    /// accumulation drivers fire `"optimizer"` themselves around
    /// their `optimizer_step()` call.
    pub async fn forward_backward_with_progress<'cb>(
        &mut self,
        input_ids: &[u32],
        target_id: u32,
        progress_cb: Option<&'cb TrainingProgressCb<'cb>>,
    ) -> Result<f32, TrainingError> {
        if input_ids.is_empty() {
            return Err(TrainingError::Config(
                "forward_backward: input_ids must be non-empty".into(),
            ));
        }
        let n_layers = self.model.forward().cfg().n_layers as usize;

        // Fresh KV cache (each call is a fresh sequence). LoRA grad
        // buffers are *not* touched — caller zeros them via
        // `zero_grads()` at the start of the accumulation cycle.
        self.model.forward_mut().reset();

        // Build per-layer LoRA slot views (forward correction inputs).
        // Stays alive for the whole step — borrows immutably from
        // `self.loras` so we can also mutably borrow `self.model`.
        let lora_slots: Vec<LayerLoraSlots> = (0..n_layers)
            .map(|li| LayerLoraSlots {
                q: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_q"))
                    .map(slot_view),
                k: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_k"))
                    .map(slot_view),
                v: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_v"))
                    .map(slot_view),
                o: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_o"))
                    .map(slot_view),
                ffn_gate: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_gate"))
                    .map(slot_view),
                ffn_up: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_up"))
                    .map(slot_view),
                ffn_down: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_down"))
                    .map(slot_view),
            })
            .collect();

        // Per-layer capture views into TrainingScratch.layers.
        let capture: Vec<LayerCaptureBuffers> = self
            .scratch
            .layers
            .iter()
            .map(|l| LayerCaptureBuffers {
                hidden_in: &l.hidden_in,
                norm_x_attn: &l.norm_x_attn,
                q_pre_norm: &l.q_pre_norm,
                q_post_rope: &l.q_post_rope,
                k_pre_norm: &l.k_pre_norm,
                v_pre_norm: &l.v_pre_norm,
                attn_out: &l.attn_out,
                attn_proj: &l.attn_proj,
                pre_ffn_rms: &l.pre_ffn_rms,
                norm_x_ffn: &l.norm_x_ffn,
                ffn_gate: &l.ffn_gate,
                ffn_up: &l.ffn_up,
                ffn_act: &l.ffn_act,
                ffn_out: &l.ffn_out,
                ple_state: &l.ple_state,
                ple_act: &l.ple_act,
                ple_proj: &l.ple_proj,
            })
            .collect();

        // Prefill prompt — capture the seq-shaped per-position
        // activations (norm_x_attn, k_pre_norm, v_pre_norm) into the
        // shared `capture` buffers at each position's offset. The
        // 11 non-seq captures get overwritten per position; only
        // the final-position values stick (which is what the regular
        // backward chain needs).
        let prefill_total = (input_ids.len().saturating_sub(1)) as u32;
        for (i, &tok) in input_ids[..input_ids.len() - 1].iter().enumerate() {
            self.model
                .forward_mut()
                .step_with_lora_seqcap(tok, &lora_slots, &capture)
                .await
                .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
            if let Some(cb) = progress_cb {
                cb("prefill", (i + 1) as u32, prefill_total);
            }
        }
        // Final position — capture activations + compute logits.
        let final_tok = *input_ids.last().unwrap();
        let _logits = self
            .model
            .forward_mut()
            .step_capture(final_tok, &capture, Some(&lora_slots))
            .await
            .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
        if let Some(cb) = progress_cb {
            cb("forward", n_layers as u32, n_layers as u32);
        }

        // Backward.
        let grads: Vec<LayerLoraGrads> = (0..n_layers)
            .map(|li| LayerLoraGrads {
                q: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_q"))
                    .map(grad_view),
                k: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_k"))
                    .map(grad_view),
                v: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_v"))
                    .map(grad_view),
                o: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_o"))
                    .map(grad_view),
                ffn_gate: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_gate"))
                    .map(grad_view),
                ffn_up: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_up"))
                    .map(grad_view),
                ffn_down: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_down"))
                    .map(grad_view),
            })
            .collect();
        let s = &self.scratch;
        // `d_attn_out` aliases `d_q_pre_rope`: the latter is never
        // written by `backward_layer` (`rope_neox_backward` modifies
        // `d_q` in place), so the buffer is idle and the right size
        // (`[n_heads · head_dim_max]`) — perfect overflow scratch.
        let scratch_view = BackwardScratchView {
            d_logits: &s.d_logits,
            loss: &s.loss,
            d_hidden_final: &s.d_hidden_final,
            d_hidden: &s.d_hidden,
            d_hidden_tmp: &s.d_hidden_tmp,
            d_hidden_tmp2: &s.d_hidden_tmp2,
            attn_probs: &s.attn_probs,
            attn_d_scores: &s.attn_d_scores,
            d_attn_out: &s.d_q_pre_rope,
            d_q: &s.d_q,
            d_k_hist: &s.d_k_hist,
            d_v_hist: &s.d_v_hist,
            d_q_pre_rope: &s.d_q_pre_rope,
            d_k_pre_rope: &s.d_k_pre_rope,
            d_q_pre_norm: &s.d_q_pre_norm,
            d_k_pre_norm: &s.d_k_pre_norm,
            d_v_pre_norm: &s.d_v_pre_norm,
            d_ffn_a: &s.d_ffn_a,
            d_ffn_b: &s.d_ffn_b,
            d_ffn_c: &s.d_ffn_c,
            d_ple_state: &s.d_ple_state,
            d_ple_act: &s.d_ple_act,
            d_ple_up_discard: &s.d_ple_up_discard,
            ple_per_layer_tmp: &s.ple_per_layer_tmp,
            norm_x_attn_window: &s.norm_x_attn_window,
            k_pre_norm_window: &s.k_pre_norm_window,
            v_pre_norm_window: &s.v_pre_norm_window,
            hidden_in_window: &s.hidden_in_window,
            q_pre_norm_window: &s.q_pre_norm_window,
            q_post_rope_window: &s.q_post_rope_window,
            attn_out_window: &s.attn_out_window,
            attn_proj_window: &s.attn_proj_window,
            pre_ffn_rms_window: &s.pre_ffn_rms_window,
            norm_x_ffn_window: &s.norm_x_ffn_window,
            ffn_gate_window: &s.ffn_gate_window,
            ffn_up_window: &s.ffn_up_window,
            ffn_act_window: &s.ffn_act_window,
            ffn_out_window: &s.ffn_out_window,
            ple_state_window: &s.ple_state_window,
            ple_act_window: &s.ple_act_window,
            ple_proj_window: &s.ple_proj_window,
        };
        let history_len = input_ids.len() as u32;
        let pos = (input_ids.len() - 1) as u32;
        // Forward the progress callback into the backward layer walk
        // — it'll fire `"backward"` per logical layer (top-down).
        let loss = self
            .model
            .forward_mut()
            .backward_step_with_progress(
                target_id,
                &capture,
                &lora_slots,
                &grads,
                &scratch_view,
                history_len,
                pos,
                self.gradient_checkpointing,
                progress_cb,
            )
            .await
            .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;

        // Debug readback of LoRA gradient norms — set
        // `RULLAMA_DEBUG_GRADS=1` to print each layer's dA/dB max-abs +
        // NaN count. Used to localise NaN sources in the backward path.
        // Native-only: env vars aren't available in wasm32 + the browser
        // path uses console-level logging through the worker instead.
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("RULLAMA_DEBUG_GRADS").is_ok() {
            self.debug_grad_norms().await;
        }

        Ok(loss)
    }

    /// Run one training step on a single example:
    ///   `loss = CE(forward(input_ids), target_id)`.
    ///
    /// Resets KV cache, **zeroes gradients**, runs forward with LoRA
    /// correction, captures activations at the final position, runs
    /// backward, applies Adam. Returns the scalar cross-entropy loss.
    ///
    /// For gradient accumulation across multiple micro-batches, call
    /// `zero_grads()` once, `forward_backward()` for each
    /// micro-batch, then `optimizer_step()` once.
    pub async fn step(&mut self, input_ids: &[u32], target_id: u32) -> Result<f32, TrainingError> {
        self.step_with_progress(input_ids, target_id, None).await
    }

    /// Variant of [`step`] that fires `progress_cb` at phase
    /// boundaries — see [`TrainingProgressCb`] for the surface. Used
    /// by the wasm-bindgen `TrainingSession::step` JS entry point so
    /// the PWA can render a VisionProgress-style status strip.
    pub async fn step_with_progress<'cb>(
        &mut self,
        input_ids: &[u32],
        target_id: u32,
        progress_cb: Option<&'cb TrainingProgressCb<'cb>>,
    ) -> Result<f32, TrainingError> {
        self.zero_grads();
        let loss = self.forward_backward_with_progress(input_ids, target_id, progress_cb).await?;
        if self.max_grad_norm > 0.0 {
            self.clip_grad_norm(self.max_grad_norm).await?;
            if let Some(cb) = progress_cb {
                cb("clip", 0, 1);
            }
        }
        self.optimizer_step();
        if let Some(cb) = progress_cb {
            cb("optimizer", 0, 1);
        }
        Ok(loss)
    }

    /// PerPosition forward+backward: single-forward variant.
    ///
    /// 1. Reset KV. Run **one** forward sweep through the full
    ///    `input_ids` capturing per-position activations into the
    ///    seq-sized `LayerActivations` (every layer's captures are
    ///    written at offset `pos·per_position_size`).
    /// 2. Save each token's pre-final-norm `self.hidden` into
    ///    `scratch.seq_pre_final_norm` at offset `pos·d_model`.
    /// 3. For each position `p` with `targets[p] != u32::MAX`:
    ///    a. Point `self.hidden` at the saved `seq_pre_final_norm[p]`.
    ///    b. Run final rmsnorm + tiled output projection to fill `self.logits` with position-`p`'s vocab distribution.
    ///    c. Call `backward_step` at `pos=p`, `history_len=p+1`, `target_id=targets[p]`. Pre-copies window slices from offset `p·size` for all 14 captures and walks the layer chain (including the per-history K/V LoRA loop over positions `0..p`).
    /// 4. Return the mean cross-entropy across active positions.
    ///
    /// Forward cost: `O(N)` layer-ops (one sweep). Backward cost:
    /// `O(C·N)` layer-ops (`C` active positions × `N` layers per
    /// position). Total: `O(N + C·N)` vs. the old loop variant's
    /// `O(C·N/2 + C·N) = O(C·N)`. ~`C/2`× forward speedup.
    ///
    /// Like [`forward_backward`], does NOT zero gradients or step
    /// Adam — caller drives the accumulation cycle.
    pub async fn forward_backward_per_position(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
    ) -> Result<f32, TrainingError> {
        self.forward_backward_per_position_with_progress(input_ids, targets, None).await
    }

    /// Variant of [`forward_backward_per_position`] that fires
    /// `progress_cb` at phase boundaries — same semantics as
    /// [`forward_backward_with_progress`] but adapted to the
    /// PerPosition loop (single forward sweep, then C backward
    /// sweeps over the active positions).
    pub async fn forward_backward_per_position_with_progress<'cb>(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
        progress_cb: Option<&'cb TrainingProgressCb<'cb>>,
    ) -> Result<f32, TrainingError> {
        if targets.len() != input_ids.len() {
            return Err(TrainingError::Config(format!(
                "forward_backward_per_position: targets.len()={} must equal input_ids.len()={}",
                targets.len(),
                input_ids.len()
            )));
        }
        let n_active = targets.iter().filter(|&&t| t != u32::MAX).count();
        if n_active == 0 {
            return Ok(0.0);
        }
        let n_layers = self.model.forward().cfg().n_layers as usize;
        let d_model = self.model.forward().cfg().d_model as u64;
        let d_model_bytes = d_model * 4;

        self.model.forward_mut().reset();

        let lora_slots: Vec<LayerLoraSlots> = (0..n_layers)
            .map(|li| LayerLoraSlots {
                q: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_q"))
                    .map(slot_view),
                k: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_k"))
                    .map(slot_view),
                v: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_v"))
                    .map(slot_view),
                o: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_o"))
                    .map(slot_view),
                ffn_gate: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_gate"))
                    .map(slot_view),
                ffn_up: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_up"))
                    .map(slot_view),
                ffn_down: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_down"))
                    .map(slot_view),
            })
            .collect();
        let capture: Vec<LayerCaptureBuffers> = self
            .scratch
            .layers
            .iter()
            .map(|l| LayerCaptureBuffers {
                hidden_in: &l.hidden_in,
                norm_x_attn: &l.norm_x_attn,
                q_pre_norm: &l.q_pre_norm,
                q_post_rope: &l.q_post_rope,
                k_pre_norm: &l.k_pre_norm,
                v_pre_norm: &l.v_pre_norm,
                attn_out: &l.attn_out,
                attn_proj: &l.attn_proj,
                pre_ffn_rms: &l.pre_ffn_rms,
                norm_x_ffn: &l.norm_x_ffn,
                ffn_gate: &l.ffn_gate,
                ffn_up: &l.ffn_up,
                ffn_act: &l.ffn_act,
                ffn_out: &l.ffn_out,
                ple_state: &l.ple_state,
                ple_act: &l.ple_act,
                ple_proj: &l.ple_proj,
            })
            .collect();

        // 1. Single forward sweep — every token's encode_layer
        //    writes seq-position-shifted captures, AND we snapshot
        //    `self.hidden` (= pre-final-norm) into the seq buffer
        //    right after the call returns.
        let ctx = self.model.forward().ctx().clone();
        let prefill_total = input_ids.len() as u32;
        for (idx, &tok) in input_ids.iter().enumerate() {
            self.model
                .forward_mut()
                .step_with_lora_seqcap(tok, &lora_slots, &capture)
                .await
                .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
            let pos_just_finished = (self.model.forward().pos() as u64).saturating_sub(1);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("train.save_pre_final_norm"),
                });
            enc.copy_buffer_to_buffer(
                self.model.forward().hidden_buffer(),
                0,
                &self.scratch.seq_pre_final_norm,
                pos_just_finished * d_model_bytes,
                d_model_bytes,
            );
            ctx.queue.submit(Some(enc.finish()));
            if let Some(cb) = progress_cb {
                cb("prefill", (idx + 1) as u32, prefill_total);
            }
        }

        // 2. Build grad views + scratch view (mirrors `forward_backward`).
        let grads: Vec<LayerLoraGrads> = (0..n_layers)
            .map(|li| LayerLoraGrads {
                q: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_q"))
                    .map(grad_view),
                k: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_k"))
                    .map(grad_view),
                v: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_v"))
                    .map(grad_view),
                o: self
                    .loras
                    .get(&LoraKey::new(li as u32, "attn_o"))
                    .map(grad_view),
                ffn_gate: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_gate"))
                    .map(grad_view),
                ffn_up: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_up"))
                    .map(grad_view),
                ffn_down: self
                    .loras
                    .get(&LoraKey::new(li as u32, "ffn_down"))
                    .map(grad_view),
            })
            .collect();
        let s = &self.scratch;
        let scratch_view = BackwardScratchView {
            d_logits: &s.d_logits,
            loss: &s.loss,
            d_hidden_final: &s.d_hidden_final,
            d_hidden: &s.d_hidden,
            d_hidden_tmp: &s.d_hidden_tmp,
            d_hidden_tmp2: &s.d_hidden_tmp2,
            attn_probs: &s.attn_probs,
            attn_d_scores: &s.attn_d_scores,
            d_attn_out: &s.d_q_pre_rope,
            d_q: &s.d_q,
            d_k_hist: &s.d_k_hist,
            d_v_hist: &s.d_v_hist,
            d_q_pre_rope: &s.d_q_pre_rope,
            d_k_pre_rope: &s.d_k_pre_rope,
            d_q_pre_norm: &s.d_q_pre_norm,
            d_k_pre_norm: &s.d_k_pre_norm,
            d_v_pre_norm: &s.d_v_pre_norm,
            d_ffn_a: &s.d_ffn_a,
            d_ffn_b: &s.d_ffn_b,
            d_ffn_c: &s.d_ffn_c,
            d_ple_state: &s.d_ple_state,
            d_ple_act: &s.d_ple_act,
            d_ple_up_discard: &s.d_ple_up_discard,
            ple_per_layer_tmp: &s.ple_per_layer_tmp,
            norm_x_attn_window: &s.norm_x_attn_window,
            k_pre_norm_window: &s.k_pre_norm_window,
            v_pre_norm_window: &s.v_pre_norm_window,
            hidden_in_window: &s.hidden_in_window,
            q_pre_norm_window: &s.q_pre_norm_window,
            q_post_rope_window: &s.q_post_rope_window,
            attn_out_window: &s.attn_out_window,
            attn_proj_window: &s.attn_proj_window,
            pre_ffn_rms_window: &s.pre_ffn_rms_window,
            norm_x_ffn_window: &s.norm_x_ffn_window,
            ffn_gate_window: &s.ffn_gate_window,
            ffn_up_window: &s.ffn_up_window,
            ffn_act_window: &s.ffn_act_window,
            ffn_out_window: &s.ffn_out_window,
            ple_state_window: &s.ple_state_window,
            ple_act_window: &s.ple_act_window,
            ple_proj_window: &s.ple_proj_window,
        };

        // 3. Per active position: point hidden at that position's
        //    pre-final-norm slice, run final norm + output proj,
        //    then backward_step.
        let mut total_loss = 0.0f32;
        for (p, &target_id) in targets.iter().enumerate() {
            if target_id == u32::MAX {
                continue;
            }
            self.model
                .forward()
                .set_hidden_from(&self.scratch.seq_pre_final_norm, (p as u64) * d_model_bytes);
            self.model
                .forward_mut()
                .run_final_norm_and_output_proj_only()
                .await
                .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
            let loss = self
                .model
                .forward_mut()
                .backward_step_with_progress(
                    target_id,
                    &capture,
                    &lora_slots,
                    &grads,
                    &scratch_view,
                    (p + 1) as u32,
                    p as u32,
                    false,
                    progress_cb,
                )
                .await
                .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
            total_loss += loss;
        }

        Ok(total_loss / n_active as f32)
    }

    /// One PerPosition training step: zeroes gradients, runs
    /// [`forward_backward_per_position`], applies Adam. Returns the
    /// mean cross-entropy across active positions.
    pub async fn step_per_position(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
    ) -> Result<f32, TrainingError> {
        self.step_per_position_with_progress(input_ids, targets, None).await
    }

    /// Variant of [`step_per_position`] with the same progress-callback
    /// surface as [`step_with_progress`].
    pub async fn step_per_position_with_progress<'cb>(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
        progress_cb: Option<&'cb TrainingProgressCb<'cb>>,
    ) -> Result<f32, TrainingError> {
        self.zero_grads();
        let loss = self
            .forward_backward_per_position_with_progress(input_ids, targets, progress_cb)
            .await?;
        if self.max_grad_norm > 0.0 {
            self.clip_grad_norm(self.max_grad_norm).await?;
            if let Some(cb) = progress_cb {
                cb("clip", 0, 1);
            }
        }
        self.optimizer_step();
        if let Some(cb) = progress_cb {
            cb("optimizer", 0, 1);
        }
        Ok(loss)
    }

    /// Number of LoRA parameters currently being trained.
    pub fn parameter_count(&self) -> u64 {
        self.loras.parameter_count()
    }

    /// Immutable handle on the wrapped model — for token encoding /
    /// inference between training steps.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// 1-based step counter. Increments at the end of every successful
    /// `step()` / `step_per_position()` call.
    pub fn step_num(&self) -> u32 {
        self.step_num
    }

    /// Consume the session and hand the wrapped `Model` back to the
    /// caller. Used by the browser path so chat can resume against the
    /// same `Model` handle after training ends, without re-loading the
    /// (multi-GB) weights from OPFS.
    pub fn into_model(self) -> Model {
        self.model
    }

    /// Cooperatively cancel any in-flight `step` / `forward_backward` /
    /// `step_per_position` call. The in-progress forward + backward
    /// layer walks check the flag between per-layer encoder submits
    /// (~300 ms - 1 s latency on browser); the awaited `step` resolves
    /// with `TrainingError::Backend("cancelled by caller")` so the JS
    /// driver loop exits cleanly. Safe to call when no step is in
    /// flight — the flag is reset at the top of each layer walk.
    pub fn cancel(&self) {
        self.model.forward().cancel();
    }

    /// Serialize the current LoRA A/B matrices into a safetensors byte
    /// buffer. Caller decides where the bytes go — disk (native) or
    /// OPFS / download blob (browser).
    ///
    /// Tensor naming: `lora.blk.{layer}.{projection}.{A|B}`. Stored as
    /// f32 row-major (or f16 if `mixed_precision` was set on the session).
    /// Metadata sidecar carries rank/alpha/target_modules so a loader
    /// can rebuild the `LoraState` without external context.
    ///
    /// Note: the `m_a/v_a/m_b/v_b` Adam state and gradient accumulators
    /// are *not* persisted — only the trainable parameters. To resume
    /// training, instantiate a fresh `TrainingSession` and call
    /// `load_adapter_into_state*` to seed A/B; Adam state restarts at
    /// step 1 (acceptable for downstream fine-tunes; matches what HF
    /// Transformers does).
    pub async fn save_adapter_to_bytes(&self) -> Result<Vec<u8>, TrainingError> {
        use safetensors::tensor::{Dtype, TensorView};
        let ctx = self.model.forward().ctx().clone();
        let f16_mode = self.mixed_precision;
        let dtype = if f16_mode { Dtype::F16 } else { Dtype::F32 };

        // 1. Pull every LoRA A/B back to host as bytes — f32 always,
        //    converted to f16 if `mixed_precision` is set.
        let mut tensors: Vec<(String, Vec<u32>, Vec<u8>)> = Vec::new();
        for (key, layer) in self.loras.iter() {
            let a_vals = read_buf_f32(&ctx, &layer.a, layer.a_len()).await;
            let b_vals = read_buf_f32(&ctx, &layer.b, layer.b_len()).await;
            let (a_bytes, b_bytes) = if f16_mode {
                let a_h: Vec<half::f16> = a_vals.iter().map(|&x| half::f16::from_f32(x)).collect();
                let b_h: Vec<half::f16> = b_vals.iter().map(|&x| half::f16::from_f32(x)).collect();
                (
                    bytemuck::cast_slice::<half::f16, u8>(&a_h).to_vec(),
                    bytemuck::cast_slice::<half::f16, u8>(&b_h).to_vec(),
                )
            } else {
                (
                    bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec(),
                    bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec(),
                )
            };
            let a_name = format!("lora.blk.{}.{}.A", key.layer, key.projection);
            let b_name = format!("lora.blk.{}.{}.B", key.layer, key.projection);
            // A shape: [rank, in_dim]; B shape: [out_dim, rank].
            let a_shape = vec![layer.rank, layer.in_dim];
            let b_shape = vec![layer.out_dim, layer.rank];
            tensors.push((a_name, a_shape, a_bytes));
            tensors.push((b_name, b_shape, b_bytes));
        }

        // 2. Build TensorViews (each borrows from the owned byte vec).
        let mut views: std::collections::HashMap<&str, TensorView<'_>> =
            std::collections::HashMap::new();
        for (name, shape, bytes) in &tensors {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let view = TensorView::new(dtype, shape_usize, bytes)
                .map_err(|e| TrainingError::Backend(format!("safetensors view: {e}")))?;
            views.insert(name.as_str(), view);
        }

        // 3. Metadata sidecar.
        let any = self.loras.iter().next();
        let (rank, alpha) = match any {
            Some((_, layer)) => (layer.rank, layer.scale * layer.rank as f32),
            None => (0u32, 0.0f32),
        };
        let mut target_modules: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for (key, _) in self.loras.iter() {
            target_modules.insert(key.projection.clone());
        }
        let target_modules_list: Vec<String> = target_modules.into_iter().collect();
        let metadata: std::collections::HashMap<String, String> = [
            ("format".to_string(), "rullama-lora-v0".to_string()),
            ("rank".to_string(), rank.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("target_modules".to_string(), target_modules_list.join(",")),
            (
                "dtype".to_string(),
                if f16_mode { "f16" } else { "f32" }.to_string(),
            ),
        ]
        .into_iter()
        .collect();

        // 4. Serialize to bytes.
        safetensors::serialize(&views, &Some(metadata))
            .map_err(|e| TrainingError::Backend(format!("safetensors serialize: {e}")))
    }

    /// Native-only convenience wrapper: serialize the adapter and write
    /// it to disk. Browser code calls [`Self::save_adapter_to_bytes`]
    /// and writes to OPFS via `FileSystemSyncAccessHandle` directly.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save_adapter(&self, path: &std::path::Path) -> Result<(), TrainingError> {
        let bytes = self.save_adapter_to_bytes().await?;
        std::fs::write(path, bytes)
            .map_err(|e| TrainingError::Backend(format!("write adapter: {e}")))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn debug_grad_norms(&self) {
        let ctx = self.model.forward().ctx().clone();
        for (key, layer) in self.loras.iter() {
            let da_vals = read_buf_f32(&ctx, &layer.da, layer.a_len()).await;
            let db_vals = read_buf_f32(&ctx, &layer.db, layer.b_len()).await;
            let (da_max, da_nans) = stats(&da_vals);
            let (db_max, db_nans) = stats(&db_vals);
            eprintln!(
                "[grad] layer={} proj={} dA max={da_max:.3e} nan={da_nans}  dB max={db_max:.3e} nan={db_nans}",
                key.layer, key.projection
            );
        }
    }
}

/// Load LoRA A/B tensors from a safetensors file into the given
/// `LoraState`. Native-only convenience wrapper around
/// [`load_adapter_into_state_from_bytes`].
#[cfg(not(target_arch = "wasm32"))]
pub fn load_adapter_into_state(
    state: &mut crate::lora::LoraState,
    path: &std::path::Path,
) -> Result<usize, TrainingError> {
    let bytes = std::fs::read(path)
        .map_err(|e| TrainingError::Backend(format!("read {}: {e}", path.display())))?;
    load_adapter_into_state_from_bytes(state, &bytes)
}

/// Load LoRA A/B tensors from a safetensors byte buffer into the given
/// `LoraState`. Each named tensor in the buffer
/// (`lora.blk.{layer}.{proj}.{A|B}`) is written to the matching
/// `LoraLayer` buffer.
///
/// The `LoraState` must already have all the expected LoRA slots
/// registered with matching shapes — i.e. caller built it via
/// `TrainingSession::new`-style construction. Tensors in the file
/// that don't have a matching slot are skipped (with a tracing
/// warning); slots with no matching tensor are left at whatever the
/// caller's initialiser produced.
pub fn load_adapter_into_state_from_bytes(
    state: &mut crate::lora::LoraState,
    bytes: &[u8],
) -> Result<usize, TrainingError> {
    use safetensors::SafeTensors;

    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TrainingError::Backend(format!("safetensors parse: {e}")))?;

    let mut loaded = 0usize;
    let ctx = state.ctx();
    for (name, tensor) in st.tensors() {
        if !name.starts_with("lora.blk.") {
            continue;
        }
        let suffix = &name["lora.blk.".len()..];
        let (layer_str, rest) = match suffix.split_once('.') {
            Some(p) => p,
            None => continue,
        };
        let layer: u32 = match layer_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let (projection, ab) = match rest.rsplit_once('.') {
            Some(p) => p,
            None => continue,
        };
        let key = crate::lora::LoraKey::new(layer, projection.to_string());
        let layer_state = match state.get(&key) {
            Some(l) => l,
            None => {
                tracing::warn!("adapter has tensor {name} but no matching LoRA slot");
                continue;
            }
        };
        let buf = match ab {
            "A" => &layer_state.a,
            "B" => &layer_state.b,
            _ => continue,
        };
        let n_elems = (buf.size() / 4) as usize;
        let data = tensor.data();
        let dtype = tensor.dtype();
        let upload_bytes: Vec<u8> = match dtype {
            safetensors::tensor::Dtype::F32 => {
                if data.len() != buf.size() as usize {
                    return Err(TrainingError::Backend(format!(
                        "tensor {name} f32 size mismatch: file={} expected={}",
                        data.len(),
                        buf.size()
                    )));
                }
                data.to_vec()
            }
            safetensors::tensor::Dtype::F16 => {
                // 2 bytes per element in the file → f32 on the GPU.
                if data.len() != n_elems * 2 {
                    return Err(TrainingError::Backend(format!(
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
                return Err(TrainingError::Backend(format!(
                    "tensor {name} unsupported dtype {other:?} (expected F32 or F16)"
                )));
            }
        };
        ctx.queue.write_buffer(buf, 0, &upload_bytes);
        loaded += 1;
    }
    Ok(loaded)
}

#[cfg(not(target_arch = "wasm32"))]
fn stats(v: &[f32]) -> (f32, usize) {
    let mut max_abs = 0.0f32;
    let mut nans = 0usize;
    for &x in v {
        if x.is_nan() {
            nans += 1;
        } else if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    (max_abs, nans)
}

async fn read_buf_f32(ctx: &rullama::backend::WgpuCtx, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let bytes = (n * 4) as u64;
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad.read"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grad.read.enc"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let slice = read_buf.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
    rx.await.unwrap().unwrap();
    let data = slice.get_mapped_range();
    let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    v
}

fn slot_view(l: &crate::lora::LoraLayer) -> LoraSlot<'_> {
    LoraSlot {
        a: &l.a,
        b: &l.b,
        z: &l.z,
        rank: l.rank,
        scale: l.scale,
    }
}

fn grad_view(l: &crate::lora::LoraLayer) -> LoraGradPair<'_> {
    LoraGradPair {
        a: &l.a,
        b: &l.b,
        z: &l.z,
        d_a: &l.da,
        d_b: &l.db,
        rank: l.rank,
        scale: l.scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraKey, LoraState};
    use rullama::backend::WgpuCtx;
    use std::sync::Arc;

    /// Build a small `LoraState`, write some known values into one
    /// layer's A/B buffers, serialize via the safetensors path, then
    /// reload into a *fresh* `LoraState` and confirm the values
    /// round-trip identically.
    #[test]
    fn adapter_save_load_round_trip() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));

        // Build LoraState A — populate with a known pattern.
        let mut state_a = LoraState::new(Arc::clone(&ctx));
        state_a
            .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
            .unwrap();
        state_a
            .insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 2)
            .unwrap();
        // Overwrite A buffer of layer 0 attn_q with known bytes.
        let known_a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125).collect();
        let known_b: Vec<f32> = (0..8).map(|i| (i as f32) * -0.25 + 0.5).collect();
        {
            let layer = state_a.get(&LoraKey::new(0, "attn_q")).unwrap();
            ctx.queue
                .write_buffer(&layer.a, 0, bytemuck::cast_slice(&known_a));
            ctx.queue
                .write_buffer(&layer.b, 0, bytemuck::cast_slice(&known_b));
        }
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        // Serialize via the same path TrainingSession::save_adapter uses.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            use safetensors::tensor::{Dtype, TensorView};
            let a_vals = pollster::block_on(read_buf_f32(
                &ctx,
                &state_a.get(&LoraKey::new(0, "attn_q")).unwrap().a,
                16,
            ));
            let b_vals = pollster::block_on(read_buf_f32(
                &ctx,
                &state_a.get(&LoraKey::new(0, "attn_q")).unwrap().b,
                8,
            ));
            let k_a_vals = pollster::block_on(read_buf_f32(
                &ctx,
                &state_a.get(&LoraKey::new(0, "attn_k")).unwrap().a,
                16,
            ));
            let k_b_vals = pollster::block_on(read_buf_f32(
                &ctx,
                &state_a.get(&LoraKey::new(0, "attn_k")).unwrap().b,
                8,
            ));
            let a_bytes = bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec();
            let b_bytes = bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec();
            let k_a_bytes = bytemuck::cast_slice::<f32, u8>(&k_a_vals).to_vec();
            let k_b_bytes = bytemuck::cast_slice::<f32, u8>(&k_b_vals).to_vec();
            let mut views: std::collections::HashMap<&str, TensorView<'_>> =
                std::collections::HashMap::new();
            let view_a = TensorView::new(Dtype::F32, vec![2usize, 8usize], &a_bytes).unwrap();
            let view_b = TensorView::new(Dtype::F32, vec![4usize, 2usize], &b_bytes).unwrap();
            let view_ka = TensorView::new(Dtype::F32, vec![2usize, 8usize], &k_a_bytes).unwrap();
            let view_kb = TensorView::new(Dtype::F32, vec![4usize, 2usize], &k_b_bytes).unwrap();
            views.insert("lora.blk.0.attn_q.A", view_a);
            views.insert("lora.blk.0.attn_q.B", view_b);
            views.insert("lora.blk.0.attn_k.A", view_ka);
            views.insert("lora.blk.0.attn_k.B", view_kb);
            safetensors::serialize_to_file(&views, &None, &path).unwrap();
        }

        // Build LoraState B with same shape (but different initial values).
        let mut state_b = LoraState::new(Arc::clone(&ctx));
        state_b
            .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 99)
            .unwrap();
        state_b
            .insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 100)
            .unwrap();

        // Load adapter into state_b.
        let loaded = load_adapter_into_state(&mut state_b, &path).unwrap();
        assert_eq!(loaded, 4, "expected to load 4 tensors (A + B for q and k)");

        // Read A and B back from state_b and confirm they match known_a/known_b.
        let layer_q = state_b.get(&LoraKey::new(0, "attn_q")).unwrap();
        let a_round = pollster::block_on(read_buf_f32(&ctx, &layer_q.a, 16));
        let b_round = pollster::block_on(read_buf_f32(&ctx, &layer_q.b, 8));
        for (orig, round) in known_a.iter().zip(a_round.iter()) {
            assert_eq!(orig, round, "A mismatch: orig={orig} round={round}");
        }
        for (orig, round) in known_b.iter().zip(b_round.iter()) {
            assert_eq!(orig, round, "B mismatch: orig={orig} round={round}");
        }
    }

    /// Same as the f32 round trip but with `mixed_precision`-style f16
    /// dtype in the safetensors file. Tolerance is f16 quantization
    /// noise (~5e-4 for values in [-1, 1]); load round-trips through
    /// f32 on the GPU side.
    #[test]
    fn adapter_save_load_round_trip_f16() {
        let ctx = Arc::new(pollster::block_on(WgpuCtx::new()).expect("wgpu"));

        let mut state_a = LoraState::new(Arc::clone(&ctx));
        state_a
            .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1)
            .unwrap();
        let known_a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125 - 0.5).collect();
        let known_b: Vec<f32> = (0..8).map(|i| (i as f32) * -0.25 + 0.5).collect();
        {
            let layer = state_a.get(&LoraKey::new(0, "attn_q")).unwrap();
            ctx.queue
                .write_buffer(&layer.a, 0, bytemuck::cast_slice(&known_a));
            ctx.queue
                .write_buffer(&layer.b, 0, bytemuck::cast_slice(&known_b));
        }
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        // Serialize as f16, manually mirroring `save_adapter`'s f16 path.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            use safetensors::tensor::{Dtype, TensorView};
            let layer_q = state_a.get(&LoraKey::new(0, "attn_q")).unwrap();
            let a_vals = pollster::block_on(read_buf_f32(&ctx, &layer_q.a, 16));
            let b_vals = pollster::block_on(read_buf_f32(&ctx, &layer_q.b, 8));
            let a_h: Vec<half::f16> = a_vals.iter().map(|&x| half::f16::from_f32(x)).collect();
            let b_h: Vec<half::f16> = b_vals.iter().map(|&x| half::f16::from_f32(x)).collect();
            let a_bytes = bytemuck::cast_slice::<half::f16, u8>(&a_h).to_vec();
            let b_bytes = bytemuck::cast_slice::<half::f16, u8>(&b_h).to_vec();
            let mut views: std::collections::HashMap<&str, TensorView<'_>> =
                std::collections::HashMap::new();
            let view_a = TensorView::new(Dtype::F16, vec![2usize, 8usize], &a_bytes).unwrap();
            let view_b = TensorView::new(Dtype::F16, vec![4usize, 2usize], &b_bytes).unwrap();
            views.insert("lora.blk.0.attn_q.A", view_a);
            views.insert("lora.blk.0.attn_q.B", view_b);
            safetensors::serialize_to_file(&views, &None, &path).unwrap();
        }

        let mut state_b = LoraState::new(Arc::clone(&ctx));
        state_b
            .insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 99)
            .unwrap();
        let loaded = load_adapter_into_state(&mut state_b, &path).unwrap();
        assert_eq!(loaded, 2, "f16 round-trip: expected 2 tensors");

        let layer_q = state_b.get(&LoraKey::new(0, "attn_q")).unwrap();
        let a_round = pollster::block_on(read_buf_f32(&ctx, &layer_q.a, 16));
        let b_round = pollster::block_on(read_buf_f32(&ctx, &layer_q.b, 8));
        for (orig, round) in known_a.iter().zip(a_round.iter()) {
            let d = (orig - round).abs();
            assert!(
                d < 1e-3,
                "A f16 round trip: orig={orig} round={round} diff={d}"
            );
        }
        for (orig, round) in known_b.iter().zip(b_round.iter()) {
            let d = (orig - round).abs();
            assert!(
                d < 1e-3,
                "B f16 round trip: orig={orig} round={round} diff={d}"
            );
        }
    }
}
