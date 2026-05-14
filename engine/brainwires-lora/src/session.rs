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
    adam_step_chained, scale_chained, sum_of_squares_chained, AdamConfig,
};
use rullama::reference::forward_chained::{
    BackwardScratchView, LayerCaptureBuffers, LayerLoraGrads, LayerLoraSlots,
    LoraGradPair, LoraSlot,
};

use crate::lora::{LoraKey, LoraState};
use crate::lr_schedule::LrSchedule;
use crate::scratch::TrainingScratch;
use crate::shared::config::{LoraConfig, LossMode, TrainingHyperparams};
use crate::shared::error::TrainingError;

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
    /// 1-based step counter used by Adam's bias correction. Increments
    /// at the end of every successful `step()` call.
    step_num: u32,
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

        let mut loras = LoraState::new(Arc::clone(&ctx));
        let d_model = cfg.d_model;
        for layer in 0..cfg.n_layers {
            let head_dim = cfg.head_dim(layer);
            let n_heads_dim = cfg.n_heads * head_dim;
            let n_kv_dim = cfg.n_kv_heads(layer) * head_dim;
            let ffn_n = cfg.ffn(layer);
            for proj in &lora_cfg.target_modules {
                let (in_dim, out_dim) = match proj.as_str() {
                    "attn_q"   => (d_model, n_heads_dim),
                    "attn_k"   => (d_model, n_kv_dim),
                    "attn_v"   => (d_model, n_kv_dim),
                    "attn_o"   => (n_heads_dim, d_model),
                    "ffn_gate" => (d_model, ffn_n),
                    "ffn_up"   => (d_model, ffn_n),
                    "ffn_down" => (ffn_n, d_model),
                    other => {
                        return Err(TrainingError::Config(format!(
                            "supported LoRA targets: attn_q/k/v/o + ffn_gate/up/down, got {other}"
                        )));
                    }
                };
                // Deterministic seed per (layer, proj) so reruns are
                // reproducible without an extra RNG.
                let proj_idx = ["attn_q", "attn_k", "attn_v", "attn_o",
                                "ffn_gate", "ffn_up", "ffn_down"]
                    .iter()
                    .position(|p| *p == proj.as_str())
                    .unwrap_or(0) as u64;
                let seed = hp.seed
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
            step_num: 1,
        })
    }

    /// The loss objective this session was constructed with.
    pub fn loss_mode(&self) -> LossMode { self.loss_mode }

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
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("train.sos"),
        });
        for (_key, layer) in self.loras.iter() {
            sum_of_squares_chained(&ctx, &pipes, &mut enc,
                &layer.da, &layer.sos_a, layer.a_len(), 1.0);
            sum_of_squares_chained(&ctx, &pipes, &mut enc,
                &layer.db, &layer.sos_b, layer.b_len(), 1.0);
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
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
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
        let mut enc2 = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        let adam = AdamConfig { step: self.step_num, lr, ..self.adam_cfg };
        let ctx = self.model.forward().ctx().clone();
        let pipes = self.model.forward().pipes().clone();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("train.adam"),
        });
        for (_key, layer) in self.loras.iter() {
            adam_step_chained(&ctx, &pipes, &mut enc,
                &layer.da, &layer.a, &layer.m_a, &layer.v_a, layer.a_len(), adam);
            adam_step_chained(&ctx, &pipes, &mut enc,
                &layer.db, &layer.b, &layer.m_b, &layer.v_b, layer.b_len(), adam);
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
                q:        self.loras.get(&LoraKey::new(li as u32, "attn_q")).map(slot_view),
                k:        self.loras.get(&LoraKey::new(li as u32, "attn_k")).map(slot_view),
                v:        self.loras.get(&LoraKey::new(li as u32, "attn_v")).map(slot_view),
                o:        self.loras.get(&LoraKey::new(li as u32, "attn_o")).map(slot_view),
                ffn_gate: self.loras.get(&LoraKey::new(li as u32, "ffn_gate")).map(slot_view),
                ffn_up:   self.loras.get(&LoraKey::new(li as u32, "ffn_up")).map(slot_view),
                ffn_down: self.loras.get(&LoraKey::new(li as u32, "ffn_down")).map(slot_view),
            })
            .collect();

        // Per-layer capture views into TrainingScratch.layers.
        let capture: Vec<LayerCaptureBuffers> = self
            .scratch
            .layers
            .iter()
            .map(|l| LayerCaptureBuffers {
                hidden_in:   &l.hidden_in,
                norm_x_attn: &l.norm_x_attn,
                q_pre_norm:  &l.q_pre_norm,
                q_post_rope: &l.q_post_rope,
                k_pre_norm:  &l.k_pre_norm,
                v_pre_norm:  &l.v_pre_norm,
                attn_out:    &l.attn_out,
                attn_proj:   &l.attn_proj,
                pre_ffn_rms: &l.pre_ffn_rms,
                norm_x_ffn:  &l.norm_x_ffn,
                ffn_gate:    &l.ffn_gate,
                ffn_up:      &l.ffn_up,
                ffn_act:     &l.ffn_act,
                ffn_out:     &l.ffn_out,
            })
            .collect();

        // Prefill prompt — every position 0..N-2 uses LoRA correction
        // but no activation capture (their K/V are stored into the
        // cache, that's all we need for backward).
        for &tok in &input_ids[..input_ids.len() - 1] {
            self.model
                .forward_mut()
                .step_with_lora(tok, &lora_slots)
                .await
                .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;
        }
        // Final position — capture activations + compute logits.
        let final_tok = *input_ids.last().unwrap();
        let _logits = self
            .model
            .forward_mut()
            .step_capture(final_tok, &capture, Some(&lora_slots))
            .await
            .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;

        // Backward.
        let grads: Vec<LayerLoraGrads> = (0..n_layers)
            .map(|li| LayerLoraGrads {
                q:        self.loras.get(&LoraKey::new(li as u32, "attn_q")).map(grad_view),
                k:        self.loras.get(&LoraKey::new(li as u32, "attn_k")).map(grad_view),
                v:        self.loras.get(&LoraKey::new(li as u32, "attn_v")).map(grad_view),
                o:        self.loras.get(&LoraKey::new(li as u32, "attn_o")).map(grad_view),
                ffn_gate: self.loras.get(&LoraKey::new(li as u32, "ffn_gate")).map(grad_view),
                ffn_up:   self.loras.get(&LoraKey::new(li as u32, "ffn_up")).map(grad_view),
                ffn_down: self.loras.get(&LoraKey::new(li as u32, "ffn_down")).map(grad_view),
            })
            .collect();
        let s = &self.scratch;
        // `d_attn_out` aliases `d_q_pre_rope`: the latter is never
        // written by `backward_layer` (`rope_neox_backward` modifies
        // `d_q` in place), so the buffer is idle and the right size
        // (`[n_heads · head_dim_max]`) — perfect overflow scratch.
        let scratch_view = BackwardScratchView {
            d_logits:       &s.d_logits,
            loss:           &s.loss,
            d_hidden_final: &s.d_hidden_final,
            d_hidden:       &s.d_hidden,
            d_hidden_tmp:   &s.d_hidden_tmp,
            d_hidden_tmp2:  &s.d_hidden_tmp2,
            attn_probs:     &s.attn_probs,
            attn_d_scores:  &s.attn_d_scores,
            d_attn_out:     &s.d_q_pre_rope,
            d_q:            &s.d_q,
            d_k_hist:       &s.d_k_hist,
            d_v_hist:       &s.d_v_hist,
            d_q_pre_rope:   &s.d_q_pre_rope,
            d_k_pre_rope:   &s.d_k_pre_rope,
            d_q_pre_norm:   &s.d_q_pre_norm,
            d_k_pre_norm:   &s.d_k_pre_norm,
            d_v_pre_norm:   &s.d_v_pre_norm,
            d_ffn_a:        &s.d_ffn_a,
            d_ffn_b:        &s.d_ffn_b,
            d_ffn_c:        &s.d_ffn_c,
        };
        let history_len = input_ids.len() as u32;
        let pos = (input_ids.len() - 1) as u32;
        let loss = self
            .model
            .forward_mut()
            .backward_step(target_id, &capture, &lora_slots, &grads, &scratch_view, history_len, pos)
            .await
            .map_err(|e| TrainingError::Backend(format!("{e:?}")))?;

        // Debug readback of LoRA gradient norms — set
        // `RULLAMA_DEBUG_GRADS=1` to print each layer's dA/dB max-abs +
        // NaN count. Used to localise NaN sources in the backward path.
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
    pub async fn step(
        &mut self,
        input_ids: &[u32],
        target_id: u32,
    ) -> Result<f32, TrainingError> {
        self.zero_grads();
        let loss = self.forward_backward(input_ids, target_id).await?;
        if self.max_grad_norm > 0.0 {
            self.clip_grad_norm(self.max_grad_norm).await?;
        }
        self.optimizer_step();
        Ok(loss)
    }

    /// PerPosition forward+backward: for every position whose
    /// `targets[p] != u32::MAX`, run a fresh forward sweep over
    /// `input_ids[..=p]`, capture activations at position `p`, and
    /// run backward seeded with `dL/d_logits` at `p`. Gradients
    /// accumulate into the same LoRA `dA`/`dB` across positions — the
    /// caller drives the accumulation cycle (`zero_grads` once before,
    /// `optimizer_step` once after, or call [`step_per_position`]).
    ///
    /// Returns the **mean** cross-entropy across the active
    /// positions. Like [`forward_backward`], does *not* zero or step.
    ///
    /// Build `targets` with
    /// [`crate::dataset_loader::Tokenizer::encode_example`] (or the
    /// `next_token_targets` helper inside the dataset loader) so the
    /// prompt positions stay masked and only completion positions
    /// contribute to the loss.
    ///
    /// Trade-off: each non-masked position runs a full forward sweep
    /// from scratch. For a completion of length `C` over a prompt of
    /// length `P`, total forward work is `O(C * (P + C/2))`. The
    /// alternative — capture activations at every position in a
    /// single forward and walk positions in reverse during backward —
    /// would be more efficient but requires multi-position activation
    /// storage and a multi-position dL/dlogits seed; deferred to a
    /// future optimization pass. Adam's scale invariance keeps the
    /// update direction equivalent regardless of summing-vs-averaging
    /// the per-position gradients before the optimizer.
    pub async fn forward_backward_per_position(
        &mut self,
        input_ids: &[u32],
        targets: &[u32],
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
        let mut total_loss = 0.0f32;
        for (p, &target_id) in targets.iter().enumerate() {
            if target_id == u32::MAX {
                continue;
            }
            let prefix = &input_ids[..=p];
            total_loss += self.forward_backward(prefix, target_id).await?;
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
        self.zero_grads();
        let loss = self.forward_backward_per_position(input_ids, targets).await?;
        if self.max_grad_norm > 0.0 {
            self.clip_grad_norm(self.max_grad_norm).await?;
        }
        self.optimizer_step();
        Ok(loss)
    }

    /// Number of LoRA parameters currently being trained.
    pub fn parameter_count(&self) -> u64 {
        self.loras.parameter_count()
    }

    /// Immutable handle on the wrapped model — for token encoding /
    /// inference between training steps.
    pub fn model(&self) -> &Model { &self.model }

    /// Serialize the current LoRA A/B matrices into a safetensors file.
    ///
    /// Tensor naming: `lora.blk.{layer}.{projection}.{A|B}`. Stored as
    /// f32 row-major. Metadata sidecar (safetensors header `__metadata__`)
    /// carries rank/alpha/target_modules so a loader can rebuild the
    /// `LoraState` without external context.
    ///
    /// Note: the `m_a/v_a/m_b/v_b` Adam state and gradient accumulators
    /// are *not* persisted — only the trainable parameters. To resume
    /// training, instantiate a fresh `TrainingSession` and call
    /// `load_adapter_into` to seed A/B from disk; Adam state restarts
    /// at step 1 (acceptable for downstream fine-tunes; matches what
    /// HF Transformers does).
    pub async fn save_adapter(&self, path: &std::path::Path) -> Result<(), TrainingError> {
        use safetensors::tensor::{Dtype, TensorView};
        let ctx = self.model.forward().ctx().clone();

        // 1. Pull every LoRA A/B back to host as bytes.
        let mut tensors: Vec<(String, Vec<u32>, Vec<u8>)> = Vec::new();
        for (key, layer) in self.loras.iter() {
            let a_vals = read_buf_f32(&ctx, &layer.a, layer.a_len()).await;
            let b_vals = read_buf_f32(&ctx, &layer.b, layer.b_len()).await;
            let a_bytes = bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec();
            let b_bytes = bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec();
            let a_name = format!("lora.blk.{}.{}.A", key.layer, key.projection);
            let b_name = format!("lora.blk.{}.{}.B", key.layer, key.projection);
            // A shape: [rank, in_dim]; B shape: [out_dim, rank].
            let a_shape = vec![layer.rank, layer.in_dim];
            let b_shape = vec![layer.out_dim, layer.rank];
            tensors.push((a_name, a_shape, a_bytes));
            tensors.push((b_name, b_shape, b_bytes));
        }

        // 2. Build TensorViews (each borrows from the owned byte vec).
        let mut views: std::collections::HashMap<&str, TensorView<'_>> = std::collections::HashMap::new();
        for (name, shape, bytes) in &tensors {
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let view = TensorView::new(Dtype::F32, shape_usize, bytes)
                .map_err(|e| TrainingError::Backend(format!("safetensors view: {e}")))?;
            views.insert(name.as_str(), view);
        }

        // 3. Metadata sidecar.
        let any = self.loras.iter().next();
        let (rank, alpha) = match any {
            Some((_, layer)) => (layer.rank, layer.scale * layer.rank as f32),
            None => (0u32, 0.0f32),
        };
        let mut target_modules: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (key, _) in self.loras.iter() {
            target_modules.insert(key.projection.clone());
        }
        let target_modules_list: Vec<String> = target_modules.into_iter().collect();
        let metadata: std::collections::HashMap<String, String> = [
            ("format".to_string(), "rullama-lora-v0".to_string()),
            ("rank".to_string(), rank.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("target_modules".to_string(), target_modules_list.join(",")),
        ].into_iter().collect();

        // 4. Serialize to file.
        safetensors::serialize_to_file(&views, &Some(metadata), path)
            .map_err(|e| TrainingError::Backend(format!("safetensors write: {e}")))?;
        Ok(())
    }

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
/// `LoraState`. Each named tensor in the file (`lora.blk.{layer}.{proj}.{A|B}`)
/// is written to the matching `LoraLayer` buffer.
///
/// The `LoraState` must already have all the expected LoRA slots
/// registered with matching shapes — i.e. caller built it via
/// `TrainingSession::new`-style construction. Tensors in the file
/// that don't have a matching slot are skipped (with a tracing
/// warning); slots with no matching tensor are left at whatever the
/// caller's initialiser produced.
pub fn load_adapter_into_state(
    state: &mut crate::lora::LoraState,
    path: &std::path::Path,
) -> Result<usize, TrainingError> {
    use safetensors::SafeTensors;

    let bytes = std::fs::read(path)
        .map_err(|e| TrainingError::Backend(format!("read {}: {e}", path.display())))?;
    let st = SafeTensors::deserialize(&bytes)
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
        let data = tensor.data();
        if data.len() != buf.size() as usize {
            return Err(TrainingError::Backend(format!(
                "tensor {name} size mismatch: file={} expected={}",
                data.len(), buf.size()
            )));
        }
        ctx.queue.write_buffer(buf, 0, data);
        loaded += 1;
    }
    Ok(loaded)
}

fn stats(v: &[f32]) -> (f32, usize) {
    let mut max_abs = 0.0f32;
    let mut nans = 0usize;
    for &x in v {
        if x.is_nan() { nans += 1; }
        else if x.abs() > max_abs { max_abs = x.abs(); }
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
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("grad.read.enc"),
    });
    enc.copy_buffer_to_buffer(buf, 0, &read_buf, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let slice = read_buf.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
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
        state_a.insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 1).unwrap();
        state_a.insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 2).unwrap();
        // Overwrite A buffer of layer 0 attn_q with known bytes.
        let known_a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125).collect();
        let known_b: Vec<f32> = (0..8).map(|i| (i as f32) * -0.25 + 0.5).collect();
        {
            let layer = state_a.get(&LoraKey::new(0, "attn_q")).unwrap();
            ctx.queue.write_buffer(&layer.a, 0, bytemuck::cast_slice(&known_a));
            ctx.queue.write_buffer(&layer.b, 0, bytemuck::cast_slice(&known_b));
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        // Serialize via the same path TrainingSession::save_adapter uses.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            use safetensors::tensor::{Dtype, TensorView};
            let a_vals = pollster::block_on(read_buf_f32(&ctx, &state_a.get(&LoraKey::new(0, "attn_q")).unwrap().a, 16));
            let b_vals = pollster::block_on(read_buf_f32(&ctx, &state_a.get(&LoraKey::new(0, "attn_q")).unwrap().b, 8));
            let k_a_vals = pollster::block_on(read_buf_f32(&ctx, &state_a.get(&LoraKey::new(0, "attn_k")).unwrap().a, 16));
            let k_b_vals = pollster::block_on(read_buf_f32(&ctx, &state_a.get(&LoraKey::new(0, "attn_k")).unwrap().b, 8));
            let a_bytes = bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec();
            let b_bytes = bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec();
            let k_a_bytes = bytemuck::cast_slice::<f32, u8>(&k_a_vals).to_vec();
            let k_b_bytes = bytemuck::cast_slice::<f32, u8>(&k_b_vals).to_vec();
            let mut views: std::collections::HashMap<&str, TensorView<'_>> = std::collections::HashMap::new();
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
        state_b.insert(LoraKey::new(0, "attn_q"), 8, 2, 4, 4.0, 99).unwrap();
        state_b.insert(LoraKey::new(0, "attn_k"), 8, 2, 4, 4.0, 100).unwrap();

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
}
