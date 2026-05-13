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
use rullama::backend::dispatch::{adam_step_chained, AdamConfig};
use rullama::reference::forward_chained::{
    BackwardScratchView, LayerCaptureBuffers, LayerLoraGrads, LayerLoraSlots,
    LoraGradPair, LoraSlot,
};

use crate::lora::{LoraKey, LoraState};
use crate::scratch::TrainingScratch;
use crate::shared::config::{LoraConfig, TrainingHyperparams};
use crate::shared::error::TrainingError;

/// One LoRA fine-tuning session over a loaded model.
pub struct TrainingSession {
    model: Model,
    loras: LoraState,
    scratch: TrainingScratch,
    adam_cfg: AdamConfig,
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
            for proj in &lora_cfg.target_modules {
                let (in_dim, out_dim) = match proj.as_str() {
                    "attn_q" => (d_model, n_heads_dim),
                    "attn_k" => (d_model, n_kv_dim),
                    "attn_v" => (d_model, n_kv_dim),
                    "attn_o" => (n_heads_dim, d_model),
                    other => {
                        return Err(TrainingError::Config(format!(
                            "M0 supports attn_q/k/v/o LoRA only, got {other}"
                        )));
                    }
                };
                // Deterministic seed per (layer, proj) so reruns are
                // reproducible without an extra RNG.
                let proj_idx = ["attn_q", "attn_k", "attn_v", "attn_o"]
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
            step_num: 1,
        })
    }

    /// Run one training step on a single example:
    ///   `loss = CE(forward(input_ids), target_id)`.
    ///
    /// Resets KV cache, runs forward with LoRA correction, captures
    /// activations at the final position, runs backward, applies Adam.
    /// Returns the scalar cross-entropy loss for this step.
    pub async fn step(
        &mut self,
        input_ids: &[u32],
        target_id: u32,
    ) -> Result<f32, TrainingError> {
        if input_ids.is_empty() {
            return Err(TrainingError::Config(
                "step: input_ids must be non-empty".into(),
            ));
        }
        let n_layers = self.model.forward().cfg().n_layers as usize;

        // Fresh KV cache + zeroed gradient buffers.
        self.model.forward_mut().reset();
        self.loras.zero_all_grads();

        // Build per-layer LoRA slot views (forward correction inputs).
        // Stays alive for the whole step — borrows immutably from
        // `self.loras` so we can also mutably borrow `self.model`.
        let lora_slots: Vec<LayerLoraSlots> = (0..n_layers)
            .map(|li| LayerLoraSlots {
                q: self.loras.get(&LoraKey::new(li as u32, "attn_q")).map(slot_view),
                k: self.loras.get(&LoraKey::new(li as u32, "attn_k")).map(slot_view),
                v: self.loras.get(&LoraKey::new(li as u32, "attn_v")).map(slot_view),
                o: self.loras.get(&LoraKey::new(li as u32, "attn_o")).map(slot_view),
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
                q: self.loras.get(&LoraKey::new(li as u32, "attn_q")).map(grad_view),
                k: self.loras.get(&LoraKey::new(li as u32, "attn_k")).map(grad_view),
                v: self.loras.get(&LoraKey::new(li as u32, "attn_v")).map(grad_view),
                o: self.loras.get(&LoraKey::new(li as u32, "attn_o")).map(grad_view),
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

        // Adam over every LoRA's (A, m_a, v_a) and (B, m_b, v_b).
        let adam = AdamConfig { step: self.step_num, ..self.adam_cfg };
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
        Ok(loss)
    }

    /// Number of LoRA parameters currently being trained.
    pub fn parameter_count(&self) -> u64 {
        self.loras.parameter_count()
    }

    /// Immutable handle on the wrapped model — for token encoding /
    /// inference between training steps.
    pub fn model(&self) -> &Model { &self.model }
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
