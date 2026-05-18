//! wasm-bindgen training surface — only compiled for `wasm32`.
//!
//! Mirrors the camelCase + async-Promise pattern of
//! `crates/rullama/src/api.rs`. The browser worker constructs a
//! `TrainingSession` over a loaded `Model`, drives steps + manual
//! gradient accumulation, then calls `saveAdapter()` to pull the
//! adapter bytes back as a `Uint8Array` (which the worker writes to
//! OPFS under `rullama-adapters/<name>.bin`).
//!
//! Model ownership: `TrainingSession::new` consumes the `Model` (same
//! contract as the native API). Call `finish()` to release the
//! `Model` back to JS so chat can resume against the same loaded
//! weights without a multi-gigabyte reload.

use rullama::api::Model;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::session::TrainingSession as NativeSession;
use crate::shared::config::{LoraConfig, TrainingHyperparams};

#[derive(Serialize)]
struct ProbeReport {
    ok: bool,
    /// Coarse GPU-byte estimate of what a `TrainingSession::new` call
    /// would allocate. Useful for "you'd need X MB" diagnostics even
    /// on the success path.
    #[serde(rename = "estimatedBytes")]
    estimated_bytes: u64,
    /// Present when `ok=false` — the wgpu/probe error message verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

/// Try the training-side allocations without consuming the Model so the
/// chat path stays alive on memory-tight devices. Returns
/// `{ok: true, estimatedBytes}` if the trial succeeded, `{ok: false,
/// estimatedBytes, reason}` if any allocation failed or wgpu surfaced
/// an OOM error.
///
/// JS pattern:
/// ```js
/// const probe = await TrainingSession.probeFit(model, loraJson, hpJson);
/// if (!probe.ok) { showError(probe.reason); return; }
/// const session = new TrainingSession(model, loraJson, hpJson);
/// ```
#[wasm_bindgen(js_name = probeTrainingFit)]
pub async fn probe_training_fit_js(
    model: &Model,
    lora_config_json: &str,
    hp_json: &str,
) -> std::result::Result<JsValue, JsError> {
    let lora_cfg: LoraConfig = serde_json::from_str(lora_config_json)
        .map_err(|e| JsError::new(&format!("invalid loraConfig JSON: {e}")))?;
    let hp: TrainingHyperparams = serde_json::from_str(hp_json)
        .map_err(|e| JsError::new(&format!("invalid hyperparams JSON: {e}")))?;

    let report = match NativeSession::probe(model, &lora_cfg, &hp).await {
        Ok(bytes) => ProbeReport { ok: true, estimated_bytes: bytes, reason: None },
        Err(e) => ProbeReport {
            ok: false,
            estimated_bytes: crate::session::estimate_training_bytes(
                model.forward().cfg(),
                &lora_cfg,
                &hp,
            ),
            reason: Some(format!("{e}")),
        },
    };
    serde_wasm_bindgen::to_value(&report)
        .map_err(|e| JsError::new(&format!("serialize probe report: {e}")))
}

/// JS-facing wrapper around the native `TrainingSession`.
///
/// All async methods return `Promise<JsValue>` resolving to a small
/// JSON-shaped result `{loss, lr, step}` (or `void` for ops that
/// only mutate state).
#[wasm_bindgen]
pub struct TrainingSession {
    inner: NativeSession,
}

#[derive(Serialize)]
struct StepReport {
    loss: f32,
    lr: f64,
    step: u32,
}

#[wasm_bindgen]
impl TrainingSession {
    /// Build a new session over the supplied `Model`. The Model is
    /// **consumed** by the session — call `finish()` to get it back.
    ///
    /// `loraConfigJson` and `hparamsJson` are JSON strings shaped by
    /// `LoraConfig` / `TrainingHyperparams`. Defaults match the native
    /// `train_jsonl` example so the browser UI's "use defaults" path
    /// behaves the same as the CLI.
    #[wasm_bindgen(constructor)]
    pub fn new(
        model: Model,
        lora_config_json: &str,
        hp_json: &str,
    ) -> std::result::Result<TrainingSession, JsError> {
        let lora_cfg: LoraConfig = serde_json::from_str(lora_config_json)
            .map_err(|e| JsError::new(&format!("invalid loraConfig JSON: {e}")))?;
        let hp: TrainingHyperparams = serde_json::from_str(hp_json)
            .map_err(|e| JsError::new(&format!("invalid hyperparams JSON: {e}")))?;
        let inner =
            NativeSession::new(model, lora_cfg, hp).map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(Self { inner })
    }

    /// One NextToken training step. Zeros grads, runs forward+backward,
    /// applies Adam. Resolves with `{loss, lr, step}`.
    #[wasm_bindgen(js_name = step)]
    pub async fn step_js(
        &mut self,
        input_ids: Vec<u32>,
        target_id: u32,
    ) -> std::result::Result<JsValue, JsError> {
        let loss = self
            .inner
            .step(&input_ids, target_id)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Self::step_report(loss, &self.inner)
    }

    /// One PerPosition training step. `targets` must have the same
    /// length as `inputIds`; positions whose target is `0xFFFFFFFF`
    /// (`u32::MAX`) are masked. Returns `{loss, lr, step}` where loss
    /// is the mean cross-entropy across active positions.
    #[wasm_bindgen(js_name = stepPerPosition)]
    pub async fn step_per_position_js(
        &mut self,
        input_ids: Vec<u32>,
        targets: Vec<u32>,
    ) -> std::result::Result<JsValue, JsError> {
        let loss = self
            .inner
            .step_per_position(&input_ids, &targets)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))?;
        Self::step_report(loss, &self.inner)
    }

    /// Zero every LoRA's gradient buffers. Call at the start of a
    /// gradient-accumulation cycle.
    #[wasm_bindgen(js_name = zeroGrads)]
    pub fn zero_grads_js(&mut self) {
        self.inner.zero_grads()
    }

    /// One forward+backward pass — does NOT zero gradients or step
    /// Adam. Use inside a manual accumulation loop:
    /// `zeroGrads()` → N × `forwardBackward(...)` → `optimizerStep()`.
    /// Returns the scalar loss for this micro-batch.
    #[wasm_bindgen(js_name = forwardBackward)]
    pub async fn forward_backward_js(
        &mut self,
        input_ids: Vec<u32>,
        target_id: u32,
    ) -> std::result::Result<f32, JsError> {
        self.inner
            .forward_backward(&input_ids, target_id)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// One forward+backward (PerPosition variant) — same accumulation
    /// contract as `forwardBackward`.
    #[wasm_bindgen(js_name = forwardBackwardPerPosition)]
    pub async fn forward_backward_per_position_js(
        &mut self,
        input_ids: Vec<u32>,
        targets: Vec<u32>,
    ) -> std::result::Result<f32, JsError> {
        self.inner
            .forward_backward_per_position(&input_ids, &targets)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Apply the accumulated gradients with Adam. Bumps `step`.
    #[wasm_bindgen(js_name = optimizerStep)]
    pub fn optimizer_step_js(&mut self) {
        self.inner.optimizer_step()
    }

    /// Serialize the current adapter to safetensors bytes. JS decides
    /// where the bytes go (OPFS write via `FileSystemSyncAccessHandle`,
    /// `Blob` download, etc.). Metadata sidecar carries
    /// rank/alpha/target_modules so `Model.loadAdapter` can reconstruct
    /// the shape table.
    #[wasm_bindgen(js_name = saveAdapter)]
    pub async fn save_adapter_js(&self) -> std::result::Result<Vec<u8>, JsError> {
        self.inner
            .save_adapter_to_bytes()
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// 1-based step counter — bumped after each `step()` /
    /// `optimizerStep()`.
    #[wasm_bindgen(js_name = stepNum, getter)]
    pub fn step_num_js(&self) -> u32 {
        self.inner.step_num()
    }

    /// Consume the session and return the wrapped `Model` to JS so
    /// chat can resume against the same loaded weights without a
    /// multi-gigabyte reload. After this call the `TrainingSession`
    /// handle is invalid.
    #[wasm_bindgen(js_name = finish)]
    pub fn finish_js(self) -> Model {
        self.inner.into_model()
    }

    /// Current learning rate (post-warmup, post-schedule). Useful for
    /// the loss-chart label.
    #[wasm_bindgen(js_name = lr, getter)]
    pub fn current_lr_js(&self) -> f64 {
        self.inner.current_lr()
    }

    /// Number of trainable LoRA parameters across all wrapped
    /// projections (rank × in_dim + out_dim × rank, summed).
    #[wasm_bindgen(js_name = parameterCount, getter)]
    pub fn parameter_count_js(&self) -> u32 {
        // u64 → u32 truncation: rank-16 over q/k/v/o on gemma4-e4b is
        // ≈4 M params, well within u32. Larger adapters would be a
        // browser memory problem long before this overflows.
        self.inner.parameter_count() as u32
    }

    /// True iff this session was built with `gradient_checkpointing=true`.
    #[wasm_bindgen(js_name = gradientCheckpointing, getter)]
    pub fn gradient_checkpointing_js(&self) -> bool {
        self.inner.gradient_checkpointing()
    }

    /// True iff this session was built with `mixed_precision=true`.
    #[wasm_bindgen(js_name = mixedPrecision, getter)]
    pub fn mixed_precision_js(&self) -> bool {
        self.inner.mixed_precision()
    }

    /// Configure an LR schedule for the next `totalSteps` optimizer
    /// steps. Respects the session's `warmup_steps` + `lr_scheduler`
    /// configuration. Without calling this, the optimizer uses the
    /// constant `hp.learning_rate`.
    #[wasm_bindgen(js_name = setLrSchedule)]
    pub fn set_lr_schedule_js(&mut self, total_steps: u32) {
        self.inner.set_lr_schedule(total_steps as u64);
    }

    fn step_report(loss: f32, inner: &NativeSession) -> std::result::Result<JsValue, JsError> {
        let report = StepReport {
            loss,
            lr: inner.current_lr(),
            step: inner.step_num(),
        };
        serde_wasm_bindgen::to_value(&report)
            .map_err(|e| JsError::new(&format!("serialize step report: {e}")))
    }
}
