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
/// Queryable GPU-memory monitor — returns the current tracked GPU
/// buffer breakdown as `tot=N w=N s=N kv=N lora=N o=N` (MiB). Free
/// function so the test harness / worker can call it any time (incl.
/// re-entrantly from a training progress callback) without touching a
/// borrowed `TrainingSession`. See `rullama::backend::gpu_mem`.
#[wasm_bindgen(js_name = gpuMemBreakdown)]
pub fn gpu_mem_breakdown_js() -> String {
    rullama::backend::gpu_mem::breakdown_str()
}

/// Just the total tracked GPU MiB (cheap; for folding into per-layer
/// beacons to capture the on-device memory trajectory).
#[wasm_bindgen(js_name = gpuMemTotalMib)]
pub fn gpu_mem_total_mib_js() -> f64 {
    rullama::backend::gpu_mem::snapshot_mib().0 as f64
}

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
        Ok(bytes) => ProbeReport {
            ok: true,
            estimated_bytes: bytes,
            reason: None,
        },
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

/// Wrap an optional JS function as a Rust `TrainingProgressCb`
/// closure. The returned `Box<dyn Fn>` is borrowed (`.as_deref()`)
/// by the training methods — its lifetime is the await — so the
/// `'static` requirement on the closure trades cheaply against the
/// extra allocation per step.
///
/// Mirrors `Model::encode_image_js`'s wrapping pattern in
/// `crates/rullama/src/api.rs`.
fn wrap_progress_cb(cb: Option<js_sys::Function>) -> Option<Box<dyn Fn(&str, u32, u32)>> {
    cb.map(|f| {
        Box::new(move |phase: &str, current: u32, total: u32| {
            // Best-effort: failure to call the JS function (e.g. it
            // threw) just drops the beacon; training continues.
            let _ = f.call3(
                &JsValue::NULL,
                &JsValue::from_str(phase),
                &JsValue::from(current),
                &JsValue::from(total),
            );
        }) as Box<dyn Fn(&str, u32, u32)>
    })
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
    ///
    /// Optional `progressCb` is a JS function called at phase
    /// boundaries: `(phase: string, current: number, total: number)`.
    /// Phases: `"prefill"` per prompt token, `"forward"` once on
    /// capture-step end, `"backward"` per layer (top-down), `"clip"`
    /// once, `"optimizer"` once. Used by the PWA to drive a
    /// VisionProgress-style status strip.
    #[wasm_bindgen(js_name = step)]
    pub async fn step_js(
        &mut self,
        input_ids: Vec<u32>,
        target_id: u32,
        progress_cb: Option<js_sys::Function>,
    ) -> std::result::Result<JsValue, JsError> {
        let cb = wrap_progress_cb(progress_cb);
        let loss = self
            .inner
            .step_with_progress(&input_ids, target_id, cb.as_deref())
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
        progress_cb: Option<js_sys::Function>,
    ) -> std::result::Result<JsValue, JsError> {
        let cb = wrap_progress_cb(progress_cb);
        let loss = self
            .inner
            .step_per_position_with_progress(&input_ids, &targets, cb.as_deref())
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
    ///
    /// IMPORTANT: callers that want to also `finish()` after save MUST
    /// use [`save_adapter_and_finish_js`] instead. wasm-bindgen's
    /// borrow-tracking for async methods (even `&mut self`) does NOT
    /// reliably release across awaits in the way we'd want — a
    /// subsequent `finish_js(self)` call will intermittently fail with
    /// "attempted to take ownership of Rust value while it was
    /// borrowed". The combined method takes `self` synchronously,
    /// avoiding the borrow conflict entirely.
    #[wasm_bindgen(js_name = saveAdapter)]
    pub async fn save_adapter_js(&mut self) -> std::result::Result<Vec<u8>, JsError> {
        self.inner
            .save_adapter_to_bytes()
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Save the adapter AND return the wrapped `Model` to JS in a single
    /// call. Takes `self` (consumes the TrainingSession), which makes
    /// the wasm-bindgen JS-side handle invalid immediately on call —
    /// no `&self` / `&mut self` borrow is tracked across the await,
    /// so the await-then-consume sequence is deterministic. This is
    /// the right path for "Save + apply to chat" and "Save (then
    /// release session)" UI flows; only the rare save-without-
    /// finishing case should use [`save_adapter_js`] alone.
    ///
    /// Returns a [`SaveAndFinishResult`] whose `bytes` getter yields
    /// the safetensors bytes (consumed on read) and whose
    /// `takeModel()` yields the `Model` handle for chat. Call both
    /// before dropping the result; either can throw if called twice.
    #[wasm_bindgen(js_name = saveAdapterAndFinish)]
    pub async fn save_adapter_and_finish_js(
        self,
    ) -> std::result::Result<SaveAndFinishResult, JsError> {
        let bytes = self
            .inner
            .save_adapter_to_bytes()
            .await
            .map_err(|e| JsError::new(&format!("{e}")))?;
        let model = self.inner.into_model();
        Ok(SaveAndFinishResult {
            model: Some(model),
            bytes: Some(bytes),
        })
    }

    /// 1-based step counter — bumped after each `step()` /
    /// `optimizerStep()`.
    #[wasm_bindgen(js_name = stepNum, getter)]
    pub fn step_num_js(&self) -> u32 {
        self.inner.step_num()
    }

    /// GPU weight-cache size in bytes — diagnostic for iOS peak-memory
    /// debugging. Beacon this at each training phase to see the
    /// resident weight VRAM trajectory.
    #[wasm_bindgen(js_name = cachedWeightBytes, getter)]
    pub fn cached_weight_bytes_js(&self) -> f64 {
        self.inner.cached_weight_bytes() as f64
    }

    /// Consume the session and return the wrapped `Model` to JS so
    /// chat can resume against the same loaded weights without a
    /// multi-gigabyte reload. After this call the `TrainingSession`
    /// handle is invalid.
    #[wasm_bindgen(js_name = finish)]
    pub fn finish_js(self) -> Model {
        self.inner.into_model()
    }

    /// Cooperatively cancel an in-flight step. Forward + backward
    /// layer walks check the flag between per-layer encoder submits;
    /// the awaited `step()` promise rejects with a "cancelled" error
    /// on the next layer boundary. No-op when nothing is in flight.
    #[wasm_bindgen(js_name = cancel)]
    pub fn cancel_js(&self) {
        self.inner.cancel()
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

/// Return value from `TrainingSession.saveAdapterAndFinish`. Holds the
/// serialized adapter bytes and the returned `Model` handle. Each
/// inner value is consumed on first read (`bytes` getter and
/// `takeModel()`) so the JS-side caller has to claim them once.
#[wasm_bindgen]
pub struct SaveAndFinishResult {
    model: Option<Model>,
    bytes: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl SaveAndFinishResult {
    /// Adapter bytes (safetensors). Consumed on first read; calling
    /// the getter twice throws "bytes already taken". JS callers should
    /// read this into a typed-array immediately and pipe it to OPFS.
    #[wasm_bindgen(getter)]
    pub fn bytes(&mut self) -> std::result::Result<Vec<u8>, JsError> {
        self.bytes
            .take()
            .ok_or_else(|| JsError::new("SaveAndFinishResult: bytes already taken"))
    }

    /// Returns the `Model` handle for the chat-side worker to resume
    /// inference against. Consumed on first call; calling twice throws.
    #[wasm_bindgen(js_name = takeModel)]
    pub fn take_model(&mut self) -> std::result::Result<Model, JsError> {
        self.model
            .take()
            .ok_or_else(|| JsError::new("SaveAndFinishResult: model already taken"))
    }
}
