//! brainwires-chat-pwa-wasm — wasm32 entry point for the chat PWA.
//!
//! Public surface:
//! - [`init`] / [`version`]: lifecycle + build-info.
//! - [`init_local_model`] + [`LocalModelHandle`] + [`local_chat_stream`]: load a
//!   Gemma-family model from JS-supplied bytes and drive streaming chat.
//! - [`count_tokens`] / [`format_prompt`]: prompt utilities.
//! - [`WebTts`] / [`WebStt`]: thin wasm-bindgen wrappers over the framework's
//!   `brainwires_provider_speech::web_speech` module.
//!
//! Streaming bridge:
//! We expose [`local_chat_stream`] as a `web_sys::ReadableStream<Uint8Array>`
//! whose chunks are JSON-encoded objects of shape:
//!
//! ```json
//! { "delta": "...", "finished": false }
//! { "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 } }
//! { "finished": true }
//! ```
//!
//! Choosing a `ReadableStream` (rather than a custom Promise-based "next()"
//! channel) keeps the JS side trivially consumable via `for await
//! (const chunk of stream)` and `TextDecoder`. We hand-roll the bridge using
//! `web_sys::ReadableStream::new_with_underlying_source` plus
//! `wasm_bindgen_futures::spawn_local` rather than pulling in `wasm-streams`,
//! whose maintenance status is unclear and would add another moving part.
//!
//! Cloud chat (Anthropic / OpenAI / Gemini / Ollama) lives entirely in JS —
//! see task #6.
//!
//! The entire surface is gated to `wasm32-unknown-unknown` — the web-speech
//! providers in the framework are wasm-only, so on a native `cargo check`
//! this crate compiles to an empty rlib (which is fine: it's only consumed
//! via wasm-pack).

#![cfg(target_arch = "wasm32")]

use std::collections::HashMap;
use std::sync::Arc;

use brainwires_core::message::{Message, StreamChunk};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_provider::local_llm::CandleLlmProvider;
use brainwires_provider::local_llm::candle_provider::default_gemma_e2b_config;
use brainwires_provider::CandleDevice as Device;
use brainwires_provider::{CandleDType as DType, CandleTensor as Tensor, CandleVarBuilder};
use brainwires_provider::{CandleStorage as Storage, WgpuDevice, WgpuStorage};
use brainwires_provider_speech::web_speech::{
    WebSpeechStt, WebSpeechSttOptions, WebSpeechTts, WebSpeechTtsOptions,
};
use futures::StreamExt;
use js_sys::{Function, Object, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{ReadableStream, ReadableStreamDefaultController};

mod embedding;
pub use embedding::{init_embedding_model, EmbeddingHandle};

mod gemma_pipeline;
pub use gemma_pipeline::{
    init_local_multimodal, init_local_multimodal_chunked, local_chat_stream_with_image,
    LocalMultiModalHandle,
};

#[wasm_bindgen(start)]
pub fn __start() {
    console_error_panic_hook::set_once();
}

/// Returns the crate version string baked in at compile time.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").into()
}

/// Initializes the chat surface. Currently a no-op (the panic hook is
/// installed by `#[wasm_bindgen(start)]`); kept so JS can call it explicitly
/// after `await init()` to express intent.
#[wasm_bindgen]
pub fn init() -> Result<(), JsValue> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Local model handle + streaming chat
// ---------------------------------------------------------------------------

/// Opaque handle around a loaded [`CandleLlmProvider`]. The handle is cheap to
/// clone (it's an `Arc`) and is meant to live for the lifetime of the page so
/// a single set of weights serves many chats.
///
/// Concurrency note: the underlying candle Gemma forward pass holds an
/// internal `Mutex` for its KV cache. We rely on that mutex for serialization
/// — running two streams concurrently against the same handle will see them
/// queue, not corrupt each other. Per the framework comment in
/// `candle_provider.rs`, true per-token streaming requires interleaving
/// sample/forward across a `yield`, which today emits one big `StreamChunk::Text`
/// followed by `Usage` + `Done`. We forward whatever the framework emits as-is.
///
/// Disposal: wasm-bindgen autogenerates a JS-side `free()` method on this
/// struct. Calling `handle.free()` from JS drops the underlying
/// `Arc<CandleLlmProvider>` and releases the wasm-side memory; if it was
/// the last `Arc` reference, the model weights are freed too. The call is
/// idempotent — `local-worker.js` swallows the second-free error so model
/// swaps stay safe even if a stale handle reference lingers.
#[wasm_bindgen]
pub struct LocalModelHandle {
    inner: Arc<CandleLlmProvider>,
}

#[wasm_bindgen]
impl LocalModelHandle {
    #[wasm_bindgen(getter)]
    pub fn model_id(&self) -> String {
        self.inner.name().to_string()
    }

    /// Returns `"webgpu"` or `"cpu"` so JS can report which device is active.
    #[wasm_bindgen(getter)]
    pub fn device_type(&self) -> String {
        let loc = self.inner.device().location();
        match loc {
            brainwires_provider::CandleDeviceLocation::Cpu => "cpu".into(),
            brainwires_provider::CandleDeviceLocation::Wgpu { .. } => "webgpu".into(),
            _ => "unknown".into(),
        }
    }
}

/// Build a [`LocalModelHandle`] from JS-supplied byte buffers (CPU only).
///
/// `weights` is the contents of a single safetensors file; `tokenizer_json`
/// is the contents of `tokenizer.json`. Both are taken by value because
/// wasm-bindgen copies `Vec<u8>` out of the JS `Uint8Array` once and we want
/// to hand ownership straight to candle.
#[wasm_bindgen]
pub fn init_local_model(
    weights: Vec<u8>,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalModelHandle, JsValue> {
    let provider = CandleLlmProvider::from_bytes(&model_id, weights, tokenizer_json)
        .map_err(|e| JsValue::from_str(&format!("init_local_model failed: {e}")))?;
    Ok(LocalModelHandle {
        inner: Arc::new(provider),
    })
}

/// Build a [`LocalModelHandle`] attempting WebGPU first, CPU fallback.
///
/// Async because WebGPU adapter/device negotiation requires awaiting the
/// browser's GPU promise. Returns a `Promise<LocalModelHandle>` to JS.
/// The resolved handle reports which device it landed on via `device_type()`.
#[wasm_bindgen]
pub async fn init_local_model_gpu(
    weights: Vec<u8>,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalModelHandle, JsValue> {

    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(&"wgpu: using WebGPU device".into());
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("wgpu: WebGPU unavailable ({e}), falling back to CPU").into(),
            );
            Device::Cpu
        }
    };

    let provider =
        CandleLlmProvider::from_bytes_on_device(&model_id, weights, tokenizer_json, &device)
            .map_err(|e| JsValue::from_str(&format!("init_local_model_gpu failed: {e}")))?;
    Ok(LocalModelHandle {
        inner: Arc::new(provider),
    })
}

async fn try_webgpu_device() -> Result<Device, String> {
    let has_gpu = js_sys::Reflect::get(
        &js_sys::global(),
        &JsValue::from_str("navigator"),
    )
    .ok()
    .and_then(|nav| js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")).ok())
    .map_or(false, |gpu| !gpu.is_undefined() && !gpu.is_null());

    if !has_gpu {
        return Err("navigator.gpu not available".into());
    }

    // PR #3379's WgpuDevice doesn't expose the bare async constructor;
    // go through Device::new_wgpu_async which internally allocates the
    // adapter, calls into wgpu_compute_layer::WgpuDevice::create_async,
    // and registers the candle kernel loader. Returns Device::Wgpu(_)
    // already wrapped — match it back out so the caller can route.
    Device::new_wgpu_async(0)
        .await
        .map_err(|e| format!("{e}"))
}

// ---------------------------------------------------------------------------
// Chunked safetensors loading (avoids 10 GB single allocation)
// ---------------------------------------------------------------------------

#[derive(Clone, Deserialize)]
pub(crate) struct StTensorInfo {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: (u64, u64),
}

pub(crate) fn st_dtype_to_candle(s: &str) -> Result<DType, String> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "U8" => Ok(DType::U8),
        "U32" => Ok(DType::U32),
        "I16" => Ok(DType::I16),
        "I32" => Ok(DType::I32),
        "I64" => Ok(DType::I64),
        "F64" => Ok(DType::F64),
        other => Err(format!("unsupported safetensors dtype: {other}")),
    }
}

/// Format a JS exception as a human-readable string. `JsValue::as_string`
/// only succeeds when the value is itself a string; thrown `Error` /
/// `DOMException` objects fall through to `<no msg>` otherwise. Probe for
/// the standard `.message` / `.name` shape first, then fall back to
/// `String(e)` via wasm-bindgen's `Debug`.
pub(crate) fn js_err_to_string(e: &JsValue) -> String {
    if let Some(s) = e.as_string() {
        return s;
    }
    let name = Reflect::get(e, &JsValue::from_str("name"))
        .ok()
        .and_then(|v| v.as_string());
    let message = Reflect::get(e, &JsValue::from_str("message"))
        .ok()
        .and_then(|v| v.as_string());
    match (name, message) {
        (Some(n), Some(m)) if !m.is_empty() => format!("{n}: {m}"),
        (Some(n), _) => n,
        (None, Some(m)) if !m.is_empty() => m,
        _ => format!("{e:?}"),
    }
}

pub(crate) fn call_read_fn(read_fn: &Function, offset: u64, length: u64) -> Result<Vec<u8>, JsValue> {
    const CHUNK: u64 = 64 * 1024 * 1024;

    let total = usize::try_from(length).map_err(|_| {
        JsValue::from_str(&format!(
            "tensor too large for wasm32: {length} bytes exceeds usize::MAX"
        ))
    })?;
    if total > isize::MAX as usize {
        return Err(JsValue::from_str(&format!(
            "tensor too large for wasm32 linear memory: {total} bytes exceeds \
             isize::MAX ({}). Model requires a native backend.",
            isize::MAX
        )));
    }

    if length <= CHUNK {
        let result = read_fn.call2(
            &JsValue::NULL,
            &JsValue::from_f64(offset as f64),
            &JsValue::from_f64(length as f64),
        )?;
        let array = Uint8Array::new(&result);
        return Ok(array.to_vec());
    }

    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve(total).map_err(|e| {
        JsValue::from_str(&format!(
            "wasm OOM: failed to reserve {total} bytes for tensor read: {e}"
        ))
    })?;
    // Pre-set length so we can copy_to directly into slices.
    // capacity >= total is guaranteed by try_reserve above,
    // so resize will not reallocate.
    buf.resize(total, 0);

    let mut pos = 0u64;
    while pos < length {
        let chunk_len = std::cmp::min(CHUNK, length - pos);
        let result = read_fn.call2(
            &JsValue::NULL,
            &JsValue::from_f64((offset + pos) as f64),
            &JsValue::from_f64(chunk_len as f64),
        )?;
        let array = Uint8Array::new(&result);
        let start = pos as usize;
        let end = start + array.length() as usize;
        array.copy_to(&mut buf[start..end]);
        pos += chunk_len;
    }
    Ok(buf)
}

/// Stream tensor bytes from OPFS into a GPU buffer.
///
/// Originally we used a chunked write-at-offset pattern
/// (`create_storage_buffer` + `write_to_buffer` + `WgpuStorage::from_raw_buffer`)
/// to keep peak wasm linear memory at one chunk (64 MiB) regardless of
/// tensor size. PR #3379's WGPU backend doesn't expose those low-level
/// buffer ops, so we now collect every chunk into a single `Vec<u8>` and
/// hand it to `alloc_from_bytes`. Peak wasm memory becomes the largest
/// tensor (Gemma 4 E2B's `embed_tokens.weight` is ~805 MB BF16, well
/// within the wasm32 4 GB linear-memory cap). Restoring true streaming
/// will require porting `create_storage_buffer` / `write_to_buffer` /
/// `from_raw_buffer` forward into PR #3379's wgpu-compute-layer.
pub(crate) fn load_tensor_to_gpu(
    read_fn: &Function,
    file_offset: u64,
    byte_length: u64,
    dtype: DType,
    shape: &[usize],
    wgpu_dev: &WgpuDevice,
) -> Result<Tensor, JsValue> {
    const CHUNK: u64 = 64 * 1024 * 1024;

    let mut bytes = Vec::with_capacity(byte_length as usize);
    let mut pos = 0u64;
    while pos < byte_length {
        let chunk_len = std::cmp::min(CHUNK, byte_length - pos);
        let result = read_fn.call2(
            &JsValue::NULL,
            &JsValue::from_f64((file_offset + pos) as f64),
            &JsValue::from_f64(chunk_len as f64),
        )?;
        let array = Uint8Array::new(&result);
        let chunk_bytes = array.to_vec();
        bytes.extend_from_slice(&chunk_bytes);
        pos += chunk_len;
    }

    let storage = wgpu_dev
        .alloc_from_bytes(dtype, &bytes)
        .map_err(|e| JsValue::from_str(&format!("alloc_from_bytes failed: {e}")))?;
    let storage = Storage::Wgpu(storage);
    let shape: candle_core::Shape = shape.into();
    Ok(Tensor::from_storage(
        storage,
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Build a [`LocalModelHandle`] by reading tensors one-at-a-time from OPFS.
///
/// Unlike [`init_local_model_gpu`] which requires the entire safetensors file
/// as a single `Uint8Array` (impossible for 10 GB+ models), this function
/// receives a synchronous JS read callback and loads each tensor individually.
/// Peak WASM memory = the single largest tensor (~2 GB for the embedding),
/// not the full model file.
///
/// `read_fn(offset: number, length: number) → Uint8Array` reads `length`
/// bytes at the given byte offset from the OPFS file. The caller is expected
/// to hold an open `FileSystemSyncAccessHandle` for the duration of this call.
#[wasm_bindgen]
pub async fn init_local_model_chunked(
    read_fn: Function,
    file_size: f64,
    tokenizer_json: Vec<u8>,
    model_id: String,
) -> Result<LocalModelHandle, JsValue> {
    let file_size = file_size as u64;
    web_sys::console::log_1(
        &format!("[wasm] init_local_model_chunked: file_size={file_size}, model={model_id}")
            .into(),
    );

    // ── 1. Read safetensors header ─────────────────────────────────
    let header_size_bytes = call_read_fn(&read_fn, 0, 8)?;
    if header_size_bytes.len() < 8 {
        return Err(JsValue::from_str("failed to read safetensors header size"));
    }
    let header_size =
        u64::from_le_bytes(header_size_bytes[..8].try_into().unwrap());
    web_sys::console::log_1(
        &format!("[wasm] safetensors header: {header_size} bytes").into(),
    );

    let header_bytes = call_read_fn(&read_fn, 8, header_size)?;
    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| JsValue::from_str(&format!("invalid header UTF-8: {e}")))?;

    // ── 2. Parse tensor metadata ───────────────────────────────────
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| JsValue::from_str(&format!("invalid safetensors header: {e}")))?;

    let mut tensor_meta: Vec<(String, StTensorInfo)> = Vec::new();
    for (name, value) in &raw {
        if name == "__metadata__" {
            continue;
        }
        let info: StTensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
            JsValue::from_str(&format!("bad tensor info for {name}: {e}"))
        })?;
        tensor_meta.push((name.clone(), info));
    }
    // Sort by file offset so reads are sequential (better for OPFS).
    tensor_meta.sort_by_key(|(_, info)| info.data_offsets.0);

    web_sys::console::log_1(
        &format!("[wasm] parsed {} tensor entries", tensor_meta.len()).into(),
    );

    let data_start: u64 = 8 + header_size;

    // ── 3. Select device (WebGPU preferred) ────────────────────────
    let device = match try_webgpu_device().await {
        Ok(dev) => {
            web_sys::console::log_1(
                &"[wasm] chunked load: using WebGPU device".into(),
            );
            dev
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[wasm] WebGPU unavailable ({e}), CPU fallback").into(),
            );
            Device::Cpu
        }
    };

    // ── 4. Load each tensor individually ───────────────────────────
    let total = tensor_meta.len();
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(total);

    let wgpu_dev = match &device {
        Device::Wgpu(w) => Some(w.clone()),
        _ => None,
    };

    for (idx, (name, info)) in tensor_meta.iter().enumerate() {
        let offset = data_start + info.data_offsets.0;
        let length = info.data_offsets.1 - info.data_offsets.0;

        let src_dtype = st_dtype_to_candle(&info.dtype).map_err(|e| {
            JsValue::from_str(&format!("tensor {name}: {e}"))
        })?;

        // Tensors that exceed wasm32 addressable memory (or ~2 GB
        // isize::MAX) are streamed directly to GPU via queue.write_buffer,
        // never touching WASM linear memory as a contiguous allocation.
        let needs_gpu_stream = length > (isize::MAX as u64);

        let tensor = if needs_gpu_stream {
            let w = wgpu_dev.as_ref().ok_or_else(|| {
                JsValue::from_str(&format!(
                    "tensor {name} is {length} bytes — too large for wasm32 \
                     and no WebGPU device available for direct upload"
                ))
            })?;
            load_tensor_to_gpu(&read_fn, offset, length, src_dtype, &info.shape, w)?
        } else {
            let bytes = call_read_fn(&read_fn, offset, length)?;
            let t = Tensor::from_raw_buffer(&bytes, src_dtype, &info.shape, &device)
                .map_err(|e| JsValue::from_str(&format!("tensor {name}: {e}")))?;
            drop(bytes);
            t
        };

        let key = name.strip_prefix("model.").unwrap_or(name).to_string();
        tensors.insert(key, tensor);

        if idx % 20 == 0 || idx == total - 1 {
            let method = if needs_gpu_stream { " [gpu-direct]" } else { "" };
            web_sys::console::log_1(
                &format!(
                    "[wasm] loaded tensor {}/{total}: {name} {:?} [{}] ({} bytes){method}",
                    idx + 1,
                    info.shape,
                    info.dtype,
                    length
                )
                .into(),
            );
        }
    }

    web_sys::console::log_1(
        &format!("[wasm] all {total} tensors loaded, building model...").into(),
    );

    // ── 5. Build VarBuilder + model ────────────────────────────────
    let vb = CandleVarBuilder::from_tensors(tensors, DType::F32, &device);
    let cfg = default_gemma_e2b_config();
    let provider =
        CandleLlmProvider::from_vb_on_device(&model_id, vb, tokenizer_json, &device, &cfg)
            .map_err(|e| {
                JsValue::from_str(&format!("init_local_model_chunked failed: {e}"))
            })?;

    let device_type = match provider.device() {
        Device::Cpu => "cpu",
        _ => "webgpu",
    };
    web_sys::console::log_1(
        &format!("[wasm] model ready on {device_type}").into(),
    );

    Ok(LocalModelHandle {
        inner: Arc::new(provider),
    })
}

/// Streaming chat parameters accepted from JS. Mirrors a useful subset of
/// [`ChatOptions`]; cloud-only fields like `cache_strategy` are intentionally
/// omitted because they do not apply to a local Gemma run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StreamParams {
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    system: Option<String>,
}

/// Wire-format chunk emitted into the [`ReadableStream`]. Encoded as
/// JSON+UTF-8 and pushed as a `Uint8Array` so the JS side can use a single
/// `TextDecoder` to parse line-delimited frames.
#[derive(Debug, Clone, Default, Serialize)]
struct WireChunk<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<WireUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    finished: bool,
}

#[derive(Debug, Clone, Serialize)]
struct WireUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Drive a streaming chat against a loaded [`LocalModelHandle`].
///
/// `messages_json` is a JSON array of `brainwires_core::message::Message`.
/// `params_json` is `{ max_tokens?, temperature?, top_p?, system? }`.
///
/// Returns a `ReadableStream<Uint8Array>` where each chunk is one
/// JSON-serialized `WireChunk` followed by a `\n` (NDJSON).
#[wasm_bindgen]
pub fn local_chat_stream(
    handle: &LocalModelHandle,
    messages_json: String,
    params_json: String,
) -> Result<ReadableStream, JsValue> {
    let messages: Vec<Message> = serde_json::from_str(&messages_json)
        .map_err(|e| JsValue::from_str(&format!("messages_json parse error: {e}")))?;
    let params: StreamParams = if params_json.trim().is_empty() {
        StreamParams::default()
    } else {
        serde_json::from_str(&params_json)
            .map_err(|e| JsValue::from_str(&format!("params_json parse error: {e}")))?
    };

    let provider = handle.inner.clone();

    // Build a JS underlying-source object with a `start(controller)` callback.
    // We capture the controller, kick off the candle stream on
    // `spawn_local`, and forward chunks as they come.
    let underlying = Object::new();
    let start_cb = Closure::once_into_js(move |controller: JsValue| {
        let controller: ReadableStreamDefaultController = match controller.dyn_into() {
            Ok(c) => c,
            Err(_) => return,
        };
        spawn_local(run_stream(provider, messages, params, controller));
    });
    Reflect::set(&underlying, &JsValue::from_str("start"), &start_cb)
        .map_err(|_| JsValue::from_str("failed to set ReadableStream start callback"))?;

    ReadableStream::new_with_underlying_source(&underlying)
}

/// Runs the candle stream and pushes encoded NDJSON frames into the
/// `ReadableStreamDefaultController`. Errors are surfaced both as a final
/// `{error: "..."}` frame and via `controller.error_with_e`.
async fn run_stream(
    provider: Arc<CandleLlmProvider>,
    messages: Vec<Message>,
    params: StreamParams,
    controller: ReadableStreamDefaultController,
) {
    let options = build_options(&params);
    let mut stream = provider.stream_chat(&messages, None, &options);

    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => match chunk {
                StreamChunk::Text(t) => {
                    enqueue_chunk(
                        &controller,
                        &WireChunk {
                            delta: Some(&t),
                            ..Default::default()
                        },
                    );
                }
                StreamChunk::Usage(u) => {
                    enqueue_chunk(
                        &controller,
                        &WireChunk {
                            usage: Some(WireUsage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            }),
                            ..Default::default()
                        },
                    );
                }
                StreamChunk::Done => {
                    enqueue_chunk(
                        &controller,
                        &WireChunk {
                            finished: true,
                            ..Default::default()
                        },
                    );
                }
                // Tool-related chunks and context-compaction notices are not
                // produced by the candle local provider today; ignore rather
                // than fail so future framework extensions don't break this
                // consumer.
                StreamChunk::ToolUse { .. }
                | StreamChunk::ToolInputDelta { .. }
                | StreamChunk::ToolCall { .. }
                | StreamChunk::ContextCompacted { .. } => {}
            },
            Err(e) => {
                let msg = format!("local_chat_stream error: {e}");
                enqueue_chunk(
                    &controller,
                    &WireChunk {
                        error: Some(msg.clone()),
                        finished: true,
                        ..Default::default()
                    },
                );
                controller.error_with_e(&JsValue::from_str(&msg));
                return;
            }
        }
    }

    let _ = controller.close();
}

fn build_options(params: &StreamParams) -> ChatOptions {
    let mut opts = ChatOptions::new();
    if let Some(t) = params.temperature {
        opts = opts.temperature(t);
    }
    if let Some(p) = params.top_p {
        opts = opts.top_p(p);
    }
    if let Some(m) = params.max_tokens {
        opts = opts.max_tokens(m);
    }
    if let Some(s) = params.system.as_deref() {
        opts = opts.system(s);
    }
    opts
}

fn enqueue_chunk(controller: &ReadableStreamDefaultController, chunk: &WireChunk<'_>) {
    let mut bytes = match serde_json::to_vec(chunk) {
        Ok(b) => b,
        Err(_) => return,
    };
    bytes.push(b'\n');
    let view = Uint8Array::from(bytes.as_slice());
    let _ = controller.enqueue_with_chunk(&view);
}

// ---------------------------------------------------------------------------
// Prompt utilities
// ---------------------------------------------------------------------------

/// Approximate token count for `text`.
///
/// We do not currently route through a loaded tokenizer — `CandleLlmProvider`
/// does not expose its `Tokenizer` publicly, and adding `tiktoken-rs` just
/// for an estimate is overkill. The `_model_id` argument is reserved so a
/// future per-model override can be added without changing the JS surface.
#[wasm_bindgen]
pub fn count_tokens(_model_id: String, text: String) -> Result<usize, JsValue> {
    // Coarse estimate: ~4 bytes per token. Good enough for UI guardrails.
    Ok(text.len().div_ceil(4))
}

/// Render a chat history into a single prompt string.
///
/// TODO: route through the Gemma chat template (`<start_of_turn>` markers)
/// once the framework exposes a public formatter on `CandleLlmProvider`.
/// For now we emit a plain `<role>: <content>\n` join, which is fine for the
/// "preview the prompt" UI affordance the PWA needs today.
#[wasm_bindgen]
pub fn format_prompt(_model_id: String, messages_json: String) -> Result<String, JsValue> {
    let messages: Vec<Message> = serde_json::from_str(&messages_json)
        .map_err(|e| JsValue::from_str(&format!("messages_json parse error: {e}")))?;
    let mut buf = String::new();
    for msg in &messages {
        let role = match msg.role {
            brainwires_core::message::Role::User => "user",
            brainwires_core::message::Role::Assistant => "assistant",
            brainwires_core::message::Role::System => "system",
            brainwires_core::message::Role::Tool => "tool",
        };
        buf.push_str(role);
        buf.push_str(": ");
        buf.push_str(&msg.text_or_summary());
        buf.push('\n');
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Voice exports
// ---------------------------------------------------------------------------

/// Wraps `brainwires_provider_speech::web_speech::tts::WebSpeechTts`.
#[wasm_bindgen]
pub struct WebTts {
    inner: WebSpeechTts,
}

#[derive(Serialize)]
struct VoiceJs {
    uri: String,
    name: String,
    lang: String,
    default: bool,
    local_service: bool,
}

#[wasm_bindgen]
impl WebTts {
    /// Construct a TTS handle bound to `window.speechSynthesis`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebTts, JsValue> {
        Ok(WebTts {
            inner: WebSpeechTts::new()?,
        })
    }

    /// Queue an utterance for playback. Optional knobs match the underlying
    /// `SpeechSynthesisUtterance` fields.
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn speak(
        &self,
        text: String,
        voice_uri: Option<String>,
        rate: Option<f32>,
        pitch: Option<f32>,
        volume: Option<f32>,
        lang: Option<String>,
    ) -> Result<(), JsValue> {
        let opts = WebSpeechTtsOptions {
            voice_uri,
            lang,
            rate,
            pitch,
            volume,
        };
        self.inner.speak(&text, opts)
    }

    /// Cancel any pending and currently-spoken utterances.
    #[wasm_bindgen]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Pause the current utterance.
    #[wasm_bindgen]
    pub fn pause(&self) {
        self.inner.pause();
    }

    /// Resume a paused utterance.
    #[wasm_bindgen]
    pub fn resume(&self) {
        self.inner.resume();
    }

    /// True if currently speaking.
    #[wasm_bindgen(js_name = isSpeaking)]
    pub fn is_speaking(&self) -> bool {
        self.inner.is_speaking()
    }

    /// True if currently paused.
    #[wasm_bindgen(js_name = isPaused)]
    pub fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// Snapshot of currently-available voices, serialized as a JS array of
    /// `{ uri, name, lang, default, local_service }`.
    #[wasm_bindgen]
    pub fn voices(&self) -> Result<JsValue, JsValue> {
        let voices: Vec<VoiceJs> = self
            .inner
            .voices()
            .into_iter()
            .map(|v| VoiceJs {
                uri: v.voice_uri,
                name: v.name,
                lang: v.lang,
                default: v.default,
                local_service: v.local_service,
            })
            .collect();
        serde_wasm_bindgen::to_value(&voices)
            .map_err(|e| JsValue::from_str(&format!("voices serialize failed: {e}")))
    }
}

/// Wraps `brainwires_provider_speech::web_speech::stt::WebSpeechStt`.
///
/// Holds the inner recognizer plus retains the JS callback closures for the
/// life of the recognition session so they aren't dropped while in flight.
#[wasm_bindgen]
pub struct WebStt {
    inner: WebSpeechStt,
}

#[wasm_bindgen]
impl WebStt {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebStt, JsValue> {
        Ok(WebStt {
            inner: WebSpeechStt::new()?,
        })
    }

    /// Configure the result callback and start recognition.
    ///
    /// `on_result` is invoked as `(text: string, is_final: bool, confidence: number)`
    /// per recognized result. The closure is kept alive by the framework's
    /// `WebSpeechStt` for the session.
    #[wasm_bindgen]
    pub fn start(
        &self,
        lang: Option<String>,
        continuous: bool,
        interim: bool,
        max_alternatives: u32,
        on_result: Function,
    ) -> Result<(), JsValue> {
        let cb = on_result;
        self.inner.on_result(move |r| {
            let this = JsValue::NULL;
            let _ = cb.call3(
                &this,
                &JsValue::from_str(&r.text),
                &JsValue::from_bool(r.is_final),
                &JsValue::from_f64(r.confidence as f64),
            );
        });

        let opts = WebSpeechSttOptions {
            lang,
            continuous,
            interim_results: interim,
            max_alternatives: if max_alternatives == 0 {
                None
            } else {
                Some(max_alternatives)
            },
        };
        self.inner.start(opts)
    }

    /// Register a JS callback fired on recognition errors. Receives
    /// `(error: string, message: string | null)`.
    #[wasm_bindgen(js_name = setOnError)]
    pub fn set_on_error(&self, cb: Function) {
        self.inner.on_error(move |e| {
            let this = JsValue::NULL;
            let msg = match &e.message {
                Some(s) => JsValue::from_str(s),
                None => JsValue::NULL,
            };
            let _ = cb.call2(&this, &JsValue::from_str(&e.error), &msg);
        });
    }

    /// Register a JS callback fired when recognition ends.
    #[wasm_bindgen(js_name = setOnEnd)]
    pub fn set_on_end(&self, cb: Function) {
        self.inner.on_end(move || {
            let this = JsValue::NULL;
            let _ = cb.call0(&this);
        });
    }

    /// Stop recognition gracefully.
    #[wasm_bindgen]
    pub fn stop(&self) {
        self.inner.stop();
    }

    /// Abort recognition immediately, dropping any pending result.
    #[wasm_bindgen]
    pub fn abort(&self) {
        self.inner.abort();
    }
}

// ---------------------------------------------------------------------------
