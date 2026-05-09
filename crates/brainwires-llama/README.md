# brainwires-llama

Gemma 4 inference engine: pure Rust + wgpu, runs natively and in the browser via wasm-pack. Loads Ollama GGUF blobs directly, dispatches forward passes through hand-written WGSL kernels, never reaches a remote server.

## Origin

Source ported in from the standalone [rullama](https://github.com/Brainwires/rullama) prototype (~5 hours of greenfield work outside the framework, validated bit-identical to Ollama on `gemma4:e2b` greedy decode). The standalone repo remains intact as the upstream reference; this crate is the framework-internal copy consumed by `extras/brainwires-chat-pwa/wasm`.

## Scope

- ✅ `gemma4:e2b` (Q4_K_M, ~7 GB GGUF)
- ✅ `gemma4:e4b` (shape-compatible, untested)
- ❌ MoE variants (`gemma4:26b`, `gemma4:31b`)
- ❌ Other architectures (llama, mistral, qwen, phi)
- ❌ Vision / audio multimodal towers (chat-pwa keeps those on candle for now)

## Public API

`brainwires_llama::Model`:

- `Model::load_streaming(fetcher)` — async load via any [`gguf::TensorFetcher`] impl
- `Model::load_native(bytes)` — for in-memory GGUF (native tests; doesn't fit gemma4:e2b in wasm32)
- `Model::encode_tokens(text)` / `Model::token_str_native(id)` — Ollama-bit-exact BPE
- `Model::render_chat_native(messages, with_bos)` — Gemma 4 chat template
- `Model::step_native(token_id)` — feed one token, return sampled next id, advance KV
- `Model::is_eos_native(id)` / `Model::reset_native()` / `Model::set_sampling_native(opts)`

In `wasm32` builds the same surface is exposed via `wasm-bindgen` with camel-cased names (`load`, `loadFromUrl`, `encode`, `step`, `tokenStr`, `renderChat`, `setSampling`, etc.).

## Layout

```
src/
├── api.rs           # public Model + ChatMessage + GenerateOptions
├── backend/         # wgpu context, pipelines, weight cache, dispatch
├── gguf/            # GGUF v3 reader, TensorFetcher trait + InMemory + HttpRange
├── kernels/wgsl/    # 13 hand-written compute shaders
├── model/config.rs  # Gemma4Config from GGUF metadata
├── reference/       # CPU oracle (f32) + chained GPU forward (one encoder/token)
├── sampling.rs      # temperature, top-k, top-p, rep penalty
├── template/        # Gemma 4 chat-template renderer
└── tokenizer/       # GGUF BPE (Ollama-bit-exact)
```

## License

Dual-licensed under MIT or Apache 2.0, matching the framework. Origin attributions retained in `LICENSE-MIT` and `LICENSE-APACHE`.
