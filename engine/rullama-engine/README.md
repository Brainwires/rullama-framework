# rullama-engine

**Browser-resident Gemma 4 inference — pure Rust → WebAssembly + WebGPU.**

`rullama-engine` is the inference engine of the [rullama](https://github.com/Brainwires/rullama-framework)
platform. It loads Ollama's on-disk **GGUF** blobs and runs the Gemma 4 forward
pass on the local GPU via hand-written **WGSL** kernels — in the browser (wasm32 +
WebGPU) or natively (wgpu on Metal/Vulkan). Your data never leaves the device.

## What it does

- **Text generation** — Gemma 4 (`e2b`/`e4b`/…) forward pass, sampling
  (temperature / top-k / top-p / repeat penalty), chat-template rendering, and a
  GPU-resident KV cache.
- **Multimodal** — ViT vision tower + Conformer audio tower splice soft tokens
  into the prompt via `<|image>` / `<|audio>` sentinels.
- **Streaming load** — GGUF is read via HTTP byte-range *or* OPFS sync access
  handles, so a multi-GB model never enters wasm linear memory in bulk.
- **Also bundled** — embeddings (`EmbeddingModel`), diffusion text
  (`DiffusionGemma`), image generation, and TTS.

## Stable API

Three modules follow semver across `0.x` patch releases:

- [`api`] — the high-level `Model` handle (`loadFrom*` / `generate` / `stop`),
  `ChatMessage`, `ChatRole`, `GenerateOptions`. This is what `#[wasm_bindgen]`
  exposes to JS and what native Rust consumers should program against.
- [`error`] — `RullamaError` and `Result`.
- [`sampling`] — `SamplingOptions` and `Sampler`.

Everything else (`backend`, `gguf`, `kernels`, `model`, `multimodal`,
`reference`, `template`, `tokenizer`) is `#[doc(hidden)]` implementation detail,
reachable only so sibling crates (`rullama-lora`) can link the kernel set.

## Native serve binary

The `serve` feature builds **`rullama-serve`**, an OpenAI-compatible
`POST /v1/chat/completions` server hosting an engine `Model` on a dedicated
thread (the engine `Model` is `!Send`).

```sh
cargo run -p rullama-engine --features serve --bin rullama-serve
```

## Building the wasm bundle

The browser app consumes the engine as a wasm bundle. It is built from the
sibling `rullama-lora` crate (which re-exports the inference surface):

```sh
wasm-pack build rullama-lora --target web --release --out-name rullama
```

## License

MIT OR Apache-2.0.
