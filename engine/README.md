# brainwires engine

Browser-resident AI runtime in pure Rust → WebAssembly + WebGPU. Loads Ollama's
on-disk GGUF blobs (no server) and runs the forward pass on the local GPU through
hand-written WGSL.

This is an **isolated sub-workspace** inside the [brainwires](../) platform repo,
excluded from the root workspace so the harness's native build never pulls in
`wgpu` and the engine's wasm build never pulls in the harness.

| Crate | What it is |
|---|---|
| [`brainwires-engine`](brainwires-engine) | The inference engine — model load/streaming, forward pass + WGSL kernels, sampling, KV cache, tokenizer/template, LoRA inference, vision/audio, diffusion, image-gen. |
| [`brainwires-lora`](brainwires-lora) | Local LoRA fine-tuning over the same wgpu kernels. Native + `wasm32`; exposes the `TrainingSession` wasm-bindgen surface the PWA's Fine-tune tab consumes. |

The engine handles **tokens**; the harness handles **turns**. See
[`../docs/ARCHITECTURE-engine-harness.md`](../docs/ARCHITECTURE-engine-harness.md).

## Build

```sh
# native parity / smoke (consumes an Ollama GGUF blob path)
cargo run -p brainwires-engine --release --example greedy_parity -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" 5

# unified wasm bundle (inference Model + training TrainingSession).
# --out-name rullama keeps the PWA's /pkg/rullama.js import stable.
wasm-pack build brainwires-lora --target web --release \
    --out-dir ../../pkg --out-name rullama

# inference-only wasm bundle (smaller; no TrainingSession)
wasm-pack build brainwires-engine --target web --release \
    --out-dir ../../pkg --out-name rullama
```

Native consumers (the rullama app's devserver, rullama-native, anything else)
reach the engine through an OpenAI-compatible `POST /v1/chat/completions` endpoint
— see the `serve` bin in `brainwires-engine`.
