# rullama-lora

**Local LoRA fine-tuning for the rullama engine — native and in the browser.**

`rullama-lora` trains LoRA adapters against [`rullama-engine`](../rullama-engine)'s
live wgpu kernels — no Burn, no PyTorch, no separate runtime. The same trainer
compiles for native and `wasm32-unknown-unknown`; the forward, backward, LoRA
state, optimizer, and dataset parsing all run on both targets. The only
native-only pieces are filesystem helpers wrapping the bytes-based core API.

## Scope

- Rank-*r* LoRA on `attn_q` / `attn_k` / `attn_v` / `attn_o` and the FFN
  projections.
- Adam optimizer over GPU buffers, global L2 grad clipping, gradient
  accumulation, mixed precision, gradient checkpointing.
- `PerPosition` cross-entropy — a single-forward variant with a ~C/2 speedup
  over the multi-forward path.
- Backward kernels for matmul (Q4_K / Q6_K), rmsnorm, rope, geglu, attention,
  and cross-entropy.

## Module map

- `shared` — config / error / progress types.
- `dataset_loader` — JSONL parser (bytes-in core + native wrapper) + a
  `Tokenizer` trait with byte-level and HuggingFace-`tokenizers` impls.
- `lr_schedule` — warmup + linear / cosine / cosine-warm-restarts.
- `lora` — LoRA A/B state, forward correction, A/B grad accumulation.
- `scratch` — per-step GPU scratch buffers for the backward pass.
- `session` — `TrainingSession`, driving one step end-to-end
  (forward → loss → backward → Adam).

## API

```rust
use rullama_lora::TrainingSession;
// Trained adapters export as safetensors and load back into a
// rullama-engine `Model` via `Model::loadAdapter(bytes)`.
```

Bytes-based entry points (`load_jsonl_from_bytes`, `save_adapter_to_bytes`,
`load_adapter_into_state_from_bytes`) work identically on wasm.

## Browser bundle

`rullama-lora` is the crate wasm-packed into the app's `/pkg/rullama.js` bundle —
it re-exports the engine's inference surface (`Model`) alongside
`TrainingSession`, so one bundle does both inference and training:

```sh
wasm-pack build rullama-lora --target web --release --out-name rullama
```

## License

MIT OR Apache-2.0.
