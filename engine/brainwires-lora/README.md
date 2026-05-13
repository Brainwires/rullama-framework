# brainwires-finetune-local

[![Crates.io](https://img.shields.io/crates/v/brainwires-finetune-local.svg)](https://crates.io/crates/brainwires-finetune-local)
[![Documentation](https://docs.rs/brainwires-finetune-local/badge.svg)](https://docs.rs/brainwires-finetune-local)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Local PEFT (parameter-efficient fine-tuning) for the Brainwires Agent
Framework. LoRA / QLoRA / DoRA on a pre-trained model, Burn-backed
(CPU + WGPU), with safetensors checkpointing.

These are **fine-tuning** methods — adapting a pre-trained model with
additional training data — not training-from-scratch. The naming pair:

- [`brainwires-finetune`](https://crates.io/crates/brainwires-finetune) — cloud fine-tune APIs (OpenAI / Anthropic / Bedrock / Vertex AI / etc.)
- **`brainwires-finetune-local`** (this crate) — local PEFT
- [`brainwires-training`](https://crates.io/crates/brainwires-training) — placeholder for actual training-from-scratch (no code yet)

## What lives here

- `local::adapters::{LoraLayer, QLoraLayer, DoraLayer}` — adapter layers.
- `local::alignment::{dpo, orpo}` — preference-alignment losses.
- `local::burn_backend` — training loop on Burn (autodiff + WGPU).
- `local::checkpointing::CheckpointManager` — safetensors checkpoint
  save / load.
- `local::dataset_loader` — tokenizer + dataset wrappers.
- `local::weight_loader::SafeTensorsLoader` — model weight loading.
- `local::quantization` — int8 / int4 quantisation helpers.

## Heavy deps

This crate pulls `burn-core`, `burn-nn`, `burn-optim`, `burn-autodiff`,
`burn-wgpu`, `burn-ndarray`, `tokenizers`, `safetensors`. Consumers that
only want cloud fine-tune APIs should depend on `brainwires-finetune`
directly.

## Usage

```toml
[dependencies]
brainwires-finetune-local = "0.10"
```

```rust,ignore
use brainwires_finetune_local::{LocalTrainingConfig, BurnBackend};
use brainwires_finetune::config::{AdapterMethod, LoraConfig};

let config = LocalTrainingConfig {
    adapter: AdapterMethod::Lora(LoraConfig::default()),
    // ...
};
```

## See also

- [`brainwires-finetune`](https://crates.io/crates/brainwires-finetune) —
  cloud fine-tune APIs + dataset pipelines (this crate depends on it for
  shared `config` / `error` / `types`).
- [`brainwires`](https://crates.io/crates/brainwires) — umbrella facade
  with `training-local` feature.

## License

MIT OR Apache-2.0
