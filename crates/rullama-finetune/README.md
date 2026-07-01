# rullama-finetune

[![Crates.io](https://img.shields.io/crates/v/rullama-finetune.svg)](https://crates.io/crates/rullama-finetune)
[![Documentation](https://docs.rs/rullama-finetune/badge.svg)](https://docs.rs/rullama-finetune)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/rullama-framework)

Cloud fine-tune APIs for rullama agents. Dataset pipelines live in the
sibling `rullama-datasets` crate.

Fine-tuning surface:

- **`rullama-finetune`** (this crate) — cloud fine-tune APIs (OpenAI / Anthropic / Together / Fireworks / Anyscale / Bedrock / Vertex AI).
- **`rullama-datasets`** — dataset pipelines (JSONL I/O, tokenization, dedup, format conversion); extracted from this crate in v0.11 and re-exposed here via the `datasets-*` features.
- **`rullama-lora`** (sibling `rullama` workspace) — local PEFT (LoRA / QLoRA / DoRA), Burn-backed. (Lived in this workspace as a separate local-PEFT crate prior to v0.11; moved out alongside the rest of the wgpu inference engine.)
- **`rullama-training`** (sibling `rullama` workspace) — placeholder for actual training-from-scratch.

## What lives here

- `manager::TrainingManager` — dispatches fine-tune jobs to whichever
  provider implements `FineTuneProvider`.
- `cloud::FineTuneProvider` + `FineTuneProviderFactory` — provider-agnostic
  trait + factory.
- `cloud::providers` (one module per cloud API) — concrete impls.
- `config` — hyperparameter / adapter / alignment-method types shared with
  `rullama-lora`.
- `error::TrainingError`, `types::{TrainingJobId, TrainingJobStatus, ...}`
  — shared infrastructure.

## Features

| Feature | Default | Notes |
|---|---|---|
| `cloud` | yes | reqwest-based cloud provider clients |
| `bedrock` | no | AWS Bedrock fine-tune (sigv4) |
| `vertex` | no | Google Vertex AI (gcp_auth) |
| `datasets-hf-tokenizer` | no | HuggingFace tokenizers |
| `datasets-tiktoken` | no | OpenAI tiktoken |
| `datasets-dedup` | no | sha2 + rand for content dedup |
| `datasets-full` | no | All three datasets sub-features |
| `full` | no | `cloud + bedrock + vertex + datasets-full` |

## Usage

```toml
[dependencies]
rullama-finetune = "0.12"
```

```rust,ignore
use rullama_finetune::{TrainingManager, CloudFineTuneConfig};

let manager = TrainingManager::new(/* ... */);
let job = manager.submit(CloudFineTuneConfig { /* ... */ }).await?;
```

## See also

- [`rullama-datasets`](https://crates.io/crates/rullama-datasets) — dataset
  pipelines extracted from this crate (re-exposed via the `datasets-*` features).
- `rullama-lora` (sibling `rullama` workspace) — local PEFT (LoRA /
  QLoRA / DoRA), Burn-backed. Reuses this crate's shared `config` / `error` /
  `types` modules.
- [`rullama-provider`](https://crates.io/crates/rullama-provider) — LLM chat clients (separate crate).
- [`rullama`](https://crates.io/crates/rullama) — umbrella facade
  with `training` / `training-cloud` features (cloud only since v0.11).

## License

MIT OR Apache-2.0
