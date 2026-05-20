# brainwires-finetune

[![Crates.io](https://img.shields.io/crates/v/brainwires-finetune.svg)](https://crates.io/crates/brainwires-finetune)
[![Documentation](https://docs.rs/brainwires-finetune/badge.svg)](https://docs.rs/brainwires-finetune)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Cloud fine-tune APIs and dataset pipelines for Brainwires agents.

Fine-tuning surface:

- **`brainwires-finetune`** (this crate) — cloud fine-tune APIs (OpenAI / Anthropic / Together / Fireworks / Anyscale / Bedrock / Vertex AI) plus dataset pipelines.
- **`rullama-finetune`** (sibling `rullama` workspace) — local PEFT (LoRA / QLoRA / DoRA), Burn-backed. (Lived in this workspace as a separate local-PEFT crate prior to v0.11; moved out alongside the rest of the wgpu inference engine.)
- **`rullama-training`** (sibling `rullama` workspace) — placeholder for actual training-from-scratch.

## What lives here

- `manager::TrainingManager` — dispatches fine-tune jobs to whichever
  provider implements `FineTuneProvider`.
- `cloud::FineTuneProvider` + `FineTuneProviderFactory` — provider-agnostic
  trait + factory.
- `cloud::providers` (one module per cloud API) — concrete impls.
- `config` — hyperparameter / adapter / alignment-method types shared with
  `rullama-finetune`.
- `datasets` — JSONL / format conversion / tokenization / dedup
  (absorbed from the deprecated `brainwires-datasets` crate).
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
brainwires-finetune = "0.11"
```

```rust,ignore
use brainwires_finetune::{TrainingManager, CloudFineTuneConfig};

let manager = TrainingManager::new(/* ... */);
let job = manager.submit(CloudFineTuneConfig { /* ... */ }).await?;
```

## See also

- `rullama-finetune` (sibling `rullama` workspace) — local PEFT (LoRA /
  QLoRA / DoRA), Burn-backed. Reuses this crate's shared `config` / `error` /
  `types` modules.
- [`brainwires-provider`](https://crates.io/crates/brainwires-provider) — LLM chat clients (separate crate).
- [`brainwires`](https://crates.io/crates/brainwires) — umbrella facade
  with `training` / `training-cloud` features (cloud only since v0.11).

## License

MIT OR Apache-2.0
