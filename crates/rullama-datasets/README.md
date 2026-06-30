# rullama-datasets

**Training-data pipelines for [rullama](https://github.com/Brainwires/rullama-framework).**

JSONL I/O, tokenization, deduplication, format conversion, and quality stats —
the shared dataset toolkit consumed by both cloud fine-tuning (`rullama-finetune`)
and local LoRA training (`rullama-lora`).

## Features

- **JSONL I/O** — streaming read/write of instruction and preference records.
- **Format conversion** — Alpaca, ChatML, OpenAI, ShareGPT, and Together
  formats, in and out.
- **Tokenization** — pluggable tokenizer trait for length stats and packing.
- **Deduplication** — exact and near-duplicate removal.
- **Quality stats** — length distributions, field coverage, and dataset health
  summaries.

## API

```rust
use rullama_datasets::{Dataset, InstructDataset, PreferenceDataset};
```

Core modules: `dataset`, `format`, `jsonl`, `quality`, `sampling`, `tokenizer`,
`types`, `error`.

## Feature flags

| Feature | Enables |
|---|---|
| `datasets-hf-tokenizer` | HuggingFace `tokenizers`-backed tokenization |
| `datasets-tiktoken` | `tiktoken-rs`-backed tokenization |
| `datasets-dedup` | hashing + near-dup detection |
| `datasets-full` | all of the above |

## License

MIT OR Apache-2.0.
