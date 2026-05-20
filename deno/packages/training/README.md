# @brainwires/training

Cloud fine-tuning orchestration.

## What ships

- **Types** — `TrainingJobId`, `DatasetId`, `TrainingJobStatus`,
  `TrainingProgress`, `TrainingMetrics`, `TrainingJobSummary`.
- **Config** — `TrainingHyperparams`, `LoraConfig`, `AdapterMethod`
  (`lora` / `qlora` / `dora` / `qdora`), `AlignmentMethod`
  (`none` / `dpo` / `orpo`).
- **Providers** — `OpenAiFineTune`, `TogetherFineTune`, `FireworksFineTune`.
  Each implements the same `FineTuneProvider` interface:
  `uploadDataset`, `createJob`, `getJobStatus`, `cancelJob`, `listJobs`,
  `deleteModel`.
- **JobPoller** — exponential-backoff polling loop (15s → 5min cap, 1.5×,
  4-hour ceiling by default).
- **TrainingManager** — registry + orchestration across providers.

## Intentionally not ported

- **Bedrock** (AWS) and **Vertex** (Google) — both need the vendor SDKs for
  request signing; implement `FineTuneProvider` directly if you need them.
  The interface is stable.
- **Anyscale, `cost.rs`** — niche; add later as needed.
- **Local training** (Burn-based fine-tuning, quantization, LR schedules,
  checkpointing, weight loading, alignment training, architectures/adapters)
  — stays Rust-side. Deno consumers that want on-device training should
  drive the Rust binary instead.
- **`datasets/` subtree** (JSONL validation, tokenizer, sampling, quality
  checks) — callers construct the JSONL themselves and upload via
  `uploadDataset`.

## Example

```ts
import {
  DatasetId,
  defaultHyperparams,
  JobPoller,
  newCloudFineTuneConfig,
  OpenAiFineTune,
  TrainingManager,
} from "@brainwires/training";

const mgr = new TrainingManager();
mgr.addCloudProvider(new OpenAiFineTune(Deno.env.get("OPENAI_API_KEY")!));

const upload = await mgr.getCloudProvider("openai")!.uploadDataset(
  await Deno.readFile("./train.jsonl"),
  "jsonl",
);
const cfg = {
  ...newCloudFineTuneConfig("gpt-4o-mini-2024-07-18", upload),
  hyperparams: { ...defaultHyperparams(), epochs: 2 },
};

const job = await mgr.startCloudJob("openai", cfg);
const final = await mgr.waitForCloudJob("openai", job);
console.log(final);
```

## Equivalent Rust crate

`brainwires-training` with the `cloud` feature enabled. The TrainingJobId /
TrainingJobStatus / hyperparameter shapes are semantically 1:1.
