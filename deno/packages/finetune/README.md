# @rullama/finetune

Cloud fine-tuning orchestration: upload a dataset, start a job, poll it to
completion, and manage the resulting models.

```sh
deno add jsr:@rullama/finetune
```

## What ships

- **Types** — `TrainingJobId`, `DatasetId`, `TrainingJobStatus`,
  `TrainingProgress`, `TrainingMetrics`, `TrainingJobSummary`, plus the helpers
  `isTerminal` / `isRunning` / `isSucceeded` / `completionFraction`.
- **Config** — `TrainingHyperparams`, `LoraConfig`, `AdapterMethod` (`lora` /
  `qlora` / `dora` / `qdora`), `AlignmentMethod` (`none` / `dpo` / `orpo`) with
  `default*` constructors.
- **Providers** — `OpenAiFineTune`, `TogetherFineTune`, `FireworksFineTune`.
  Each implements the same `FineTuneProvider` interface: `uploadDataset`,
  `createJob`, `getJobStatus`, `cancelJob`, `listJobs`, `deleteModel`.
- **JobPoller** — exponential-backoff polling loop (15 s → 5 min cap, 1.5×,
  4-hour ceiling by default). Throws a `timeout` `TrainingError` when the
  ceiling passes before the job reaches a terminal state.
- **TrainingManager** — registry keyed by provider name + `startCloudJob` /
  `waitForCloudJob` / `checkCloudJob` / `cancelCloudJob` / `listAllCloudJobs`.
- **TrainingError** — every failure carries a `kind` (`api`, `provider`,
  `upload`, `job_not_found`, `validation`, `timeout`, …) and, for HTTP failures,
  the `status_code`.

## Example

```ts
import {
  defaultHyperparams,
  newCloudFineTuneConfig,
  OpenAiFineTune,
  TrainingError,
  TrainingManager,
} from "@rullama/finetune";

const mgr = new TrainingManager();
mgr.addCloudProvider(new OpenAiFineTune(Deno.env.get("OPENAI_API_KEY")!));
const openai = mgr.getCloudProvider("openai")!;

// 1. Upload a JSONL dataset (chat-format lines, one example per line).
const dataset = await openai.uploadDataset(
  await Deno.readFile("./train.jsonl"),
  "jsonl",
);

// 2. Configure the job — defaults, then override what you need.
const cfg = {
  ...newCloudFineTuneConfig("gpt-4o-mini-2024-07-18", dataset),
  hyperparams: { ...defaultHyperparams(), epochs: 2 },
  suffix: "support-bot",
};

// 3. Start it and wait, polling with exponential backoff (15 s → 5 min).
const jobId = await mgr.startCloudJob("openai", cfg);
try {
  const final = await mgr.waitForCloudJob("openai", jobId, {
    initial_interval_ms: 15_000,
    max_interval_ms: 300_000,
    backoff_multiplier: 1.5,
    timeout_ms: 2 * 60 * 60 * 1000,
    on_status: (s) => console.log("status:", s.status),
  });
  if (final.status === "succeeded") {
    console.log("fine-tuned model:", final.model_id);
  } else {
    console.log("job ended:", final); // failed | cancelled
  }
} catch (err) {
  if (err instanceof TrainingError && err.kind === "timeout") {
    await mgr.cancelCloudJob("openai", jobId);
  }
  throw err;
}
```

## Provider notes

| provider    | LoRA fields forwarded                  | DPO                                    | ORPO                            |
| ----------- | -------------------------------------- | -------------------------------------- | ------------------------------- |
| `openai`    | none (ignored)                         | yes (`method.type = "dpo"`)            | rejected (`validation` error)   |
| `together`  | `lora_r`, `lora_alpha`, `lora_dropout` | yes (`training_method.method = "dpo"`) | rejected (`validation` error)   |
| `fireworks` | `lora_rank`                            | no — alignment is not forwarded        | no — alignment is not forwarded |

- `uploadDataset` on Together and Fireworks always sends the bytes as
  `training_data.jsonl`; only OpenAI honours the `format` argument (`jsonl` /
  `parquet` / `csv`).
- `listJobs` returns `metrics: null` for every provider; Fireworks additionally
  reports `created_at` as the listing time.
- OpenAI takes a learning-rate _multiplier_; `hyperparams.learning_rate` is
  divided by OpenAI's `2e-5` base, so the default maps to `1.0`.

## Intentionally not ported

- **Bedrock** (AWS) and **Vertex** (Google) — both need the vendor SDKs for
  request signing; implement `FineTuneProvider` directly if you need them. The
  interface is stable. `DatasetId.fromS3Uri` / `fromGcsUri` exist for such
  implementations.
- **Anyscale, `cost.rs`** — niche; add later as needed.
- **Local training** (Burn-based fine-tuning, quantization, LR schedules,
  checkpointing, weight loading, alignment training, architectures/adapters) —
  stays Rust-side. Deno consumers that want on-device training should drive the
  Rust binary instead.
- **`datasets/` subtree** (JSONL validation, tokenizer, sampling, quality
  checks) — callers construct the JSONL themselves and upload via
  `uploadDataset`.

## Equivalent Rust crate

`rullama-training` with the `cloud` feature enabled. The TrainingJobId /
TrainingJobStatus / hyperparameter shapes are semantically 1:1.
