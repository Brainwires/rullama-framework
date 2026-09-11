/**
 * Cloud fine-tuning provider interface.
 *
 * Equivalent to Rust's `rullama_training::cloud::mod` types.
 */

import type {
  AlignmentMethod,
  LoraConfig,
  TrainingHyperparams,
} from "../config.ts";
import { defaultAlignment, defaultHyperparams } from "../config.ts";
import type {
  DatasetId,
  TrainingJobId,
  TrainingJobStatus,
  TrainingJobSummary,
} from "../types.ts";

/** Dataset format for uploads. */
export type DataFormat = "jsonl" | "parquet" | "csv";

/** Upload MIME type and file name for a dataset format. */
export function datasetFile(
  format: DataFormat,
): { mime: string; name: string } {
  switch (format) {
    case "parquet":
      return {
        mime: "application/vnd.apache.parquet",
        name: "training_data.parquet",
      };
    case "csv":
      return { mime: "text/csv", name: "training_data.csv" };
    default:
      return { mime: "application/jsonl", name: "training_data.jsonl" };
  }
}

/** Configuration for a cloud fine-tuning job. */
export interface CloudFineTuneConfig {
  /** Provider model identifier to fine-tune (see `FineTuneProvider.supportedBaseModels`). */
  base_model: string;
  /** Id returned by `uploadDataset` for the training set. */
  training_dataset: DatasetId;
  /** Optional held-out validation set; `null` to skip validation. */
  validation_dataset: DatasetId | null;
  /** Training hyperparameters; providers forward the subset their API accepts. */
  hyperparams: TrainingHyperparams;
  /** LoRA settings, or `null` for full fine-tuning (OpenAI ignores this field). */
  lora: LoraConfig | null;
  /** Alignment stage; `{ kind: "none" }` for plain SFT. */
  alignment: AlignmentMethod;
  /** Provider-specific name suffix / output-model name, or `null` for the default. */
  suffix: string | null;
}

/** Create a config with defaults. */
export function newCloudFineTuneConfig(
  base_model: string,
  training_dataset: DatasetId,
): CloudFineTuneConfig {
  return {
    base_model,
    training_dataset,
    validation_dataset: null,
    hyperparams: defaultHyperparams(),
    lora: null,
    alignment: defaultAlignment(),
    suffix: null,
  };
}

/** Interface implemented by every cloud fine-tune backend. */
export interface FineTuneProvider {
  /** Stable provider key (`"openai"`, `"together"`, `"fireworks"`); used by `TrainingManager`. */
  readonly name: string;
  /** Base-model identifiers the provider is known to accept. */
  supportedBaseModels(): string[];
  /** Whether `createJob` accepts a `dpo` alignment method. */
  supportsDpo(): boolean;

  /** Upload raw dataset bytes; resolves to the id to put in `CloudFineTuneConfig`. */
  uploadDataset(data: Uint8Array, format: DataFormat): Promise<DatasetId>;
  /** Submit a fine-tuning job and return its id without waiting for completion. */
  createJob(config: CloudFineTuneConfig): Promise<TrainingJobId>;
  /** Fetch the job's current status; rejects with `job_not_found` on 404. */
  getJobStatus(job_id: TrainingJobId): Promise<TrainingJobStatus>;
  /** Request cancellation of a job. */
  cancelJob(job_id: TrainingJobId): Promise<void>;
  /** List the account's fine-tuning jobs. */
  listJobs(): Promise<TrainingJobSummary[]>;
  /** Delete a fine-tuned model by its provider model id. */
  deleteModel(model_id: string): Promise<void>;
}
