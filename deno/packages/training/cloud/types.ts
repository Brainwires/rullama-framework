/**
 * Cloud fine-tuning provider interface.
 *
 * Equivalent to Rust's `brainwires_training::cloud::mod` types.
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

/** Configuration for a cloud fine-tuning job. */
export interface CloudFineTuneConfig {
  base_model: string;
  training_dataset: DatasetId;
  validation_dataset: DatasetId | null;
  hyperparams: TrainingHyperparams;
  lora: LoraConfig | null;
  alignment: AlignmentMethod;
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
  readonly name: string;
  supportedBaseModels(): string[];
  supportsDpo(): boolean;

  uploadDataset(data: Uint8Array, format: DataFormat): Promise<DatasetId>;
  createJob(config: CloudFineTuneConfig): Promise<TrainingJobId>;
  getJobStatus(job_id: TrainingJobId): Promise<TrainingJobStatus>;
  cancelJob(job_id: TrainingJobId): Promise<void>;
  listJobs(): Promise<TrainingJobSummary[]>;
  deleteModel(model_id: string): Promise<void>;
}
