/**
 * Training types shared across cloud providers.
 *
 * Equivalent to Rust's `brainwires_training::types` module.
 */

/** Unique identifier for a training job. */
export class TrainingJobId {
  readonly value: string;
  constructor(value: string) {
    this.value = value;
  }
  toString(): string {
    return this.value;
  }
}

/** Unique identifier for an uploaded dataset. */
export class DatasetId {
  readonly value: string;
  constructor(value: string) {
    this.value = value;
  }
  static fromS3Uri(uri: string): DatasetId {
    return new DatasetId(uri);
  }
  static fromGcsUri(uri: string): DatasetId {
    return new DatasetId(uri);
  }
  toString(): string {
    return this.value;
  }
}

/** Progress information for a running training job. */
export interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  train_loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  /** Elapsed time in seconds. */
  elapsed_secs: number;
}

export function defaultProgress(): TrainingProgress {
  return {
    epoch: 0,
    total_epochs: 0,
    step: 0,
    total_steps: 0,
    train_loss: null,
    eval_loss: null,
    learning_rate: null,
    elapsed_secs: 0,
  };
}

export function completionFraction(p: TrainingProgress): number {
  return p.total_steps === 0 ? 0 : p.step / p.total_steps;
}

/** Status of a training job. */
export type TrainingJobStatus =
  | { status: "pending" }
  | { status: "validating" }
  | { status: "queued" }
  | { status: "running"; progress: TrainingProgress }
  | { status: "succeeded"; model_id: string }
  | { status: "failed"; error: string }
  | { status: "cancelled" };

export function isTerminal(s: TrainingJobStatus): boolean {
  return s.status === "succeeded" || s.status === "failed" || s.status === "cancelled";
}

export function isRunning(s: TrainingJobStatus): boolean {
  return s.status === "running";
}

export function isSucceeded(s: TrainingJobStatus): boolean {
  return s.status === "succeeded";
}

/** Metrics from a completed training job. */
export interface TrainingMetrics {
  final_train_loss: number | null;
  final_eval_loss: number | null;
  total_steps: number;
  total_epochs: number;
  total_tokens_trained: number | null;
  duration_secs: number;
  estimated_cost_usd: number | null;
}

export function defaultMetrics(): TrainingMetrics {
  return {
    final_train_loss: null,
    final_eval_loss: null,
    total_steps: 0,
    total_epochs: 0,
    total_tokens_trained: null,
    duration_secs: 0,
    estimated_cost_usd: null,
  };
}

/** Summary of a training job for listing. */
export interface TrainingJobSummary {
  job_id: TrainingJobId;
  provider: string;
  base_model: string;
  status: TrainingJobStatus;
  /** ISO 8601. */
  created_at: string;
  metrics: TrainingMetrics | null;
}
