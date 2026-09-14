/**
 * Training types shared across cloud providers.
 *
 * Equivalent to Rust's `rullama_training::types` module.
 */

/** Unique identifier for a training job. */
export class TrainingJobId {
  /** The provider-assigned job id string (e.g. OpenAI's `ftjob-…`). */
  readonly value: string;
  /** Wrap a provider-assigned job id string. */
  constructor(value: string) {
    this.value = value;
  }
  /** Return the raw job id string. */
  toString(): string {
    return this.value;
  }
}

/** Unique identifier for an uploaded dataset. */
export class DatasetId {
  /** The provider-side file id (e.g. OpenAI's `file-…`) or a storage URI. */
  readonly value: string;
  /** Wrap a provider file id or storage URI as a DatasetId. */
  constructor(value: string) {
    this.value = value;
  }
  /**
   * Wrap an `s3://` URI verbatim — no validation or transformation is
   * applied. Intended for providers that read datasets from S3 (Bedrock).
   */
  static fromS3Uri(uri: string): DatasetId {
    return new DatasetId(uri);
  }
  /**
   * Wrap a `gs://` URI verbatim — no validation or transformation is
   * applied. Intended for providers that read datasets from GCS (Vertex).
   */
  static fromGcsUri(uri: string): DatasetId {
    return new DatasetId(uri);
  }
  /** Return the raw file id / URI string. */
  toString(): string {
    return this.value;
  }
}

/** Progress information for a running training job. */
export interface TrainingProgress {
  /** Current epoch (0-based); `0` when the provider does not report epochs. */
  epoch: number;
  /** Planned number of epochs; `0` when unknown. */
  total_epochs: number;
  /** Current step — provider-specific unit (OpenAI reports trained tokens here). */
  step: number;
  /** Planned number of steps; `0` when unknown (makes {@link completionFraction} return 0). */
  total_steps: number;
  /** Most recent training loss, or `null` when the provider does not report it. */
  train_loss: number | null;
  /** Most recent evaluation loss, or `null` when not reported. */
  eval_loss: number | null;
  /** Current learning rate, or `null` when not reported. */
  learning_rate: number | null;
  /** Elapsed time in seconds. */
  elapsed_secs: number;
}

/** A zeroed progress record: no epochs/steps, all losses `null`. */
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

/**
 * Fraction of the job completed, as `step / total_steps`.
 *
 * @returns A value in `[0, 1]`, or `0` when `total_steps` is unknown (`0`).
 */
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

/** `true` for `succeeded`, `failed`, and `cancelled` — states a job never leaves. */
export function isTerminal(s: TrainingJobStatus): boolean {
  return s.status === "succeeded" || s.status === "failed" ||
    s.status === "cancelled";
}

/** `true` only for the `running` state (not `pending`/`queued`/`validating`). */
export function isRunning(s: TrainingJobStatus): boolean {
  return s.status === "running";
}

/** `true` when the job finished with a fine-tuned `model_id`. */
export function isSucceeded(s: TrainingJobStatus): boolean {
  return s.status === "succeeded";
}

/** Metrics from a completed training job. */
export interface TrainingMetrics {
  /** Training loss at the last step, or `null` when not reported. */
  final_train_loss: number | null;
  /** Evaluation loss at the last step, or `null` when not reported. */
  final_eval_loss: number | null;
  /** Number of optimizer steps executed. */
  total_steps: number;
  /** Number of epochs executed. */
  total_epochs: number;
  /** Tokens consumed during training, or `null` when not reported. */
  total_tokens_trained: number | null;
  /** Wall-clock duration of the job in seconds. */
  duration_secs: number;
  /** Provider-estimated cost in USD, or `null` when unknown. */
  estimated_cost_usd: number | null;
}

/** A zeroed metrics record: zero steps/epochs/duration, all optional values `null`. */
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
  /** Provider-assigned job id. */
  job_id: TrainingJobId;
  /** Name of the provider that owns the job (`"openai"`, `"together"`, `"fireworks"`, …). */
  provider: string;
  /** Base model the job fine-tunes. */
  base_model: string;
  /** Current status as of the listing call. */
  status: TrainingJobStatus;
  /** ISO 8601. */
  created_at: string;
  /** Final metrics when available; the shipped providers always return `null` from `listJobs`. */
  metrics: TrainingMetrics | null;
}
