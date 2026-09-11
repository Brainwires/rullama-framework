/**
 * Fireworks AI fine-tuning provider.
 *
 * Equivalent to Rust's `rullama_training::cloud::fireworks` module.
 */

import { TrainingError } from "../error.ts";
import {
  DatasetId,
  type TrainingJobId,
  TrainingJobId as TrainingJobIdClass,
  type TrainingJobStatus,
  type TrainingJobSummary,
} from "../types.ts";
import type {
  CloudFineTuneConfig,
  DataFormat,
  FineTuneProvider,
} from "./types.ts";

/** Default Fireworks AI API root used when no `base_url` is passed. */
export const FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1";

/**
 * Fireworks AI implementation of {@link FineTuneProvider} over `/files` + `/fine_tuning/jobs`.
 */
export class FireworksFineTune implements FineTuneProvider {
  /** Provider key used by `TrainingManager`. */
  readonly name = "fireworks";
  /** API root every request URL is built from. */
  readonly base_url: string;
  /** Bearer token sent in the `Authorization` header of every request. */
  private readonly api_key: string;

  /**
   * Create a provider bound to one API key.
   *
   * @param api_key Fireworks AI API key.
   * @param base_url Override the API root (e.g. a proxy or a test server).
   */
  constructor(api_key: string, base_url: string = FIREWORKS_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  /** Fireworks-hosted models accepted for fine-tuning (Llama-v3 8B/70B, Mixtral-8x7B). */
  supportedBaseModels(): string[] {
    return [
      "accounts/fireworks/models/llama-v3-8b-instruct",
      "accounts/fireworks/models/llama-v3-70b-instruct",
      "accounts/fireworks/models/mixtral-8x7b-instruct",
    ];
  }

  /** Fireworks' fine-tuning API has no preference-optimisation mode. */
  supportsDpo(): boolean {
    return false;
  }

  /** Map Fireworks' status into the neutral enum. Exposed for tests. */
  static parseJobStatus(
    status_str: string,
    body: Record<string, unknown>,
  ): TrainingJobStatus {
    switch (status_str) {
      case "JOB_STATE_PENDING":
      case "pending":
        return { status: "pending" };
      case "JOB_STATE_QUEUED":
      case "queued":
        return { status: "queued" };
      case "JOB_STATE_VALIDATING":
      case "validating":
        return { status: "validating" };
      case "JOB_STATE_RUNNING":
      case "running":
        return {
          status: "running",
          progress: {
            epoch: 0,
            total_epochs: 0,
            step: typeof body.current_step === "number" ? body.current_step : 0,
            total_steps: typeof body.total_steps === "number"
              ? body.total_steps
              : 0,
            train_loss: null,
            eval_loss: null,
            learning_rate: null,
            elapsed_secs: 0,
          },
        };
      case "JOB_STATE_COMPLETED":
      case "succeeded": {
        const model_id = typeof body.model === "string"
          ? body.model
          : "unknown";
        return { status: "succeeded", model_id };
      }
      case "JOB_STATE_FAILED":
      case "failed": {
        const err = body.error;
        const msg = typeof err === "string"
          ? err
          : (err as { message?: string } | undefined)?.message;
        return { status: "failed", error: msg ?? "Unknown error" };
      }
      case "JOB_STATE_CANCELLED":
      case "cancelled":
        return { status: "cancelled" };
      default:
        return { status: "pending" };
    }
  }

  /**
   * Multipart-upload the bytes to `/files` with `purpose=fine-tune`. The `format` argument is ignored — the file is always sent as `training_data.jsonl` with type `application/json`.
   *
   * @returns The Fireworks file id.
   *
   * @throws `TrainingError` (`api`) on a non-2xx response, (`upload`) if the response has no `id`.
   */
  async uploadDataset(
    data: Uint8Array,
    _format: DataFormat,
  ): Promise<DatasetId> {
    const form = new FormData();
    form.append("purpose", "fine-tune");
    form.append(
      "file",
      new Blob([data as BlobPart], { type: "application/json" }),
      "training_data.jsonl",
    );
    const res = await fetch(`${this.base_url}/files`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.api_key}` },
      body: form,
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      throw TrainingError.api(
        errorMessage(body, "Fireworks upload error"),
        res.status,
      );
    }
    const id = typeof body.id === "string" ? body.id : null;
    if (id === null) throw TrainingError.upload("Missing file ID in response");
    return new DatasetId(id);
  }

  /**
   * POST `/fine_tuning/jobs`. Sends `epochs`, `learning_rate`, `batch_size`, plus `validation_dataset`, `output_model` (from `suffix`), and `lora_rank` when set. `config.alignment` is not forwarded — Fireworks has no DPO/ORPO mode (and, unlike the other providers, a non-`none` alignment is silently ignored rather than rejected).
   *
   * @throws `TrainingError` (`api`) on a non-2xx response, (`provider`) if the response has no `id`.
   */
  async createJob(config: CloudFineTuneConfig): Promise<TrainingJobId> {
    const body: Record<string, unknown> = {
      base_model: config.base_model,
      training_dataset: config.training_dataset.value,
      epochs: config.hyperparams.epochs,
      learning_rate: config.hyperparams.learning_rate,
      batch_size: config.hyperparams.batch_size,
    };
    if (config.validation_dataset) {
      body.validation_dataset = config.validation_dataset.value;
    }
    if (config.suffix) body.output_model = config.suffix;
    if (config.lora) body.lora_rank = config.lora.rank;

    const res = await fetch(`${this.base_url}/fine_tuning/jobs`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const resBody = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      throw TrainingError.api(
        errorMessage(resBody, "Unknown error"),
        res.status,
      );
    }
    const id = typeof resBody.id === "string" ? resBody.id : null;
    if (id === null) throw TrainingError.provider("Missing job ID in response");
    return new TrainingJobIdClass(id);
  }

  /**
   * GET `/fine_tuning/jobs/{id}` and map `state` (falling back to `status`) via {@link FireworksFineTune.parseJobStatus}.
   *
   * @throws `TrainingError` (`job_not_found`) on 404, (`api`) on other non-2xx responses.
   */
  async getJobStatus(job_id: TrainingJobId): Promise<TrainingJobStatus> {
    const res = await fetch(
      `${this.base_url}/fine_tuning/jobs/${job_id.value}`,
      {
        headers: { Authorization: `Bearer ${this.api_key}` },
      },
    );
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      if (res.status === 404) throw TrainingError.jobNotFound(job_id.value);
      throw TrainingError.api(errorMessage(body, "Unknown error"), res.status);
    }
    const state = typeof body.state === "string"
      ? body.state
      : typeof body.status === "string"
      ? body.status
      : "unknown";
    return FireworksFineTune.parseJobStatus(state, body);
  }

  /** POST `/fine_tuning/jobs/{id}:cancel`; throws `TrainingError` (`api`) on a non-2xx response. */
  async cancelJob(job_id: TrainingJobId): Promise<void> {
    const res = await fetch(
      `${this.base_url}/fine_tuning/jobs/${job_id.value}:cancel`,
      {
        method: "POST",
        headers: { Authorization: `Bearer ${this.api_key}` },
      },
    );
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<
        string,
        unknown
      >;
      throw TrainingError.api(
        errorMessage(body, "Failed to cancel job"),
        res.status,
      );
    }
  }

  /**
   * GET `/fine_tuning/jobs` (reads `jobs`, else `data`) and map each entry to a summary, accepting `id`/`name`, `base_model`/`model`, and `state`/`status`. `created_at` is set to now — the endpoint's timestamp is not read; `metrics` is always `null`.
   */
  async listJobs(): Promise<TrainingJobSummary[]> {
    const res = await fetch(`${this.base_url}/fine_tuning/jobs`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      throw TrainingError.api(
        errorMessage(body, "Failed to list jobs"),
        res.status,
      );
    }
    const data = Array.isArray(body.jobs)
      ? body.jobs
      : Array.isArray(body.data)
      ? body.data
      : [];
    const out: TrainingJobSummary[] = [];
    for (const raw of data) {
      const job = raw as Record<string, unknown>;
      const id = job.id ?? job.name;
      const base_model = job.base_model ?? job.model;
      const state = job.state ?? job.status;
      if (
        typeof id !== "string" || typeof base_model !== "string" ||
        typeof state !== "string"
      ) continue;
      out.push({
        job_id: new TrainingJobIdClass(id),
        provider: "fireworks",
        base_model,
        status: FireworksFineTune.parseJobStatus(state, job),
        created_at: new Date().toISOString(),
        metrics: null,
      });
    }
    return out;
  }

  /** DELETE `/models/{model_id}`; throws `TrainingError` (`api`) on a non-2xx response. */
  async deleteModel(model_id: string): Promise<void> {
    const res = await fetch(`${this.base_url}/models/${model_id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<
        string,
        unknown
      >;
      throw TrainingError.api(
        errorMessage(body, "Failed to delete model"),
        res.status,
      );
    }
  }
}

function errorMessage(body: Record<string, unknown>, fallback: string): string {
  const err = body.error;
  if (typeof err === "string") return err;
  if (err && typeof err === "object") {
    const m = (err as Record<string, unknown>).message;
    if (typeof m === "string") return m;
  }
  const msg = body.message;
  return typeof msg === "string" ? msg : fallback;
}
