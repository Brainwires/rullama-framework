/**
 * Fireworks AI fine-tuning provider.
 *
 * Equivalent to Rust's `brainwires_training::cloud::fireworks` module.
 */

import { TrainingError } from "../error.ts";
import {
  DatasetId,
  type TrainingJobId,
  TrainingJobId as TrainingJobIdClass,
  type TrainingJobStatus,
  type TrainingJobSummary,
} from "../types.ts";
import type { CloudFineTuneConfig, DataFormat, FineTuneProvider } from "./types.ts";

export const FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1";

export class FireworksFineTune implements FineTuneProvider {
  readonly name = "fireworks";
  readonly base_url: string;
  private readonly api_key: string;

  constructor(api_key: string, base_url: string = FIREWORKS_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  supportedBaseModels(): string[] {
    return [
      "accounts/fireworks/models/llama-v3-8b-instruct",
      "accounts/fireworks/models/llama-v3-70b-instruct",
      "accounts/fireworks/models/mixtral-8x7b-instruct",
    ];
  }

  supportsDpo(): boolean {
    return true;
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
            total_steps: typeof body.total_steps === "number" ? body.total_steps : 0,
            train_loss: null,
            eval_loss: null,
            learning_rate: null,
            elapsed_secs: 0,
          },
        };
      case "JOB_STATE_COMPLETED":
      case "succeeded": {
        const model_id = typeof body.model === "string" ? body.model : "unknown";
        return { status: "succeeded", model_id };
      }
      case "JOB_STATE_FAILED":
      case "failed": {
        const err = body.error;
        const msg = typeof err === "string" ? err : (err as { message?: string } | undefined)?.message;
        return { status: "failed", error: msg ?? "Unknown error" };
      }
      case "JOB_STATE_CANCELLED":
      case "cancelled":
        return { status: "cancelled" };
      default:
        return { status: "pending" };
    }
  }

  async uploadDataset(data: Uint8Array, _format: DataFormat): Promise<DatasetId> {
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
    if (!res.ok) throw TrainingError.api(errorMessage(body, "Fireworks upload error"), res.status);
    const id = typeof body.id === "string" ? body.id : null;
    if (id === null) throw TrainingError.upload("Missing file ID in response");
    return new DatasetId(id);
  }

  async createJob(config: CloudFineTuneConfig): Promise<TrainingJobId> {
    const body: Record<string, unknown> = {
      base_model: config.base_model,
      training_dataset: config.training_dataset.value,
      epochs: config.hyperparams.epochs,
      learning_rate: config.hyperparams.learning_rate,
      batch_size: config.hyperparams.batch_size,
    };
    if (config.validation_dataset) body.validation_dataset = config.validation_dataset.value;
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
    if (!res.ok) throw TrainingError.api(errorMessage(resBody, "Unknown error"), res.status);
    const id = typeof resBody.id === "string" ? resBody.id : null;
    if (id === null) throw TrainingError.provider("Missing job ID in response");
    return new TrainingJobIdClass(id);
  }

  async getJobStatus(job_id: TrainingJobId): Promise<TrainingJobStatus> {
    const res = await fetch(`${this.base_url}/fine_tuning/jobs/${job_id.value}`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      if (res.status === 404) throw TrainingError.jobNotFound(job_id.value);
      throw TrainingError.api(errorMessage(body, "Unknown error"), res.status);
    }
    const state = typeof body.state === "string" ? body.state : typeof body.status === "string" ? body.status : "unknown";
    return FireworksFineTune.parseJobStatus(state, body);
  }

  async cancelJob(job_id: TrainingJobId): Promise<void> {
    const res = await fetch(`${this.base_url}/fine_tuning/jobs/${job_id.value}:cancel`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<string, unknown>;
      throw TrainingError.api(errorMessage(body, "Failed to cancel job"), res.status);
    }
  }

  async listJobs(): Promise<TrainingJobSummary[]> {
    const res = await fetch(`${this.base_url}/fine_tuning/jobs`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) throw TrainingError.api(errorMessage(body, "Failed to list jobs"), res.status);
    const data = Array.isArray(body.jobs) ? body.jobs : Array.isArray(body.data) ? body.data : [];
    const out: TrainingJobSummary[] = [];
    for (const raw of data) {
      const job = raw as Record<string, unknown>;
      const id = job.id ?? job.name;
      const base_model = job.base_model ?? job.model;
      const state = job.state ?? job.status;
      if (typeof id !== "string" || typeof base_model !== "string" || typeof state !== "string") continue;
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

  async deleteModel(model_id: string): Promise<void> {
    const res = await fetch(`${this.base_url}/models/${model_id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<string, unknown>;
      throw TrainingError.api(errorMessage(body, "Failed to delete model"), res.status);
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
