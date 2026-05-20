/**
 * OpenAI fine-tuning provider.
 *
 * Equivalent to Rust's `brainwires_training::cloud::openai` module.
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

export const OPENAI_API_BASE = "https://api.openai.com/v1";

export class OpenAiFineTune implements FineTuneProvider {
  readonly name = "openai";
  readonly base_url: string;
  private readonly api_key: string;

  constructor(api_key: string, base_url: string = OPENAI_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  private filesUrl(): string {
    return `${this.base_url}/files`;
  }

  private finetuneUrl(): string {
    return `${this.base_url}/fine_tuning/jobs`;
  }

  supportedBaseModels(): string[] {
    return [
      "gpt-4o-mini-2024-07-18",
      "gpt-4o-2024-08-06",
      "gpt-4-0613",
      "gpt-3.5-turbo-0125",
      "gpt-3.5-turbo-1106",
    ];
  }

  supportsDpo(): boolean {
    return true;
  }

  /** Parse the OpenAI-side status string into a TrainingJobStatus. Exposed for tests. */
  static parseJobStatus(
    status_str: string,
    body: Record<string, unknown>,
  ): TrainingJobStatus {
    switch (status_str) {
      case "validating_files":
        return { status: "validating" };
      case "queued":
        return { status: "queued" };
      case "running": {
        const step = typeof body.trained_tokens === "number" ? body.trained_tokens : 0;
        return {
          status: "running",
          progress: {
            epoch: 0,
            total_epochs: 0,
            step,
            total_steps: 0,
            train_loss: null,
            eval_loss: null,
            learning_rate: null,
            elapsed_secs: 0,
          },
        };
      }
      case "succeeded": {
        const model_id = typeof body.fine_tuned_model === "string"
          ? body.fine_tuned_model
          : "unknown";
        return { status: "succeeded", model_id };
      }
      case "failed": {
        const err = body.error as { message?: string } | undefined;
        return { status: "failed", error: err?.message ?? "Unknown error" };
      }
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
    const res = await fetch(this.filesUrl(), {
      method: "POST",
      headers: { Authorization: `Bearer ${this.api_key}` },
      body: form,
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      const msg = (body.error as { message?: string } | undefined)?.message ?? "Unknown upload error";
      throw TrainingError.api(msg, res.status);
    }
    const id = body.id;
    if (typeof id !== "string") throw TrainingError.upload("Missing file ID in response");
    return new DatasetId(id);
  }

  async createJob(config: CloudFineTuneConfig): Promise<TrainingJobId> {
    const body: Record<string, unknown> = {
      training_file: config.training_dataset.value,
      model: config.base_model,
      hyperparameters: {
        n_epochs: config.hyperparams.epochs,
        batch_size: config.hyperparams.batch_size,
        learning_rate_multiplier: config.hyperparams.learning_rate / 2e-5,
      },
    };
    if (config.validation_dataset) body.validation_file = config.validation_dataset.value;
    if (config.suffix) body.suffix = config.suffix;

    const res = await fetch(this.finetuneUrl(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const resBody = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      const msg = (resBody.error as { message?: string } | undefined)?.message ?? "Unknown error";
      throw TrainingError.api(msg, res.status);
    }
    const id = resBody.id;
    if (typeof id !== "string") throw TrainingError.provider("Missing job ID in response");
    return new TrainingJobIdClass(id);
  }

  async getJobStatus(job_id: TrainingJobId): Promise<TrainingJobStatus> {
    const res = await fetch(`${this.finetuneUrl()}/${job_id.value}`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      if (res.status === 404) throw TrainingError.jobNotFound(job_id.value);
      const msg = (body.error as { message?: string } | undefined)?.message ?? "Unknown error";
      throw TrainingError.api(msg, res.status);
    }
    const status_str = typeof body.status === "string" ? body.status : "unknown";
    return OpenAiFineTune.parseJobStatus(status_str, body);
  }

  async cancelJob(job_id: TrainingJobId): Promise<void> {
    const res = await fetch(`${this.finetuneUrl()}/${job_id.value}/cancel`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<string, unknown>;
      const msg = (body.error as { message?: string } | undefined)?.message ?? "Failed to cancel job";
      throw TrainingError.api(msg, res.status);
    }
  }

  async listJobs(): Promise<TrainingJobSummary[]> {
    const res = await fetch(this.finetuneUrl(), {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      const msg = (body.error as { message?: string } | undefined)?.message ?? "Failed to list jobs";
      throw TrainingError.api(msg, res.status);
    }
    const data = Array.isArray(body.data) ? body.data : [];
    const out: TrainingJobSummary[] = [];
    for (const raw of data) {
      const job = raw as Record<string, unknown>;
      const id = job.id;
      const model = job.model;
      const status_str = job.status;
      const ts = job.created_at;
      if (
        typeof id !== "string" || typeof model !== "string" ||
        typeof status_str !== "string" || typeof ts !== "number"
      ) continue;
      out.push({
        job_id: new TrainingJobIdClass(id),
        provider: "openai",
        base_model: model,
        status: OpenAiFineTune.parseJobStatus(status_str, job),
        created_at: new Date(ts * 1000).toISOString(),
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
      const msg = (body.error as { message?: string } | undefined)?.message ?? "Failed to delete model";
      throw TrainingError.api(msg, res.status);
    }
  }
}
