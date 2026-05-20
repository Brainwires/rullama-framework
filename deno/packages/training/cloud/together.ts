/**
 * Together AI fine-tuning provider.
 *
 * Equivalent to Rust's `brainwires_training::cloud::together` module.
 * Implements the same FineTuneProvider contract as OpenAI over Together's
 * `/v1/files` + `/v1/fine-tunes` endpoints.
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

export const TOGETHER_API_BASE = "https://api.together.xyz/v1";

export class TogetherFineTune implements FineTuneProvider {
  readonly name = "together";
  readonly base_url: string;
  private readonly api_key: string;

  constructor(api_key: string, base_url: string = TOGETHER_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  supportedBaseModels(): string[] {
    return [
      "meta-llama/Llama-3-8b-chat-hf",
      "meta-llama/Llama-3-70b-chat-hf",
      "mistralai/Mistral-7B-Instruct-v0.2",
      "Qwen/Qwen2-7B-Instruct",
    ];
  }

  supportsDpo(): boolean {
    return true;
  }

  /** Map Together's status strings onto the neutral enum. Exposed for tests. */
  static parseJobStatus(
    status_str: string,
    body: Record<string, unknown>,
  ): TrainingJobStatus {
    switch (status_str) {
      case "pending":
        return { status: "pending" };
      case "queued":
        return { status: "queued" };
      case "validating":
        return { status: "validating" };
      case "training":
      case "running":
        return {
          status: "running",
          progress: {
            epoch: 0,
            total_epochs: 0,
            step: typeof body.step === "number" ? body.step : 0,
            total_steps: typeof body.total_steps === "number" ? body.total_steps : 0,
            train_loss: null,
            eval_loss: null,
            learning_rate: null,
            elapsed_secs: 0,
          },
        };
      case "completed":
      case "succeeded": {
        const model_id = typeof body.output_name === "string" ? body.output_name : "unknown";
        return { status: "succeeded", model_id };
      }
      case "failed": {
        const err = body.error;
        const msg = typeof err === "string" ? err : (err as { message?: string } | undefined)?.message;
        return { status: "failed", error: msg ?? "Unknown error" };
      }
      case "cancelled":
      case "user_cancelled":
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
    if (!res.ok) throw TrainingError.api(errorMessage(body, "Together upload error"), res.status);
    const id = typeof body.id === "string" ? body.id : null;
    if (id === null) throw TrainingError.upload("Missing file ID in response");
    return new DatasetId(id);
  }

  async createJob(config: CloudFineTuneConfig): Promise<TrainingJobId> {
    const body: Record<string, unknown> = {
      training_file: config.training_dataset.value,
      model: config.base_model,
      n_epochs: config.hyperparams.epochs,
      batch_size: config.hyperparams.batch_size,
      learning_rate: config.hyperparams.learning_rate,
    };
    if (config.validation_dataset) body.validation_file = config.validation_dataset.value;
    if (config.suffix) body.suffix = config.suffix;
    if (config.lora) {
      body.lora = true;
      body.lora_r = config.lora.rank;
      body.lora_alpha = config.lora.alpha;
      body.lora_dropout = config.lora.dropout;
    }

    const res = await fetch(`${this.base_url}/fine-tunes`, {
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
    const res = await fetch(`${this.base_url}/fine-tunes/${job_id.value}`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) {
      if (res.status === 404) throw TrainingError.jobNotFound(job_id.value);
      throw TrainingError.api(errorMessage(body, "Unknown error"), res.status);
    }
    const status_str = typeof body.status === "string" ? body.status : "unknown";
    return TogetherFineTune.parseJobStatus(status_str, body);
  }

  async cancelJob(job_id: TrainingJobId): Promise<void> {
    const res = await fetch(`${this.base_url}/fine-tunes/${job_id.value}/cancel`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({})) as Record<string, unknown>;
      throw TrainingError.api(errorMessage(body, "Failed to cancel job"), res.status);
    }
  }

  async listJobs(): Promise<TrainingJobSummary[]> {
    const res = await fetch(`${this.base_url}/fine-tunes`, {
      headers: { Authorization: `Bearer ${this.api_key}` },
    });
    const body = await res.json() as Record<string, unknown>;
    if (!res.ok) throw TrainingError.api(errorMessage(body, "Failed to list jobs"), res.status);
    const data = Array.isArray(body.data) ? body.data : [];
    const out: TrainingJobSummary[] = [];
    for (const raw of data) {
      const job = raw as Record<string, unknown>;
      const id = job.id;
      const model = job.model;
      const status_str = job.status;
      if (typeof id !== "string" || typeof model !== "string" || typeof status_str !== "string") continue;
      const created = typeof job.created_at === "number"
        ? new Date(job.created_at * 1000).toISOString()
        : new Date().toISOString();
      out.push({
        job_id: new TrainingJobIdClass(id),
        provider: "together",
        base_model: model,
        status: TogetherFineTune.parseJobStatus(status_str, job),
        created_at: created,
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
