/**
 * TrainingManager — thin registry + orchestrator over cloud providers.
 *
 * Equivalent to Rust's `brainwires_training::manager::TrainingManager`.
 * The local-backend slice (Burn-based) stays in Rust; Deno consumers use
 * cloud fine-tuning only.
 */

import type { TrainingJobId, TrainingJobStatus, TrainingJobSummary } from "./types.ts";
import type { CloudFineTuneConfig, FineTuneProvider } from "./cloud/types.ts";
import { JobPoller, type JobPollerConfig } from "./cloud/polling.ts";
import { TrainingError } from "./error.ts";

export class TrainingManager {
  private readonly providers = new Map<string, FineTuneProvider>();

  addCloudProvider(provider: FineTuneProvider): void {
    this.providers.set(provider.name, provider);
  }

  cloudProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  getCloudProvider(name: string): FineTuneProvider | null {
    return this.providers.get(name) ?? null;
  }

  async startCloudJob(provider_name: string, config: CloudFineTuneConfig): Promise<TrainingJobId> {
    const provider = this.providers.get(provider_name);
    if (!provider) {
      throw TrainingError.provider(
        `Unknown provider: ${provider_name}. Available: [${this.cloudProviders().join(", ")}]`,
      );
    }
    return await provider.createJob(config);
  }

  async waitForCloudJob(
    provider_name: string,
    job_id: TrainingJobId,
    poller_config?: JobPollerConfig,
  ): Promise<TrainingJobStatus> {
    const provider = this.providers.get(provider_name);
    if (!provider) throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    const poller = new JobPoller(poller_config);
    return await poller.poll(provider, job_id);
  }

  async checkCloudJob(provider_name: string, job_id: TrainingJobId): Promise<TrainingJobStatus> {
    const provider = this.providers.get(provider_name);
    if (!provider) throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    return await provider.getJobStatus(job_id);
  }

  async cancelCloudJob(provider_name: string, job_id: TrainingJobId): Promise<void> {
    const provider = this.providers.get(provider_name);
    if (!provider) throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    await provider.cancelJob(job_id);
  }

  /** List jobs across every registered provider. Per-provider errors are swallowed. */
  async listAllCloudJobs(): Promise<TrainingJobSummary[]> {
    const out: TrainingJobSummary[] = [];
    for (const provider of this.providers.values()) {
      try {
        const jobs = await provider.listJobs();
        out.push(...jobs);
      } catch {
        // swallow per-provider failures so one bad provider doesn't kill the listing
      }
    }
    return out;
  }
}
