/**
 * TrainingManager — thin registry + orchestrator over cloud providers.
 *
 * Equivalent to Rust's `rullama_training::manager::TrainingManager`.
 * The local-backend slice (Burn-based) stays in Rust; Deno consumers use
 * cloud fine-tuning only.
 */

import type {
  TrainingJobId,
  TrainingJobStatus,
  TrainingJobSummary,
} from "./types.ts";
import type { CloudFineTuneConfig, FineTuneProvider } from "./cloud/types.ts";
import { JobPoller, type JobPollerConfig } from "./cloud/polling.ts";
import { TrainingError } from "./error.ts";

/**
 * Registry of {@link FineTuneProvider}s keyed by `provider.name`, with
 * convenience methods that dispatch a job operation to one provider by name.
 */
export class TrainingManager {
  /** Registered providers, keyed by `provider.name`. */
  private readonly providers = new Map<string, FineTuneProvider>();

  /** Register (or replace, on the same `name`) a provider. */
  addCloudProvider(provider: FineTuneProvider): void {
    this.providers.set(provider.name, provider);
  }

  /** Names of every registered provider, in registration order. */
  cloudProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  /** Look up a provider by name; `null` when not registered. */
  getCloudProvider(name: string): FineTuneProvider | null {
    return this.providers.get(name) ?? null;
  }

  /**
   * Create a fine-tuning job on the named provider.
   *
   * @throws `TrainingError` (`provider`) listing the available names when
   *   `provider_name` is not registered.
   */
  async startCloudJob(
    provider_name: string,
    config: CloudFineTuneConfig,
  ): Promise<TrainingJobId> {
    const provider = this.providers.get(provider_name);
    if (!provider) {
      throw TrainingError.provider(
        `Unknown provider: ${provider_name}. Available: [${
          this.cloudProviders().join(", ")
        }]`,
      );
    }
    return await provider.createJob(config);
  }

  /**
   * Block until the job reaches a terminal state, polling with a
   * {@link JobPoller} (defaults from `defaultPollerConfig()` when
   * `poller_config` is omitted).
   *
   * @throws `TrainingError` (`provider`) for an unknown provider, or
   *   (`timeout`) if the poller's deadline passes first.
   */
  async waitForCloudJob(
    provider_name: string,
    job_id: TrainingJobId,
    poller_config?: JobPollerConfig,
  ): Promise<TrainingJobStatus> {
    const provider = this.providers.get(provider_name);
    if (!provider) {
      throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    }
    const poller = new JobPoller(poller_config);
    return await poller.poll(provider, job_id);
  }

  /** Fetch the job's current status once, without polling. */
  async checkCloudJob(
    provider_name: string,
    job_id: TrainingJobId,
  ): Promise<TrainingJobStatus> {
    const provider = this.providers.get(provider_name);
    if (!provider) {
      throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    }
    return await provider.getJobStatus(job_id);
  }

  /** Ask the named provider to cancel the job. */
  async cancelCloudJob(
    provider_name: string,
    job_id: TrainingJobId,
  ): Promise<void> {
    const provider = this.providers.get(provider_name);
    if (!provider) {
      throw TrainingError.provider(`Unknown provider: ${provider_name}`);
    }
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
