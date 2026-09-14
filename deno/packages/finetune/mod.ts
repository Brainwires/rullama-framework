/**
 * Cloud fine-tuning orchestration for rullama: upload a dataset, start a
 * job, poll it to completion, and manage the resulting models.
 *
 * Ships OpenAI, Together, and Fireworks implementations of the
 * {@link FineTuneProvider} interface, a backoff-based {@link JobPoller},
 * and a {@link TrainingManager} that dispatches to providers by name.
 * Bedrock and Vertex require their vendor SDKs and are not ported;
 * implement {@link FineTuneProvider} directly if you need them.
 *
 * Local training (Burn, GPU kernels) stays Rust-side — this package
 * intentionally exposes only the cloud path. Equivalent to the
 * `rullama-training` crate built with the `cloud` feature.
 *
 * @module
 */

// Shared types
export {
  completionFraction,
  DatasetId,
  defaultMetrics,
  defaultProgress,
  isRunning,
  isSucceeded,
  isTerminal,
  TrainingJobId,
  /** @deprecated `TrainingJobId` is now exported as a value; alias kept for 0.12. */
  TrainingJobId as TrainingJobIdClass,
  type TrainingJobStatus,
  type TrainingJobSummary,
  type TrainingMetrics,
  type TrainingProgress,
} from "./types.ts";

// Config
export {
  type AdapterMethod,
  type AlignmentMethod,
  defaultAlignment,
  defaultHyperparams,
  defaultLoraConfig,
  dpoAlignment,
  isQuantized,
  type LoraConfig,
  type LrScheduler,
  orpoAlignment,
  quantizationBits,
  type TrainingHyperparams,
} from "./config.ts";

// Errors
export { TrainingError, type TrainingErrorKind } from "./error.ts";

// Cloud interface + config
export {
  type CloudFineTuneConfig,
  type DataFormat,
  type FineTuneProvider,
  newCloudFineTuneConfig,
} from "./cloud/types.ts";

// Cloud providers
export { OPENAI_API_BASE, OpenAiFineTune } from "./cloud/openai.ts";
export { TOGETHER_API_BASE, TogetherFineTune } from "./cloud/together.ts";
export { FIREWORKS_API_BASE, FireworksFineTune } from "./cloud/fireworks.ts";

// Poller + manager
export {
  defaultPollerConfig,
  JobPoller,
  type JobPollerConfig,
} from "./cloud/polling.ts";
export { TrainingManager } from "./manager.ts";
