/**
 * @module @brainwires/training
 *
 * Cloud fine-tuning orchestration for the Brainwires Agent Framework.
 *
 * Ships OpenAI, Together, and Fireworks providers plus a backoff-based
 * {@link JobPoller}. Bedrock and Vertex require their vendor SDKs and are
 * not ported in this first slice; implement {@link FineTuneProvider}
 * directly if you need them.
 *
 * Local training (Burn, GPU kernels) stays Rust-side — the Deno package
 * intentionally exposes only the cloud path.
 *
 * Equivalent to the `brainwires-training` crate built with the `cloud`
 * feature.
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
  type TrainingJobId,
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
