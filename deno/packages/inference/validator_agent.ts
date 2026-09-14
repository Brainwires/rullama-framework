/**
 * ValidatorAgent - Standalone read-only agent that runs external validators.
 *
 * Unlike the inline validation inside TaskAgent, the ValidatorAgent can be
 * triggered independently by an orchestrator -- e.g., after multiple task
 * agents finish work -- without coupling validation to any single task agent.
 *
 * This is intentionally not an AgentRuntime implementation: it is a
 * deterministic pipeline (no AI provider loop).
 *
 * @module
 */

import {
  formatValidationFeedback,
  runValidation,
  type ValidationConfig,
  type ValidationResult,
} from "./validation_loop.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Current status of a ValidatorAgent. */
export type ValidatorAgentStatus =
  | { kind: "idle" }
  | { kind: "acquiring_locks" }
  | { kind: "validating" }
  | { kind: "passed" }
  | { kind: "failed"; issueCount: number }
  | { kind: "error"; message: string };

/** Format a ValidatorAgentStatus as a display string. */
export function formatValidatorStatus(status: ValidatorAgentStatus): string {
  switch (status.kind) {
    case "idle":
      return "Idle";
    case "acquiring_locks":
      return "Acquiring locks";
    case "validating":
      return "Validating";
    case "passed":
      return "Passed";
    case "failed":
      return `Failed (${status.issueCount} issues)`;
    case "error":
      return `Error: ${status.message}`;
  }
}

/** Configuration for the ValidatorAgent. */
export interface ValidatorAgentConfig {
  /** The underlying validation pipeline configuration. */
  validationConfig: ValidationConfig;
  /** Wall-clock timeout in milliseconds for the entire validation run. Default: 120000. */
  timeoutMs: number;
}

/** Default validator agent config. */
export function defaultValidatorAgentConfig(
  validationConfig: ValidationConfig,
): ValidatorAgentConfig {
  return { validationConfig, timeoutMs: 120_000 };
}

/** Result returned by ValidatorAgent.validate(). */
export interface ValidatorAgentResult {
  /** The validator agent's unique ID. */
  agentId: string;
  /** Whether all checks passed. */
  success: boolean;
  /** The raw validation result from the pipeline. */
  validationResult: ValidationResult;
  /** Human-readable feedback string. */
  feedback: string;
  /** Wall-clock duration of the validation run in ms. */
  durationMs: number;
  /** Number of files that were checked. */
  filesChecked: number;
}

// ---------------------------------------------------------------------------
// ValidatorAgent
// ---------------------------------------------------------------------------

/**
 * A standalone, read-only agent that runs external validators and returns
 * a structured result to the orchestrator.
 */
export class ValidatorAgent {
  /** Unique validator ID. */
  readonly id: string;
  /** Configuration (checks, working directory, timeout). */
  readonly config: ValidatorAgentConfig;
  private _status: ValidatorAgentStatus = { kind: "idle" };

  /** Create a validator agent with `id` and `config`. */
  constructor(id: string, config: ValidatorAgentConfig) {
    this.id = id;
    this.config = config;
  }

  /** Get the current status. */
  get status(): ValidatorAgentStatus {
    return this._status;
  }

  /** Run the full validation pipeline. */
  async validate(): Promise<ValidatorAgentResult> {
    const start = Date.now();

    this._status = { kind: "validating" };

    let validationResult: ValidationResult;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      // Create a timeout promise; the timer is cleared once the race settles so
      // a finished validation does not keep the event loop alive.
      const timeoutPromise = new Promise<never>((_, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new Error(
                `Validation timed out after ${this.config.timeoutMs}ms`,
              ),
            ),
          this.config.timeoutMs,
        );
      });

      validationResult = await Promise.race([
        runValidation(this.config.validationConfig),
        timeoutPromise,
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._status = { kind: "error", message: msg };
      throw err;
    } finally {
      clearTimeout(timer);
    }

    const success = validationResult.passed;
    const issueCount = validationResult.issues.length;
    const filesChecked = this.config.validationConfig.workingSetFiles?.length ??
      0;
    const feedback = formatValidationFeedback(validationResult);

    if (success) {
      this._status = { kind: "passed" };
    } else {
      this._status = { kind: "failed", issueCount };
    }

    return {
      agentId: this.id,
      success,
      validationResult,
      feedback,
      durationMs: Date.now() - start,
      filesChecked,
    };
  }
}
