/**
 * Training-specific errors.
 *
 * Equivalent to Rust's `brainwires_training::error::TrainingError`.
 */

export type TrainingErrorKind =
  | "api"
  | "provider"
  | "upload"
  | "backend"
  | "job_not_found"
  | "validation";

export class TrainingError extends Error {
  readonly kind: TrainingErrorKind;
  readonly status_code: number | null;

  constructor(kind: TrainingErrorKind, message: string, status_code: number | null = null) {
    super(message);
    this.kind = kind;
    this.status_code = status_code;
    this.name = "TrainingError";
  }

  static api(message: string, status_code: number): TrainingError {
    return new TrainingError("api", message, status_code);
  }

  static provider(message: string): TrainingError {
    return new TrainingError("provider", message);
  }

  static upload(message: string): TrainingError {
    return new TrainingError("upload", message);
  }

  static backend(message: string): TrainingError {
    return new TrainingError("backend", message);
  }

  static jobNotFound(job_id: string): TrainingError {
    return new TrainingError("job_not_found", `Job not found: ${job_id}`);
  }

  static validation(message: string): TrainingError {
    return new TrainingError("validation", message);
  }
}
