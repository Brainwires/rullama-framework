/**
 * Training-specific errors.
 *
 * Equivalent to Rust's `rullama_training::error::TrainingError`.
 */

/**
 * Category of a {@link TrainingError}: `api` (non-2xx provider response),
 * `provider` (unknown provider / malformed response), `upload` (dataset
 * upload response lacked a file id), `backend`, `job_not_found` (404 on a
 * job lookup), `validation` (unsupported config for the provider), or
 * `timeout` (poller deadline passed).
 */
export type TrainingErrorKind =
  | "api"
  | "provider"
  | "upload"
  | "backend"
  | "job_not_found"
  | "validation"
  | "timeout";

/** Error thrown by providers, the poller, and the manager. */
export class TrainingError extends Error {
  /** Failure category. */
  readonly kind: TrainingErrorKind;
  /** HTTP status of the failing provider response; `null` for non-HTTP failures. */
  readonly status_code: number | null;

  /**
   * Build an error of the given category.
   *
   * @param kind Failure category.
   * @param message Human-readable detail (used verbatim as `Error.message`).
   * @param status_code HTTP status when the failure came from a provider response.
   */
  constructor(
    kind: TrainingErrorKind,
    message: string,
    status_code: number | null = null,
  ) {
    super(message);
    this.kind = kind;
    this.status_code = status_code;
    this.name = "TrainingError";
  }

  /** A provider returned a non-2xx HTTP response. */
  static api(message: string, status_code: number): TrainingError {
    return new TrainingError("api", message, status_code);
  }

  /** Unknown provider name, or a provider response missing a required field. */
  static provider(message: string): TrainingError {
    return new TrainingError("provider", message);
  }

  /** Dataset upload succeeded at the HTTP level but returned no file id. */
  static upload(message: string): TrainingError {
    return new TrainingError("upload", message);
  }

  /** Generic backend failure. Reserved for the (Rust-side) local backend; not thrown by the shipped providers. */
  static backend(message: string): TrainingError {
    return new TrainingError("backend", message);
  }

  /** A job lookup returned 404; message is `Job not found: <job_id>`. */
  static jobNotFound(job_id: string): TrainingError {
    return new TrainingError("job_not_found", `Job not found: ${job_id}`);
  }

  /** A poll deadline passed before the job reached a terminal state. */
  static timeout(message: string): TrainingError {
    return new TrainingError("timeout", message);
  }

  /** The config asks for something the provider cannot do (e.g. ORPO alignment). */
  static validation(message: string): TrainingError {
    return new TrainingError("validation", message);
  }
}
