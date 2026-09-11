/**
 * Resilience-specific error types.
 *
 * Equivalent to Rust's `rullama_resilience::error::ResilienceError`.
 */

/** Kind tag discriminating the specific resilience error. */
export type ResilienceErrorKind =
  | "budget_exceeded"
  | "circuit_open"
  | "retries_exhausted"
  | "deadline_exceeded";

/** Errors surfaced by resilience decorators. */
export class ResilienceError extends Error {
  /** Which decorator condition produced this error. */
  readonly kind: ResilienceErrorKind;
  /**
   * Structured context for the `kind` — e.g. `{ kind, consumed, limit }` for
   * `budget_exceeded`, `{ provider, model, failures }` for `circuit_open`,
   * `{ attempts }` / `{ attempts, elapsed_ms }` for the retry kinds.
   */
  readonly detail: Record<string, unknown>;

  /**
   * Construct directly; the static factories below build the standard
   * messages and `detail` shapes.
   *
   * @param kind Error discriminator.
   * @param message Human-readable description.
   * @param detail Structured context stored on {@link detail}.
   */
  constructor(
    kind: ResilienceErrorKind,
    message: string,
    detail: Record<string, unknown>,
  ) {
    super(message);
    this.kind = kind;
    this.detail = detail;
    this.name = "ResilienceError";
  }

  /** Budget cap reached before the request could be sent. */
  static budgetExceeded(
    kind: "tokens" | "usd_cents" | "rounds",
    consumed: number,
    limit: number,
  ): ResilienceError {
    return new ResilienceError(
      "budget_exceeded",
      `budget exceeded: ${kind} (${consumed}/${limit})`,
      { kind, consumed, limit },
    );
  }

  /** Circuit breaker is open and rejecting calls for the cooldown window. */
  static circuitOpen(
    provider: string,
    model: string,
    failures: number,
  ): ResilienceError {
    return new ResilienceError(
      "circuit_open",
      `circuit open for ${provider}/${model}: ${failures} consecutive failures`,
      { provider, model, failures },
    );
  }

  /** Retries exhausted — final attempt's error attached as `cause`. */
  static retriesExhausted(attempts: number, source: Error): ResilienceError {
    const e = new ResilienceError(
      "retries_exhausted",
      `retries exhausted after ${attempts} attempts: ${source.message}`,
      { attempts },
    );
    e.cause = source;
    return e;
  }

  /** `RetryPolicy.overall_deadline` elapsed before the call could succeed. */
  static deadlineExceeded(
    attempts: number,
    elapsed_ms: number,
    source: Error,
  ): ResilienceError {
    const e = new ResilienceError(
      "deadline_exceeded",
      `retry deadline exceeded after ${elapsed_ms}ms (${attempts} attempts): ${source.message}`,
      { attempts, elapsed_ms },
    );
    e.cause = source;
    return e;
  }
}
