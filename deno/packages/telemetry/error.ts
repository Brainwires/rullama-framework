/**
 * Errors returned by the analytics subsystem.
 *
 * Equivalent to Rust's `rullama_telemetry::error::AnalyticsError`.
 */

/**
 * Category of an {@link AnalyticsError}: `"channel_closed"` — the sink
 * pipeline was shut down; `"io"` — a sink failed to read/write; `"serde"` —
 * an event could not be (de)serialized; `"other"` — anything else.
 */
export type AnalyticsErrorKind =
  | "channel_closed"
  | "io"
  | "serde"
  | "other";

/** Error raised by the analytics pipeline (collector / sinks), tagged with a {@link AnalyticsErrorKind}. */
export class AnalyticsError extends Error {
  /** Which category of failure this is. */
  readonly kind: AnalyticsErrorKind;

  /**
   * Create an error of category `kind`; prefer the static constructors,
   * which apply the conventional message prefixes.
   * @param kind Failure category.
   * @param message Human-readable message, used verbatim as `Error.message`.
   */
  constructor(kind: AnalyticsErrorKind, message: string) {
    super(message);
    this.kind = kind;
    this.name = "AnalyticsError";
  }

  /** A `"channel_closed"` error with the fixed message `"Analytics sink channel closed"`. */
  static channelClosed(): AnalyticsError {
    return new AnalyticsError(
      "channel_closed",
      "Analytics sink channel closed",
    );
  }

  /** An `"io"` error; the message is prefixed with `"I/O error: "`. */
  static io(message: string): AnalyticsError {
    return new AnalyticsError("io", `I/O error: ${message}`);
  }

  /** A `"serde"` error; the message is prefixed with `"Serialization error: "`. */
  static serde(message: string): AnalyticsError {
    return new AnalyticsError("serde", `Serialization error: ${message}`);
  }

  /** An `"other"` error carrying `message` unchanged. */
  static other(message: string): AnalyticsError {
    return new AnalyticsError("other", message);
  }
}

/** Errors returned by BillingHook implementations. */
export type BillingErrorKind =
  | "hook"
  | "budget_exhausted"
  | "serde";

/**
 * Error raised by a `BillingHook`, tagged with a {@link BillingErrorKind}:
 * `"hook"` — the hook itself failed; `"budget_exhausted"` — the agent's
 * spend limit was reached; `"serde"` — a usage event could not be serialized.
 */
export class BillingError extends Error {
  /** Which category of failure this is. */
  readonly kind: BillingErrorKind;
  /** Structured context for the failure; for `budget_exhausted` it holds `agent_id`, `spent` and `limit`. Empty otherwise. */
  readonly detail: Record<string, unknown>;

  /**
   * Create an error of category `kind`; prefer {@link hook} /
   * {@link budgetExhausted}, which fill in message and `detail`.
   * @param kind Failure category.
   * @param message Human-readable message, used verbatim as `Error.message`.
   * @param detail Structured context attached to the error (default `{}`).
   */
  constructor(
    kind: BillingErrorKind,
    message: string,
    detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.kind = kind;
    this.detail = detail;
    this.name = "BillingError";
  }

  /** A `"hook"` error; the message is prefixed with `"billing hook error: "`. */
  static hook(message: string): BillingError {
    return new BillingError("hook", `billing hook error: ${message}`);
  }

  /**
   * A `"budget_exhausted"` error for `agent_id` whose message reports
   * `spent / limit` in USD (6 decimals) and whose `detail` carries the same
   * three values.
   */
  static budgetExhausted(
    agent_id: string,
    spent: number,
    limit: number,
  ): BillingError {
    return new BillingError(
      "budget_exhausted",
      `budget exhausted for agent '${agent_id}': ${spent.toFixed(6)} / ${
        limit.toFixed(6)
      } USD`,
      { agent_id, spent, limit },
    );
  }
}
