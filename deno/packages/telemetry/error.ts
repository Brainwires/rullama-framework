/**
 * Errors returned by the analytics subsystem.
 *
 * Equivalent to Rust's `brainwires_telemetry::error::AnalyticsError`.
 */

export type AnalyticsErrorKind =
  | "channel_closed"
  | "io"
  | "serde"
  | "other";

export class AnalyticsError extends Error {
  readonly kind: AnalyticsErrorKind;

  constructor(kind: AnalyticsErrorKind, message: string) {
    super(message);
    this.kind = kind;
    this.name = "AnalyticsError";
  }

  static channelClosed(): AnalyticsError {
    return new AnalyticsError("channel_closed", "Analytics sink channel closed");
  }

  static io(message: string): AnalyticsError {
    return new AnalyticsError("io", `I/O error: ${message}`);
  }

  static serde(message: string): AnalyticsError {
    return new AnalyticsError("serde", `Serialization error: ${message}`);
  }

  static other(message: string): AnalyticsError {
    return new AnalyticsError("other", message);
  }
}

/** Errors returned by BillingHook implementations. */
export type BillingErrorKind =
  | "hook"
  | "budget_exhausted"
  | "serde";

export class BillingError extends Error {
  readonly kind: BillingErrorKind;
  readonly detail: Record<string, unknown>;

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

  static hook(message: string): BillingError {
    return new BillingError("hook", `billing hook error: ${message}`);
  }

  static budgetExhausted(agent_id: string, spent: number, limit: number): BillingError {
    return new BillingError(
      "budget_exhausted",
      `budget exhausted for agent '${agent_id}': ${spent.toFixed(6)} / ${limit.toFixed(6)} USD`,
      { agent_id, spent, limit },
    );
  }
}
