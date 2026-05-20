/**
 * @module @brainwires/telemetry
 *
 * Unified telemetry for the Brainwires Agent Framework — analytics events,
 * sinks, Prometheus-formatted outcome metrics, and a billing-hook interface.
 *
 * Deno-port pragmatics vs the Rust crate:
 * - SQLite sink and SQL query layer are omitted (use your own AnalyticsSink
 *   implementation — Deno KV, Postgres, OTLP — per your deployment).
 * - The `tracing` crate integration layer is replaced with an `onEvent`
 *   callback on {@link AnalyticsCollector} so consumers pipe events to OTLP
 *   or any logger themselves.
 * - All counters are plain numbers (JS isolate is single-threaded).
 *
 * Equivalent to Rust's `brainwires-telemetry` crate.
 */

export { AnalyticsError, type AnalyticsErrorKind, BillingError, type BillingErrorKind } from "./error.ts";
export {
  type AnalyticsEvent,
  type ComplianceMetadata,
  eventSessionId,
  eventTimestamp,
  eventType,
} from "./events.ts";
export {
  agentIdOf,
  apiCallEvent,
  costUsdOf,
  kindOf,
  sandboxSecondsEvent,
  timestampOf,
  tokensEvent,
  toolCallEvent,
  toolCallPaidEvent,
  type UsageEvent,
} from "./usage.ts";
export { type BillingHook } from "./billing_hook.ts";
export { type AnalyticsSink, DEFAULT_CAPACITY, MemoryAnalyticsSink } from "./sink.ts";
export { AnalyticsCollector, type EventCallback } from "./collector.ts";
export {
  avgCostPerRunUsd,
  avgProviderLatencyMs,
  avgRunDurationMs,
  cacheHitRate,
  MetricsRegistry,
  type OutcomeMetrics,
  successRate,
  toolErrorRate,
} from "./metrics.ts";
export { hashSessionId, redactSecrets } from "./pii.ts";
