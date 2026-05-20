/**
 * AnalyticsCollector — fan-out to multiple sinks plus an optional `onEvent`
 * callback for consumers who want to pipe to OTLP or any other destination.
 *
 * JS isolates are single-threaded, so we skip the tokio mpsc+drain dance the
 * Rust crate needs: `record` is fire-and-forget with the sinks awaited
 * sequentially in a microtask. Consumers can call {@link flush} to wait for
 * in-flight deliveries and then flush each sink.
 *
 * Equivalent to Rust's `brainwires_telemetry::collector::AnalyticsCollector`.
 */

import type { AnalyticsEvent } from "./events.ts";
import type { AnalyticsSink } from "./sink.ts";

/** Callback invoked for every recorded event — cheap hook for OTLP export. */
export type EventCallback = (event: AnalyticsEvent) => void;

export class AnalyticsCollector {
  private readonly sinks: AnalyticsSink[] = [];
  private readonly callbacks: EventCallback[] = [];
  private pending: Promise<void> = Promise.resolve();

  constructor(sinks: AnalyticsSink[] = []) {
    this.sinks.push(...sinks);
  }

  /** Add a sink. */
  addSink(sink: AnalyticsSink): void {
    this.sinks.push(sink);
  }

  /** Register a fire-and-forget callback — invoked synchronously per record. */
  onEvent(cb: EventCallback): void {
    this.callbacks.push(cb);
  }

  /**
   * Emit an event. Returns immediately; delivery to sinks is awaited in the
   * background via a chained promise. Sink errors are swallowed (fail-open).
   */
  record(event: AnalyticsEvent): void {
    for (const cb of this.callbacks) {
      try {
        cb(event);
      } catch {
        // callbacks must never abort the run
      }
    }
    const sinks = this.sinks;
    this.pending = this.pending.then(async () => {
      for (const sink of sinks) {
        try {
          await sink.record(event);
        } catch {
          // sink errors are fail-open
        }
      }
    });
  }

  /**
   * Wait for all queued events to reach every sink, then call `flush()` on
   * each sink. Returns once durable sinks have acknowledged the flush.
   */
  async flush(): Promise<void> {
    await this.pending;
    for (const sink of this.sinks) {
      try {
        await sink.flush?.();
      } catch {
        // flush errors are fail-open
      }
    }
  }

  /** Signal shutdown — drain pending events and flush sinks. */
  async shutdown(): Promise<void> {
    await this.flush();
  }
}
