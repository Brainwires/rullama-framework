/**
 * AnalyticsSink interface and the in-memory default implementation.
 *
 * Equivalent to Rust's `rullama_telemetry::sink` + `sinks::memory` modules.
 */

import type { AnalyticsEvent } from "./events.ts";

/** Pluggable output for analytics events. */
export interface AnalyticsSink {
  /** Persist or forward a single event. */
  record(event: AnalyticsEvent): Promise<void>;
  /** Flush any buffered events. Default implementations may be a no-op. */
  flush?(): Promise<void>;
}

/** Default capacity for the in-memory ring buffer. */
export const DEFAULT_CAPACITY = 1024;

/**
 * Ring-buffer sink used for tests and short-lived processes.
 *
 * Events are kept in insertion order up to `capacity`; older events are
 * dropped from the front once the buffer is full.
 */
export class MemoryAnalyticsSink implements AnalyticsSink {
  /** Maximum number of events retained; the oldest is evicted once exceeded. */
  readonly capacity: number;
  private readonly buf: AnalyticsEvent[] = [];

  /**
   * Create an empty sink.
   * @param capacity Ring-buffer size (default {@link DEFAULT_CAPACITY}).
   */
  constructor(capacity: number = DEFAULT_CAPACITY) {
    this.capacity = capacity;
  }

  record(event: AnalyticsEvent): Promise<void> {
    this.buf.push(event);
    while (this.buf.length > this.capacity) this.buf.shift();
    return Promise.resolve();
  }

  flush(): Promise<void> {
    return Promise.resolve();
  }

  /** Number of events currently buffered. */
  len(): number {
    return this.buf.length;
  }

  /** `true` when no events are buffered. */
  isEmpty(): boolean {
    return this.buf.length === 0;
  }

  /** Copy of the buffered events, oldest first. */
  events(): AnalyticsEvent[] {
    return [...this.buf];
  }

  /** Empty the buffer. */
  clear(): void {
    this.buf.length = 0;
  }
}
