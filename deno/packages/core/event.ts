/**
 * Unified event schema with trace IDs and sequence numbers.
 *
 * Defines a common {@link Event} interface and {@link EventEnvelope} wrapper
 * that can carry any domain event with correlation metadata. This enables
 * cross-system tracing across the provider, A2A, and agent-network layers
 * without forcing a breaking change on existing event types.
 *
 * Equivalent to Rust's `brainwires_core::event` module.
 */

/** Common interface for structured events that carry correlation metadata. */
export interface Event {
  /** Unique ID for this specific event instance. */
  readonly event_id: string;
  /** Trace ID shared by all events in a single logical operation. */
  readonly trace_id: string;
  /** Monotonically increasing sequence number within the trace. */
  readonly sequence: number;
  /** Wall-clock timestamp (ISO 8601) when this event was emitted. */
  readonly occurred_at: string;
  /** Short lowercase label identifying the event kind. */
  readonly event_type: string;
}

/** Generate a fresh trace ID for a new logical operation. */
export function newTraceId(): string {
  return crypto.randomUUID();
}

/**
 * Generic envelope that wraps an arbitrary payload with correlation metadata.
 *
 * Wrap at the point where events are produced or consumed (audit logger,
 * OTel exporter) rather than retrofitting existing event structs.
 */
export class EventEnvelope<E> implements Event {
  readonly event_id: string;
  readonly trace_id: string;
  readonly sequence: number;
  readonly occurred_at: string;
  readonly event_type: string = "envelope";
  payload: E;

  constructor(trace_id: string, sequence: number, payload: E) {
    this.event_id = crypto.randomUUID();
    this.trace_id = trace_id;
    this.sequence = sequence;
    this.occurred_at = new Date().toISOString();
    this.payload = payload;
  }

  /** Map the payload to a different type, preserving all correlation fields. */
  map<U>(f: (payload: E) => U): EventEnvelope<U> {
    const mapped = new EventEnvelope<U>(this.trace_id, this.sequence, f(this.payload));
    // Preserve the original event_id and occurred_at across map operations.
    (mapped as { event_id: string }).event_id = this.event_id;
    (mapped as { occurred_at: string }).occurred_at = this.occurred_at;
    return mapped;
  }
}
