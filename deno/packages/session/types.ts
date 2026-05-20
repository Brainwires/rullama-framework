/**
 * Shared value types used by every {@link SessionStore} implementation.
 *
 * Equivalent to Rust's `brainwires_session::types` module.
 */

/** Opaque identifier for a persisted session. */
export class SessionId {
  readonly value: string;

  constructor(value: string) {
    this.value = value;
  }

  /** Build a SessionId from any string. */
  static new(s: string): SessionId {
    return new SessionId(s);
  }

  /** Borrow the id as a plain string. */
  asStr(): string {
    return this.value;
  }

  toString(): string {
    return this.value;
  }

  equals(other: SessionId | string): boolean {
    return this.value === (other instanceof SessionId ? other.value : other);
  }
}

/** Metadata row returned by {@link SessionStore.list}. */
export interface SessionRecord {
  id: SessionId;
  /** Number of messages in the transcript. */
  message_count: number;
  /** ISO 8601 timestamp — when the session was first persisted. */
  created_at: string;
  /** ISO 8601 timestamp — when the session was last written. */
  updated_at: string;
}

/**
 * Pagination window passed to {@link SessionStore.listPaginated}.
 *
 * `offset` rows are skipped; `limit` (when non-null) caps the returned count.
 * Defaults to `{ offset: 0, limit: null }`, equivalent to an unbounded `list`.
 */
export interface ListOptions {
  offset: number;
  limit: number | null;
}

/** Default ListOptions — no offset, no limit. */
export function defaultListOptions(): ListOptions {
  return { offset: 0, limit: null };
}
